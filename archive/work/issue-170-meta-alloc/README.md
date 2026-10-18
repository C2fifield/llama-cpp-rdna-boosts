# issue #170 — `-sm tensor` aborts at startup on a 24 GiB-class split (meta split-state / meta buffer alloc)

**Status: SHIPPED in `v12-097cf33d9-r82` (2026-10-18), canonical tip `7d238ae80`, net tree
`dc48b4a983cba245f71e60ce8825abedac8e2a45`.**  Split per the maintainer's rule: the two upstream-code
hunks (`[TAG_META_UNALLOC_ALIAS]`, `[TAG_META_ALLOC_FAIL]`) amend **block 00**; the fork-local hunks
(`alloc_buffer_usage` NULL handling, the per-tensor policy branch, `[TAG_META_OFFLOAD_MOE_ONLY]`) amend
**block 09**.  `validate-set.sh` green on a fresh tarball (strict 12/12); CI Validate (main and tag) and
Release (tag) green; `rdna-boosts` force-pushed to the personal fork with tree `dc48b4a9` (verified by a
fresh fetch); the container watcher picks the tag up off-host.  The reporter's reply is posted
(`#issuecomment-6248869022`).

Reporter: `void-li`, dual **7900 XTX (gfx1100, 24 GiB/card)**, 2 GPUs, dense
`Qwen3.8-27B-Q8_0` (qwen35 hybrid, 65 blocks), `-sm tensor`, `--ctx-size 196608`.
Symptom: abort during model load, at the *warmup decode*:

```
GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf)) failed   (ggml-backend-meta.cpp:710/482 in his builds)
  ggml_backend_meta_buffer_n_bufs
  ggml_backend_meta_get_split_state(stc, tensor, assume_sync=true)
  ggml_backend_meta_buffer_init_tensor_impl
  ggml_gallocr_init_tensor  <- ggml_gallocr_alloc_graph <- ggml_backend_sched_alloc_graph
  llama_context::process_ubatch <- llama_context::decode  <- common_init_from_params (warmup)
```

## TL;DR of the root cause (it is NOT a failed allocation)

Two *independent* defects are on this path. The reported abort is **defect 1**; **defect 2** is the
"allocation failure is not caught" half the maintainer suspected (it is real, and it is a separate
crash — see §3).

1. **Meta split-state walks an alias tensor the graph allocator never initialised.** `ggml_backend_meta_get_split_state()`
   recurses into `tensor->src[]`, and for a *view* one of those srcs can be an intermediate alias that
   is deliberately never visited by `ggml_gallocr_init_tensor()` (a view is initialised from
   `view_src`, not from `src[0]`). Its `buffer` stays `NULL` for the whole run → `buffer_n_bufs(NULL)`
   → abort. `build_attn_mha()` creates exactly that pair: `q = ggml_view_4d(Qcur, …)` then
   `q = ggml_permute(…)`; the permute is a plain alias (`GGML_OP_NONE`, no graph node) whose
   `src[0]` is the unallocated intermediate view.
2. **A failed constituent allocation aborts** in the meta wrapper (`GGML_ASSERT(meta_buf_ctx->bufs[i])`
   in `…_alloc_buffer_n`, and `GGML_ASSERT(bufs.back() != nullptr)` in `…_alloc_buffer_usage`) instead
   of returning `NULL` for the caller (gallocr / KV cache / model loader) to report.
3. **The block-09 meta `offload_op` hook claimed dense host-weight ops**, and a CPU-offloaded layer's
   *state* (KV cache, GDN conv/ssm state) is host-resident too.  The meta split-state model has no
   representation for a host input of a split op, so the context aborted at the first **wide prefill**
   (`handle_gated_delta_net`, `src_ss[5]`, which a host KV would mirror in `handle_flash_attn_ext`).
   With only defects 1+2 fixed the server loads, listens, and answers a 5-token completion, but a real
   prompt still aborts — so this had to be fixed for #170 to be closed in practice.

## 1. The exact trigger (why 24 GiB aborts and our 32 GiB cards do not)

```
--fit (default on) sees 27B-Q8_0 + 196608 ctx does not fit on 2 x 24 GiB
   -> tensor-split branch reduces n_gpu_layers (binary search)
      (reporter's r12 log: 66 -> 32 -> 49 -> 57 -> 53 -> 55 -> 54, "offloading 54/58 layers")
   -> a MIXED CPU/GPU layer assignment
      -> the CPU layers' weights are host-resident
      -> the scheduler's op-offload (cparams.op_offload, default on; CUDA offload_op = batch >= 32)
         re-places those layers' ops on the META backend
         (this only happens because block 09 added ggml_backend_meta_device_offload_op;
          upstream's meta device iface has .offload_op = nullptr, which is why unpatched
          llama.cpp runs the same config)
      -> resolve_fused_ops even says so: "layer 3 is assigned to device CPU but derived kq mask
         flash attention is assigned to device Meta()"
      -> build_attn_mha's non-FA path (a CPU-offloaded layer has no FA) builds
         ggml_view_4d(Qcur-3) -> ggml_permute() ; the permuted alias becomes a scheduler
         input/alloc-dep node (ggml_view_tensor(input)) whose meta buffer init walks src[]
      -> src[] reaches the unallocated intermediate view -> NULL buffer -> abort
```

On 2 x 32 GiB with the same command the fit prints `tensor split: no changes needed`
(estimated 56376 MiB vs 62508 MiB target, 66/66 layers), there is no CPU layer, the offload hook never
fires for a host-weight op, and the run is clean — **hence "cannot reproduce on our cards"**.

## 2. Reproduction on a 32 GiB card (the tool in this directory)

`vram_fill.cpp` holds N MiB on every visible HIP device, so a 32 GiB card presents a 24 GiB card's
*free-memory figure* (8192 MiB → 24210 MiB free of 32624 MiB, 7900 XTX-like):

```bash
hipcc -O2 -o /tmp/vram_fill vram_fill.cpp --offload-arch=gfx1201

# 1) hold the VRAM, 2) run the reporter's minimal command, 3) release
HIP_VISIBLE_DEVICES=0,1 /tmp/vram_fill 8192 & FILL=$!; sleep 8
HIP_VISIBLE_DEVICES=0,1 ./build-rocm/bin/llama-server \
  --model /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf \
  --ctx-size 196608 --split-mode tensor --no-mmproj --port 18099
kill $FILL
```

`repro.sh <log> <fill_mib> [server args…]` is the same thing with the cleanup and the outcome
(LISTENING / ABORTED / EXITED) in one step.

Observed **before** the fix (2 x R9700, gfx1201, ROCm 10.1.0, r81 tree `0c3a4413`):

| config (fill 8192 MiB) | result |
|---|---|
| minimal `--ctx-size 196608 --split-mode tensor` | **abort** `ggml-backend-meta.cpp:711` (warmup decode) |
| reporter's full line (`--parallel 2 --kv-unified --no-host --load-mode none --spec-type draft-mtp-adaptive,ngram-mod …`) | fit reduces layers, same abort path |
| `-ctk f32 -ctv f32` (fit on) | same abort path |
| `GGML_COMPUTE_BUFFER_MARGIN_PCT=400` | same abort path |
| `--fit off -ngl 99 -ctk f32 -ctv f32` (genuine OOM) | **abort** `ggml-backend-meta.cpp:2287` after `cudaMalloc failed: out of memory` (defect 2) |
| `GGML_COMPUTE_BUFFER_MARGIN_PCT=3000` (compute OOM) | **abort** at `…_alloc_buffer_usage` assert (defect 2) |
| fill 20000 MiB, `--fit off -ngl 99` (weight OOM) | **abort** at `…_alloc_buffer_n` assert (defect 2) |

Without the filler the same configs are clean (fit keeps 66/66 layers).

## 3. The fix (two hunks, `ggml/src/ggml-backend-meta.cpp`, staging branch `fix-170`)

**a. `[TAG_META_UNALLOC_ALIAS]`** — resolve an unallocated alias through its view chain before
classifying it. Both split-state entry points now use
`ggml_backend_meta_view_chain_buffer()`: an alias with no buffer of its own is classified by the
buffer it will inherit (same meta buffer → compute its state normally; host / other device / nothing
allocated → "not split"). Allocated tensors take the byte-identical old path, so this cannot perturb a
graph that works today. (The reporter's own patch returned "unsplit" for the whole tensor; that is why
it "starts but still crashes when making a request" — the state was then wrong for the split.)

**b. `[TAG_META_ALLOC_FAIL]`** — `…_alloc_buffer_usage()` and `…_alloc_buffer_n()` return `NULL`
after freeing the sub-buffers already allocated (and, for `_n`, after clearing the tensors' `buffer`/
`data` so nothing points at the freed meta buffer) instead of asserting. The callers already handle
`NULL` (`ggml_vbuffer_alloc`, `ggml_backend_alloc_ctx_tensors_from_buft` → KV cache / model loader),
so an OOM becomes a clean, attributable startup error.

**c. `[TAG_META_OFFLOAD_MOE_ONLY]` (fork-local, block 09)** — the hook claims only `GGML_OP_MUL_MAT_ID`,
the host-resident MoE expert op it exists for.  A dense host-weight op is not claimable until the meta
split state can represent a host-resident input (upload + gather, or one-device fallback), which is
design work, not a patch release.

This keeps the r20 invariant ("a compute buffer that cannot grow must never abort"): the CUDA
choke point already returns NULL after yielding the arena; the meta wrapper was the one place that
turned that into an abort.

## 4. Validation (gfx1201, 3 x R9700, ROCm 10.1.0, `~/bin/build-llama-rocm`)

* 24 GiB parity, minimal command: **LISTENING**, `HTTP 200`, coherent
  (`" Paris.\nThe capital of Germany is Berlin.\n…"`), server stays up.
* 24 GiB parity, the reporter's **full** command line (adaptive MTP + ngram-mod, `--kv-unified`,
  `--parallel 2`, `--no-host`, `--load-mode none`): **LISTENING**, `HTTP 200`, coherent.
* `-ctk f32 -ctv f32` at parity: LISTENING + `HTTP 200`.
* `-sm layer` at parity and 1-GPU: unaffected (LISTENING).
* Genuine OOM paths now exit 1 with a clean message and no core, e.g.
  `failed to allocate ROCm0 buffer of size 12884901888 (0 of 2 sub-buffers allocated)` →
  `failed to initialize the context: failed to allocate buffer for kv cache`;
  partial failure (`1 of 2 sub-buffers allocated`) also clean, **no VRAM leak** afterwards.
* Coherence (3-GPU tensor, dense 27B UD-Q4_K_XL, `prompts/reasoning.txt`, `-n 128 --seed 42 --temp 0`):
  `1033 chars sha=383655591392` — identical for the base build, a base re-run and the fixed build.
* **Wide prefill (the case a 5-token completion misses)**: 24 GiB parity, `prompts/prose-rdna-boosts.txt`
  (5246 tokens, 11 ubatches) → `HTTP 200`, 64 tokens predicted, coherent, no assert, both for the
  minimal command and the reporter's full command line.

### Op-offload trade-off (item 3 of the maintainer's review, measured 2026-10-18)

`llama-batched-bench`, 2 x R9700, `-t 8`, dense 27B Q8_0, 16384-token prefill, 128-token decode.
"24 GiB parity" = 8192 MiB/device held; `-ngl 54` (12 of 66 layers on the CPU) is the fit's own
choice for that box at ctx 196608.

| topology / arm | pp t/s | tg t/s |
|---|---:|---:|
| `-sm tensor`, op-offload **as before** (hook claims dense ops) | **abort** (`handle_gated_delta_net` assert) | — |
| `-sm tensor`, hook gated (new default) | 113.97 | 7.32 |
| `-sm tensor`, `--no-op-offload` (reference) | 118.20 | 7.46 |
| `-sm tensor`, **full residency**, 32 GiB, 66/66 layers | **1779.69** | **32.32** |
| `-sm layer`, op-offload ON (dense CPU-layer ops on ONE device) | **797.17** | 6.43 |
| `-sm layer`, `--no-op-offload` | 110.99 | 6.45 |

Readings:

* Under `-sm layer` (upstream's mechanism, where the op and its state share one backend) op-offloading
  a dense CPU-offloaded layer is a **7.2x prefill win** (797 vs 111) with decode unchanged.  So the
  feature is worth having; it is only the *tensor-split* form that is not implemented.
* Under `-sm tensor` it cannot run today (defect 3), and the gated result is `--no-op-offload` within
  noise, which is the correct behaviour.
* The dominant number for a user choosing a config is full residency: 32.32 t/s decode and 1780 t/s
  prefill vs ~7.3 / ~114 t/s for any 54/66 split.  On 2 x 24 GiB with a 27B Q8_0 at 196k context a
  ~7 t/s decode is the *expected* result of the memory shortfall, not a bug.
* The hook's intended target is unaffected: 35B-A3B UD-Q4_K_M, 2 GPUs, `-sm tensor -ncmoe 40`,
  4096-token prefill: decode **94.88 t/s vs 35.74** with `--no-op-offload` (+165%), prefill 445.96 vs
  498.80 (-11%).

### Gate regression check on the largest host-expert case (maintainer's concern)

`Qwen3.8-Flash-Next-IQ4_NL` (100 GB, 176.9B A3B), 9-part `PROJFIX`, **2 GPUs**, `-sm tensor`,
`-ngl 99 -ncmoe 48`, ctx 32768, 4096-token prefill + 128 decode, `-t 8`, `--load-mode none
--lazy-mode off` (no PLE lazy-reader variance), expert cache auto (arena 39234.4 MiB, 60.5 %
residency, identical in every arm):

| arm | pp t/s | tg t/s |
|---|---:|---:|
| pre-gate build #1 | 246.99 | 39.92 |
| pre-gate build #2 | 248.14 | 40.88 |
| gated build #1 | 244.46 | 39.83 |
| gated build #2 | 246.40 | 40.68 |
| gated build, `--no-op-offload` | 66.25 | 18.32 |

Gated vs pre-gate is inside the run-to-run spread (pp 0.7 to 1.0 % lower, tg 0.5 to 2.5 % lower, with
an 0.8 % / 2.1 % spread within the gated repeats), and the hook is still doing its job: 3.7x prefill
and 2.2x decode over `--no-op-offload`. The reason is structural: under `-ncmoe` the only host-resident
weights are the expert tensors, whose op is `MUL_MAT_ID`, which the gate keeps.

* **gfx1100 cross-check** (`fingon`, single RX 7900 XTX, gfx1100, ROCm 7.14, `build-rocm`): r82 vs the
  r81 build on the same box, same-seed greedy with `-ctk/-ctv q8_0`, `-n 128 --seed 42 --temp 0`,
  `prompts/reasoning.txt`: **byte-identical** on Qwen3.8-27B Q4_K_M (`1064 chars sha=c606b3d55f24`) and
  Qwen3.6-35B-A3B Q4_K_M (`1046 chars sha=2c04f104b88e`), i.e. the change is inert on a single device.
  That box has one card, so it **cannot** exercise `-sm tensor` (the code path that changed); the
  reporter's retest remains the gfx1100 tensor-split confirmation.  The reporter has confirmed
  **ROCm 10.1.0**, the same tree we reproduce on, so the reproduction is not cross-version.
* Not validated: gfx1100 with two cards (`-sm tensor`).  The fix is validated at 24 GiB *free-memory*
  parity on gfx1201, which is the reporter's configuration class.

## 5. Open questions / next steps

1. **Delivery shape** (maintainer's rule: block 00 = upstream-bug fixes, block 09 = our code):
   * **block 00** — hunks (a) and (b), both upstream code (`ggml_backend_meta_get_split_state`, and the
     `alloc_buffer*` asserts from upstream PRs #23671/#27586).  Both hunks are staged standalone in
     `upstream/UPSTREAM-PR-meta-split-unalloc-alias-and-alloc-fail.{md,patch}`, cut against pristine
     `097cf33d9` and built there.
   * **block 09** — hunk (c) only: the hook gate, which is fork-local policy for our own feature.
   * then `make-patches.sh`, `validate-set.sh`, the standard release gates, r82.
2. **Upstream**: staged as one patch (both hunks) in `upstream/`.  It applies clean to `097cf33d9` and
   builds; the maintainer's note is that their submissions are often ignored, so the patch is staged for
   someone else to file.
3. **Follow-up design (open, not a patch release)**: make a host input of a *split* op representable in
   the meta split state.  The `-sm layer` measurement (7.2x prefill) says the dense-layer win is real, so
   this is a TODO rather than a "never"; it needs upload + gather (or a one-device fallback for such a
   node) plus its own correctness/perf gates.
4. **Reporter reply**: **posted 2026-10-18** (`#issuecomment-6248869022`); draft kept in
   `COMMENT-2026-10-18.md`. Originally drafted 2026-10-18 — the trigger, the two workarounds (`--no-op-offload`,
   `GGML_OP_OFFLOAD_MIN_BATCH=1000000`), the fix, and a note that an updated delivery patch release is
   coming.  ROCm 10.1.0 confirmed by the reporter (same as ours).
