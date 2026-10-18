# issue #170 — `-sm tensor` aborts at startup on a 24 GiB-class split (meta split-state / meta buffer alloc)

**Status: FIXED + validated in the fork tree (2026-10-17). Not in `patches/` yet — needs the
maintainer's call on the delivery shape (proposed: amend block 09, cut r82).**

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
* Not validated: gfx1100 itself (no hardware). The fix is validated at 24 GiB *free-memory* parity,
  which is the reporter's configuration class.

## 5. Open questions / next steps

1. **Delivery shape**: amend **block 09** (it owns the Meta device iface, the `offload_op` hook that
   makes this reachable, and the split-state cache) with both hunks, then `make-patches.sh` + r82.
   Blocks 00/09/10 all touch `ggml-backend-meta.cpp`; 09 is the natural owner.
2. **Upstream**: both hunks are upstream code (`ggml_backend_meta_get_split_state`, and the meta
   `alloc_buffer*` asserts from PR #23671/#27586). Defect 2 is plainly upstream-relevant (an OOM aborts
   instead of erroring). Defect 1 is a latent upstream invariant violation; in unpatched upstream it is
   hard to reach because the meta device has no `offload_op` (candidate `upstream/` PR).
3. **Policy question (not fixed here)**: with the hook in place, `-sm tensor` op-offloads a *dense*
   CPU-offloaded layer's ops to the Meta device (weights streamed per op over the x4 host link).
   Upstream does the same under `-sm layer`, so this is not a regression of ours, but under `-sm tensor`
   it partly defeats layer offload and deserves a bandwidth/benefit gate (TODO candidate).
4. **Reporter reply**: mention the two working workarounds (`--no-op-offload`,
   `GGML_OP_OFFLOAD_MIN_BATCH=1000000`), that r75+ still asserts, the exact trigger, and ask for the
   ROCm version (never stated: he builds with `CMAKE_PREFIX_PATH=/opt/rocm/core` and
   `-DGGML_HIP_RCCL=ON`, `GPU_TARGETS=gfx1100`; his container tags are ours, so the line is not
   identifiable from the log).
