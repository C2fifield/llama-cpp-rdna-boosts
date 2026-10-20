# The delivery patches — what each block is and what it does

**12 blocks** (block 00 + blocks 01-11) against `release.json.base`, one `git am` commit each.  This file
is the per-block reference: what a block contains, what it changes, the knobs its code reads and the
purity notes that constrain it.  It is **not** a changelog — the release-by-release record is
[`../WORKLOG.md`](../WORKLOG.md), the apply order and verification contract is
[`../MANIFESTS.md`](../MANIFESTS.md), provenance and drift policy is [`../BASELINE.md`](../BASELINE.md),
and every environment variable is catalogued in [`../ENVIRONMENT.md`](../ENVIRONMENT.md).

> **Dated notes.**  This file used to be the running record of block amendments (16-block numbering,
> `r1`..`r93`).  That history is preserved verbatim in
> [`../archive/docs/patch-notes-history.md`](../archive/docs/patch-notes-history.md) and is what
> `WORKLOG.md` entries and `archive/work/<campaign>/` records cite.  Everything below describes the
> **current** blocks in **V2.0** numbering.

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(jq -r .base /path/to/rdna-boosts/release.json)"
bash /path/to/rdna-boosts/scripts/apply-all.sh .          # creates branch rdna-boosts, 12 commits
bash /path/to/rdna-boosts/scripts/apply-all.sh --include-beta .   # + beta/*/*.patch (NOT the release tree)
```

All blocks with **`git am`**, in order.  `git apply` on the concatenated series silently drops hunks; the
manual equivalent is `git am patches/000[0-9]-*.patch patches/001[01]-*.patch`.  `apply-all.sh` asserts
the applied tree equals `release.json.tree` on the strict path.

## Block index

Sizes are `git diff` line counts against the block's parent (i.e. what that one commit adds), and matter
only as a rough shape.

| # | patch | subject | +/− | files |
|---|---|---|---|---|
| 00 | `0000-…-block-00-foundation-band-uniformity-Meta` | foundation: band uniformity, Meta wrapper, allocator | +1250 / −1001 | 48 |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-RDNA4-gat` | hybrid HIP all-reduce | +2364 / −45 | 4 |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth | +1051 / −41 | 19 |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill + rollback tracker | +3182 / −16 | 22 |
| 04 | `0004-…-block-04-BF16-KV-cache-and-native-BF16-f` | BF16 KV cache + native-BF16 FA | +630 / −215 | 8 |
| 05 | `0005-…-block-05-flash-attention-band-native-qua` | FA band / native quantized K-V / derived kq mask | +4596 / −577 | 287 |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM | +2620 / −4 | 5 |
| 07 | `0007-…-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM` | k-quant boosts (VDR, Q6_K MMQ, Q1_0, grid pad) | +884 / −145 | 9 |
| 08 | `0008-…-block-08-fused-kernels-old-08-fusions-ol` | fused kernels + fused MoE | +5369 / −445 | 36 |
| 09 | `0009-…-block-09-host-resident-MoE-iface-schedul` | host-resident MoE: iface, scheduler, cache, slab hooks | +13962 / −396 | 67 |
| 10 | `0010-…-block-10-qwen4exp-model-hyper-connection` | qwen4exp, the slab TU, the fusion dispatcher, async input | +21964 / −5063 | 104 |
| 11 | `0011-…-block-11-campaign-remainder-CPU-wins-bac` | campaign remainder | +881 / −101 | 30 |

Block 05's file count is `template-instances/*.cu` (the FA kernel instantiations); block 10's size is
dominated by `qwen4exp` + the slab/fusion translation units that were split out of `ggml-cuda.cu`.

---

## Block 00 — foundation: band uniformity, the Meta wrapper, the allocator

**What it is.**  The rules and the plumbing every later block assumes: *decode and verify must take the
same arithmetic*, the tensor-split wrapper must never abort on a legal graph, and the allocator must not
make output a function of the layout.  If a later block's purity claim is "one arm across `W = 1..8`", the
band rule it satisfies starts here.

**Contents.**

* **FA small-batch KV-split width invariance** (issue #25) — `launch_fattn`'s `parallel_blocks` heuristic
  maximised wave efficiency over `ntiles_dst`, a function of `Q->ne[1]`, so single-token decode and a
  verify batch grouped the online-softmax/PV partials differently.  It is evaluated as if `n_q == 1` for
  every `n_q <= 8` (`ntiles_dst_eff`); prefill (`n_q > 8`) is untouched.
* **CPU decode/verify band** — the llamafile `sgemm` path is gated to prefill (`n <= 8` takes `vec_dot`),
  because llamafile rejected `n < 2`, i.e. decode and verify accumulated differently on CPU.
* **Vulkan masked-V / freed-cell fixes** — `flash_attn.comp` / `flash_attn_cm1.comp` never read V for
  dead columns (baseline shaders, hence this block).
* **`-Wshadow` for the `llama` target** (GCC `-Wshadow=local`, clang plain) plus the rename sweep that
  makes it clean — the class of bug behind the 2026-09-11 dead-mask-chain finding
  (`GREEDY-PURITY.md` §23).
* **`ggml-backend-meta`**: resolve a split-state through the view chain when the alias was never
  initialised (issue #170), return NULL from a failed per-device allocation instead of asserting (a
  clean OOM exit), and keep the split-state hot path at its pre-#170 shape with the #170 resolution in a
  `noinline, cold` helper.
* **`ggml-alloc`**: release view sources whose views are never consumed (a real leak), and fingerprint the
  graph layout by node order + output marking so a *different* graph cannot reuse a layout computed for
  another (issue #183).
* **`MUL_MAT_ID` repeated ids** — emit every `(token, iex)` occurrence as its own compact row and size the
  column grids for the expanded expert list (upstream bug; the delivery was the reproducer).
* **Shared-NextN MTP heads keep their own KV** — the draft driver inferred KV sharing from `ctx_other`
  alone, which is wrong for a `gemma4-assistant` head (upstream bug `04eb4c446`).
* **Multi-ubatch logits readback ordering** — the pinned async D2H copy is completed before the next
  ubatch runs, so a `llama-perplexity` `n_seq >= 4` run cannot read a torn readback.

**Files.**  `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml-cuda/mmid.cu`/`mmf.cu`/`mmq.cu`,
`ggml-backend-meta.cpp`, `ggml-alloc.c`, `ggml-cpu/llamafile/sgemm.cpp`,
`ggml-vulkan/vulkan-shaders/flash_attn*.comp`, `src/llama-{graph,context,model,kv-cache}.cpp`,
`src/CMakeLists.txt`, `src/models/*` (the shadow rename), `src/llama-grammar.cpp`, `common/speculative.cpp`.

**Knobs.**  `GGML_READBACK_SYNC` (diagnostic).  The rest are behaviour, not switches.

**Notes.**  Block 00 is the *only* block whose changes are not gated on RDNA: it fixes code paths every
backend shares.  A re-base conflict here is usually an upstream fix arriving for the same bug — keep
upstream's version and drop the local hunk.

---

## Block 01 — hybrid HIP all-reduce

**What it is.**  A device-side all-reduce for the small tensors `-sm tensor` reduces on every decode step,
plus a per-size hybrid dispatch: small tensors take the internal pipeline, large ones go to RCCL/NCCL.
Four files, ~2.4k lines, all in `allreduce-hip.cu` + `ggml-cuda.cu`.

**Contents.**  Internal host-staged pipeline and butterfly kernels; a wall-clock soft/hard spin budget so a
lagging rank fails soft instead of hanging (issue #139); eager `ncclCommInitAll` (a lazy mid-run init
perturbed the split compute and broke `-sm tensor -ncmoe` MTP purity); a first-runtime-failure fallback
that warns once, disables NCCL and re-routes to the internal path; an opt-in P2P/large path and an opt-in
copy-engine (SDMA) scratch path; and the `AR_PROFILE` device-list initialisation fix.

**Knobs.**  `GGML_CUDA_ALLREDUCE` (`hybrid` default, `nccl`, `internal`, `meta`), `GGML_CUDA_AR_P2P*`,
`GGML_CUDA_AR_CE_SCRATCH_MIB`, `GGML_CUDA_AR_SPIN_TIMEOUT_MS` / `_HARD_MS`,
`GGML_CUDA_AR_SMALL_BLOCKS` / `_SMALL_BYTES`, `GGML_CUDA_AR_KER_EVENTS`, `GGML_CUDA_AR_PROFILE`.

**Purity.**  `GGML_CUDA_ALLREDUCE=nccl` is **not** bit-identical under `-sm tensor` (the internal pipeline
BF16-round-trips), so it is a smoke comparison, never a purity reference.  The hybrid dispatch's
small/large crossover must not move with batch width — the 2-device crossover was raised to the 3-device
value (131072 elements) for exactly that reason (`GREEDY-PURITY.md` §11, cause A).

**Arch gate.**  The internal AR is RDNA4-only; on other arches it falls back to RCCL.  Cross-backend
consistency: the HIP split lives in `allreduce-hip.cu`, upstream's CUDA file is untouched.

---

## Block 02 — adaptive MTP draft depth

**What it is.**  The `draft-mtp-adaptive` speculator: a controller that sets the MTP draft depth from the
observed acceptance, instead of a fixed `--spec-draft-n-max`.  The tuned bucketed controller is the
shipped one (`delta = n_accepted - depth`, `climb_budget(d) = 20 + 6*(d-1)`,
`drop_pressure(d) = max(60, 10*d)`, cold start `max(floor, cap-3)`, `--spec-draft-n-start N`).

**Contents.**  `common/speculative-adaptive.h` + `.cpp` and the CLI plumbing
(`--spec-type draft-mtp-adaptive`, `--spec-draft-n-start`); the draft-depth clamp raised from 7 to 15
(`LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escapes it; purity is promised only to 7 — see
`GREEDY-PURITY.md` §11); an opt-in MTP prompt-tail ingest and a last-row prompt copy; a `--fit` SIGSEGV
fix with `draft-mtp-adaptive` (issue #38); `tests/test-speculative-adaptive.cpp` and
`tests/test-recurrent-state-depth.cpp`; and the **embeddings-layer-input preset API**
(`llama_model_set_embeddings_layer_inp_preset`) that the draft setup calls before the target context
reserves its compute buffers (r94 moved it here, the block that calls it).

**Knobs.**  `--spec-draft-n-max`, `--spec-draft-n-start`, `--spec-draft-p-min`,
`LLAMA_SPEC_DRAFT_N_MAX_CLAMP`, `LLAMA_MTP_INGEST_TAIL`, `LLAMA_MTP_DRAFT_OP_OFFLOAD`.

**Gate.**  Judge any controller change on the four axes (reasoning / prose / code / recall) **and** the
phase-switching prompt (`prompts/code-reasoning-mixed.txt`), at `-n 3000` (`-n 2000` floor),
`--reasoning on` for R and `off` for P/C/K.  Short runs are a correctness smoke test only
(`benchmarks/mtp-adaptive-methodology.md` rule 0).  Records:
`benchmarks/2026-09-15-adaptive-mtp-tuning.md`, `archive/work/adaptive-mtp-ceiling-scaling/`.

---

## Block 03 — chunked gated-delta-net prefill + the recurrent rollback tracker

**What it is.**  The fused chunked GDN prefill kernel (bf16/WMMA), shipped as **two arch-segregated
files**: the RDNA4 kernel (`gated_delta_net_chunked_bf16.cu`) and a first-gen-WMMA port for gfx11
(`gated_delta_net_chunked_bf16_gfx11.cu`), selected at runtime by device `cc`, with no-op stubs so the
host launchers link on either pass.  Plus the per-sequence rollback tracker, which is what makes a
speculative rewind into a prefill safe.

**Contents.**

* Chunked prefill is **K-independent**: threshold `max(K > 16 ? K : 16, n_rs_batch)`.
* **Rollback snapshots**: only the sequential kernel writes the `K = n_rs_seq + 1` snapshot planes, and a
  whole-batch rollback also needs a pre-batch plane (copied into slot `n_tokens` when
  `0 < n_tokens < K`).
* `prepare()` folds in **every** ubatch of a prepare list (r91) — recording only `ubatches.back()` made a
  sequence that ran in a non-final ubatch keep an earlier prefill chunk as its "last batch", which is what
  the issue-#131 warning was reporting.
* `LLAMA_RS_ROLLBACK_DEBUG` (`RSPREP`/`RSROLL`) is the instrument that proved both.

**Knobs.**  `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel (bit-identical, slow),
`GGML_CUDA_GDN_CHUNKED_BF16=0` forces the F32 chunked path, `LLAMA_RS_ROLLBACK_DEBUG` (diagnostic),
`GDN_DBG_*` (kernel-debug only).  The sequential-tail knob (`GGML_CUDA_GDN_TAIL`) is read by block 10's
copy of the launch site.

**Purity.**  The pure `none == draft-mtp` range is `n_max <= 7`; `n_rs_batch` is derived from the widest
enabled speculator (`common_speculative_n_max() + 1`, every draft *and* ngram type), so a verify batch can
never take the chunked path (`GREEDY-PURITY.md` §27).  The BF16 chunked path was the issue-#113 prefill-logit
regression: **run `scripts/gate-prefill-logits.sh` on any change here.**

**Gates.**  `test-backend-ops -o GATED_DELTA_NET`, `test-recurrent-state-depth`,
`test-recurrent-state-rollback` (compare the per-arm FAIL/PASS *set*, never the count — it is per box).

---

## Block 04 — BF16 KV cache + native-BF16 flash attention

**What it is.**  Makes bf16 a first-class K/V cache type with a **native** FA path instead of an F16
conversion, and fixes the gfx11 masked-V signed-zero leak that a bf16 cache exposes.

**Contents.**  `V_DOT2_F32_BF16_AVAILABLE` for the RDNA3/3.5/4 targets (`common.cuh`); native-BF16 tile
instantiations for every head size plus a BF16→F16 converting load for the F16-math fallback
(`fattn-tile.cuh`); the tile case-type chooser and `launch_fattn`'s `need_f16_K/V` decision
(`fattn-tile.cu`, `fattn.cu`); the gfx11 signed-zero masked-V guard in `fattn-mma-f16.cuh` (a fully masked
KV row fed its `x + (-0.0)` products to the WMMA and leaked stale/freed content); `rope.cu` bf16 support;
`-Wshadow`/warning cleanup; the `FLASH_ATTN_EXT` head-size matrix in `tests/test-backend-ops.cpp`.

**Knobs.**  None — bf16 is a cache type (`-ctk bf16 -ctv bf16`).  The staging/native *policy* switch
(`GGML_CUDA_FA_KV_NATIVE`) is block 05's.

**Purity.**  `-ctk`/`-ctv` must **match**: mixed K/V types are hard-rejected at context creation (every
mixed pair measured 1.7-3.6x slower and never smaller).

---

## Block 05 — the FA band, native quantized K-V, the derived kq mask

**What it is.**  The RDNA flash-attention work: a family of band kernels that serve the whole
decode/verify band with one reduction order, native (non-staging) reads of quantized K/V, and the derived
kq mask that removes the packed `n_kv × n_q` F16 mask from both compute and host memory.  287 files —
almost all of them `template-instances/*.cu`.

**Contents.**

* **The band kernels** (`fattn-rdna.cuh`): the RDNA4 head-256 GQA-6 band (the GQA group folded into
  `ncols2 = 8`, a round-robin KV split over a fixed `P`), extended to the 2-byte (f16/bf16) arm and capped
  by `GGML_HIP_FA_BAND_WMMA_GQA_MAX` (default 12) for GQA-12 models.  Since r92 the 2-byte arm uses the
  same 32-column block as the quantized arm with a `nsm/2` band split.
* **The FA chooser band rule (F1)**: a quantized K/V cache took VEC for `n_q <= 2` and TILE from `n_q = 3`;
  both VEC conditions sit inside the band, so the branch is deleted and the band is TILE
  (`GREEDY-PURITY.md` §14).
* **Native quantized K/V (V4/V5)**: the sub-F16 quant types (`q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`)
  read the cache natively instead of a whole-cache F16 conversion per decode step — a decode-*depth* win,
  and the margin that lets a high-context MTP config load (`GREEDY-PURITY.md` §34).
* **`q4_1`/`q5_0`/`q5_1`/`iq4_nl` as FA cache types** (previously reachable only via
  `GGML_CUDA_FA_ALL_QUANTS`, which also permits mixed K/V), each with its own band sweep and oracle
  (`GREEDY-PURITY.md` §20-22).
* **The derived kq mask (V3)**: per-token bounds + one per-cell position replace the packed mask; the FA
  kernel computes visibility on the fly.  Implemented in **both** the MMA and the tile kernels, and only
  for prefill-shaped batches (`n_tokens > 8`), so decode/verify are untouched.
* **Fully-masked KV group skip** (`GGML_CUDA_FA_MASK_SKIP`, prefill only) — exact zeros, bit-identical.
* **Hygiene**: the `(512, 512, 32)` RDNA MMA row retuned to a 4-warp shape (issue #188, Gemma-4 head-512
  prefill), the RDNA2 (gfx1030) gate fixes of PR #181, the masked-V HIP fixes, and the FA staging arena
  (prealloc + graceful degradation instead of abort).

**Knobs.**  `LLAMA_KQ_MASK_DERIVED`, `GGML_CUDA_FA_KV_NATIVE` (auto / force on / force staging),
`GGML_CUDA_FA_MASK_SKIP`, `GGML_HIP_FA_BAND_WMMA*` (`_SPLIT`, `_GQA_MAX`, `_WIDE`),
`GGML_CUDA_FA_WMMA_256` / `_MAX_HEAD`, `GGML_CUDA_FA_ALL_QUANTS` (build), `GGML_CUDA_OP_TIMING`
(diagnostic).

**Purity.**  One arm across `W = 1..8` is the contract; the guarantee is per KV type
(`GREEDY-PURITY.md` §§11-14, 36).  **Any change to an FA kernel or the derived mask must pass
`scripts/gate-prefill-logits.sh`** — decode-side coherence and MTP are blind to a uniform prefill shift.

**Upstream candidates.**  `fa-kv-split-width` (block 00), `fa-decode-verify-kernel-family`,
`fa-512-rdna-head-cap`, `attn-k-null-mask-guard` — see [`../upstream/`](../upstream/README.md).

---

## Block 06 — MMB: the bf16-WMMA dequant-weight prefill GEMM

**What it is.**  A prefill GEMM (`mmb.cu`, ~2.4k lines) that dequantizes weights into bf16 WMMA fragments
on the fly, with per-arch and per-shape tuning: dense, routed/compact (MoE), GLU, F32-split, tall/narrow,
tiny-M, IQ3_XXS and HC16 variants, plus a shadow cache.  Adapted from `pwilkin`'s `strix-halo` fork.

**Contents.**  The `mmb` kernel family and its dispatch (`ggml-cuda.cu`); a per-quant type list
(`GGML_CUDA_MMB_TYPES`); per-arch RDNA3/RDNA4 tuning knobs; and
`tests/test-logits-width-probe.cpp` — the **width probe** that every purity claim in the delivery is
measured with.

**Knobs.**  `GGML_CUDA_MMB` (master switch), `_TYPES`, `_DENSE`, `_ROUTED`/`_ROUTED_THRESH`, `_GLU`/`_GLU_THRESH`,
`_F32SPLIT`(+`_MIN_K`/`_MIN_M`), `_TALL`(+`_MIN_M`), `_TINY_M`/`_TINY_TT`, `_DOWN16`, `_HC16`, `_IQ3XXS`,
`_RDNA3`, `_CFG`, `_CACHE`/`_SHADOW*`, `_MIN_T`, `_TILE`, `_BF16W`, `_LOG`; `LLAMA_MMB_CVT_LOG`.
Everything is default-on and documented in `ENVIRONMENT.md` §5.

**Purity.**  MMB changes prefill arithmetic, not decode: run the **prefill-logit gate** on any tuning
change, and the width probe when a fusion/dispatch predicate moves.

---

## Block 07 — k-quant boosts: `mmvq` VDR, Q6_K MMQ, Q1_0, grid pad

**What it is.**  The decode-numerics block.  `VDR` (vectors-per-thread reduction) for `Q4_K`/`Q5_K`/`Q8_0`
(2→4) and `Q6_K` (1→2), the RDNA3_5 `mmvq` parameter table split out of the RDNA2 fallback, a Q6_K MMQ
prefill path, the Q1_0 `__byte_perm` → `v_perm_b32` port, and a grid-stall pad.

**Contents.**  `vecdotq.cuh` (VDR constants + the `ggml_cuda_byte_perm` helper), `mmvf.cu` (the Q6_K
decode path), `mmvq.cu` (the VDR-aware K-split loop and the RDNA3_5 table), `mmq-vec-dot.cuh` /
`mmq-load-tiles.cuh` / `mmq.cu` (Q6_K MMQ), `common.cuh` (grid pad), and the Q1_0/Q6_K cases in
`tests/test-backend-ops.cpp`.

**Knobs.**  `GGML_CUDA_GRID_STALL_PAD`, `GGML_CUDA_MMQ_J_MAX`, `GGML_CUDA_OP_TIMING` (diagnostic).

**Purity.**  **This block changes decode numerics on every backend** — same real-number result, different
fp32 addition order.  Read `GREEDY-PURITY.md` §§1-9 before touching a VDR value: stock bits then require
re-cutting the set without this block (it is no longer the last patch, so it cannot be dropped with a
one-line edit).  A PPL run before/after is the gate: it must be **unchanged**.

---

## Block 08 — fused kernels (including the fused MoE gate+up+GLU)

**What it is.**  The CUDA fusion layer: matchers in `ggml_cuda_try_fuse` plus the kernels they dispatch to,
each one replacing a small chain of standalone launches, all constrained by the rule that a fused path
must be **bit-identical** to what it replaces at the widths it serves.

**Contents.**

* `mmvq.cu`: the dense `ncols == 1` K-split alignment, the fused dense gate+up+GLU, and (r93) the
  **RDNA3_0 per-shape `nwarps`** (`Q4_0/Q4_1/Q5_0/Q5_1` narrow at every row count, `Q8_0` narrow for
  `2048 <= M <= 8192`), forwarded to every fusion that reproduces a dense weight launch
  (`GREEDY-PURITY.md` §19 addendum).
* `mmq.cuh` / `mmq.cu`: the fused MoE gate+up+GLU MMQ prefill, the routed-compact dispatch, `mul_mat_q_pair`,
  the FP4/gate precision contract, and the Q6_K compact-J path.
* `moe-weighted-reduction.cu`: the MoE weighted-down reduction (+ column-blocked band epilogue).
* `norm.cu` / `quantize.cu` / `unary.cu`: `rms_norm`→Q8_1 folds, the verify-band q8_1 cache, scale-unary,
  norm-rows, the conv1d/PLE fusions and the shared-expert down gate.
* `topk_moe`: the fused MoE router, made bit-identical to the softmax→argsort→get_rows→norm chain
  (`GREEDY-PURITY.md` §31) with a sorted-key argmax and a stable tie-break.
* The **RDNA3_5 single-token-only fusion skip** (the dense GLU and the weighted-down tail are not
  decode/verify bit-identical), `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` as the A/B.

**Knobs.**  `GGML_CUDA_DISABLE_FUSION` (master), `_DISABLE_MOE_MMQ_FUSION`, `_DISABLE_MMQ_ROUTED`,
`_DISABLE_TOPK_MOE_FUSION`, `_DISABLE_CONV_FUSION`, `_DISABLE_NORM_ROWS`, `GGML_CUDA_TOPK_SORTED`,
`GGML_CUDA_FUSE_GDN_GATE`, `GGML_CUDA_FUSE_Q8_1_VERIFY`, `GGML_CUDA_MMVQ_RDNA3_SMALL_M`,
`GGML_CUDA_SCALE_UNARY`, `GGML_PAIR_2X`, `GGML_CUDA_Q8_1_ARENA_FREE_OLD`,
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS`.  Block 10 adds the rest of the `_DISABLE_*` / `_FUSE_*`
family (the qwen4exp/GDN fusions); the full catalogue is `ENVIRONMENT.md` §7.

**Purity.**  Every fusion carries the same requirement: one arm across `W = 1..8`, and — because some
matchers fire only at one width — the fused kernel must reproduce the standalone launch at that width
(`GREEDY-PURITY.md` §§39, 42, 25).  The `nwarps`/VDR choice is part of the reduction order: a
shape-scoped value must be forwarded to *every* kernel that reproduces that launch.

**Gates.**  `test-backend-ops -o MUL_MAT`/`MUL_MAT_ID`, the width probe on a MoE model, and the
stock-relative `llama-batched-bench -npl 1,4,8` arm.

---

## Block 09 — host-resident MoE: iface, scheduler, expert cache, staging rings

**What it is.**  Everything that makes `-ncmoe`/`-cmoe` (and `--host-experts`) fast and safe without
fitting the whole model on the device: a pinned host master, a per-op H2D staging ring, a scheduler that
can hand an op to a host-resident table, and the MoE expert cache that keeps the hottest experts in
whatever VRAM is left.

**Contents.**

* `moe-expert-cache.cu` (~5.1k lines) + `.h`: the cache engine — arena sizing, per-layer uniform slots,
  the slot remap, the routed-MMVQ decode/verify band (`moe_cache_band`), admission/heat policy, the
  persistent cross-run heat map (`MOE_EXPERT_HEATMAP`) and prefill cache staging.
* `ggml-cuda.cu`: the **op-offload H2D staging ring** (three invariants: prefill is never graph-captured,
  the width gate is floored at 64 tokens, the arena lives outside the compute-graph reserve), the pinned
  `ROCm_Host` master, the per-device host buffer types, and the base of the async graph-input ring.
* `ggml-backend.cpp`: the scheduler's staged cross-backend upload (including the cross-device
  `wait_before_overwrite` ordering that fixed issue #103), routed-expert rebalancing and env-read caching.
* **A second consumer of a staged expert table gets its own split** (r96, PR #186): a cache-managed
  `MUL_MAT_ID` whose weights are already an input of the current split now starts a new split, so the
  existing re-registration path stages them again for that consumer's own routing.  qwen4exp's gathered
  logits tail and its full-row MTP export tail are the motivating pair: merged, neither consumer could be
  served by the expert cache and the scheduler's guarded fast paths declined, so the last layer's whole
  table was copied for every prompt chunk on `-sm tensor` (-30 %/-47 % TTFT on follow-up turns,
  bit-identical text) and on `-sm layer` the second consumer read the arena slots staged for the first
  routing, which silently produced non-greedy-pure text with 0.00000 draft acceptance.  `GGML_SCHED_MOE_DUP_SPLIT=0`
  restores the merged split.  Invariant and evidence: `GREEDY-PURITY.md` §43.
* `ggml-backend-meta.cpp`: `MIRRORED` tensors serving arbitrary byte ranges and the per-device split of a
  host expert table.
* **The meta `get_tensor_async` reads an arbitrary row range** (r96, PR #189): it shares the synchronous
  read's split arithmetic instead of asserting `offset == 0` and one segment, so a caller can batch many
  per-device reads behind one `synchronize`.  Calls that worked before issue the same copies as before.
  The caller is block 11's batched state save.
* **The host-expert account keys a real device only** (r97): `moe_host_expert_account` skips a host
  buffer type that carries no device.  `ggml_backend_cpu_buffer_type()` is the plain **pageable** CPU
  buffer and has `.device = NULL` upstream (the FIXME in `ggml-backend.cpp`), so a CPU-only run, or
  `--override-tensor exps=CPU`, used to insert a null key and abort in `ggml_backend_dev_memory()` when
  the context auto-sized the staging ring — a CPU-only model could not create a context at all, and
  `test-llama-archs -a deepseek4` aborted on its first row.  A pageable master is not served by this
  block's machinery (the slab arena, the ring and the cache read the master in place from a device
  kernel, which needs a pinned buffer): it is copied by the upstream `llama_moe_cache` engine instead,
  so it must not be accounted.  Only a pinned host buft — which always carries its device — is
  accounted, and the per-device distribution is unchanged.
* `common/fit.cpp`: `--fit` reserves the arena and the slab headroom per device.
* Host-side: mmap reclaim (`LLAMA_LOAD_RELEASE_MMAP`), rear-fill host layers
  (`LLAMA_HOST_EXPERTS_FROM_END`), `--host-experts pool` (`MOE_HOST_POOL_MIB`), a second draft scheduler
  (`LLAMA_DRAFT_SCHED2`), compute-buffer drop/re-reserve, and the CPU MoE-offload thread cap.
* **The batched f32 CPY run** (`cpy-batch.cu` / `.cuh`, r95): a run of up to 16 consecutive same-layout
  f32 `GGML_OP_CPY` nodes (the per-position DeltaNet conv-state snapshots of a speculative verify batch
  are the motivating shape) as **one launch**, element copies in `ggml_cpy`'s flat order, so
  bit-identical.  Admitted on **RDNA4 and RDNA3_0 (gfx1100-1102; gfx1101-gfx1102 not explicitly
  validated)**, `GGML_CUDA_FUSE_CPY_BATCH=0` disables it, and the marker `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1`
  prints every run.  `ggml_cuda_cpy_batch_try_fuse` (block 10) is the caller; the fusion is skipped while
  a split's concurrent events are live, but it still fires heavily on a multi-device graph (measured
  ~450 runs under `-sm layer` / ~1350 under `-sm tensor` per 60-token MTP run on 3x gfx1201).
  RDNA3_0 multi-device is an **open, accepted gap** (no RDNA3_0 multi-GPU box).
* **The appended device-iface fills.**  Every `ggml_backend_device_i` / `ggml_backend_i` entry this set adds
  (`stage_gather`, `slab_ring_set`, `slab_narrow_floor`, `slab_reserve_wide`, `slab_narrow2_floor`,
  `slab_compute_narrow2`, `slab_compute_plain`, the slab-ring hooks, `moe_cache_*`) is wired here as a real
  hook (CUDA/Meta) or NULL/no-op (CPU, RPC, every other backend).  The list is **positional**: the block
  that declares a field must also fill it in every backend, or the initializer silently shifts (r94
  block-boundary repair; previously deferred to block 11, which does not build on its own).
* **The pre-loaded draft-model handoff** (r86; ownership completed in r94):
  `common_speculative_load_model`, the 4-argument `common_speculative_init_result` / `init_from_params`,
  and `common_init_result::take_draft_model` live in this block together, so the draft model is resident
  before the target context sizes the slab.  A caller and its definition must not be split across blocks.

**Knobs.**  `MOE_EXPERT_CACHE_MIB` (auto default; `0` disables), `_MIN_MIB`, `_MIN_RES_PCT`,
`_RESERVE_MIB`, `_AUX_RESERVE_MIB`, `_UNIFIED` (APU opt-in), `_VALIDATE*`, `MOE_EXPERT_HEATMAP[+_DIR]`,
`MOE_HOST_POOL_MIB` / `MOE_HOST_POOL_FRAC`, `GGML_MOE_CACHE_MAX_TOK`, `GGML_MOE_GATHER_ONCE`,
`GGML_CUDA_SLAB`/`_RESERVE_MIB`/`_HEADROOM_MIB`/`_MIN_ARENA_MIB`/`_RING_MIB`/`_CHUNK_MIB`,
`GGML_COMPUTE_BUFFER_MARGIN_PCT`/`_CHUNK_MIB`, `GGML_CUDA_ASYNC_INPUT`, `GGML_SCHED_STAGE*`,
`GGML_STAGE_*`, `LLAMA_LOAD_RELEASE_MMAP`, `LLAMA_DROP_COMPUTE_BUFFERS`, `LLAMA_DROP_EXTRA_RESERVE_MIB`,
`GGML_CPU_MOE_OFFLOAD_THREADS`, `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD`, `LLAMA_TENSOR_HOST_BUFT`,
`GGML_HIP_GRAPH_FORCE_UPDATE`, `GGML_META_SS_VERIFY`, `GGML_SCHED_MOE_DUP_SPLIT`.

**Invariants.**  Host experts are always pinned; a partially failed cache **falls back wholesale**; no
arena is freed while a kernel carrying its address is in flight; the slab is **MoE-only** (armed solely by
the MoE preflight — never enable it for a dense model); a `-ncmoe` host table is split per device, not
mirrored (`GGML_META_SPLIT_COPY=1`); a staged expert table gets **one consumer per split** (a second
`MUL_MAT_ID` consumer of the same weights starts its own split, because the cache arena can only hold one
routing's slots — `GREEDY-PURITY.md` §43).

**Gates.**  Host-expert coherence (`-ncmoe`/`-cmoe`, and `cache-on == cache-off` text), `test-llama-archs`,
`MOE_EXPERT_CACHE_VALIDATE=1`, `scripts/check-gpu-faults.sh` on the run's journal + log.

---

## Block 10 — qwen4exp, the slab, the fusion dispatcher, the async input ring

**What it is.**  The largest block, and the one that owns the translation units split out of
`ggml-cuda.cu`: the qwen4exp (Qwen3.8-Flash-Next) model support, the movable-boundary **slab** allocator,
the `ggml_cuda_try_fuse` dispatcher, the async graph-input ring's batched uploads, and the qwen4exp-family
fusions.

**Contents.**

* **qwen4exp** (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`): QSA sparse attention and the
  fused indexer top-k/score (`indexer-topk.cu`, `lightning-indexer`), hyper-connections (`hc-mix.cu`,
  `hyperconn.cu`), the managed lazy reader, PLE n-gram loading, the MTP draft head, and the **keys-only
  indexer cache** (`LLAMA_QSA_KEYS_ONLY`; no V tensor at all).
* **The slab** (`ggml-cuda-slab.cu`, `ggml-cuda-arena.cu`): one VMM reserve per device split by a movable
  boundary — compute below, MoE arena above, no HIP call at runtime.  Plus the narrow-band learning, the
  wide-phase auxiliary region, pool-wide routing, and the slab transient absorption.
* **The fusion dispatcher** (`ggml-cuda-try-fuse.cu`): the matcher arms moved out of `ggml-cuda.cu`.
* **Rope/FWHT fusions** (M-RoPE and NEOX, with the q8_0 K/V cache `SET_ROWS` folded in), the paired GDN
  q/k norms, and the `ADD + rms_norm_q8_1` one-token fusion (`GREEDY-PURITY.md` §42).
* **The verify-band arch extension** (r95, PR #192 + PR #193):
  `ggml_cuda_should_fuse_mul_mat_vec_q(..., verify_band = true)` admits **RDNA3_0** as well as RDNA4 via
  one `verify_band_arch` term.  That predicate is the single gate the three 2..8-token Q8_1
  pre-quantize fusions share — residual `ADD` + `rms_norm` → Q8_1, plain `rms_norm` → Q8_1, and gated
  unary → Q8_1 (plus the `GLU` → Q8_1 mark) — so one change admits all of them; their kernels are
  row-generic and write the values `quantize_q8_1` writes, so `W = 1..8` stay bit-identical.
  `gdn-conv.cu`'s `verify_ok` takes the same `(RDNA4 || RDNA3_0)` form, so the direct GDN conv covers
  the 2..255-token verify batches too and keeps its pre-existing single-device restriction.
  `GGML_CUDA_FUSE_Q8_1_VERIFY=0` / `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` still disable them; RDNA3_5 is
  excluded by the macro (`0x1100 <= cc < 0x1150`).
* **The DSV4_HC_POST arm is chosen by the split mode** (r97).  The `hc_combine_norm` chain spelled out
  (scale → sigmoid → scale → repeat → mul → add → rms → mul) is faster than the op on **one** device
  (+0.9..+1.8 % prefill on qwen4exp IQ4_NL, gfx1151) and stays the default there, but the chain's
  broadcasts are split-state sensitive and it is **not correct under the Meta (tensor-split) device**:
  `test-llama-archs -a deepseek4` on 3× gfx1201 read NMSE 1.74e-01 vs the CPU reference through the
  chain and 6.5e-06 through the op.  `cparams.fused_dsv4_hc_post` therefore defaults to *on* whenever a
  Meta device is in play and to *off* otherwise; `LLAMA_FUSED_DSV4_HC_POST` still forces either arm.
* **The ring's teardown flush is unconditional** (r97): `ggml_cuda_async_input_flush_done` early-returned on
  `!g_async_input_on`, but since r90 the ring also carries the `set_tensor_async` snapshots *unconditionally*,
  so `GGML_CUDA_ASYNC_INPUT=0` left a pending done-event naming the stream the context destructor destroys
  and the next context on that device recorded on the dead handle (`invalid resource handle`).  A no-op when
  nothing is pending; both arms now give the same results.
* **Async graph-input ring**: batched uploads + lazy done event (PR #182), and the r90 fix that snapshots
  the source through the pinned ring — the scheduler uses `set_tensor_async` for a *CPU split's* output,
  which the next ubatch rewrites, so a direct copy from the caller's live pointer could read the next
  ubatch's bytes.  Default-on; `GGML_CUDA_ASYNC_INPUT=0` disables.
* **`GGML_CUDA_GDN_TAIL`**: the fixed-`D` sequential tail that makes a chunked-GDN prefill batch write its
  rollback snapshots (r91; `GREEDY-PURITY.md` §27).
* Graph-memory generations (`GGML_CUDA_GRAPH_MEM_GEN`) so a graph is recaptured when the pool / FA staging /
  H2D ring memory it captured is freed, **and when `ggml_cuda_slab_work_alloc` shrinks the work boundary**
  (the post-prefill drop hands the wide region back to the arena; a captured graph baked its addresses).
  Issue #187: without that invalidation a 4-stream MTP decode replayed a stale `set_rows` source and
  faulted with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.

**Knobs.**  The slab/cache set listed under block 09, plus `GGML_CUDA_POOL_WIDE[_MIN_MIB]`,
`GGML_CUDA_SLAB_STATS` / `_GRAPH_TRACE`, `GGML_CUDA_ASYNC_INPUT_BATCH` / `_LAZY_DONE`,
`GGML_CUDA_GDN_TAIL`, `GGML_CUDA_GRAPH_MEM_GEN`, `GGML_META_SPLIT_COPY`,
`LLAMA_KV_N_PAD_PREFILL`(+`_FROM`/`_MIN`), `LLAMA_QSA_*`, `GGML_QSA_*`, `LLAMA_HC_*` / `GGML_HC_*`,
`LLAMA_MTP_SPARSE`.

**Purity.**  qwen4exp's decode/verify band is `HC_FUSED_MAX_TOKENS = 8` and the QSA decode arm band is
`max(8, n_rs_batch)`; the arm must be chosen from the band and `n_kv`, never `n_tokens == 1`
(`GREEDY-PURITY.md` §§13, 16, 26, 32, 38).  The QSA op has a CPU oracle — run
`test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum) and the sparse-vs-dense perplexity ratio, because the
`W = 1..8` matrix cannot see a width-uniform bug (§21).  Also run `scripts/gate-qwen4exp-quant-coherence.sh`
before changing the device gather or the head zero.

**Gate.**  Debugging the slab or the arena: `GGML_CUDA_SLAB_STATS=1` (arm/drop/destructor accounting,
`hole=0` and `outside_failed=0` are the acceptance signals).

---

## Block 11 — campaign remainder

**What it is.**  The tail of the campaigns: CPU wins, the backend-iface fills that every non-CUDA backend
needs for the new hooks, DFlash, the FA instance build grouping, and the small allocator/model fixes that
had no other home.

**Contents.**

* **CPU wins**: an AVX2 block prefilter in the sampler (block max + NaN detection over 64 logits, with the
  scalar fallback preserved), and an AVX2 Q2_0 dot product that reproduces the generic two-rounding
  arithmetic bit-for-bit.
* **DFlash**: the F1 device-resident layer-input path made default-on with a warn-once fallback
  (`GGML_LF_DFLASH_DEV=0` disables), `GGML_LF_FAST_TOPK`, and the clean `-sm tensor` rejection for a
  DFlash2 selector (a Meta distributed top-k is unstarted — TODO #78).
* **Allocator/model**: `ggml_gallocr_drop_buffers` (free compute buffers and invalidate the layout so the
  next reserve sizes from the current graph), the per-device `max_host_expert_slice_bytes` split rule, the
  application of the embeddings-layer-input preset, and the `-sm tensor` H2D-ring auto-size + narrow-band
  pin.
* **The batched state save** (r96, PR #189): `llama_io_write_host` queues its deferred reads as
  `ggml_backend_tensor_get_async` into one pinned staging buffer kept by the context, synchronizes each
  backend once, then copies to the caller's buffer.  A recurrent model's checkpoint is about 75 small
  ranges, and under `-sm tensor` each was one synchronous read per device, so a checkpoint paid about 150
  device syncs before every prompt chunk of a follow-up turn (Flash-Next + MTP: -55 ms/-59 ms TTFT,
  isolated by the same-binary `LLAMA_STATE_ASYNC=0` arm).  States above `LLAMA_STATE_ASYNC_MAX_MIB` and any
  state with a tensor no context backend owns keep the synchronous path, and restores are unchanged.  The
  file also gains `#include <algorithm>` (the patch as submitted does not compile without it).
* **Build time**: one FA template instantiation TU per `(ncols1, ncols2, head size)` instead of per
  `(ncols1, ncols2)` — same instantiations, grouped so the backend build is no longer one-file-bound.

**Knobs.**  `LLAMA_OUTPUT_PINNED`, `MTP_DRAFT_N_UBATCH`, `GGML_LF_DFLASH_DEV`, `GGML_LF_FAST_TOPK`,
`LLAMA_STATE_GET_MERGE_GAP`, `LLAMA_STATE_ASYNC`, `LLAMA_STATE_ASYNC_MAX_MIB`,
`GGML_CPU_MOE_OFFLOAD_THREADS`, `GGML_CUDA_SLAB_RING_MIB`.

**Notes.**  A backend-iface entry is a compile-time contract: adding one to the struct means adding it to
**every** backend's initializer, or the build breaks for that backend.  Keep the difference documented when
a hook is CUDA-only (the non-CUDA entries stay NULL and the feature is simply inactive).

---

## Facts that span blocks

* **The band rule.**  Decode and every speculative verify width must take the same kernels, chosen from the
  `n_tokens` band and `n_kv` — never from the exact width.  Blocks 00/03/05/06/08/10 each contribute a piece
  of it; `GREEDY-PURITY.md` is the rulebook.
* **A block owns its code.**  A fix goes into the block whose surface it changes (even an early block); the
  per-block build gate (`scripts/gate-per-block-build.sh`) exists because appending fixes to the last block
  once broke blocks 06-14 silently.
* **Cross-backend consistency.**  The set is AMD-targeted but generic code: a new predicate, KV type or
  iface entry must stay consistent on every backend (an inconsistent set is a crash on whichever backend
  reaches it).  Behaviour/perf on non-AMD is out of scope — see AGENTS.md.
* **RDNA4 is the reference arch.**  The WMMA FA path, the chunked-GDN kernel and the VDR work were built
  and validated there first; RDNA3.0/3.5 get back-ports (their own kernel files, tuned head caps, their own
  `mmvq` parameter table) and RDNA2 is best-effort.
* **The arch gate is runtime** everywhere in the set, so one multi-arch binary picks the right path.
* **`ggml-cuda.cu` is shared by blocks 00/07/08/09/10** and is the usual re-base conflict site; the
  `template-instances/*.cu` files are block 05's and are generated (`generate_cu_files.py`).
* **A knob's `getenv` call site may land in a later block than the feature it controls** (the series is
  linear and `ggml-cuda.cu` accumulates): `ENVIRONMENT.md` is authoritative for what a variable does and
  its default, and `git grep` on the applied branch for its call sites.  A block's "knobs" list here is the
  set its own hunks read/write, not a claim of exclusive ownership.

## Dated history

[`../archive/docs/patch-notes-history.md`](../archive/docs/patch-notes-history.md) — every amendment note
from `r1` to `r93`, including the pre-V2.0 block-by-block notes, kept verbatim because `WORKLOG.md` and the
campaign records cite it.
