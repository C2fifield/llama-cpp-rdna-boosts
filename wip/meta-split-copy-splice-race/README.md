# `meta-split-copy-splice-race` — the `-sm tensor` + `-ncmoe` below-gate upload races its own buffer

**Status: FIXED + VALIDATED (2026-10-14), awaiting the maintainer's go-ahead to fold into block 10.**
The one-line fix is `meta-split-copy-splice-race.patch` (applies to the current `release.json` tree).
Surfaced while parking `wip/moe-layer-stagger/`; the stagger campaign's `-sm tensor` basis was on this
path, which is why it was corrupt.

## 0. Symptom

`-sm tensor` + host-resident MoE experts (`-ncmoe > 0`) produces garbage on the **below-gate** upload
path (ubatch smaller than `sched_stage_min_tokens`, i.e. the default llama-cli `-ub 512`, and any run
with `GGML_SCHED_STAGE=0` / `GGML_SCHED_STAGE_MIN_TOKENS=100000`):

| model | config | output |
|---|---|---|
| 35B-A3B Q8_0 | `-sm tensor -ncmoe 20` | `3333…` |
| gemma4 26B-A4B Q4_K_XL | `-sm tensor -ncmoe 20` | `<unused32>…` |
| Flash-Next GSQ-IQ3_XXS | `-sm tensor -ncmoe 24` | `3333…` (intermittent) |

`llama-perplexity` is unambiguous: `[1]nan … [8]nan`, `Unexpected negative standard deviation of
log(prob)`.  Clean: `-sm layer`, `-ncmoe 0`, `GGML_META_SPLIT_COPY=0` (mirrored), and
`GGML_SCHED_STAGE_MIN_TOKENS=0` (the staging path).  Not the heat map (`MOE_EXPERT_HEATMAP=0` and
default both fail, no sidecar), not the expert cache (`MOE_EXPERT_CACHE_MIB=0` and auto both fail), not
`--lazy-mode`, not the load mode, and not a regression (`build-r55-ref` fails identically).

## 1. Root cause

`ggml_backend_cuda_set_tensor_2d_async` (`ggml/src/ggml-cuda/ggml-cuda.cu:3720`) has two arms:

* `GGML_CUDA_SPLICE_GATHER=0` — plain `cudaMemcpy2DAsync(..., cuda_ctx->stream())`, i.e. the **main
  stream**.  Correct.
* default (`splice_gather`) — gather the strided host slice into a pinned ring slot, then one 1-D
  `cudaMemcpyAsync(..., h2d_stream)` with `h2d_stream = cuda_ctx->copy_stream()`.

The scheduler calls `wait_before_overwrite()` before this upload because the graph allocator **reuses
the split-input buffer across splits**: the previous split may still be reading it.  That wait is
enqueued on the **main** stream.  The plain arm inherits it; the gather arm issued the H2D on the
**copy stream** and only ordered the *consumer* afterwards
(`cudaStreamWaitEvent(cuda_ctx->stream(), s.ev, 0)`).  So the H2D could overwrite the split-input
buffer while the previous split's kernels were still reading it.

Isolation (gemma4, `-sm tensor -ncmoe 20`, forced below-gate):

| arm | result |
|---|---|
| `GGML_CUDA_SPLICE_GATHER=1` (default) | `<unused32>` |
| `GGML_CUDA_SPLICE_GATHER=0` (plain 2-D) | coherent |
| gather H2D moved to the main stream | coherent |

## 2. The fix

`ggml/src/ggml-cuda/ggml-cuda.cu`: use `cuda_ctx->stream()` for the gather's 1-D H2D (one line + a
comment).  The host pinned gather and the single 1-D H2D are unchanged; only the stream moves.  The
copy stream bought nothing here anyway — the transfer has to wait for the reused-buffer read regardless
— and the measurement agrees (below).

## 3. Validation (gfx1201, 2× R9700, ROCm 7.14.1, `~/llama.cpp/build-rocm-hybrid`)

* **Coherence, product env (heatmap on, cache auto)**, 32-token and 250-token greedy runs:
  35B-A3B `-sm tensor -ncmoe 20`, gemma4 `-sm tensor -ncmoe 20`, Flash-Next GSQ `-sm tensor -ncmoe 24`
  — all coherent, 0 `////`, 0 `<unused32>`.
* **Perplexity** (wikitext-2, ctx 512, 8 chunks): 35B-A3B `-sm tensor -ncmoe 20` = **6.1802**, exactly
  equal to `-sm tensor -ncmoe 0` (6.1802) — the split copy now delivers the correct bytes;
  gemma4 `-sm tensor -ncmoe 20` = 1140.85 (finite).
* **No regression elsewhere:** dense 4B `Qwen3.5-Q8_0 -sm tensor` same-seed hash **`1c5d32ac537d`**
  (golden); `gate-prefill-logits.sh` **PASS** (mean KLD 0.000707 / same-top-p 98.755 %, identical to
  base).
* **Perf-neutral** (35B-A3B, `-sm tensor -ncmoe 20 -ub 2048 -p 2048 -n 128`): fixed gather
  **2487.9 pp / 73.35 tg** vs plain 2-D 2495.3 / 73.36 vs mirrored 1501.5 / 21.33.  The split still
  beats mirrored by **+66 % prefill / +244 % decode**.
* **Below-gate** (gemma4, forced copy, `-ub 512`): fixed 193.2 pp vs plain 188.1 vs the old copy-stream
  gather 194.0 — identical within noise.

## 4. Fold plan / open decisions

* **Owning block: 10** (`patches/0010-...qwen4exp...` owns
  `ggml_backend_cuda_set_tensor_2d_async`; the splice gather was added there).  Fold as a block-10
  amendment + re-cut blocks 11 (no new block).
* `release.json` is untouched by this campaign; nothing here is in the delivery until the maintainer
  approves the fold.
* **Not chased here (separate, pre-existing):** `test-backend-ops -o MUL_MAT_ID` hangs/spins on the
  `dup=1` cases with the MoE cache active (the process had to be killed; the two `type_a=f16, m=1024,
  n=33/64, dup=1` cases FAIL before the hang).  The delivery's `MUL_MAT_ID` gate is recorded 931/931 —
  re-check it with the cache off before treating the fold as gate-green.
