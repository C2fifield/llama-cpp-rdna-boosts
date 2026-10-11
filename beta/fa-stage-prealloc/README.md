# `fa-stage-prealloc` — BETA (issue #138): FA prefill staging arena preallocation

> **STATUS (2026-10-17): BETA — not part of the delivery set.**  The patch applies on top of the
> current `v16-a55e952b8-r48` delivery tree and is handed to the issue reporter (@DanoPTT) for a
> Windows re-test.  It is not folded into any block yet.  The code it changes lives in **block 15**
> (`fattn_stage_try_get` in `ggml/src/ggml-cuda/common.cuh`); a promotion target is deliberately left
> open until the maintainer decides where it belongs.

## Artifact

| field | value |
|---|---|
| patch | `fa-stage-prealloc.patch` (in this directory) |
| sha256 | `f1a0f0b1c87dc77e6bce796af94b63262c109dce5a8a93d09685d262137d7f2a` |
| base | delivery `v16-a55e952b8-r48`, tree `3b9f403544169544f312bd24e4de31a554bf189d` |
| applied tree | `24fdc3c7f641e7b37a0d4b0a6d4024622bde94ce` |
| applies | `git apply` and `git am` clean on a fresh `apply-all.sh` checkout (verified) |
| kill switch | `GGML_CUDA_FA_STAGE_PREALLOC=0` restores the pre-change growth-only path |

## The problem

`ggml_backend_cuda_context::fattn_stage_try_get()` allocates the per-stream F16 prefill staging arena
with a direct `cudaMalloc` and grows it in 1.25x steps (`new_size = max(16 MiB, size + size/4)`),
allocating the new arena and then `cudaFree`-ing the old one.  The arena request grows with the
prefix, so one deep prefill of a quantized (q8_0/q4_0/bf16) K/V cache walks the ladder ~16 times; the
cumulative size of every step is about `final * (1 + 1/1.25 + 1/1.25^2 + ...) = 5x final`.

On Linux/KFD a freed step is returned to the driver and reused, so the high-water mark is just the
current arena (~1.25x the request).  On **Windows/HIP (WDDM)** a freed allocation stays committed
against the process: over a 119k-token q8_0 prefill the reporter measured +2.65 GiB of dead VRAM
(final arena ~530 MiB), and +3.6 GiB at 164k.  Near the VRAM limit the next long prefill then spills
to shared memory (pp 793 -> 255 t/s, MTP acceptance 65% -> 41%).  The reporter's instrumented build and
repeat runs pin the steps to the arena (geometric, deterministic, matching the new arena size).

The existing `--fit` accounting already reserves this arena as `min(2*GGML_CUDA_FA_STAGE_MAX_MB,
f16_kv_bytes)` per device, i.e. the full preallocated bound is budgeted on a default `--fit on` run.
The ladder is exactly what can exceed that budget on WDDM.

## The fix (Option A)

`launch_fattn` passes a size hint: the F16 size of the K/V **view roots** (`view_src`, which ggml
flattens to the root tensor), capped per operand by `GGML_CUDA_FA_STAGE_MAX_MB` exactly as the request
is.  An unbounded cap (`GGML_CUDA_FA_STAGE_MAX_MB=0`) passes no hint.  `fattn_stage_try_get` then:

1. tries the hint first, but **only** while `optional_alloc_within_free_cap(total + hint)` allows it
   (the r39 G4 slab guard), so a slab-active run cannot have its free VRAM consumed by the prealloc;
2. falls through to the existing `max(16 MiB, size + size/4)` step on a G4 refusal **or** a
   `cudaMalloc` failure;
3. keeps the existing warn-once + native-read fallback if neither succeeds.

Because the hint is only a first attempt, an oversized (or unbounded) hint can never cost more than
the pre-change path.  `GGML_CUDA_FA_STAGE_PREALLOC=0` disables the hint for A/B and bisection.

Note the peak behaviour: at the deepest prefill the prealloc (`min(root_F16, cap)` per operand) is
*smaller* than the ladder's `1.25x request`; the prealloc only increases commitment for shallow
prefills, up to the same bound `--fit` already reserves.  On Linux this is a commitment-timing change,
not a peak-memory regression.

## Validation on Linux (gfx1201, ROCm 7.14, r48 + this patch)

Build: `build-rocm-hybrid` at the r48 tip `0fb51947d`.  All runs single card unless noted.

| gate | result |
|---|---|
| golden coherence, f16 KV, 3-GPU `-sm tensor` (`Qwen3.5-4B-Q8_0`) | `1c5d32ac537d` — matches the current delivery |
| coherence A/B, q8_0 KV, 31k-token prompt (`GGML_CUDA_FA_STAGE_PREALLOC=0` vs default) | byte-identical (`ab8e156cca11`) |
| arena A/B via prefill-logit KLD, q8_0 KV (4B, ctx 16384, 16 chunks, `--ab PREALLOC=0`) | mean KLD **0.000000**, same-top-p **99.997%** (PASS) |
| standard prefill-logit gate (27B Q8_0, f16 KV, ctx 512, 40 chunks) | PASS, mean KLD **0.000707**, same-top-p **98.755%** (unchanged from base) |
| pp16384 A/B, 27B Q5_K_M, q8_0 KV (`llama-bench -r 3`) | ladder **1108.94 ± 0.92** vs prealloc **1108.03 ± 0.61** t/s (within noise) |
| env smoke: `GGML_CUDA_FA_STAGE_MAX_MB=0`, `GGML_CUDA_FA_STAGE_PREALLOC=0`, `GGML_CUDA_FA_STAGE_MAX_MB=1` | all identical output (`691f212ed0c0`) |

The arena path is exercised by the q8_0 KV runs (the f16-KV gates do not touch it).  The reporter's
Windows symptom cannot be reproduced on Linux/KFD by construction, so the confirmation that matters is
their re-test with `BETA-TESTING.md`.

## When promoted

* Move the hunk into the owning block (`fattn_stage_try_get` is defined in block 15; the maintainer may
  prefer another grouping).
* Add a `GGML_CUDA_FA_STAGE_PREALLOC` row to `ENVIRONMENT.md` (default on, kill switch, per the
  default-on policy) and fold the note into `patches/README.md`.
* Re-run `validate-set.sh`, the coherence gate, `gate-prefill-logits.sh` and the MTP/batched-bench
  gates on the regenerated set, then move this directory to `archive/work/fa-stage-prealloc/`.
