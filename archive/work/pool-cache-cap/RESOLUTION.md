# RESOLUTION — pool-cache-cap (PR #132)

**Status:** FOLDED into the delivery as **block 16** on 2026-10-14 (release `v16-a55e952b8-r45`),
**opt-in**.  **Owner:** `@briansp2020` (PR #132).

## What was adopted

`0001-ggml-cuda-GGML_CUDA_POOL_CACHE_MB-caps-cached-bytes-in-the-legacy-pool.patch` —
`ggml_cuda_pool_leg` tracks the bytes of the freed buffers it keeps cached and, before a new device
allocation, frees them once they exceed `GGML_CUDA_POOL_CACHE_MB` MiB.  Unset/0 keeps today's
behaviour exactly.  Only the legacy pool is touched; the VMM pool is unchanged; the flush is
skipped during graph capture.

## Validation on the delivery tree (gfx1201)

- 2x R9700, Flash-Next GSQ IQ3_XXS, `--lazy-mode off --fit on -sm tensor`, 32k-token prompt,
  `-c 65536`: runs cleanly, same output as without the cap, peak VRAM unchanged at this context
  (~25.6 GiB/card, dominated by weights).
- The reporter's VRAM win is at ~256K context (32,551 -> 30,945 MiB/card, `=128`); not reproduced
  here at 64K.  Kept **opt-in** because the useful cap is model/context dependent.
- Prefill-logit KLD gate: mean **0.000707** / same-top-p **98.755 %** (PASS, default arm).
