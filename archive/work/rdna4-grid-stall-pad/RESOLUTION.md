# RESOLUTION — rdna4-grid-stall-pad (PR #128)

**Status:** FOLDED into the delivery as **block 16** on 2026-10-14 (release `v16-a55e952b8-r45`).
**Owner:** `@briansp2020` (PR #128, follow-up to #78).

## What was adopted

`0001-ggml-cuda-pad-mmvf-mmvq-launches-off-the-gfx1201-grid-size-dispatch-stall.patch` —
`ggml_cuda_grid_stall_pad()` in `common.cuh` detects a launch whose block count is a multiple of
256 (65536 work-items at 256 threads) and the three affected launch sites (`mul_mat_vec_f`,
`mul_mat_vec_q_ksplit`, `mul_mat_vec_q_moe`) add one block that returns immediately.  Every row
keeps its computation, so the launch is bit-identical.  Default-on kill-switch
`GGML_CUDA_GRID_STALL_PAD=0`; the helper and its three call sites can be deleted once AMD fixes the
stall (ROCm/legacy-rocm-build#6689, ROCm/TheRock#8634).

## Validation on the delivery tree (gfx1201)

- 2x and 3x R9700, Flash-Next GSQ IQ3_XXS + shared MTP head, `--lazy-mode off --fit on`,
  `-n 3000` MTP `n_max 3`: same text sha as the r44 baseline (`2b2341670f23`);
  `--spec-type none == draft-mtp` byte-identical.
- 2-GPU A/B on one binary: pad on 130.1 t/s vs `GGML_CUDA_GRID_STALL_PAD=0` 130.3 t/s — within
  noise on this box (the reporter's 2x R9700 on PCIe 5.0 x8 measured +1.6 %; our cards are x4).
  No regression.
- `test-backend-ops -o MUL_MAT_ID`, `GATED_DELTA_NET`: OK.
- Prefill-logit KLD gate: mean **0.000707** / same-top-p **98.755 %** (PASS).
