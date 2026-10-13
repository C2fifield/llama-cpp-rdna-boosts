# RESOLUTION — moe-rpb-short (PR #140)

**Status:** FOLDED into the delivery as **block 10** (`ggml/src/ggml-cuda/mmvq-moe.cu`, r52, 2026-10-13),
**default-on**.  **Owner:** `@briansp2020` (PR #140).

## What was adopted

`0001-ggml-cuda-8-rows-per-block-for-short-K-MoE-expert-GE.patch` — in `mul_mat_vec_q_moe_launch`, a
RDNA4 `MUL_MAT_ID` expert matrix with fewer than 8 K-blocks per row uses **8 rows per block** instead
of `ceil(8 / K-blocks)` (the Flash-Next q2_0 down under `-sm tensor` is K = 320 per GPU, 5 q2_0
blocks, where the old rule gave 2 rows and each lane did at most one dot product).

`GGML_CUDA_MOE_RPB_SHORT=0` restores the old rule; `2`/`4`/`8` force a value on any architecture
(other values are ignored).

## Where it moved

The PR was cut against r48 and patched `mmvq.cu`.  The V2.0 Phase-2 extraction moved
`mul_mat_vec_q_moe` + `mul_mat_vec_q_moe_launch` into `mmvq-moe.cu` (block 10), so the hunk was
re-based to the new location; the logic is unchanged.

## Validation on the delivery tree (gfx1201, 3x R9700)

- `test-backend-ops -o MUL_MAT_ID -p 'n=1,'` (the mmvq path): **166/166**, 0 FAIL.
- `test-logits-width-probe`: **PASS (worst maxdiff 0)** on 4B and 35B-A3B.
- Flash-Next GSQ-IQ3_XXS, 3 GPU all-resident, `-sm tensor`, code prompt, 400 greedy tokens:
  default **63.5 / 63.7 t/s** vs `GGML_CUDA_MOE_RPB_SHORT=0` **63.2 / 63.1 t/s** (~+0.6 %, the PR
  reports a modest +1.4 %).
- Dense 27B Q4_K_XL stock-relative `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`: no worse than the
  r50 reference at B=1/4/8.
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS).
