# moe-rpb-short: 8 rows per block for short-K MoE expert GEMMs on RDNA4

One format-patch on top of `v16-a55e952b8-r48`, applied with `git am` (it also applies cleanly on top of r49's
16-patch set, tree `45ba5da2`; the measurements below are from r48).  It only changes the rows-per-block choice in
`mul_mat_vec_q_moe_launch` (mmvq.cu) for expert matrices with fewer than 8 K-blocks per row, and only on RDNA4.
`GGML_CUDA_MOE_RPB_SHORT=0` restores the current rule; `2`, `4` or `8` force a value (on any architecture, short-K
matrices only); any other value is treated as unset.

## What it addresses

Under `-sm tensor` the Flash-Next down projection is split along K, so each GPU sees K = 320 per expert row: 5 q2_0
blocks.  The current rule picks `ceil(8 / blocks_per_row)` = 2 rows per block, so every lane does at most one dot
product and the warp finishes almost as soon as it starts.  With 8 rows per block the same launch keeps the warps
busy long enough to stream.  16 and 32 rows were slower again on gfx1201 (one full-warp reduction per row).

## Measurements (2 x R9700 32 GB, gfx1201)

Per-call kernel time, 4-token verify call:

| expert type | 2 rows per block (current) | 8 rows per block |
|---|---:|---:|
| q2_0 down (K = 320, 5 K-blocks) | 43.0 us | 29.8 us |

(An iq4_nl down projection, 12 K-blocks and so not touched by the patch, re-measured 44.0 vs 42.9 us as a control.)

End to end on stock r48 + this patch, Flash-Next GSQ-RCO IQ3_XXS + MTP (n-max 3), `-sm tensor`, all experts in VRAM,
same binary with `GGML_CUDA_MOE_RPB_SHORT=0` vs default, alternated, 2 runs each:

| | `GGML_CUDA_MOE_RPB_SHORT=0` | default (8 rows) |
|---|---:|---:|
| greedy 400 tokens, t/s | 130.6 / 129.1 | 131.9 / 130.8 |
| greedy sha | `c39d78416cd1` | `c39d78416cd1` |

Prose, code and 2-concurrent runs moved within their usual noise.  `test-backend-ops -o MUL_MAT_ID`: 931/931 on both
GPUs.  The output equality is measured, not structural (with 8 rows a row's K-blocks land on different lanes, so the
warp reduction can associate differently): same greedy sha over 400 tokens here, and over a 100K-token run on our
r39 build (`6cd43e863d00`).  We have run this rule in every build since r39; it has never changed the greedy output of
any config we run.

## Notes

- RDNA4 only (`GGML_CUDA_CC_IS_RDNA4`); other architectures keep the current rule unless the env var forces a value.
- Only MUL_MAT_ID matrices with 1-7 K-blocks per row are affected: K < 2048 for the 256-wide block types (k-quants,
  IQ2/IQ3), K < 512 for q2_0, K < 256 for q4_0/q8_0/iq4_nl, split or not.  In our model that is only the K-split q2_0
  down projection (K = 320).  The 4 -> 8 step for a 256-wide type with 2-7 K-blocks (e.g. the K = 512 down that the
  existing comment names) is not something we could measure here.
