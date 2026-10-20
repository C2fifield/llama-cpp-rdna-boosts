# patches-rdma2/ — RDNA2 (gfx1030) flash-attention follow-ups

Add-on set for RDNA2 (gfx1030 / Radeon Pro V620), applied **on top of** the
12-block rdna-boosts delivery (current: `v12-097cf33d9-r93`). Apply with
`scripts/apply-rdma2.sh`; the base and applied trees are asserted in
`release-rdma2.json`.

## 0000-rdma2-fa-followups-todo71-72.patch

Follow-ups from the review of PR #181 (folded into block 05 of r88), tracked
as TODO 71 and 72:

1. **TODO 71 — shared tile K/V type selection.** The tile kernel's K/V type
   was chosen by hand in two places (`ggml_cuda_flash_attn_ext_tile_case_type`
   in fattn-tile.cu, the TILE branch of `ggml_cuda_flash_attn_ext_get_alloc_size`
   in fattn.cu) and drifted three times (missing q4_0 arm, mixed-K/V NaNs,
   missing RDNA2 bf16 scratch). Both now call one
   `ggml_cuda_fattn_tile_type(K, V, cc)` in fattn-rdna.cuh, so "the allocator
   and the launcher agree" is a property of the code.

2. **TODO 72 — checked tail for the last unchecked tile.** The `ncols2 > 1`
   tile branch loads KV tiles without OOB checks, so its last tile can read up
   to `nbatch_fa - 1` rows past the KV data. In a real model that stays inside
   the KV cache tensor's slack, but an exact-size operand (test-backend-ops)
   can reach the neighbouring allocation. On RDNA2 the last tile now takes the
   checked form (zero-fills the tail rows, keeps them out of the softmax);
   every other target is untouched. Measured on the V620: register usage
   (VGPR/AGPR/scratch) unchanged on all 70 flash_attn tile instantiations.

3. **Guard-tail comment.** The 511-row converted-operand guard tail (r88) is
   kept; its comment now documents the derivation: `nbatch_fa` is 9-bit but
   `GGML_CUDA_FATTN_TILE_CONFIG_CASE` static_asserts `nbatch_fa <= 256`, the
   over-read is at most `nbatch_fa - 1 <= 255` rows, 511 = 2 x 256 - 1.

## Gate (ROCm 7.2.4, `GPU_TARGETS=gfx1030;gfx1100` dual build)

| card | op | result |
|------|----|--------|
| Radeon Pro V620 (gfx1030) | FLASH_ATTN_EXT | 6389/6389 passed |
| Radeon Pro V620 (gfx1030) | MUL_MAT | 2444/2444 passed |
| Radeon RX 7900 XTX (gfx1100) | FLASH_ATTN_EXT | (gated on the same dual binary) |
| Radeon RX 7900 XTX (gfx1100) | MUL_MAT | (gated on the same dual binary) |

The r93 baseline (before this add-on) passes the same two gates on the V620
(6389/6389 FA, 2444/2444 MM), confirming the r88 fold on r93.

## Non-RDNA2 reach

The `ggml_cuda_fattn_tile_type` refactor is semantically neutral (same
selection order as before, in one place). The checked-tail change is gated on
the per-arch `RDNA2` macro, so gfx1100/gfx1151/gfx1201 and NVIDIA/MUSA builds
of the same tree keep the unchecked branch byte-for-byte.
