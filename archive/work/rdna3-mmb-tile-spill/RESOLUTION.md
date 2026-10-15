# RESOLUTION — `rdna3-mmb-tile-spill` (PR #155, @briansp2020)

**Status: RESOLVED (2026-10-15), promoted into the delivery as part of `v12-097cf33d9-r65` (block 06
amended).**  No `wip/` entry; the PR's format-patch and write-up are preserved in this directory.

## What landed

A new per-arch field `mmb_arch_cfg.big_q4k_only`, set for RDNA3_0 only: on gfx1100 the 128x256 dense MMB
tile is kept for Q4_K weights (which do not spill) and every other weight type takes the 128x128 tile
(which does not spill either), avoiding the scratch-spill condition that faults the queue with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` under WSL2 when a spilling MMB prefill kernel and the
spilling head-256 WMMA FA kernel run together.  `GGML_CUDA_MMB_TILE` still overrides and the field is
printed in the `GGML_CUDA_MMB_CFG=1` dump.  RDNA3_5 and RDNA4 are untouched (`big_q4k_only` stays 0).

## Validation (r65 candidate)

- **gfx1201 (this session):** the change is a runtime no-op (`big_q4k_only == 0` on RDNA4).  The
  prefill-logit KLD gate is **PASS** (mean KLD 0.000707, same-top-p 98.755 %, identical to the recorded
  base) and the rule-5 `llama-batched-bench` B=1/4/8 is no worse than the r64 reference.
- **gfx1100 (reporter, RX 7900 XTX, WSL2, ROCm 10.2):** r62 faulted 4/4 at `pp2048`; this change 0/4,
  with `1234.9 / 989.2` t/s at d0 / d16384 versus `1181.3 / 952.9` for `GGML_CUDA_MMB_TILE=0`
  (+4.5 % / +3.8 %), 26k-token MTP greedy text unchanged.

**Open caveat carried forward:** only validated under WSL2 by the reporter; native Linux gfx1100 was not
tested (the WSL HSA scratch handling differs), and only one model.

## Record

`WORKLOG.md` 261015.002 (r65).  Original PR description and patch: `README.md`,
`0001-mmb-gfx1100-q4k-big-tile.patch`.
