# RESOLUTION — `rdna3-mmvq-q4k-rpb` (PR #156, @briansp2020)

**Status: RESOLVED (2026-10-15), promoted into the delivery as part of `v12-097cf33d9-r65` (block 08
amended).**  No `wip/` entry; the PR's format-patch and write-up are preserved in this directory.

## What landed

`calc_rows_per_block_weight` now returns 4 rows per block for a Q4_K verify batch (`ncols_dst` 2..8) on
the RDNA3_0 table, and the `mmvq.cu` multi-row guard covers `MMVQ_PARAMETERS_RDNA3_0` using the device's
own table (RDNA4 takes the identical `rpb_w` as before).  Each row keeps its own thread mapping, K order
and warp reduction, so the result is bit-identical; only the q8_1 activation columns are loaded once per
block instead of once per row.  `GGML_MMVQ_RDNA3_Q4K_WEIGHT_RPB` (compile-time, default 4) sets the
count.  Single-token decode is unchanged.

## Validation (r65 candidate)

- **gfx1201 (this session):** the RDNA4 path is unchanged.  The rule-5
  `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (dense Qwen3.8-27B UD-Q4_K_XL, q8_0 KV, 3 GPUs) is no
  worse than the r64 reference: B=1 `28.51 -> 29.12`, B=4 `102.85 -> 108.81`, B=8 `180.29 -> 188.96`
  t/s.  `test-backend-ops -o MUL_MAT -p type_a=q4_K` and `-o MUL_MAT_ID -p type_a=q4_K` are OK.
- **gfx1100 (reporter, RX 7900 XTX, WSL2):** MTP `n_max 3`, Qwen3.8-27B Q4_K_M: prose `59.2 -> 63.7`
  (+7.6 %), code `69.6 -> 75.1` (+7.8 %), 26k depth `47.9 -> 51.0` (+6.4 %); plain decode unchanged;
  output bit-identical with and without MTP.

**Open caveat carried forward:** one model/quant, WSL2 only; gfx1101/gfx1102 take the rule by family.

## Record

`WORKLOG.md` 261015.002 (r65).  Original PR description and patch: `README.md`,
`0001-mmvq-gfx1100-q4k-rpb.patch`.
