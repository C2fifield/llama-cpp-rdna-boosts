# RESOLUTION — `rdna3-mmvq-vecdot-alu` (PR #159, @briansp2020)

**Status: RESOLVED (2026-10-15), promoted into the delivery as part of `v12-097cf33d9-r65` (block 07
amended).**  No `wip/` entry; the PR's format-patches and write-up are preserved in this directory.

## What landed

Two instruction-count reductions in the mmvq dot products, both output-preserving:

- **0001 (shared code, `vec_dot_q4_K_q8_1_impl_vmmq`, VDR 2):** the mins term becomes
  `(d8 * float(dot2)) * float(m)` instead of `d8 * float(dot2 * m)`.  `d8` (a half, 11 significant bits)
  times `|dot2| <= 1024` (11 bits) is exact in fp32 and `float(dot2 * m)` is also exact, so both forms
  perform a single rounding of the same exact value.  The two factors are separately column- and
  row-dependent, so a multi-row, multi-column block computes each once.  The 6-bit scales/mins are packed
  into one word built 32 bits at a time, replacing the two 16-bit `aux` halves.
- **0002 (HIP only, `vec_dot_q6_K_q8_1_impl_mmvq_vdr2`):** `(vil | 16*h) - 32` becomes
  `vil | __builtin_amdgcn_perm(0, 0x1000F0E0, h)`, one `v_perm_b32` plus an OR, replacing the
  shift/mask and the emulated byte subtract.  CUDA keeps `__vsubss4`.  Reached on RDNA4 / RDNA3_0 only.

Reported instruction counts: gfx1100 Q4_K 4-col 566 -> 475, Q6_K 4-col 213 -> 160; gfx1201 Q4_K 4-col
567 -> 485, Q6_K 4-col 654 -> 478.

## Validation (r65 candidate, gfx1201)

- Dense Qwen3.8-27B UD-Q4_K_XL, single R9700, draft-mtp `n_max 3`, `-n 3000`, q8_0 KV: r64 reference
  `69.4 / 69.5` t/s, candidate `70.0 / 70.0` t/s (**+0.8 %**), acceptance `0.78865` identical.
- Rule-5 `llama-batched-bench` (27B UD-Q4_K_XL, q8_0 KV, 3 GPUs): B=1/4/8 improved, no regression.
- Flash-Next GSQ-RCO IQ3_XXS MTP, 2 GPUs: within noise (133.3 vs 133.1 t/s); acceptance and text
  identical.
- `test-backend-ops -o MUL_MAT` / `-o MUL_MAT_ID` with `type_a=q4_K` and `type_a=q6_K`: OK.
- Reporter: +2-3 % gfx1201 (R9700), +4-6 % gfx1100/Q4_K_M; output identical on both.

**Open caveat carried forward:** 0001 is shared code; CUDA was not built here or by the reporter, so its
bit-identity on nvcc rests on the single-rounding argument above.  RDNA3_5 was not run (0001 applies,
0002 does not).

## Record

`WORKLOG.md` 261015.002 (r65).  Original PR description and patches: `README.md`,
`0001-vecdotq-q4k-alu.patch`, `0002-vecdotq-q6k-alu.patch`.
