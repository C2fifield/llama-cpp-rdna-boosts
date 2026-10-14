# `mmvq-rpb-retune` — retune the RDNA4 mmvq rows-per-block table for `q5_1` / `iq4_xs` on ROCm 7.14

**Status: OPEN (seeded 2026-10-11), small follow-up.**  Not part of the delivery; nothing here may be
applied to the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).
This is the **only** residual of the now-archived `mmvq-verify-rows` campaign
([`../../archive/work/mmvq-verify-rows/`](../../archive/work/mmvq-verify-rows/README.md)), which was
fully integrated in `v16-84e76d8a2-r21` (PR #57, blocks 10 + 13).  It is the same item as **TODO #28**.

## The residual

The multi-row `calc_rows_per_block_weight()` table in `ggml/src/ggml-cuda/mmvq.cu` was swept by the
contributor on **ROCm 10.0**.  It transfers cleanly to the maintainer's **ROCm 7.14** build for the main
types (q3_K +7..+24 %, q4_K +8..+38 %, q6_K +3..+27 %, q2_K +12..+26 %, plus +14 % at n=1 from
`__mul24`), but carries two small dips at the narrow verify widths:

| type | n=2 | n=3 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| `q5_1` | -3.3 % | -4.2 % | -2.4 % | +12.2 % |
| `iq4_xs` | -2.6 % | -4.4 % | +1.3 % | +10.9 % |

The common MTP/DFlash verify widths are 4 and 8, so the aggregate band is a large net win and the table
was kept as tuned.  The candidate fix is a **per-type n-dependent** rule: 1 row for these two types at
2..3 columns (their +10..+12 % at n=8 must be preserved).  The present rule is deliberately per-type
only ("a per-(type, n) rule would add at most 2.5 % (IQ4_XS)", `archive/work/mmvq-verify-rows/README.md`
§1), so any change here must beat that tradeoff.

## What to do

1. Sweep `calc_rows_per_block_weight` for `q5_1` / `iq4_xs` at **every** verify width (n = 2..8) on
   gfx1201 / ROCm 7.14, plus a `q5_1`/`iq4_xs` MUL_MAT microbench (`test-backend-ops perf -o MUL_MAT`,
   m = 4096, k = 14336) as in `VERIFICATION-r21.md`.
2. If a wider rule (1 row only at n = 2..3, current 2 rows at n = 4..8) wins, implement it so the
   host dispatch and the kernel agree (the `has_fusion`/`small_m` host/kernel mismatch class,
   `GREEDY-PURITY.md` §20).
3. **Purity is the hard gate:** per-row arithmetic, thread mapping, K order and warp reduction must not
   change, so `W = 1..8` stays one hash.  A per-`(type, n)` rows choice is safe for the same reason the
   per-type one is (rows-per-block is a block decomposition, not a reduction-order knob) — but prove it
   with the width probe, do not assume it.
4. Re-measure the verify-width band and MTP acceptance, and re-check the scratch-spill list (a new
   `(type, n)` instance must not spill; the Q2_K 4-row kernel spilled at 7..8 columns and is capped at 2
   there for exactly this reason).

## Gates

- `test-logits-width-probe` on the affected models at `W = 1..8` — one hash per tensor across widths
  (stock == patched).
- `test-backend-ops perf -o MUL_MAT` (m = 4096, k = 14336): no type below stock at any width; the two
  regressions removed.
- `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (rule 5, `benchmarks/mtp-adaptive-methodology.md`):
  decode must not regress.
- MTP acceptance identical (`-n 2000`, q8_0 KV) and the scratch-spill list identical to stock.
- `scripts/validate-set.sh` if anything is promoted.

## Pointers

- Integrated campaign (now archive): [`../../archive/work/mmvq-verify-rows/`](../../archive/work/mmvq-verify-rows/README.md),
  in particular its `VERIFICATION-r21.md` (the ROCm-7.14 re-check that found the two dips).
- `WORKLOG.md` 2026-09-28 (r21, PR #57); `patches/README.md` block-10 + block-13 amendments; TODO #28.
- `GREEDY-PURITY.md` §19/§24/§25 (the band-uniform knobs that must stay pinned).
