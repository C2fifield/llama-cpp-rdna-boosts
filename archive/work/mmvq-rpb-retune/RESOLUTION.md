# `mmvq-rpb-retune` — RESOLVED (r55, 2026-10-13)

**Partially promoted into the delivery as block 08; the `iq4_xs` half closed as a no-win.**

## Result

Swept `calc_rows_per_block_weight` for `q5_1` / `iq4_xs` at n=2..8 on gfx1201 / ROCm 7.14
(`test-backend-ops perf`, m=4096 k=14336, us/run; lower is better):

| type | rule | n=2 | n=3 | n=4 | n=5..8 |
|---|---|---|---|---|---|
| `q5_1` | **1 row at n<=4, 2 rows at n>=5** | **-6.9 %** | **-6.6 %** | **-7.1 %** | unchanged |
| `iq4_xs` | unchanged (4 rows) | 1 row +10.7 %, 2 rows +11.4 % | 1 row +9.2 %, 2 rows worse | — | — |

`q5_1` now returns `ncols_dst <= 4 ? 1 : 2` in the RDNA4 table (**block 08**).  The n>=5 two-row rule is
preserved (up to +16 % at n=8).  `iq4_xs` keeps 4 rows: the campaign's "dip at n=2/3" did **not**
reproduce -- 1 row and 2 rows are both slower than the current 4 rows.

## Limitation

No model in `/llm/models` contains `q5_1` tensors, so the standard MTP gate cannot exercise the change;
the evidence is the microbenchmark.  The change is width-safe by construction (rows-per-block is a
weight-row block decomposition, not a K-reduction knob) and cannot affect any other quant type.

## Validation

`validate-set.sh` strict 12/12; 4B greedy `1c5d32ac537d` unchanged.
