# RESOLUTION -- `spec-batch-budget` (PR #169, @briansp2020) -- REJECTED

**Status: REJECTED (2026-10-16) after review on gfx1201.  Not shipped; no block change.**  The
contributor's format-patch and write-up are preserved in this directory so the measurement is not
re-litigated from scratch.  The PR is answered on GitHub with the numbers and with what a re-entry
would need.

## What it proposed

`LLAMA_SPEC_BATCH_BUDGET=N` (server only, `tools/server/server-context.cpp`): while more than one slot
is generating, cap each slot's draft at `max(1, N / n_generating - 1)`, so the verify batch stays within
`N` tokens.  Unset = off, one draft token per slot minimum, single request never capped.  The reporter
measured +8.7 % aggregate at 4 concurrent requests on 2x R9700 (IQ3_XXS Flash-Next, 300-token requests)
and no change to single-request speed or output.

## Why it was rejected

It regresses on this box, in the reporter's own 2-GPU topology as well as at 3 GPUs.  Same binary
(r76 + PR #171 + PR #168 + PR #169, so the accepted PRs are held constant), fresh server per arm, warmup
first, q8_0 KV, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`, four concurrent
300-token requests:

| config | budget unset (cap 3) | `=8` (cap 1) |
|---|---|---|
| MoE 35B-A3B UD-Q4_K_M, 3 GPUs tensor | 279.5 / 259.1 / 278.0 t/s | 232.8 / 211.9 / 213.0 t/s (**-16 %**) |
| MoE 35B-A3B UD-Q4_K_M, **2 GPUs** tensor | 314.6 / 320.7 / 326.9 | 304.7 / 308.7 / 323.4 (**-2.7 %**) |
| dense 27B UD-Q4_K_XL, 3 GPUs tensor | 134.3 / 142.4 / 151.9 | 128.9 / 126.2 / 122.6 (**-12 %**) |
| single request (same arms) | 130.7 / 104.9 / 130.2 | 129.7 / 102.3 / 130.1 |

Budget sweeps show the shape of it: 8 (cap 1) is the losing arm, 12 (cap 2) is neutral, and 16/32 (cap 3,
i.e. no cap) reproduce the unset column.  On this hardware a 16-token verify batch (4 slots x 4) is
**cheaper per accepted token** than an 8-token one, so the premise ("at that width the extra draft
tokens cost more step time than they save") does not hold here.

## Second finding (structural, independent of the numbers)

The cap is applied to `common_speculative_get_draft_params(...).n_max`, i.e. to the *per-call* bound,
but the adaptive controller (`--spec-type draft-mtp-adaptive`) is fed `n_accepted` against its own
`n_cur` (`common_speculative_adaptive::update`), not against the load-capped depth.  During a multi-slot
burst the controller therefore sees a miss of `n_cur - n_accepted` every round and drains to the floor;
when the burst ends, a single-request run has to climb back from the floor (about 20-26 net full accepts
per step).  The PR's measurements are all fixed-depth (`draft-mtp`), so this interaction is untested.

## Re-entry conditions

1. Reproduce a win on a config this project can gate: the 2-GPU row above is the reporter's topology and
   it is -2.7 % here; a model/quant/GPU combination where the cap actually pays would need to be named,
   with the before/after per-step step time and accepted tokens (not only aggregate t/s).
2. Make the cap interact correctly with `draft-mtp-adaptive`: feed the controller the effective
   (capped) depth, or suspend its update while the load cap is active, so a burst does not drain it.
3. Keep it an opt-in tuning value (`0`/unset = off) -- a hard-coded 8 is device- and model-dependent.
4. Gate it like any other change: `llama-batched-bench -npl 1,4,8` at the verify widths plus acceptance
   from multi-slot servers, since "4 concurrent chat requests" is the only shape that can move.

## Record

`WORKLOG.md` 261016.012 (r77) records the rejection and the table.  No `ENVIRONMENT.md` entry, because
nothing shipped.
