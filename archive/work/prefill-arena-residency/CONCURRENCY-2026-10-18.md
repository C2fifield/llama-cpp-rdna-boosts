# Concurrency stress — 2026-10-18

> The freeze finding below stands on its measurements, but its *interpretation* was superseded after r83:
> the reason a phase freeze loses is pin **scope** (the contended arena, for a whole prefill), and the
> shape that keeps the coherency guarantee without that stall is the copy's copy-time pin.  See
> [`README.md`](README.md) §9.

Two-slot server (`-np 2`), R2 prototype (`GGML_MOE_CACHE_STAGE=0/1`), field loader.
Harness: [`tools/concurrency_check.py`](tools/concurrency_check.py).  Raw JSON:
`/tmp/conc_full_{a,b}*.json`; server logs `/tmp/conc_{a,b}_stage{0,1}.log`.

## Setup

`Qwen3.8-Flash-Next IQ4_NL` (100 GB, 176.9B A3B), 2× R9700 gfx1201, ROCm 10.1.0:

```sh
HIP_VISIBLE_DEVICES=0,1 GGML_MOE_CACHE_STAGE={0,1} llama-server \
  -m .../Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  -ngl 99 -sm tensor -ncmoe 48 -fa 1 -ctk f16 -ctv f16 \
  -c 131072 --parallel 2 -b 4096 -ub 4096 -t 8 \
  --load-mode none --lazy-mode off --fit off --spec-type none
```

`-c 131072` / 2 slots ⇒ 64k per slot; arena **43 410.9 MiB** at sizing (66.9 % of the 64 800 MiB
host expert set, lower than the single-request 47 207.8 MiB / 72.9 % because the KV is larger).

## Scenario A — long prefill (slot 0) + short prefill/decode (slot 1)

Slot 0: 15 748-token prompt, `n_predict 8`.  Slot 1 starts 0.3 s later: 5 246-token prompt,
`n_predict 32` (only one short request fits inside the long prefill).

| | stage 0 (whole-table) | stage 1 (R2) | R2 gain |
|---|---:|---:|---:|
| long prefill | 754.9 t/s | **889.4 t/s** | **+17.8 %** |
| short request prompt | 293.3 t/s | **339.2 t/s** | **+15.7 %** |
| short decode (32 tok) | 26.7 t/s | 26.8 t/s | ~0 |
| long wall | 24.78 s | 20.94 s | −15.5 % |

Cache: stage 1 staged **H2D 33.2 %** (arena 177 134.6 MiB vs host 87 915.4 MiB), `geom_decline=0`,
`unusable=230`, **PRS rebuilds = 1 180 = every staged call**.  The slab evicted/re-armed 54 tables on
**4** prefill→decode transitions (which is the 4 ubatches of the long prefill).

## Scenario B — long decode (slot 0) + stream of short prefills (slot 1)

Fixed **60 s window**; the decode request asks for 10 000 tokens with `ignore_eos` and is abandoned at
the window end (the point is the rate during contention).  Short prefills: 3 925-token prompt,
`n_predict 4`, `cache_prompt:false`, streamed back-to-back.

| | stage 0 (whole-table) | stage 1 (R2) |
|---|---:|---:|
| long decode alone (60 s) | **49.02 t/s** (2 942 tok) | **49.28 t/s** (2 957 tok) |
| long decode + short prefills (60 s) | **3.76 t/s** (230 tok) | **4.17 t/s** (255 tok) |
| short prefill median | **1 162.8 t/s** | **1 479.1 t/s** (+27.2 %) |
| short prefill min/max | 1 057.9 / 1 226.8 | 1 333.6 / 1 588.5 |
| short 4-token decode | 14.7 t/s | 14.7 t/s |

Cache (stage 1): **H2D 33.4 %**, `geom_decline=0`, `unusable=966`, **PRS rebuilds = 4 956 = every
staged call**, arena 43 410.9 MiB, hit `h=0.9779`.

## Findings

* **F1 — R2 survives concurrency and still helps.**  +16-27 % on the prefill side in both scenarios,
  with the same 27 %→33 % H2D split (the arena is smaller at 64k/slot).
* **F2 — the long-decode starvation is the scheduler, not R2.**  A stream of short prefills drops a
  long decode from ~49 t/s to **3.8-4.2 t/s** with R2 **off and on** (3.76 vs 4.17).  It is the
  two-slot continuous-batching policy: the decode gets roughly one token per prefill request.  The
  field config uses `--parallel 1`, so this is a `-np > 1` behaviour; it is worth tracking separately
  (it is not caused by, and cannot be fixed by, the expert cache).
* **F3 — the current prototype churns on every staged call.**  With a decode session active, each
  decode admission bumps `g_write_seq`, so every `moe_cache_stage_table` rebuilds its PRS and the copy
  stream waits on the compute stream (`PRS rebuilds == staged calls`: 1 180 in A, 4 956 in B).  This
  is exactly what the 75/25 freeze is for: while the top 75 % is frozen, decode writes only touch the
  bottom 25 %, so the snapshot stays valid and the rebuild count should collapse to one per phase.
* **F4 — the slab boundary is the other churn source, and the freeze cannot fix it.**  Every
  4k-token short prefill grows the compute buffer, the slab takes chunks (`evicted 3950.0 + 4014.0
  MiB` per device), and the prefill→decode drop re-arms **54 tables** (`arena now 43 410.9 MiB`).
  In scenario B this is a ~2.7 s cycle for every short request.  That is the movable-boundary slab
  (`../slab-transient-absorption/`, `../moe-cache-coverage/`), not the residency policy; a
  residency-aware cache can only decide *which* tables are lost, not stop the work region taking the
  chunks.
* **F5 — what the freeze must key on.**  The PRS invalidation must move from "any cache write" to "the
  **frozen set** changed".  A decode admission that evicts only the mutable 25 % does not change the
  frozen `expert -> slot` map, so the PRS stays valid and no copy-stream wait is needed.  A slab
  stand-down/re-arm (F4) still invalidates it and must fall back / rebuild.

## Next actions

1. Implement the 75/25 freeze: per table, mark the top ~75 % of slots by heat/count as frozen while a
   prefill is in flight; decode admission evicts only the mutable 25 %; the PRS marks only frozen
   slots; invalidate on frozen-set change or arena move, not on every `g_write_seq`.
2. Re-run scenarios A and B (stage 0 vs 1) and compare PRS rebuilds, H2D split, prefill rate, decode
   rate.  The expected effect is F3 only; F2/F4 should not move.
3. Then decide on R1 (in-place MMQ read) for the remaining ring copy.
   *(Settled after r83: **not** R1 — the ring and its copy stay, and the coexistence primitive is the
   copy-time pin (one table's fill, then release), not a phase freeze.  See `README.md` §9.)*

---

## Follow-up: the 75/25 freeze, implemented and measured (2026-10-18)

**Activity signal.**  The per-batch token classification was too coarse (the server alternates
prefill-only and decode-only batches, and the freeze toggled 16×/60 s with no benefit).  The server now
passes its exact **slot states**: `server_context::decode` counts slots in
`PROCESSING_PROMPT`/`STARTED` as prefill and `GENERATING` as decode, and calls
`llama_moe_cache_set_activity(model, n_prefill, n_decode)` → device hook → `moe_cache_set_activity`.
A 4-decode-batch hold covers the decode-only batches that can separate two prefill chunks of the same
request.  Arm only while `n_prefill > 0 && n_decode > 0`; a pure prefill or a pure decode uses the whole
arena.  Kill switch: `GGML_MOE_CACHE_FREEZE=0`.

**Frozen set.**  Per table, the top ~75 % of the **occupied** slots by `(count, heat, last)` are
immutable; decode admission may only evict the other ~25 %, and at least `n_used_hint` (8) slots stay
mutable.  The PRS marks only frozen slots and is keyed on the freeze generation, so a decode fill into
the mutable region does **not** rebuild it or order the copy stream.

| scenario | metric | stage 0 | R2, no freeze | R2 + freeze |
|---|---|---:|---:|---:|
| A (long prefill + short prefill/decode) | long prefill | 754.9 | 889.4 | **830.4** |
| A | short prompt | 293.3 | 339.2 | **309.6** |
| A | short decode | 26.7 | 26.8 | 26.3 |
| A | H2D staged | 100 % | 33.2 % | 36.5 % |
| A | freeze arms | — | — | **1** |
| B (long decode + short prefill stream) | short prefill | 1 162.8 | 1 479.1 | **1 150.1** |
| B | long decode (concurrent) | 3.76 | 4.17 | 3.73 |
| B | H2D staged | 100 % | 33.4 % | 49.1 % |
| B | freeze arms | — | — | **16** |
| A/B | decode hit rate | — | A 0.8466 / B 0.9795 | A 0.8453 / B 0.9796 |

**Interpretation.**

* The phase logic is exactly as specified.  Scenario A (two prefills, one switches to decode) arms
  **once** and unfreezes at the end.  Scenario B arms **once per short request** (its prefill freezes,
  its own 4-token decode unfreezes) — the literal "freeze only while a prefill and a decode are active
  together" rule.
* **The freeze costs prefill throughput.**  The frozen set is 75 % of the *arena*, and the arena is
  ~67 % of the expert set at 64k/slot, so the prefill's effective residency is ~50 % and H2D rises from
  33 % to 49 % (A: 36.5 %, because only part of the run is armed).  Scenario A loses 6.6 % prefill;
  scenario B loses 22 %.
* **No measurable decode-side benefit:** the decode hit rate is unchanged, and the long decode is
  scheduler-starved (3.7-4.2 t/s) with or without the freeze.  The freeze is therefore a **correct,
  phase-accurate but net-loss** policy on these workloads.  Keep it opt-in inside the prototype
  (`GGML_MOE_CACHE_FREEZE=0` disables it); do **not** default it on until a workload where decode
  latency/hit rate is the objective shows a benefit, or the arena is fully resident (then the frozen
  75 % is 75 % of the experts, not 50 %).
* **Correctness does not depend on the freeze.**  The PRS is rebuilt when the cache writes or the arena
  moves; the freeze only changes *which* residents the snapshot exposes.  So the non-freeze R2 path is
  the one to ship for throughput; the freeze is a policy option for a decode-latency-sensitive mixed
  workload.

**Decision (maintainer, 2026-10-18): the 75/25 freeze is DROPPED.**  It is correct and phase-accurate,
but a net prefill loss with no measurable decode gain on the tested workloads, and the per-layer-pin
concept the maintainer proposed collapses to the same thing in this scheduler (see below).  The prototype
patch is R2-only again; the freeze implementation remains recoverable at campaign commit `8134e23`.

**Why the per-layer pin is not a third option in this scheduler.**  Prefill and decode are separate,
serialized graphs: one `llama_decode` graph at a time, and a ubatch runs **all layers**.  Inside a graph
the arena is already stable, so a pin that lasts for the layer/ring slot being prefilled protects a
window in which nobody can evict anyway.  The erosion happens *between* graphs, when a decode ubatch runs
`moe_cache_update_host`.  A pin that survives that must cover every table the next prefill ubatch stages
— i.e. all of them — which is a 100 % freeze; a fraction of it is the 75/25 policy just measured.  The
ring read-ahead is inside the same graph, so it does not change this.  **The true limitation is the
serialized scheduler**; a per-layer/per-ring-slot pin becomes the right primitive only once a decode graph
can overlap a prefill graph (the `moe-cpu-overlap` / pipelined-scheduler work).

**Beta-12 compatibility:** deferred to a follow-up session at the maintainer's request.  The beta patch's
only overlap with this prototype is `ggml-cuda-moe-cache-hooks.cu`, where beta 12 changes
`stage_h2d_gbps` (the calibration probe) and this prototype changes `stage_from_cache`; the hunks are
disjoint, as are `moe-expert-cache.*` (not touched by beta 12) and the new activity plumbing.  It must
still be verified by applying both orders and building/running the 16k A/B.
