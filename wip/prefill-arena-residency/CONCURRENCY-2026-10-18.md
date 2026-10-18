# Concurrency stress — 2026-10-18

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
