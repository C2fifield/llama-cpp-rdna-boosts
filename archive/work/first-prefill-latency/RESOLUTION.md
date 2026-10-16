# `first-prefill-latency` — PROMOTED (r74, 2026-10-16): arena sized at warmup, seed capped at the reclaim boundary

## Result

The campaign's arena-ordering fix shipped in **`v12-097cf33d9-r74`** (blocks 09/10/11; canonical tip
`9450cbb89`, net tree `0fe32003b4014c2adf520eccc2e2ba1647ae3511`).  Default on
(`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` restores the deferred sizing).

What was promoted:

* the priming pass also calls `alloc_all_locked` (completeness-gated), so the arena is sized at load;
* `slab_extend_all` is skipped at the priming sizing (it over-mapped the reserve and a later graph
  capture OOM'd);
* the arena is sized to the **full** region, but the context records the wide prefill size
  (`ggml_cuda_slab_reserve_wide`) and the **seed stops at `narrow_floor + ring + wide`** — the band a
  wide prefill immediately reclaims.  That band's tables are the coldest (heat-ordered last), stay
  unseeded, are reclaimed cheaply by the first prefill, and fill lazily during decode.  The seed boundary
  is the shared `ggml_cuda_slab_ring_layout` helper's `wide_floor` output (also used by the real wide
  placement, so the two cannot disagree).

Measured (maintainer's exact GSQ command, 1 GPU, 13.8K prompt): arena **15666.6 MiB** at load (the full
15.62 GiB, matching the pre-session sizing), **97/144** tables seeded, first-prefill eviction **6150 MiB**
(the band), `moe_cache_rearm` restores the arena to 15666.6 MiB, decode **75.0 t/s**.  Log:
`seed-floor-full-arena-1gpu.log`.

**Rejected draft.**  The first r74 attempt made the reservation a *persistent boundary floor* instead:
that kept the ring+wide out of the arena permanently, so the sizing budget shrank to 9.75 GiB and the
arena could never grow back into it (measured 9671 MiB vs the correct 15666.6).  It was reverted before
release — the boundary is **not** raised; only the seed is capped.

## Not promoted

`keep-survivors.patch` (skip the free+re-seed of a cut layer's still-valid survivors on a re-arm) is
**inert in the measured config** — the re-arm never fired on a straddling layer.  It stays here as a
prototype for a workload that does stand tables down.

## Residuals / follow-ups

* The 3-GPU decode deficit this work surfaced is a separate, revived campaign:
  [`wip/three-gpu-performance/`](../../wip/three-gpu-performance/README.md) (TODO #51).
* The 2/3-GPU arms of the corrected design must be re-measured (the earlier 84.4/100 % numbers were the
  rejected reservation draft).
* Async seeding (overlap the H2D once the arena exists) remains an alternative to the load-time seed.

See `README.md` in this directory for the full measurement, diagnosis and repro.
