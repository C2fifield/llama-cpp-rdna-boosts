# `first-prefill-latency` — PROMOTED (r74, 2026-10-16): the arena is sized at warmup with ring+wide reserved

## Result

The campaign's arena-ordering fix shipped in **`v12-097cf33d9-r74`** (blocks 09/10/11; canonical tip
`d6f59a05d`, net tree `599befc66eba9ebba1ed38956e3fedd2fc32bd37`).  Default on
(`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` restores the deferred sizing).

What was promoted:

* the priming pass also calls `alloc_all_locked` (completeness-gated), so the arena is sized at load;
* `slab_extend_all` is skipped at the priming sizing (it over-mapped the reserve and a later graph
  capture OOM'd);
* the context reserves the **ring + wide prefill** region before the cache sizes
  (`ggml_cuda_slab_reserve_wide`), with the layout from the shared `ggml_cuda_slab_ring_layout` helper
  (also used by the real wide placement, so the two cannot disagree).

Measured (GSQ-IQ3_XXS `-ncmoe 48 -sm tensor`, 13.8K prompt): arena 5355 MiB at load, first-prefill
eviction **5810 -> 84 MiB**, decode **29.2 -> 37.1 t/s** (1 GPU), **84.4 t/s** (2 GPU), **100 % resident**
(3 GPU), no OOM.

## Not promoted

`keep-survivors.patch` (skip the free+re-seed of a cut layer's still-valid survivors on a re-arm) is
**inert in the measured config** — the re-arm never fired (the only cache event was an 84 MiB,
non-straddling eviction).  It stays here as a prototype for a workload that does stand tables down.

## Residuals / follow-ups

* The 3-GPU decode deficit this work surfaced is a separate, revived campaign:
  [`wip/three-gpu-performance/`](../../wip/three-gpu-performance/README.md) (TODO #51).
* Async seeding (overlap the H2D once the arena exists) remains an alternative to the load-time seed.

See `README.md` in this directory for the full measurement, diagnosis and repro.
