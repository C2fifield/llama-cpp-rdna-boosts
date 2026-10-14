# RESOLVED — promoted in `v12-097cf33d9-r60` (2026-10-14)

**The campaign is PROMOTED: the persistent cross-run expert heat map is folded into block 09.**  The
delivery is `v12-097cf33d9-r60` (base `097cf33d9`, canonical tip
`a3459aada97288b520504f5716c37d628f41a58d`, net tree `ec37675ec797392508e560616d320d8cc09efd86`,
`validate-set.sh` green).  Block 09's patch now carries the change; `release.json`, `ENVIRONMENT.md`,
`patches/README.md` and `WORKLOG.md` (261014.005) are updated.

## Follow-up: r61 freeze semantics (2026-10-14)

r61 (block 09) made the heat map's **file mode the freeze signal**, because r60's save path never checked
the target mode: a `tmp`+`rename` replaces a read-only file whenever the directory is writable, so the
announced `chmod 400` trick did not freeze anything (measured), and a read-only model directory relocated
to the cache instead of freezing.  Now a read-only sidecar is loaded and used to seed, is never
overwritten, and is not relocated; `MOE_EXPERT_HEATMAP=load` (aliases `ro`, `read-only`) is the explicit
freeze; `=0` still disables the feature entirely.  See `WORKLOG.md` 261014.006 (r61) and
`ENVIRONMENT.md` section 1.2.  The r60 measurements below are unchanged (the fix is on the persistence
path only).

## What was promoted

Everything in `heatmap-persistence.patch` (R2 + persistence), default-on:

* **R2 device-side routing tally** — `moe_cache_heat_observe` launched from the outer `mul_mat_id`
  entry (covers identity, eager and staged-prefill paths) + a host tally on the eager decode band.
  This closes the F7 gap: a fully-resident run previously recorded no per-expert heat at all
  (`takeover=1`, `hits=16`), now it produces a full map from the device prefill tally.
* **Persistent sidecar** `<gguf>.moe-heatmap` — `MHT1` fixed records keyed by
  `(layer, sem_role, n_experts, host_bytes)`, atomic `tmp`+`fsync`+`rename`, 60 s debounce with a
  mandatory exit flush, read-only-model-dir fallback (`MOE_EXPERT_HEATMAP_DIR` → XDG cache → HOME →
  `/tmp`), non-fatal failures that disable persistence for the session after 3 consecutive errors,
  merge-on-save (a long EWMA across runs).
* **Seed at sizing** — `apply_heat_seed_locked` bulk-admits the top-`slots` experts as provisional
  slots, before the (still inert) prefill seed.
* **Heat-ordered layer placement (campaign `moe-arena-defrag` phase 1)** — `alloc_all_locked`
  allocates layers in descending heat so the hottest sit highest in VA and are reclaimed last; frozen
  at sizing.

Env: `MOE_EXPERT_HEATMAP` (default **1**, kill switch), `MOE_EXPERT_HEATMAP_DIR`,
`MOE_EXPERT_HEATMAP_DEBOUNCE_S`, `MOE_EXPERT_HEATMAP_MAX_FAILS` — documented in `ENVIRONMENT.md` §1.2.

## Gates (all passed)

* Same-seed coherence: heat-on vs heat-off output **byte-identical** (only the spinner/timing lines
  differ).
* `scripts/gate-prefill-logits.sh`: **PASS**, mean KLD **0.000707**, same-top-p **98.755 %** (exactly
  the recorded base — no prefill-kernel drift).
* Adaptive-MTP purity on the field config (`-sm tensor -ncmoe 99`, auto cache): `none == n3` text
  identical.
* `test-backend-ops -o MUL_MAT_ID`: the same **4 pre-existing** upstream `f16 dup=1` fails as the r53
  baseline (documented), no regression.
* `scripts/validate-set.sh`: green on a fresh tarball (checksums + strict `git am` + tree match).

## Measured win

35B-A3B UD-Q4_K_M, `-sm layer -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`, 5256-token prompt, same seed:

| turn | run | hit rate | fills | decode t/s |
|---|---|---:|---:|---:|
| 48 tok | cold (no map) | 0.7674 | 11748 | 38.15 |
| 48 tok | warm (seeded) | 0.9402 | 2976 | **51.30 (+34 %)** |
| 256 tok | cold (no map) | 0.9060 | 19452 | 48.53 |
| 256 tok | warm (seeded) | 0.9561 | 6912 | **53.55 (+10.3 %)** |

The arena no longer starts cold.  Prefill unchanged; the seed cost lands in warmup/sizing.

## NOT promoted — residuals (now in `TODO.md`)

1. **`MOE_EXPERT_CACHE_PREFILL_SEED` repair (F1/F6).**  The in-run `prefill_count` tally is still dead
   (gated on `!g_sized`, staged path bypasses the hook), so the host-pool prewarm order
   (`pool_prepopulate_locked`) still falls back to expert-id order and `pool_rerank_from_tally_locked`
   is still an uncalled function.  The heat vector now covers the same need for the arena, but the
   *pool* warm order is a separate, cheap fix.
2. **Decode-identity heat gap.**  A fully-resident table's decode traffic is still not tallied (the
   identity take-over skips `access_locked`); the prefill tally covers the map, which the measurement
   showed is a good proxy, so this is low priority.
3. **`moe_cache_observe` is still dead code** (superseded by `moe_cache_heat_observe`); remove it or
   wire it to the tally.

Campaign `moe-arena-defrag` phases 2-4 (eviction-order policy, idle repack, partial eviction) remain
parked in `wip/moe-arena-defrag/`; phase 1 is what shipped here.  Campaign `slab-transient-absorption`
is untouched.

## Record

* Patches: `patches/0009-...` (block 09); this directory's `heatmap-persistence.patch` is the
  pre-promotion prototype (R2 + persistence) and `r2-device-tally.patch` the earlier R2-only snapshot.
* `WORKLOG.md` 261014.005 (r60); `patches/README.md` block 09; `ENVIRONMENT.md` §1.2.
* The two design Q&A sections (§0.1 decisions, §0.2 "does persistence depend on the prefill tally?")
  are the useful part of this README for future readers; the implementation detail now lives in the
  delivered block.
