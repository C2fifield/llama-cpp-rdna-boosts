# `beta/` — promotion-candidate patches (NOT part of the delivery)

This directory holds **self-contained patches that are candidates for folding into the delivery block
set**, staged for a validation window before the maintainer decides.  Nothing here is part of
`patches/`, `release.json` or `rdna-boosts-all.patch`, and a fresh `scripts/apply-all.sh` checkout
never contains it.

**Why it exists.**  A candidate that changes a file a delivery block owns cannot be shipped as a
separate `wip/` item (it would conflict with the release flow) and cannot be committed straight into
the blocks (it has not been validated on the reporter's platform yet).  `beta/` is the holding area
for exactly that: the patch is cut against the current delivery tree, handed to the reporter for a
beta test, and only folded into the owning block once the result comes back.  This is the same area
that carried `beta/mmb-general` and `beta/tensor-fit-fix` (both promoted; see `archive/work/`).

## Conventions

* One sub-directory per candidate: `beta/<name>/`.  A numeric prefix orders it relative to the block
  set (`0012-...` applies after block 11).
* `README.md` — what the candidate is, why, the patch, and the status (dated entries, newest first).
* `BETA-TESTING.md` — the apply + build + gate checklist handed to the tester.
* `<name>.patch` — the patch itself, cut against the current `release.json` tree.  It must `git apply`
  and `git am` cleanly on a fresh `apply-all.sh` checkout; record its **sha256** and the applied tree
  in `README.md`.
* **Nothing here is applied by default.**  `scripts/apply-all.sh` ignores `beta/` unless the tester asks
  for it explicitly with `--include-beta`, which applies every `beta/*/*.patch` (sorted) on top of the
  verified release tree, one commit each, and then reports the resulting tree (which is *not*
  `release.json`'s tree -- that is the point).  Without the flag the script behaves exactly as before.

## Promotion

When the tester confirms the candidate, it is folded into the block that owns the code it changes (or
another block, at the maintainer's discretion), the block set is regenerated and re-validated, and the
candidate moves to `archive/work/<name>/` and is recorded in `WORKLOG.md`.  If it is refuted, the same
move happens with a "refuted" status.  A finished candidate leaves `beta/`; no stubs.

## Current candidates

* **`0013-slab-narrow-floor`** — **candidate, 2026-10-18 (pending the r84 base).**  Fixes a degenerate
  slab eviction loop on host-expert MoE + MTP decodes: the always-resident narrow band was estimated from
  a `split_only` `n_rs_batch` reserve (1792 MiB) and frozen once the ring held blocks, so the wider MTP
  decode/verify graph (2304 MiB) was classified *wide*, placed above the H2D staging ring, and evicted
  3.65 GiB/device of arena — and the prefill→decode drop then recorded the smaller reserve as its
  reference, so it repeated **every decode step** (66 tables re-armed, 18 cycles ≈ 3 s at 17–25 t/s vs
  45).  Reproduced identically on **r82** (pre-existing, not an r83 regression); scales with the draft
  depth (n-max 6 → 18 cycles, n-max 2 → 2, no MTP → 0).  The candidate learns the band from the real
  graph (`work_alloc`), reports a refused raise instead of dropping it silently, and takes the drop
  reference as `max(observed, re-reserved)`.  Measured: **1** re-arm, decode **45.4 t/s** (`-sm tensor`)
  / **35.8** (`-sm layer`), prefill unchanged at 74k tokens; 16k prefill −2.2 % (the correctly sized band
  moves the ring up 512 MiB).  Folds into **block 10** (slab) + **block 09** (drop reference) once the
  r84 block-00 fix lands; see `beta/0013-slab-narrow-floor/README.md`.

* **`0012-slab-transient-absorption`** — **BETA 2, 2026-10-18.**  Absorbs the slab's outside transients
  into the slab, loads the MTP **draft model before** the target context so the outside reserve is
  *measured* instead of predicted, makes a graph capture impossible while the wide band is armed, and
  raises the default headroom to `640 + 128 x n_gpu` (768 / 896 / 1024 MiB).  Validated on 2× R9700
  gfx1201 (`Qwen3.8-Flash-Next` IQ4_NL, `-ncmoe 48`, 131k ctx, MTP, 73k-token prefill) and gate-validated on
  that box: coherence identical, prefill-logit KLD 0.000000, adaptive-MTP 4 axes + phase-switching exactly
  equal on both arms.  See
  `beta/0012-slab-transient-absorption/README.md`; record in `wip/slab-transient-absorption/README.md`
  §10.16 and `WORKLOG.md` 261018.002.

_`rdna2` (issue #158) was folded into r70 blocks 06/10 and moved to `archive/work/rdna2/`
(record: `WORKLOG.md` 261016.002)._

_`fa-stage-prealloc` (issue #138) was promoted into r53 blocks 09/10 and moved to
`archive/work/fa-stage-prealloc/` (record: `WORKLOG.md` 261013.003)._

**Recently promoted.**  `fa-stage-prealloc/` (issue #138): the FA prefill staging arena is allocated
once at its eventual size, folded into blocks 09/10, default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` as
the kill switch.
