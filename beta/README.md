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

* One sub-directory per candidate: `beta/<name>/`.
* `README.md` — what the candidate is, why, the patch, and the status (dated entries, newest first).
* `BETA-TESTING.md` — the apply + build + gate checklist handed to the tester.
* `<name>.patch` — the patch itself, cut against the current `release.json` tree.  It must `git apply`
  and `git am` cleanly on a fresh `apply-all.sh` checkout; record its **sha256** and the applied tree
  in `README.md`.
* Nothing here is applied by `apply-all.sh`; the tester applies it explicitly on top of a clean
  delivery tree.

## Promotion

When the tester confirms the candidate, it is folded into the block that owns the code it changes (or
another block, at the maintainer's discretion), the block set is regenerated and re-validated, and the
candidate moves to `archive/work/<name>/` and is recorded in `WORKLOG.md`.  If it is refuted, the same
move happens with a "refuted" status.  A finished candidate leaves `beta/`; no stubs.

## Current candidates

_None — `rdna2` (issue #158) was folded into r70 blocks 06/10 and moved to `archive/work/rdna2/`
(record: `WORKLOG.md` 261016.002)._

_`fa-stage-prealloc` (issue #138) was promoted into r53 blocks 09/10 and moved to
`archive/work/fa-stage-prealloc/` (record: `WORKLOG.md` 261013.003)._

**Recently promoted.**  `fa-stage-prealloc/` (issue #138): the FA prefill staging arena is allocated
once at its eventual size, folded into blocks 09/10, default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` as
the kill switch.
