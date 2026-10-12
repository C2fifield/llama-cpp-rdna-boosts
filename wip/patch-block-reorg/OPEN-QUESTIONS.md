# Open questions — decisions required from the maintainer

Per `AGENTS.md` ("Report findings before shipping"), these are the decisions that must be made **before**
any commit is moved. Each has a recommendation; the maintainer's call is final and should be recorded
(here, and in the relevant `WORKLOG.md` entry).

---

## Q1 — Target block count and the two contested seams (BLOCKING)

The recommended target is **19 blocks (00–18)** (`README.md` §4, `TARGET-BLOCKS.md`). Two seams are
genuinely arguable:

- **Q1a — MoE: keep target 12 (interface/scheduler/Meta/slab) and 13 (cache engine/hooks) separate, or
  merge into one "Host-Resident MoE Experts + Slab handler" block?**
  The user's brief named "Host-Resident MoE Experts + Slab handler" as *one* block. The two-block split
  follows the layer seam (generic plumbing vs. policy + hooks) and the user's other instruction to break
  up oversized blocks. They share `moe-expert-cache.{cu,h}` and `ggml-cuda.cu` regions.
  **Recommendation: keep them separate** (12 vs 13), because a merged block would again be ~7k lines and
  ~20 modified upstream files — a rebase super-block. If the maintainer prefers the brief's single block,
  merge 12+13 and accept the size.
- **Q1b — qwen4exp: keep target 16 (model/HC/lazy/MTP head) and 17 (QSA/indexer/QSA memory) separate, or
  merge into one qwen4exp block?**
  Same tension. 17 is the larger half (~5k lines) but the QSA op code is fairly self-contained; the seam
  is `src/models/qwen4exp.cpp` and `src/llama-graph.cpp`.
  **Recommendation: keep them separate** (16 vs 17), with the caveat that the W1–W3 memory wins touch
  graph/model files shared with 16, so the two must be adjacent and 17 after 16.

**Decision:** _pending._

## Q2 — Reorder aggressively or minimally?

- **Aggressive** (recommended): adopt the full order in `README.md` §5. Adds `01 All-reduce` early and
  moves qwen4exp/MoE late; minimises hub-function interleaving and forward deps. Cost: the redistribution
  is a real reconstruction, and every block's commit content changes.
- **Minimal**: keep the existing order, only absorb the tiny blocks and split the giants. Far less work,
  but leaves the forward-dep-prone order and does not group related blocks.

**Recommendation: aggressive**, because the whole point is to avoid the next rebase's pain; a minimal
reorder only fixes the current one (which r50 already did).

**Decision:** _pending._

## Q3 — Tree-invariant only, or allow small content clean-ups?

Phase 1 should keep the net tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8` byte-identical. During
redistribution, small content problems may surface (a stray hunk, a duplicate from the r50 lateral move,
a comment citing a moved symbol).

- **Option A (recommended):** Phase 1 is strictly tree-invariant; any content issue is recorded and, if
  the maintainer wants it fixed, deferred to a separate follow-up commit (not silently folded).
- **Option B:** allow a small, listed set of clean-ups inside Phase 1 and re-baseline the tree.

**Recommendation: A.** Tree invariance is the strongest proof the re-org is correct; mixing content fixes
makes a failed per-block build ambiguous.

**Decision:** _pending._

## Q4 — Distribute the getenv sweep (target 18) or keep it a sweep block?

Target 18 caches `getenv` on hot paths. As a block it touches ~40 files owned by other blocks, so it is
the highest-coupling block and can only be last.
- **Option A (recommended):** keep it a single trailing block and accept that it is a sweep; it is
  pure-perf, droppable, and last by construction.
- **Option B:** distribute each cached flag to the block that introduced the flag. Zero cross-block
  coupling, but 10–15 blocks each carry an unrelated micro-hunk, and the "cache the getenv" change
  becomes invisible as a concern.
- **Option C:** drop target 18 from the re-org and keep the sweep as a standalone later campaign.

**Recommendation: A**, with C as the escape hatch if the maintainer dislikes sweep blocks.

**Decision:** _pending._

## Q5 — What happens to the r50 per-block build gate and `release.json`?

The re-org changes `n_blocks`. `scripts/gate-per-block-build.sh` reads `release.json`'s `base..tip`, so it
must be regenerated before the gate runs. The tip commit changes even though the tree does not.
- **Recommendation:** run `scripts/make-release.sh` for a scratch manifest, run the gate against the new
  chain, and only then overwrite `release.json`; `validate-set.sh` proves the tree. No hand-edited hashes.

**Decision:** _pending (mechanical; flag if a different flow is wanted)._

## Q6 — Renumbering / documentation strategy (BLOCKING for the doc pass)

Renumbering invalidates the numeric handles used everywhere (`block 15` alone has ~1132 references in
303 markdown files). Dated records must stay historical. Options:

- **Option A — Renumber + mapping table (recommended for a fundamental re-org).** New numbers 00–18;
  update only *current-state* docs (`patches/README.md` top, `README.md`, `ENVIRONMENT.md`,
  `AGENTS.md` Critical facts, `TODO.md`, `release.json`, `MANIFESTS.md`/`BASELINE.md` headers). Add an
  old→new table to this campaign and to `WORKLOG.md`. Dated records keep their original numbers.
  Risk: `AGENTS.md`'s "Critical facts" block references are the most-used and must be updated carefully.
- **Option B — Keep the old numbers; change only contents; append new blocks as 16, 17, ….** Minimises doc
  churn but produces a confusing order and cannot express the reorder cleanly.
- **Option C — Mnemonic names as the primary handle** (`block 03-gdn`, `block 09-kquant`, …) with numbers
  secondary. Most future-proof (G7) and the recommended end state combined with A, but it is a larger doc
  rewrite now.
- **Option D — Hybrid:** new numbers + a short mnemonic alias per block, published in one table that the
  current-state docs can use (`block 09-kquant`). Historical docs keep old numbers.

**Recommendation: D** (new numbers + mnemonic aliases), which is A plus the G7 stability benefit; fall
back to A if the alias table is judged too much extra work.

**Decision:** _pending._

## Q7 — Adopt a feature/integration block split (Phase 2)?

A per-feature "integration block" (all dispatch/registration in a dedicated block, features as new files
only) would give the best rebase resilience (`REBASE-RESILIENCE.md` §6), at the cost of two blocks per
feature and a larger Phase-2 refactor.
- **Recommendation: no as a separate-block scheme; yes to its spirit** — enforce "one-line arm → feature-
  local function" so the upstream-file hunk is tiny, without a second block. Revisit after Phase 1's
  measurement.

**Decision:** _pending._

## Q8 — Is Phase 2 in scope for this campaign at all?

Phase 1 is a boundary re-org; Phase 2 changes code. `AGENTS.md` requires reporting and a maintainer
decision before delivery-affecting work.
- **Recommendation:** scope this campaign to **Phase 1 + the Phase-2 measurement**; open a **separate**
  campaign for any Phase-2 refactor once the post-Phase-1 numbers are in.

**Decision:** _pending._

---

## Other findings to report (not decisions, but noted)

- The local `~/llama.cpp` `rdna-boosts` tip (`af0595360`) is tree-identical to the published
  `release.json.tree`, but its commit hashes differ from the published `patches/` `From:` hashes. All
  redistribution must start from a canonical chain rebuilt at `release.json.base` via
  `scripts/apply-all.sh`, per the existing "Regenerate the patches" flow.
- Block 05 shares **no file** with any other old block — the cleanest possible absorb into target 00.
- Block 09 is genuinely two concerns (`compute_headroom` for recurrent views vs. slab headroom for the
  MoE arena); the split is not optional if target 00 and 12 are to be orthogonal.
- The r50 hygiene's lateral move of the `ggml_cuda_try_fuse` dispatch/matchers into block 15 **grew the
  super-block**; the new taxonomy must reverse that by sending each matcher to its owner block
  (`MIGRATION-MAP.md` old-06). If Phase 1 keeps the r50 placement, the goal is not met.
