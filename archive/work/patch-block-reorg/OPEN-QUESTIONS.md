# Open questions — decisions from the maintainer

Per `AGENTS.md` ("Report findings before shipping"), these are the decisions that gate the re-org. The
maintainer's answers of **2026-10-12** are recorded; only Q4/Q5/Q7/Q8 remain open or for clarification.

---

## Q1 — Separate the MoE and qwen4exp concerns; order qwen4exp first — **DECIDED (separate)**

**Decision (maintainer):** keep the MoE pair and the qwen4exp pair **separate** (targets 14/15 and 12/13),
and order **qwen4exp before the host-resident MoE blocks**. Rationale: qwen4exp is independent, but its
architecture is what makes host-resident MoE effective; host MoE is the broad enabler across all MoE
models and therefore lands after the model-specific work.

**Consequence applied:** the r50 lateral move of the `ggml_cuda_match_hc_mix` /
`_hc_combine_norm_*` / `_match_idx_relu_sum` matchers (which need `ggml_cuda_hc_mix_args`) is **reversed**:
they belong to the qwen4exp block (12), not the MoE blocks. Verified that old-13 (Fused MoE) fusions are
shape-matched and have no compile dependency on qwen4exp types, so Fused MoE (11) stays before qwen4exp.

## Q2 — Aggressive reorder; V2.0 clean break — **DECIDED (aggressive / V2.0)**

**Decision (maintainer):** be as aggressive as needed. Rewrite the top-level documents, move much of the
old block history to `archive/`, and establish a clean break-point — treat this as **V2.0** of the
original work rather than an incremental renumbering. Historical detail remains discoverable in the
archive.

**Consequence applied:** target order `00 … 17` per `README.md` §4; dated records are not rewritten, but
current-state docs are re-issued for V2.0; the old→new map is published here/`WORKLOG.md` and the old
`patches/README.md` history is archived rather than maintained forward.

## Q3 — Tree-invariant where practical, document exceptions — **DECIDED (practical invariant)**

**Decision (maintainer):** Phase 1 should be as tree-invariant as practical, but the choice is not
black-and-white; any content cleanup that is needed is documented as a **Phase-2 item** rather than
silently folded. This matches the original Option A with an explicit exception list.

**Consequence applied:** `CONTENT-CLEANUP` items are collected in `REBASE-RESILIENCE.md` §5 / a running
list; each must be listed and approved, never hidden inside a boundary move.

## Q6 — New numbers + mnemonic aliases, clean doc break — **DECIDED**

**Decision (maintainer):** yes. New numbers plus mnemonic aliases (`00-foundation`, `09-kquant`, …), with
the V2.0 clean break from Q2.

**Consequence applied:** current-state docs (`README.md`, `AGENTS.md` Critical facts, `ENVIRONMENT.md`
headers, `patches/README.md` top, `TODO.md`, `release.json`, `MANIFESTS.md`/`BASELINE.md` headers) use the
new numbers/aliases; dated records keep their historical numbers; the old→new table is published.
Block-15 alone had ~1132 references across 303 markdown files, so the doc pass is a real work package.

---

## Q4 — getenv hot-path caching: folded into owning blocks — **DECIDED (fold)**

**Decision (maintainer):** fold every cached `getenv` into the block that owns the flag. The maintainer
explicitly accepted that this means multiple blocks touch the same file; keeping the concern aligned
with its owner is worth it, and env-var reads are the easiest class to resolve at rebase time. Target is
**18 blocks (00–17)**; there is no sweep block.

**Consequence applied:** each cached read rides in `TARGET-BLOCKS.md`'s owning block; the `MIGRATION-MAP.md`
old-15 entry marks the issue-#65 sweep as "folded".

## Q5 — `release.json` / per-block gate flow — **RESOLVED (procedure, no decision)**

No choice exists. The re-org changes `n_blocks` and the tip, so `release.json` is regenerated with
`scripts/make-release.sh` (never hand-edited) and `scripts/gate-per-block-build.sh` then reads the new
`base..tip`. Downstream this now applies at two checkpoints: the end of Phase 1 and the end of Phase 2.

## Q7 — Self-contained functions/files, call-point-only interleaving — **DECIDED**

**Decision (maintainer):** the general Phase-2 approach is to extract each concern into **self-contained
functions, ideally its own file**, and interleave into upstream/churning files as little as possible —
only the call point. Interleaving should be minimised everywhere; where a concern can live entirely in
its own file with a single plug-in call site, do that. Some cases will not allow it, but it is the default.

**Consequence applied:** this is now a first-class design principle (G8) in `README.md` and the drive of
`PHASE2-REFACTOR.md`, which lists the per-concern extraction targets and their (single) call points.
The literal "second integration block per feature" scheme is **not** adopted; the self-contained-function
form gives the benefit without splitting one concern across two blocks (G1) or complicating the per-block
build gate.

## Q8 — Phase 2 is in scope now; V2.0 = Phase 1 + Phase 2 — **DECIDED**

**Decision (maintainer):** do Phase 2 now. **V2.0 is Phase 1 and Phase 2 together**; Phase 1 is simply the
intermediate checkpoint (block-boundary re-org, validated, tree-invariant where practical). Phase 2 is
the self-contained refactor (Q7). A cheap straggler that does not fit the Phase-2 pass (e.g. a residual
`try_fuse` one-liner) can be picked up in a **Phase 3**, if needed.

**Consequence applied:** `README.md` §6 is now a three-phase V2.0 plan; `PHASE2-REFACTOR.md` is the
Phase-2 work plan; the DoD requires the Phase-1 checkpoint to be gated before Phase 2 begins, and the
final V2.0 tree to be re-baselined (`release.json`) and gated again. Phase 2 changes the net tree from the
Phase-1 baseline, so the tree-invariance proof applies to Phase 1 only.

---

## Other findings (not decisions)

- The local `~/llama.cpp` `rdna-boosts` tip (`af0595360`) is tree-identical to the published
  `release.json.tree`, but its commit hashes differ from the published `patches/` `From:` hashes. All
  redistribution must start from a canonical chain rebuilt at `release.json.base` via
  `scripts/apply-all.sh`.
- Block 05 shares **no file** with any other old block — the cleanest possible absorb into target 00.
- Block 09 is two concerns (`compute_headroom` for recurrent views vs. slab headroom for the MoE arena);
  the split is required if targets 00 and 14 are to be orthogonal.
- The r50 lateral move of the `ggml_cuda_try_fuse` dispatch/matchers into block 15 **grew the
  super-block**; Q1 reverses it by sending the qwen4exp matchers to 12 and the cache-blocks matcher to 15.
