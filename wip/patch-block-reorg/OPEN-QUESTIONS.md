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

## Q4 — getenv hot-path caching: sweep block vs. folded — **OPEN (recommended: fold)**

**What the choice actually is.** Old-15's issue-#65 amendment caches ~25 `getenv()` reads that sit on
per-op / per-graph / per-tensor hot paths (`LLAMA_HC_CN_DEBUG`, `GGML_CUDA_DISABLE_CONV_FUSION`,
`GGML_CUDA_GCDBG`, `GGML_CUDA_OP_TIMING`, `GGML_STREAMDBG`, `GGML_META_*`, `GGML_SCHED_*`,
`GGML_CUDA_MMQ_J_MAX`, `GGML_CUDA_GDN_CHUNKED(_BF16)`, …). Each cached read is one or two lines in the
file that *owns the flag*.

- **Fold (recommended):** each cache-the-getenv line goes into the block that introduced the flag. No
  block touches ~40 files; the change is invisible as a concern; perfectly orthogonal (G2). The new
  total is **18 blocks (00–17)**.
- **Sweep block (18):** keep them as one trailing block. Easier to revert as a unit, but it is the most
  coupled block in the set (touches ~40 files owned by others) and can only be last.

**Recommendation: fold.** This is consistent with the Q2 V2.0/orthogonality direction. This is not about
`ENVIRONMENT.md` (which already groups variables by feature) — it is purely about the **patch-block**
structure.

**Decision:** _pending (fold assumed in the docs)._

## Q5 — `release.json` / per-block gate flow — **NOT A DECISION (procedure)**

There is no real choice here; I should not have listed it as one. It is procedure: the re-org changes
`n_blocks` and the tip, so `release.json` must be regenerated with `scripts/make-release.sh` (never
hand-edited) and `scripts/gate-per-block-build.sh` then reads the new `base..tip`. No action required.

## Q7 — Feature-block + integration-block split — **OPEN (clarified)**

**What the choice is.** A *code-structure* change (Phase 2), not a Phase-1 taxonomy choice: for the
features whose logic is interleaved into an actively-churning upstream function (`ggml_cuda_try_fuse`,
the mmvq dispatch, `fattn.cu`), extract the feature's logic into a feature-local function/file and reduce
the block's upstream-file hunk to a **one-line call**. The alternative is a literal second "integration
block" per feature holding all dispatch edits.

- **Recommended:** the one-line-call extraction, **not** a second block per feature. It gets most of the
  rebase-resilience benefit without splitting one concern across two blocks (G1) or breaking the
  per-block build gate.
- **Decision needed only if** you want the literal integration-block scheme.

**Decision:** _pending._

## Q8 — Is Phase 2 in scope for this campaign? — **OPEN (clarified)**

**What the choice is.** Phase 1 = move hunks between blocks (no code change). Phase 2 = refactor the code
so feature logic is local to feature files (`REBASE-RESILIENCE.md` §5). Phase 2 is delivery-affecting and
needs its own gates (it can change the net tree from the Phase-1 baseline).

- **Recommended:** finish **Phase 1** (and re-measure the modified-file count per block), then open a
  **separate campaign** for the Phase-2 items that still dominate. The cheap item (one-line `try_fuse`
  arms) can be pulled into the V2.0 effort if you want.
- Given the Q2 V2.0 framing you may prefer Phase 2 to be part of V2.0. Either is fine; the only thing to
  avoid is doing a code refactor inside the boundary move and losing the tree-invariant proof.

**Decision:** _pending._

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
