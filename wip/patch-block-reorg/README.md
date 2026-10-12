# `patch-block-reorg` — fundamental re-organisation of the block taxonomy

**Status: OPEN / scoping + plan (seeded 2026-10-12).** Not part of the delivery; nothing here may be
applied to the fork without the maintainer's explicit go-ahead (see the WIP and promotion rules in
`AGENTS.md`).

> **This is a fundamental re-org, not another r50 micro-fix.** The archived r50 merge-hygiene work
> (`archive/work/r50-merge-hygiene/`) was a *narrow, tree-invariant redistribution* whose only goal was to
> make the existing 16 block commits each build, so the rebased chain could be re-applied. It is useful
> prior art (its per-hunk move list is incorporated into `MIGRATION-MAP.md`), but it did **not** change
> the taxonomy, the ordering, or the hub-file coupling. This campaign does. Its success criterion is
> **the next re-base being cheap**, not merely the current one fitting.

> **Cold start.** Read this file, then `REBASE-RESILIENCE.md`, `TARGET-BLOCKS.md`, `MIGRATION-MAP.md`,
> `FILE-OWNERSHIP.md` and `OPEN-QUESTIONS.md`. Then read `AGENTS.md` ("Report findings before shipping",
> "WIP and promotion rules"), `archive/work/r50-merge-hygiene/README.md` + `REDISTRIBUTION.md`, and
> `scripts/gate-per-block-build.sh`. **Report the open questions to the maintainer and wait for decisions
> before moving any commit.** Do not restructure the `patches/` set until the maintainer picks the target
> and the granularity.

---

## 0. The problem: rebasing the block set is a hunk-jigsaw

A horror rebase onto a newer `master` exposed the structural reasons. They are not really "the 16 blocks
were slightly out of order"; they are:

1. **Giant merged blocks bundle unrelated concerns.** Block 15 is `+14 197 / −1 430` over ~212 files and
   carries a dozen unrelated wins (QSA memory, derived kq-mask, native KV, MoE-cache hooks, FA band,
   verify-band fusions, qwen4exp policy, build-time, env caching). Block 06 is `+10 108 / −348`; block 14
   is `+7 050`; block 08 is `+5 073`. When upstream moves one function a giant block touches, the whole
   block conflicts — including the parts that have nothing to do with the upstream change — and there is
   no way to skip/defer just the affected concern.
2. **Cross-patch forward dependencies.** Block 06's `ggml_cuda_try_fuse` additions referenced types/ops
   owned by 08/10/13/14/15. Any reorder or partial apply breaks, and the resolution is manual.
3. **Hub-file interleaving.** `ggml/src/ggml-cuda/ggml-cuda.cu` is touched by 10 of the 16 blocks;
   `common.cuh`, `mmvq.cu`, `mmq*.cu(h)`, `src/llama-context.cpp`, `tests/test-backend-ops.cpp`,
   `common/common.cpp`, `ggml-backend-meta.cpp` and `ggml-alloc.c` by 4–8 each. Upstream actively reworks
   several of these, so every rebase re-resolves the same functions for several independent features.
4. **Tiny blocks burn numbers.** 05 (7 lines), 07 (25), 09 (33), 11 (108) — each is a concern that belongs
   to a larger block, yet each is a separate apply/conflict unit.
5. **Generated instance TUs stranded from their type axis.** Because an FA type axis was introduced in one
   block and its `template-instances/*.cu` emitted in another, block 06 could not build on its own. This
   is a symptom of (1)–(3), not a separate disease.

## 1. What "rebase-resilient" means here (design goals)

`REBASE-RESILIENCE.md` works the failure model and ranks the target blocks by likely rebase cost. The
design goals that drive the target taxonomy:

- **G1 — One concern per block.** An upstream conflict must be attributable to exactly one concern, so a
  block can be temporarily dropped or deferred without losing unrelated work.
- **G2 — File orthogonality.** A block should own the files it modifies, and two blocks should not both
  modify the same file where it can be avoided. For hub files, one block owns a *region* and later blocks
  are append-only (see `FILE-OWNERSHIP.md`). **New files never conflict; edits to upstream functions
  always can.**
- **G3 — No forward dependencies.** A block may only depend on earlier blocks. The per-block `all` build
  gate (`scripts/gate-per-block-build.sh`, a *fresh* build per commit) is the mechanical proof.
- **G4 — Blocks should be droppable.** Each block should be independently revertable/skippable so a rebase
  can defer the one or two blocks that hit an upstream change, then land them separately.
- **G5 — Order by upstream stability and subsystem layering.** Foundational/low-churn first; actively
  churning upstream functions touched by as few blocks as possible; model-specific and cross-cutting
  sweeps last.
- **G6 — Generated instances travel with their type axis.** Never split an instantiation from the
  dispatch case that selects it (it breaks G3).
- **G7 — A stable, mnemonic taxonomy.** Names should survive renumbering so docs and `WORKLOG` history
  remain intelligible (see Q6).

The tension between orthogonality (G2) and hub-file reality is the central design problem; §"Hub-file
strategy" of `REBASE-RESILIENCE.md` proposes the split into a *feature* half and a small *integration*
half where the churn can be isolated.

## 2. Audit method and evidence

Measured on the re-cut chain in `~/llama.cpp` (`rdna-boosts` tip `af0595360`, tree
`f102245dfb116f02f6b3f572df7fb413a5c65cf8` — **identical to `release.json.tree`**, so this chain is
content-equivalent to the published `84058162b`; the published `patches/` carry different `From:` hashes
because a re-cut re-applies the series).

```bash
git -C ~/llama.cpp show --numstat --format="" <block> | awk '{a+=$1;d+=$2} END{print a,d}'
git -C ~/llama.cpp show --name-only --format="" <block> | grep -cv template-instances
git -C ~/llama.cpp show -s --format="%h %s%n%b" <block>
# pairwise overlap matrix -> FILE-OWNERSHIP.md
```

## 3. Audit findings

### 3.1 Size and file counts

| old block | subject | added | removed | all files | non-template files |
|---|---|---:|---:|---:|---:|
| 00 | structural and architecture fixes | 90 | 8 | 4 | 4 |
| 01 | adaptive MTP draft depth | 959 | 38 | 19 | 19 |
| 02 | fused chunked GDN prefill | 3 141 | 15 | 22 | 22 |
| 03 | BF16 KV cache + native-BF16 FA | 630 | 214 | 8 | 8 |
| 04 | RDNA4 WMMA FA + Q6_K MMQ | 284 | 21 | 9 | 9 |
| 05 | CPU bit-identical decode/verify | 7 | 1 | 1 | 1 |
| 06 | general system-operations bucket | 10 108 | 348 | 194 | 51 |
| 07 | meta device-wrapper skip | 25 | 15 | 1 | 1 |
| 08 | fused-core prefill kernels | 5 073 | 135 | 47 | 31 |
| 09 | meta-buffer compute-container headroom | 33 | 3 | 4 | 4 |
| 10 | k-quant boosts | 411 | 37 | 5 | 5 |
| 11 | skip CUDA graphs for multi-token prefill | 108 | 14 | 2 | 2 |
| 12 | hybrid HIP all-reduce | 2 231 | 44 | 4 | 4 |
| 13 | fused MoE gate+up+GLU MMQ + mmvq | 2 134 | 332 | 19 | 13 |
| 14 | qwen4exp support | 7 050 | 398 | 54 | 54 |
| 15 | campaign memory wins | 14 197 | 1 430 | 212 | 107 |

Generated `template-instances/*.cu` dominate the file counts of 06 (143), 15 (105), 08 (16) and 13 (6).

### 3.2 The coupling matrix

The full matrix and per-file owner map live in `FILE-OWNERSHIP.md`. The headline: `ggml-cuda.cu` is
touched by 10/16 blocks, and nine other files by 4–8 blocks.

### 3.3 The specific pathologies

- **Block 06 is a bucket.** Generic expert-cache iface + scheduler rebalance + Meta delegation + CUDA
  engine + `GGML_ENV_STR` + host mmap reclaim + async inputs/pool cap + 143 FA template instances +
  scattered flags. The r50 record lists the forward-dependency hunks it moved out; the residual is still
  a bucket.
- **Block 15 is a super-block** — the single biggest rebase liability (see §0.1).
- **Block 04 mixes FA and MMQ.** The `GGML_CUDA_MMQ_J_MAX` knob and the `mmq-vec-dot.cuh` x_scale hoist
  belong to quant-GEMM (target 09), not FA (target 05).
- **Tiny blocks are homeless but owned:** 05 → foundation band-uniformity; 07 → foundation Meta wrapper;
  09 → split foundation (compute headroom) vs MoE slab; 11 → expert-offload graph handling.
- **Instances stranded from their axis** (see §0.5).

## 4. Target organisation (summary)

Full spec: `TARGET-BLOCKS.md`. Recommended: **19 blocks (00–18)**, grouped foundation → multi-GPU →
decode/speculation → flash attention → quantized GEMM → kernel fusions → host-resident MoE → memory →
models → host hot-path.

| new | name | absorbs |
|---|---|---|
| 00 | Foundation: band-uniformity, Meta-wrapper, allocator & base correctness | 00, 05, 07, 09a, 06-logits-order, 15-W4, 15-cpy-batch |
| 01 | Hybrid HIP all-reduce | 12 |
| 02 | Adaptive MTP draft depth | 01 |
| 03 | Chunked gated-delta-net prefill | 02 |
| 04 | BF16 KV cache + native-BF16 flash attention | 03 |
| 05 | Flash-attention RDNA4 WMMA band (GQA fold, wide verify band) | 04-FA, 15-band |
| 06 | Native quantized K/V reads in flash attention | 15-V4/V5, stage arena, prefill band split |
| 07 | Derived kq-mask attention memory | 15-V3 |
| 08 | MMB quantized prefill GEMM | 08-MMB |
| 09 | K-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0 perm, grid pad) | 10, 04-MMQ, 15-Q1_0/grid/PR57 |
| 10 | Fused prefill kernels (conv1d, RMS-norm, top-k MoE, verify fusions) | 08-fusions, 15-PR63 |
| 11 | Fused MoE kernels (gate+up+GLU MMQ, item-split, shared-expert, MWR) | 13, 08-mmID, 15-PR114 |
| 12 | Host-resident MoE: interface, scheduler, Meta, slab | 06-iface/sched/meta/slab, 09b, 15-r48 |
| 13 | CUDA decode expert-cache engine, consumer hooks & arena | 06-engine, 13-engine, 15-hooks |
| 14 | CUDA/HIP graph handling for expert-offload decode | 11, 06-async/pool |
| 15 | Host memory reclaim (mmap release) | 06-mmap, 15-r49-mmap |
| 16 | qwen4exp model support (arch, lazy reader, HC, MTP head) | 14-model/HC/lazy, 15-HC16/policy |
| 17 | QSA sparse attention, indexer and QSA memory | 14-QSA/indexer, 15-W1–W3 |
| 18 | Host hot-path flag caching (getenv) | 15-env-caching |

Two seams are deliberately left as maintainer decisions (`OPEN-QUESTIONS.md` Q1): whether the two MoE
blocks (12/13) stay separate or merge, and whether the two qwen4exp blocks (16/17) stay separate or merge.

## 5. Ordering and dependency DAG

```
00 Foundation ─────────────────────────────────────────────┐
  ├─ 01 All-reduce                                         │
  ├─ 02 Adaptive MTP                                       │
  ├─ 03 Chunked GDN                                        │
  ├─ 04 BF16 KV / native BF16 FA                           │
  │     ├─ 05 FA band ── 06 native K/V ── 07 derived mask  │
  ├─ 08 MMB ─── 09 K-quant boosts ─── 11 MoE fusions ──┐   │
  ├─ 10 Fused prefill kernels ─────────────────────────┤   │
  │                                                    ├─ 13 expert-cache engine
  │                        12 MoE iface/sched/slab ────┤
  │                                                    └─ 14 graph handling
  ├─ 15 Host memory reclaim
  ├─ 16 qwen4exp model ── 17 QSA/indexer (needs 05/06/07/08/11)
  └───────────────────────────────────────────────────── 18 host flag caching
```

Hard dependency notes: 05 needs 04's `type_KV` axis; 06/07 need 05; 08 sets the FA-quants / MMB policy;
11 interleaves with 09's mmvq dispatch; 13 needs 12's iface and 11's kernels; 14 needs 13's arming; 18 is
last by construction; 17 needs 05–08/11 and 16's model plumbing.

## 6. Two-phase plan

**Phase 1 — taxonomy re-org (this campaign's first deliverable).** Redistribute the existing hunks into
the 19-block target, reorder, absorb the tiny blocks, and split the super-blocks — **as far as possible a
pure commit-boundary move keeping the net tree `f102245…` invariant.** Validate with a fresh per-commit
`all` build + `validate-set.sh` + the release gates. This alone removes the forward deps and the giant
blocks.

**Phase 2 — rebase-resilience refactor (optional, decided after Phase 1).** Where a feature's code is
interleaved into an actively-churning upstream function (`ggml_cuda_try_fuse`, the mmvq dispatch,
`llama-context.cpp` glue), extract the feature's logic into a **feature-local file/function** and reduce
the block's upstream-file footprint to a small, contiguous insertion. `REBASE-RESILIENCE.md` ranks the
candidates. Phase 2 is allowed to change the code (not just the boundaries); it must re-run the net tree
comparison against its own new baseline and the relevant gates. **Do not start Phase 2 until the
maintainer agrees** — it is a larger change than a re-org.

## 7. Migration method

The full step map is `MIGRATION-MAP.md` (it incorporates the r50 per-hunk move list). Mechanics, from the
r50 lesson:

1. Freeze the tip on a `pin/` branch, record the tree hash.
2. Rebuild in the new order by `git rebase -i 097cf33d9` with `edit` stops and direct file edits
   (**never** `--autosquash`: a fixup's base is the tip, so it conflicts/double-applies), **or** — cleaner
   when the target order differs a lot — reconstruct from the pin tree: for each target block, extract
   exactly its hunks and commit.
3. After each target block: fresh `cmake … && cmake --build --target all -k` before continuing. A block
   that builds only at the tip is not a block.
4. Force the tree back to the pin at the end so the final tree is byte-identical by construction, then
   `scripts/validate-set.sh`.
5. `scripts/make-release.sh` writes `release.json` (never hand-edit hashes); `n_blocks` changes.

## 8. Validation / definition of done

- **Tree invariant** (Phase 1): final tip tree == `f102245dfb116f02f6b3f572df7fb413a5c65cf8`.
- **Per-block build**: `scripts/gate-per-block-build.sh` → N/N (fresh `all` per commit).
- **Patch apply**: `scripts/validate-set.sh` green.
- **Release gates re-run**: prefill-logit (`0.000707 / 98.755 %`), qwen4exp quant coherence, same-seed
  coherence, 35B-A3B draft-mtp.
- **Doc freshness**: update *current-state* docs only (`patches/README.md` top, `README.md`,
  `ENVIRONMENT.md`, `AGENTS.md` Critical facts, `TODO.md`, `release.json`); dated records stay historical
  and the old→new mapping is published here + `WORKLOG.md`.
- **Fork branch refresh + container images** per the Pushing policy when this becomes a release.

## 9. Documentation / renumbering impact (a real cost)

Approximate `block NN` reference counts in `*.md` at r50: block 15 = 1132, block 13 = 565, block 14 = 539,
block 06 = 492, block 08 = 226, block 12 = 213, block 10 = 160, block 00 = 171, blocks 04/11 = 49,
blocks 05/07/09 = 8–19; 303 markdown files carry block references. Dated records are not edited
(historical), but current-state docs are, and `AGENTS.md` Critical facts alone carries ~50. `Q6` records
the options (renumber-and-map vs. keep old numbers and change contents only vs. adopt mnemonic names).
Do not start the renumber until the maintainer picks.

## 10. Open decisions

See `OPEN-QUESTIONS.md`. Blocking: **Q1** target count and the MoE/qwen4exp seams; **Q2** aggressive vs
minimal reorder; **Q3** tree-invariant only vs. allow small content clean-ups; **Q6** renumbering/doc
strategy. `REBASE-RESILIENCE.md` adds **Q7** whether to adopt a feature/integration block split
(Phase 2) and **Q8** whether to accept Phase 2 at all in this campaign.

## 11. Files in this campaign

| file | contents |
|---|---|
| `README.md` | this handover: problem, design goals, audit, target, order, method, DoD |
| `REBASE-RESILIENCE.md` | why re-bases hurt, per-block rebase-cost ranking, hub-file strategy, Phase-2 candidates |
| `TARGET-BLOCKS.md` | per-target-block spec (concern, sources, owner files, deps, gates) |
| `MIGRATION-MAP.md` | old→new mapping, file/hunk level, incl. the r50 hygiene moves |
| `FILE-OWNERSHIP.md` | coupling matrix, hub-file region map, generated-instance rule |
| `OPEN-QUESTIONS.md` | decisions required from the maintainer, with recommendations |
