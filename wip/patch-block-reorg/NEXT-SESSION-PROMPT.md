# Next-session prompt (copy-paste)

Paste the block below as the first message of a fresh session. It is self-contained; the session should
read `HANDOFF.md` and continue the Phase-1 execution.

---

We are in `/home/stew675/llama-cpp-rdna-boosts` working the `wip/patch-block-reorg/` campaign: a
fundamental re-organisation of the 16-patch delivery into an orthogonal, rebase-resilient block set
(V2.0 = Phase 1 boundary re-org + Phase 2 self-contained refactor).

**Read first, in this order:**
`wip/patch-block-reorg/HANDOFF.md` (cold-start execution state and next steps),
`wip/patch-block-reorg/PHASE1-STATUS.md` (findings + decisions),
`wip/patch-block-reorg/README.md`, `REBASE-RESILIENCE.md`, `FILE-OWNERSHIP.md`,
`wip/patch-block-reorg/tools/`, `AGENTS.md` ("Report findings before shipping", "WIP and promotion
rules", "Critical facts"), and `archive/work/r50-merge-hygiene/REDISTRIBUTION.md`.

**Current state.** Target blocks **00–04** are rebuilt and committed on the Phase-1 worktree
`~/llama-v2`, branch `v2-phase1`, clean checkpoint tag `pin/v2-phase1-00-04` (`2a1502311`), base
`097cf33d9`, reference tip `84058162b` (tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8`). The delivery
`patches/` are untouched. All Q1–Q8 decisions are recorded. A partial FA attempt is on scratch branch
`v2-fa-wip` (`116ff4ccf`) — reference only.

**Final approved target: 14 blocks** (00 foundation, 01 allreduce, 02 mtp-adaptive, 03 gdn, 04 bf16-kv,
05 fa, 06 mmb, 07 kquant, 08 fused-prefill, 09 fused-moe, 10 slab-moe, 11 qwen4exp, 12 qsa,
13 mmap-reclaim). Slab+MoE is grouped ahead of qwen4exp; the `fattn_stage` arena is deferred out of `05 fa`
into `10 slab-moe`.

**Your task: continue Phase 1.** Start with `05 fa` (see HANDOFF.md §5 for the exact apply commands and
the two known conflicts; exclude the `fattn_stage` arena and reconstruct the pre-arena staging in
`launch_fattn`). Then proceed through `06 mmb`, `07 kquant`, `08 fused-prefill`, `09 fused-moe`,
`10 slab-moe`, `11 qwen4exp`, `12 qsa`, `13 mmap-reclaim`, one block per commit on `v2-phase1`.

**Git checkpointing:** after each block, commit on `v2-phase1`; do **not** push anything to the fork
without an explicit instruction. The `~/llama-cpp-rdna-boosts` repo (`main`) is the place for campaign
docs; commit + push those as you go.

**Mandatory:**
- Every block must build on its own with a **fresh** `all` build:
  `~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama-v2 --base 097cf33d9 --tip "$(git -C ~/llama-v2 rev-parse v2-phase1)" --build-dir build-per-block`.
- Run the owning region's gates (HANDOFF.md §7) and the release gates
  (`scripts/gate-prefill-logits.sh`, `scripts/gate-qwen4exp-quant-coherence.sh`, same-seed coherence,
  35B-A3B `--spec-type draft-mtp`).
- The delivery `patches/`/`release.json` are **not** modified until the full 14-block chain is built and
  gated; then regenerate with `scripts/make-patches.sh` / `scripts/make-release.sh` and `validate-set.sh`.
- Surface any new structural finding to the maintainer and wait before changing the plan (AGENTS.md).

**Pitfalls:** use `git apply -3` and `tools/select_hunks.py`; never `--autosquash`; do not keyword-triage
the squashed old-15/old-06 without reading each hunk (a `band` keyword matches the GLU→Q8_1 fusion). See
HANDOFF.md §8.

Report progress after each block (what moved, the gate result), and ask the maintainer before any
delivery-affecting commit.
