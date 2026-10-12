# Rebase resilience — why re-bases hurt and what the new taxonomy does about it

Companion to `README.md`. This is the *design* document for the fundamental re-org: it names the failure
mechanisms, ranks the target blocks by how much they are likely to hurt on the next upstream re-base, and
proposes the structural choices that reduce that cost. It also scopes an optional Phase 2 code refactor.

Everything here is a plan; nothing is applied.

---

## 1. Failure model of a re-base

A re-base replays every block commit onto a new upstream base with `git am` / 3-way merge. It is cheap
exactly when the diff between the old and new upstream base does not overlap a block's hunks. Each
mechanism below turns a non-overlapping change into manual work:

| # | mechanism | why it hurts | what reduces it |
|---|---|---|---|
| F1 | A block edits an upstream function that upstream also changed | 3-way conflict in that function; the block's hunk context is gone | smaller hunks; feature logic in new/local functions; fewer blocks per upstream function |
| F2 | A block edits a file that many blocks edit | every block that touches the file is at risk from one upstream change; resolution order matters | one block owns a file (G2); hub files split into feature vs integration |
| F3 | A block is a bundle of unrelated concerns | a conflict in any one concern fails the whole block, and the unrelated parts cannot be skipped | one concern per block (G1) |
| F4 | A block references code added by a later block | reorder/partial apply breaks; the resolution is manual | no forward deps (G3); fresh per-block build gate |
| F5 | An instance TU is split from its dispatch case | the owning block does not build alone; the split is invisible until link | instances travel with the axis (G6) |
| F6 | A block's hunks are large contiguous rewrites | upstream context drift moves every line | prefer additive/new files; keep edits contiguous and small |
| F7 | Historical block numbers are the only handle | a renumber invalidates docs/handles and makes history unreadable | stable mnemonic names (G7); mapping table |

The 2026-10 horror rebase hit F2 (10 blocks on `ggml-cuda.cu`), F1 (`ggml_cuda_try_fuse`, `mmq.cu` J
selection, `llama-context.cpp`), F3 (blocks 06/15), and F4 (block 06 forward deps).

## 2. The raw metric: modified vs. added files

New files (`A`) never conflict. Modified upstream files (`M`) are the entire conflict surface. Measured
per old block (`git show --name-status`):

| old block | added files | modified files | deleted |
|---|---:|---:|---:|
| 00 | 0 | 4 | 0 |
| 01 | 3 | 16 | 0 |
| 02 | 4 | 18 | 0 |
| 03 | 0 | 8 | 0 |
| 04 | 0 | 9 | 0 |
| 05 | 0 | 1 | 0 |
| 06 | 124 | 49 | 16 |
| 07 | 0 | 1 | 0 |
| 08 | 24 | 23 | 0 |
| 09 | 0 | 4 | 0 |
| 10 | 0 | 5 | 0 |
| 11 | 0 | 2 | 0 |
| 12 | 1 | 3 | 0 |
| 13 | 0 | 19 | 0 |
| 14 | 14 | 40 | 0 |
| 15 | 95 | 105 | 8 |

Block 06 has 124 added files — but 143 of them are *generated* `template-instances` that belong to other
blocks. Once those travel with their axis (G6), 06's real added-file count collapses. Block 15's 95 added
files are mostly generated instances and new `.cu` kernels; its 105 modified files are the real problem.

**Target design consequence:** maximise the number of *new* files per target block and minimise the
number of *modified upstream* files, especially shared ones. The 19-block target already does this by
giving the big generated/feature code to its own blocks; Phase 2 (§5) pushes further.

## 3. Per-target-block rebase-cost ranking

Estimated from the migration map. "Modified-upstream files" is the conflict surface; "hub share" is how
many of them are files other target blocks also modify. Cost = likelihood that an upstream change lands
in the block's surface × the interleaving depth.

| target | modified-upstream files (est.) | hub share | interleaving depth | rebase cost | notes |
|---|---|---:|---|---|---|
| 00 Foundation | ~12 | high | shallow | **medium** | touches `ggml-alloc.c`, `ggml-backend-meta.cpp`, `common/speculative.cpp`, `src/llama.cpp`, `llama-context.cpp` — but each hunk is small and independent |
| 01 All-reduce | ~4 | low | deep in `ggml-cuda.cu` | **medium** | own allreduce files are safe; the comm-init/dispatch hunks are in a churning region |
| 02 Adaptive MTP | ~19 | medium | shallow | **medium** | touches `common/*`, `llama-context.cpp`, `llama-cparams.h`, docs; userland, low upstream churn |
| 03 Chunked GDN | ~20 | low | medium | **medium** | own `gated_delta_net*` files; recurrent/hybrid plumbing is upstream-active |
| 04 BF16 KV / native BF16 FA | ~8 | high | deep in `fattn*` | **high** | `fattn-tile.cuh`/`fattn-mma-f16.cuh` are heavily reworked upstream |
| 05 FA band | ~7 | high | deep in `fattn*` + `ggml-cuda.cu` | **high** | same files as 04; keep adjacent so one rebase resolves them together |
| 06 Native KV | ~7 | high | deep in `fattn*` | **high** | same files as 04/05 |
| 07 Derived kq-mask | ~8 | high | deep in `fattn*` + graph | **high** | adds op inputs (upstream op-shape churn) |
| 08 MMB | ~10 | medium | medium | **medium** | mostly own `mmb.*`; `ggml-cuda.cu` dispatch churn |
| 09 K-quant boosts | ~8 | high | deep in `mmvq.cu`/`mmq*` | **high** | upstream actively reworks MMQ/MMVQ (e.g. #29953, #28935) |
| 10 Fused prefill | ~20 | high | deep in `ggml-cuda.cu` try_fuse | **high** | many arms in one function |
| 11 Fused MoE | ~15 | high | deep in `ggml-cuda.cu`/`mmq*`/`mmvq*` | **high** | the hardest seam; adjacent to 09 |
| 12 MoE iface/sched/slab | ~20 | high | medium | **high** | `ggml-backend.cpp`/`ggml-backend-meta.cpp`/`ggml-cuda.cu` alloc region are upstream-active |
| 13 Expert-cache engine | ~15 | high | deep | **high** | engine file is ours; the hooks interleave with 10/11 |
| 14 Graph handling | ~4 | medium | deep in `ggml-cuda.cu` | **medium** | small, but in a churning region |
| 15 Host memory reclaim | ~4 | low | shallow | **low** | loader-only; independent |
| 16 qwen4exp model/HC | ~35 | medium | medium | **medium** | many files but mostly qwen4exp/HC-specific |
| 17 QSA/indexer | ~25 | medium | medium | **medium** | own indexer/QSA files; model file shared with 16 |
| 18 getenv sweep | ~40+ | very high | shallow | **high** | by design it touches every cached flag; it is last and droppable |

The concentration is clear: **the flash-attention and quant-GEMM blocks (04–11) carry most of the rebase
cost**, because upstream churns those files. The taxonomy cannot remove that, but it can (a) keep each
concern separate so a rebase can defer one at a time, and (b) make each block's upstream-function hunk as
small and as local as possible — the Phase-2 refactor.

## 4. Hub-file strategy

For each hub file, name the owner and keep later blocks append-only / additive. The full region map is in
`FILE-OWNERSHIP.md`; the strategy:

1. **Own the file where possible.** `mmb.{cu,cuh}` → 08; `gated_delta_net*` → 03;
   `allreduce-hip.*` → 01; `hc-mix.*`/`hyperconn.*` → 16; `indexer-*`/`fattn-qsa.*` → 17;
   `moe-expert-cache.{cu,h}` → 13. These never conflict unless upstream creates the same file.
2. **Split a hub file by region and assign one owner per region.** `ggml-cuda.cu`: comm/dispatch (01),
   FA dispatch (04/05/06/07), MMQ/MMVQ dispatch + J selection (08/09), `try_fuse` arms (10/11), slab/alloc
   (12), cache hooks (13), graph (14), QSA/HC dispatch (16/17). `src/llama-context.cpp`: readback (00),
   MTP (02), GDN (03), FA hint (05), MoE/loader (12), qwen4exp (16/17). Record the region map so the
   ordering rebases survive.
3. **Isolate the churning-function insertion.** The Phase-2 rule: a feature's `ggml_cuda_try_fuse` arm
   should be a *one-line call* to a feature-local matcher function (the current pattern), so the
   upstream-function hunk is contiguous and small. F1 then costs a few lines, not a function rewrite.
4. **Generated instances**: `template-instances/*.cu` and `generate_cu_files.py` travel with the axis
   (G6). Never a separate block.
5. **`tests/test-backend-ops.cpp`** is a permanent hub. Each block appends its own cases at its own
   region; accept the conflict and keep each block's test addition contiguous. Do **not** create a
   trailing "tests" block (it would depend on every feature and would be the worst possible rebase unit).
6. **`common/arg.cpp` / `common/common.*` / `include/llama.h`** flags: each flag lives with its feature
   block; the `18` getenv sweep is the only block allowed to touch all of them, and it is last and
   droppable.

## 5. Phase-2 candidates (optional, needs maintainer go-ahead)

Phase 2 extracts feature logic from actively-churning upstream functions into feature-local code, so the
upstream-file hunk becomes small. Ranked by rebase-pain relief per unit of risk:

| rank | candidate | current home | proposed shape | risk |
|---|---|---|---|---|
| 1 | `ggml_cuda_try_fuse` arms (MoE, prefill fusions, cache hooks) | 10/11/13 | each arm already calls a matcher; ensure **every** arm is one call + a feature-local `_match`/`_op` pair; no inline setup in `try_fuse` | low (mechanical, already the pattern) |
| 2 | MMQ `J` selection / config dispatch | 08/09 | move our config rows into a feature-local config function/table (partly done) | medium (numerics-sensitive; gate it) |
| 3 | `mmvq.cu` dispatch (item-split vs ksplit vs VDR) | 09/11 | feature-local dispatch helper + own kernel file where possible | medium/high (band-uniform purity) |
| 4 | FA `type_KV`/band dispatch in `fattn.cu` | 04/05/06/07 | a feature-local chooser already exists; move the remaining inline branches out | high (numerics/purity; large gates) |
| 5 | `llama-context.cpp` glue (FA hint, readback, MoE) | 00/05/12 | keep each hunk a small standalone block; do not refactor for its own sake | low |
| 6 | `src/llama.cpp` / `llama-model-loader.cpp` loader glue | 12/15 | already mostly local functions | low |

**Recommendation:** do Phase 1 first; then re-measure the rebase cost (the modified-file count per target
block) and only do the Phase-2 items that still dominate. Item 1 is the cheapest and highest-value.

## 6. Would a "feature + integration" block split help?

One tempting end-state is: every feature block is **new files only** (feature-local), and a companion
**integration block** holds all dispatch/registration edits. On a rebase, only the few integration blocks
can conflict; features are free.

- **Pro:** the best possible F1/F2 relief; features become fully droppable (G4).
- **Con:** it breaks the per-block build gate unless the feature block is reachable without the
  integration block; and the integration block depends on all feature blocks, so ordering becomes
  features-then-integration per subsystem. It also splits each concern across two blocks (violates G1).
- **Verdict:** not the default, but keep the *spirit* — one-line arms calling feature-local functions
  (§5.1) gives most of the benefit without a second block per feature. Flagged as Q7.

## 7. What to measure after Phase 1

To turn this from a hypothesis into evidence, re-run the modified-vs-added metric on the **new** block
chain and record it in this file:

```bash
# per target block, count modified upstream files (conflict surface) vs new files
git -C ~/llama.cpp show --name-status --format="" <target-block> | awk '$1!="A"{m++} $1=="A"{a++} END{print a, m}'
```

Success = the maximum modified-file count across blocks drops, and the number of blocks touching each hub
file drops toward 1–2. If a target block still touches >10 shared modified files, it is a Phase-2
candidate or a split candidate.

## 8. Relationship to r50 merge-hygiene

r50's `REDISTRIBUTION.md` is the per-hunk move list for the *existing* 16-block chain; `MIGRATION-MAP.md`
reuses it where the target block coincides with the old one. But r50 optimised for "each of the 16 commits
builds", not for future re-bases: it kept block 06 as a bucket and moved the forward deps to block 15 as a
lateral move, which grew the super-block. The new taxonomy fixes the cause; the r50 moves are a useful
inventory, not the target.
