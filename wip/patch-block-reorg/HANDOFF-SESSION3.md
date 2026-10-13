# HANDOFF — patch-block-reorg, session 3 (cold start)

**Written 2026-10-12 (session 2 end).** This is the cold-start entry point. Read `HANDOFF.md`
(design + the original 05 execution steps), then `PHASE1-STATUS.md` §9–§14 (the dated findings for
blocks 05–09), then this file. `AGENTS.md` ("Report findings before shipping", "WIP and promotion
rules", "Critical facts") still applies.

## TL;DR

Blocks **00–09 are rebuilt, committed and gated** on `v2-phase1` (**10/10** fresh per-block builds).
Block **10 `qwen4exp` is WIP on scratch branch `v2-qwen4exp-wip`** (`0d3e20323`): it **builds** and
`test-lazy-reader` passes, but `test-llama-archs` hits a **GPU memory fault to diagnose**. The delivery
`patches/`/`release.json` are untouched.

The maintainer decided (this session) that the planned 08+09 are **one `fused-kernels` block** (see
§13), and observed that splitting `ggml-cuda.cu` into multiple files is the right Phase-2 follow-up.

## 1. Ref map (exact)

| ref | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts` (branch `main`, **pushed**) |
| Phase-1 worktree | `~/llama-v2` (worktree of `~/llama.cpp`), branch **`v2-phase1`** = **`0a5d26280`** (blocks 00–09, clean) |
| clean checkpoint tag | `pin/v2-phase1-00-04` = `2a1502311` |
| block-10 WIP | branch **`v2-qwen4exp-wip`** = **`0d3e20323`** (do **not** use for anything else) |
| block-08 source | branch `v2-prefill-wip` = `5f11da3ce` (squashed into block 08 `5ae8cb848`) |
| reference release | `release.json` base `097cf33d9`, tip `84058162b` (tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8`) |
| canonical chains | `~/llama.cpp` (rdna-boosts), `~/llama-rebase` |
| r49 known-good build | `~/llama.cpp/build-rocm-hybrid` (tip `f0a74f4cc`) |
| gate build dir | `~/llama-v2/build-per-block` |

Block map so far: 00 foundation, 01 allreduce, 02 mtp-adaptive, 03 gdn, 04 bf16-kv, 05 fa, 06 mmb,
07 kquant, **08 fused-kernels** (merged old 08+09), **09 slab-moe** (old 06+09b+11+15), then
10 qwen4exp, 11 qsa, 12 mmap-reclaim.

## 2. State of blocks 00–09 (committed, gated)

| block | commit | files | gates run |
|---|---|---|---|
| 00 | `4143c62af` | 9 | per-block build |
| 01 | `320fecf0d` | 4 | per-block build |
| 02 | `8401ee71f` | 19 | per-block build |
| 03 | `8f244219d` | 22 | per-block build |
| 04 | `2a1502311` | 8 | per-block build |
| 05 fa | `05e88da52` | 24 + 286 instances | `FLASH_ATTN_EXT` 6389/6389 ×3 arms; prefill-logit **0.000586 / 98.725 %**; same-seed == r49 |
| 06 mmb | `298060c73` | `mmb.{cu,cuh}` +3 | `MUL_MAT` 2444/2444; MMB prefill-only |
| 07 kquant | `95d15bad9` | 9 | `MUL_MAT` 2444/2444 |
| 08 fused-kernels | `5ae8cb848` | 27 + 6 | fusion on == off; `width_purity` PASS; MUL_MAT 2444/2444; prefill-logit PASS |
| 09 slab-moe | `0a5d26280` | 55 | 35B-A3B `-ncmoe 40` cache off == on; prefill-logit PASS |

**Per-block fresh `all` build gate:** `scripts/gate-per-block-build.sh --repo ~/llama-v2 --base
097cf33d9 --tip "$(git -C ~/llama-v2 rev-parse v2-phase1)" --build-dir build-per-block` → **10/10**.

Known pre-existing test failures (present since 06, fixed by a later/upstream block — NOT caused by
the delivery): `test-backend-ops -o MUL_MAT_ID` has 4 `f16 / dup=1` failures + an
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (r49 passes 931/931).

## 3. Block 10 `qwen4exp` — WIP, one fault to diagnose

Branch `v2-qwen4exp-wip` = `0d3e20323`. Contents: **old-14 whole** (qwen4exp model, lazy reader, HC
machinery `hc-mix`/`hyperconn`/`dsv4-hc`, HC ops in `ggml.h`/`ggml.c`, `llama-memory-hybrid-idx`,
`llama-arch`, the old-14 QSA/indexer files — see the note below) **+ old-15's** `hc-mix`/`hyperconn`/
`dsv4-hc`, `mmb.{cu,cuh}` extensions, the `norm.cu` MMB bf16-copy hook, the `meta` `DSV4_HC_*`/
`LIGHTNING_INDEXER`/`HC_MIX` cases, and the `HC_COMBINE→HC_MIX` + `cpy_batch` `try_fuse` arms.

- **Builds** (`all`), `test-lazy-reader` **PASSES**.
- **`test-llama-archs` faults** in
  `mul_mat_vec_f<float,float,1,64,false,true>` (i.e. `is_multi_token_id=true, ncols_dst=1,
  block=64, no fusion`) on the qwen4exp/Meta rows — a GPU memory access fault. It **also faults with
  `GGML_CUDA_DISABLE_FUSION=1`**, so it is in the mmvf kernel/launch, not the fusion arms.
- The **HC/indexer fusion matchers** (`ggml_cuda_match_hc_mix`, `hc_combine_norm`, `idx_relu_sum`,
  `sigmoid_mul_add`) are **NOT applied** — they are interleaved with old-15 definitions that the
  keyword triage misses (recorded as a Phase-2 item).

**First thing to do next session (diagnose the fault):**
1. Build `0a5d26280` (block 09) into `build-per-block` and run `test-llama-archs` — does the fault
   predate block 10? (The qwen4exp rows will be absent at 09, but the fault may be arch-independent.)
2. If it predates 10, bisect block 07's `mmvf.cu` (the old-15 `nrows` guard + `ggml_cuda_grid_stall_pad`
   launch) and block 08's `mmvf.cu` `batched_reduce` — those are the two changes to that kernel.
3. If it is block-10-caused, check the qwen4exp HC/`MUL_MAT_ID` launch geometry against the kernel's
   `nrows` argument.

> **DIAGNOSED (session 3): see `PHASE1-STATUS.md` §15.**  The fault **predates block 10** but is **not
a reorg/block-10 artifact and not even a v2 artifact**: with a fixed seed (`-s 42`, `-a deepseek2`)
the upstream base `097cf33d9` **passes 7/7**, v2 block 08 `5ae8cb848` **passes**, v2 block 09
`0a5d26280` **faults**, and the untouched **published reference tip `84058162b` also faults** (identical
`mul_mat_vec_f<float,float,1,64,false,true>` signature).  It is a **pre-existing delivery defect inside
the block-09 `slab-moe` concern** (host-expert pinned loader / Meta / scheduler / cache hooks) on the
upstream `--moe-cache-mib` engine path; the grid-pad and `batched_reduce` hypotheses are excluded
(`GGML_CUDA_GRID_STALL_PAD=0` still faults).  It does not change the block plan — the maintainer's
decision on how/when to fix it is pending (options in §15).  Do **not** block `10 qwen4exp` on it.

> **BLOCK 09 FIXED (session 3, maintainer decision):** §17's dropped base content (the four
`GGML_OP_LIGHTNING_INDEXER` / `GGML_OP_DSV4_HC_*` Meta split-state cases) is repaired; block 09 was
amended to **`0a5d26280`** (see `PHASE1-STATUS.md` §18).  Per-block build gate **10/10**; base's
qwen4exp model now loads+generates.  The §17.3 blocks-00–09 dropped-content sweep is **still open**.

> **BLOCK 10 PROGRESS (session 3): see `PHASE1-STATUS.md` §16 + §19.**  The maintainer chose "move
old-14's QSA/indexer to `11 qsa`"; scoping it showed the WIP has **no QSA dispatch at all** and that the
QSA graph is a Phase-2-class rewrite of `qwen4exp.cpp` (§16).  The pragmatic boundary adopted: block 10 =
WIP minus the six QSA/indexer `.cu/.cuh` files; `11 qsa` = those + the CUDA dispatch (to be written) +
old-15 W1–W3.  On the amended block 09 the WIP still **aborted on load** — the missing
`GGML_OP_HC_MIX`/`HC_COMBINE` CUDA dispatch (old-15 content); adding it makes the model load+generate,
but the **dense arm still diverges from r50** ("question" vs "query"), so block 10 is **not finished**
(§19).  Scratch: `v2-block10` (rebased onto `0a5d26280`).

**Note on old-14's QSA/indexer files** (`fattn-qsa.cu`, `indexer-*.cu`, `mmvf.cu` additions,
`mmq.cu`/`mmq.cuh`): they are currently inside the block-10 WIP. The plan puts QSA/indexer in `11 qsa`.
**DECIDED (maintainer, session 3): move them to `11 qsa`.**  Because the QSA graph is a rewrite of
upstream's QSA inside `qwen4exp.cpp` (not separable by file), the interim boundary is: block 10 = the
WIP **minus** the six QSA/indexer kernel files; `11 qsa` = those **plus the CUDA dispatch that must be
written** (the WIP has none) plus old-15 W1–W3.  See `PHASE1-STATUS.md` §16 for the full analysis and the
plan-exact alternative.

## 4. Deferred / stripped items (must be re-added or given a home)

- **`GGML_CUDA_OP_TIMING`** (block 04's `ggml-cuda.cu`) is a generic diagnostic currently living in
  `05 fa`. Decision pending: leave it or move to `00 foundation`.
- The `ggml-cuda.cu` **HC/indexer fusion matchers** and the **`cuda_ctx->bf16_src1_cache`/`sigmoid_mul_add`
  /`idx_relu_sum` args** (old-15) are not in blocks 08–10.
- old-15's **cache/qwen4exp `try_fuse` arms** are partially re-added (the cache ones in 09, the HC/
  cpy_batch ones in 10).
- `ggml-cuda.cu` **R1 `lf_nval` reduction**: applied in 08; the `topk_moe_args` (old-15) fields are
  not.
- The maintainer's **`ggml-cuda.cu` split** observation: the Phase-2 files should be
  `ggml-cuda-try-fuse.cu`, `ggml-cuda-slab.cu`, `ggml-cuda-moe-cache-hooks.cu`, `ggml-cuda-arena.cu`.

## 5. Remaining work

- **10 qwen4exp** — finish from `v2-qwen4exp-wip`: fix the `test-llama-archs` fault, run
  `test-llama-archs`/`test-lazy-reader`/`test-backend-ops HC_MIX`/qwen4exp coherence, decide the QSA
  split with the maintainer, then commit on `v2-phase1` and run the per-block gate.
- **11 qsa** — old-15 W1–W3 + `qsa3` + the `LLAMA_QSA_SPARSE_FA=0` shadowing fix (and old-14's
  QSA/indexer if not already in 10).
- **12 mmap-reclaim** — old-06/15's `LLAMA_LOAD_RELEASE_MMAP` (loader `munmap` after upload).

Then: regenerate `patches/` with `scripts/make-patches.sh`, `scripts/make-release.sh`,
`scripts/validate-set.sh`, run the full release gates, refresh the fork branch + container images per
the Pushing policy.

## 6. Method / pitfalls (learned this session)

- **Run builds in the FOREGROUND** with a generous `timeout`, then grep the log. Do not background
  and sleep — ccache makes them finish in under a minute once warm; the background/sleep pattern just
  wastes wall-clock. (Maintainer asked for this explicitly.)
- **`git show <src> --format="" -- . ':(exclude)<path>' | git apply -3`** for whole-block moves;
  `tools/select_hunks.py '<regex>'` for hunk subsets. **Keyword triage is unreliable** on the squashed
  old-06/13/15: a hunk that mentions one symbol routinely also carries a later-block definition, which
  is why blocks 08/09/10 needed manual stripping. **Read every selected hunk.**
- **`git checkout --theirs -- <file>`** in a `-3` conflict takes the WHOLE stage-3 file (the later
  block's version), not just the conflicting region — that is how `mmvq.cu`/`mmq.cuh` picked up
  cache/qwen4exp code. Prefer manual per-conflict resolution.
- **The interleaving is real and deep.** 07↔09 (mmvq), 08↔09 (mmvq/norm), 09↔10 (cache/qwen4exp),
  10↔11 (old-14 QSA). The honest unit for the fusions was `08 fused-kernels`; the same reasoning may
  apply to 10+11. **Surface it and ask the maintainer before merging.**
- **Per-block gate is the build gate only.** The correctness gates are per-region and must be run
  manually (`test-backend-ops` subsets, `test-logits-width-probe`, `gate-prefill-logits.sh`, the
  fusion on/off coherence, the 35B-A3B `-ncmoe` cache coherence). All pass for 00–09.
- **`llama-cli` needs `--single-turn`.** Wrapping a model run: `timeout … ./bin/llama-cli … | sed
  's/\x1b\[[0-9;]*m//g' | sed -n '/The capital of France is/,$p' | tail -n +2 | grep -vE
  'Prompt:|Generation:|Exiting|^\s*$'`.
- **`test-llama-archs` is the block-10+ gate that caught the fault** — run it early on any qwen4exp/
  HC change.
- Commit campaign docs to `~/llama-cpp-rdna-boosts` (`main`, pushed) as you go; `~/llama-v2` is the
  delivery build worktree.

## 7. The exact next steps (copy-paste)

```bash
cd ~/llama-v2
git log -1 --format='%h %s' v2-phase1          # 0a5d26280 (blocks 00-09, clean)
git checkout v2-qwen4exp-wip                   # the block-10 WIP (0d3e20323)
# build is already in ~/llama-v2/build-per-block for this branch
cd build-per-block && timeout 400 ./bin/test-llama-archs 2>&1 | tail -3   # reproduces the fault

# 1) does the fault predate block 10?
cd ~/llama-v2 && git worktree add /tmp/v2-b09 0a5d26280   # or: git checkout 0a5d26280
# build ggml-hip + test-llama-archs there and run it
```
