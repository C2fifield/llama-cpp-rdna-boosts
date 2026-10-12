# HANDOFF — patch-block-reorg, Phase 1 execution (cold start)

**Written 2026-10-12.** This is the cold-start entry point for the next session. Read this first, then the
files in the reading order below. It supersedes the 16/18-block target tables in `README.md` and
`TARGET-BLOCKS.md` (the final approved set is **14 blocks**, below); `PHASE1-STATUS.md` is the dated
findings log.

## TL;DR

The **audit is complete**, the **method is proven**, and **target blocks 00–09 are rebuilt and
committed** on a dedicated branch (`v2-phase1` = `4463f18c7`).  Per-block fresh builds are green
(10/10).  Block 05: FLASH_ATTN_EXT 6389/6389 ×3, prefill-logit 0.000586 / 98.725 %.  Blocks 06–09:
MUL_MAT 2444/2444, width_purity PASS, fusion on == off, 35B-A3B `-ncmoe` cache off == on.  Nothing is
broken or half-applied.

**Remaining:** `10 qwen4exp`, `11 qsa`, `12 mmap-reclaim`.  See `PHASE1-STATUS.md` §9–§14.  `GGML_CUDA_OP_TIMING`
still needs a home decision; the old-15 qwen4exp/HC and cache-blocks `ggml-cuda.cu` arms stripped from
08/09 are re-added with `10 qwen4exp`.

## 1. Ref map (exact)

| ref | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts` (branch `main`, pushed to `origin` = `git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| reference release | `release.json` base `097cf33d9` (tree `30c400ccea2e2fb47b663036a6cee08a4dc542d8`), tip `84058162b` (tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8`) |
| Phase-1 worktree | `~/llama-v2` (git worktree of `~/llama.cpp`), branch **`v2-phase1`** |
| clean checkpoint | tag **`pin/v2-phase1-00-04`** = `2a1502311` (blocks 00–04); **`v2-phase1`** now at `4463f18c7` (blocks 00–09) |
| scratch (partial FA) | branch `v2-fa-wip` = `116ff4ccf` (do **not** use; kept only as a record) |
| scratch (08 build) | branch `v2-prefill-wip` = `5f11da3ce` (the merged-block source; block 08 `a32c95a6e` is the squashed version) |
| other worktrees | `~/llama.cpp` (user's build), `~/llama-rebase` (canonical published chain), `~/llama-src` |

`~/llama-v2` shares the `~/llama.cpp` git object store. The delivery patches (`~/llama-cpp-rdna-boosts/patches`)
are untouched by this work.

## 2. Reading order

1. `README.md` (this campaign's charter — design goals, audit, method, DoD)
2. `REBASE-RESILIENCE.md` (why re-bases hurt; hub-file strategy; Phase-2 candidates)
3. `PHASE1-STATUS.md` (dated findings log — **the decisions and the four structural findings are here**)
4. `HANDOFF.md` (this file — the execution state and next steps)
5. `MIGRATION-MAP.md` + `FILE-OWNERSHIP.md` (old→new hunk map, hub-file regions)
6. `OPEN-QUESTIONS.md` (all Q1–Q8 decisions recorded)
7. `PHASE2-REFACTOR.md` (the self-contained-extraction work plan)
8. `tools/select_hunks.py`, `tools/hunk_summary.py` (the redistribution tooling)
9. `AGENTS.md` ("Report findings before shipping", "WIP and promotion rules", "Critical facts"),
   `archive/work/r50-merge-hygiene/REDISTRIBUTION.md` (per-hunk move inventory)

## 3. What is done

Target blocks **00–05** are committed on `v2-phase1`, each rebuilt from base by replaying the union of
its source hunks with `git apply -3` (zero conflicts; file/line counts match the source blocks):

| target | alias | source | files/lines |
|---|---|---|---|
| 00 | foundation | old 00+05+07+09a+06-readback+15-W4 | 9 files |
| 01 | allreduce | old 12 | 4 files |
| 02 | mtp-adaptive | old 01 | 19 files |
| 03 | gdn | old 02 | 22 files, +3141/−15 |
| 04 | bf16-kv | old 03 | 8 files, +630/−214 |
| 05 | fa | old 04-FA + 08-FA + 15-FA, minus the arena | 24 files + 286 instances, +3082/−313 |
| 06 | mmb | old 08-MMB half | `mmb.{cu,cuh}` + 3 files, +2603/−3 |
| 07 | kquant | old 04-MMQ + old 10 + old 15 Q1_0/grid/mmvf | 9 files, +725/−119 |
| 08 | fused-kernels | old 08 fusions + old 13 + old 15 PR57/63 (cache/qwen4exp arms stripped) | 27 files + 6 instances, +4774/−420 |
| 09 | slab-moe | old 06 + old 09b + old 11 + old 15 slab/cache extensions | 55 files, +11553/−221 |

**Tooling proven:** `git show <src> --format= -- <paths> | git apply -3` for whole-file moves, and
`tools/select_hunks.py '<regex>'` to lift hunks by content. See `PHASE1-STATUS.md` §2.

**Correction recorded:** `cpy-batch.*` is a block-15 file (cache-blocks feature), not a target-00 idiom.

## 4. Final approved target — 14 blocks

Approved by the maintainer (see `PHASE1-STATUS.md` §8). The two big decisions: **qwen4exp is independent
but host MoE is the broad enabler, so Slab+MoE is grouped and placed ahead of qwen4exp**; **FA ships
without the `fattn_stage` arena** (the arena is a memory optimisation and lives in `slab-moe`).

| # | alias | contents | old sources |
|---|---|---|---|
| 00 | `foundation` | band-uniformity, Meta-wrapper, allocator W4, logits readback | 00,05,07,09a,06-readback,15-W4 |
| 01 | `allreduce` | hybrid HIP all-reduce | 12 |
| 02 | `mtp-adaptive` | adaptive MTP | 01 |
| 03 | `gdn` | chunked gated-delta-net | 02 |
| 04 | `bf16-kv` | BF16 KV cache + native-BF16 FA | 03 |
| 05 | `fa` | band + native-KV + derived-mask (**no arena**) | 04-FA, 15-FA |
| 06 | `mmb` | MMB quantized prefill GEMM | 08-MMB |
| 07 | `kquant` | mmvq VDR, Q6_K MMQ, Q1_0, grid pad | 10, 04-MMQ, 15-Q1_0/grid/PR57 |
| 08 | `fused-prefill` | conv1d/RMS/top-k MoE/verify fusions | 08-fusions, 15-PR63 |
| 09 | `fused-moe` | gate+up+GLU MMQ, item-split, shared-expert, MWR | 13, 08-mmID, 15-PR114 |
| 10 | `slab-moe` | host-resident MoE (iface+scheduler+Meta+slab+cache engine+hooks) + `fattn_stage` arena + graph-offload | 06-MoE, 09b, 11, 13-engine, 15-hooks/arena/r48, 06-async |
| 11 | `qwen4exp` | model/HC/lazy/MTP head + qwen4exp policy | 14-model/HC, 15-HC16/policy |
| 12 | `qsa` | QSA/indexer + QSA memory wins | 14-QSA, 15-W1–W3 |
| 13 | `mmap-reclaim` | host memory reclaim | 06/15-r49 |

Dependency notes: 05 needs no slab (arena deferred to 10); 09 precedes 10 so the cache hooks can
reference the fusions; 10 precedes 11 (qwen4exp consumes it); 12 after 11; 13 last. The getenv caching is
folded into each flag's owning block. Generated `template-instances/*` travel with the type axis that
dispatches to them.

## 5. `05 fa` – `09 slab-moe` DONE; next: `10 qwen4exp`

> **Status 2026-10-12 (session 2):** blocks 05–09 are built and committed; `v2-phase1` = `4463f18c7`
> (clean, 10/10).  **Next task:** `10 qwen4exp` (`src/models/qwen4exp.cpp`, the lazy reader, the HC
> machinery `hc-mix`/`hyperconn`/`dsv4-hc`, the HC matchers, the qwen4exp prefill policy) — this is also
> where the qwen4exp/HC and cache-blocks `ggml-cuda.cu` arms stripped from 08/09 are re-added.

Worktree `~/llama-v2`, branch `v2-phase1` at `2a1502311`.

1. **old-04 FA** (excluding `mmq.cu`, `mmq-vec-dot.cuh`):
   ```
   git show a7204fa20 --format= -- ggml/include/ggml.h ggml/src/ggml.c \
     ggml/src/ggml-cuda/fattn-mma-f16.cuh ggml/src/ggml-cuda/fattn.cu \
     ggml/src/ggml-cuda/ggml-cuda.cu src/llama-context.cpp | git apply -3
   ```
2. **old-15 FA core** (exclude `fattn-qsa*`):
   ```
   git show 84058162b --format= -- \
     ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/fattn-mma-f16.cuh \
     ggml/src/ggml-cuda/fattn-tile.cu ggml/src/ggml-cuda/fattn-tile.cuh \
     ggml/src/ggml-cuda/fattn-vec.cuh ggml/src/ggml-cuda/fattn.cu \
     'ggml/src/ggml-cuda/template-instances/fattn-tile-*' \
     'ggml/src/ggml-cuda/template-instances/fattn-mma-*' \
     'ggml/src/ggml-cuda/template-instances/fattn-vec-*' | git apply -3
   ```
   Two conflicts resolve by keeping **theirs** (the added band/verify blocks): `fattn.cu` and
   `fattn-tile.cuh` — delete the `<<<<<<< ours` / `=======` / `>>>>>>> theirs` marker lines (ours is
   empty there). See `v2-fa-wip` (`116ff4ccf`) for the exact state.
3. **Derived API:** `ggml.c` hunk applies cleanly (`select_hunks.py 'ggml_flash_attn_ext_add_kq_derived'`);
   for `ggml.h`, apply the same hunk but the trailing context includes QSA lines not present at 05 — either
   resolve by hand (insert the derived block after `ggml_get_fa_tensor_parallel`) as done on `v2-fa-wip`.
4. **Shared FA hunks from old-15** (`ggml-cuda.cu` FA dispatch, `common.cuh` band/native, `llama-graph.*`,
   `src/llama-context.cpp`, `tests/test-backend-ops.cpp`): select by content, but **do not rely on a bare
   `band` keyword** — it matches the GLU→Q8_1 verify-band fusion (`fused-moe`). Use `band_wmma`,
   `fattn_band`, `kv_native`, `fattn_dequantize`, `kq_derived`, `cell_pos`/`tok_lo`/`tok_hi`,
   `flash_attn_ext_add_kq_derived`.
5. **Exclude the arena:** do not take `common.cuh`'s `fattn_stage` fields / `fattn_stage_try_get` /
   `ggml_cuda_fattn_stage_max_bytes`, and reconstruct the pre-arena staging in
   `fattn-common.cuh`'s `launch_fattn` (the pre-amendment implementation is in the block-14/15 history;
   it staged into the node's reserved scratch). Omit the `ggml-cuda.cu` destructor free + the
   `fattn_stage_bound`/`_used` accounting. The arena belongs to `10 slab-moe`.
6. Commit `05 fa`; then **fresh-build gate** (see §6) and the FA gates (see §7).

## 6. Build / apply commands

```bash
# build the Phase-1 worktree (ROCm 7.14, gfx1201); ccache is on
cd ~/llama-v2 && BUILD_DIR=build-v2 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714

# fresh per-block `all` sweep for the new chain (NOT a reused build dir)
~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh \
    --repo ~/llama-v2 --base 097cf33d9 --tip "$(git -C ~/llama-v2 rev-parse v2-phase1)" \
    --build-dir build-per-block
```

`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is required with CMake >= 4.3. See `AGENTS.md` → "Build the fork".

## 7. Gates (run per owning block)

- **FA (05):** `test-backend-ops -o FLASH_ATTN_EXT` (full matrix incl. GQA band, native KV types, mixed
  pairs, derived/mask-hole); `scripts/gate-prefill-logits.sh` (KLD ≤ 0.005, same-top-p ≥ 98 %);
  width purity `W=1..8` / `plain == draft-mtp n_max ≤ 7`; same-seed coherence.
- **MMB (06):** `test-logits-width-probe`; MUL_MAT family; prefill perf.
- **kquant (07):** `llama-batched-bench -npl 1,4,8`; band-uniform nwarps/VDR (`GREEDY-PURITY.md` §§19/24/25).
- **fusions (08/09):** fusion on/off same-seed; MTP gate; the relevant `GGML_CUDA_DISABLE_*` switches.
- **slab-moe (10):** `scripts/gate-qwen4exp-quant-coherence.sh`; `MUL_MAT_ID`; byte-identity oracles;
  `MOE_EXPERT_CACHE_MIB` auto; `GGML_CUDA_SLAB=0`; `--fit off`; `-sm tensor` + `-ncmoe` coherence.
- **qwen4exp (11):** qwen4exp coherence (dense+sparse arms); `HC_MIX`; `test-llama-archs`;
  `test-lazy-reader`; MTP-export width probe.
- **qsa (12):** `FLASH_ATTN_QSA`, `LIGHTNING_INDEXER`, `TOPK_QSA`; `LLAMA_QSA_OFF=1` plain==draft-mtp.
- **reclaim (13):** peak VmRSS; same-seed + `-sm tensor` coherence; `--load-mode none` unaffected.

Release gates (must stay green): prefill-logit `0.000707 / 98.755 %`, qwen4exp quant coherence, same-seed,
35B-A3B draft-mtp.

## 8. Method and pitfalls (learned this session)

- **Move hunks with `git apply -3`.** `git show <src> --format= -- <paths>` for whole-file moves;
  `tools/select_hunks.py '<regex>'` for hunk subsets. `git show` patch `index` lines give the 3-way merge
  its blobs, so context drift is handled.
- **Never `--autosquash`.** A fixup's base is the tip, so it conflicts/double-applies (r50 lesson). Edit
  at `git rebase -i` `edit` stops, or reconstruct.
- **Prefer reconstruction from base** for reorders; reorder-via-rebase conflicts on shared files.
- **Keyword triage is unreliable** on the squashed old-15/old-06: hunks mix concerns (`band` matched
  the GLU→Q8_1 fusion). Verify every selected hunk by reading it.
- **Per-block build must be fresh** (`rm -rf` + configure + `all`); a reused cache hides errors.
- **Phase 1 is tree-invariant where practical;** each content exception is a documented Phase-2 item (Q3).
- **The hub files** (`ggml-cuda.cu`, `common.cuh`, the FA kernels, `ggml-alloc.c`, `llama-context.cpp`)
  are where the work is; `FILE-OWNERSHIP.md` has a region map.
- **Report findings and wait** before delivery-affecting commits (AGENTS.md).

## 9. Definition of done (V2.0)

- 14 blocks; every block builds fresh (`gate-per-block-build.sh` 14/14).
- `patches/` regenerated via `scripts/make-patches.sh` from a canonical chain rebuilt at
  `release.json.base`; `scripts/validate-set.sh` green; new `release.json` via `scripts/make-release.sh`.
- All release gates green on the final tree.
- Current-state docs re-issued for V2.0 (`README.md`, `AGENTS.md` Critical facts, `ENVIRONMENT.md`,
  `patches/README.md` top, `TODO.md`, `release.json`); dated records keep historical block numbers; the
  old→new map published (`PHASE1-STATUS.md`).
- Fork `rdna-boosts` refreshed + container images per the Pushing policy (only on release).

## 10. Key records

- Findings/decisions: `PHASE1-STATUS.md`, `OPEN-QUESTIONS.md`.
- Per-hunk move inventory: `archive/work/r50-merge-hygiene/REDISTRIBUTION.md`.
- Incremental FA patches (older bases, for reference): `archive/work/rdna4-fa-band/`,
  `archive/work/rdna4-fa-band-wide/`, `archive/work/arch-independent-memory/patches/0006-0007`,
  `archive/work/kq-derived-tile/`, `archive/work/kq-mask-derived-ab/`,
  `archive/work/block-15-campaign-wins/block-15-campaign-wins.patch`.
- Tooling: `tools/select_hunks.py`, `tools/hunk_summary.py`.
