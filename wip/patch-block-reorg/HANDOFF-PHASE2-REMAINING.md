# HANDOFF — patch-block-reorg Phase 2, remaining work (fresh-session entry point)

**Written 2026-10-12, end of the Phase-2 session that completed the `ggml_cuda_try_fuse` extraction.**
Read this first, then `PHASE2-STATUS.md` (the dated Phase-2 log — it has the per-cluster gate tables),
`PHASE2-REFACTOR.md` (the plan), `HANDOFF-PHASE2.md` (the Phase-2 cold start), `REBASE-RESILIENCE.md`,
`FILE-OWNERSHIP.md`, and `AGENTS.md` ("Report findings before shipping", "WIP and promotion rules",
"Critical facts").

## 0. TL;DR

- **Phase 1 is complete** at tag `v2-phase1` = `a40841e1e` (12 blocks 00–11; tree byte-identical to the
  published delivery `af0595360`).
- **Phase 2 is in progress** on branch **`v2-phase2`** = **`18a2fd0f8`** (a re-cut of Phase 1 with the
  Phase-2 refactors folded into the owning blocks).
- **Done:** the whole `ggml_cuda_try_fuse` extraction (work item 1 — `try_fuse` 2244 → **1064** lines,
  the rest being upstream arms); work item 2 (MMQ `J` selection was already self-contained); work item 3
  (steps 1–2 the RDNA `nwarps`/rpb policy; **step 3 the MoE kernel file split** `mmvq-moe.cu` +
  `mmvq-vecdot.cuh`, re-homed in block 10).
- **Remaining:** item 4 (FA chooser), item 5 (context/loader glue), item 6 (the maintainer's
  `ggml-cuda.cu` split), then the **V2.0 release cut**.
- **The delivery `patches/` and `release.json` are untouched**; nothing has been pushed to the fork.
- Checkpoints 1–8 exist as tags + bundles (see §2). `v2-phase1` is the safe rollback.

## 1. What this work is

V2.0 = Phase 1 (block-boundary re-org, done) **+ Phase 2 (self-contained refactor, in progress)**. The
goal (G8/Q7) is to move each delivery concern into its own function/file so a future upstream re-base
collides with a small call point, not a rewritten shared function. The authoritative work list is
`PHASE2-REFACTOR.md`; the dated execution record is `PHASE2-STATUS.md`.

## 2. Refs and checkpoints (exact)

| ref | value |
|---|---|
| Phase-1 checkpoint | `v2-phase1` = `a40841e1e` (tree == `af0595360`, the delivery) |
| **Phase-2 branch** | **`v2-phase2` = `18a2fd0f8`** |
| block 08 (amended) | `fcce5864e` |
| block 09 (amended) | `96a7d0d16` |
| block 10 (amended) | `fdd20f2c6` |
| block 11 (amended) | `18a2fd0f8` |
| base | `097cf33d9` (tree `30c400cc…`) |
| delivery tip (r50) | `af0595360` / `84058162b` (tree `f102245d…`) |
| reference build r50 | `~/llama-rebase/build-r50` |
| upstream-base build | `~/llama-base/build-base` |
| Phase-1 worktree | `~/llama-v2` (worktree of `~/llama.cpp`) |
| delivery repo | `~/llama-cpp-rdna-boosts` (`main`, pushed) |

Checkpoints (annotated tags in `~/llama.cpp`) — each also has a bundle in
`~/llama-cpp-rdna-boosts/wip/patch-block-reorg/checkpoints/`:

| tag | tip | contents |
|---|---|---|
| `pin/v2-phase2-ckpt1` | `d8ad56737` | 4 block-10 try_fuse extractions |
| `pin/v2-phase2-ckpt2` | `d86bcf088` | + conv1d/narrow-norm, cpy/gdn/ssm/norm arms, MWR |
| `pin/v2-phase2-ckpt3` | `463c1d598` | + weighted-down, mm_q pair |
| `pin/v2-phase2-ckpt4` | `9c1ec61d6` | + all remaining block-10 arms |
| `pin/v2-phase2-ckpt5` | `8326733e1` | + block-08/09 clean arms |
| `pin/v2-phase2-ckpt6` | `cf9c2e474` | + glu_q8_1/shared_expert; try_fuse COMPLETE |
| `pin/v2-phase2-ckpt7` | `24ef342b3` | item 3 s1: `calc_nwarps` → `mmvq-rdna.cuh` |
| `pin/v2-phase2-ckpt8` | `9d6730dc8` | item 3 s2: launch-param helpers → `mmvq-rdna-params.cuh` |
| **`pin/v2-phase2-ckpt9`** | **`18a2fd0f8`** | **item 3 s3: MoE kernel → `mmvq-moe.cu` + shared selectors → `mmvq-vecdot.cuh`** |

## 3. Done (do not redo)

### 3.1 Work item 1 — `ggml_cuda_try_fuse` extraction: **COMPLETE**

`ggml_cuda_try_fuse` is an ordered list of guarded call points (1064 lines; the remainder are upstream's
own arms). Every delivery arm now lives in its concern file and was re-homed into the block that owns it.
New/updated homes:

| concern | file | functions |
|---|---|---|
| HC combine/mix/gate-mix | `hyperconn.cu` | `ggml_cuda_hc_combine_mix_try_fuse`, `ggml_cuda_hc_mix_try_fuse`, `ggml_cuda_hc_combine_norm_try_fuse`, `ggml_cuda_hc_gate_mix_try_fuse`, `ggml_cuda_hc_combine_norm_identify/set_bf16*` |
| indexer relu sum | `indexer-score.cu` | `ggml_cuda_indexer_relu_sum_try_fuse`, `ggml_cuda_match_idx_relu_sum` |
| sigmoid·mul·add, conv, gate | `unary.cu` | `ggml_cuda_sigmoid_mul_add_try_fuse`, `ggml_cuda_add_softplus_mul_try_fuse`, `ggml_cuda_scale_unary_try_fuse` |
| MMB / MWR | `moe-weighted-reduction.cu` | `ggml_cuda_moe_weighted_reduction_try_fuse` |
| mmvq side | `mmvq.cu` | `mul_mat_id_weighted_down`, `ssm_conv_in`, `ssm_prescan`, `dual_output`, `moe_down_fold`, `gate_beta_verify`, `shexp_down_gate`, `glu_q8_1_mark`, `match_shared_expert[_try_fuse]`, `should_fuse_mul_mat_vec_q` |
| mmq side | `mmq.cu` | `mul_mat_q_pair_try_fuse`, `mul_mat_q_swiglu[_dense]_try_fuse`, `mul_mat_takes_mmq` (exposed) |
| conv1d / norm | `ple-conv.cu`, `gdn-conv.cu`, `norm-gated.cu`, `norm.cu` | `ple_conv_try_fuse`, `gdn_conv_try_fuse`, `norm_rows_try_fuse`, `l2_norm_pair_try_fuse`, `add_rms_q8_1_try_fuse`, `norm_q8_1_try_fuse` |
| gdn/ssm cpy | `gated_delta_net.cu`, `ssm-scan.cu` | `gdn_beta_sigmoid_try_fuse`, `gdn_cache_cpy_try_fuse`, `ssm_scan_cache_cpy_try_fuse` |
| cpy batch | `cpy-batch.cu` | `cpy_batch_try_fuse` |

Helper hoists made for these (exposed via `common.cuh` / concern headers):
`ggml_cuda_compute_forward`, `ggml_cuda_check_fusion_memory_ranges`, `ggml_cuda_mul_mat_takes_mmq`,
`ggml_cuda_should_fuse_mul_mat`, `ggml_cuda_should_fuse_mul_mat_vec_q`, `ggml_cuda_is_view_or_noop`
(inline in `common.cuh`), `ggml_cuda_bf16_src1_cache_on` (inline in `common.cuh`),
`ggml_cuda_try_gdn_cache_fusion` / `ggml_cuda_try_ssm_scan_cache_fusion` (→ their files),
`ggml_cuda_match_shared_expert` (→ `mmvq.cu`).

**Deliberately skipped:** `topk_moe` — the matcher and the `//topk-moe` arm are **upstream-unchanged**;
the only delivery edit is a two-line `!ggml_cuda_topk_moe_fusion_disabled()` guard, so moving upstream
code would not help.

### 3.2 Work item 2 — MMQ `J` selection: already done

`ggml_cuda_mmq_select_J` (`mmq.cu`) owns the RDNA3_5 `J` selection and the `GGML_CUDA_MMQ_J_MAX` cap
(3 call sites); the config rows live in `mmq-config-rdna3-5.cuh`. No action.

### 3.3 Work item 3 — `mmvq.cu` dispatch: steps 1–2 done

- `mmvq-rdna.cuh`: `mmvq_rdna_nwarps()` — the band-uniform policy (RDNA4=1; RDNA3_0=8 for
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL else 1; RDNA3_5=2 for Q8_0 else 1; 0 otherwise). Included in
  `mmvq.cu` **after** `enum mmvq_parameter_table_id` and **before** `calc_nwarps`; `calc_nwarps` keeps
  upstream's bodies and adds:
  ```c
  const int rdna_nwarps = mmvq_rdna_nwarps(type, ncols_dst, table_id);
  if (rdna_nwarps > 0) { return rdna_nwarps; }
  ```
- `mmvq-rdna-params.cuh`: `calc_nwarps_weight`, `calc_rows_per_block_override`,
  `calc_rows_per_block_weight`, `mmvq_decode_grid_loop_type`, the `GGML_MMVQ_RDNA4_*` macros. Included
  in `mmvq.cu` **after** `calc_rows_per_block` (they call it). `calc_rows_per_block` is
  upstream-unchanged and stays in `mmvq.cu`.

## 4. Remaining work, in recommended order

> **Everything below is behaviour-preserving refactor unless stated.** Each step: make the change in
> `~/llama-v2` on `v2-phase2`, build, gate, then re-home into the owning block by
> `git apply -3` + `commit --amend` + `git rebase --onto` (§5). Run the mandatory full per-block build
> gate after each re-home (§6).

### 4.1 Item 3 step 3 — MoE kernel file split — **DONE (ckpt9)**

`mul_mat_vec_q_moe` + `mul_mat_vec_q_moe_launch` now live in `mmvq-moe.cu` (with `mmvq-moe.cuh`
declaring the launch template and **explicit instantiation for the 23 dispatch types**); the shared
selectors (`vec_dot_q_cuda_t`, the `GGML_CUDA_MMVQ_MOE_DENSE` guard, `get_vec_dot_q_cuda`,
`get_vdr_mmvq`) moved to `mmvq-vecdot.cuh`.  `mmvq.cu` 3745 → 3368 lines.  Re-homed in **block 10**
(the last editor of the launch) — the tip diff applies cleanly there and only block 11 replays (it does
not touch `mmvq*`).  Gates: 4B/qwen4exp dense+sparse/35B identical to r50, width purity PASS, prefill-logit
0.000707 / 98.755 %, per-block `all` 12/12.  Detail in `PHASE2-STATUS.md` §9.

### 4.2 Item 4 — FA `type_KV` / band / native chooser → `fattn-rdna.cuh`

Current interleave (all in `fattn.cu` / `fattn-common.cuh`): the band predicate
`ggml_cuda_fattn_band_wmma_applies`, `ggml_cuda_fattn_band_wmma_ncols1`, the native-KV type selector
`ggml_cuda_fattn_kv_native_type`, the derived-mask reachability (`kq_mask_derivable`), and the
`launch_fattn` branches. Target: move the predicate *bodies* into `fattn-rdna.cuh` and leave predicate
calls in `launch_fattn`; the tile/mma **generated instances stay in the template-instance TUs** (G6 —
never move them). Highest risk (numerics-sensitive, large gates). Owner blocks 04–07/05.

Gate: `test-backend-ops -o FLASH_ATTN_EXT` full matrix (6389 cases; also with `GGML_CUDA_FA_KV_NATIVE=0`
and `GGML_HIP_FA_BAND_WMMA=0`), width purity, `gate-prefill-logits.sh` (prefill kernels!), and a
native-vs-staging same-seed check, in addition to the standard set (§6).

### 4.3 Item 5 — `llama-context.cpp` / `llama-model-loader.cpp` glue

Small, low-risk. Move each concern's setup into a named helper in its owner file, leaving a one-line
call. Concrete sites seen: `llama-context.cpp`'s FA tensor-parallel hint
(`ggml_set_fa_tensor_parallel(...)` at ~line 271), the logits readback ordering, the GDN/MoE/qwen4exp
setup; `llama-model-loader.cpp`'s host-expert policy (`host_experts_mode`, ~line 547) and the mmap
release (`LLAMA_LOAD_RELEASE_MMAP`, ~line 1566). Gate: per-block build, same-seed coherence, the
mmap-VmRSS measurement from the block that owns it.

### 4.4 Item 6 — the maintainer's `ggml-cuda.cu` split (CAPSTONE)

`ggml-cuda.cu` is still **9819 lines**. Split whole function groups into their own TUs:
- `ggml-cuda-try-fuse.cu` — `ggml_cuda_try_fuse` (+ `ggml_cuda_can_fuse` and the fusion-range helpers);
- `ggml-cuda-slab.cu` — the movable-boundary slab (`ggml_cuda_vmm_map_phys` ~516, `struct
  ggml_cuda_slab` ~599, `g_slabs` ~648, `ggml_cuda_slab_*` ~670–1208, `ggml_cuda_slab_work_alloc`,
  `ggml_cuda_device_malloc`, the pool cap);
- `ggml-cuda-moe-cache-hooks.cu` — `ggml_backend_cuda_moe_cache_*` (~8507–8526), the `stage_*` hooks,
  `ggml_backend_cuda_device_moe_cache_*` (~9414–9470), `fattn_stage`/`h2d_stage` accounting (~9650–9681);
- `ggml-cuda-arena.cu` — the expert-cache arena / `optional_alloc_within_free_cap` / compute-margin
  helpers.

Method: pure `git mv` of function groups + `#include` of a shared `ggml-cuda-internal.cuh` (or expose
each via `common.cuh`). Keep the `ggml-cuda.cu` region map in `FILE-OWNERSHIP.md §3` consistent. **Do not
change behaviour**; the slab↔MoE-cache coupling documented in `PHASE2-STATUS.md`/`AGENTS.md` "Critical
facts" (r20, r37) is real — the cache hooks and the slab call each other, so those two splits must land
together and keep the invariants. Gates: per-block build, 4B/qwen4exp/35B coherence, 35B `-ncmoe 40`
cache on/off coherence, prefill-logit.

### 4.5 Work item 6 of the plan — env reads

Already folded per-owner (Q4). No action unless new flags appear.

## 5. Method for a re-home (proven this session)

For a change that belongs in block **N** (the block whose commit currently owns the code; for all the
concern extractions so far that was block 10 for the try_fuse arms, block 09 for the block-09 arms,
block 08 for the RDNA launch policy):

1. Make the change on `v2-phase2` (tip), build, gate (§6).
2. `git diff > /tmp/change.patch` (and copy any NEW files aside — `git stash create` does **not** include
   untracked files, so capture new headers separately).
3. `git checkout -f <block-N> && git apply -3 /tmp/change.patch` (use `-3`: the tip diff may carry later
   blocks' context; e.g. a header declaration added by block 10).
4. Resolve conflicts. **The typical conflict is two independent appends at the same end-of-file point
   (block 10 appended its wrappers where this block now appends) or a block-10 declaration leaking into
   an earlier header.** Resolution is a clean **union**: keep both the earlier block's wrapper/declaration
   and block 10's; reconstruct from the pre-rebase tip file + the owner-block text if needed.
5. `git add -A && git commit --amend --no-edit`; capture `NEW`; `git rebase --onto NEW <old-block-N> v2-phase2`.
6. **Verify the rebased tip equals the validated working tree** for every file the change touched
   (`diff -q /tmp/validated-<file> <(git show HEAD:<file>)`). Note `git diff <commit> af0595360` is
   **no longer empty** after Phase 2 — that old check does not apply; the correct check is "the tree
   delta is exactly the intended refactor".
7. Write a `PHASE2-STATUS.md` entry, add a checkpoint tag + bundle, commit/push the docs to the delivery
   repo `main`.

Useful: `git branch -f tmpNN $NEW` to keep the amended commit reachable during the rebase; delete it
after.

## 6. Gates (run after every step)

```bash
# 1. fresh per-block build of ALL 12 commits (mandatory; ~10-15 min warm via ccache)
~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama-v2 --base 097cf33d9 \
  --tip "$(git -C ~/llama-v2 rev-parse v2-phase2)" --build-dir build-per-block

# 2. prefill-logit release gate (mandatory for prefill-kernel changes)
LLAMA_PPL=~/llama-v2/build-per-block/bin/llama-perplexity \
  ~/llama-cpp-rdna-boosts/scripts/gate-prefill-logits.sh     # needs KLD <= 0.005, same-top-p >= 98 %

# 3. width purity (the band-uniform mmvq gate) — must print "width_purity=PASS (worst maxdiff 0)"
B=~/llama-v2/build-per-block
HIP_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:$B/bin \
  $B/bin/test-logits-width-probe /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt 1024 512 | tail -1

# 4. same-seed coherence vs r50 (text identical; the timing line differs)
#   4B: Qwen3.5/4B/Q8_0, -ngl 99 -sm tensor -mg 0 -n 20 --seed 42 --temp 0
#   qwen4exp: Qwen3.8/Flash-Next/Q4_K_M, -ngl 99 -sm tensor -c 4096 -n 24  (dense: LLAMA_QSA_OFF=1)
#   35B-A3B: Qwen3.6/35B-A3B/Q8_0, -ngl 99 -sm tensor -c 4096 -n 16
#   all with --no-display-prompt --single-turn, wrapped in timeout
```

Known-good reference outputs for 4B / qwen4exp dense+sparse / 35B r50 are in `/tmp/r50-*.txt` on the
last session's host (re-generate from `~/llama-rebase/build-r50` if gone). Known delivery defects
`TODO.md` #56/#57 are **pre-existing, not reorg artifacts — do not fix them here**.

## 7. Pitfalls (learned this session)

- **Run builds in the FOREGROUND**, `timeout`-bounded, grep the log. `ccache` makes reconfigures fast.
  Iterate with `cmake --build build-phase2 -j 16` (no `rm -rf`) before the full `gate-per-block-build.sh`.
- **`git stash create` omits untracked files** — capture new headers/files explicitly when recording a
  validated tree.
- **`git apply -3` is required** when applying a tip diff to an earlier block; plain `git apply` fails on
  later blocks' header context.
- **A re-home can cascade helper hoists.** If block N's extracted function needs a helper that block 10
  hoisted earlier, the hoist must move to block N too (this is what item 3's block-08/09 work did). Move
  the helper, then drop block 10's now-duplicate hunk during the rebase (§5).
- **`topk_moe` is upstream** — do not "extract" it.
- **FA generated instances stay with their axis** (G6) — never move `template-instances/*.cu`.
- **llama-cli MUST use `--single-turn`** (+ `--no-display-prompt`), wrapped in `timeout`.
- **Never push to upstream `ggml-org/llama.cpp`.** Campaign pushes go to the delivery repo `main`. The
  fork (`git@github.com:stew675/llama.cpp.git`) is only touched at the release cut (§8) and only with the
  maintainer's explicit go-ahead.

## 8. Definition of done (V2.0 release cut)

1. All remaining Phase-2 items landed on `v2-phase2`, each block building on its own (12/12) and the full
   gate set green.
2. Re-measure the modified-vs-added metric per block (`REBASE-RESILIENCE.md` §7) and record it; update
   `FILE-OWNERSHIP.md`'s region map for the new files.
3. `scripts/make-patches.sh` + `scripts/make-release.sh` to regenerate `patches/` and `release.json` from
   the canonical `v2-phase2` chain — **the delivery `patches/`/`release.json` are only touched here**.
4. `scripts/validate-set.sh` green (fresh-tarball strict apply).
5. Full release gates: prefill-logit, qwen4exp quant coherence, same-seed coherence, 35B-A3B draft-MTP,
   width purity.
6. Fork-branch refresh: clean apply at `release.json.base`, push `rdna-boosts` to
   `git@github.com:stew675/llama.cpp.git` (`--force-with-lease`) per `AGENTS.md` Pushing policy.
7. Publish the ROCm container images locally (`scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`).
8. Re-issue the current-state docs (`README.md`, `ENVIRONMENT.md`, `AGENTS.md` Critical facts,
   `patches/README.md`, `TODO.md`, `release.json`) and add the `WORKLOG.md` entry; dated records stay
   historical. Move the finished campaign out of `wip/` per the WIP rules.

## 9. Restore a checkpoint

```bash
cd ~/llama.cpp
git fetch /path/to/v2-phase2-ckpt8.bundle refs/heads/v2-phase2:v2-phase2-restored
~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama.cpp \
  --base 097cf33d9 --tip v2-phase2-restored --build-dir build-per-block
```
