# r50 merge hygiene — block-06 redistribution (reviewable spec)

**Status: in progress, 2026-10-12.** Frozen target and gate are in place; this file is the
reviewable specification of the redistribution. It is *not* the delivery and nothing here is
applied by `apply-all.sh`.

## 0. Frozen target (do not move)

| ref | value |
|---|---|
| pinned tip branch | `pin/r50-tip` |
| tip commit | `9a9e6aa6c` (`rdna-boosts: block 15: campaign memory wins`) |
| **tip tree (must not change)** | `f102245dfb116f02f6b3f572df7fb413a5c65cf8` |
| base | `097cf33d9` |
| base tree | `30c400ccea2e2fb47b663036a6cee08a4dc542d8` |

Everything below is **commit-boundary redistribution only**: the tip tree above is the invariant.
If a step changes it, the step is wrong. Verify with
`git -C <repo> rev-parse <new-tip>^{tree}` and `scripts/validate-set.sh`.

## 1. The gate

`scripts/gate-per-block-build.sh` — fresh `rm -rf` + configure + `cmake --build --target all -k`
for **every** commit in `release.json`'s `base..tip`, one at a time, with ccache.

```bash
./scripts/gate-per-block-build.sh --repo ~/llama.cpp            # uses release.json base/tip
./scripts/gate-per-block-build.sh --repo ~/llama-rebase
```

It must not use a reused build dir (the r2 lesson: a stale cache hides configure/header errors
and reports a wrong per-block list). Exit 0 = 16/16.

## 2. Failure map on the frozen chain

```
OK    block 00..05
FAIL  block 06  ("general system-operations bucket")
FAIL  block 07..14
OK    block 15  (tip)
```

Blocks 07–14 fail because they inherit block 06's errors. Block 14 additionally needs its own
`ggml-backend-meta.cpp` move (see §5). So **block 06 is the job.**

## 3. Block 06 — small hoists (block 06 owns these, they are just misfiled)

These are self-contained definitions currently introduced by block 15; hoist them to block 06.

| file | symbol | note |
|---|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `h2d_stage_slots()` | block 06's `h2d_stage_budget(size_t)` calls it |
| `ggml/src/ggml-cuda/common.cuh` | `h2d_stage_budget_explicit()` | same |
| `ggml/src/ggml-cuda/common.cuh` | **remove** the `h2d_stage_slab[slot]` guard from `h2d_stage_buffer` | it is block-15 slab-ring code; it needs `h2d_stage_slab` + `ggml_cuda_slab_ring_slot_capacity`, neither of which exists at block 06. The full slab path stays in block 15. |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `std::atomic<uint64_t> g_write_seq{1};` + `#include <atomic>` | block 06 already uses `g_write_seq++` |
| `ggml/src/ggml-cuda/mmvq.cuh` | `#define MMVQ_MOE_MAX_BATCH_SIZE 16` | block 06's `moe_cache_max_tok_dev` uses it |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `#include <set>` | `ggml_cuda_vmm_map_phys` uses `std::multiset` |

## 4. Block 06 — missing includes (headers added by later blocks)

Block 06's `ggml-cuda.cu` includes headers that do not exist until the block below adds them.
**Remove the include from block 06 and add it at the owning block** (keeps the tip byte-identical
because the header content still lands where the tip has it).

| header | owning block |
|---|---|
| `ggml-cuda/gdn-conv.cuh`, `ple-conv.cuh`, `mmb.cuh`, `norm-gated.cuh` | 08 |
| `ggml-cuda/indexer-topk.cuh`, `indexer-score.cuh`, `fattn-qsa.cuh` | 14 |
| `ggml-moe-weighted-reduction.h` (note: **not** under `ggml-cuda/`) | 14 |
| `ggml-cuda/cpy-batch.cuh` | 15 |

(The compiler reports only the *first* missing include per TU; unblocking one reveals the next.)

## 5. Block 06 — feature code that belongs to later blocks

Block 06's `ggml-cuda.cu` diff (`85965d912..1292dc77f`, +2560 lines) added matchers and
`ggml_cuda_try_fuse` branches for features owned by blocks 08/10/13/14/15. These are compile-time
(types) and link-time (ops) forward dependencies.

### 5.1 Whole functions added by block 06 to move

| function (block 06 `ggml-cuda.cu`) | uses (owner) | destination |
|---|---|---|
| `ggml_cuda_match_hc_mix` | `ggml_cuda_hc_mix_args` (14) | **14** |
| `ggml_cuda_hc_mix_closed` | `ggml_cuda_hc_mix_args` (14) | **14** |
| `ggml_cuda_hc_combine_norm_identify` | `ggml_cuda_hc_combine_norm_args` (14) | **14** |
| `ggml_cuda_hc_combine_norm_set_bf16` | `hc_combine_norm_args` (14), `ggml_cuda_mmb_*` (08), `ggml_cuda_idx_relu_sum_enabled` (14) | **14** (it is the hc/combine feature) |
| `ggml_cuda_match_idx_relu_sum` | `ggml_cuda_idx_relu_sum_args/enabled` (14) | **14** |
| `ggml_cuda_cache_blocks_fusion` | `cpy_batch` (15), `idx_relu_sum` (14), `ple_conv_match` (08), `gdn_conv_match` (08), `norm_gated_match` (08), `glu_q8_1_*` (15) | **15** (verify: it is the cache-blocks campaign) |

### 5.2 `ggml_cuda_try_fuse` branches added by block 06

`ggml_cuda_try_fuse` pre-exists block 05; block 06 **added** dispatch branches for later features.
Move each branch to the owning block (in the same relative order, so the rejoined tip is
byte-identical). The added lines (block-06 diff line numbers) and owners:

| diff lines | symbol(s) | owner |
|---|---|---|
| 1956 | `ggml_cuda_cpy_batch` | 15 |
| 1963, 1984-1986, 2377 | `ggml_cuda_should_fuse_mul_mat_vec_q`, `cuda_ctx->glu_q8_1_*` | 15 |
| 2005 | `ggml_cuda_idx_relu_sum_args` | 14 |
| 2021-2037 | `ggml_cuda_ple_conv_match`, `ggml_cuda_gdn_conv_match` | 08 |
| 2050-2051 | `ggml_cuda_norm_gated_match` | 08 |
| 2072-2111 | `ggml_cuda_mul_mat_id_weighted_rdna3_5[_ok]` | 10/15 |
| 2119-2170 | `ggml_cuda_mul_mat_q_pair`, `ggml_cuda_mmb_active/routed_will_take/dense_will_take` | 13 (pair), 08 (mmb) |
| 2179-2189 | `ggml_cuda_match_moe_weighted_reduction` (the bare `ggml_match_...` name is **correct** — the missing piece is the `ggml-moe-weighted-reduction.h` include, §4), `ggml_cuda_mmb_blk16/is_bf16_only` | 15 / 08 |
| 2232-2233, 2274-2276 | `ggml_cuda_op_add_rms_norm_q8_1`, `ggml_cuda_op_rms_norm_q8_1` | 15 |
| 2295 | `ggml_cuda_rdna3_5_dense_glu_disabled` | 15 |
| 2380 | `x_scale_channel_dst` (member of `ggml_cuda_mm_fusion_args_host`) | 15 |
| 2425 | `ggml_cuda_op_l2_norm_pair` | 15 |
| 2492 | `ggml_cuda_op_ssm_gate_beta` | 15 |
| 2569 | `ggml_cuda_op_shexp_down_gate` | 15 |

Also arity: block 06's `ggml_cuda_mmq_get_vdr` / `ggml_cuda_mul_mat_q` calls use the post-block-13
signatures — those signature changes must move to block 06 (or the calls later), matching the
block-13 arity fix already applied.

### 5.3 MoE-cache helper references

| block-06 function | references (owner) | action |
|---|---|---|
| `ggml_backend_cuda_moe_cache_gather` | `ggml_backend_cuda_stage_gather` (15) | hoist the small `stage_gather` wrapper to block 06, or move the gather function to 15 |
| `ggml_backend_cuda_device_moe_cache_preflight` | `h2d_stage_region_bytes` (15, common.cuh) | hoist `h2d_stage_region_bytes` to block 06 (it only depends on `h2d_stage_bound`/`_explicit`) |

## 6. Block 14 and block 15 (after block 06 builds)

- **block 14** — move ~435 lines of its canonical `ggml-backend-meta.cpp` changes (currently in
  block 15) into block 14, plus the `ggml_cuda_mul_mat_takes_mmq` `warp_size`/`MMVQ_MOE_MAX_BATCH_SIZE`
  usage in `ggml-cuda.cu`. Re-add `indexer-topk.cuh`/`indexer-score.cuh`/`fattn-qsa.cuh` here (§4).
- **block 15** — shrink by exactly the block-14 subset and the block-06 hoists; keep its own fixes
  (ssm-fusion restore, fattn `get_swizzled`/`load_tile` dedup, mmq duplicate template, `sched_new`).

## 7. Execution method

`git rebase -i 097cf33d9` with `edit` stops at blocks **06, 08, 10, 13, 14, 15**; at each stop:

1. apply that block's moves (direct file edits — **not** `--autosquash`; the r2 trap: a fixup's
   base is the tip, so it conflicts or double-applies),
2. `git add -A && git commit --amend --no-edit`,
3. `./scripts/gate-per-block-build.sh` limited to that commit (or a direct `cmake --build` of the
   worktree) before `git rebase --continue`.

At the final stop, force the tree to `pin/r50-tip` (`git checkout pin/r50-tip -- .`) so the tip is
byte-identical by construction; `validate-set.sh` then proves it.

## 8. Definition of done

- `./scripts/gate-per-block-build.sh` → **16/16**.
- `release.json` tree unchanged (`f102245…`); `scripts/validate-set.sh` green.
- gates re-run green: prefill-logits, q4exp quant coherence, same-seed coherence, 35B draft-mtp.
- `WORKLOG.md` r50 entry updated; this campaign leaves `wip/` (no stub) when done.
