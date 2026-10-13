# Phase 2 — execution status (self-contained refactor, G8)

Companion to `PHASE2-REFACTOR.md` (the work plan) and `HANDOFF-PHASE2.md` (the cold start).
Phase 1 is complete at the checkpoint tag `v2-phase1` = `a40841e1e` (12 blocks, tree == the
published delivery).  Phase 2 works on top of it on branch **`v2-phase2`**.

## 0. Refs (current)

| ref | value |
|---|---|
| Phase-1 checkpoint | `v2-phase1` = `a40841e1e` (12 blocks, tree == `af0595360`) |
| Phase-2 branch | `v2-phase2` = **`3802688c1`** (block 11) |
| block 8/9/10 | **`fcce5864e` / `96a7d0d16` / `6d6f9692e`** |
| delivery repo | `~/llama-cpp-rdna-boosts` (`main`) |

## 1. Method (as executed)

- **One extraction per commit, inside the owning block.**  The extraction is applied to the owning
  block's commit (`git commit --amend`) and the later block(s) are rebased (`git rebase --onto`).
  Block-10-owned arms are the cheapest: only block 11 has to be rebased, and block 11 touches none of
  the concern files, so it replays clean.  An amend cascades the block-10 hash; the rebased block 11
  gets a new hash too.
- **Validate before re-homing.**  The extraction is first made on the tip, built fresh
  (`~/bin/build-llama-rocm-714`, `build-phase2`), gated, and only then applied to the owning block.
  The re-homed tree equals the validated tip tree, so the gate carries over.
- **Gates per extraction:** the feature's own gate, a same-seed/byte-identity check, the full per-block
  fresh `all` build (`gate-per-block-build.sh`, 12/12), and — because the extracted arm lives in a
  prefill fusion path — `gate-prefill-logits.sh`.
- **Nothing in `patches/` / `release.json` is touched.**  The Phase-2 tree is the new baseline only
  after V2.0's final gates; earlier work is recoverable at `v2-phase1`.

## 2. Extraction 1 — HC_COMBINE → HC_MIX arm → `hc-mix.cu` (owner block 10)

`ggml_cuda_try_fuse`'s `HC_COMBINE -> BF16 HC_MIX` arm (matcher call + op dispatch) moved into
`ggml_cuda_hc_combine_mix_try_fuse()` in `hc-mix.cu` (declared in `hc-mix.cuh`).  `try_fuse` now has a
single guarded call.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| `test-backend-ops -o HC_MIX` | **30/30** |
| qwen4exp `Qwen3.8-Flash-Next-Q4_K_M` `-sm tensor` dense + sparse vs r50 | **identical** |
| `GGML_CUDA_FUSE_HC_COMBINE_MIX=0` vs default (dense) | **identical** (bit-identical fusion) |
| `GGML_CUDA_DISABLE_FUSION=1` vs r50 `GGML_CUDA_DISABLE_FUSION=1` | **identical** (pre-existing non-bit-identical fusions, not this arm) |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 3. Extraction 2 — indexer relu-sum matcher+arm → `indexer-score.cu` (owner block 10)

`ggml_cuda_match_idx_relu_sum` (~83 lines, previously a `static` in `ggml-cuda.cu`) moved to
`indexer-score.cu` (now non-static, declared in `indexer-score.cuh`; the graph-optimizer call site in
`ggml-cuda.cu` keeps working off the header).  The `try_fuse` arm became
`ggml_cuda_indexer_relu_sum_try_fuse()` in `indexer-score.cu` (it carries the `RDNA3_5` cc guard), so
`try_fuse` is a single guarded call.

Note: the arm is `RDNA3_5`-only, so on this gfx1201 box it is inert — the gate is the build plus the
unchanged-path coherence.  The move is a pure relocation.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| qwen4exp dense + sparse vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 4. Extraction 3 — HC_MIX arm + matchers → `hyperconn.cu` (owner block 10)

`ggml_cuda_match_hc_mix` + `ggml_cuda_hc_mix_closed` (previously `static` in `ggml-cuda.cu`) and the
`hc_mix_on` prefill arm moved into `hyperconn.cu` (`ggml_cuda_hc_mix_try_fuse()`; the kill-switch
`GGML_CUDA_DISABLE_HC_FUSION`/`GGML_CUDA_DISABLE_HC_MIX` semantics are reproduced exactly).  The
matchers are declared in `hyperconn.cuh` (the gate-mix arm in `ggml-cuda.cu` still calls
`ggml_cuda_hc_mix_closed`).  `try_fuse`'s `hc_mix_on` block is now a single guarded call; the
`hc_comb_on` block follows it in the same order.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| `test-backend-ops -o HC_MIX` | **30/30** |
| qwen4exp dense + sparse vs r50 | **identical** |
| `GGML_CUDA_DISABLE_HC_MIX=1` vs default (dense) | **identical** |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 5. Extraction 4 — sigmoid-mul-add arm → `unary.cu` (owner block 10)

The `UNARY(sigmoid) -> MUL -> ADD` per-row-gate arm moved into
`ggml_cuda_sigmoid_mul_add_try_fuse()` in `unary.cu` (declared in `unary.cuh`); the matcher and op
were already there.  `try_fuse` is a single guarded call.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| qwen4exp dense + sparse vs r50 | **identical** |
| `GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0` vs default (dense) | **identical** |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 6. Next (remaining `try_fuse` arms, highest value next)

The big remaining interleaves, in rough value order:

- **hc-comb cluster** (`hc_comb_on`: the REPEAT-anchored combine+norm matcher and the SCALE A-chain,
  plus `ggml_cuda_hc_combine_norm_identify` and the `set_bf16`/`set_bf16_src1` helpers) →
  `hyperconn.cu`.  Owner block 10, the largest remaining piece (~480 lines), but it needs two
  `ggml-cuda.cu` statics hoisted first: `ggml_cuda_is_view_or_noop` and `ggml_cuda_bf16_src1_cache_on`
  (both used widely in `ggml-cuda.cu`), and `hyperconn.cu` must include `mmb.cuh`.  Deferred to a
  focused step.
- **hc gate-mix arm** → `hyperconn.cu` (needs `mmb.cuh` for `ggml_cuda_hc_gate_mix` /
  `ggml_cuda_mmb_gatemix`).
- **conv1d** (`ple_conv`, `gdn_conv`) → `ple-conv.cu` / `gdn-conv.cu`.  Owner block 08 (needs rebasing
  09/10/11; expect `ggml-cuda.cu` conflicts).
- **cpy_batch** → `cpy-batch.cu`.  Owner block 09 (rebase 10/11).
- **norm-gated / norm_rows** → `norm-gated.cu` (uses `ggml_cuda_compute_forward`: needs a hoist).
- **shared-expert / topk-moe / MWR / mmb / q_pair / shexp / weighted-down** matcher relocations.

Then the maintainer's `ggml-cuda.cu` split (`ggml-cuda-try-fuse.cu`, `-slab.cu`,
`-moe-cache-hooks.cu`, `-arena.cu`).

## 7. Process decision and checkpoint

**Resolved (maintainer, 2026-10-12):** amend-owning-block + rebase is correct; hoisting more per change
(batching a whole concern rather than one tiny arm) is encouraged.  The earlier `git diff <commit>
<delivery-tip>` empty-check does not apply after Phase 2 changes the tree; the correct check is that the
tree delta is exactly the intended extraction.

**Checkpoint 8** (`checkpoints/v2-phase2-ckpt8.bundle`, tag `pin/v2-phase2-ckpt8`): `v2-phase2` =
`9d6730dc8`; block 08 = `fcce5864e`, block 09 = `96a7d0d16`, block 10 = `5d472a3eb`.  **Checkpoint 7**:
`24ef342b3` / block 08 `51bfc6c7f` / 09 `eb2105b51` / 10 `50f5fa1d1`.  **Checkpoint 6**:
`cf9c2e474` / block 08 `8296626db` / 09 `9cf2c7181` / 10 `bf1fbfa55`.  **Checkpoint 5**: `8326733e1` /
block 08 `8296626db` / 09 `7ffdd65e9` / 10 `11ddfb213`.  **Checkpoint 4**: `9c1ec61d6` / block 10
`3238b04e9`.  **Checkpoint 3**: `463c1d598` / `952aa59f7`.  **Checkpoint 2**: `d86bcf088` /
`fbed3f4c0`.  **Checkpoint 1**: `d8ad56737` / `1fa36b3ea`.  Restore instructions in
`checkpoints/README.md`.

## 7b. Batching pass 1 — conv/norm, decode/norm folds, MWR (2026-10-12, session 2)

After the maintainer's go-ahead, the remaining block-10-owned arms were extracted in clusters.  All are
pure moves; every cluster cleared a fresh tip build, 4B same-seed, qwen4exp dense+sparse, 35B-A3B,
`test-backend-ops` where applicable, the full per-block `all` build (**12/12**) and
`gate-prefill-logits.sh` (**0.000707 / 98.755 %**).

### Cluster A — conv1d + narrow-row norm → `ple-conv.cu` / `gdn-conv.cu` / `norm-gated.cu`

`ggml_cuda_ple_conv_try_fuse`, `ggml_cuda_gdn_conv_try_fuse`, `ggml_cuda_norm_rows_try_fuse`.  Hoist:
`ggml_cuda_compute_forward` is now non-static (declared in `common.cuh`) so `norm-gated.cu` can
dispatch the gate `MUL_MAT` before the fused norm.  Kill-switch A/Bs (`GGML_CUDA_DISABLE_CONV_FUSION=1`,
`GGML_CUDA_DISABLE_NORM_ROWS=1`) bit-identical.

### Cluster B — cpy-batch / GDN / SSM / norm-fold arms + helper hoists

- `ggml_cuda_cpy_batch_try_fuse` → `cpy-batch.cu`.
- `ggml_cuda_gdn_beta_sigmoid_try_fuse` + `ggml_cuda_gdn_cache_cpy_try_fuse` → `gated_delta_net.cu`.
- `ggml_cuda_ssm_scan_cache_cpy_try_fuse` → `ssm-scan.cu`.
- `ggml_cuda_add_rms_q8_1_try_fuse` + `ggml_cuda_norm_q8_1_try_fuse` → `norm.cu`.
- Hoists: `ggml_cuda_is_view_or_noop` → `common.cuh` (inline); `ggml_cuda_should_fuse_mul_mat_vec_q`
  → `mmvq.cu`/`.cuh`; `ggml_cuda_try_gdn_cache_fusion` → `gated_delta_net.cu`;
  `ggml_cuda_try_ssm_scan_cache_fusion` → `ssm-scan.cu`; `ggml_cuda_check_fusion_memory_ranges`
  now non-static (declared in `common.cuh`).

### Cluster C — MoE weighted reduction (MWR) → `moe-weighted-reduction.cu`

`ggml_cuda_moe_weighted_reduction_try_fuse` (matcher already portable in `ggml-moe-weighted-reduction.h`).

### Cluster D — routed weighted-down + MUL_MAT pair

- `ggml_cuda_mul_mat_id_weighted_down_try_fuse` → `mmvq.cu` (the qwen4exp 10-expert weighted sum).
- `ggml_cuda_mul_mat_q_pair_try_fuse` → `mmq.cu` (the gate+up / shared-expert pair), with the `cc`
  lookup moved into the helper.

### Cluster E — the rest of the block-10 arms (try_fuse 2244 → 1262 lines)

- `ggml_cuda_mul_mat_q_swiglu_try_fuse` + `ggml_cuda_mul_mat_q_swiglu_dense_try_fuse` → `mmq.cu`;
  `ggml_cuda_mul_mat_takes_mmq` exposed via `common.cuh`.
- `ggml_cuda_ssm_conv_in_try_fuse` + `ggml_cuda_ssm_prescan_try_fuse` + dual-output + MoE down fold → `mmvq.cu`.
- `ggml_cuda_hc_combine_norm_try_fuse` (the whole `hc_comb_on` block) and `ggml_cuda_hc_gate_mix_try_fuse`
  → `hyperconn.cu`, together with `ggml_cuda_hc_combine_norm_identify` / `_set_bf16` / `_set_bf16_src1`
  (the graph optimizer calls `_identify` off the header); `ggml_cuda_bf16_src1_cache_on` hoisted to
  `common.cuh`.

All of A–E: 4B + qwen4exp dense/sparse + 35B-A3B greedy text **identical to r50**; HC kill-switches
(`GGML_CUDA_DISABLE_HC_COMB/MIX/FUSION`) identical; `test-backend-ops -o HC_MIX` 30/30; per-block `all`
**12/12**; prefill-logit **0.000707 / 98.755 %**.  `ggml-cuda.cu` 11307 → **10069** lines.

**Block-10 arms are complete.**

### Cluster F — the clean block-08/09 arms re-homed in their introducer blocks (option A)

`glu_q8_1`, `shared_expert`, `topk_moe` need the helpers hoisted in block 10, so they are handled
separately (see §7c).  The other five only call helpers already in their concern files and were
amended into their introducer blocks:

- **block 08**: `gdn_gate` + `scale_unary` → `unary.cu` (`ggml_cuda_add_softplus_mul_try_fuse`,
  `ggml_cuda_scale_unary_try_fuse`).
- **block 09**: `l2_norm_pair` → `norm.cu`; `gate_beta_verify` + `shexp_down_gate` → `mmvq.cu`.

Method: extract at the tip, validate, then `git apply -3` onto the introducer block, `commit --amend`,
and `git rebase --onto <new-block> <old-block> v2-phase2`.  The only conflicts were in the concern
files: block 10's diff appends its wrappers at the same end-of-file point, and the tip diff carries
block-10 declarations into the earlier header.  Resolution is a clean **union** (keep both the earlier
block's wrapper/declaration and block 10's), reconstructed from the pre-rebase tip file plus the
owner-block wrapper text; `ggml-cuda.cu` auto-merged.  The tree delta of each rebase equals the
extraction.

`try_fuse` **1262 → 1102 lines**.  Gates: 4B + qwen4exp dense/sparse + 35B-A3B identical; per-block
`all` **12/12**; prefill-logit **0.000707 / 98.755 %**.

Remaining: `glu_q8_1` + `shared_expert` (block 09) and `topk_moe` (block 08), which need
`ggml_cuda_should_fuse_mul_mat_vec_q`, `ggml_cuda_check_fusion_memory_ranges` and
`ggml_cuda_match_shared_expert` moved from block 10 into block 08/09 first.

### Cluster G — the hoist arms, and `try_fuse` COMPLETE (try_fuse 1102 → 1064 lines)

- **`topk_moe` skipped**: the `ggml_cuda_topk_moe_fusion` matcher and the `//topk-moe` arm are
  **upstream-unchanged** (the only delivery edit is the two-line `!ggml_cuda_topk_moe_fusion_disabled()`
  guard).  Moving upstream code would not improve rebase resilience, so it is left alone.
- **`glu_q8_1` + `shared_expert` (block 09)** re-homed, together with the helper hoists they need,
  moved from block 10 to block 09: `ggml_cuda_should_fuse_mul_mat_vec_q` → `mmvq.cu`,
  `ggml_cuda_match_shared_expert` → `mmvq.cu`, `ggml_cuda_should_fuse_mul_mat` exposed,
  `ggml_cuda_check_fusion_memory_ranges` exposed.  New wrappers: `ggml_cuda_glu_q8_1_mark` (a void
  marking call) and `ggml_cuda_match_shared_expert_try_fuse`.
- Rebase procedure: amend block 09 with the moves + extraction, rebase 10/11, and on the single
  `ggml-cuda.cu` conflict restore the validated final files (the tip tree was recorded first and the
  rebased tip's tree is **identical**).  `ggml-cuda.cu` auto-merged the rest.

**`ggml_cuda_try_fuse` = 1064 lines** (Phase-2 start 2244, upstream base 792); `ggml-cuda.cu`
11307 → **9819**.  The only remaining arms in `try_fuse` are upstream's own.

The `ggml_cuda_try_fuse` extraction (Phase-2 work item 1) is therefore **complete**.

## 8. Scale of the remaining work

The `try_fuse` split is the bulk.  `ggml_cuda_try_fuse` is **2212 lines** now (upstream base: 792), and
the delivery matchers *before* it (`topk_moe`, `MWR`, `shared_expert`, the `hc_combine_norm` helpers)
add ~700 more lines in `ggml-cuda.cu` (11307 lines total).  Roughly **25+ delivery arms / ~2300 lines**
remain inside `try_fuse`, plus: `mmvq.cu` dispatch (3257 lines; ~700 lines of `calc_nwarps`/
`calc_rows_per_block`/item-split/ksplit), the FA chooser (`fattn.cu` 919 + the MMA/tile kernels), the
`llama-context.cpp`/`llama-model-loader.cpp` glue, and finally the maintainer's `ggml-cuda.cu` split
(`-try-fuse.cu`, `-slab.cu`, `-moe-cache-hooks.cu`, `-arena.cu`).  The MMQ `J` selection of plan item 2
is **already** a self-contained `ggml_cuda_mmq_select_J` (3 call sites) — largely done.

Batching plan (more per change, per the maintainer): one commit per concern cluster, not per arm —
(a) conv1d + norm arms (block 08); (b) MoE/MMVQ decode cluster: cpy_batch, glu_q8_1, shared_expert,
weighted-down, q_pair (blocks 08/09/10); (c) the ssm/gdn cluster: ssm_conv_in, ssm_prescan, gdn_gate,
gdn_beta_sigmoid, gdn/ssm cpy, gate_beta_verify, l2_norm_pair (block 08/10); (d) the MWR matcher;
(e) the hc_comb cluster (needs `ggml_cuda_is_view_or_noop` / `ggml_cuda_bf16_src1_cache_on` hoists);
(f) the remaining upstream-shared arms.  Each batch still gets a fresh per-block build N/N + its
gate(s).

## 9. Item 3 — `mmvq.cu` dispatch (start, 2026-10-12)

**Item 2 is confirmed done**: `ggml_cuda_mmq_select_J` (`mmq.cu`) already owns the RDNA3_5 `J`
selection and the `GGML_CUDA_MMQ_J_MAX` cap (3 call sites); the config rows live in
`mmq-config-rdna3-5.cuh`.

**Item 3, step 1 — `calc_nwarps` override hook.**  Our band-uniform RDNA policy was a rewrite of
`calc_nwarps`'s RDNA4/RDNA3_0 bodies; upstream also churns that function.  The refactor keeps
**upstream's** `calc_nwarps` bodies intact and moves our policy to a new `mmvq-rdna.cuh`:

```c
// mmvq.cu, top of calc_nwarps(), before upstream's RDNA4 branch:
const int rdna_nwarps = mmvq_rdna_nwarps(type, ncols_dst, table_id);
if (rdna_nwarps > 0) { return rdna_nwarps; }   // 0 = not ours -> upstream logic
```

`mmvq_rdna_nwarps()` returns the whole-band policy for `RDNA4` (1), `RDNA3_0` (8 for
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL, else 1) and `RDNA3_5` (2 for Q8_0, else 1), and 0 for every
other table id.  It is included after `enum mmvq_parameter_table_id` (so it can name the table ids)
and before `calc_nwarps`.

Re-homed in **block 08** (the block that introduced the band-uniform policy); the rebase of
09/10/11 was clean, and the tip's `mmvq.cu`/`mmvq-rdna.cuh`/`mmvq.cuh` are byte-identical to the
validated working tree.

Gates: `test-logits-width-probe` **width_purity=PASS (worst maxdiff 0)** on 4B and 35B-A3B; 4B +
qwen4exp dense/sparse identical to r50; per-block `all` **12/12**; prefill-logit **0.000707 /
98.755 %**.

**Item 3, step 2 — the rest of the RDNA launch policy → `mmvq-rdna-params.cuh`.**  `calc_nwarps_weight`,
`calc_rows_per_block_override`, `calc_rows_per_block_weight`, `mmvq_decode_grid_loop_type` and the
`GGML_MMVQ_RDNA4_*` macros moved to a second header (`mmvq-rdna-params.cuh`) included by `mmvq.cu`
right after `calc_rows_per_block()` (they call `calc_nwarps`/`calc_rows_per_block`, so they need a
later include point than the `mmvq-rdna.cuh` nwarps hook).  `calc_rows_per_block` itself is
**upstream-unchanged** and stays in `mmvq.cu`.  Re-homed in block 08; the 09/10/11 rebase was clean
and the four `mmvq*` files are byte-identical to the validated tree.

Gates: `test-logits-width-probe` **PASS** (4B + 35B, worst maxdiff 0); 4B + qwen4exp dense/sparse
identical; per-block `all` **12/12**; prefill-logit **0.000707 / 98.755 %**.

**Item 3, step 3 — MoE kernel file split: DONE (2026-10-13).**  The shared selectors
(`vec_dot_q_cuda_t`, the `GGML_CUDA_MMVQ_MOE_DENSE` guard, `get_vec_dot_q_cuda`, `get_vdr_mmvq`) were
lifted into a new `mmvq-vecdot.cuh` (included by `mmvq.cu` at the old line 39); `mul_mat_vec_q_moe`
(the dedicated multi-token MUL_MAT_ID kernel) and `mul_mat_vec_q_moe_launch` then moved to a new
`mmvq-moe.cu` with a `mmvq-moe.cuh` template declaration.  The kernel is file-local (`static`); the
launch is non-static and **explicitly instantiated for the 23 types the dense dispatch can reach**
(the surrounding `mul_mat_vec_q_switch_type` instantiates the per-type caller, so a declaration-only
header would otherwise leave the specialization undefined at link).  `mmvq.cu` 3745 → **3368** lines;
`mmvq-moe.cu` 346.  The regions were removed by exact text markers (the earlier line-range attempt had
left the kernel-template line behind), and the file set is picked up by the `.cu` glob (no CMake edit).

**Re-home:** applied to **block 10** (the last block to edit the MoE launch — it widened
`MMVQ_MOE_MAX_BATCH_SIZE` and dropped a cold-seam comment); the tip diff applies there cleanly and only
block 11 (which does not touch `mmvq*`) replays, so the rebase is conflict-free.  Block 09 was the
alternative owner but the tip diff conflicts there (the kernel still carries block 09's `MMVQ_MAX_BATCH_SIZE`
and cold-seam comment), so the split would have had to be re-cut against the block-09 form.  The rebased
tip's four `mmvq*` files are byte-identical to the validated tree.

Gates: 4B + qwen4exp dense/sparse + 35B-A3B same-seed greedy **identical to r50**; `test-logits-width-probe`
**width_purity=PASS (worst maxdiff 0)** on 4B and 35B; `gate-prefill-logits.sh` mean KLD **0.000707**,
same-top-p **98.755 %** PASS; full per-block `all` build **12/12**.  **Item 3 is complete.**

## 10. Item 4 — the FA native-KV / band chooser → `fattn-rdna.cuh` (2026-10-13)

The delivery's flash-attention *predicate bodies* moved out of upstream's `fattn-common.cuh` into a new
`fattn-rdna.cuh` (429 lines), leaving only the include / call points behind.  Moved, as two contiguous
pure-relocation regions:

- **native-KV policy + selectors** — `GGML_CUDA_FA_Q8_CHUNK`, the `fattn_kv_native_policy` enum and
  policy read, the `q8_0`/`q4_0`/`bf16` supported predicates, `ggml_cuda_fattn_kv_layout_ok`, the
  `GGML_CUDA_FATTN_KV_NATIVE_ARM` macro (q4_1/q5_0/q5_1/iq4_nl), `ggml_cuda_fattn_kv_native_type`,
  `ggml_cuda_fattn_tile_kv_native_type`, `ggml_cuda_fattn_native_type_from_kernel`,
  `ggml_cuda_fattn_native_ggml_type`;
- **band-WMMA** — `ggml_cuda_fattn_band_wmma_{ncols1_env,enabled,gqa_max,two_byte,ncols1,split,applies}`.

`fattn-common.cuh` 2421 → **2007** lines; it `#include`s the new header at the old region-1 location
(after `#define FATTN_KQ_STRIDE`, which the header relies on — noted in its include-order comment).
**Re-homed in block 05** — the block that owns the 1053-line `fattn-common.cuh` diff (the FA band /
native-KV / derived-mask block); the tip diff applies there cleanly and the 06–11 replay is
conflict-free (only block 10 touches `fattn-common.cuh` below the moved regions, inside `launch_fattn`).
The rebased tip's `fattn-common.cuh` and `fattn-rdna.cuh` are byte-identical to the validated tree.

**Deliberately left inline** (would have rewritten upstream control flow, not a clean move): the
`launch_fattn` native-staging decision block (it consumes the predicates and the context staging arena)
and the `ggml_cuda_fattn_dequantize_*_chunk` device helpers.  Generated instances stay with their axis
(G6).  There is no `kq_mask_derivable()` function — the name is only a comment alias for the
`dst->src[5] != nullptr` derived-op check.

Gates: `test-backend-ops -o FLASH_ATTN_EXT` **6389/6389** in all three arms (default,
`GGML_CUDA_FA_KV_NATIVE=0`, `GGML_HIP_FA_BAND_WMMA=0`); 4B + qwen4exp dense/sparse + 35B-A3B same-seed
greedy **identical to r50**; q8_0-KV native-vs-staging same-seed **identical**;
`test-logits-width-probe` **PASS (worst maxdiff 0)** on 4B and 35B; `gate-prefill-logits.sh` mean KLD
**0.000707**, same-top-p **98.755 %** PASS; full per-block `all` build **12/12**.
