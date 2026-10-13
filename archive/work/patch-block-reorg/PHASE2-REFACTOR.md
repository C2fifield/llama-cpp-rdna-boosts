# Phase 2 — self-contained refactor work plan

Companion to `README.md` §6 and `REBASE-RESILIENCE.md` §5. **Decision (Q7/Q8, 2026-10-12):** Phase 2 is
in scope now; V2.0 = Phase 1 + Phase 2. The general principle is **G8**: extract each concern into a
self-contained function (ideally its own file) and leave behind only the **call point**. Interleave into
upstream/churning files as little as possible. A Phase 3 may catch residual stragglers.

This document is the working list of extraction targets, ordered by rebase-pain relief. Each entry names
the current interleave site, the target function/file, the (single) call point, and the gate that must
stay green. It is expected to be refined as Phase 1's redistribution exposes the exact hunks.

---

## 0. Rules for every extraction

1. **One concern, one home.** Move the logic into a function whose name names the concern; put it in the
   concern's own file where one exists (`gdn-conv.cu`, `moe-expert-cache.cu`, `hc-mix.cu`, …), otherwise
   a new `<concern>.cu`/`.cuh`.
2. **Only the call point stays in the shared file.** The shared file gets an `#include` and one call
   (or one small dispatch line). No setup, no locals, no inline conditionals beyond the call's guard.
3. **Preserve ordering.** `ggml_cuda_try_fuse`'s arm order determines which fusion wins; the extracted
   call sequence must reproduce the current order exactly (same-seed gates are the proof).
4. **No forward deps.** The extracted function may only reference earlier blocks' symbols (G3); the fresh
   per-block `all` build is the check.
5. **Generated instances stay with their axis** (G6), even if the axis logic moved to a new file.
6. **Behaviour-preserving unless separately gated.** Each extraction is a pure move first; any semantic
   change is its own commit with its own A/B and gate.
7. **One extraction per commit**, inside the owning block (amend the block, or add a sub-commit that the
   release cut squashes). This keeps `git rebase -i` stops small.

---

## 1. `ggml_cuda_try_fuse` arms — highest value, lowest risk

**Current interleave.** ~600+ lines of matcher bodies and arm setup live in
`ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_match_*`, `ggml_cuda_*_args`, inline arm bodies).

**Target.** One self-contained "try-fuse" function per feature concern, living in the concern's file,
with `ggml_cuda_try_fuse` reduced to an ordered sequence of guarded one-line calls. Example shape:

```c
// ggml-cuda.cu (shared; call points only)
if (ggml_cuda_gdn_conv_try_fuse(cgraph, i, ...)) { ... }   // gdn-conv.cu (owner 10)
if (ggml_cuda_norm_gated_try_fuse(cgraph, i, ...)) { ... } // norm-gated.cu (owner 10)
if (ggml_cuda_moe_pair_try_fuse(cgraph, i, ...)) { ... }   // mmq.cu (owner 11)
if (ggml_cuda_cache_blocks_try_fuse(cgraph, i, ...)) { ... } // moe-expert-cache.cu (owner 15)
if (ggml_cuda_hc_mix_try_fuse(cgraph, i, ...)) { ... }     // hc-mix.cu (owner 12)
if (ggml_cuda_idx_relu_sum_try_fuse(cgraph, i, ...)) { ... } // indexer-score.cu (owner 13)
```

| feature | current site | target file | call point | owner block | gate |
|---|---|---|---|---|---|
| `cpy_batch` | `ggml-cuda.cu` | `cpy-batch.cu` | one arm | 15 | same-seed + `GGML_CUDA_DISABLE_FUSION` A/B |
| `glu_q8_1` / `mul_mat_vec_q` fuse | `ggml-cuda.cu` | `mmvq.cu` | one arm | 15/11 | same-seed; width purity |
| `idx_relu_sum` | `ggml-cuda.cu` | `indexer-score.cu` | one arm | 13 | qwen4exp coherence |
| `ple_conv` / `gdn_conv` / `norm_gated` | `ggml-cuda.cu` | `ple-conv.cu` / `gdn-conv.cu` / `norm-gated.cu` | one arm each | 10 | same-seed fusion on/off |
| `mul_mat_id_weighted_rdna3_5` | `ggml-cuda.cu` | `mmvq.cu` | one arm | 9/11 | `MUL_MAT_ID`; width purity |
| `mul_mat_q_pair` | `ggml-cuda.cu` | `mmq.cu` | one arm | 11 | `test-backend-ops`; same-seed |
| `mmb_active/routed/dense` | `ggml-cuda.cu` | `mmb.cu` | one arm | 8 | `test-logits-width-probe` |
| `match_moe_weighted_reduction` | `ggml-cuda.cu` | `moe-weighted-reduction.cu` | one arm | 11 | `MOE_WEIGHTED_REDUCTION` |
| `mmb_blk16/is_bf16_only` | `ggml-cuda.cu` | `mmb.cu` | one arm | 8 | MMB gates |
| `op_add_rms_norm_q8_1` / `op_rms_norm_q8_1` | `ggml-cuda.cu` | `norm.cu` | one arm | 10 | same-seed; PR #63 gate |
| `rdna3_5_dense_glu_disabled` | `ggml-cuda.cu` | `mmvq.cu` / `mmq.cu` | predicate call | 9/11 | RDNA3_5 gates |
| `x_scale_channel_dst` | `ggml-cuda.cu` | `mmvq.cu` | arm | 11 | multi-token `MUL_MAT_ID` |
| `op_l2_norm_pair` | `ggml-cuda.cu` | `norm.cu` | one arm | 10/15 | same-seed |
| `op_ssm_gate_beta` | `ggml-cuda.cu` | `gdn-conv.cu` | one arm | 10 | PR #63 gate |
| `op_shexp_down_gate` | `ggml-cuda.cu` | `mmvq.cu` | one arm | 11 | `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` |
| `hc_mix` / `_hc_mix_closed` / `_hc_combine_norm_identify` / `_hc_combine_norm_set_bf16` | `ggml-cuda.cu` | `hc-mix.cu` / `hyperconn.cu` | one arm each | 12 | `HC_MIX` 30/30; qwen4exp coherence |
| `cache_blocks_fusion` | `ggml-cuda.cu` | `moe-expert-cache.cu` | one arm | 15 | cache A/B; width purity |

**Outcome.** `ggml-cuda.cu`'s `try_fuse` becomes a short ordered dispatch list; each feature owns its
matcher in its own file. An upstream change to `try_fuse` then collides with a handful of call lines, not
with every feature's body.

---

## 2. MMQ `J` selection and config dispatch

**Current interleave.** `ggml_cuda_mul_mat_q` in `mmq.cu` contains our `GGML_CUDA_MMQ_J_MAX` cap and
per-type `J` selection inline; config rows live in `mmq.cuh` / `mmq-config-rdna3-5.cuh`.

**Target.** A self-contained `ggml_cuda_mmq_select_J_rdna0(type, config_fn, ...)` (in `mmq-config-rdna3-5.cuh`
or an `mmq-rdna.cuh`) plus a config lookup that owns all our rows. `mmq.cu` calls it once.

**Owner.** 8/9. **Call point.** one call in `ggml_cuda_mul_mat_q`. **Gate.** stock-relative
`llama-batched-bench -npl 1,4,8`; `test-backend-ops` MUL_MAT; prefill perf.

---

## 3. `mmvq.cu` dispatch (item-split / ksplit / VDR)

**Current interleave.** `mul_mat_vec_q_switch_ncols_dst` / `mul_mat_vec_q_switch_fusion` /
`calc_nwarps` / `calc_rows_per_block*` hold several distinct concerns: the band-uniform nwarps/VDR
policy, the short-K MoE item-split, the dense ksplit fallback, and the multi-row verify blocks.

**Target.** A feature-local dispatch helper (e.g. `mmvq-rdna.cuh`) that chooses the kernel and the
`nwarps`/VDR/rpb parameters, with `mmvq.cu` keeping only the call. Where a kernel is separable, move it
to its own `.cu` (e.g. the MoE item-split kernel to `mmvq-moe.cu`).

**Owner.** 9/11. **Call point.** the dispatch line in `ggml_cuda_mul_mat_vec_q`. **Gate.** band-uniform
nwarps/VDR rule (`GREEDY-PURITY.md` §§19/24/25); `llama-batched-bench -npl 1,4,8`; width purity.

---

## 4. FA `type_KV` / band / native-KV dispatch

**Current interleave.** `fattn.cu`'s chooser and `launch_fattn` hold the band predicate, the native-KV
type selection, the derived-mask support check and the type-KV case dispatch.

**Target.** Keep the self-contained predicates (`ggml_cuda_fattn_band_wmma_applies`,
`ggml_cuda_fattn_kv_native_type`, `kq_mask_derivable`) as the interface, move their bodies and the inline
branches into a `fattn-rdna.cuh`; `launch_fattn` calls them. The tile/mma instance dispatch stays in the
generated TUs (G6).

**Owner.** 4/5/6/7. **Call point.** the predicate calls in `launch_fattn` / the chooser. **Gate.**
`FLASH_ATTN_EXT` full matrix; width purity; prefill-logit; native-vs-staging same-seed.

---

## 5. Context/loader glue

**Current interleave.** `src/llama-context.cpp` carries the FA tensor-parallel hint, the logits readback
ordering, GDN/MoE/qwen4exp setup; `src/llama-model-loader.cpp` carries the host-expert policy and the mmap
release.

**Target.** Move each concern's setup into a named helper in its owner file (e.g.
`llama_context_fa_tensor_parallel_hint()` in the FA hint's file, the readback ordering into the existing
readback path), leaving a one-line call. The loader's mmap release and host-expert policy become helper
functions in `llama-mmap`/`llama-model-loader` with one call site.

**Owner.** 00/05/14/17. **Gate.** per-block build; same-seed coherence; the mmap VmRSS measurement.

---

## 6. Environment-variable reads (Q4)

**Current interleave.** old-15's issue-#65 caches many hot-path `getenv` reads; as re-homed (Q4) each
cached read belongs to the owning block.

**Target.** A small `static`/file-static cache at first use in the owning file (the current pattern), or
a tiny `env_<feature>.h` inline helper when the same flag is read from several files. Never a central
env file (that recreates the sweep). **Owner.** the flag's feature block. **Gate.** clean build;
behaviour unchanged; the Windows host-lock rationale is a perf claim only.

---

## 7. Phase-2 validation and checkpoints

1. **Phase-1 checkpoint (before Phase 2):** 18 blocks, per-block `all` build N/N, `validate-set.sh`
   green, tree `f102245…` where practical; scratch `release.json`. Tag it (e.g. `v2-phase1`). Phase 2
   starts from here.
2. **Per extraction:** fresh `all` build of the modified block and every later block; the feature's gate;
   a same-seed/byte-identity check where applicable.
3. **After each block's extractions:** re-run `scripts/gate-per-block-build.sh` for that block onward.
4. **Final V2.0:** re-measure the modified-vs-added metric per block (`REBASE-RESILIENCE.md` §7) and
   record it; re-baseline `release.json` (new tree); full release gates (prefill-logit, qwen4exp quant
   coherence, same-seed, 35B draft-mtp); `validate-set.sh`; fork-branch refresh + containers per the
   Pushing policy.

## 8. Success metrics

- Maximum modified-upstream-file count per block drops (target: no block touches > ~10 shared modified
  files; the FA/quant blocks are the expected residual).
- No block has a forward dependency (per-block `all` build N/N).
- `ggml_cuda_try_fuse` contains no feature setup, only ordered call points.
- The number of blocks touching each hub file is stable/known (the region map holds).
