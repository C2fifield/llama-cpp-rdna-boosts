# Migration map — old block → target block

Companion to `README.md` and `TARGET-BLOCKS.md`. This is the work list. It maps every old block to its
target block(s) at the **file** level, and where the split is inside a file, to the **function/region**
level. The r50 hygiene hunk moves (`archive/work/r50-merge-hygiene/REDISTRIBUTION.md`) are incorporated
verbatim in §"old-06".

Rules:
- Target numbers refer to `TARGET-BLOCKS.md`.
- "→ 00" etc. means the hunk moves to target block 00.
- If a file is split across targets, the split key is named (function or `@@` region).
- The final tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8` must not change. A move that would change
  it is a finding: stop and report (Q3).

---

## old-00 — structural and architecture fixes → **00**
| hunk | target |
|---|---|
| `launch_fattn` KV-split width invariance | 00 |
| shared-NextN `is_mem_shared` gate (`common/speculative.cpp`) | 00 |
| Vulkan `flash_attn{,_cm1}.comp` (freed-cell / mask) | 00 |

## old-01 — adaptive MTP draft depth → **02** (whole block)
No split. The `src/models/delta-net-base.cpp` comment-only hunk can stay here.

## old-02 — fused chunked gated-delta-net prefill → **03** (whole block)
No split.

## old-03 — BF16 KV cache and native-BF16 FA → **04** (whole block)
No split.

## old-04 — RDNA4 WMMA FA + Q6_K MMQ prefill → split **05** / **09**
| hunk | target |
|---|---|
| `fattn-mma-f16.cuh`, `fattn.cu` (WMMA FA) | 05 |
| `ggml.c`/`ggml.h` `ggml_set_fa_tensor_parallel` + `src/llama-context.cpp` hint | 05 |
| `ggml-cuda.cu` FA dispatch hunk | 05 |
| `mmq.cu` `GGML_CUDA_MMQ_J_MAX` | 09 |
| `mmq-vec-dot.cuh` x_scale/x_df hoist | 09 |
| `tests/test-backend-ops.cpp` (both halves) | 05 and 09 (split by case) |

## old-05 — CPU bit-identical decode/verify → **00** (`ggml-cpu/llamafile/sgemm.cpp`)

## old-06 — general system-operations bucket → **00 / 04 / 05 / 06 / 10 / 11 / 12 / 13 / 14 / 15 / 17 / 18**
The r50 hygiene record is the authority for the forward-dependency moves. Summary:

| hunk / file | target |
|---|---|
| multi-ubatch logits readback ordering (`src/llama-context.cpp`) | 00 |
| `GGML_ENV_STR` macro | 12 (or 18) |
| generic backend expert-cache iface (`ggml-backend-impl.h`, `ggml-backend.{cpp}`, `ggml-backend.h`) | 12 |
| `ggml-backend-meta.cpp` Meta delegation + slab iface entries (the ~435-line canonical block-14 subset that r50 left in 15) | 12 (generic/slab) **and** 16 (qwen4exp/HC subset) |
| scheduler routed-expert rebalance + merged per-layer MoE split (`ggml-backend.cpp`) | 12 |
| device gather / input-takeover / `GGML_SCHED_STAGE*` + `every 224 MiB` gate (`ggml-cuda.cu`, `common.cuh`, `moe-expert-cache.{cu,h}`) | 13 |
| expert-cache engine `moe-expert-cache.{cu,h}` | 13 |
| host-expert pinned `ROCm_Host` loader / `--host-experts` policy (`llama-model-loader.*`, `llama-model.*`, `common/arg.cpp`) | 12 |
| slab API (`ggml-cuda-vmm.h`, `ggml-cuda.cu` slab/device alloc, `GGML_CUDA_SLAB_HEADROOM_MIB`) | 12 |
| `moe_cache_rearm`, `llama_kv_cache::seq_rm` used-range scan | 12 |
| async graph-input uploads `GGML_CUDA_ASYNC_INPUT=1` + `GGML_CUDA_POOL_CACHE_MB` | 14 |
| mmap host-memory reclaim (r49) | 15 |
| 143 generated `template-instances/*` | 04/05/06 (by FA type axis) and 08/11 (MMB/MoE gates) |
| `common/arg.cpp`, `common/common.*`, `common/speculative.cpp` flags | 12 (MoE), 13 (cache), 16 (qwen4exp), 18 (cache sweep) |
| `ggml/src/ggml-cuda/CMakeLists.txt` / `ggml-hip/CMakeLists.txt` / `generate_cu_files.py` | 04/05/06/08/11 (by the instance they build) |
| `tools/llama-bench/llama-bench.cpp` | 12 |

**r50-hygiene forward-dep moves** (from `REDISTRIBUTION.md` §5, already applied at r50 but confirm they
survive the new taxonomy):
- `ggml_cuda_match_hc_mix`, `_hc_mix_closed`, `_hc_combine_norm_identify`, `_hc_combine_norm_set_bf16`,
  `ggml_cuda_match_idx_relu_sum` → **16/17** (qwen4exp/HC)
- `ggml_cuda_cache_blocks_fusion` → **13**
- `ggml_cuda_try_fuse` branches: `cpy_batch`→13/10; `glu_q8_1_*`→13/11;
  `idx_relu_sum_args`→17; `ple_conv_match`/`gdn_conv_match`/`norm_gated_match`→10;
  `mul_mat_id_weighted_rdna3_5`→09/11; `mul_mat_q_pair`→11, `mmb_active/routed/dense`→08;
  `match_moe_weighted_reduction` + `mmb_blk16/is_bf16_only`→11/08;
  `op_add_rms_norm_q8_1`/`op_rms_norm_q8_1`→10; `rdna3_5_dense_glu_disabled`→09/11;
  `x_scale_channel_dst`→11; `op_l2_norm_pair`→10/13; `op_ssm_gate_beta`→10;
  `op_shexp_down_gate`→11.
- `ggml_backend_cuda_moe_cache_gather`/`stage_gather` and `h2d_stage_region_bytes` → **12** hoist or
  **13** (r50 moved them laterally to 15; the target owner is 12 if generic, 13 if cache-specific).
- missing-include owners: `gdn-conv.cuh`/`ple-conv.cuh`/`mmb.cuh`/`norm-gated.cuh`→08/10;
  `indexer-topk.cuh`/`indexer-score.cuh`/`fattn-qsa.cuh`→17; `ggml-moe-weighted-reduction.h`→10/11;
  `cpy-batch.cuh`→13.

## old-07 — meta device-wrapper skip → **00** (`src/llama.cpp`)

## old-08 — fused-core prefill kernels and GPU bit-identical → **10 / 11 / 04 / 05 / 09**
| hunk / file | target |
|---|---|
| `mmb.cu`, `mmb.cuh`, `dequantize.cuh`, `convert.cu`, `getrows.cu`, `argsort.cu`, `ggml/CMakeLists.txt`, `ggml/cmake/common.cmake`, `docs/build.md`, `tests/test-logits-width-probe.cpp`, `tests/CMakeLists.txt` (MMB half) | 08 |
| `gdn-conv.{cu,cuh}`, `ple-conv.{cu,cuh}`, `norm.cu`/`norm.cuh`, `norm-gated.{cu,cuh}`, `unary.{cu,cuh}`, `topk-moe.cu`, `moe-weighted-reduction.*`, `ggml-cpu/ops.cpp` (fusion half) | 10 |
| `mmvq.cu`/`mmvq.cuh` band-uniform nwarps, item-split, `has_fusion`, `mul_mat_vec_q_ksplit`, `calc_rows_per_block_override` | 11 (adjacent to 09) |
| `ggml-cuda.cu` `try_fuse` fusion arms + `ggml_cuda_mul_mat_id` dispatch | 10/11 (by arm) |
| `fattn-common.cuh`, `fattn.cu`, `fattn-tile.cuh` (FA quant coverage / `GGML_CUDA_FA_QUANTS`) | 05 |
| Q8_1 arena retention (graph lifetime) | 14 (arena in 13) |
| 16 generated `template-instances/*` | 04/05/06 or 08/11 by type axis |
| `tests/test-backend-ops.cpp` | 08/10/11 by case |

## old-09 — meta-buffer compute-container headroom → split **00 / 12**
| hunk | target |
|---|---|
| `ggml-backend-meta.cpp` `compute_headroom 16 → 128` | 00 |
| `ggml-cuda-vmm.h` `ggml_cuda_slab_headroom_at_least_mib` + `ggml-cuda.cu` slab headroom | 12 |
| `moe-expert-cache.cu` `-sm layer` 6 GiB headroom call | 13 (or 12) |

## old-10 — k-quant boosts → **09** (whole block)

## old-11 — skip CUDA graphs for multi-token prefill → **14** (whole block)
Includes the `hipGraphExecUpdate` → destroy/re-instantiate recapture (ROCm/rocm-systems#10713).

## old-12 — hybrid HIP all-reduce → **01** (whole block)

## old-13 — fused MoE gate+up+GLU MMQ + mmvq item-split → **11 / 13 / 09**
| hunk / file | target |
|---|---|
| fused MoE gate+up+GLU MMQ, routed compact, pair fusion, shared-expert epilogue, MoE `MUL_MAT_ID` dispatch, MWR, split-load fix, `mmq-config-rdna3-5.cuh`, `quantize.{cu,cuh}` | 11 |
| `ggml-cuda.cu` `try_fuse` MoE arms + iface assignment | 11 (fusions) / 13 (cache-aware) |
| `moe-expert-cache.{cu,h}` engine | 13 |
| `mmvq.cu`/`mmvq.cuh` short-K item-split + ksplit | 11 (with 09) |
| `src/llama-model-loader.cpp` unaligned split-load | 11 |
| `generate_cu_files.py` + 6 generated MoE gate instances | 11 |
| `common/common.cpp` | 11/12 |
| `tests/test-backend-ops.cpp` | 11 |

## old-14 — qwen4exp support → **16 / 17 / 05 / 08 / 10 / 11**
| hunk / file | target |
|---|---|
| `src/models/qwen4exp.cpp` model support, lazy reader, HC, MTP export | 16 |
| `hc-mix.{cu,cuh}`, `hyperconn.{cu,cuh}`, `dsv4-hc.cu`, `gdn-conv.cu` HC arms | 16 |
| `indexer-topk.{cu,cuh}`, `indexer-score.{cu,cuh}`, `fattn-qsa.{cu,cuh}`, `lightning-indexer.cu`, QSA ops | 17 |
| QSA/HC CPU references `ggml-cpu/ops.{cpp,h}` | 16/17 |
| `ggml-cuda.cu` QSA/HC dispatch | 16/17 |
| `fattn-common.cuh` `iq4_nl` first-class KV / MMB stand-downs / PR #63 fusions | 05/08/10/11 per the arm |
| `ggml-hip/CMakeLists.txt` `GGML_USE_HIP` publish | 16 |
| `llama-arch.cpp`, `llama-model.*`, `llama-model-loader.*`, `llama-memory-hybrid-idx.*`, `llama-kv-cache.cpp` qwen4exp plumbing | 16 |
| `src/CMakeLists.txt`, `tests/*` | 16/17 |

**Note.** old-14's commit body says it "folds beta/mmb-general" and PR #63 — those hunks must land in
08/10/11, not in the qwen4exp blocks, or the orthogonality goal fails.

## old-15 — campaign memory wins → **00 / 05 / 06 / 07 / 09 / 10 / 11 / 12 / 13 / 14 / 16 / 17 / 18**
The super-block. Per the commit body:

| win / amendment | target |
|---|---|
| W1 QSA score-chain memory (`GGML_QSA_SCORE_MEM`) | 17 |
| W2 derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS/_VIS`) | 17 |
| W3 keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`) | 17 |
| W4 `ggml-alloc` view-source release | **00** |
| V3 derived kq-mask | 07 |
| V4/V5 native q8_0/q4_0/bf16/… KV + prefill band split + stage arena | 06 |
| MoE-cache consumer hooks + arena + slab ring + `stage_gather` | 13 |
| RDNA4 GQA-6 band + PR #62 64-wide + f16/bf16 arm + #133 GQA-12 | 05 |
| PR #64 verify-band fusions (wide ncols-64, `ssm_gate_beta` ncols, `add_rms_norm_q8_1`) | 05 (FA) / 10 (ssm/add-rms) |
| PR #114 qwen4exp decode fusions | 17 |
| qwen4exp prefill policy (qsa3 RDNA3_0, lightning-indexer, FA stage accounting) | 16 |
| HC16 / hc_mix BF16 / `hc_gate_mix` | 16 |
| build-time: tile per-type instance generation, `-Wno-pass-failed`, tile `type_KV` axis | 04/05/06 |
| r49 Q1_0 `__byte_perm` port | 09 |
| r49 `cpy-batch.cu` device idiom | 00 |
| r49 mmap reclaim | 15 |
| #124 GDN state-gather skip | 03 (or 10) |
| #125 S2 exact top-k AVX2 prefilter | 09 (or 10) |
| #128 gfx1201 grid-stall pad | 09 |
| #133 band-WMMA GQA cap | 05 |
| getenv hot-path caching (issue #65) + clean-build warnings | **18** |
| 105 generated `template-instances/*` | 04/05/06/08/11 by type axis |

---

## Net result (target volumes, approx.)

| target | old contributors | rough added lines |
|---|---|---:|
| 00 | 00,05,07,09a,06part,15part | ~0.4k |
| 01 | 12 | 2.2k |
| 02 | 01 | 1.0k |
| 03 | 02 | 3.1k |
| 04 | 03 | 0.6k |
| 05 | 04FA,15band,15PR62/64(FA) | ~1.2k |
| 06 | 15V4/V5 | ~1.5k |
| 07 | 15V3 | ~0.7k |
| 08 | 08MMB | ~2.6k |
| 09 | 10,04MMQ,15Q1_0/grid/PR57 | ~1.0k |
| 10 | 08fusions,15PR63(ssm/arms) | ~1.2k |
| 11 | 13,08mmID,15PR114 | ~2.3k |
| 12 | 06iface/sched/meta/slab,09b | ~3.0k |
| 13 | 06engine,13engine,15hooks/arena | ~4.0k |
| 14 | 11,06async | ~0.5k |
| 15 | 06mmap | ~0.3k |
| 16 | 14model/HC/lazy,15HC16/policy | ~3.5k |
| 17 | 14QSA,15W1–W3 | ~5.0k |
| 18 | 15env | ~0.3k |

(The volumes are indicative only; the exact split is discovered during redistribution, not by line
budget. This table exists so an unexpected blow-up in one target is visible.)
