# Target blocks — detailed spec

Companion to `README.md`. One section per target block. "Sources" are the **old** blocks/hunks that
feed it. "Owner files" are the files the block is the primary modifier of in the target taxonomy;
secondary touches are noted where a hub file is unavoidable (see `FILE-OWNERSHIP.md`). "Deps" are the
target blocks that must precede it.

**Decisions applied (2026-10-12, maintainer):** Q1 keep the MoE and qwen4exp concerns separate, and order
**qwen4exp before the host-resident MoE blocks** (the host-MoE work is the broad enabler that lands after
the model-specific work; the qwen4exp/HC matchers are its only compile dependency). Q2 aggressive reorder /
V2.0 clean break. Q3 tree-invariant where practical, content cleanups documented for Phase 2. Q6 new
numbers + mnemonic aliases. The getenv sweep (old target 18) is **folded into the owning blocks** by
default (Q4, recommended); leaving it as a trailing block would make 19.

**Target: 18 blocks (00–17).** Aliases in parentheses are the stable mnemonics.

---

## 00 — Foundation: band-uniformity, Meta-wrapper, allocator and base correctness (`foundation`)

**Concern.** Independent correctness/structural fixes later blocks build on. Multiple concerns are
allowed here (it is the groundwork block).

**Sources.** old-00 (FA KV-split width invariance, shared-NextN MTP gate, Vulkan freed-cell kernels);
old-05 (CPU `llamafile_sgemm` `n <= 8`); old-07 (single-device Meta wrapper skip); old-09a
(`compute_headroom 16 -> 128`); old-06-part (multi-ubatch logits readback ordering; the generic Meta
`host_buffer`/`events` upload fix if it must precede qwen4exp); old-15.W4 (`ggml-alloc` view-source
release). (`cpy-batch.*` is a block-15 introduction and belongs to target 15, not here.)

**Owner files.** `common/speculative.cpp`; `ggml/src/ggml-cuda/fattn-common.cuh`;
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn{,_cm1}.comp`; `ggml/src/ggml-cpu/llamafile/sgemm.cpp`;
`src/llama.cpp`; `ggml/src/ggml-backend-meta.cpp`; `ggml/src/ggml-alloc.c` + `ggml/include/ggml-alloc.h`;
`src/llama-context.cpp`; `ggml/src/ggml-cuda/cpy-batch.{cu,cuh}`.

**Deps.** None.

**Gates.** Clean apply; per-block `all` build; same-seed coherence; W4 has an A/B revert and
`test-alloc`/`test-batch-alloc`.

---

## 01 — Hybrid HIP all-reduce (`allreduce`)

**Concern.** Hybrid dispatcher (small → internal pipeline, large → NCCL), internal pipeline, opt-in
copy-engine (SDMA) `ce`, bounded in-kernel spin, NCCL-failure fail-soft, AR_PROFILE devices[] fix,
small-message 4-block form, opt-in 2-device P2P.

**Sources.** old-12 (entire block).

**Owner files.** `ggml/src/ggml-cuda/allreduce-hip.*`; `ggml/src/ggml-cuda/ggml-cuda.cu` (comm init /
dispatch region); `ggml/src/ggml-cuda/common.cuh`; `common/arg.cpp` / `include/llama.h`.

**Deps.** 00.

**Gates.** 2- and 3-GPU coherence (hybrid vs RCCL smoke; `nccl` not bit-identical under `-sm tensor`);
MTP purity; depth-16384 decode.

---

## 02 — Adaptive MTP draft depth (`mtp-adaptive`)

**Concern.** `--spec-type draft-mtp-adaptive` credit-bucket controller, start/floor/cap, `accept_partial()`,
docs, tests.

**Sources.** old-01 (entire block).

**Owner files.** `common/speculative-adaptive.h`, `common/speculative.{cpp,h}`, `common/arg.cpp`,
`common/common.{cpp,h}`, `include/llama.h`, `src/llama-context.cpp`, `src/llama-cparams.h`,
`src/models/delta-net-base.cpp`, `tests/test-speculative-adaptive.cpp`,
`tests/test-recurrent-state-depth.cpp`, `tests/test-arg-parser.cpp`, docs/READMEs,
`tools/server/server-context.cpp`, `examples/speculative-simple/`.

**Deps.** 00.

**Gates.** Four-axis MTP gate (rule 0); `prompts/code-reasoning-mixed.txt`; byte-identical greedy text.

---

## 03 — Chunked gated-delta-net prefill (`gdn`)

**Concern.** Fused chunked GDN prefill (fp32 default, BF16/WMMA gfx12 + gfx11), K-independent whole-batch
boundary, `n_rs_batch` snapshot bound, per-sequence rollback guard, memory-hybrid/recurrent plumbing.

**Sources.** old-02 (entire block).

**Owner files.** `ggml/src/ggml-cuda/gated_delta_net_chunked{,_bf16,_bf16_gfx11}.cu{,.cuh}`,
`gated_delta_net.cu`, `src/llama-memory-recurrent.{cpp,h}`, `src/llama-memory-hybrid{,-idx,-iswa}.{cpp,h}`,
`src/models/delta-net-base.cpp`, `ggml.c`/`ggml.h`, `src/llama-context.cpp`, `src/llama-model.cpp`,
`common/common.cpp`, `common/speculative.cpp`, `tests/test-backend-ops.cpp`.

**Deps.** 00.

**Gates.** `test-backend-ops -o GATED_DELTA_NET`; prefill-logit; plain == draft-mtp `n_max <= 7`;
`GGML_CUDA_GDN_CHUNKED=0`; `GGML_CUDA_GDN_CHUNKED_BF16=0`.

---

## 04 — BF16 KV cache and native-BF16 flash attention (`bf16-kv`)

**Concern.** FA `type_KV` template axis, BF16 KV cache, native-BF16 FA reads, roped KV path.

**Sources.** old-03 (entire block).

**Owner files.** `ggml/src/ggml-cuda/fattn-tile.{cu,cuh}`, `fattn-mma-f16.cuh`, `fattn.cu`,
`common.cuh`, `rope.cu`, `ggml-cuda.cu` (FA dispatch), `tests/test-backend-ops.cpp`, plus the BF16-axis
generated instance TUs.

**Deps.** 00.

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT`; width purity; prefill-logit; same-seed.

---

## 05 — Flash-attention RDNA4 WMMA band (`fa-band`)

**Concern.** RDNA4 decode/verify band: WMMA `ncols2 = 8` GQA fold, fixed round-robin KV split, wide
(`ncols == 64`) verify row, f16/bf16 band arm, GQA cap (#133, default 12),
`ggml_set_fa_tensor_parallel` hint, RDNA4 WMMA dispatch, `ncols_opt` arch gate.

**Sources.** old-04:FA; old-15:RDNA4 band (GQA-6, PR #62, f16/bf16 arm, #133).

**Owner files.** `fattn-mma-f16.cuh`, `fattn.cu`, `fattn-tile.cuh`, `ggml-cuda.cu`,
`ggml.c`/`ggml.h`, `src/llama-context.cpp`, `tests/test-backend-ops.cpp`, plus the band instance TUs.

**Deps.** 00, 04.

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT` incl. GQA-6 band; width purity at `n_max 3/5/7`;
`GGML_HIP_FA_BAND_WMMA=0`.

---

## 06 — Native quantized K/V reads in flash attention (`fa-native-kv`)

**Concern.** `ggml_cuda_fattn_dequantize_*_chunk` native reads (q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl/bf16);
`GGML_CUDA_FA_KV_NATIVE` three-state policy; prefill band split + per-context `fattn_stage` arena;
mixed-K/V contract; alloc-size TILE q4_0 arm.

**Sources.** old-15:V4/V5 and follow-ups.

**Owner files.** `fattn-tile.cu{h}`, `fattn-mma-f16.cuh`, `fattn.cu`, `fattn-common.cuh`, `common.cuh`,
`ggml-cuda.cu`, `src/llama-context.{cpp,h}`, `tests/test-backend-ops.cpp`.

**Deps.** 04, 05.

**Gates.** `FLASH_ATTN_EXT` all eight KV types + mixed pairs; same-seed native-vs-staging; width purity;
prefill-logit; reserve matrix; `GGML_CUDA_FA_KV_NATIVE=0`; `GGML_CUDA_FA_STAGE_MAX_MB`.

---

## 07 — Derived kq-mask attention memory (`kq-derived`)

**Concern.** `GGML_OP_FLASH_ATTN_EXT` derived visibility inputs (`cell_pos`/`tok_lo`/`tok_hi`), MMA/tile
in-kernel derivation, `kq_mask_derivable()`, packed-mask fallback, device-side
`flash_attn_kq_derived_blocks` window (issue #53), single-stream guard.

**Sources.** old-15:V3 (all amendments).

**Owner files.** `fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cu{h}`, `fattn.cu`, `ggml-cuda.cu`,
`ggml.c`/`ggml.h`, `src/llama-context.cpp`, `src/llama-graph.{cpp,h}`, `tests/test-backend-ops.cpp`.

**Deps.** 04, 05.

**Gates.** derived/mask-hole cases; same-seed derived on/off; multi-stream keeps the packed mask;
`--kv-unified` `-np 4` smoke; reserve `-799 MiB`.

---

## 08 — MMB quantized prefill GEMM (`mmb`)

**Concern.** MMB GEMM core, F32/tiny-M kernels + width probe, RDNA4 fragment port, dense tile geometry +
quant coverage, per-arch defaults, routed policy, F32 policy split, gfx1100 F32 split.

**Sources.** old-08:MMB (`mmb-general` 0001/0003/0006/0007/0008/0009/0010/0012) + `test-logits-width-probe.cpp`.

**Owner files.** `ggml/src/ggml-cuda/mmb.{cu,cuh}`, `ggml-cuda.cu`, `ggml/CMakeLists.txt`,
`ggml/cmake/common.cmake`, `docs/build.md`, `tests/test-logits-width-probe.cpp`, `tests/CMakeLists.txt`,
`tests/test-backend-ops.cpp`.

**Deps.** 00.

**Gates.** `test-logits-width-probe`; MUL_MAT family; prefill perf per the r50-flagged sweep; width purity.

---

## 09 — K-quant boosts (`kquant`)

**Concern.** Per-type mmvq VDR/nwarps for Q4_K/Q5_K/Q6_K/Q8_0; MoE-expert VDR split; dense Q6_K `VDR=2`;
multi-row verify blocks (PR #57) + exact `__mul24`; `GGML_CUDA_MMQ_J_MAX` + `mmq-vec-dot.cuh` x_scale
hoist; Q1_0 `ggml_cuda_byte_perm`; gfx1201 grid-stall pad.

**Sources.** old-10; old-04:MMQ; old-15:Q1_0/grid-pad/PR57.

**Owner files.** `mmvq.cu`, `mmvq.cuh`, `vecdotq.cuh`, `mmq-vec-dot.cuh`, `mmq.cu`, `mmq.cuh`,
`mmq-load-tiles.cuh`, `ggml-cuda.cu`, `tests/test-backend-ops.cpp`.

**Deps.** 08.

**Gates.** stock-relative `llama-batched-bench -npl 1,4,8`; band-uniform nwarps/VDR rule;
`test-backend-ops` MUL_MAT; q1_0 1304/1304.

---

## 10 — Fused prefill kernels (`fused-prefill`)

**Concern.** GDN/PLE conv1d fusions, narrow-row RMS-norm fusion, `norm-gated`, fused MoE router
(`topk-moe`), `iq4_nl` `GET_ROWS` fallback, fused MoE weighted reduction, PR #63 verify-band fusions
(`ssm_gate_beta` ncols, `add_rms_norm_q8_1`).

**Sources.** old-08:fusions, old-15:PR63, old-14's deferred conv switches into 08's files.

**Owner files.** `gdn-conv.{cu,cuh}`, `ple-conv.{cu,cuh}`, `norm.cu`/`norm.cuh`, `norm-gated.{cu,cuh}`,
`unary.{cu,cuh}`, `topk-moe.cu`, `moe-weighted-reduction.{cu,cuh}`, `getrows.cu`, `argsort.cu`,
`convert.cu`, `dequantize.cuh`, `ggml/src/ggml-cpu/ops.cpp`, `ggml-cuda.cu`, `tests/test-backend-ops.cpp`.

**Deps.** 08, 09.

**Gates.** same-seed fusion on/off; `test-backend-ops` fusion cases; MTP gate; CPU reference where present.

---

## 11 — Fused MoE kernels (`fused-moe`)

**Concern.** Fused MoE gate+up+GLU MMQ (prefill), mmvq short-K item-split + ksplit regression fixes,
multi-token `x_scale_channel_dst` fold, band-uniform shared-expert down epilogue, RDNA3_5/RDNA3_0 gate
relaxations, `mul_mat_q_pair`, MoE `MUL_MAT_ID` decode/verify dispatch, routed compact MMQ, Strix MoE
folds, PR #114 decode fusions (MoE-class).

**Sources.** old-13; old-08:MMID/mmvq; old-15:PR114 (MoE-class).

**Owner files.** `ggml-cuda.cu` (try_fuse MoE arms), `mmq.cu`/`mmq.cuh`, `mmq-vec-dot.cuh`,
`mmq-config-rdna3-5.cuh`, `mmvq.cu`/`mmvq.cuh`, `quantize.{cu,cuh}`, `moe-weighted-reduction.{cu,cuh}`,
`common/common.cpp`, `src/llama-model-loader.cpp`, `tests/test-backend-ops.cpp`, `generate_cu_files.py`
+ MoE gate instance TUs.

**Deps.** 08, 09, 10.

**Gates.** `MUL_MAT_ID`/`MUL_MAT_ID_FUSION`; same-seed fused on/off; MTP; `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`
/ `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`; MoE byte-identity oracles; 2-GPU unaligned split-load timing.

---

## 12 — qwen4exp model support (`qwen4exp`)

**Concern.** `src/models/qwen4exp.cpp` architecture, managed lazy reader + PLE n-gram rows +
`LLAMA_LAZY_IO_THREADS`, HC machinery (`hc-mix`, `hyperconn`, `dsv4-hc`, `hc_combine` norms), the HC16
native-BF16 + hc_gate_mix arms and the BF16 hc_mix fusion, the MTP draft-head/`t_h_nextn` export purity
fix, `llm_arch_supports_sm_tensor`, the `GGML_USE_HIP` dynamic-backend macro, the qwen4exp prefill policy
(qsa3 RDNA3_0 gating, Lightning-Indexer registration, device-query arm), and the **hc_mix/combine/idx
matchers** (r50 had them in block 15).

**Sources.** old-14:model/HC/lazy/MTP-export + 2nd/7th/9th amendments; old-15:HC16/hc-mix/policy/qsa3;
the r50 `ggml_cuda_match_hc_mix`/`_hc_mix_closed`/`_hc_combine_norm_*`/`_match_idx_relu_sum` matchers.

**Owner files.** `src/models/qwen4exp.cpp`, `hc-mix.{cu,cuh}`, `hyperconn.{cu,cuh}`, `dsv4-hc.cu`,
`src/llama-lazy-reader.{cpp,h}`, `src/llama-arch.cpp`, `ggml/src/ggml-hip/CMakeLists.txt`,
`tests/test-lazy-reader.cpp`, `tests/test-llama-archs.cpp`, `src/llama-graph.cpp` (MTP export),
`ggml-cuda.cu` (HC matchers/dispatch), `ggml-cpu/ops.{cpp,h}`, `tests/test-backend-ops.cpp`.

**Deps.** 04–11.

**Gates.** qwen4exp coherence (dense + sparse arms), `test-backend-ops HC_MIX` Q8_0+BF16,
`test-llama-archs`, `test-lazy-reader`, MTP-export width probe (0 mismatches), prefill-logit.

---

## 13 — QSA sparse attention, indexer and QSA memory (`qsa`)

**Concern.** QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`), sparse flash attention
(`GGML_OP_FLASH_ATTN_QSA`) + CPU oracle, `qsa3` packed WMMA, fused `INDEXER_POOL`/`SCORE` ops, f16
indexer caches, derived block-vector cache, two-round 16-bit topk, `indexer-topk.cu` + K/V-head chunking,
decode/verify band-uniform QSA score, `LLAMA_QSA_*` policy, W1–W3 memory wins (score-chain
relu+chunked concat, derived per-block bias/visibility + null guards, keys-only indexer cache) + the
`LLAMA_QSA_SPARSE_FA=0` dense-arm shadowing fix.

**Sources.** old-14:QSA/indexer; old-15:W1/W2/W3 + shadowing fix + qsa3.

**Owner files.** `indexer-topk.{cu,cuh}`, `indexer-score.{cu,cuh}`, `fattn-qsa.{cu,cuh}`,
`lightning-indexer.cu`, `ggml-cuda.cu` (QSA dispatch), `ggml.c`/`ggml.h` (ops), `ggml-cpu/ops.cpp`,
`src/models/qwen4exp.cpp` (QSA graph), `src/llama-graph.{cpp,h}`, `tests/test-backend-ops.cpp`.

**Deps.** 05, 06, 11, 12.

**Gates.** `FLASH_ATTN_QSA` 18/18+, `LIGHTNING_INDEXER`, `TOPK_QSA`; `LLAMA_QSA_OFF=1` plain == draft-mtp
byte-identical; `LLAMA_QSA_SPARSE_FA=0` perplexity ratio; QSA reserve matrix; width purity.

---

## 14 — Host-resident MoE: interface, scheduler, Meta and slab (`moe-iface`)

**Concern.** Generic backend expert-cache iface (`moe_cache_update`/`_take_over`/`_gather`), scheduler
routed-expert rebalance + merged per-layer MoE split, Meta delegation to per-device simple backends
(incl. `stage_gather` guard), movable-boundary slab + slab headroom/aux-reserve plumbing, `GGML_ENV_STR`,
host-expert pinned `ROCm_Host` loader / `--host-experts`, `moe_cache_rearm`, `llama_kv_cache::seq_rm`
used-range scan. The generic Meta `host_buffer`/`events` upload fix belongs to 00 if qwen4exp needs it.

**Sources.** old-06:iface/scheduler/meta/slab/loader (non-engine, non-mmap bulk); old-09b; old-15:r48 Meta
upload/`stage_gather`; old-06 block-16 host half.

**Owner files.** `ggml-backend-impl.h`, `ggml-backend.cpp`, `ggml-backend-meta.cpp`,
`ggml/include/ggml-backend.h`, `ggml-cuda-vmm.h`, `ggml-cuda.cu` (slab/device alloc),
`src/llama-model-loader.{cpp,h}`, `src/llama-model.{cpp,h}`, `src/llama-kv-cache.cpp`,
`src/llama-mmap.{cpp,h}`, `common/{arg,common}.{cpp,h}`, `common/fit.cpp`, `include/llama.h`,
`tools/llama-bench/llama-bench.cpp`, `ggml/src/ggml-{cpu,rpc,...}/*.cpp` (iface NULL fills).

**Deps.** 12, 13 (the hc_mix/idx matchers it moved with), 11.

**Gates.** per-block `all` build (critical); `GGML_CUDA_SLAB=0`; `--fit off`; `-sm tensor` + `-ncmoe`
coherence; `MOE_EXPERT_CACHE_MIB` unset inert.

---

## 15 — CUDA decode expert-cache engine, consumer hooks and arena (`moe-cache`)

**Concern.** Per-layer VRAM slot arena, LFRU admission, UVA cold reads, full-residency identity fast
path; cache-aware fused gate+up+GLU/down folds, `MUL_MAT_ID` takeover, cache-band fusion stand-down;
arena slot-count/layer-uniform re-size fixes; MTP draft compute-buffer cap; wide-prefill compute-layout
drop; device gather + head-pad fix; H2D staging ring.

**Sources.** old-06:engine (`moe-expert-cache.{cu,h}`); old-13:engine; old-15:MoE consumer hooks + arena
fixes + wide-prefill drop + r46/r49 host-pool/rearm remainder; old-06:r12/r26 H2D ring.

**Owner files.** `moe-expert-cache.{cu,h}`, `ggml-cuda.cu` (cache-aware try_fuse arms + iface assignment),
`common.cuh` (`h2d_pin`/`h2d_scratch`), `mmvq.cu`/`mmq.cu`, `ggml-backend-meta.cpp` (`stage_gather`),
`src/llama-context.cpp`, `src/llama-model-loader.cpp`, `tests/test-backend-ops.cpp`.

**Deps.** 11, 14.

**Gates.** `MOE_EXPERT_CACHE_MIB` auto; `gate-qwen4exp-quant-coherence.sh`; `MUL_MAT_ID` 929/929;
byte-identity `de8be4d0c90c`/`15038c19ddc8`; width purity; MTP n3; arena UB-tension gates;
`GGML_CUDA_Q8_1_ARENA_FREE_OLD`.

---

## 16 — CUDA/HIP graph handling for expert-offload decode (`graph-offload`)

**Concern.** Prefill-vs-decode graph-skip token-count detection, HIP `hipGraphExecUpdate` →
destroy/re-instantiate recapture (ROCm kernarg leak), pool/FA-staging/H2D-ring graph-memory generation
recapture, opt-in async graph-input uploads (`GGML_CUDA_ASYNC_INPUT=1`) + legacy-pool cache cap
(`GGML_CUDA_POOL_CACHE_MB`).

**Sources.** old-11; old-06:async inputs/pool cap; old-08:Q8_1 arena graph-lifetime half.

**Owner files.** `ggml-cuda.cu` (graph cache/recapture), `common.cuh`, `ggml-backend.cpp`,
`ggml-cuda-vmm.h`, `tests`, `tools/llama-bench`.

**Deps.** 15.

**Gates.** `-ncmoe` decode replay count; greedy byte-identity; `GGML_HIP_GRAPH_FORCE_UPDATE=1`;
`GGML_CUDA_GRAPH_MEM_GEN=0`; Q8_1 arena-free regression.

---

## 17 — Host memory reclaim (`mmap-reclaim`)

**Concern.** Loader no longer prefetches the whole file; `munmap`s each tensor's mapping range after its
device upload, preserving host-pointer-backed ranges; `LLAMA_LOAD_RELEASE_MMAP=0` kill-switch; skipped
under `--mlock`/`--check_tensors`/lazy.

**Sources.** old-06:r49 mmap reclaim.

**Owner files.** `src/llama-model-loader.{cpp,h}`, `src/llama-mmap.{cpp,h}`,
`ggml-backend-meta.cpp`, `common/arg.cpp`.

**Deps.** 14, 15.

**Gates.** peak `VmRSS` (122.5 → 55.4 GiB case); same-seed + `-sm tensor` coherence; `--load-mode none`
unaffected; `--mlock`/`--check_tensors`/lazy skip.

---

## (18) — Host hot-path getenv caching — **folded, not a separate block** (`host-hotpath`)

By default (Q4 recommendation) each cached `getenv` from old-15's issue-#65 sweep is folded into the
block that introduced the flag, so no block touches ~40 files. If the maintainer prefers a sweep block,
it becomes **18** and must be last and droppable. The cached flags are enumerated in
`MIGRATION-MAP.md` old-15.
