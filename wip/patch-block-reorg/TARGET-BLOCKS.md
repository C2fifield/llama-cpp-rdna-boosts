# Target blocks — detailed spec

Companion to `README.md`. One section per target block. "Sources" are the **old** blocks/hunks that
feed it. "Owner files" are the files the block is the primary modifier of in the target taxonomy;
secondary touches are noted where a hub file is unavoidable (see `FILE-OWNERSHIP.md`). "Deps" are the
target blocks that must precede it. This is a proposal; the contested seams are Q1 in `OPEN-QUESTIONS.md`.

Notation used in the source column: `old-NN` = existing block NN; `old-NN:part` = a named sub-concern
of that block (the parts are enumerated in `MIGRATION-MAP.md`).

---

## 00 — Foundation: band-uniformity, Meta-wrapper, allocator and base correctness

**Concern.** Independent correctness/structural fixes that later blocks build on. The user's brief
explicitly allows multiple concerns here because it is the groundwork block.

**Sources**
| part | old source | lines | note |
|---|---|---|---|
| FA KV-split width invariance (`launch_fattn` evaluates the heuristic as `n_q == 1` for every `n_q <= 8`) | old-00.1 | ~30 | makes decode and verify pick the same KV split; issue #25 |
| shared-NextN MTP head keeps its own KV (`is_mem_shared` gate) | old-00.2 | ~60 | upstream bug 04eb4c446 / #23398 |
| CPU `llamafile_sgemm` routes `n <= 8` to `vec_dot` | old-05 | +7/−1 | decode/verify band uniformity on CPU |
| single-device `-sm tensor` skips the Meta wrapper | old-07 | +25/−15 | `src/llama.cpp` `llama_prepare_model_devices` |
| meta-buffer compute-container headroom `16 → 128` | old-09a | ~10 | views for GDN/recurrent snapshots |
| multi-ubatch logits readback ordering | old-06-part | ~40 | async D2H race; `src/llama-context.cpp` |
| `ggml-alloc` release view sources whose views are never consumed | old-15.W4 | ~20 | pure allocator bugfix (issue repro 56→16 MiB) |
| `cpy-batch.cu` uses `ggml_cuda_get_device()` | old-15.r49 | ~5 | no-op idiom fix |

**Owner files.** `common/speculative.cpp`; `ggml/src/ggml-cuda/fattn-common.cuh`;
`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn{,_cm1}.comp`; `ggml/src/ggml-cpu/llamafile/sgemm.cpp`;
`src/llama.cpp`; `ggml/src/ggml-backend-meta.cpp` (compute-headroom hunk only);
`ggml/src/ggml-alloc.c` + `ggml/include/ggml-alloc.h` (view-release hunk only);
`src/llama-context.cpp` (readback-ordering hunk only); `ggml/src/ggml-cuda/cpy-batch.{cu,cuh}`.

**Deps.** None.

**Gates.** Clean apply; same-seed coherence; per-block `all` build; the allocator fix gets an A/B revert
(old-15.W4 had one) and `test-alloc`/`test-batch-alloc`.

---

## 01 — Hybrid HIP all-reduce

**Concern.** The multi-GPU all-reduce alternatives: hybrid dispatcher (small → internal host-staged
pipeline, large → NCCL), internal pipeline, opt-in copy-engine (SDMA) `ce`, bounded in-kernel spin,
NCCL-failure fail-soft, AR_PROFILE devices[] fix, small-message 4-block form, opt-in 2-device P2P.

**Sources.** old-12 (entire block: hybrid + nccl fail-soft + ce + AR_PROFILE + r46 small-msg/P2P).

**Owner files.** `ggml/src/ggml-cuda/allreduce-hip.{cu,cuh}` (or whatever the current set is);
`ggml/src/ggml-cuda/ggml-cuda.cu` (comm init + dispatch region);
`ggml/src/ggml-cuda/common.cuh` (small-msg helpers); `common/arg.cpp`/`include/llama.h` if the
`GGML_CUDA_ALLREDUCE` plumbing is there.

**Deps.** 00.

**Gates.** 2- and 3-GPU coherence (hybrid vs RCCL smoke; note `nccl` is not bit-identical under
`-sm tensor`); MTP purity; depth-16384 decode. AR_PROFILE default-off must not regress.

**Rationale for placement.** It is multi-GPU transport used by `-sm tensor` MoE and qwen4exp; putting it
early (own files, low conflict) removes it from the later `ggml-cuda.cu` jigsaw. It is the only real
candidate to leave entirely where it was; if the maintainer prefers, keep it late (see Q2).

---

## 02 — Adaptive MTP draft depth

**Concern.** `--spec-type draft-mtp-adaptive` credit-bucket controller, start/floor/cap options,
`accept_partial()`, docs, tests.

**Sources.** old-01 (entire block).

**Owner files.** `common/speculative-adaptive.h`, `common/speculative.{cpp,h}`, `common/arg.cpp`,
`common/common.{cpp,h}`, `include/llama.h`, `src/llama-context.cpp`, `src/llama-cparams.h`,
`src/models/delta-net-base.cpp`, `tests/test-speculative-adaptive.cpp`,
`tests/test-recurrent-state-depth.cpp`, `tests/test-arg-parser.cpp`, docs/READMEs,
`tools/server/server-context.cpp`, `examples/speculative-simple/`.

**Deps.** 00 (shared-NextN gate).

**Gates.** Four-axis MTP gate (`benchmarks/mtp-adaptive-methodology.md` rule 0: `-n 3000`,
`--reasoning on/off`); `prompts/code-reasoning-mixed.txt` phase-switching; byte-identical greedy text
(adaptive cap 7/12 == fixed draft-mtp).

---

## 03 — Chunked gated-delta-net prefill

**Concern.** The fused chunked GDN prefill kernel (fp32 default, BF16/WMMA gfx12 + gfx11 arms),
K-independent whole-batch boundary, `n_rs_batch` snapshot bound, the per-sequence rollback guard,
memory-hybrid/recurrent plumbing.

**Sources.** old-02 (entire block) + old-09a's compute-container headroom is re-homed to 00.

**Owner files.** `ggml/src/ggml-cuda/gated_delta_net_chunked{,_bf16,_bf16_gfx11}.cu{,.cuh}`,
`ggml/src/ggml-cuda/gated_delta_net.cu`, `src/llama-memory-recurrent.{cpp,h}`,
`src/llama-memory-hybrid{,-idx,-iswa}.{cpp,h}`, `src/models/delta-net-base.cpp`, `ggml/src/ggml.c` +
`ggml/include/ggml.h` (op), `src/llama-context.cpp`, `src/llama-model.cpp`, `common/common.cpp`,
`common/speculative.cpp`, `tests/test-backend-ops.cpp`.

**Deps.** 00.

**Gates.** `test-backend-ops -o GATED_DELTA_NET`; prefill-logit gate (the bf16 arm is the issue-#113
example); plain == draft-mtp `n_max <= 7`; `GGML_CUDA_GDN_CHUNKED=0` bit-identical fallback;
`GGML_CUDA_GDN_CHUNKED_BF16=0` A/B.

---

## 04 — BF16 KV cache and native-BF16 flash attention

**Concern.** The FA `type_KV` template axis, BF16 KV cache support, native-BF16 FA reads, the roped
KV path.

**Sources.** old-03 (entire block).

**Owner files.** `ggml/src/ggml-cuda/fattn-tile.{cu,cuh}`, `fattn-mma-f16.cuh`, `fattn.cu`,
`ggml/src/ggml-cuda/common.cuh` (type-KV helpers), `rope.cu`, `ggml-cuda.cu` (FA dispatch case),
`tests/test-backend-ops.cpp`. Generated `template-instances/fattn-tile-instance-*` and
`fattn-mma-f16-instance-*` for the BF16 axis move here.

**Deps.** 00.

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT`; width purity `W=1..8`; prefill-logit; same-seed.

---

## 05 — Flash-attention RDNA4 WMMA band (GQA fold, wide verify band)

**Concern.** The RDNA4 decode/verify band: WMMA `ncols2 = 8` GQA fold, the fixed round-robin KV split,
the wide (`ncols == 64`) verify row, the f16/bf16 band arm, the GQA cap (#133, default 12), the
`ggml_set_fa_tensor_parallel` hint, the RDNA4 WMMA FA dispatch and the `ncols_opt` arch gate.

**Sources.** old-04:FA (WMMA FA + `ggml_set_fa_tensor_parallel` + `src/llama-context.cpp` hint);
old-15:RDNA4 band (GQA-6, PR #62 64-wide batches, f16/bf16 arm, #133 GQA-12).

**Owner files.** `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (config rows), `fattn.cu` (chooser/dispatch),
`fattn-tile.cuh` (band cross-checks), `ggml-cuda.cu` (FA dispatch), `ggml.c`/`ggml.h`
(`ggml_set_fa_tensor_parallel`), `src/llama-context.cpp` (hint), `tests/test-backend-ops.cpp`
(band matrix). Generated WMMA instance TUs for the band rows.

**Deps.** 00, 04.

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT` including the GQA-6 band cases; width purity at
`n_max 3/5/7`; `GGML_HIP_FA_BAND_WMMA=0` off switch; the band must not change W-dependence.

---

## 06 — Native quantized K/V reads in flash attention

**Concern.** `ggml_cuda_fattn_dequantize_*_chunk` native reads for `q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/
`iq4_nl`/`bf16`; the `GGML_CUDA_FA_KV_NATIVE` three-state policy; the prefill band split (native decode,
staged prefill) and the per-context `fattn_stage` arena; the mixed-K/V kernel contract; the alloc-size
TILE q4_0 arm.

**Sources.** old-15:V4/V5 and the follow-ups (last-four native arms, q4_0 NaN fix, prefill band split,
FA stage arena accounted in `--fit`).

**Owner files.** `ggml/src/ggml-cuda/fattn-tile.cu{h}`, `fattn-mma-f16.cuh`, `fattn.cu`,
`fattn-common.cuh`, `common.cuh` (`ggml_cuda_fattn_kv_native_type`), `ggml-cuda.cu` (FA dispatch),
`src/llama-context.{cpp,h}` (stage arena), `tests/test-backend-ops.cpp`.

**Deps.** 04, 05.

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT` (all eight KV types, mixed-pair cases);
same-seed greedy native-vs-staging per type; width purity; prefill-logit; memory reserve matrix;
`GGML_CUDA_FA_KV_NATIVE=0` fallback and `GGML_CUDA_FA_STAGE_MAX_MB`.

---

## 07 — Derived kq-mask attention memory

**Concern.** `GGML_OP_FLASH_ATTN_EXT` derived visibility inputs (`cell_pos`/`tok_lo`/`tok_hi`), the MMA
and tile kernels deriving the mask in-kernel, `kq_mask_derivable()`, the packed-mask fallback, the
device-side `flash_attn_kq_derived_blocks` window (issue #53 host-read fix), the single-stream guard.

**Sources.** old-15:V3 (all amendments: IGPU gate, multi-stream guard, tile kernel, kernel shape,
issue #53).

**Owner files.** `ggml-cuda/fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cu{h}`, `fattn.cu`,
`ggml-cuda.cu`, `ggml.c`/`ggml.h` (op inputs), `src/llama-context.cpp`,
`src/llama-graph.{cpp,h}` (mask inputs), `tests/test-backend-ops.cpp`.

**Deps.** 04, 05 (needs the FA mask path and MMA kernel).

**Gates.** `test-backend-ops -o FLASH_ATTN_EXT` derived/mask-hole cases; same-seed derived on/off;
multi-stream context keeps the packed mask (no abort); `--kv-unified` + `-np 4` server smoke;
memory reserve `-799 MiB`.

---

## 08 — MMB quantized prefill GEMM

**Concern.** The MMB GEMM core (`mmb.cu`), F32/tiny-M kernels and the width probe, the RDNA4 fragment
port, dense tile geometry + quant coverage, per-arch defaults, the routed policy, the F32 policy split,
the gfx1100 F32 split, `mmb.cuh` dispatch, `GGML_CUDA_FA_QUANTS`-adjacent defaults.

**Sources.** old-08:MMB (the `mmb-general` folds: 0001/0003/0006/0007/0008/0009/0010/0012), the
`test-logits-width-probe.cpp` tool.

**Owner files.** `ggml/src/ggml-cuda/mmb.{cu,cuh}`, `ggml/src/ggml-cuda/ggml-cuda.cu` (MMB
dispatch/legacy), `ggml/CMakeLists.txt`/`ggml/cmake/common.cmake`, `docs/build.md`,
`tests/test-logits-width-probe.cpp`, `tests/CMakeLists.txt`, `tests/test-backend-ops.cpp`.

**Deps.** 00.

**Gates.** `test-logits-width-probe`; `test-backend-ops` MUL_MAT family; prefill perf gates per the
r50-flagged open sweep; width purity.

---

## 09 — K-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0 perm, gfx1201 grid pad)

**Concern.** Per-type mmvq VDR and nwarps for Q4_K/Q5_K/Q6_K/Q8_0; the MoE-expert VDR split; dense
Q6_K `VDR=2`; multi-row verify blocks (PR #57) and the exact `__mul24`; the `GGML_CUDA_MMQ_J_MAX` cap
and `mmq-vec-dot.cuh` x_scale hoist; Q1_0 `ggml_cuda_byte_perm` port; the gfx1201 grid-stall pad.

**Sources.** old-10 (entire block); old-04:MMQ (`mmq.cu` J cap + `mmq-vec-dot.cuh` hoist);
old-15:Q1_0 (r49) + old-15:grid-pad + old-15:PR57 (if not in old-10).

**Owner files.** `ggml/src/ggml-cuda/mmvq.cu`, `mmvq.cuh`, `vecdotq.cuh`, `mmq-vec-dot.cuh`, `mmq.cu`,
`mmq.cuh`, `mmq-load-tiles.cuh`, `ggml-cuda.cu` (VDR/J selection), `tests/test-backend-ops.cpp`.

**Deps.** 08 (MMB sets the J/type policy; keep adjacent).

**Gates.** Stock-relative `llama-batched-bench -npl 1,4,8`; the band-uniform nwarps/VDR rule
(`GREEDY-PURITY.md` §§19/24/25); `test-backend-ops` MUL_MAT; `MUL_MAT` q1_0 1304/1304.

---

## 10 — Fused prefill kernels (conv1d, RMS-norm, top-k MoE, verify fusions)

**Concern.** GDN/PLE conv1d fusions, narrow-row RMS-norm fusion, `norm-gated`, the fused MoE router
(`topk-moe`), the `iq4_nl` `GET_ROWS` fallback, the fused MoE weighted reduction, and the PR #63
verify-band fusions (`ssm_gate_beta` ncols, `add_rms_norm_q8_1`).

**Sources.** old-08:fusions (mmb-General 0014/0015 + the 09-13 amendments + PR #63); old-15:PR63;
old-14:deferred `GGML_CUDA_DISABLE_CONV_FUSION` exact switches (folded into 08's files per old-15
body). The qwen4exp fusion stand-downs stay in 16/17.

**Owner files.** `ggml/src/ggml-cuda/gdn-conv.{cu,cuh}`, `ple-conv.{cu,cuh}`, `norm.cu`/`norm.cuh`,
`norm-gated.{cu,cuh}`, `unary.{cu,cuh}`, `topk-moe.cu`, `moe-weighted-reduction.{cu,cuh}`,
`getrows.cu`, `argsort.cu`, `convert.cu`, `dequantize.cuh`, `ggml/src/ggml-cpu/ops.cpp`,
`ggml-cuda.cu` (try_fuse arms for these), `tests/test-backend-ops.cpp`.

**Deps.** 08, 09.

**Gates.** Same-seed fusion on/off; `test-backend-ops` fusion cases; MTP gate; the CPU reference where
one exists.

---

## 11 — Fused MoE kernels (gate+up+GLU MMQ, item-split, shared-expert, MWR)

**Concern.** Fused MoE gate+up+GLU MMQ (prefill), mmvq short-K item-split + the ksplit regression fixes,
the multi-token `x_scale_channel_dst` fold, the band-uniform shared-expert down epilogue, the RDNA3_5 /
RDNA3_0 gate relaxations, `mul_mat_q_pair`, the MoE `MUL_MAT_ID` decode/verify dispatch, the routed
compact MMQ, the Strix MoE folds, the PR #114 decode fusions (if MoE-class).

**Sources.** old-13 (entire block); old-08:MMID/mmvq (the block-13-adjacent mmvq/topk changes);
old-15:MoE fusions (PR #114) if MoE-class.

**Owner files.** `ggml/src/ggml-cuda/ggml-cuda.cu` (try_fuse MoE arms), `mmq.cu`/`mmq.cuh`,
`mmq-vec-dot.cuh`, `mmq-config-rdna3-5.cuh`, `mmvq.cu`/`mmvq.cuh`, `quantize.{cu,cuh}`,
`moe-weighted-reduction.{cu,cuh}`, `common/common.cpp`, `src/llama-model-loader.cpp`,
`tests/test-backend-ops.cpp`, `generate_cu_files.py` + the MoE gate instance TUs.

**Deps.** 08, 09, 10 (the MoE fusion arm interleaves with `ggml_cuda_try_fuse` from 10).

**Gates.** `test-backend-ops` `MUL_MAT_ID`/`MUL_MAT_ID_FUSION`; same-seed fused on/off; MTP gate;
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` / `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`; the MoE byte-identity
oracles; the ROCm unaligned split-load fix (2-GPU load timing).

---

## 12 — Host-resident MoE: interface, scheduler, Meta and slab

**Concern.** The generic backend expert-cache interface (`moe_cache_update` / `_take_over` / `_gather`
in the backend iface), the scheduler routed-expert rebalance + merged per-layer MoE split, the Meta
delegation to per-device simple backends (incl. `stage_gather` guard), the movable-boundary slab and the
slab headroom/aux-reserve plumbing, the `GGML_ENV_STR` macro, the host-expert pinned `ROCm_Host`
loader/`--host-experts` policy, `moe_cache_rearm`, the `llama_kv_cache::seq_rm` used-range scan,
`GGML_CUDA_SLAB_HEADROOM_MIB` policy. (The `--load-mode none` Meta host_buffer/events fix and the r48
Meta upload path also live here, as they are the Meta-wrapper owner.)

**Sources.** old-06:iface/scheduler/meta/slab/loader (the non-engine, non-mmap bulk); old-09b (slab
headroom-at-least); old-15:r48 Meta upload/`stage_gather`; old-06-block16 host half.

**Owner files.** `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp`,
`ggml/src/ggml-backend-meta.cpp`, `ggml/include/ggml-backend.h`,
`ggml/src/ggml-cuda/ggml-cuda-vmm.h` (slab API), `ggml/src/ggml-cuda/ggml-cuda.cu` (slab/device alloc
region), `src/llama-model-loader.{cpp,h}`, `src/llama-model.{cpp,h}`, `src/llama-kv-cache.cpp`,
`src/llama-mmap.{cpp,h}`, `common/{arg,common}.{cpp,h}`, `common/fit.cpp`, `include/llama.h`,
`ggml/src/ggml-backend-meta.cpp`, `tools/llama-bench/llama-bench.cpp`,
`ggml/src/ggml-{cpu,rpc,...}/*.cpp` (iface NULL fills).

**Deps.** 00; **may** require 10/11 for the try_fuse types if a generic/engine helper references them —
keep this block free of `ggml_cuda_try_fuse` feature arms (that was the r50 failure). If a helper cannot
be made generic, move it to 13 instead.

**Gates.** Per-block `all` build is the critical gate; `GGML_CUDA_SLAB=0`, `--fit off`, `-sm tensor`
+ `-ncmoe` coherence; `MOE_EXPERT_CACHE_MIB` unset must be inert.

---

## 13 — CUDA decode expert-cache engine, consumer hooks and arena

**Concern.** The per-layer VRAM slot arena, LFRU admission, UVA cold reads, full-residency identity
fast path; the cache-aware fused gate+up+GLU/down folds, `MUL_MAT_ID` takeover, cache-band fusion
stand-down; the arena slot-count/layer-uniform re-size fixes; the MTP draft compute-buffer cap; the
wide-prefill compute-layout drop; the device gather + head-pad fix; the H2D staging ring.

**Sources.** old-06:engine (`moe-expert-cache.{cu,h}`); old-13:engine; old-15:MoE consumer hooks +
arena fixes + wide-prefill drop + r46/r49 host-pool/rearm remainder; old-06:r12/r26 H2D ring.

**Owner files.** `ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`, `ggml-cuda.cu` (cache-aware try_fuse
arms + iface assignment), `common.cuh` (`h2d_pin`/`h2d_scratch`), `mmvq.cu`/`mmq.cu` (cache-aware
kernels), `ggml-backend-meta.cpp` (`stage_gather`), `src/llama-context.cpp`, `src/llama-model-loader.cpp`,
`tests/test-backend-ops.cpp`.

**Deps.** 11, 12.

**Gates.** `MOE_EXPERT_CACHE_MIB` auto; `scripts/gate-qwen4exp-quant-coherence.sh`; `MUL_MAT_ID`
929/929; byte-identity `de8be4d0c90c`/`15038c19ddc8`; width purity; MTP n3; arena UB-tension gates
(`GREEDY-PURITY.md` §13/§14); `GGML_CUDA_Q8_1_ARENA_FREE_OLD`.

---

## 14 — CUDA/HIP graph handling for expert-offload decode

**Concern.** The prefill-vs-decode graph-skip token-count detection (MUL_MAT_ID `ne[2]` / weight
`src1->ne[1]`), the HIP `hipGraphExecUpdate` → destroy/re-instantiate recapture (ROCm kernarg leak),
the pool / FA-staging / H2D-ring graph-memory generation recapture, opt-in async graph-input uploads
(`GGML_CUDA_ASYNC_INPUT=1`) and the legacy-pool cache cap (`GGML_CUDA_POOL_CACHE_MB`).

**Sources.** old-11 (entire block); old-06:async inputs/pool cap; old-08:retired Q8_1 arena retention
(the graph-lifetime half; the arena itself lives in 13) and the graph recapture generation.

**Owner files.** `ggml/src/ggml-cuda/ggml-cuda.cu` (graph cache/recapture region),
`ggml/src/ggml-cuda/common.cuh` (graph-mem generation), `ggml/src/ggml-backend.cpp` (sched),
`ggml/src/ggml-cuda/ggml-cuda.cu` + `ggml-cuda-vmm.h` (pool), `tests`, `tools/llama-bench`.

**Deps.** 13 (the cache/arena frees must participate in the recapture generation).

**Gates.** `-ncmoe` decode graph replay count; greedy byte-identity (decode graph on/off);
`GGML_HIP_GRAPH_FORCE_UPDATE=1`; `GGML_CUDA_GRAPH_MEM_GEN=0`; the Q8_1 arena-free regression test.

---

## 15 — Host memory reclaim (mmap release)

**Concern.** The loader no longer prefetches the whole file; it `munmap`s each tensor's mapping range
right after its device upload, keeping host-pointer-backed ranges, with `LLAMA_LOAD_RELEASE_MMAP=0` as
the kill-switch; skipped under `--mlock`/`--check_tensors`/lazy.

**Sources.** old-06:r49 mmap reclaim (the entry is still listed under old-06 in the changelog);
old-15:r49 if the record places it there.

**Owner files.** `src/llama-model-loader.{cpp,h}`, `src/llama-mmap.{cpp,h}`,
`ggml/src/ggml-backend-meta.cpp` (only if the upload path must change), `common/arg.cpp` (flag).

**Deps.** 12 (loader ownership), 13 (host expert master interaction).

**Gates.** Peak `VmRSS` measurement (the 122.5 → 55.4 GiB case); same-seed + `-sm tensor` coherence;
`--load-mode none` unaffected; `--mlock`/`--check_tensors`/lazy skip.

---

## 16 — qwen4exp model support (arch, lazy reader, HC, MTP head)

**Concern.** `src/models/qwen4exp.cpp` architecture support, the managed lazy reader with PLE n-gram
rows and `LLAMA_LAZY_IO_THREADS`; the hyper-connection (HC) machinery (`hc-mix`, `hyperconn`,
`dsv4-hc`, `hc_combine` norms), the HC16 native-BF16 and hc_gate_mix arms, the BF16 hc_mix fusion; the
MTP draft-head/`t_h_nextn` export purity fix; the `llm_arch_supports_sm_tensor` gate and the
`GGML_USE_HIP` dynamic-backend macro; the qwen4exp prefill policy (qsa3 RDNA3_0 gating,
Lightning-Indexer registration, device-query arm).

**Sources.** old-14:model/HC/lazy/MTP-export + 2nd/7th/9th amendments; old-15:HC16/hc-mix/policy/qsa3
gating (the non-QSA-memory HC pieces).

**Owner files.** `src/models/qwen4exp.cpp`, `ggml/src/ggml-cuda/hc-mix.{cu,cuh}`,
`hyperconn.{cu,cuh}`, `dsv4-hc.cu`, `gdn-conv.cu` (if HC-shared), `src/llama-lazy-reader.{cpp,h}`,
`src/llama-arch.cpp`, `ggml/src/ggml-hip/CMakeLists.txt`, `tests/test-lazy-reader.cpp`,
`tests/test-llama-archs.cpp`, `src/llama-graph.cpp` (MTP export).

**Deps.** 04–11.

**Gates.** qwen4exp coherence (dense arm, sparse arm), `test-backend-ops HC_MIX` Q8_0+BF16,
`test-llama-archs`, `test-lazy-reader`, the MTP-export width probe (0 mismatches), prefill-logit.

---

## 17 — QSA sparse attention, indexer and QSA memory

**Concern.** The QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`), sparse flash attention
(`GGML_OP_FLASH_ATTN_QSA`) + CPU oracle, `qsa3` packed WMMA, the fused `INDEXER_POOL`/`SCORE` ops,
f16 indexer caches, derived block-vector cache, two-round 16-bit topk, `indexer-topk.cu` + the
K/V-head chunking fix, the decode/verify band-uniform QSA score fix, the `LLAMA_QSA_*` policy knobs,
and the QSA memory wins W1–W3 (score-chain relu+chunked concat, derived per-block bias/visibility with
the null guards, keys-only indexer cache).

**Sources.** old-14:QSA/indexer (fused top-k, sparse FA, QSA decode campaign, arch policy, quantized
QSA, band-uniform score, K/V-head chunk, configurable prefill arm); old-15:W1/W2/W3 + the
`LLAMA_QSA_SPARSE_FA=0` dense-arm V2/V3 shadowing fix + qsa3.

**Owner files.** `ggml/src/ggml-cuda/indexer-topk.{cu,cuh}`, `indexer-score.{cu,cuh}`,
`fattn-qsa.{cu,cuh}`, `lightning-indexer.cu`, `ggml-cuda.cu` (QSA/HC op dispatch),
`ggml.c`/`ggml.h` (ops), `ggml-cpu/ops.cpp` (CPU references), `src/models/qwen4exp.cpp` (QSA graph),
`src/llama-graph.{cpp,h}`, `tests/test-backend-ops.cpp`.

**Deps.** 05, 06, 11, 16.

**Gates.** `test-backend-ops -o FLASH_ATTN_QSA` 18/18+ and `LIGHTNING_INDEXER`/`TOPK_QSA`;
`LLAMA_QSA_OFF=1` plain == draft-mtp byte-identical; `LLAMA_QSA_SPARSE_FA=0` perplexity ratio vs dense;
QSA memory reserve matrix; width purity.

**Seam caveat.** The W1–W3 memory wins touch `src/models/qwen4exp.cpp` and `src/llama-graph.cpp`, both
also touched by 16's MTP-export/graph work. This is the strongest argument for merging 16+17 (Q1).

---

## 18 — Host hot-path flag caching (getenv)

**Concern.** Cache every debug/A-B `getenv()` read that sits on a per-op / per-graph / per-tensor path
(`LLAMA_HC_CN_DEBUG`, `GGML_CUDA_DISABLE_CONV_FUSION`, `GGML_CUDA_GCDBG`, `GGML_CUDA_OP_TIMING`,
`GGML_STREAMDBG`, `GGML_CUDA_MMB_MARK_LOG`, `GGML_META_*`, `GGML_CUDA_MMQ_J_MAX/_ROUTED`,
`GGML_Q6_COMPACT_J`, `GGML_CUDA_DISABLE_MMID_512`, `GGML_PAIR_2X`, `GGML_CUDA_GDN_CHUNKED(_BF16)`,
`GDN_DBG_*`, `GGML_CUDA_FA_WMMA_256/MAX_HEAD`, `GGML_CUDA_QSA_SLICES/IDENTITY`, `GGML_SCHED_*`,
`GGML_META_*`, `GGML_CPU_MOE_OFFLOAD_THREADS`), plus the clean-build warning fixes that ride with it.

**Sources.** old-15:env-caching amendment (issue #65) + the old-15 clean-build warnings.

**Owner files.** Every file with a hot-path `getenv` — this is why it is **last**. It is the only block
that is a pure cross-cutting sweep; if the maintainer objects to a sweep block, distribute each cached
flag to its owning block instead (Q4).

**Deps.** All blocks that introduce a cached flag (i.e. everything).

**Gates.** Same-seed coherence; clean build with zero warnings; Windows lock-cost rationale is a
correctness/perf claim only.
