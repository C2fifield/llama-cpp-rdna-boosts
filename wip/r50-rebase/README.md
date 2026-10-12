# r50 re-base onto upstream `097cf33d9` — in-flight handover

**Status: blocks 00–14 resolved; block 15 has 24 conflicts mapped but not applied. No build yet.**
The delivery repo (`patches/`, `release.json`, `rdna-boosts-all.patch`) is **untouched**. All work is in
the `~/llama-rebase` worktree of `~/llama.cpp`.

- new base: `097cf33d9` (`model: support vllm decision 2 model`, 2026-10-12; +195 commits over `a55e952b8`)
- old canonical tip: `f0a74f4cc` (tree `45ba5da24f872c5f9aa1b70b85470b5eb4d98395`)
- rebase is paused mid-block-15: `cd ~/llama-rebase && git rebase --continue`
- durable backups (branches in `~/llama.cpp`): `wip-r50-blocks-12`, `wip-r50-blocks-14`

## Key upstream collisions (why this rebase is hard)

| upstream | effect | decision |
|---|---|---|
| `d6cf9acb2` #29887 + `c811cb8f0` #30112 | new `src/llama-moe-cache.{cpp,h}`, opt-in `--moe-cache-mib`, hooks `llama_context::sched_copy_experts` | keep present but **dormant**; our `MOE_EXPERT_CACHE_MIB` path stays default. Do not delete upstream's file. |
| `6753a033f` #29943 | moved selective expert copying out of `ggml-backend.cpp` into `llama_context::sched_copy_experts` (a `callback_copy`) | block 06 keeps its own scheduler copy path (staging + `moe_cache_*` + pruning) and **does not call `callback_copy`**, so upstream's cache never runs. |
| `fc9ce6b9d` #29953 | removed `ggml_cuda_mmq_get_J_max`; `ncols_opt` → `J_best` chosen in `ggml_cuda_mul_mat_q`; `needs_fallback` | adopted. Our J selection moved into a shared `ggml_cuda_mmq_select_J` helper (mmq.cu); `GGML_CUDA_MMQ_J_MAX` is a cap in it. |
| `1bb2b9fcb` #30103 | FA `__syncthreads` when `nbatch_combine != DKQ/2` | **union** with our `k00 + nbatch_combine < DV/2` (neither subsumes the other). |
| `050439614` #14891 | added `params.load_mtp` | block 01 → `mparams.load_mtp = params.load_mtp || params.speculative.has_mtp();` |
| `9d3aba6b5` #29633 | `ggml_cuda_should_use_mmvf` gained `warp_size` | our MMB/verify-clamp call passes `warp_size`. |

## Block-by-block resolution log (blocks 00–14, applied)

- **00,02,03,05,07,09,10,11,12** — clean.
- **01** — `common/common.cpp`: `params.load_mtp || params.speculative.has_mtp()`.
- **04** — `fattn-mma-f16.cuh` sync union; `mmq.cuh` drop `get_J_max`; `GGML_CUDA_MMQ_J_MAX`
  re-expressed in upstream's `J_best` loop (later folded into `ggml_cuda_mmq_select_J`).
- **06** — `common/common.cpp`-style `cparams` union in `llama-context.cpp` (keep
  `moe_cache_size` + `drop_compute_buffers`). `ggml-backend.cpp`: adopted upstream's
  `ggml_backend_sched_copy_input` helper + two-pass loop and **moved our full inline copy logic into
  the helper** (staging consumption, `stage_input`, tripwire, `wait_before_overwrite`, single
  MUL_MAT_ID-consumer discovery, `moe_cache_take_over`/`gather`/`update`, host→device
  `set_tensor_async`, cross-device event fix). Added `ggml_backend_sched_copy_state` (ids readback
  cache) and `struct ggml_backend_sched_copy_state st;` in `compute_splits`. `callback_copy` is
  deliberately not called.
- **08** — `ggml-cuda.cu`: our MMB/verify `ne11_mmvf` clamp + upstream `warp_size`; kept both
  upstream's `ggml_cuda_try_ssm_scan_cache_fusion` and our `ggml_cuda_topk_moe_fusion_disabled`.
- **13** (the big one) — `mmq.cuh`: `mmq_args` = upstream `int J_best` + our gate fields
  (`x_gate`/`glu_op`/`glu_limit`); added `has_gate` template param to
  `mul_mat_q_process_tile`/`mul_mat_q`/`launch_mul_mat_q`/`mul_mat_q_switch_J`/`mul_mat_q_case`;
  `switch_J` now switches on `args.J_best`; routed-compact kernel gained `prec_src1`.
  `mmq.cu`: added `ggml_cuda_mmq_select_J` (block 04 `J_MAX` cap + block 13 RDNA3.5 routed widths +
  fused-gate `J_max_gate`), called from `ggml_cuda_mul_mat_q` and from both branches of
  `ggml_cuda_mul_mat_q_pair` (replacing the dead `get_J_max` sizing with `J_best` +
  `nthreads_best`-derived padding, and setting `args.J_best`). `mmvq.cu`: merged the band wrapper +
  upstream's PASCAL arch. `ggml-cuda.cu`: kept both the SSM and swiglu fusions.
- **14** — `ggml-backend-meta.cpp`: host-tensor check (ours) **and** upstream `[TAG_META_HOST_VIEWS]`
  check. `ggml-cuda.cu`: our `MMVF_MAX_BATCH_SIZE_FLAT` + upstream `warp_size`; took upstream's
  `MOE_WEIGHTED_REDUCTION` matcher. `mmq.cu`/`mmq.cuh`: kept block-13 `J_best` resolutions.
  `mmvq.cu`: `get_mmvq_mmid_max_batch_impl` rename + upstream PASCAL; keep the 2-arg `band` call.
  `tests/test-llama-archs.cpp`: `get_model_and_ctx(..., tensor_buft_overrides=nullptr,
  moe_cache_size=0, lazy_buf_size=0)` and the call passes
  `overrides, dc.moe_cache_size, qwen4exp ? 256<<10 : 0`.
  **`src/models/qwen4exp.cpp`: our canonical block-14 file (`a0204158c`) + the upstream U0→U1 delta's
  non-kpool hunks** (adopted `nextn_flags`, the `#30160` ple_emb/embd adjacency, the
  `llm_graph_input_embd_h(n_embd_inp, n_embd_out)` + raw-token (`ubatch.token`) change, and the PLE
  `is_mixed()` change; kept our QSA/`mtp_export_defer` design and skipped upstream's kpool/`cache_safe`
  hunks, which our block 14 removed). The upstream delta patch is at `/tmp/qwen_up.patch`
  (`git -C ~/llama.cpp diff a55e952b8 master -- src/models/qwen4exp.cpp`).

## Block 15 — remaining conflicts (9 files, 24 hunks) and how to resolve

Most are "keep both": upstream added a parallel feature at the same spot. Resolutions already decided
from inspection:

- `ggml-cuda/mmq.cuh:374` — take **HEAD** (drop our `get_J_max`; same as block 04). *1529* — take
  **ours + HEAD** (our `mmq_routed_src` struct before upstream's `launch_mul_mat_q` template). *1841* —
  take **HEAD** (`switch (args.J_best)`; discard the old J computation block).
- `ggml-cuda/fattn-common.cuh:1869` — merge the `launch_fattn` tail params:
  `const int kv_native_kernel = FATTN_KV_NATIVE_PER_OPERAND, const int warp_size = WARP_SIZE, const bool async_kv_preload = false`.
- `ggml-cuda/fattn-mma-f16.cuh` — 366 take **ours** (bank-aligned/swizzled helpers); 446 merge the
  `load_tile` template (add `swz`); 731, 956, 988, 1331, 1353, 1729 take **ours** (native-KV staging /
  issue #47 typed store); 2639 merge the `launch_fattn` call to pass
  `FATTN_KV_NATIVE_PER_OPERAND, warp_size_host, async_kv_preload`.
- `ggml-cuda/ggml-cuda.cu:5461` — take **ours** (block 15's `ggml_cuda_gdn_state_gather_deferrable`);
  upstream's `ssm_scan` fusion is already present from block 08.
- `ggml-cuda/lightning-indexer.cu:721` — take **ours** (scoped `{ constexpr ... }` block + comment).
- `ggml-cuda/unary.cu:188` — merge: upstream's contiguous/strided dispatch **with** our
  `dst16`/`store_f32` MMB handling threaded into both `unary_cuda` and `unary_cuda_strided`.
- `ggml-cuda/norm.cu` — 130/172 take **ours** (inline multiply/add pointwise loop; upstream's version
  is a different refactor of the same kernel); 479 merge the signature (ours adds
  `uint16_t * dst16 = nullptr, bool store_f32 = true`, and note ours changed `blocks_num` to
  `(nrows, nchannels, nsamples)` — verify which grid upstream intends).
- `src/llama-context.cpp` — 930 take **ours** (`kq_mask_packed_reachable`); 1048 take **ours**
  (`sched.reset(ggml_backend_sched_new(...))` + `graph_reserve(..., packed_kq_mask)`); 4436 merge the
  `memory_breakdown` bodies (upstream's `moe_cache->memory_breakdown()` **and** our
  `fa_stages`/`f16_kv_bytes` accounting).
- `src/models/dflash.cpp` — 648/918 take **ours** (`ubatch.embd || cparams.lf_dev_inject` →
  `lf_dflash_target_features`), keeping the rest of upstream's graph input construction.

## After block 15 (do not skip)

1. `git rebase --continue`; confirm the branch tip tree, then **build the final tip** first
   (`~/bin/build-llama-rocm-714`), fix compile errors, then do the per-block fresh-configure `all`
   sweep (the r2 rule: every one of the 16 commits must build).
2. Gates: same-seed coherence, `-n 2000/3000` adaptive-MTP, `scripts/gate-prefill-logits.sh`,
   `scripts/gate-qwen4exp-quant-coherence.sh`, `test-backend-ops` subsets (`MUL_MAT_ID`,
   `FLASH_ATTN_EXT`, `HC_MIX`, `GATED_DELTA_NET`, `ARGSORT`, `INDEXER_TOPK`).
3. Re-cut the delivery: `scripts/make-patches.sh` with the new base/tip, `scripts/make-release.sh`,
   `scripts/validate-set.sh` green.
4. Update `patches/README.md` + `WORKLOG.md` (new dated entry, r50), then the fork refresh and local
   container build per the Pushing policy — only after maintainer sign-off.
5. Delete this `wip/r50-rebase/` dir once the release is cut (finished campaigns leave `wip/`).

## Gotchas found

- `git apply --reject` on the net patch over-reports (~84 hunks); the faithful rebase had only the
  conflicts above.
- Upstream's `ggml_cuda_mmq_get_config`/`get_nthreads`/`get_I`/`get_occupancy` lost their `prec_src1`
  default — any 3-arg call added by our patches must pass it.
- `mmq_args` aggregate initializers must now end with `J_best` (a missing trailing value silently
  zero-inits `J_best`, which aborts in `switch_J`).

---

## Session 2 update — rebase complete and building

- The rebase is **complete**: 16 commits on `097cf33d9`, tip `29d75f0970b69c76aea33e7377f6bf8030dcea20`,
  tree `9c6eeb3c4dccd774afa669de3cc3e86c875099c2`. Backups: `wip-r50-blocks-12`, `wip-r50-blocks-14`,
  `wip-r50-rebased-tip`.
- **Block 15 is resolved.** The merges, beyond the "keep both" cases:
  - `mmq.cuh` — dropped the duplicate `get_J_max`; kept the `mmq_routed_src` struct + upstream
    `launch_mul_mat_q` template (with `prec_src1`); `switch(args.J_best)`.
  - `mmq.cu`/`mmq-vec-dot.cuh` — added the missing `prec_src1`/`GGML_PREC_Q8` args to
    `get_nthreads`/`get_I`/`get_occupancy`/`get_stream_k` calls.
  - `fattn-mma-f16.cuh` — removed the duplicated `get_swizzled` (kept our `ncols2`-aware one) and the
    duplicated `load_tile` template; updated the `get_swizzled` call sites to `ncols1,ncols2`.
  - `norm.cu` — upstream's clamped-grid loop + our `dst16`/`store_f32` merged into one kernel.
  - `unary.cu` — upstream's contiguous/strided dispatch + our `dst16`/`store_f32`.
  - `ggml-backend.cpp` / `ggml-backend-meta.cpp` — fixed brace-count artifacts from the block-06/14
    merges (the `copy_input` helper's outer-else close, the two-pass loop, and the meta host-tensor
    branch body).
  - `llama-context.cpp` — restored upstream's `sched_new()` definition (the block-15 conflict had
    replaced it with `kq_mask_packed_reachable`), and used it in the retry path.
  - `ggml-cuda.cu` — restored upstream's `ggml_cuda_try_ssm_scan_cache_fusion` (dropped when the block-15
    conflict kept our `ggml_cuda_gdn_state_gather_deferrable`) and fixed the `should_use_mmvf` arity.
- **Build is green**: `~/bin/build-llama-rocm-714` configure, then `cmake --build build-r50 -j16` →
  `llama-cli`, `llama-server`, `llama`, `llama-app` all built (100%).
- Coherence smoke (4B Q8_0, 3-GPU `-sm tensor`, seed 42, temp 0): coherent text.

### OPEN regression — `test_mul_mat_id_dup` (new upstream test)

`test-backend-ops -o MUL_MAT_ID -p '^(?!.*dup=1).*'` is **green** (0 fails, no fault). The new
`test_mul_mat_id_dup` cases (added upstream by `966baae76`, the Vulkan dup fix; only Vulkan was fixed)
fail and then fault:

- `type_a=f16`, `n_mats=28, n_used=10, b=0, m=1024, n=9/16/33/64, k=256, dup=1` → `ERR ~1.5-1.9`.
- `q8_0` / `q4_0` `dup=1` cases pass.
- Later a `quantize_mmq_q8_1<...,false,false>` page fault (grid 42240).

Ownership is **not yet confirmed**: the F16 `MUL_MAT_ID` dup path is not touched by our MMQ patches
(f16 is not quantized), so this is plausibly upstream's CUDA/HIP dup gap. To confirm, build a plain
`097cf33d9` `test-backend-ops` and run the same cases. Do that before treating it as ours.

### Not yet done
Per-block fresh-configure `all` sweep, the full gate suite (coherence vs a known-good, `-n 2000/3000`
MTP, `scripts/gate-prefill-logits.sh`, `scripts/gate-qwen4exp-quant-coherence.sh`), `make-patches.sh` +
`make-release.sh` + `validate-set.sh`, and the fork/container publish (after maintainer sign-off).
