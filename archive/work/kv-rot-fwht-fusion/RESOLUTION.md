# RESOLUTION -- `kv-rot-fwht-fusion` (PR #172, @briansp2020)

**Status: RESOLVED (2026-10-17), both parts promoted into `v12-097cf33d9-r78` as block 10.**
The contributor's two format-patches and their write-up are preserved in this directory; nothing was
left behind as a `wip/` stub (the campaign arrived as a PR branch, never on `main`'s `wip/`).

## What landed (block 10)

- `0001`: a new `fwht_n` instantiation of the fused `rms_norm_mul_rope_f32` kernel.  With a quantized KV
  cache Q and K each ran `RMS_NORM, MUL, ROPE, RESHAPE, MUL_MAT(Hadamard)` = the fused rope launch plus
  a `fwht_cuda` launch.  The new variant stages the roped row in shared memory and runs the same in-warp
  transform, moved into `ggml_cuda_fwht_warp()` in `fwht.cuh` and shared with `fwht_cuda` (one helper in
  a header included before `rope.cu`'s `#pragma clang fp contract(off)`, so both compile the transform
  under the same contraction mode).  Matcher `ggml_cuda_should_fuse_rms_norm_mul_rope_fwht`; one rotation
  block per head (`n` = 64/128/256); the allocator's usual `dst == x` exact alias is allowed (each block
  reads its whole row of `x` before a barrier, then writes the same row); any other overlap rejects.
  Kill switch `GGML_CUDA_DISABLE_ROPE_FWHT` (presence-only).
- `0002`: `fwht_cuda<N, T, has_q8_1>` also writes the following mmvq matmul's Q8_1 activation into the
  quantize cache, with `quantize_q8_1`'s own reduction and rounding, so the output projection skips its
  `quantize_q8_1` launch.  The key is the mmvq launcher's (view root of `src1` = the Hadamard node, its
  shape/strides, `curr_stream_no`), the row length must be a multiple of `MATRIX_ROW_PADDING` (no padded
  blocks), and an already-cached entry stands the fusion down.  The F32 output is still written.
  Kill switch `GGML_CUDA_DISABLE_FWHT_Q8_1` (presence-only).

## Where it is reachable (important -- read before quoting the speedup)

The fusion needs `ROPE` mode **NORMAL/NEOX** with the `RMS_NORM, MUL, ROPE` chain immediately followed by
the rotation `MUL_MAT`, and `0001` needs `rope->ne[0] == n` (one rotation block per head).

- **Active on this box:** Gemma4 12B (`rope_neox`, qk-norm) for `0001` + `0002`; gpt-oss-20b for `0002`
  (it has no qk-norm, so there is no `RMS_NORM, MUL, ROPE` chain for `0001`).
- **Inert on the mrope models:** Qwen3.5-4B/9B/122B, Qwen3.6-35B-A3B and Qwen3.8-27B all run
  `rope_multi` (`GGML_ROPE_TYPE_MROPE`), which the shared rope matcher rejects.  A trace on each shows
  the unfused `rope_multi` + `fwht_cuda` pair.  So the +4-8 % the contributor measured on
  Qwen3-30B-A3B (a NEOX model) does not appear on the maintainer's Qwen3.x models.

## Validation (r78 candidate, gfx1201, 3x R9700, ROCm 7.14.1)

- **Text byte-identical to r77**: Gemma4 12B QAT UD-Q4_K_XL, 1 GPU layer split, q8_0 KV, 256-token
  greedy `sha 970325058470` (965 chars), also with `GGML_CUDA_GRAPH_OPT=1`; gpt-oss-20b MXFP4
  `sha a296beac7818` (1160 chars).
- `test-logits-width-probe` (layer, q8_0 KV): **PASS, worst maxdiff 0**, row-0 hashes identical to r77
  at P=512 (`ffc573dcc0c3878f`) and P=3584 (`72fb964721e643fb`) -- i.e. the W=1..8 verify band and the
  prefill path go through the fused kernel with the same bits.
- **Prefill-logit gate 0.000000 / 100.000 %** against a base recorded from the r77 build on the same
  Gemma4 12B (this is the instrument that sees a uniform prefill shift; the fused rope runs at prefill
  too).
- `test-backend-ops -o ROPE` 450/450, `-o MUL_MAT` 2444/2444, `-o SET_ROWS` / `RMS_NORM` / `ADD` pass.
- Speed (llama-bench `-p 512 -n 256 -r 5`, two alternating rounds): Gemma4 12B tg **60.15 -> 61.05 t/s
  (+1.5 %)**; `GGML_CUDA_DISABLE_ROPE_FWHT=1` 60.41, `GGML_CUDA_DISABLE_FWHT_Q8_1=1` 60.80 (the two
  parts are ~0.65 and ~0.25 t/s).  pp512 2975.8 -> 2970.3 (flat).  Dense 27B / MoE 35B-A3B unchanged
  (the fusion is inert there; their gain this release comes from PR #174).
- The contributor's own gfx1100 (WSL2) and gfx1201 (native) hashes, probe results and `test-backend-ops`
  runs were reproduced here.

## Review notes

- The `dst == x` alias is safe as argued (all reads of the row precede the barrier; the write is the same
  row), and the matcher only permits the **exact** alias.
- `el_w = fwht_n / warp_size` and the shared transform keep every reduction order of `fwht_cuda`; the
  `#pragma clang fp contract(off)` interaction was checked by ISA inspection in the PR and is covered
  here by the bit-identical probe and prefill-logit results.
- The `0002` cache key was checked against `ggml_cuda_mul_mat_vec_q`'s (view root, `ne10..ne13`, `nb`
  strides, stream) and the size formula is identical; `q8_1_cache_get` cannot return null, so the
  "entry created, must be filled" assert cannot fire on an allocation failure.
