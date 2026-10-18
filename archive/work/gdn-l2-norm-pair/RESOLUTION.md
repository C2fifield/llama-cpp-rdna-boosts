# RESOLUTION -- `gdn-l2-norm-pair` (PR #179, @briansp2020)

**Status: SHIPPED in `v12-097cf33d9-r81` (2026-10-17) as part of block 10.**  The contributor's
format-patch and write-up are preserved here unchanged.

## What landed (block 10)

- `rms_norm_scale_pair_f32<256>` (`norm.cu`): blocks `[0, nrows)` take the first `(x, dst)` pair, blocks
  `[nrows, 2*nrows)` the second; the per-row body is `rms_norm_f32<256, false, false, true>`'s, so each
  output is bit-identical to its own launch.
- `ggml_cuda_op_rms_norm_scale_pair_fused` (`ncols < 1024` only) and a matcher extension in the existing
  `{RMS_NORM, SCALE}` arm: only no-op nodes between the two pairs, both inputs F32 views of the same
  `view_src` with equal `ne` / `nb`, equal eps and scale params, and the second input must not read the
  first pair's output.
- Kill switch `GGML_CUDA_DISABLE_NORM_PAIR=1`.

## Validation here (gfx1201, 3x R9700, ROCm 10.1.0, r80 vs r80 + patch, tree `a297c909`)

- Same-seed greedy text byte-identical to r80 in all six arms (35B-A3B `sha ecb4e0796ba5`, dense 27B
  `sha 980719527364`; 3-GPU tensor, q8_0 KV, `none == draft-mtp n3`, with and without
  `GGML_CUDA_GRAPH_OPT=1`).
- `test-logits-width-probe` PASS, maxdiff **0**, hashes identical to r80 at P=256 and P=3584 on both models.
- Prefill-logit KLD **0.000000 / 100.000 %** against a base recorded from stock r80.
- `test-backend-ops -o RMS_NORM` 53/53, `-o SCALE` 7/7.
- Launch count (rocprofv3, 35B-A3B 1 GPU, 120-token decode): `rms_norm_f32<256,false,false,true>`
  **7500 -> 3750** `rms_norm_scale_pair_f32<256>`; total kernels 140131 -> 136381.
- Perf (2-GPU tensor, 35B-A3B Q4_K_M, q8_0 KV, three interleaved rounds): tg256 **114.49 / 114.48 / 114.41
  -> 115.65 / 115.80 / 115.85 t/s (+1.2 %, 3/3 rounds positive)**, MTP decode +1.0 %
  (178.3 / 178.1 / 177.3 -> 179.9 / 179.6 / 179.6), `pp31500` +0.3-0.6 % (noise),
  `llama-batched-bench -npl 1,4,8` flat.
- The contributor's 27B numbers (+0.2 % decode 1 GPU, +0.3 % 2-GPU tensor) are within this box's noise.

## Record

`WORKLOG.md` 261017.008 (r81); `patches/README.md` (r81 release note); `ENVIRONMENT.md` §7.2.
