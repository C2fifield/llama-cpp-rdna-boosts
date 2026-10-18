# RESOLUTION -- `mrope-rope-fusion` (PR #175, @briansp2020)

**Status: SHIPPED in `v12-097cf33d9-r81` (2026-10-17), both patches, as part of block 10.**  The
contributor's two format-patches and their write-up are preserved here unchanged.

## What landed (block 10)

- **0001**: `ggml_cuda_should_fuse_rms_norm_mul_rope` admits `GGML_ROPE_TYPE_MROPE` and
  `GGML_ROPE_TYPE_IMROPE` besides NORMAL / NEOX, and `rms_norm_mul_rope_f32` computes theta with
  `rope_multi`'s own section selection (the `sect_dims` / `sec_w` / `sector` arithmetic, the IMROPE `% 3`
  interleave, `pos[channel + nchannels*k]`, `powf(theta_scale, i0/2)`), rotated in NEOX pairing.  Every
  other matcher check is unchanged, including `n_offs == 0` (so `iw == i0`) and VISION rejected.
  Kill switch `GGML_CUDA_DISABLE_ROPE_MROPE_FUSE` (presence-only).
- **0002**: the 5-op `RMS_NORM, MUL, ROPE, VIEW, SET_ROWS` match declines a **BF16** `SET_ROWS`.
  `ggml_cuda_op_rms_norm_mul_rope_fused` writes F32 / F16 rows only and aborts otherwise, while
  `ggml_cuda_should_fuse_rope_set_rows` admits BF16.  `test-backend-ops` gains the BF16 set_rows cases.
- `test-backend-ops`: `GGML_ROPE_TYPE_MROPE` added to the `RMS_NORM_MUL_ROPE` mode list (72 new cases).

This closes the TODO item recorded against PR #172 (the MROPE gap: #172's fusion was inert on every
Qwen3.x model because they all run `rope_multi`).

## Why 0002 matters beyond the feature

It is a **latent abort fix**, not only a guard for the new code path:

- with 0001 and without 0002, the 27B Q4_K_XL with `-ctk bf16 -ctv bf16` aborts at `rope.cu:1087` on this
  box (confirmed: `GGML_ASSERT` -> core dump),
- with 0001 + 0002 the same run is clean and **byte-identical** to stock r80 (`sha 3681f80ee0bf`, 881
  chars, 200 tokens),
- the stock-r80 half of the claim (a NEOX qk-norm model such as Qwen3-30B-A3B with `-ctk bf16`) could
  **not** be reproduced here: that model is not in this box's set, and Gemma4 12B QAT and gpt-oss-20b with
  bf16 KV are clean on stock r80.  The guard is taken anyway: it is two lines, it is the documented
  behaviour of the fused op, and the reachable half is proven.

## Validation here (gfx1201, 3x R9700, ROCm 10.1.0, r80 vs r80 + both patches, tree `955009dd`)

- Same-seed greedy text byte-identical to r80 in all six arms (35B-A3B Q4_K_M `sha ecb4e0796ba5`, dense 27B
  UD-Q4_K_XL `sha 980719527364`; 3-GPU tensor, q8_0 KV, `none == draft-mtp n3`, each with and without
  `GGML_CUDA_GRAPH_OPT=1`), plus bf16 KV on the dense model.
- `test-logits-width-probe` PASS, maxdiff **0**, hashes identical to r80 at P=256 and P=3584 on both models.
- Prefill-logit KLD **0.000000 / 100.000 %** against a base recorded from stock r80 (27B Q8_0).
- `test-backend-ops -o RMS_NORM_MUL_ROPE` **298/298** (includes the new MROPE cases), `-o ROPE` 450/450.
- Launches gone (rocprofv3, 35B-A3B 1 GPU, 120-token decode): `rope_multi` 2500 -> 40, `fwht_cuda<256>`
  2500 -> 40, `rms_norm_f32<256,true,false,false>` 6120 -> 3660, replaced by 2460
  `rms_norm_mul_rope_f32<256,false,float,256>`.
- Perf (2-GPU tensor, 35B-A3B): tg256 114.57 / 114.42 -> 115.89 / 115.85 (**+1.15 %**, two rounds), MTP
  decode +1.0 %.  Dense 27B +0.6 % as reported.

## Contributor numbers that did NOT reproduce

Their **2-GPU tensor 31.5k prefill +11.7 %** (5638 -> 6300 t/s) does not reproduce here: `llama-bench
-p 31500` on the same topology is **4470 -> 4469 t/s (flat)**, twice, and `pp512` is flat.  They said they
had not traced where it came from; on this box there is nothing to trace.  It does not affect the
decision (the decode win and the abort fix are independent of it), but the figure should not be quoted as
a delivery claim.

## Record

`WORKLOG.md` 261017.008 (r81); `patches/README.md` (r81 release note); `ENVIRONMENT.md` §7.3.
