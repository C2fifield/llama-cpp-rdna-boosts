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
  not be reproduced on the gfx1201 box: that model is not in its set, and Gemma4 12B QAT and gpt-oss-20b
  with bf16 KV are clean on stock r80.  **It was then reproduced on gfx1100** (`fingon`, RX 7900 XTX,
  ROCm 7.14, native Linux, stock r80 reference built by reverse-applying the four patches to the r81 tree;
  tree verified as `873d0335...`): Qwen3-Coder-30B-A3B Q4_K_M with `-ctk bf16 -ctv bf16` **aborts on stock
  r80 at `rope.cu:1050`** and **completes on r81** (128 tokens, and 136.0 t/s at 64 tokens), with the
  q8_0 arm byte-identical to r80.  So 0002 closes **two** aborts: the fused-kernel one at
  `rope.cu:1087`, which 0001 makes reachable on the Qwen3.x models, and the pre-existing 5-op-fusion one
  at `rope.cu:1050`, which a NEOX qk-norm model reaches on r80 without any of these patches.

## gfx1100 cross-check (`fingon`, RX 7900 XTX, ROCm 7.14, 1 GPU)

* Same-seed greedy text identical to stock r80 (300 tokens, q8_0 KV, layer) on Qwen3-Coder-30B-A3B
  (`f5af07dc6502`), Qwen3.6-35B-A3B Q4_K_M (`86b1b02d18a0`) and Qwen3.8-27B Q4_K_M (`3b949b9d64ea`).
* `test-logits-width-probe` PASS maxdiff 0 with identical W8 and row-0 hashes to r80 at P=256 and P=3584
  on the 30B-A3B and the 35B-A3B (30B-A3B P=256 `W8=a5097bc9ad1f13e1`, `row0=98a558d45f81e80a` in both).
* Speed (`llama-bench`, 2 rounds): 35B-A3B tg256 126.70 / 126.60 -> 128.80 / 128.88 (**+1.7 %**), dense
  27B tg256 41.69 / 41.68 -> 42.11 / 42.09 (**+1.0 %**), pp512 flat on both.

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

**Refinement (same day): our host link is PCIe 5.0 x4 per card, so read that as "not measurable here",
not as refuted.**  Each R9700 has a x16 device-side link but a **x4 @ 32 GT/s uplink to the host**
(`current_link_width` = 4 on `0000:01:00.0` / `04:00.0` / `07:00.0`), which is half of the contributor's
x8 host and a quarter of x16, so our multi-GPU prefill is link-bound: `pp31500` scales 3613.8 (1 GPU) ->
4481.5 (2 GPU tensor, +24 %) -> 4347.0 (3 GPU, -3 % vs 2 GPU), and `wip/three-gpu-performance/` shows the
same +17 % / +1 % shape on the field config.  Isolating the fusion with its own kill switch in the r81
binary (same build, so both arms carry the same link cost) gives:

| prefill, 35B-A3B q8_0 | r81 default | `GGML_CUDA_DISABLE_ROPE_MROPE_FUSE=1` | delta |
|---|---:|---:|---:|
| 1 GPU layer, pp31500 | 3613.8 | 3579.1 | +0.97 % |
| 2 GPU tensor, pp31500 | 4481.5 | 4471.4 | +0.23 % |
| 3 GPU tensor, pp31500 | 4347.0 | 4334.3 | +0.29 % |
| 2 GPU tensor, real 31.5k prompt, `-ub 2048` | 5258.5 / 5247.4 | 5241.2 / 5237.9 | +0.2-0.3 % |

So the fusion is worth ~+1 % of prefill on one card and ~+0.25 % on a tensor split here, which is the same
order as the decode win and does not support an +11.7 % reading on any link.  The decode numbers, which
do carry over because the per-token all-reduce traffic is negligible, are the ones to trust from this PR.


## Record

`WORKLOG.md` 261017.008 (r81); `patches/README.md` (r81 release note); `ENVIRONMENT.md` §7.3.
