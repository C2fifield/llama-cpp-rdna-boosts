# RESOLUTION — `q8-quantize-fusion` (PR #163, @briansp2020)

**Status: RESOLVED (2026-10-16), promoted into the delivery as part of `v12-097cf33d9-r69` (block 10),
accepted as a no-regression launch-count cleanup.**  The PR's format-patch and write-up are preserved in
this directory; the new `wip/` copy was never left behind as a stub.

## What landed (block 10)

Three output-identical parts, each with its own A/B switch:

- `q8_1_cache_get` matches on the canonical row layout (`GGML_CUDA_Q8_1_CACHE_EXACT` restores the old
  exact-shape match).  The MUL_MAT_ID view `[K, 1, n]` and the MUL_MAT view `[K, n]` of one activation
  now share one quantization.
- qwen4exp builds the shared expert before the MoE so it shares the MoE segment's Q8_1
  (`LLAMA_SHEXP_FIRST=0` restores the old node order).
- `hc_mix_bf16_up_collapse2` writes the mixed rows' Q8_1 straight into the cache
  (`GGML_HC_MIX_Q8=0` restores the old path and row mapping).

## Validation (r69 candidate, gfx1201, 3x R9700)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `-sm tensor` + shared MTP Q8_0, q8_0 KV:

- **Output bit-identical** on/off: `bbb21c2fb19a` for the default, all-switches-off, and
  old-order + `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` arms.  `none == draft-mtp n3` also
  `bbb21c2fb19a`.
- **Speed**: on `108.0` vs off `108.0` t/s (means of 3 interleaved rounds).  The reporter's ~1 %
  single-request win did not reproduce on this box (3 GPUs, strong host), so this was taken as a
  launch-count reduction with no regression rather than a measured local win.
- The shared-expert down+gate fusion that the node reorder could split is **inert** on this model
  (output identical with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`), so the reorder's latent risk is not
  realized here; it remains documented for other qwen4exp models.
- Flash-Next `--spec-type none` aborted once with a transient VRAM-fit failure; it did not reproduce
  (the same config then ran clean), and is not related to this change.

## Record

`WORKLOG.md` 261016.001 (r69).  Original PR description and patch: `README.md`,
`0001-ggml-cuda-qwen4exp-fewer-quantize_q8_1-launches-in-s.patch`.
