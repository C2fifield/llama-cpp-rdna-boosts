# hc-mix-bf16-16: fused BF16 hyper-connection mixer and combine up to 16 tokens (bit-identical)

One format-patch on top of `v12-097cf33d9-r52`, for `git am`.  It touches `ggml/src/ggml-cuda/hc-mix.cu` and
`src/models/qwen4exp.cpp`.

## Why

Several concurrent speculative streams verify more than 8 tokens per step: 4 users x (1 + 3 drafts) = 16.  Above 8
tokens (`HC_FUSED_MAX_TOKENS`) the BF16 mixer falls back to the unfused chain.  There the generic mmvf costs about
118 us per mixer, against about 41 us for the fused kernel at 8 tokens.

## The patch

- A new `HC_BF16_MAX_TOKENS = 16` gates the BF16 fused mixer and the fused combine, in both the CUDA op and
  `qwen4exp.cpp`.  The combine uses it only for a BF16-mixer model (`hc_head_down` is BF16).
- The launcher runs a 9-16-token batch as chunks of at most 8 tokens through the existing kernels.  Every token is
  computed independently (its own `xn` / `lo` / output rows), so a chunk gives the same bits as one launch.  We also
  tried a single 16-token instantiation; it was slower (~94 us for the down/inject kernel).
- The Q8_0 variant keeps 8.

## Greedy purity

The fused kernels replay the unfused BF16 chain's per-token arithmetic, so widening the band doesn't change any
output.  We checked prompts of exactly 4, 8, 9 through 16, 17 and 24 tokens: output and first-token probabilities
match the build without the patch.  This covers the case where a 9-16-token prompt chunk now takes the fused path.

## Measurements (2 x R9700 32 GB, gfx1201)

Flash-Next IQ3_XXS + MTP (n-max 3), all experts in VRAM, `-sm tensor`, no tuning env, 600-token greedy chats
(aggregate t/s), one run each:

| users | r52 | r52 + this patch |
|---:|---:|---:|
| 1 | 134.3 | 131.9 |
| 3 | 165.5 | 178.3 (+7.7 %) |
| 4 | 144.8 | 150.6 (+4.0 %) |
| 6 | 129.6 | 132.2 (+2.0 %) |

Greedy sha `a7a81f01b3bc` in both.  (The concurrent runs' output shas change from run to run in either build, because
the batch mix above 8 tokens depends on timing, so they aren't comparable.)

On our earlier local build (mean of 3): 3 users 163.9 -> 169.3, 4 users 160.8 -> 168.6, 6 users 146.9 -> 158.6.  1-2
users are unchanged (their verify batches are at most 8 tokens).

## Note

The combine's gate looks only at the head weight type, while the mixer's also honours `LLAMA_HC_MIX_BF16=0` and a
host-resident hc weight.  In those two debug setups a 9-16-token batch fuses the combine while the mixer runs the
chain; the fused combine matches the chain, so the output is the same.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd prefer it done differently.
