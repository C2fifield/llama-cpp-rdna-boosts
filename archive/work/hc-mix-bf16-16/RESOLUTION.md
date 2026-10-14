# RESOLUTION — hc-mix-bf16-16 (PR #148)

**Status:** FOLDED into the delivery as **block 10** (`ggml/src/ggml-cuda/hc-mix.cu` +
`src/models/qwen4exp.cpp`, r56, 2026-10-13), **default-on**.  **Owner:** `@briansp2020` (PR #148).

## What was adopted

`0001-qwen4exp-hc-mix-fused-BF16-hyper-connection-mixer-an.patch` — the BF16 fused hyper-connection
mixer and the fused combine serve up to `HC_BF16_MAX_TOKENS = 16` (the Q8_0 mixer keeps 8).  The
launcher runs a 9-16-token batch as chunks of at most `HC_FUSED_MAX_TOKENS = 8` tokens through the
existing kernels; every token is computed independently (its own `xn` / `lo` / output rows, located by
the per-token strides), so a chunk is bit-identical to one launch.  Motivation: several concurrent
speculative streams verify 9-16 tokens per step (4 users x (1 + 3 drafts)), where the unfused chain
costs ~118 us per mixer against ~41 us fused.

## Validation on the delivery tree (gfx1201, 3x R9700, ROCm 7.14.1)

- `test-logits-width-probe` on Flash-Next GSQ-IQ3_XXS, `SPLIT=tensor`, `P=256 ub=512`: W=1..8 matrix
  **byte-identical to r55** (`width_purity=PASS`).
- An extended probe (NW=16, local validation edit, not shipped) shows the fused mixer+combine produce
  the **same per-width logits as the unfused chain** (`LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0`)
  at every width 1..16, including the newly fused 9-16 band.
- The same extended probe also confirms the cross-width divergence at W>=9 exists in **both** the fused
  and the unfused build: it is the pre-existing >8 width dependence the delivery only promises to
  `n_max` 7, not something this PR introduces.
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS); 4B coherence identical.

Perf: the contributor measured 3 / 4 / 6 concurrent 600-token greedy chats on Flash-Next IQ3_XXS +
MTP (`n-max 3`, all experts in VRAM, `-sm tensor`) at +7.7 % / +4.0 % / +2.0 % with an unchanged greedy
sha; 1-2 users are unchanged (their verify batches are at most 8 tokens).
