# RESOLUTION — allreduce-small-msg-latency (PR #129)

**Status:** BOTH FOLDED into the delivery as **block 16** on 2026-10-14 (release
`v16-a55e952b8-r45`).  **Owner:** `@briansp2020` (PR #129).

## What was adopted

| patch | what | switch |
|---|---|---|
| `0001` | decode/verify all-reduces (<= 64 KiB) launch `GGML_CUDA_AR_SMALL_BLOCKS` (default 4) instead of 8 blocks, so the call does not wait for the slowest of 8 self-signalling blocks; the fused stage and larger messages keep 8 | default-on; `GGML_CUDA_AR_SMALL_BLOCKS=8` restores; `GGML_CUDA_AR_SMALL_BYTES` tunes the threshold |
| `0002` | opt-in P2P push path: each device writes its wire data and arrival token into the peer's fine-grained VRAM and publishes the arrival with a system-scope release store, dropping the three all-thread `__threadfence_system()` calls (2 devices, not fused) | opt-in `GGML_CUDA_AR_P2P=1`; `GGML_CUDA_AR_P2P_LITE=0` keeps the full fences |

## Validation on the delivery tree (gfx1201, 2x R9700, Flash-Next GSQ IQ3_XXS + shared MTP head)

`--lazy-mode off --fit on -sm tensor`, q8_0 KV, MTP `n_max 3`, `-n 3000`, `prompts/prose-rdna-boosts.txt`,
identical text sha `2b2341670f23` in every arm:

- default: 130.1 / 130.1 t/s (`GGML_CUDA_AR_SMALL_BLOCKS=8`: 131.0 t/s — within noise here).
- `GGML_CUDA_AR_P2P=1`: 132.8 t/s, log confirms `P2P push all-reduce ... on`;
  `--spec-type none == draft-mtp` byte-identical.
- Prefill-logit KLD gate: mean **0.000707** / same-top-p **98.755 %** (PASS).
