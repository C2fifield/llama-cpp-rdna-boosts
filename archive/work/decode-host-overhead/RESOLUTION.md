# RESOLUTION — decode-host-overhead (PR #125)

**Status:** ALL THREE FOLDED into the delivery as **block 16** on 2026-10-14 (release
`v16-a55e952b8-r45`).  **Owner:** `@briansp2020` (PR #125).

## What was adopted

| patch | what | switch |
|---|---|---|
| `0001` | async graph-input uploads: `set_tensor` writes `<= 1 MiB` into a COMPUTE buffer stage through a per-device 16 MiB pinned ring, copied asynchronously with a done-event ordering; any setup failure falls back to synchronous | opt-in `GGML_CUDA_ASYNC_INPUT=1` |
| `0002` | `llama_kv_cache::seq_rm` scans only `[used_min(), used_max_p1())` instead of all `-c` cells | none (always on) |
| `0003` | `common_sampler` S2 exact top-k fast path: AVX2 block max + NaN prefilter, scalar fallback; `_WIN32` uses `IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)` (the `@DanoPTT` fix for the `__cpu_model` link error) | runtime AVX2 check |

## Validation on the delivery tree (gfx1201, 2x R9700, Flash-Next GSQ IQ3_XXS + shared Q8_0 MTP head)

`--lazy-mode off --fit on -sm tensor`, q8_0 KV, MTP `n_max 3`, `-n 3000`, `prompts/prose-rdna-boosts.txt`:

| arm | Generation t/s | text sha |
|---|---:|---|
| r44 baseline | 127.7 / 127.7 | `2b2341670f23` |
| r45 default (0002 + 0003) | 130.1 / 130.1 (+1.9 %) | `2b2341670f23` |
| r45 with `GGML_CUDA_ASYNC_INPUT=1` | 135.9 / 135.9 (+6.4 % over r44) | `2b2341670f23` |

The async arm was repeated 4 times (3-GPU and 2-GPU) with the same sha every time.
`--spec-type none == draft-mtp` byte-identical; prefill-logit KLD mean **0.000707** /
same-top-p **98.755 %** (PASS).  `0001` stays opt-in because of the historical single-run impurity
described in the handover; the hardened done-event ordering closes the window found and 4 repeat
runs were identical.
