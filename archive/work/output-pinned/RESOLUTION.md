# RESOLUTION — `output-pinned` (PR #157, @briansp2020)

**Status: RESOLVED (2026-10-15), promoted into the delivery as part of `v12-097cf33d9-r65` (block 09
amended).**  No `wip/` entry; the PR's format-patch and write-up are preserved in this directory.

## What landed

`llama_context::output_reserve()` now uses the first simple device's pinned host buffer type when the
output device is a meta (tensor-split) device with no host buffer type of its own.  The output buffer is
never scheduled on, so its `.device` does not matter, and without this the per-step D2H reads of the
logits, embeddings and MTP hidden state land in pageable memory and each blocks the host.  Kill switch
**`LLAMA_OUTPUT_PINNED=0`** (default on).  The meta device's own `get_host_buffer_type` is unchanged.

## Validation (gfx1201, ROCm 7.14.1, r65 candidate)

Harness: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS + shared `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`,
`-sm tensor`, q8_0 KV, `-lm none`, `-ot per_layer_token_embd.*=CPU`,
`--spec-type draft-mtp --spec-draft-n-max 3`, `-n 3000`, prose prompt, 2 GPUs.  See
`../draft-sched2/RESOLUTION.md` for the full arm table.

| arm (2 GPUs) | Generation t/s |
|---|---:|
| r64 reference | 132.75 |
| this arm on, second scheduler off (`LLAMA_DRAFT_SCHED2=0`) | 137.8 |
| second scheduler on, this arm off (`LLAMA_OUTPUT_PINNED=0`) | 135.1 |

So the pinned output buffer is worth about **+2.7 %** on this harness (137.8 vs 135.1).  On dense
Qwen3.8-27B UD-Q4_K_XL `-sm tensor` 3-GPU MTP the equivalent split is 107.0 vs 102.2 t/s (**+4.7 %**).
Acceptance (0.81673) and the generated text (`5d51438c7a26`) are identical across all arms.

## Record

`WORKLOG.md` 261015.002 (r65).  Original PR description and patch: `README.md`, `0001-output-pinned.patch`.
