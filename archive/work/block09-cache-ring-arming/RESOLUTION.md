# `block09-cache-ring-arming` -- RESOLUTION (r67, 2026-10-15)

**Promoted.**  All three host-expert regressions and the ring-arming gap are fixed and shipped in
**r67** (`v12-097cf33d9-r67`, canonical tip `c536e9703`).  Because the block-10 refactor owns
`ggml-cuda.cu`, `ggml-cuda-slab.cu` and `mmid.cu`, and block 11 last touches `llama-context.*`, each fix
was folded into the last block that owns the file (**09** `moe-expert-cache.cu`; **10** `mmid.cu`,
`ggml-cuda-slab.cu`, `ggml-cuda.cu`; **11** `llama-context.{h,cpp}`).  No new block.

## What shipped

* **#1 cache completeness** (`moe-expert-cache.cu`, block 09): the deferred arena sizing compares the
  **total** registered host-expert bytes against the preflight total, not each device's even
  `bytes/n_dev` guess.  On the GSQ 40/60 split the per-device gate never latched, so the arena was never
  sized and every step streamed (GSQ `-ncmoe 48`: 14.8 -> 62 t/s).
* **#2 pageable weight source** (`ggml-cuda.cu`, block 10): `ggml_backend_cuda_buffer_set_tensor`
  bounces **any** pageable source (the `usage != WEIGHTS` exemption is gone), so the `-lm none` +
  `-sm tensor` Meta whole-tensor upload no longer issues a bare `cudaMemcpyAsync` from anonymous memory
  (the `SDMA0` fault, issue #137 class).
* **#3 ring arming** (`ggml-cuda-slab.cu` block 10, `llama-context.{h,cpp}` block 11): the narrow floor
  is pinned **before** the first prefill (not only at the prefill->decode drop), and a prefill ubatch
  (`n_tokens >= 64`) is placed **above** the ring even when its compute buffer is no larger than the
  narrow floor.  The 35B-A3B / GSQ layouts are size-independent, so the original size-only test left the
  carved ring unarmed.  A narrow->wide transition drops the reused narrow buffer so the prefill actually
  re-reserves; the width threshold is `< 64` to match the ring's own gate (GSQ's first prefill ubatch is
  42 tokens).  Proof: `WORK_ALLOC ... narrow=0 ret_off=narrow_floor+ring_region` on all three models and
  `ring_slot_capacity > 0`.  GSQ `-ncmoe 48` prefill 645 -> 687 t/s (r65 651).

The ring-arming fix surfaced a fourth, larger regression (not in the original §0 list): the **r66
MUL_MAT_ID duplicate-ids fix** had rewritten `mm_ids_helper` into a generic one-token-per-iteration
kernel (~20-23 % of the fully-resident 35B-A3B prefill).  That was root-caused and fixed in the same
release (block 10 `mmid.cu`); see `WORKLOG.md` 261015.005.

## DoD (gfx1201, 2x R9700, 2 GPUs, `-b/-ub 2048`, 31.5K prompt, 0 faults, ring armed)

| model | ncmoe | host | pp t/s | tg t/s | r65 pp |
|---|---|---|---|---|---|
| 35B-A3B Q8_0 | 20 | pinned | 2057 | 98.0 | 2052 |
| 35B-A3B Q8_0 | 20 | pool | 2057 | 97.8 | -- |
| GSQ-IQ3_XXS | 20 | pinned | 1102 | 61.9 | 1067 |
| GSQ-IQ3_XXS | 20 | pool | 1101 | 61.9 | -- |
| GSQ-IQ3_XXS | 48 | pinned | 687 | 57.8 | 651 |
| GSQ-IQ3_XXS | 48 | pool | 697 | 57.8 | -- |
| IQ4_NL | 36 | pinned | 611 | 42.3 / 49.2 | 583 |
| IQ4_NL | 36 | pool | 602 | 18.2 / 40.5 | -- |
| IQ4_NL field 70K | 48 | pinned | 615.8 | 32.4 | 434 |
| IQ4_NL field 70K | 48 | pool | 497 | 2.1 | -- |

Gates: `gate-prefill-logits.sh` PASS (0.000707 / 98.755 %), `gate-qwen4exp-quant-coherence.sh` PASS,
adaptive-MTP `none == draft-mtp n3` byte-identical on prose (`49eea5e81393`) and mixed
(`5571e0424738`), width purity `none == n1 == n3 == n7` (`7c709bb0c6fa`), `test-backend-ops -o MUL_MAT_ID`
947/947, `validate-set.sh` green.

## Residual

**IQ4_NL field (MTP n7) + `--host-experts pool`** decodes at 2.1 t/s against the handover's recorded
13.7, with the prefill fine (497), the ring armed and 0 faults.  The non-MTP `-ncmoe 36/48` pool decode
(18-40) matches the handover's ~2.4x pool penalty, so this reads as a pool **memory shortfall** on that
config.  Parked as `TODO.md` #65 (maintainer's call to debug later).

## Artifacts

The handover, the original working-tree diff
(`block09-cache-ring-arming-r66.patch`, sha256 `44ea933d...`) and the DoD harness/logs are preserved in
this directory (`README.md`, `tools/`, `logs/`).  The delivered form is `patches/0009..0011` in r67.
