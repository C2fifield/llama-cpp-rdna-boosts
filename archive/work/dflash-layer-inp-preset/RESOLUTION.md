# RESOLUTION — dflash-layer-inp-preset (PR #150)

**Status:** FOLDED into the delivery, **no new block**, split by concern as **block 02**
(`common/common.cpp`) + **block 11** (`src/llama-context.cpp`, `src/llama-ext.h`,
`src/llama-model.{h,cpp}`), release **`v12-097cf33d9-r58`** (2026-10-14).  **Owner:** `@overdoingism`
(PR #150).  Upstream: <https://github.com/stew675/llama-cpp-rdna-boosts/pull/150>.

## What was adopted

`dflash-layer-inp-preset.patch` (`sha256 f243673750b94f916f88ccbbaa1fe50503d4fac5c0d3c535f6a45b82e4674c86`,
made on r56 and applying cleanly to the r57 tree):

- **block 02** (`common/common.cpp`): in `common_init_result`, before `llama_init_from_model()` creates
  the target context, read the configured draft GGUF's `<arch>.target_layers` (metadata-only
  `gguf_init_from_file`, `no_alloc`) and call `llama_model_set_embeddings_layer_inp_preset()`; clear it
  right after context creation so a later MTP context on the same model does not inherit it.  Gated on
  `params.speculative.has_dft()` plus a `draft-dflash`/`draft-dspark` type.
- **block 11** (`src/llama-context.cpp`, `src/llama-ext.h`, `src/llama-model.{h,cpp}`): the model-side
  preset vector + `llama_model_set_embeddings_layer_inp_preset()`; the `llama_context` ctor applies the
  preset before the first `sched_reserve()`; and `set_embeddings_layer_inp()` returns early when the
  flag does not change (a real change still sets `sched_need_reserve`, so the EAGLE3 path is unchanged).

Motivation: a DFlash draft turns the extraction on *after* the target context was reserved and warmed
up, so the first request re-ran `sched_reserve()`, freeing and reallocating every compute buffer (the
main one larger, since the extracted layer inputs become graph outputs).  On the reporter's near-full
Windows card the growth landed in shared GPU memory (~200 t/s prefill); Linux reclaims the freed block.

**Adaptation made while folding (behaviour-neutral):** the upstream
`llama_model_set_embeddings_layer_inp_preset()` used `vector.assign(lids, lids + n)`; the clear call
passes `nullptr, 0`, i.e. `assign(nullptr, nullptr)` — technically UB pointer arithmetic.  The folded
version guards `n == 0` -> `clear()`.

**Block-split note:** the `common/common.cpp` hunk lands in block 02 (the maintainer's choice, the
draft-setup concern) while the API it calls lands in block 11, so block 02 references a symbol declared
in block 11.  This is the same intermediate-block compile ordering the existing block 09 -> 11
`ggml_backend_buffer_type_i` members already have; the applied 12-block set builds and per-block
standalone builds were never a contract.

## Validation on the delivery tree (gfx1201, ROCm 7.14.1, 2x R9700)

- Configuration: `Qwen3.8-27B-UD-Q5_K_M` target + `Qwen3.8-27B-DFlash2-Q4_K_M` drafter,
  `--spec-type draft-dflash --spec-draft-n-max 3 -sm layer -c 16384 -ub 1280 -ctk q8_0 -ctv q8_0`,
  `prompts/prose-rdna-boosts.txt`.
- **5 -> 4 reserves.**  The target context's first reserve is already sized for the extraction
  (ROCm0/ROCm1/ROCm_Host `842.88/622.84/411.30` -> `897.88/705.34/411.30` MiB) and the first-request
  re-reserve is gone.
- **Text byte-identical** at `-n 512` (greedy, seed 42) and **draft acceptance identical** (62 drafted /
  42 accepted at `-n 64`).
- 4B `Qwen3.5-Q8_0` 3-GPU `-sm tensor` same-seed greedy **byte-identical** to r57.
- `scripts/gate-prefill-logits.sh` **PASS** (mean KLD `0.000707`, same-top-p `98.755 %`).
- `scripts/validate-set.sh` strict 12/12, tree `84c1089798ace45224456cfb147e6e6fb9fabaff`;
  `git am --whitespace=error-all` clean.
