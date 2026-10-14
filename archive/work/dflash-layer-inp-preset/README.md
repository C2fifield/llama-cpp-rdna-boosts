# dflash-layer-inp-preset — DFlash: size the target's compute buffers for the layer-input extraction up front

**Resolved (r58, 2026-10-14).**  Contributor PR #150 (`@overdoingism`), folded into the delivery as
**block 02** (`common/common.cpp` draft-setup) + **block 11** (the `llama_context`/`llama-model`
consumption).  See [`RESOLUTION.md`](RESOLUTION.md); the upstream `git am` patch is preserved verbatim
as [`dflash-layer-inp-preset.patch`](dflash-layer-inp-preset.patch)
(`sha256 f243673750b94f916f88ccbbaa1fe50503d4fac5c0d3c535f6a45b82e4674c86`).

One `git am` file, made on `v12-097cf33d9-r56` (net tree `d718f97a…`; with the commit applied, tree
`a42fd651…`).  It touches `common/common.cpp`, `src/llama-context.cpp`, `src/llama-ext.h`,
`src/llama-model.cpp` and `src/llama-model.h`; nothing under `ggml/`.  Outputs are unchanged.

**The problem.**  A DFlash draft turns on `llama_set_embeddings_layer_inp()` for its target layers when
it attaches.  That happens after the target context was created, reserved and warmed up.  The flag
change asks for a reserve, so the first request re-reserves the graph.  `sched_reserve()` builds a new
scheduler, so every compute buffer is freed and allocated again.  The main one also grows: with
Qwen3.8-27B Q5_K_M + DFlash2, `-c 220000 -ub 1280`, it goes from 954.6 to 1064.6 MiB on the device, and
the 637.8 MiB host buffer is reallocated too.  Linux reuses the freed memory at once.  On Windows, with
the card nearly full, the new buffer landed in shared GPU memory and prefill fell to about 200 t/s.

**The change.**
- Before `common_init_from_params()` creates the target context, it reads the draft GGUF's
  `<arch>.target_layers`.  This runs only when a DFlash or DSpark draft is configured.
- It presets those layers on the target model with the new
  `llama_model_set_embeddings_layer_inp_preset()`.  The context applies them before its first reserve,
  and the preset is cleared again right after, so an MTP context made later from the same model does
  not inherit it.
- `set_embeddings_layer_inp()` returns early when the flag does not change.  The reserve that upstream's
  comment says is needed still runs whenever the flag actually changes.

The mechanism is upstream code; `097cf33d9` has the same `set_embeddings_layer_inp()` and
`sched_reserve()`.

*The patch and this note were written by Claude (AI), as reported by the contributor.*
