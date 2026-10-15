# `moe-upload-overlap` — RESOLUTION (r63, 2026-10-14)

**Promoted.**  The campaign's real finding — `-ncmoe N` should host the **last N** trunk layers, not the
first N — is folded into the delivery as **r63** (`v12-097cf33d9-r63`, blocks 09/10 amended).  The
original premise (make the H2D expert upload overlap compute) was refuted; the pivots and the evidence
are in `FINDINGS-2026-10-14.md`, `HANDOVER-2026-10-14.md` and `FIX-2026-10-14.md`.

## What shipped

* **`-ncmoe N` = last N trunk layers** via a real `llama_model_params.n_cpu_moe_from_end` field
  (`common/arg.cpp`, `common/common.cpp`, `common/speculative.cpp`, `src/llama-model-loader.{h,cpp}`,
  `src/llama.cpp`, `src/llama-quant.cpp`, `tools/llama-bench`).  Kill switch
  `LLAMA_HOST_EXPERTS_FROM_END=0` restores the first-N placement.  Measured 35B-A3B Q8_0, 2×R9700,
  `-ncmoe 20`: `pp4096` 4264 vs 4120, `tg500` 87.5 vs 82.6, `pp512` 871 vs 635.
* **Block-09 cache/preflight/`-lm none` fixes** (the decode blocker): the MoE-cache preflight is now
  universal (`llama_model_load_from_file`), the deferred arena sizing waits for the full host-table set,
  and the Meta device takes whole-tensor uploads under `-lm none` (r54 regression).
* The scratch `LLAMA_NCMOE_FROM_END` / `LLAMA_NCMOE_LAYERS` knobs are removed.

## Gates

Coherence (35B front == tail, `e2dde701fc04`); `gate-prefill-logits.sh` PASS (0.000707 / 98.755 %);
`gate-qwen4exp-quant-coherence.sh` PASS (IQ3_XXS/IQ4_NL/Q4_K_M); adaptive-MTP `none == draft-mtp n3`
bit-identical on the 35B (`-n 3000`, +82 % prose / +49 % mixed); `llama-batched-bench -npl 1,4,8` tail ≥
front; r62 below-gate corruption repro clean; `validate-set.sh` green on a fresh tarball.

## Residuals → follow-up campaign

`wip/gemma4-drift-and-meta-async/`:

1. **Gemma4 26B-A4B front vs tail greedy text differs** (both coherent; flips when the expert cache is
   toggled → near-tie MoE reduction-order drift).  Accepted for r63; decide whether to chase it.
2. **`ggml_backend_meta_set_tensor_async`'s segmented mid-row fast path** is a latent hazard: the loader
   guard makes it unreachable from the loader, but any other caller sending a mid-row chunk can still
   assert (axis 0) / write a bogus tail (axis 1).  A candidate fix (require `size % row_stride == 0` or
   the axis-1 ≤512 tail) **hung the rear placement** and was reverted — needs a proper fix, not the
   heuristic.

## Artifacts

`tools/` keeps the scratch prototype (`rear-fill-prototype.patch`), the promotion patch
(`rear-fill-promotion.patch`, the r63 delta against r62), the isolated block-09 fix
(`block09-cache-sizing-and-preflight.patch`) and the instrumentation patches, for reference.
Nothing here is applied by `apply-all.sh`; the delivered form is in `patches/`.
