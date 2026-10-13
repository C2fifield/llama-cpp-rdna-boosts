# `meta-segmented-expert-upload` — finish the segmented host-resident expert upload under `-sm tensor` (TODO, parked)

**Status: PARKED (seeded 2026-10-11).** Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the parked TODO
section *"Parked: finish the meta segmented expert upload (host-resident experts under `-sm tensor`)"*.

> **Cold start.** Read this file, then
> [`archive/work/moe-expert-cache/item3-findings-session20.md`](../../archive/work/moe-expert-cache/item3-findings-session20.md)
> and [`archive/work/issue-99/README.md`](../../archive/work/issue-99/README.md). Report findings to the
> maintainer before any delivery change and wait.

## 0. The item

Under `-sm tensor`, a **host-resident gemma4 expert table** (`-ncmoe`/`-cmoe`) is **hard-rejected**:

```cpp
// src/llama-model.cpp:406 (r48)
if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && arch == LLM_ARCH_GEMMA4 &&
    llm_params_have_host_expert_override(params)) {
    throw ... "the fused expert tensor's segmented per-ubatch upload has no tensor-split path
               -- use -sm layer for that config";
}
```

The reason is that gemma4's fused expert tensor (`ffn_gate_up_exps`) has a **segmented** split layout
(`n_segments`/`nr`), and `ggml_backend_meta_set_tensor_async` (`ggml-backend-meta.cpp:2274`) only handles a
**contiguous single-segment slice**; the MMQ tail pad that `copy_experts` appends makes the range
non-row-aligned, so the meta assert fires. An all-resident model, or one offloaded by `-ngl`/`--fit`
(whole layers on the CPU backend, no per-ubatch upload), is fine.

The remaining work is the **segmented per-ubatch upload**: make the async setter understand the segmented
layout so gemma4 + host experts + `-sm tensor` works (and the rejection at `llama-model.cpp:406` can be
lifted).

## 1. The partial fix that exists (and why it is not landed)

A partial fix was implemented and **hangs non-deterministically** (one GPU spinning):

1. Port `ggml_backend_meta_buffer_set_tensor`'s segmented handling into the **async** setter.
2. Copy the row-aligned part through the segments, and the ≤512-byte pad as a flat per-device range.
3. Use per-row **1-D async sets** instead of the 2-D pageable `hipMemcpy2DAsync` (which faults on this
   shape).

All of it is in `archive/work/moe-expert-cache/item3-findings-session20.md`; the issue-#99 gate change (the
`llm_arch_supports_sm_tensor` relaxation for gemma4) is in `archive/work/issue-99/README.md`.

## 2. What a future session must do first

1. **Get a deterministic reproducer.** The hang is non-deterministic, so the first deliverable is a repro
   that fails reliably — candidates in the TODO: `compute-sanitizer`, or the
   `GGML_CUDA_SPLICE_GATHER` / `GGML_META_PINHOST` force-modes. Without a deterministic repro, do not retry
   the fix.
2. Re-read the current `ggml_backend_meta_set_tensor_async` / `ggml_backend_meta_buffer_set_tensor` on r48:
   the **r48 block-06 change** (issue #137) added `host_buffer`/`events` proxying and an async upload
   pipeline, so the surrounding code moved. Check whether the split-state cache (`n_segments`/`nr`, r25's
   `GGML_META_SS_VERIFY=1`) now helps or constrains the fix.
3. Only then implement the segmented async upload.

## 3. Gates

- The target config (gemma4, `-ncmoe`/`-cmoe` > 0, `-sm tensor`) must start, load and produce coherent text
  — plus the standard coherence gate and the MoE-cache gates.
- Must not regress the cases that work today (all-resident `-sm tensor`, `-sm layer`, `--fit`-offloaded
  whole layers).
- `scripts/validate-set.sh` if promoted.

## 4. Pointers

- `TODO.md` → "Parked: finish the meta segmented expert upload".
- [`archive/work/moe-expert-cache/item3-findings-session20.md`](../../archive/work/moe-expert-cache/item3-findings-session20.md)
  (the `stage_input` measurement, the segmented-layout findings, and the partial fix).
- [`archive/work/issue-99/README.md`](../../archive/work/issue-99/README.md) (the `llm_arch_supports_sm_tensor`
  relaxation and the explicit rejection).
- `src/llama-model.cpp:357-415` (`llm_params_have_host_expert_override`, the throw),
  `ggml/src/ggml-backend-meta.cpp:1743` (`buffer_set_tensor`), `:2274` (`set_tensor_async`),
  `:3440` (iface), and the segmented split-state code (~566-1620).
- `WORKLOG.md` 2026-09-30 / r48 (`--load-mode none` meta event proxy) and `AGENTS.md` "Cross-device split
  inputs" for the meta-backend invariants.
