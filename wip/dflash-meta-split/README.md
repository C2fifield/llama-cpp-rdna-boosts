# `dflash-meta-split` — `-sm tensor` + DFlash aborts in the Meta split-state build (TODO #58)

**Status: OPEN (seeded 2026-10-13), pre-existing correctness/limitation; needs a deterministic repro and
a root cause.**  Not part of the delivery; nothing here may be applied to the fork without the
maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).  Same item as **TODO #58**.

> **Cold start.**  Read this file, then `AGENTS.md` ("Report findings before shipping", "WIP and promotion
> rules", "Critical facts") and the DFlash bullets in `patches/README.md` and `ENVIRONMENT.md`.  This
> campaign was seeded while validating PR #144 (`archive/work/dflash-multiseq/`) on r53; the abort there
> is unrelated to that PR and reproduces on r52.

## 0. Symptom

A DFlash target plus its drafter (`-md`, `--spec-type draft-dflash`) under `-sm tensor` aborts during
model/context load:

```
E llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set
    (this warning is normal during memory fitting)
/home/.../ggml/src/ggml-backend-meta.cpp:707: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

Exit 134 (SIGABRT), core dump.  The first line is expected during `--fit`; the assert is the failure.

## 1. Reproduction

```bash
cd ~/llama.cpp
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
D=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1 ./build-rocm-hybrid/bin/llama-cli -m "$M" -md "$D" \
  --spec-type draft-dflash --spec-draft-n-max 3 -ngl 99 -sm tensor -c 16384 \
  -ctk q8_0 -ctv q8_0 -f prompts/prose-rdna-boosts.txt -n 512 --seed 42 --temp 0 \
  --reasoning off --no-display-prompt --single-turn
```

Confirmed 2026-10-13: **r52 (`2f3349306`) and r53 (`4b27ac644`) both abort**; `-sm layer` works
(the r53 DFlash gates used `-sm layer`).  `-sm tensor` was the mode in the older r26/r27 DFlash records,
so this is the current blocker for a multi-GPU tensor-split DFlash run.  Note the related, separately
documented case: a standalone `-md` MTP head under `-sm tensor` hits
`GGML_ASSERT(!suffix_fallback.empty())` at `src/llama-model.cpp:470` (see
`benchmarks/mtp-adaptive-methodology.md` "a draft head cannot be loaded standalone"); check whether both
are the same root cause before treating them as two items.

## 2. Current reality / hypotheses

- The draft context is a second `llama_context` over the same target model (`ctx_other`).  The Meta
  backend's split-state cache is built per graph; the assert rejects a split state whose first axis is
  `GGML_BACKEND_SPLIT_AXIS_0` (`ggml/src/ggml-backend-meta.cpp:707`).
- Prime suspects: the draft's graph contains a tensor/op the target graph does not (a repeat/view/rope
  chain over the drafter's layers), or the per-context split-state cache key (`GGML_META_SS_VERIFY`,
  the 8-version compact cache) reuses a target entry for a draft tensor with a different shape.
- The `--fit` path is a secondary suspect: the "dflash requires ctx_other" warning fires while fitting
  probes contexts without `ctx_other`, and the assert may be the fitting pass building a draft graph
  before the target context is set.  Try `-fit off` first (it did not help in the first attempt; record
  it).
- `GGML_CUDA_FA_KV_NATIVE` / the FA band arms differ for the draft (its head shape differs); rule them
  out or in with the existing kill switches.

## 3. Investigation plan

1. Get a deterministic small repro (prefer a single GPU + `-sm tensor`? — the assert is Meta-specific,
   so a 2-GPU `-sm tensor` is the minimum; a `-sm tensor` 2-GPU run with the smallest model pair that
   still selects DFlash).
2. Localise the offending tensor: add a temporary print of the tensor name/op/shape at
   `ggml-backend-meta.cpp:707`, or run under a debug build, and identify which op creates a
   `SPLIT_AXIS_0` split state on the draft graph.
3. Compare against the working `-sm layer` split state for the same tensors.
4. Decide: a Meta split-state cache fix (if the cache is keyed wrong), a graph/context fix (if the draft
   should not take the Meta path), or an explicit "unsupported, use `-sm layer`" documentation if the
   draft's separate graph cannot share the target's Meta split state.

## 4. Gates (when a fix exists)

- The repro above loads and generates coherently with `-sm tensor`.
- Single-sequence DFlash `-sm tensor` F1 on == off, and `--parallel 2` host vs F1 identical (the
  `archive/work/dflash-multiseq/` protocol).
- `test-backend-ops` unaffected; `LLAMA_QSA_*` / Meta split-state gates (`GGML_META_SS_VERIFY=1`) green.
- If promoted: `scripts/validate-set.sh` + the prefill-logit gate (a Meta/graph change).

## 5. Pointers

- TODO #58.
- `ggml/src/ggml-backend-meta.cpp:707` (the assert) and the split-state cache.
- `src/llama-model.cpp:470` (`suffix_fallback`; the standalone-MTP-head case).
- `archive/work/dflash-multiseq/` (the r53 PR #144 fold and its validation; it used `-sm layer`).
- `benchmarks/mtp-adaptive-methodology.md` ("a draft head cannot be loaded standalone").
