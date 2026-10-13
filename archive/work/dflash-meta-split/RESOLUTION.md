# RESOLUTION — `dflash-meta-split`: clean rejection folded into block 11 (r54); real fix re-seeded

**Status: RESOLVED (2026-10-13), folded into delivery `v12-097cf33d9-r54` (block 11).**  Record:
`WORKLOG.md` 261013.004.  Real fix tracked by the new campaign
[`wip/dflash-meta-split-impl/`](../../../wip/dflash-meta-split-impl/README.md) (TODO #58).

## Root cause

A DFlash2 drafter (`-md`, `--spec-type draft-dflash`) under `-sm tensor` aborts during the draft
graph build at `ggml/src/ggml-backend-meta.cpp:707`:

```
GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

Instrumenting the `handle_per_row` guard named the op:

```
per-row op TOP_K on 'node_630' [16,2,1,1], src0='result_output' logits [248320,2,1,1], axis SPLIT_AXIS_0
```

Source: the DFlash2 selector `build_dflash2_selector()` (`src/models/dflash.cpp:517`, called at `:888`)
runs `ggml_top_k(ctx0, res->t_logits, top_k)` at `:538`.  Under `-sm tensor` the draft's logits
(`result_output`) are **vocab-split (axis 0)**; the meta backend's per-row handler cannot reduce an op
across a split axis, so it asserts.  `-sm layer` works because the logits are not vocab-split there.
It reproduces on r52 and r53 (and with `--fit off`), so it is pre-existing, not a PR #144 regression.

## Decision

The maintainer chose the **clean rejection** (over a meta-backend distributed top-k, which is a real
new feature).  The guard lives at the model-load point in `src/llama.cpp` (after `load_hparams`, where
`dflash_selector_rank` is known and before any graph is built):

```
LLAMA_SPLIT_MODE_TENSOR is not supported for a DFlash2 drafter with a selector:
its per-row TOP_K over the vocab-split draft logits has no tensor-split path -- use -sm layer for that config
```

## Validation

- `-sm tensor` repro: exit 1, clean message, **no abort** (was SIGABRT/134).
- `-sm layer` DFlash still loads and generates coherently.
- The dense 4B coherence gate is byte-identical to r53.

## What was NOT done (the re-seeded follow-up)

A real fix needs a Meta **distributed top-k** (or a split→mirrored gather) so `TOP_K` can reduce a
vocab-split input across devices.  That is the new WIP campaign; until it lands, `-sm layer` is the
supported DFlash2 tensor config.

## Pointers

- `archive/docs/HANDOVER-2026-10-13-wip-cleanup.md` (the four-campaign cleanup handover).
- `ggml/src/ggml-backend-meta.cpp` (`handle_per_row`, the `GGML_OP_TOP_K` case ~:1210).
- `src/models/dflash.cpp:517/538/888` (the selector).
- `src/llama.cpp` (`load_hparams` guard; block 11).
