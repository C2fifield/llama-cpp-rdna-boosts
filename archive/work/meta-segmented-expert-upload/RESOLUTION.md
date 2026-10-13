# RESOLUTION — `meta-segmented-expert-upload`: segmented per-op upload fixed, rejection lifted (r54)

**Status: RESOLVED (2026-10-13), folded into delivery `v12-097cf33d9-r54` (block 09).**  Record:
`WORKLOG.md` 261013.004.  The performance follow-up (async segmented upload) is re-seeded as
[`wip/meta-segmented-upload-async/`](../../../wip/meta-segmented-upload-async/README.md).

## What was wrong

gemma4 + host-resident experts (`-ncmoe`/`-cmoe` > 0) + `-sm tensor` was hard-rejected at
`src/llama-model.cpp` ("the fused expert tensor's segmented per-ubatch upload has no tensor-split
path").  Lifting the rejection exposed two deterministic bugs (the campaign had been blocked on a
non-deterministic hang from an earlier partial fix; neither bug hangs):

1. **`ggml_backend_meta_set_tensor_async`** (`ggml/src/ggml-backend-meta.cpp`).  For a
   segmented tensor (`n_segments != 1 || nr[0] != 1`) the *whole*-tensor accumulate fallback (adapted
   from upstream PR #22466, written for the loader's chunked upload) accumulates into `ctx->async_set`
   and only flushes when `offset + size == total`.  The scheduler's per-op upload of a **pruned expert
   group** (`copy_experts`) is a partial range that never reaches `total`, so the fused
   `ffn_gate_up_exps` experts were **silently dropped** — coherent-looking but wrong decode (measured
   `<unused32>` garbage).  A **zero-element** tensor also returned `UNKNOWN` from the split-state
   computation and hit the `default:` abort.
2. **`ggml_backend_meta_buffer_set_tensor`** (the sync setter), axis-1 segmented branch.  It asserted
   `size % row_stride == 0`, but `copy_experts` appends `min(expert_size, 512)` bytes of the next
   expert so the MMQ padding is finite — the **tail pad** breaks row alignment.

An earlier experiment (skipping the cache's forced op-offload for the fused table) was rejected: the
forced offload is required for graph split-state consistency (`unsupported mul_mat split states`).

## The fix

- `set_tensor_async`: a `size == 0` upload is a no-op; a **row-aligned** segmented range (the
  scheduler's per-op upload, possibly with the tail pad) is handed to the sync setter; the mid-row
  accumulate fallback is kept for the loader's pinned async-upload chunks.
- `set_tensor` (sync), axis-1 segmented branch: after the row-aligned splice, the tail
  (`size % row_stride`) is placed flat at each device's next expert slot (the exact bytes are
  irrelevant — only that they are finite).
- The gemma4 rejection (and its `llm_params_have_host_expert_override` helper) is **removed** from
  `src/llama-model.cpp`.

## Validation (gfx1201, 2x R9700)

- gemma4 26B-A4B + `-ncmoe 40 -sm tensor`, **default cache-on**: loads and generates coherently
  ("…is **Paris**."), no asserts (was: SIGABRT / clean rejection).
- **cache-on output == cache-off output byte-identical** (the expert cache does not change numerics).
- `-sm layer -ncmoe 40` and all-resident `-sm tensor -ngl 99` unchanged/coherent.
- `test-backend-ops -o MUL_MAT_ID`: no regression — the same 4 pre-existing `dup=1` fails on both the
  untouched r53 baseline and the fix (the documented upstream duplicate-ids case).
- `scripts/validate-set.sh` green (strict 12/12 apply on a fresh tarball, new tree `71f524a5…`).

## Known cost (re-seeded follow-up)

The per-op segmented upload now goes through the **synchronous** setter, so that config reaches only
≈**9 t/s** prompt and gen (cache-off ≈31–45; `-sm layer`/all-resident unaffected).  The blast radius
is exactly: **gemma4 MoE + `-ncmoe`/`-cmoe` + `-sm tensor`** — a config that previously aborted or was
rejected, so no working config regresses.  The async/per-row-1-D-set optimization is the new campaign.

## Pointers

- `archive/docs/HANDOVER-2026-10-13-wip-cleanup.md`.
- `archive/work/moe-expert-cache/item3-findings-session20.md` (the earlier partial fix + hang).
- `archive/work/issue-99/README.md` (the rejection's origin).
- `src/llama-model.cpp`, `ggml/src/ggml-backend-meta.cpp` (`set_tensor`, `set_tensor_async`).
