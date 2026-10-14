# RESOLUTION — `meta-split-copy-splice-race`

**PROMOTED (r62, 2026-10-14).**  The race is fixed in **block 10** and shipped as
`v12-097cf33d9-r62` (canonical tip `e909c529a`, net tree
`386a6a3d4e202d46fad21f9b63ee9a4aba3fb80e`).  See `WORKLOG.md` 261014.007; the delivery's release notes
are in `patches/README.md`, `MANIFESTS.md` and `BASELINE.md`.

## What shipped

`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_set_tensor_2d_async`: the compact splice-gather's
1-D H2D now runs on `cuda_ctx->stream()` (the main stream) instead of `cuda_ctx->copy_stream()`.  The
scheduler's `wait_before_overwrite()` — which guards the graph allocator's reuse of the split-input
buffer across splits — is enqueued on the main stream; the copy-stream H2D escaped it and could
overwrite the buffer while the previous split still read it.  Garbage/NaN for `-sm tensor` + `-ncmoe`
on the below-gate path (35B-A3B, gemma4 26B-A4B, Flash-Next GSQ).  `GGML_CUDA_SPLICE_GATHER=0` (plain
2-D, main stream) and `GGML_META_SPLIT_COPY=0` (mirrored) were always clean.

The pinned gather ring and the single 1-D H2D are unchanged; only the stream moves.  Perf-neutral
(35B-A3B `-sm tensor -ncmoe 20 -ub 2048`: 2487.9 pp / 73.35 tg vs plain 2-D 2495.3 / 73.36).

## Validation (recorded in detail in `README.md`)

35B-A3B `-sm tensor -ncmoe 20` PPL = 6.1802 == `-ncmoe 0`; gemma4 finite; coherence clean (32/250 tok)
on three models; dense 4B `1c5d32ac537d`; `gate-prefill-logits.sh` PASS; MTP `none == n3`
(`89951c23872e`); `validate-set.sh` green.  `test-backend-ops -o MUL_MAT_ID` has the same 4 pre-existing
upstream `f16 dup=1` fails as r53 (also on `build-r55-ref`).

## Residual

* The `MUL_MAT_ID` `dup=1` fails and the accompanying `test-backend-ops` spin are pre-existing (upstream)
  and unrelated — reproduce on `build-r55-ref`; they are already documented in `MANIFESTS.md`.
* This was found while parking `wip/moe-layer-stagger/`; that campaign is parked/refuted and unrelated.
