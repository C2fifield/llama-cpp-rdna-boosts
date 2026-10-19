# RESOLUTION — `async-input-graph-ordering` (TODO #73)

**Status: RESOLVED (2026-10-19), promoted into `v12-097cf33d9-r90` as a block-10 amendment.**
`GGML_CUDA_ASYNC_INPUT` was opt-in since r69 because a run was not start-to-start deterministic when the
consuming graph was captured.  This campaign found the defect, fixed it, and the async path is now
**default-on** (`GGML_CUDA_ASYNC_INPUT=0` is the kill-switch).

## Root cause

`ggml_backend_cuda_set_tensor_async()` issued a pinned-source `cudaMemcpyAsync` **directly from the
caller's live host pointer**.  A pinned-source async copy is read by the copy engine when the stream
reaches it, not when it is issued, so the caller must not modify the source until the copy completes.
The scheduler uses that path for a host-resident *split* input, and the one that bites is a **CPU split's**
output (`node_1` / `model.input_embed`, a `ROCm_Host` buffer), which the next ubatch's CPU split rewrites
in place.  With the async path the host runs ahead, so the copy could read the next ubatch's bytes (or a
torn mix): a rare single-graph corruption in **layer 0** that propagated to the end of the chunk.
`ggml_backend_cuda_set_tensor_2d_async()` had the same shape on the meta/tensor-split splice path.

Why the earlier readings were consistent with it: it needs a *captured* graph (without capture the host is
the bottleneck and never gets ahead), any host delay lets the copy win, and the perplexity tool clears its
cache per chunk, which made the damage look "bounded and re-converging" instead of propagating.  The ring
itself was innocent throughout: 769 graphs x 9-10 uploads verified byte-correct at the graph's first read,
no inter-graph overlap, HIP's ordering primitives correct (two standalone reproducers), and forcing every
upload onto the compute stream (stream order alone) still diverged.

## What landed (block 10, `ggml/src/ggml-cuda/ggml-cuda.cu`)

- Both async set paths **snapshot the source through the pinned upload ring** (a ring slot is not reused
  until the ring wraps, and the wrap is guarded by a device sync).  **Unconditional** — it is a
  correctness fix, not an async-path optimisation; the old direct copy is only the fallback when the ring
  cannot take the copy (size cap / non-COMPUTE buffer).
- The ring's completion wait (`ggml_cuda_async_input_wait`) and its "previous graph done" guard become
  **unconditional**, so those snapshots are ordered with `GGML_CUDA_ASYNC_INPUT=0` too.
- `GGML_CUDA_SPLICE_GATHER` now only chooses the *shape* of the queued 2-D copy (compacted 1-D vs strided
  2-D); every arm is snapshotted.
- **Default flip**: `g_async_input_on` is on unless `GGML_CUDA_ASYNC_INPUT=0`, and the stale block-10
  comment (which blamed the `done_ev` wait for "not ordering the HIP graph replay's read") is replaced
  with the corrected rationale.
- `scripts/gate-prefill-logits.sh` gained `--ub N` (with a per-ubatch base file), so the captured-graph
  band (`-ub 8`) can be gated explicitly.

## Evidence

`FINDINGS-2026-10-19.md` (session 1: eliminated hypotheses), `FINDINGS-2026-10-19b.md` (session 2: the
instrument, the root cause, the validation table), `HANDOVER.md` (state of play), `tools/` (the diagnostic
patches and the two standalone HIP reproducers) and `r90-block10-amendment.patch` (the exact change).
Self-KLD control 91.0-95.1 % -> 99.967-100.000 %; greedy text, width purity and the prefill-logit gate
unchanged (0.000707 / 98.755 %, identical to the r89 base); decode +9.3 % (35B-A3B 2 GPU tensor tg400),
+5.2 % (27B MTP n3 2 GPU tensor end-to-end); no GPU faults.  Full record: `WORKLOG.md` 261019.007.

## Instruments added for this campaign (kept, diagnostic only)

`tools/DEBUGBUILD.patch` + `tools/DEBUGBUILD2.patch` (session-1/2 scratch instrumentation on
`~/llama.cpp` `ain-debug`): split/copy-path prints, per-graph timeline, per-node full-tensor device hashes
(`AIN_HK`, with `AIN_HK_NOSRC` to avoid the fused-node stale-buffer artefact), whole-run upload
verification (`AIN_VK`), `AIN_MAIN_COPY`, `AIN_CAP_WAIT`, `AIN_NO_FREE`, `AIN_Q81_NOCACHE`,
`AIN_MMVQ_ITEMS`.  `tools/ain_hip_test{,2}.cpp` remain the standalone HIP ordering reproducers.
