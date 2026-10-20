# RESOLUTION — `set-tensor-async-ring` (PR #194) — DECLINED at r98

**Status: reviewed 2026-10-20 and NOT promoted.**  The submitted patch plus README are kept here because
the diagnosis is worth keeping; the change is being revised by the submitter.

## What was proposed

Keep r90's `set_tensor_async` snapshot guarantee (the copy reads bytes taken when the upload is issued) but
take the snapshot into a pinned two-half ring owned by the backend context and issue the copy on
`cuda_ctx->stream()`, so the upload no longer joins the graph-input upload ring's queue.

## What was verified

* The r90 fix survives: `llama-perplexity --kl-divergence-base` self-comparison (27B UD-Q4_K_M, `-c 512`,
  `-b 8`, 24 chunks, `GGML_CUDA_ASYNC_INPUT=1`) reads **100.000 %** at `-ub 8` and at `-ub 1`, same as the
  r97 base.
* Greedy text is byte-identical to r97 in every config tried (dense 27B at 8K and 110K, 27B 2 GPU tensor,
  35B-A3B 2 GPU tensor, Flash-Next 2 GPU tensor, gemma4 12B).  No GPU faults.

## Why it was declined

It is a **regression on this tree**, because of one line: `ggml_cuda_async_input_drain(cuda_ctx->device)`
before the copy drains the whole graph-input queue, so queued uploads are issued one at a time
(`cudaMemcpyAsync` on `cudaStreamPerThread` plus an event record each) instead of being folded into one
`ggml_cuda_ain_batch_kernel` launch (PR #182).  The `-ncmoe` host-expert uploads use that same ring.

3 interleaved pairs each, seed 42, temp 0, `-n 400`:

| config | r97 | r97 + PR #194 |
|---|---:|---:|
| Flash-Next GSQ IQ3_XXS + shared Q8_0 MTP, 2 GPU tensor, `-ncmoe 10`, q8_0 K/V | 131.4 t/s | 129.8 t/s (-1.2 %) |
| 35B-A3B UD-Q4_K_M, 2 GPU tensor, `-ncmoe 10`, q8_0 K/V, MTP n3 | 184.6 t/s | 180.3 t/s (-2.3 %) |
| 27B UD-Q4_K_XL, 1 GPU, plain, 8K | 29.35 t/s | 29.2 t/s (neutral) |
| 27B UD-Q4_K_XL, 2 GPU tensor, MTP n3 | 95.0 t/s | 95.1 t/s (neutral) |

Mechanism confirmation: with `GGML_CUDA_ASYNC_INPUT_BATCH=0` (per-copy path everywhere) both builds read
**exactly 129.1 t/s** on the Flash-Next cell.

Two further notes given to the submitter: the per-context ring takes no lock (the per-device ring it
replaces takes `d.mtx`), and `cudaEventSynchronize` on a half switch is a host-blocking wait inside the
upload path that r90's ring does not have.

## Re-entry conditions

Scope the drain to the destination-overlap case only (`ggml_cuda_async_input_set` already does exactly
that) and re-run this table plus the self-KLD control.  Nothing else in the patch needs to change.
