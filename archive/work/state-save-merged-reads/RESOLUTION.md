# RESOLUTION — state-save-merged-reads (PR #149)

**Status:** FOLDED into the delivery as **block 11** (`src/llama-context.cpp`, r56, 2026-10-13),
**default-on**.  **Owner:** `@briansp2020` (PR #149).

## What was adopted

`0001-llama-read-a-state-save-s-deferred-tensor-ranges-as-.patch` — in the `llama_io_write_host`
destructor, consecutive deferred `write_tensor` entries of the same tensor whose gaps are at most
`LLAMA_STATE_GET_MERGE_GAP` bytes (default 256 KiB, `0` = the old per-range reads) are read as one
`ggml_backend_tensor_get` span and scattered on the host; if the pieces are contiguous in both the
tensor and the output, the span is read straight into the output.  A failed host span allocation falls
back to per-range reads (the destructor must not throw).

Motivation: a sequence's KV cells are interleaved with the other slots' cells a few tokens at a time
under concurrent decoding, so a slot save was ~8,000-10,000 short synchronous reads (~40 us each on
ROCm, 340-400 ms per save); merged, it is one get per tensor, 25-65 ms.  The bytes written are the
same.

## Validation on the delivery tree (gfx1201, ROCm 7.14.1)

- `test-save-load-state` (4B Q8_0): **all tests pass**, including Test 8 (state blob round-trip) and
  Tests 4-7 (seq copy host/device/scatter) which go through `llama_io_write_host`.
- `test-state-restore-fragmented`: **SUCCESS** ("state restore works with fragmented KV cache").
- A `--cache-ram` `llama-server` session with `-np 4` saves/evicts entries with 0 errors / aborts.
- 4B same-seed coherence and `gate-prefill-logits.sh` (0.000707 / 98.755 %) unchanged.
