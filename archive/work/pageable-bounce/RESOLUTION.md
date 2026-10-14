# RESOLUTION — pageable-bounce (PR #147)

**Status:** FOLDED into the delivery as **block 09** (`ggml/src/ggml-cuda/ggml-cuda.cu` +
`vendors/hip.h`, r56, 2026-10-13), **default-on**.  **Owner:** `@briansp2020` (PR #147).

## What was adopted

`0001-ggml-cuda-bounce-pageable-host-device-tensor-copies-.patch` — `buffer_get_tensor`,
`buffer_set_tensor` (not weight buffers) and `buffer_get_tensor_2d` copy pageable host data through one
reused 64 MiB pinned chunk per device.  A pinned host pointer keeps the direct copy; a failed chunk
allocation leaves the copy direct and is not retried.  `GGML_CUDA_BOUNCE_MIN` sets the threshold in
bytes (default `1` = every size, `0` = off).

The motivation is the server's context checkpoint (~118 MB of qwen4exp recurrent state in 512 KiB
tensors) and its `--cache-ram` slot saves, which make thousands of small pageable D2H reads (twice as
many under `-sm tensor`, one 2-D copy per device).

## Adaptation made during integration

`ggml_cuda_host_ptr_pageable` is now compiled only for `GGML_USE_HIP` / `GGML_USE_CUDA` and returns
`false` on MUSA.  The contributor flagged that `vendors/musa.h` has no
`cudaPointerGetAttributes` / `cudaMemoryType*` aliases; rather than add untested MUSA defines, the
delivery keeps the MUSA build compiling with the bounce inert there (it is a ROCm win).

## Validation on the delivery tree (gfx1201, ROCm 7.14.1)

- `test-save-load-state` (4B Q8_0): **all tests pass**, including the state blob round-trip and the
  host/device/scatter seq copies, which exercise the pageable get/set path this PR re-routes.
- `test-state-restore-fragmented`: **SUCCESS**.
- A `--cache-ram` `llama-server` session with `-np 4`: saves and evicts prompt-cache entries
  (`making room for prompt cache entry, removing oldest entry`) with 0 errors / aborts.
- 4B same-seed coherence and `gate-prefill-logits.sh` (0.000707 / 98.755 %) unchanged.
