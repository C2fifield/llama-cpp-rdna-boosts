# RESOLUTION — `async-input-batch` (PR #182, @briansp2020)

**Status: RESOLVED (2026-10-19), both parts promoted into `v12-097cf33d9-r89` as a block-10 amendment.**
The contributor's two format-patches and their write-up are preserved in this directory.  The campaign
arrived as a PR branch (`wip/async-input-batch/` on the PR head), i.e. it was never a `wip/` entry on
`main`, so no stub was created or removed here.

## What landed (block 10, `ggml/src/ggml-cuda/ggml-cuda.cu`)

Block 10 owns the async graph-input ring (`GGML_CUDA_ASYNC_INPUT`, r46/PR #125, default-off since r69),
so both halves fold into it verbatim except for the default flip.

- **`0001` `GGML_CUDA_ASYNC_INPUT_BATCH`** (now **default on**, `=0` restores the per-copy uploads): a
  graph's staged inputs are queued per device and copied by one `ggml_cuda_ain_batch_kernel` on the
  compute stream when the next graph starts, instead of one `cudaMemcpyAsync` + one event record per
  input.  Every other buffer/backend access (get/set/memset/clear/cpy, the 2-D and async forms,
  `synchronize`, `free_buffer`) drains the queue through the old per-copy path first and orders
  `cudaStreamPerThread` after the last batch kernel (`flush_ev`); an overlapping upload or a full queue
  (64) drains too, and a ring wrap additionally waits for `flush_ev`.  `cudaHostGetDevicePointer`
  failure falls back to the per-copy path.
- **`0002` `GGML_CUDA_ASYNC_INPUT_LAZY_DONE`** (default = the batch default; `=0`/`=1` override):
  `done_ev` is no longer recorded after every `graph_compute`; the graph marks it pending with its
  stream and the next upload records it there, which covers at least the same work.  A context
  destructor flushes a still-pending event before its streams are destroyed.

`GGML_CUDA_ASYNC_INPUT` itself **stays opt-in**.  Its r69 nondeterminism is not fixed by this PR (see
`wip/async-input-graph-ordering/`), so the default product is unchanged by the fold; what changes is
that `ASYNC_INPUT=1` is now worth +4.6 % (MoE, 2 GPU tensor) instead of +1 %.

## Validation (gfx1201, 3× R9700, ROCm 10.1)

A/B on one binary, env-selected arms, interleaved rounds, 400-token greedy, `-fa on`, q8_0 KV,
`-ub/-b 2048` (per-arm spread ≤ 0.4 %):

| config | async off (delivery) | `ASYNC_INPUT=1` before | `ASYNC_INPUT=1` after |
|---|---|---|---|
| 35B-A3B Q4_K_M, 2 GPU tensor | 109.5 t/s | 110.7 (+1.1 %) | **115.8 (+5.8 %)** |
| 27B Q4_K_M MTP n3, 2 GPU tensor | 83.2 t/s | 85.0 (+2.2 %) | **86.2 (+3.6 %)** |
| 35B-A3B, 1 GPU | 100.0 t/s | 97.8 (−2.2 %) | 99.6 (−0.4 %) |
| 27B, 1 GPU | 31.7 t/s | 31.8 | 31.8 |

Prefill `pp4096` (35B, 2 GPU tensor): 5449 → 5437 → 5486 t/s (flat).  Contribution split at 35B-A3B
2 GPU tensor: `0001` alone +1.4 %, `0002` alone +3.4 %, both +4.6 % over `ASYNC_INPUT=1` alone —
matching the contributor's own table (+0.23 / +3.87 / +4.26 %) and their 1-GPU and MTP results.

Gates: 4B Q8_0 `-n 64` seed 42 `7386359e5dac` with async off / on / `BATCH=0` (== r68's known-good);
`test-logits-width-probe` W=1..8 identical hashes + PASS; 27B 1-GPU and 27B MTP 2-GPU greedy
byte-identical across arms; prefill-logit **0.000707 / 98.755 %** (unchanged); `test-backend-ops`
`ADD`/`GET_ROWS`/`CPY`/`MUL_MAT` (2864 cases) pass with the batch path on; no GPU faults.  35B-A3B
greedy text is **not** a usable gate on this box (the stock r88 binary is itself run-to-run
nondeterministic there), which is recorded in the wip campaign.

## Packaging note

The PR's `wip/async-input-batch/` packaging was not taken (same rule as #167/#172/#181): the delivery
takes the two format-patches folded into the owning block, and the record lives here.
