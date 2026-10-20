# UPSTREAM-PR: ggml-cuda — handle repeated expert ids in `MUL_MAT_ID`

> **Upstream candidate** · status: prepared 2026-10-13; applies clean to `097cf33d9` and current master `1b70350c9`; dup cases 16/16, full suites green · not filed.
> Delivery: **block 00** (since r66); the delivery ships pure on this bug · patch: [`UPSTREAM-PR-mul-mat-id-dup.patch`](UPSTREAM-PR-mul-mat-id-dup.patch) (this directory).

**Status:** prepared 2026-10-13; applies clean (`git apply --check`) to the fork point `097cf33d9` and
to current upstream master `1b70350c9`.  Validated on gfx1201 / ROCm 7.14.1.  **Not filed.**

**Where it lives in the delivery:** **folded into block 00** as of `v12-097cf33d9-r66` as a generic
upstream foundation fix (block 00 touches none of its files, and every later MUL_MAT_ID block 06-10 then
builds on correct duplicate handling).  This file remains the standalone upstream candidate; **filing it
upstream is still the maintainer's call.**  The campaign
record and the reproduction are `archive/work/upstream-mul-mat-id-dup/` (formerly
`wip/upstream-mul-mat-id-dup/`).  Do **not** add this to `patches/`.

## The bug

`test-backend-ops -o MUL_MAT_ID -p 'dup=1'` fails F16/q8_0/q4_0 and then page-faults the GPU in
`quantize_mmq_q8_1`.  The `test_mul_mat_id_dup` cases (added upstream by the Vulkan duplicate-ids fix
`966baae76`, #29998) put repeated expert ids in a token's id row; the CPU reference and the Vulkan
prepass treat **every occurrence** as its own destination column and its own `src1` column.

CUDA/HIP had three related defects:

1. **`mmid.cu` `mm_ids_helper` deduplicated.**  It kept one compact column per `(token, expert)` pair,
   so when a row repeated an expert, only one occurrence was written.  The compact `ids_src1` tail left
   for the skipped occurrences was never initialised, and the MMQ path then read that uninitialised
   tail in `quantize_mmq_q8_1_cuda` — the `quantize_mmq_q8_1` memory fault is a *symptom* of the dedup,
   not a separate sizing bug.
2. **The MMQ/MMF column-tile grids were sized from the token count.**  `mmq_args::ncols_max` was
   `ne12` (tokens) and the MMF compact launch used `ceil(ncols_dst/cols_per_block)`; with duplicates an
   expert can own up to `n_tokens*n_expert_used` columns, so the higher column tiles were never
   launched.
3. **The MMF non-compact path stored one slot per token** (`slot_map[j]` plus `break`), so it could not
   represent a repeated id even before the grid issue; `ggml_cuda_mul_mat_f` only built the compact map
   for `ncols_dst > 16`, leaving F16 at `ncols_dst = 9/16` on the broken path.

## The fix

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmid.cu` | rewrite `mm_ids_helper`: pass 1 counts the lower-indexed experts' columns (`nex_prev`), pass 2 writes **every** matching `(token, iex)` pair as its own compact row in `(token, iex)` order directly to `ids_dst`/`ids_src1`.  No per-token shared staging buffer, so the shared-memory need no longer grows with `n_tokens*n_expert_used`; the templated per-`n_expert_used` specialisations and the 22/10-bit store are dropped. |
| `ggml/src/ggml-cuda/mmq.cu` | set `ncols_max` to `ne12` for the non-stream-k kernel (which now loops) and to the full `ne_get_rows` only for stream-k (whose flat k-space needs the worst-case width). |
| `ggml/src/ggml-cuda/mmq.cuh` | non-stream-k path: wrap the column tile in `for (int jt = blockIdx.y; jt < ntiles; jt += ntx.z)` (the fastdiv triple's divisor), with a barrier between iterations, instead of one tile per block. |
| `ggml/src/ggml-cuda/mmf.cu` | build the compact id map for **every** `MUL_MAT_ID`, not only `ncols_dst > 16`. |
| `ggml/src/ggml-cuda/mmf.cuh` | `mul_mat_f_ids`: loop the column tile over a tight `ceil(ncols_dst/cols_per_block)` grid, re-zeroing `C` each tile, and replace the store loop's `return` with `continue`; `mul_mat_f_switch_ids` always takes the compact kernel. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the host sorted fallback: remove the `break` after the first expert match, so every occurrence is emitted. |

## Validation (gfx1201, ROCm 7.14.1, unadulterated base + this patch)

| gate | result |
|---|---|
| `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` | **16/16** (base 0/16) |
| `test-backend-ops -o MUL_MAT_ID` | **947/947** |
| `test-backend-ops -o MUL_MAT` | **2444/2444** |
| `scripts/check-gpu-faults.sh` after the runs | clean (no `quantize_mmq_q8_1` fault) |
| `llama-bench` 35B-A3B Q8_0, 2x R9700, `-sm tensor`, `-fa 1` | pp512 **2587** / pp2048 **2577** t/s vs the unpatched tree's **2588** / **2563** — within noise |

## Notes / what was NOT validated

* The stream-k MMQ path (`config.stream_k`, not used by the RDNA configs) keeps the full worst-case
  `ncols_max` because its tile index is derived from the flat k-space; correct, but it over-launches on
  the architectures that use it (CDNA/NVIDIA).  A tile loop there, or a device-side max column count,
  is the follow-up.
* No NVIDIA/CUDA hardware was available: the `__reduce_add_sync` int reduction and the non-HIP shuffle
  fallbacks compile the same shape as upstream's existing warp reductions, but were not run there.
* MUSA/SYCL/Vulkan are untouched.
* The full non-`dup` matrices were run on gfx1201 only.

Not filed: awaiting the maintainer's go-ahead.
