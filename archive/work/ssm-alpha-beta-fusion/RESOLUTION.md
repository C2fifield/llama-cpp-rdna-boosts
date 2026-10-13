# RESOLUTION — `ssm-alpha-beta-fusion`: closed below the adoption bar (2026-10-13)

**Status: CLOSED (2026-10-13).**  Re-measured on r53 and closed as below the adoption bar; the
structural blocker is unchanged.  Not promoted.

## What was measured

The idea: fuse the two F32 GEMMs on the same `src1` in the qwen4exp/hybrid-SSM layers
(`src/models/qwen4exp.cpp:2511` `beta = build_lora_mm(ssm_beta, cur, …)` and `:2518`
`alpha = build_lora_mm(ssm_alpha, cur, …)`) into a single `[2560x96]` walk.

Re-confirmed on r53/gfx1201:

- Both MMs still share the **same `cur` tensor object** (`build_lora_mm(w, cur)` →
  `ggml_mul_mat(w, cur)`), with beta's `reshape+sigmoid` between them — the pair is structurally safe.
- Fresh `pairbench` (2x M48 vs stacked M96, F32 `[2560x96]`, same `cur`):

  | N | pair (2x M48) | stacked (M96) | ratio | maxdiff |
  |---:|---:|---:|---:|---:|
  | 2048 | 0.331 ms | 0.194 ms | 1.71x | 0.0 |
  | 8192 | 0.587 ms | 0.412 ms | **1.43x** | **0.0** |

  The 1.43x at N=8192 reproduces the 2026-09-06 archive (`0.588` vs `0.410`).  New: on **gfx1201 the
  two paths are bit-identical** here (the archive's ~3e-7 was gfx1151 / a different rocblas config).
- Wall estimate: ~0.14 ms/layer saved at N=2048 x ~36 recurrent layers ≈ 4.9 ms per 2048-token pass →
  **~0.3–0.6 % prefill**, consistent with the archive.

## Why it is closed

- The two MMs remain **non-adjacent** in the scheduled DFS graph (the general `nodes i, i+1`
  pair-matcher cannot fire), so the work is either load-time stacked weights + per-half views (loader
  + model-struct + graph plumbing) or a graph restructure + custom single-walk kernel.
- At 0.3–0.6 % prefill for a numerics-moving (and now, on gfx1201, bit-identical) change, the
  implementation cost is above the delivery's bar.  No route was attempted.

## Re-open if

A future release wants the last fraction of qwen4exp prefill and can take the loader-plumbing route
(a load-time stacked `[2560x96]` weight + one graph `MUL_MAT` + per-half views).

## Pointers

- `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`
  (the original microbench, the rejected inject fusion, the adjacency analysis).
- `src/models/qwen4exp.cpp:2511/2518`; `src/llama-graph.cpp` (`build_lora_mm`).
- `benchmarks/prefill-logit-methodology.md` (if ever implemented).
