# `rdna35-mmvq-fusion-perf` — restore the block-13 RDNA3_5 single-token fusion perf (TODO item 16)

**Status: PARKED (seeded 2026-10-11), low priority.** Not part of the delivery; nothing here may be
applied to the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).
This is the parked TODO item *"Restore the block-13 RDNA3_5 single-token fusion perf (item 16)"*.

> **Cold start.** Read this file, then
> [`archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md`](../../archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md)
> **§9** (the post-fold analysis that re-scoped this item), and `GREEDY-PURITY.md` §36. Report findings to
> the maintainer before any delivery change and wait.

## 0. The item

On **RDNA3_5 (gfx1151 / Strix Halo)** the dense gate+up+GLU `mmvq` fusion is **single-token-only**
(`ncols_dst == 1`) and its fused kernel does **not** reproduce the standalone `mul_mat_vec_q` arithmetic,
so a 1-token decode and an n-token verify batch of the same layer are not bit-identical — the
`decode == verify` invariant greedy MTP depends on. The delivery therefore **skips** it on that arch
(`ggml_cuda_rdna3_5_dense_glu_disabled()`, `ggml-cuda.cu:3932`), which restores `W = 1..8` purity but costs
**~0.9 % `tg128`** on qwen4exp (25.53 vs 25.77 t/s, prefill flat). The skip is opt-out-able for A/B with
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`; the sibling weighted-down MoE tail is gated in
`ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`.

## 1. Current reality — the original fix plan is dead

The TODO's proposed fix was *"pin `nwarps`/`rps`/item-split"*. The archive §9 proves that plan **does not
apply**: the fused and unfused dense paths are already **launch-identical**.

- Both go through the same template `mul_mat_vec_q_ksplit<type, ncols_dst=1, has_fusion, …>`; the
  `ncols_dst == 1` dense dispatch is the unconditional ksplit branch, so no item-split/ksplit asymmetry
  remains.
- `nwarps` comes from the same `calc_nwarps(type, 1, table_id)`; `rows_per_block` from the same table;
  `calc_launch_params` produces the same blocks/dims.
- The epilogue (`result_val *= ggml_cuda_op_silu_single(gate_value)`) uses the same device function the
  standalone GLU kernel uses.

So a fix must move the **arithmetic**, not a dispatch parameter. Two live candidates:

1. **Codegen / FMA contraction.** `has_fusion=true` adds registers and a second `vec_dot` in the inner
   loop; if clang contracts/associates the `tmp` FMAs differently in the fused instantiation, `up` shifts
   by a ULP. (Precedent: the r25 rope fix needed `#pragma clang fp contract(off)` for exactly this class —
   see `GREEDY-PURITY.md` §41.)
2. **The Q8_1 cache.** `ggml_cuda_mul_mat_vec_q` quantizes `src1` via `ctx.q8_1_cache_get(src1_key, …)`
   (`common.cuh:1611`), keyed on the src1 tensor/layout only — **not on the weight type**. Fusing changes
   which call fills the cache for a given `x`; if two consumers of one `x` have different weight types, the
   Q8_1 they see can differ (and the fused arm only quantizes with the `up` type).

## 2. Next instrumentation (the only way to separate 1 from 2)

Dump `tmp`/`tmp_gate` from the ksplit kernel for the fused launch and `tmp` from the two unfused launches,
at `W=1`, and compare the `up` values bit-for-bit. If `up` matches → the epilogue/Q8_1 cache; if it differs
→ codegen. Add the dump behind an existing debug env (there is a `GGML_CUDA_MMVQ_*` dump convention in
`mmvq.cu`); do not leave it in a promoted change.

## 3. Gates

- **Purity is the hard gate:** `test-logits-width-probe` on qwen4exp / 35B `W = 1..8` must be one hash.
  The whole point is to make the fused path reproduce the standalone arithmetic, so the gate is
  `W=1..8 ==` (not merely W-pure within the fused path).
- Same-seed greedy coherence on gfx1151; MTP acceptance unchanged.
- `scripts/validate-set.sh` if promoted; note the ~0.9 % ceiling means this may not be worth a dedicated
  release — report the measured win/loss before proposing a block change.

## 4. Pointers

- `TODO.md` → "Parked (not planned now)" → item 16.
- [`archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md`](../../archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md)
  §9 (the re-scoping; §9 lists both candidates and the instrumentation).
- `ggml/src/ggml-cuda/ggml-cuda.cu:3932` (`ggml_cuda_rdna3_5_dense_glu_disabled`),
  `ggml/src/ggml-cuda/mmvq.cu` (`calc_nwarps`, `calc_rows_per_block`, `mul_mat_vec_q_ksplit`),
  `common.cuh:1611` (`q8_1_cache_get`).
- `GREEDY-PURITY.md` §36 and §41.
