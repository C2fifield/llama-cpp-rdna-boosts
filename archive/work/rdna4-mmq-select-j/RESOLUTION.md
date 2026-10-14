# RESOLUTION — rdna4-mmq-select-j (PR #146)

**Status:** FOLDED into the delivery as **block 08** (`ggml/src/ggml-cuda/mmq.cu`, r56, 2026-10-13),
**default-on** (it restores the r49 selection).  **Owner:** `@briansp2020` (PR #146).

## What was adopted

`0001-ggml-cuda-routed-MMQ-tile-widths-for-RDNA4-again-r50.patch` — in `ggml_cuda_mmq_select_J`, the
routed-compact J selection gates on `mmq_routed_compact_arch_ok(cc)` again (RDNA3_5 **and** RDNA4) and
passes `GGML_CUDA_CC_IS_RDNA4(cc)` to `mmq_rdna3_5_id_get_J`.

The r50 re-base onto upstream `097cf33d9` (#29953) moved the J selection out of `mul_mat_q_switch_J`
into `ggml_cuda_mmq_select_J` and, in the move, left the gate RDNA3_5-only and dropped the `rdna4`
argument.  That had two effects, both fixed here:

- **gfx1201:** the IQ expert types at ~40 rows per expert fell to the generic tile-count loop (J = 48)
  instead of `MMQ_IQ_ID_J_MID` = 64, even though the compact enumeration was already enabled on RDNA4
  (`mmq_routed_compact_arch_ok`).
- **gfx1151:** the default `rdna4 = true` meant the IQ2 / Q2_0 types wrongly picked up the RDNA4 bands
  that r49 kept on the generic path.

No env var: J only changes the tiling, not the arithmetic per row.

## Validation on the delivery tree (gfx1201, 3x R9700, ROCm 7.14.1)

- 4B Q8_0 same-seed coherence: identical text to r55 (`build 5d51f478c` vs `1a77cb6d2`).
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS, unchanged).
- `test-logits-width-probe` on Flash-Next GSQ-IQ3_XXS, `SPLIT=tensor`, `P=256 ub=512` (which drives the
  routed-compact `MUL_MAT_ID` path): **byte-identical to r55** at W=1..8, `width_purity=PASS`.
- `test-save-load-state`, `test-state-restore-fragmented`: PASS (unrelated paths, no regression).
