# RESOLUTION — the RDNA3_5 W=1 impurity is the `GLU -> MMQ down` fusion being preempted, not the fused gate/up arithmetic

Date: 2026-10-19.  Box: Strix Halo APU, Radeon 8060S (**gfx1151**), 1 device, 123 GiB unified,
ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx1151`).  Tree: delivery **r91** (`45f817cc3`, tree
`90507dfb528cbfe0c688261226d18934653b1b61`).  Model: qwen4exp Flash-Next **UD-Q4_K_XL** (the Q8_0
`ffn_{gate,up}_shexp` case; the dense shexp of the last block, `blk.47`, is the W=1-only dense GLU
fusion in this model).  Prompt: `prompts/prose-rdna-boosts.txt` (P=256) for the width probe.

**Status: CLOSED as ending B (no delivery change).**  The skip stays.  A bit-exact candidate (ending A)
was validated and is recorded below, but it requires a second default change (suppress the
`GLU -> MMQ down` fusion on RDNA3_5 dense GLUs) and changes the band-wide hash from `578528...` to
`1732828...` (text unchanged on the tested prompts).  That is a maintainer decision, not a mechanical
promotion.

## 1. Hashes on the current tree (row-0 logits hash, `test-logits-width-probe <model> <prompt> 256 512`)

| arm | W=1 row 0 | W=8 row 0 | verdict |
|---|---|---|---|
| default (skip active) | `578528d234936576` | `578528d234936576` | PASS |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` | `173282800f97009f` | `578528d234936576` | **FAIL** (worst maxdiff 0.323269) |
| default + `GGML_CUDA_DISABLE_SWIGLU_MMQ=1` (temporary probe) | `173282800f97009f` | `173282800f97009f` | PASS (all W) |
| env ON + `GGML_CUDA_DISABLE_SWIGLU_MMQ=1` | `173282800f97009f` | `173282800f97009f` | PASS (all W) |
| env ON + `GGML_CUDA_DISABLE_SWIGLU_MMQ=1`, weighted-down **on** | `173282800f97009f` | `173282800f97009f` | PASS (all W) |

The last row is the sibling check: with the down path held uniform, the weighted-down MoE tail is
**not** an independent impurity in this configuration.

## 2. The gate/up arithmetic is bit-identical (the old premise is refuted)

Temporary debug dumps (reverted; not in the delivery) of the ksplit kernel's reduced `tmp` (up) and
`tmp_gate` (gate) for the `blk.47` dense shexp launch (Q8_0, K=2560, N=640, `nwarps=2`,
`rows_per_cuda_block=1`), comparing the fused GLU launch with the default path's **dual-output**
fusion (`ggml_cuda_mul_mat_dual_output_try_fuse`, `src0=gate`, `fusion.gate=up`, `dst_gate` set):

* all **640 output rows**, **3 forward passes**: fused `up` == dual-output `up`, fused `gate` ==
  dual-output `gate`, **bit for bit**;
* the fused epilogue `up * silu(gate)` matches the standalone expression (the row-0 case was checked
  against the fp32 expression; the all-row comparison is the width probe itself).

So **codegen/FMA contraction is exonerated**, and the Q8_1 cache hypothesis is exonerated too:
`quantize_row_q8_1_cuda` ignores `type_src0`, the gate/up share the same `src1`, and identical
gate/up values cannot come from a different Q8_1.  The fused kernel *does* reproduce the standalone
`mul_mat_vec_q` arithmetic.

## 3. The mechanism: the down matmul changes family

The default path fuses **`GLU + down`** into an **MMQ** down matmul:
`ggml_cuda_mul_mat_q_swiglu_try_fuse` (`mmq.cu`) matches the qwen4exp dense shape
(`GLU ne0=640`, next `ne0=2560`, Q8_0, RDNA3_5, `should_use_mmq`), and its fusion trace confirms
`ffn_swiglu-47 -> ffn_shexp-47` on every forward pass.  That fused GLU+MMQ chain is what produces the
default arm's `578528...`.

Enabling the dense gate+up+GLU fusion **consumes the GLU node** (`nodes_to_skip` from
`ffn_swiglu-47`), so the `GLU -> MMQ down` fusion can no longer fire.  At W=1 the down matmul then
takes the **normal** path; at W>=2 the dense GLU fusion does not fire (it is `ncols_dst==1` only), so
the `GLU -> MMQ down` fusion is still active.  W=1 and W>=2 therefore run **different down kernels**,
which is the entire W=1/W=8 difference.

That is why the "equals the unfused arm" target is subtle: the default arm is not the plain unfused
chain, it is `dual-output gate+up + GLU->MMQ down`.  The plain chain (GLU materialized, normal down)
is `1732828...`, and it is **width-pure** (the `DISABLE_SWIGLU_MMQ` rows above).

## 4. Candidate A (validated, not promoted)

Proposed change (no source change was promoted):

1. On RDNA3_5, make the dense gate+up+GLU fusion **default-on** with
   `GGML_CUDA_DISABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` as the kill switch (default-on policy).
2. In `ggml_cuda_mul_mat_q_swiglu_try_fuse`, **do not** fuse the GLU into the MMQ down matmul for
   dense GLUs on RDNA3_5 when the gate+up fusion is enabled.  The down path is then the normal path
   for the whole W=1..8 band (which the probe shows is pure).

Measured with the equivalent env A/B (`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` +
`GGML_CUDA_DISABLE_SWIGLU_MMQ=1`):

| gate | default | candidate A | verdict |
|---|---|---|---|
| `test-logits-width-probe` qwen4exp, W=1..8 row 0 | `578528...` (pure via GLU+MMQ) | `1732828...` (pure via normal down) | PASS (band-pure; **different band hash**) |
| `llama-bench -p 0 -n 128 -r 3` tg128 | 26.39 ± 0.05 | 26.55 ± 0.06 | **+0.6 %** |
| `llama-bench -p 512,2048 -n 0 -r 3` | 953.87 / 964.89 | 954.97 / 961.04 | flat (+0.1 % / -0.4 %) |
| `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` TG | 25.92 / 64.45 / 85.76 | 26.05 / 64.55 / 85.59 | +0.5 / +0.15 / -0.2 % (noise) |
| qwen4exp plain vs `draft-mtp n3` text (seed 42, temp 0) | — | 35 chars `sha=359ff4337837` both | PASS |
| 4B Q8_0 f16/bf16/q8_0 same-seed text (both arms) | 280 chars `sha=7386359e5dac` | same | PASS |
| qwen4exp default vs candidate plain text | 35 chars `sha=359ff4337837` | same | unchanged |
| `scripts/gate-prefill-logits.sh` | n/a | **not run** | no source path changed for the B ending; **required if candidate A is promoted**, because it suppresses a prefill fusion |
| `scripts/validate-set.sh` | n/a | **not run** | no delivery change; required for any promotion |

**The caveat that keeps this out of the delivery:** candidate A is bit-identical to the **plain**
unfused chain, but it changes the **current default** arm's band hash (`578528...` -> `1732828...`).
The text is unchanged on the tested prompts, but preserving the current hash would require a
**three-way** `gate+up+GLU + MMQ-down` fusion (so the down path never changes), which is a much larger
change than the ~0.6-0.9 % win justifies (`AGENTS.md`: do not gold-plate).  The candidate also
disables a prefill fusion for all dense GLUs on RDNA3_5; it measured flat here, but that is one model.

## 5. Disposition

* **Ending B** for the delivery: the skip stays; no block amendment.
* The old premise in `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9 ("the fused
  kernel does not reproduce the standalone arithmetic") is **wrong** and is corrected by §2-§3 here.
* Re-entry conditions for ending A: a three-way fusion, or an explicit maintainer decision to accept
  the `578528...` -> `1732828...` band-wide change plus the RDNA3_5 GLU+MMQ-down suppression.
