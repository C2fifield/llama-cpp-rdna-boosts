# RESOLUTION -- `rdna3-mmvq-moe-q4k-vdr4` (PR #167, @briansp2020)

**Status: RESOLVED (2026-10-17), promoted into `v12-097cf33d9-r79` as blocks 07/10.**  Patch 0001 is
the default on RDNA3_0; patch 0002 ships as the RDNA4 build-time opt-in (default off).  The
contributor's format-patches and write-up are preserved in this directory; nothing was left behind as a
`wip/` stub (the campaign arrived as a PR branch, never on `main`'s `wip/`).

## What landed

- **block 07** (`vecdotq.cuh`): `vec_dot_q4_K_q8_1_vdr4_dense` is available to the MoE expert kernel
  (`GGML_CUDA_Q4K_MOE_VDR4`), default on RDNA3_0, and the `ggml_cuda_rdna4_q4k_moe_vdr4` host-visible
  constant keys ccache correctly when the RDNA4 flag is toggled.
- **block 10** (`mmvq-vecdot.cuh`): the Q4_K selector takes the dense VDR-4 dot product on RDNA3_0 (MoE
  and dense) and, on RDNA4, for the MoE arm only when `-DGGML_HIP_RDNA4_Q4K_MOE_VDR4` is set.  The
  dense RDNA4 arm keeps its separate `GGML_HIP_RDNA4_Q4K_DENSE_VDR4` flag.  The MoE VDR is 4 either
  way, and the selector is still width-uniform, so the mmvq band and the pinned fusion ops are
  unchanged.
- **block 10** also carries the PR #168 follow-up: `LLAMA_KV_N_PAD_PREFILL` defaults to 8192 on a
  multi-device split and stays inactive on one GPU.

## Why 0001 is default-on but 0002 is opt-in

RDNA3_0 (independent gfx1100 rerun, r78 -> r79): plain decode **+0.83 %**, batched B=1/B=4/B=8
**+1.4 / +1.9 / +2.7 %**, 4-axis MTP R **+1.8 %** / C **+1.2 %** / K **+1.4 %** / phase-switching
**+3.0 %**, mean KLD **0.004544** (same-top 97.124 %), prefill-logit **0.000000 / 99.990 %**, and
`none == draft-mtp n3` byte-identical.  The P axis is **-2.7 %**, reproducible at `-n 2000` and
`-n 3000` and with fixed `n_max 3`; it is acceptance-driven (0.772 -> 0.718, mean accepted length
3.32 -> 3.15) because the greedy continuation changes, while plain decode on the same prompt is
**+0.7 %**.  That is rounding-level content drift, not a kernel slowdown.

RDNA4 (gfx1201, 3x R9700): 0001 is inert by construction and the default text is byte-identical to
r78.  The 0002 opt-in on **2 GPUs** is flat on single-token decode (+0.06 %), slightly positive at the
verify widths (B=1 +0.9 %, B=4 +0.5 %, B=8 +1.5 %), and mixed on MTP (R +3.4 %, K +0.8 %, P -1.65 %,
C -0.5 %).  That is not a clear default-on win, so 0002 remains an opt-in; the default RDNA4 path is
unchanged.  The 3-GPU P drop is content-driven and is cross-referenced from
`wip/three-gpu-performance/`.

## Validation (r79 candidate, gfx1201 and gfx1100)

- `MUL_MAT_ID` q4_K 84/84 on both builds; prefill-logit gate **0.000000** on both families.
- `none == draft-mtp n3` byte-identical on the RDNA3 default build and on the RDNA4 flag build.
- Full numbers and the exact commands: `../../benchmarks/2026-10-17-pr167-r79-acceptance.md`.
- RDNA3 P-axis finding: `../three-gpu-performance/README.md` (cross-reference section).

## Record

`WORKLOG.md` 261017.002 (r79); `ENVIRONMENT.md` (`GGML_HIP_RDNA4_Q4K_MOE_VDR4`,
`LLAMA_KV_N_PAD_PREFILL`); `patches/README.md` (current release block).
