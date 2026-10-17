# `archive/work/rdna2/pr165` - slavap's RDNA2 add-on set (PR #165), folded into r76

**Status: FOLDED into r76 (2026-10-16)** into blocks 05 and 07 of the 12-block delivery. This
directory preserves the original add-on exactly as merged in PR #165, for provenance only. Nothing
here is applied by `scripts/apply-all.sh`.

## What it was

Five patches cut against the r68 tree (`release-rdna2.json` base `ffe83269`), applied after the
12-block set by `scripts/apply-rdna2.sh` on branch `rdna-boosts-rdna2`: the RDNA2 tile-FA config and
launch caps, the partial-batch load fix, the Q-from-global path, the Navi21 MMQ/MMVQ work, and a
gfx1030-only WMMA build guard. The layout here is the original one (`patches-rdna2/`,
`scripts/apply-rdna2.sh`, `release-rdna2.json`), so the archived script still resolves its own paths.

## How it landed

- blocks 00-02 (tile FA) folded into **block 05**; block 03 (k-quant) folded into **block 07**.
- block 04 (WMMA guards) was **dropped**: r70 already folded the equivalent guards into blocks 06/10
  (`mmb.cu`, `fattn-qsa3.cu`), and a gfx1030-only build passes without it.
- the partial-batch branch and `ggml_cuda_mmq_scale_sumi` were made `#if defined(RDNA2)`, because the
  original add-on changed every AMD build. gfx1100 / gfx1151 / gfx1201 keep their previous codegen and
  numerics (verified: identical greedy text and mean KLD 0.000000 on all three).
- the Q-from-global path is compile-gated by `GGML_CUDA_FATTN_TILE_Q_GLOBAL` (HIP+RDNA2) and
  host-gated by `q_h2`; the Navi21 table is selected by `GGML_CUDA_CC_IS_RDNA2`.
- the only unconditional change is an inert extra `Q_h2` parameter on the shared `fattn_kernel_t`
  typedef and the three FA kernels; it is never used off RDNA2.

Release record: `../../../../WORKLOG.md` 261016.011 and `../../../../patches/README.md` (r76).

## Using it

It is not a current artifact. If a future re-base needs the original series, re-cut from the folded
blocks instead of applying these patches: they target r68 and predate the r70 guards.
