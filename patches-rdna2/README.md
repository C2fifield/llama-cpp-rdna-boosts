# rdna-boosts RDNA2 (gfx1030) add-on set

**5 patches** (blocks 00-04) that apply on top of the 12-block delivery set
(`release.json`, net tree `ffe83269`), bringing the flash-attention tile kernel,
its launch path, and the K-quant MMQ/MMVQ paths to the tuned RDNA2 (gfx1030)
state.  Apply with `../scripts/apply-rdna2.sh` **after** `../scripts/apply-all.sh`;
the result is committed on branch `rdna-boosts-rdna2` and asserted against
`../release-rdna2.json`.

Blocks 00-03 are gated to gfx1030 at compile time (`GGML_CUDA_CC_IS_RDNA2` /
`GGML_CUDA_FATTN_TILE_Q_GLOBAL`) or by device table lookup, so gfx1100/gfx1151/
gfx1201 and NVIDIA builds of the same tree are bit-for-bit the 12-block set.
Block 04 is the other way around: a *build* fix for RDNA2-only target sets.
The base set's `qsa3` and `mmb` WMMA kernels used the gfx11/gfx12
`__builtin_amdgcn_wmma_*` builtins unconditionally, so a
`-DGPU_TARGETS=gfx1030` build failed at the first WMMA site. Block 04 wraps
those sites in `#if defined(AMD_WMMA_AVAILABLE)` with no-op stubs (the base
set's own issue #1 pattern, cf. `gated_delta_net_chunked_bf16*.cu` and
`mma.cuh`); the stubs are never launched (both paths are runtime-gated to
RDNA3+), and on RDNA3/RDNA4 and NVIDIA the original code compiles unchanged.

## Provenance

Ported from `sixvolts/llama-halo-hybrid` (the RDNA2/V620 tuning line, validated
by SixVolts on 2x Radeon AI PRO R9700).  Each block's commit body lists the
source commits:

| block | content | halo commits |
|-------|---------|--------------|
| 00 | the per-DKQ RDNA2 FA tile config table (`ggml_cuda_fattn_tile_get_config_amd_rdna2`), the launch caps (cols_per_block / ncols2 GQA selection) that keep wide kernels inside the gfx1030 register file + 64 KB LDS/WGP, and the RDNA2 occupancy compute in `launch_fattn` (HIP's occupancy query undercounts the RDNA2 register file 4x and reported 0 for the tuned configs) | 8689ed6e 33cf3433 f90e1c08 875b4e69 babf99e0 7c43b8bb fc62f154 |
| 01 | partial-batch (unaligned KV length) tile loads branch instead of selecting a pointer to a local zero array -- the gfx1030 no-flat-scratch fault fix (kv=113 / sinks / ALiBi abort class) | cc1179bc |
| 02 | wide tile kernels (ncols >= 16, DKQ <= 256) read Q from a pre-converted global half2 buffer instead of the Q_tmp LDS staging (+30% measured on the 512-column kernel); compile-gated `GGML_CUDA_FATTN_TILE_Q_GLOBAL`, host-gated `q_h2` | 40408c28 4d93e972 |
| 03 | K-quant MMQ float scale (`ggml_cuda_mmq_scale_sumi`) + the Navi21 MMVQ rows_per_block/nwarps table | 1a803b9b 131d89e9 |
| 04 | build fix for RDNA2-only target sets: WMMA guards + no-op stubs in `fattn-qsa3.cu` and `mmb.cu` (see above) | (base-set defect; no halo commit) |

Not ported, with reasons:

- **4d90f69f** (MMVQ float scale) -- a no-op on this base: the 12-block set
  already ships the equivalent `ggml_cuda_mul_small` (24-bit exact mul, full
  rate on AMD) in the MMVQ paths.
- **60ecdf8b + e53cee3e** (sparse tile walk on the FA tile kernel) -- deferred
  to a follow-up add-on: it rewrites the same load/iter regions as block 05 of
  the main set and needs a dedicated pass; it is a performance feature, not a
  correctness one.

## Gates (gfx1030)

- `test-backend-ops --op=FLASH_ATTN_EXT` -- full pass (block 01 fixes the
  kv=113 abort class; halo measured 3982/3982 on a V620 before the set existed).
- `test-backend-ops --op=MUL_MAT` -- full pass.
- Build: `cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release`.

The per-halo-commit V620 numbers (prefill/verify deltas, occupancy aborts) are
in the halo commit messages; the re-run of the two backend-ops gates on this
exact tree is the delivery check.
