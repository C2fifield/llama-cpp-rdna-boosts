# rdna4-mmq-select-j: routed MMQ tile widths for RDNA4 again (output unchanged)

One format-patch on top of `v12-097cf33d9-r52`, for `git am`.  It touches `ggml/src/ggml-cuda/mmq.cu` only (+3 / -2).

## What we noticed in r50

In r49, `mul_mat_q_switch_J` applied the rows-per-expert J bands when `mmq_routed_compact_arch_ok(cc)` (RDNA3.5 or
RDNA4), and it passed `GGML_CUDA_CC_IS_RDNA4(cc)` to `mmq_rdna3_5_id_get_J`.  The move into
`ggml_cuda_mmq_select_J` in r50 tests `GGML_CUDA_CC_IS_RDNA3_5(cc)`, and the call no longer passes the `rdna4`
argument (its default is `true`).  As we read it, that has two effects:

- **gfx1201:** the IQ expert types at ~40 rows per expert (ubatch 2048, 512 experts, 10 used) fall through to the
  generic tile-count loop, which picks J = 48 instead of `MMQ_IQ_ID_J_MID` = 64.
- **gfx1151:** it now gets the RDNA4 bands for the IQ2 / Q2_0 types, which r49 kept on the generic path.

The patch restores both: the branch is gated on `mmq_routed_compact_arch_ok(cc)` again, and the `rdna4` flag is
passed through.

## Measurements (2 x R9700 32 GB, gfx1201)

From the r50 kernel traces on our local build (Flash-Next IQ3_S + MTP, `-sm tensor`, 32k-token prompt):
`mul_mat_q_routed_compact` was +110-125 ms of GPU time per GPU against r49.  With this patch it went
2192 -> 2043 ms per prompt on one GPU.  A forced-J A/B with `GGML_MMQ_ID_J_DEBUG` gave 2120 t/s at J 64 against
2098 at J 48.

On r52 itself (IQ3_XXS + MTP, all experts in VRAM, `-sm tensor`, no tuning env, two fresh ~31k-token prompts per
build, one run each):

| build | prefill t/s |
|---|---:|
| r52 | 2575, 2509 |
| r52 + this patch | 2621, 2631 (+2 to +5 %) |

Greedy output is unchanged (`a7a81f01b3bc` both), as are the long-prompt shas on our earlier build: J only changes
the tiling, not the arithmetic per row.

## Notes

- We have no gfx1151 to test the second half on; it restores the r49 selection for that arch, as we read it.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd prefer it done differently.
