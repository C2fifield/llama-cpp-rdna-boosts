# RESOLUTION — `rdna3-mmvq-q4k-vdr4` (PR #162, @briansp2020)

**Status: RESOLVED (2026-10-16), promoted into the delivery as part of `v12-097cf33d9-r69`.**  The
PR's format-patches and write-up are preserved in this directory; the new `wip/` copy was never left
behind as a stub.

## What landed

Four format-patches on top of r67, folded into the block that owns each concern (delivery stays a
12-block set):

- **block 07** (`ggml/src/ggml-cuda/vecdotq.cuh`): 0002's 32-element dense Q4_K dot product
  `vec_dot_q4_K_q8_1_vdr4_dense` (VDR 4, one integer `dp4a` chain per sub-block), plus 0004's
  `GGML_CUDA_Q4K_DENSE_VDR4` guard so the function is reachable from the RDNA4 opt-in.
- **block 08** (`ggml/src/ggml-cuda/mmvq-rdna-params.cuh`): 0001's 2 rows per block for Q6_K verify
  batches on gfx1100, and 0003's 4 -> 2 rows for Q4_K verify batches with the 32-element dot product.
- **block 10** (`mmvq.cu`, `mmvq-vecdot.cuh`): 0002's `get_vdr_mmvq_dense()` host/device VDR selection
  and the `vec_dot_q_cuda_t` switch (the last block that owns those files), plus 0004's RDNA4
  build-time opt-in (`-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4`, **off by default**).

`ggml_hip_RDNA4_Q4K_DENSE_VDR4` is a **compile-time** switch (the choice is in a `constexpr __device__`
and the host K-split must match), so it cannot follow the delivery's "default-on, env `=0` disables"
convention.  It is kept as an explicit build-time opt-in, exactly as the PR proposed.

## Validation (r69 candidate)

- **gfx1100** (fingon, native Linux, ROCm 7.14, effective x4 link): Qwen3.8-27B UD-Q4_K_M, MTP draft
  `n_max 3`, code prompt with **identical text** (`5b98d5f7e862`): r68 baseline `101.4 / 101.6` ->
  candidate `103.7 / 103.5` t/s (**+2.1 %**).  `llama-bench tg128` interleaved `42.00 -> 42.27`
  (**+0.6 %**, the earlier `29.77 ± 1.24` row was a first-run warm-up artifact; the settled
  `build-pr` value is `30.84 ± 0.03` on the R9700 sweep below).  `none == draft-mtp n3` byte-identical
  (`5b98d5f7e862`); width probe **PASS** (worst maxdiff 0) on both r68 and the candidate.
- **gfx1201** (3x R9700): 0001-0003 are RDNA3_0-gated no-ops.  The no-flag candidate is byte-identical
  to the pre-existing r65/r68 numerics (27B Q4_K_XL plain greedy `f05cec89f4cb` x3, 4B `1c5d32ac537d`,
  MTP verify `36f05bf6f888`); `none == draft-mtp n3` (`36f05bf6f888`); width probe **PASS** (worst
  maxdiff 0); prefill-logit gate **PASS** (mean KLD 0.000707 / same-top-p 98.755 %, identical to the
  recorded base).  The 0004 flag measured only `30.84 -> 30.93` tg128 (about +0.3 %, inside noise) and
  changes greedy text on some prompts (`f05cec89f4cb` vs `3c109f236a3f`), deterministically; it stays
  off by default.
- **Determinism.**  All builds are deterministic run-to-run (plain greedy 3x, MTP 2x).  A decode-path
  `-ub 8` perplexity self-comparison `4.2-5.7 %` was root-caused to the pre-existing r68
  `GGML_CUDA_ASYNC_INPUT` default-on, not to this PR (see the r69 WORKLOG entry); r69 reverts that
  default to off.

**Open caveat carried forward:** 0001/0002/0003 are validated here on native Linux gfx1100 and gfx1201;
CUDA and RDNA3_5 were not built.  The 0004 RDNA4 path is compile-checked and A/B'd on one R9700 only.

## Record

`WORKLOG.md` 261016.001 (r69).  Original PR description and patches: `README.md`, `0001`-`0004` above.
