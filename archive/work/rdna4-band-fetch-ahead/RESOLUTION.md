# RESOLUTION — `rdna4-band-fetch-ahead` (PR #195)

**Status: PROMOTED (2026-10-20) into `v12-097cf33d9-r98` as a block-05 amendment, default on**
(`GGML_HIP_FA_BAND_FETCH_AHEAD=0` is the kill-switch).  Author: `@overdoingism`.  The submitted
`git am` file, cut against r92, applies to the current tree and is folded into block 05 unchanged
apart from the rebase.

This record exists because the PR arrived as a branch under its own `wip/rdna4-band-fetch-ahead/`;
it was never an entry in this repo's `wip/` index, so no stub was created or removed there.

## What it does

With q8_0 K and V read in place (decode/verify, `n_q <= 8`, head 256) the RDNA4 band launches a second
kernel variant, `flash_attn_ext_f16<..., band_fetch_ahead = true>`:

- The raw q8_0 bytes of the next K/V batches are fetched into registers ahead of use; a batch is converted
  and stored to shared memory only when it is due, and staging a batch fetches the batch `depth` places
  later into the registers it freed.
- The iteration's barriers order shared memory only (release / `s_barrier` / acquire on `"local"`).
  `__syncthreads()` also waits for every outstanding global load, which would drain the fetches.
- The quants stay packed in 32-bit words and an empty `asm volatile` pins the conversion at the stage
  point; without it the compiler unpacks the bytes right after the fetch and needs 4x the registers.
- 64-column block (widths 5..8): two batches in flight, Q in registers.  32-column block (widths 1..4):
  one batch in flight, the stream continues across the iteration boundary into the next unmasked KV
  block, Q read from shared memory.
- Compiled in two generated translation units of its own
  (`template-instances/fattn-mma-f16-band-fetch-ahead-ncols1_{4,8}-ncols2_8-dkq256-dv256.cu`, emitted by
  `generate_cu_files.py`), so the kernels without the variant keep their compiled code.

## Verification (independent of the submitter, gfx1201 / ROCm 10.1.0)

* **Assembly level, the PR's central claim.**  Device objects extracted from both builds and the gfx1201
  ISA diffed with mangled names normalised: **0 differing instructions** in
  `fattn-mma-f16-instance-ncols1_{4,8}-ncols2_8-dkq256-dv256`, `fattn-mma-f16-instance-ncols1_8-ncols2_8-
  dkq128-dv128`, `fattn-tile-instance-dkq256-dv256`, `fattn-vec-instance-f16-f16` and `fattn.cu`.  The
  variant kernel is `W` in its own generated TU and `U` in the two `-case-` TUs that select it, which is
  the FA instantiation rule in `patches/README.md`.
* `test-backend-ops -o FLASH_ATTN_EXT`: **6391 OK / 0 FAIL** on both builds, identical per-case results.
* Prefill-logit KLD gate: **0.000000 / 100.000 %** (27B Q8_0, and with `-ub 8` and `-ub 1` on 27B UD-Q4_K_M).
* Greedy text byte-identical to r97 at 8K and 110K context, on a 2 GPU tensor split, on gemma4 12B (its
  head-256 SWA layers take the variant too) and with the kill-switch off.
* 4-axis adaptive MTP at `-n 3000`: text and draft acceptance identical on all four axes (P 0.78738,
  C 0.91041, K 0.99200, R 0.58584); `n7` is identical to plain, so the 64-column / depth-2 path is covered.
* `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`: B=1 29.47 -> 30.12 t/s, B=4 95.73 -> 97.70, B=8 flat.
* Speed (27B UD-Q4_K_XL, q8_0 K/V, 3 interleaved pairs each): **29.3 -> 29.5 t/s at 8K** (+0.7 %),
  **24.5 -> 25.1 t/s at 110K** (+2.4 %), 45.2 -> 45.6 on the 2 GPU tensor split; prefill unchanged
  (840.0 -> 839.2 t/s at 110K).
* `scripts/check-gpu-faults.sh`: clean.

## Accepted limitations

- Head 256 only (`DKQ == DV == 256`, `ncols2 == 8`), and only the 32/64-column band blocks.  A different
  head size or the prefill widths run the unchanged kernel, which is the point of the separate TUs.
- The q8_0 kernel without the variant is launched through the same compiled code as before, so an A/B is
  available but there is no path that runs the *variant* for non-q8_0 K/V.
- The submitter's end-to-end figures were measured on an earlier single-kernel form; the numbers above are
  the shipped form on this box and are slightly better than their note.

## Re-entry conditions

None.  If a later change gives the band another K/V type the same in-place read, re-evaluate the variant
for it (the fetch-ahead is type-agnostic in shape but the chunk loader is q8_0-specific).
