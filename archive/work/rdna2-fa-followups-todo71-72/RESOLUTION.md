# RESOLUTION — `rdna2-fa-followups-todo71-72` (PR #198) — PROMOTED into block 05 (r99)

**Status: PROMOTED (2026-10-20) into `v12-097cf33d9-r99` as a block-05 amendment**, not as the separate
`patches-rdma2/` add-on channel the PR proposed.  Author: `@slavap`.  The submitted patch (cut against r93)
is kept beside this file; it applied to the r99 tree unchanged.

## Why folded rather than a second channel

The delivery is a single 12-block series: `release.json` describes it, `scripts/apply-all.sh` applies it,
`scripts/validate-set.sh` (and CI) verify it, and the tags and container images carry it.  A parallel
`patches-rdma2/` directory plus its own `release-rdma2.json` would be a second delivery that none of those
check, and RDNA2 work has been folded into block 05 before (PR #181 -> r88, PR #165 -> the r64-era set).  The
changes here are arch-neutral refactoring plus an RDNA2-gated kernel change, so block 05 is their owner.

## What landed (block 05)

1. **One tile K/V type selection (TODO 71).**  `ggml_cuda_fattn_tile_type(K, V, cc)` in `fattn-rdna.cuh` is
   now the single source for both the instantiation (`ggml_cuda_flash_attn_ext_tile_case_type`) and the F16
   scratch sizing (`ggml_cuda_flash_attn_ext_get_alloc_size`), which had drifted three times.
2. **Checked tail for the last unchecked tile (TODO 72), RDNA2 only.**  The `ncols2 > 1` tile branch's last
   tile (`k_VKQ_0 + nbatch_fa > k_VKQ_max`) takes the already-compiled `oob_check = true` form so an
   exact-size operand cannot read into the neighbouring allocation.
3. **The 511-row guard tail is kept** and its derivation documented (`nbatch_fa <= 256`, over-read
   `<= 255`, 511 = 2 x 256 - 1).  Comment only.

## Non-RDNA2 reach — verified here, because the maintainer's rule is that the three supported arches must
not move unless it is proven they do not

| delta | reach | proof |
|---|---|---|
| `fattn-tile.cuh` checked tail | gfx103x device pass only | `RDNA2` is defined only for `__gfx103x__` in `vendors/hip.h`; the added block is inside `#if defined(RDNA2)`; checked with the device-pass preprocessor on gfx1030/gfx1100/gfx1151/gfx1201 |
| `fattn-common.cuh` | comments only | the `if (guard_tail)` code is untouched |
| `fattn-rdna.cuh` / `fattn-tile.cu` / `fattn.cu` (host) | all arches | the selection is 1:1 with the previous order (`bf16_mma_hardware_available` BF16 first, then the native dequant type, `FATTN_KV_NATIVE_BF16` -> F16, else F16); the tile instantiation set is unchanged (96 objects before and after); the four tile TUs checked on gfx1201 compile to **byte-identical ISA** (0 differing instructions over 108k-168k instructions each) |

Gates on gfx1201 (our release config): `test-backend-ops -o FLASH_ATTN_EXT` 6391/6391 with identical per-case
results, prefill-logit KLD 0.000000 / 100.000 %, greedy text identical to r98, host-expert greedy text
identical, `validate-set.sh` green, per-block build green for the amended block 05.  A `gfx1201;gfx1030`
dual-target build compiles the RDNA2 arm.

## Accepted limitations

- The RDNA2 runtime gate is the reporter's, not ours: Radeon Pro V620 (gfx1030, ROCm 7.2.4)
  FLASH_ATTN_EXT 6389/6389 and MUL_MAT 2444/2444 with the add-on, the same on a 7900 XTX (gfx1100) via the
  dual-arch binary, and the r93 baseline passing the same two gates.  We have no RDNA2 hardware.
- gfx1101-gfx1102 are inside the `RDNA2` predicate but not explicitly validated on hardware
  (`TODO.md` #79 covers the same gap for the RDNA3_0 work).
