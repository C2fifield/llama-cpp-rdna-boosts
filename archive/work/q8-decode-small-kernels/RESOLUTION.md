# RESOLUTION -- `q8-decode-small-kernels` (PR #171, @briansp2020)

**Status: RESOLVED (2026-10-16), promoted into `v12-097cf33d9-r77` as **block 08** (both patches).**
The contributor's two format-patches and their write-up are preserved in this directory; nothing was
left behind as a `wip/` stub (the campaign arrived as a PR branch, never on `main`'s `wip/`).

## What landed (block 08)

- `0001`: the q8_0 `SET_ROWS` path (the K/V cache write) runs **one 32-lane group per `block_q8_0`**
  instead of one thread looping over 32 values.  `k_set_rows_q8_0_group`: amax by xor-shuffle
  `fmaxf` seeded with 0 (as `quantize_f32_q8_0_block`), the same `d`, `id = d ? 1/d : 0` and
  `roundf`, the same index math, coalesced byte stores.  Other quant destinations keep the generic
  kernel (`if constexpr (std::is_same_v<block_type, block_q8_0>)`).
- `0002`: `rms_norm_q8_1` folds the dst write into the quantize loop and quantizes from the register,
  dropping the `__syncthreads()` and the dst reload; the `has_add && y == nullptr` early-out becomes a
  uniform `continue`.

No `try_fuse` change and no mmvq band participation (`nwarps`/VDR untouched), so no reduction-order
anchor moves.

## Validation (r77 candidate, gfx1201, 3x R9700)

- **Output byte-identical to r76** in every arm: dense 27B UD-Q4_K_XL 3-GPU tensor `sha 16c09ff04038`
  (6228 chars) and MoE 35B-A3B UD-Q4_K_M `sha 2bd86932038b` (6177 chars), `--spec-type none` ==
  `draft-mtp` (n-max 3) on each, seed 42, temp 0, `-n 3000`, q8_0 KV, prose prompt.
- **MTP acceptance identical**: dense 0.79055 (1940/2454), MoE 0.75653 (1507/1992).
- `test-logits-width-probe` (`SPLIT=tensor`, `KV=q8_0`): **PASS, worst maxdiff 0**, with the r76 hashes
  at P=256 and P=3584 on both models (this exercises the q8_0 SET_ROWS write and the fused
  `rms_norm_q8_1` in the verify band).
- `test-backend-ops -o SET_ROWS` 159/159, `-o RMS_NORM` 53/53.
- Prefill-logit gate: **0.000000 / 100.000 %** (512-ctx base), 0.000000 / 99.996 % (8 k-ctx base).
- Speed: MTP decode **+0.6-0.7 %** (dense 113.3 -> 114.1, MoE 181.2 -> 182.3 t/s); `llama-batched-bench
  -npl 1,4,8` TG B=1 28.19 -> 28.41, B=4 105.27 -> 106.46, B=8 185.88 -> 185.56; `llama-bench` tg128 /
  pp512 within +0.1-0.3 %.  Nothing regressed in the defaults.
- The contributor's own numbers (gfx1100 under WSL2 and gfx1201 native: set_rows 5.1 -> 3.7 us and
  1.51 -> 1.33 us per call, rms_norm_q8_1 6.1 -> 5.8 us and 2.72 -> 2.47 us, decode +1.7 % / +0.6 %)
  are consistent with what this box measures, and their reported greedy hashes / width probe were
  reproduced here exactly.

## Review notes (no action needed)

The tail block exits whole groups early (`if (i >= ne_total) return;`) while the shuffle uses the full
`0xffffffff` mask.  That is safe as written: a CUDA warp is exactly one `QK8_0` group (so a live group
never shares a warp with an exited one) and HIP ignores the mask operand, using the wave's exec mask.
No change was made to the contributor's tested code.

## Record

`WORKLOG.md` 261016.012 (r77); `patches/README.md` (current-release block); `ENVIRONMENT.md` unchanged
(no new variable).
