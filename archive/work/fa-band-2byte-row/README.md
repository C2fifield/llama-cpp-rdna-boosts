# `fa-band-2byte-row` — extend the RDNA4 GQA band's 64-wide K/V row to the 2-byte arm (TODO #27)

**Status: CLOSED at r92 (promoted into block 05).** The plan below (add a `ncols == 16` band row) was
measured and superseded; the delivered fix gives the 2-byte arm the 32-column band block and lowers its
band split. See [`RESOLUTION.md`](RESOLUTION.md) for the numbers, the gates and what was not done.
Kept as the campaign's original brief; it is no longer work in flight. Same item as **TODO #27**. The predecessor campaign
[`archive/work/rdna4-fa-band/`](../../archive/work/rdna4-fa-band/) added the `ncols == 32` band row; its
`VERIFICATION-r21.md` §"One factual correction" identified this as the follow-up. This directory is the
live campaign.

> **Cold start.** Read this file, then `AGENTS.md` ("Report findings before shipping", "WIP and promotion
> rules", "Critical facts" FA bullet), `GREEDY-PURITY.md` §36 (dense greedy purity) and §24/§25 (band
> knobs), then
> [`archive/work/rdna4-fa-band/VERIFICATION-r21.md`](../../archive/work/rdna4-fa-band/VERIFICATION-r21.md)
> and [`archive/work/rdna4-fa-band-wide/README.md`](../../archive/work/rdna4-fa-band-wide/README.md). Report
> findings to the maintainer before any delivery change and wait.

## 0. The gap

The RDNA4 GQA decode/verify **band** (`flash_attn_ext_f16<256, 256, …, 8>`, `n_q <= 8`, GQA ratio
4 < r <= `GGML_HIP_FA_BAND_WMMA_GQA_MAX`) folds the GQA group into `ncols2 = 8` and has dedicated MMA
config rows in `ggml_cuda_fattn_mma_get_config_rdna()` (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`):

```c
if (is_rdna4 && ncols2 == 8 && DKQ == 256 && DV == 256 && ncols == 32) {   // ncols1 = 4 (quantized arm)
    return fattn_mma_config(256, 1, 64, 64, 64, 64, 1, true);              // 64-half2 K/V batches, 8 warps
}
if (is_rdna4 && ncols2 == 8 && DKQ == 256 && DV == 256 && ncols == 64) {   // ncols1 = 8 (verify widths)
    return fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true);              // 16 warps, one KV pass
}
```

The **2-byte (f16/bf16) K/V arm** selects `ncols1 = 2` (`ggml_cuda_fattn_band_wmma_ncols1()` returns 2 when
`ggml_cuda_fattn_band_wmma_two_byte(dst)`), so it compiles to **`ncols = 16`** and matches **no** band row.
It falls through to the generic `256, 256, 16` row and therefore keeps the **128-half2** K/V load batches
(the row the 64-wide band work replaced for the quantized arm). The r21 measurement of the 2-byte arm was
noise (`f16 KV tg128 @ d16384`: 27.50 → 27.57 t/s), i.e. it did **not** get the band's 3–4 % — the follow-up
is a real, small hole.

## 1. Current reality check (r48)

- The band now **includes** the 2-byte arm: `ggml_cuda_fattn_band_wmma_applies()` requires
  `kv_quant || kv_f16`, and `ggml_cuda_fattn_band_wmma_two_byte()` covers `GGML_TYPE_F16` **and**
  `GGML_TYPE_BF16`. Native BF16 is default-on (r6), so both are live. The `ncols1 = 2` choice is
  deliberate: the 2-byte arm has no dequantisation to hide unused columns, and the measured 2-column tile
  is much cheaper than 4 columns at `n_q = 1` (747 µs vs 967 µs, head 256 GQA 6 kv 102400).
- r46 (#133) raised the GQA cap to 12 (default); the band's `gqa_ratio > 4` lower bound is unchanged, so
  GQA 6 (Qwen3.8-27B) and GQA 12 (qwen4exp) both reach it.
- r45/r46 added `GGML_HIP_FA_BAND_WMMA_SPLIT` for the 2-byte arm (default `max(2, 3*nsm/4)` blocks per
  output tile). Whatever row is added must not change the **split** choice — `GGML_HIP_FA_BAND_WMMA_SPLIT`
  and the band predicate must stay width-independent (decode/verify must split identically).
- `GGML_HIP_FA_BAND_WMMA=2` forces `ncols1 = 2` even for quantized types — a useful way to isolate the
  `ncols == 16` row's effect without depending on the K/V type.
- The band's `ncols1` is part of the **width-purity contract**: `W = 1..8` must keep one logits hash
  (`GREEDY-PURITY.md` §36). The row sits *inside* the band, so it must not change the W-dependence.

## 2. The change to try

Add the missing row next to the existing two (the exact config is a tuning parameter to sweep, not a
given):

```c
// 2-byte (f16/bf16) arm: ncols1 = 2 -> ncols = 16
if (is_rdna4 && ncols2 == 8 && DKQ == 256 && DV == 256 && ncols == 16) {
    return fattn_mma_config(/* nthreads */ 128, /* occupancy */ 1,
                            /* nbatch_fa */ 64, /* nbatch_K2 */ 64, /* nbatch_V2 */ 64,
                            /* nbatch_combine */ 32, 1, true);
}
```

Rationale for the starting point: keep the band's 64-half2 K/V batches and the `ncols == 32` row's
`nbatch_combine` discipline, but keep 4 warps (128 threads) so each of the 2 Q columns still has the same
KV-row split as the 32-column block would give it. Sweep `nthreads` 64/128/256, `nbatch_combine`, and
`occupancy`; the goal is the band's 64-half2 K-batch win at `n_q <= 4` without regressing the 5–8 verify
widths (those use `ncols1 = 2` twice, i.e. two 16-column tiles, because `band_wide` applies only to the
`ncols1 == 4` arm — check whether a 16-column-wide variant is worth it too).

Constraints to preserve:

- **Bit-exactness if claimed:** the `ncols == 32` row is bit-exact because the 256-dim non-MLA K batches
  iterate forward (`!V_is_K_view && DKQ == 256`). Confirm the same holds for the new row's arity before
  claiming bit-exactness; otherwise it is a **rounding** change and must be shown W-pure.
- **Width purity:** `W = 1..8` one hash per tensor (the row must not make the split depend on `n_q`).
- Only the band matches the row (the prefill `ncols == 16` instances use `ncols2 != 8`, so the guard is
  already band-only).

## 3. Measurement

Per-call microbench (the r21/r24 method), one R9700, gfx1201:

```bash
cd ~/llama.cpp
cmake --build build-rocm-hybrid --target test-backend-ops -j 16
HIP_VISIBLE_DEVICES=0 ./build-rocm-hybrid/bin/test-backend-ops perf -o FLASH_ATTN_EXT \
  | grep -A2 -i 'f16'     # head 256, GQA 6, kv 1024..102400, n_q 1..8
```

Then the model-level check with an **f16** KV cache (the 2-byte arm), Qwen3.8-27B-UD-Q4_K_XL:

```bash
# deep-context single-token decode: the band's worst case
HIP_VISIBLE_DEVICES=0 ./build-rocm-hybrid/bin/llama-bench -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -p 0 -n 128 -d 16384
# and the verify width (n_q 4..8) via the MTP gate
```

Also profile which kernel runs for the f16 case (the row is only reached if the band actually selected the
MMA kernel): the existing `GGML_HIP_FA_BAND_WMMA=2` force plus a kernel trace will show whether the new row
is taken. Confirm the f16 band is even on for the model/head shape (it needs `Q->ne[0] == V->ne[0] == 256`
and `gqa_ratio > 4`; the 27B has 24 Q / 4 KV heads = 6).

## 4. Gates

- `test-backend-ops` full `FLASH_ATTN_EXT` (backend correctness) must stay green.
- `test-logits-width-probe` on the affected models at `W = 1..8` — one hash across widths, and identical
  to stock if the row claims bit-exactness.
- The dense `f16/bf16/q8_0` greedy text is kept by `GREEDY-PURITY.md` §36; re-run the greedy A/B for f16 KV.
- The adaptive-MTP gate (`benchmarks/mtp-adaptive-methodology.md`, verify-width F16 arm) must not regress.
- `scripts/validate-set.sh` if promoted; add any new env knob to `ENVIRONMENT.md` (or none, if the row is
  unconditional).

## 5. Pointers

- `TODO.md` #27.
- [`archive/work/rdna4-fa-band/README.md`](../../archive/work/rdna4-fa-band/README.md) +
  `VERIFICATION-r21.md` (the `ncols == 32` row and the "One factual correction" that defines this item).
- [`archive/work/rdna4-fa-band-wide/README.md`](../../archive/work/rdna4-fa-band-wide/README.md) (the
  `ncols == 64` row; note it says "The 2-column f16/bf16 mode is untouched").
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh:130-150` (the band rows), `fattn-common.cuh`
  (`ggml_cuda_fattn_band_wmma_ncols1` / `_two_byte` / `_split` / `_applies`), `fattn.cu:201-250`
  (the dispatch and `GGML_HIP_FA_BAND_WIDE`).
- `GREEDY-PURITY.md` §36 (dense greedy purity) and §24/§25 (band-uniform knobs).
