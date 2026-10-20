# FA band 2-byte arm + head-512 prefill row (2026-10-19, r92)

Dated record for `WORKLOG.md` 261019.012. Two flash-attention changes, both in **block 05**:

1. the RDNA4 GQA band's **2-byte (f16/bf16) arm** moves from a 2-column to the 32-column block
   (TODO #27 / `archive/work/fa-band-2byte-row/`);
2. the RDNA `(512, 512, 32)` MMA row is retuned, which restores head-512 prefill on RDNA4
   (issue #188).

## Environment

* 1x R9700 (gfx1201, nsm 32), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), `~/bin/build-llama-rocm`
  (`BUILD_DIR=build-rocm`), `-t 8` for every model run (the GPU IRQs own cores 13-15).
* Baseline **r91** (`45f817cc3`, tree `90507dfb...`), candidate **r92** (`47c5b2dd6`, tree
  `da55e828...`). A/B is interleaved on the same box; kernel numbers are `test-backend-ops perf`, model
  numbers are `llama-bench` (`-r 3`, `--no-warmup` where noted).
* Kernel shape for the band: head 256, 4 KV heads, 24 Q heads (GQA 6), mask, `kv_view` (the Qwen3.8-27B
  full-attention layer). Kernel shape for head 512: D=512, 4 KV heads, 32 Q heads, nb=512 (the
  Gemma 4 31B global layer at ubatch 512).

## 1. Band 2-byte arm: `ncols1 = 4` + split `nsm/2` (was `ncols1 = 2`, split `3*nsm/4`)

The campaign's `ncols == 16` row, measured at kv 16384 (r91 generic row as the reference):

| row for `ncols == 16` | n_q 1 | n_q 3 | n_q 5 | n_q 8 |
|---|---|---|---|---|
| r91 generic `64, 2, 32, 128, 128, 128, 1, true` | 104 | 131 | 202 | 221 |
| `128, 1, 64, 64, 64, 64, 1, true` | 90 | 151 | 165 | 220 |
| `128, 2, 64, 64, 64, 64, 1, true` | 92 | 150 | 164 | 219 |

2-column arms need `ceil(n_q/2)` output tiles and each tile streams the whole KV cache, so the win at
`n_q = 1` is bought at `n_q = 3..8`. The delivered change instead folds the arm into the existing
32-column row (`(256, 256, 32)`, `bit-exact` per column with the wide row used at `n_q = 5..8`) and
sets the 2-byte split to `nsm/2`.

Full matrix (f16 and bf16, us per call):

| kv | type | n_q 1 | n_q 2 | n_q 3 | n_q 4 | n_q 5 | n_q 8 |
|---|---|---|---|---|---|---|---|
| 1024 | f16 | 26.6 -> 27.3 | 26.0 -> 27.3 | 29.5 -> 27.7 | 30.3 -> 27.5 | 39.0 -> 44.2 | 45.3 -> 45.6 |
| 4096 | f16 | 40.4 -> 40.3 | 40.7 -> 40.9 | 44.5 -> 41.5 | 46.2 -> 42.4 | 68.3 -> 80.7 | 78.7 -> 81.3 |
| 16384 | f16 | 106.5 -> 102.2 | 107.8 -> 102.6 | 130.8 -> 105.2 | 133.9 -> 105.9 | 202.3 -> 172.9 | 221.1 -> 175.9 |
| 51200 | f16 | 423.0 -> 382.4 | 426.6 -> 383.1 | 423.5 -> 384.5 | 423.5 -> 387.4 | 579.4 -> 456.0 | 614.2 -> 460.8 |
| 98304 | f16 | 760 -> 684 | 757 -> 687 | 770 -> 689 | 763 -> 695 | 1061 -> 810 | 1142 -> 823 |
| 1024 | bf16 | 27.9 -> 26.3 | 27.4 -> 26.1 | 30.4 -> 26.4 | 30.8 -> 27.2 | 40.3 -> 44.8 | 47.6 -> 46.1 |
| 4096 | bf16 | 45.2 -> 41.1 | 44.8 -> 41.6 | 47.6 -> 42.0 | 48.7 -> 43.0 | 76.6 -> 80.2 | 86.6 -> 80.7 |
| 16384 | bf16 | 129.1 -> 105.2 | 132.1 -> 105.4 | 152.4 -> 107.4 | 153.9 -> 108.5 | 254.8 -> 174.2 | 273.0 -> 177.6 |
| 51200 | bf16 | 439.4 -> 381.8 | 442.8 -> 384.3 | 470.0 -> 385.7 | 475.1 -> 388.0 | 740.8 -> 459.6 | 782.3 -> 464.6 |
| 98304 | bf16 | 812.9 -> 685.0 | 819.0 -> 687.4 | 860.8 -> 690.0 | 869.4 -> 695.8 | 1385 -> 818.6 | 1451 -> 832.0 |

Regressions are confined to `n_q = 5` at `kv <= 4096` (+13 % at 1024, +5 % at 4096, i.e. ~5 us per call);
model-level `tg128` at d0/d1024/d4096 is flat. The quantized arm is unchanged (q8_0 head 256:
108.4 -> 108.9 us kv 16384 `n_q = 1`, 199.6 -> 199.9 `n_q = 8`).

Split sweep for the 2-byte arm (f16, P = 8/12/16/20/24/32), 4-column block:

| kv / n_q | P 8 | P 12 | P 16 | P 20 | P 24 | P 32 |
|---|---|---|---|---|---|---|
| 16384 / 1 | 119 | 105 | **102** | - | 125 | 126 |
| 16384 / 8 | 143 | 186 | 175 | - | 192 | 211 |
| 98304 / 1 | 705 | 685 | **685** | 812 | 727 | 727 |
| 98304 / 8 | 803 | 996 | **824** | 942 | 843 | - |

P = 16 (`nsm/2`) is best or tied except a 2-3 % loss at kv 16384 `n_q = 1`; P must stay width- and
KV-length-independent, so a single value is required.

## 2. Band model numbers (27B UD-Q4_K_XL, 1 GPU)

| measurement | r91 | r92 |
|---|---|---|
| `tg128` @ d0 / d1024 / d4096 (f16) | 30.69 / 30.59 / 30.21 | 30.74 / 30.61 / 30.14 |
| `tg128` @ d16384 f16 / bf16 | 29.02 / 28.68 | 29.00 / 28.95 |
| `tg128` @ d98304 f16 / bf16 | 22.63 / 21.88 | **23.28 / 23.25** |
| deep MTP 53k prompt, bf16 / f16 | 63.8 / 64.3 | **68.3 / 68.4** |

Four-axis Protocol A gate (bf16, `-n 3000`, `--spec-draft-n-max 3`): `draft-mtp` t/s R 58.9 -> 60.7,
P 72.1 -> 75.3, C 80.1 -> 80.2, K 86.2 -> 86.5; `plain` 30.3-30.6 both arms; acceptance 0.576 / 0.799 /
0.904 / 0.992 -> 0.601 / 0.844 / 0.902 / 0.992 (healthy; the shift is the rounding change).

## 3. Head-512 prefill row (issue #188)

The `(512, 512, 32)` row is the only head-512 MMA row the AMD dispatcher can reach
(`switch_ncols1` forces `ncols = 32/ncols2` for `DKQ > 256`).  The 2026-09-14 RDNA sweep replaced
upstream's `128, 2, 32, 128, 128, 128, 1, true` with `256, 2, 128, 96, 16, 16, 1, true`.  Candidates
measured on the Gemma 4 31B global-attn shape (ms/call, q8_0 KV at kv 98304 / f16 at kv 98304):

| row | q8_0 98304 | f16 98304 | q8_0 32768 | f16 32768 |
|---|---|---|---|---|
| r91 `256, 2, 128, 96, 16, 16, 1, true` | 377.0 | 374.2 | 126.1 | 125.3 |
| `256, 2, 128, 96, 64, 16, 1, true` | 344.8 | 343.8 | 114.9 | - |
| `128, 2, 32, 128, 128, 128, 1, false` | 153.2 | 180.2 | 46.9 | 60.9 |
| `128, 2, 32, 128, 128, 128, 1, true` | 127.5 | 127.5 | 42.2 | 42.7 |
| `128, 2, 32, 128, 128, 64, 1, true` (**r92**) | **128.1** | **127.9** | **42.3** | **43.1** |
| `128, 2, 64, 128, 128, 128, 1, true` | 223.6 | 222.3 | 74.2 | 73.4 |
| `128, 2, 32, 64, 64, 64, 1, true` | 166.2 | 164.8 | 55.6 | 59.4 |
| `GGML_CUDA_FA_WMMA_MAX_HEAD=256` (tile kernel) | 283.6 | 250.9 | 92.4 | 84.3 |

Across depths (r92, q8_0 / f16, ms): kv 4096 5.29 / 5.37, 8192 10.7 / 10.7, 16384 20.9 / 21.0,
32768 42.3 / 43.1, 98304 128.1 / 127.9 (tile at 4096/8192/16384: 12.0/23.5/46.7 q8_0,
10.9/21.3/42.4 f16).

End to end (`llama-bench pp512 -ub 512 -b 1024 -ctk/-ctv q8_0 -r 2`):

| model | d0 | d32768 | d98304 |
|---|---|---|---|
| Gemma-4-31B QAT Q4_K_XL | 1132.6 -> 1181.6 (+4.3 %) | 298.7 -> 582.8 (+95 %) | 122.2 -> 298.5 (+144 %) |
| Gemma-4-26B-A4B QAT Q4_K_XL | 4514 -> 4684 (+3.8 %) | 1184 -> 2320 (+96 %) | 486 -> 1171 (+141 %) |

`tg32` on the 26B: 113.2 / 101.0 / 87.6 -> 112.4 / 101.7 / 87.9 (flat; head-512 decode stays on tile).

### Prefill-logit: gemma4-12b is not a usable instrument for head 512

| arm (r92 build) | KLD vs r91 base | same top-p | PPL |
|---|---|---|---|
| r92 default (new 512 row) | 0.0957 | 90.1 % | 224.55 |
| tile kernel (`GGML_CUDA_FA_WMMA_MAX_HEAD=256`) | 0.1180 | 89.1 % | 223.92 |
| r92 default vs tile (tile base) | 0.1286 | 88.8 % | - |
| r91 default (old row) | - | - | 225.16 |
| `-fa off` (exact reference) | - | - | 220.30 |

Every legal head-512 implementation differs from every other by 0.10-0.13 mean KLD on this model, while
PPL (all inside the +/-12.0 standard error) and 120-token greedy text against the `-fa off` reference
(473 chars reference, 472 tile, 485 r92, 342 r91) put r92 at least as close as the row it replaces. The
canonical instruments (27B Q8_0 0.000707 / 98.755 %, Flash-Next) have no head-512 layers. Recorded as a
property of that model, not a threshold change.

## Verdict

Both changes ship in block 05. Gates: `FLASH_ATTN_EXT` 6391/6391 OK; `test-logits-width-probe` PASS on
27B/35B-A3B/Flash-Next (f16 and bf16); `plain == draft-mtp n3/n7` on 27B and the 4B P-grid (and
byte-identical to r91 at 200 tokens); prefill-logit 27B Q8_0 0.000707 / 98.755 %; `validate-set.sh` green.
