# RESOLUTION - the 2-byte band arm joins the 32-column block (TODO #27), and the same sweep found the head-512 row

Date: 2026-10-19. Box: 1x R9700 (**gfx1201**, nsm 32), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`),
`~/bin/build-llama-rocm` (`BUILD_DIR=build-rocm`). Tree: delivery **r91** (`45f817cc3`, tree
`90507dfb...`) as the baseline, **r92** (`47c5b2dd6`, tree `da55e828...`) as the candidate. Model for
the model-level numbers: `Qwen3.8-27B-UD-Q4_K_XL` (65 blocks, 64 dense full-attention layers, head 256,
24 Q / 4 KV heads = GQA 6). Records: this file, `benchmarks/2026-10-19-fa-band-2byte-and-head512.md`,
`WORKLOG.md` 261019.012.

**Status: CLOSED (promoted into block 05 at r92).** The campaign's proposal (add a band row for
`ncols == 16`, keeping the 2-byte arm at `ncols1 = 2`) was implemented, measured and **superseded**: the
right answer is to give the 2-byte arm the *same* 32-column band block as the quantized arm and to lower
its band split. Both changes are in block 05; the `ncols == 16` row is **not** in the delivery.

## 1. Why the README's `ncols == 16` row is not the fix

The `ncols == 16` row was built in its two plausible forms and measured against the r91 generic row
(`test-backend-ops perf`, head 256 / GQA 6, f16 KV, us per call, band split P = 3*nsm/4 = 24):

| row for `ncols == 16` | n_q 1 | n_q 3 | n_q 5 | n_q 8 |
|---|---|---|---|---|
| r91 generic `64, 2, 32, 128, 128, 128, 1, true` (128-half2 K/V) | 104 | 131 | 202 | 221 |
| `128, 1, 64, 64, 64, 64, 1, true` (README proposal; np = 2) | 90 (-14 %) | 151 (**+15 %**) | 165 (-18 %) | 220 (0 %) |
| `128, 2, 64, 64, 64, 64, 1, true` (same, occupancy 2) | 92 | 150 (**+15 %**) | 164 | 219 |

(kv 16384; the same shape holds at 51200/98304.) The regression at `n_q = 3..4` is structural, not a
wave artifact (occupancy 2 did not move it): with `ncols1 = 2` a 3- or 4-token verify launches **two**
output tiles, and each tile streams the whole KV cache. The 64-half2 K/V batch only pays back where the
block is at least half full, i.e. at `n_q = 1`.

## 2. The fix (block 05)

```c
// fattn-rdna.cuh
ggml_cuda_fattn_band_wmma_ncols1()  -> 4 for every K/V type   (GGML_HIP_FA_BAND_WMMA=2 still forces 2)
ggml_cuda_fattn_band_wmma_split()   -> 2-byte: max(2, nsm/2)  (was max(2, 3*nsm/4))
```

The 2-byte arm now uses the band's existing 32-column row
(`(256, 256, 32) = 256, 1, 64, 64, 64, 64, 1, true`) and the wide row at `n_q = 5..8`
(`(256, 256, 64) = 512, 1, 64, 64, 64, 32, 1, true`), which is bit-identical per column to the
32-column block, so `W = 1..8` still reduce identically (the `GGML_HIP_FA_BAND_WIDE` dispatch and the
split P are both width-independent). `ncols1` is not allowed to depend on `n_q` in either arm
(GREEDY-PURITY.md §36), so this is one config for the whole band, not a per-width switch.

## 3. Kernel time (`test-backend-ops perf`, head 256 / GQA 6, us per call, r91 -> r92)

f16 K/V:

| kv | n_q 1 | n_q 2 | n_q 3 | n_q 4 | n_q 5 | n_q 8 |
|---|---|---|---|---|---|---|
| 1024 | 26.6 -> 27.3 | 26.0 -> 27.3 | 29.5 -> 27.7 | 30.3 -> 27.5 | 39.0 -> 44.2 | 45.3 -> 45.6 |
| 4096 | 40.4 -> 40.3 | 40.7 -> 40.9 | 44.5 -> 41.5 | 46.2 -> 42.4 | 68.3 -> 80.7 | 78.7 -> 81.3 |
| 16384 | 106.5 -> 102.2 | 107.8 -> 102.6 | 130.8 -> 105.2 | 133.9 -> 105.9 | 202.3 -> 172.9 | 221.1 -> 175.9 |
| 51200 | 423.0 -> 382.4 | 426.6 -> 383.1 | 423.5 -> 384.5 | 423.5 -> 387.4 | 579.4 -> 456.0 | 614.2 -> 460.8 |
| 98304 | 760 -> 684 | 757 -> 687 | 770 -> 689 | 763 -> 695 | 1061 -> 810 | 1142 -> 823 |

bf16 K/V:

| kv | n_q 1 | n_q 4 | n_q 5 | n_q 8 |
|---|---|---|---|---|
| 4096 | 45.2 -> 41.1 | 48.7 -> 43.0 | 76.6 -> 80.2 | 86.6 -> 80.7 |
| 16384 | 129.1 -> 105.2 | 153.9 -> 108.5 | 254.8 -> 174.2 | 273.0 -> 177.6 |
| 51200 | 439.4 -> 381.8 | 475.1 -> 388.0 | 740.8 -> 459.6 | 782.3 -> 464.6 |
| 98304 | 812.9 -> 685.0 | 869.4 -> 695.8 | 1385 -> 818.6 | 1451 -> 832.0 |

The only regressions are at `kv <= 4096` with `n_q = 5` (+13 % at 1024, +5 % at 4096), i.e. 5 us on a
call that is 25-40 us; the model-level `tg128` at d0/d1024/d4096 is flat (30.69/30.59/30.21 ->
30.74/30.61/30.14). The quantized arm is untouched (q8_0 head 256: 108.4 -> 108.9 us at kv 16384
`n_q = 1`, 199.6 -> 199.9 at `n_q = 8`).

## 4. Model level (27B UD-Q4_K_XL, 1 GPU, `-t 8`)

| measurement | r91 | r92 |
|---|---|---|
| `llama-bench tg128` @ d16384, f16 / bf16 | 29.02 / 28.68 | 29.00 / 28.95 |
| `llama-bench tg128` @ d98304, f16 / bf16 | 22.63 / 21.88 | **23.28 (+2.9 %) / 23.25 (+6.3 %)** |
| `llama-bench tg128` @ d0/d1024/d4096 (f16) | 30.69 / 30.59 / 30.21 | 30.74 / 30.61 / 30.14 |
| deep MTP, 53k prompt, `draft-mtp n3`, bf16 | 63.8 t/s | **68.3 (+7.0 %)** |
| deep MTP, 53k prompt, `draft-mtp n3`, f16 | 64.3 t/s | **68.4 (+6.4 %)** |

Four-axis MTP gate (Protocol A, bf16, `-n 3000`, r91 -> r92, draft-mtp n3 t/s):
R 58.9 -> 60.7, P 72.1 -> 75.3, C 80.1 -> 80.2, K 86.2 -> 86.5; plain decode flat
(30.3-30.6 both arms); acceptance 0.576 / 0.799 / 0.904 / 0.992 -> 0.601 / 0.844 / 0.902 / 0.992 (all
healthy; the small shift is the rounding change, not a collapse).

## 5. Gates

* `test-backend-ops -o FLASH_ATTN_EXT` (ROCm0): **6391/6391 OK, 0 FAIL**.
* `test-logits-width-probe` W = 1..8, `maxdiff_vs_W8 = 0` (PASS): 27B f16 + bf16, 35B-A3B f16 + bf16,
  Flash-Next GSQ-IQ3_XXS 2 GPU f16 + bf16.
* Greedy text: `plain == draft-mtp n3 == n7` on the 27B (f16 and bf16) and on the 4B grid
  (P = 192/200/224/256, f16 and bf16). At 200 tokens the text is also byte-identical to the r91 build
  (f16 `9b2cd33bdacf`, bf16 `60fca529f141`).
* `scripts/gate-prefill-logits.sh`, 27B Q8_0: **0.000707 / 98.755 %**, identical to r91 (prefill does
  not reach the band).
* `scripts/validate-set.sh`: green (12/12 strict apply, tree `da55e828...`).

## 6. What was not done

* The `ncols == 16` row is not in the delivery. `GGML_HIP_FA_BAND_WMMA=2` still forces `ncols1 = 2` for
  A/B; that arm falls through to the generic 128-half2 row again, as it did before r92.
* The quantized arm's split stays at `nsm`. Only the 2-byte arm's P was re-swept (P = 8/12/16/20/24/32
  at kv 1024..98304; P = 16 = `nsm/2` was best or tied everywhere except a 2-3 % loss at `n_q = 1`,
  kv 16384).
* The head-512 prefill row found during the same sweep is a **separate** change, also in block 05:
  see `benchmarks/2026-10-19-fa-band-2byte-and-head512.md` §3 and
  `upstream/UPSTREAM-PR-fa-512-rdna-head-cap.md` (issue #188).
