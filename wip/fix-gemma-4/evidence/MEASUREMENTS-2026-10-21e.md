# Measurements 2026-10-21 (e) — two real defects fixed: `--fit` draft-KV over-reserve, and gemma4 `-sm tensor`

Box: gfx1201, 3x R9700, ROCm 10.1.0, Linux.  Tree: `~/llama.cpp` @ `972f3b757` + `patch-state/0001..0008`.
**Every measurement below was taken with `ROCBLAS_USE_HIPBLASLT=0`** so the per-start hipBLASLt solution
pick (`MEASUREMENTS-2026-10-21d.md`) cannot confound it.  `-t 8`, one job at a time.

## 1. §7 item 1 — `--fit` charged the drafter KV it does not own (FIXED, patch 0007)

**Symptom** (reproduced, default context = 262144): MTP was *slower* than plain, 41.0 vs 74.6 t/s.

**Cause.**  `--fit` cannot *measure* the gemma4 drafter (creating its probe context fails: "Gemma4Assistant
requires ctx_other"), so it falls back to `common_draft_reserve_estimate()`.  That estimator charged the
drafter a full `n_ctx` KV — **18022 of the 19152 MiB** it reserved per device — for a drafter that
allocates **none**: `llama_kv_cache`'s `layer_share_cb` reuses the *target's* tensors
(`layers.push_back(layer_share)`) and overrides `kv_size` to the source's.  The draft GGUF says so:
`gemma4-assistant.attention.shared_kv_layers = 4` (= `block_count`).

**Fix (0007):** the estimator reads `attention.shared_kv_layers` and charges KV only for the layers the
draft owns (`kv_own = kv_layers - shared`, and 0 for `gemma4-assistant`, which shares unconditionally).
The indexer-cache term follows the same count, and the log line reports the owned count.

| gemma4 26B-A4B Q8_0 `-ncmoe 40 -sm layer`, MTP n3, default ctx | before | after |
|---|---:|---:|
| derived draft reserve /device | 19152 MiB | **1129 MiB** |
| MoE arena | 13752.2 MiB | **23141.2 MiB** |
| cache slots | 4564 | **7680** |
| cache hit rate `h` | 0.8969 (176373/196656, 887 evictions) | **0.0000 (0/0, 0 evictions)** |
| gen t/s | **41.0** | **93.2** |

Plain (no drafter) on the same config: 74.6 t/s — so MTP at the default context went from -45 % to
**+25 %** over plain, and the pinned-context reference (93.1 t/s at `-c 16384`) is now the *default*.

Second gemma4 case, `Gemma4 31B-QAT Q4_K_XL` at the default context: plain 25.0 t/s vs **MTP 51.2 t/s**
(2.0x; that model's drafter declares the same `shared_kv_layers = 4`).

**Non-regression (by construction + measured):** any draft GGUF without `shared_kv_layers` and not
`gemma4-assistant` keeps `kv_own == kv_layers`.  Checked on the qwen4exp sidecar
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, arch `qwen4exp`, `nextn_predict_layers = 1`, no
`shared_kv_layers`): the estimate is `weights 2647 + kv 563 + compute 512 + slack 192 = 3914 MiB` with
`(1 x 1 layers)`, i.e. **arithmetically identical to before**.  Embedded MTP heads (35B-A3B) never reach
the estimator (`path.empty()`), so they are untouched.

## 2. §7 item 2 — gemma4 dense + `-sm tensor` aborted in `handle_per_row` (FIXED, patch 0008)

**Repro:** `gemma4 12B Q8_0 -sm tensor` (f16 cache) aborted at the warmup decode:
`ggml-backend-meta.cpp:838: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed`.

**Cause — the row is never straddled.**  New diagnostic print at the abort:

```
PERROWDBG op=RMS_NORM out=Vcur_normed-5 out_ne=[512,1,1,1] src0=Vcur-5 (reshaped) src0_ne=[512,1,1,1]
          axis=0 n_seg=1 nr0=1 ne_per_dev=[0 512] src0_op=RESHAPE
```

gemma4's **full-attention** layers have a single KV head, so their 512-wide K/V projection row is placed
**whole on one device** (`ne_per_dev = [0, 512]`).  The per-row ops over it (the per-head K norm, the V
`rms_norm`, the rope) then see an axis-0 split, and `handle_per_row` rejected *any* axis-0 split.  It only
needs to reject one **straddling** a row (e.g. `[256, 256]`), which really is a cross-device reduction.

**Fix (0008), two parts:**
1. `handle_per_row` accepts an axis-0 split when every per-device extent is either empty or the *whole* row
   (whole rows ⇒ each device reduces locally).  A genuine straddle still asserts, with a message saying it
   needs a cross-device reduction.
2. `handle_bin_bcast` gains the elementwise case: a split tensor times a **MIRRORED, same-shape** operand
   yields the split operand's state (nothing is reduced, and the operand is present in full everywhere).
   Without it the K norm's `MUL` hit `handle_generic` and aborted (`GENERICDBG op=MUL out=Kcur_normed-5
   src0=norm-5 axis=0 src1=blk.5.attn_k_norm.weight axis=10(MIRRORED)`).  The existing broadcast case only
   covered `ne[axis] == 1`.

**Result**

| gate | result |
|---|---|
| `gemma4 12B Q8_0 -sm tensor`, 2 GPU, `-n 64` | **runs**; text **`c740be0e131a`** (143 chars) — byte-identical to `-sm layer` |
| same at **3 GPU** `-sm tensor` | **`c740be0e131a`** (143 chars) |
| gemma4 26B-A4B `-ncmoe 40 -sm tensor`, plain, `-n 300` | **runs**: prompt 182.8, gen **72.2 t/s** (the arm the campaign could not measure at all) |
| gemma4 26B-A4B `-ncmoe 40 -sm layer` plain / MTP n3 | 74.7 / **93.0** t/s |
| `test-llama-archs -s 42` | **all 644 tests passed** |
| rotated q8_0 KV + `-sm tensor` (TODO #68 1a) | **still aborts** at `handle_mul_mat` — unchanged, next item |

## 3. Config nits found on the way (handover, not fixed)

1. **`-md <gemma4-assistant> --spec-type none` aborts** (`llama-graph.cpp:228 GGML_ASSERT(ubatch->embd_state)`)
   — with and without `--fit`.  A drafter loaded but unused still gets a context whose graph needs the MTP
   state embedding.  Repro: `-md mtp-gemma-4-26B-A4B-it.gguf --spec-type none` (any `-c`).
2. `--fit` logs `E llama_init_from_model: failed to initialize the context: Gemma4Assistant requires
   ctx_other` before falling back to the derived estimate.  Harmless (the message says so, and the fallback
   is now correct) but E-level; it should be a warning when the extra model declares shared KV.
3. `test-llama-archs` writes `(file*).moe-heatmap` into the CWD — the same artifact block 09 ships by
   accident; drop it at the promotion cut.
