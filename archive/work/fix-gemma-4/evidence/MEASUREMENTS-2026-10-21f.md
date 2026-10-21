# Measurements 2026-10-21 (f) — session d: TODO #68 1a fixed, a multi-sequence `-sm tensor` defect found and fixed, two config nits

Box: gfx1201, 3x R9700, ROCm 10.1.0, Linux, `-t 8`.  Tree: `~/llama.cpp` @ `972f3b757` +
`../patch-state/0001..0012`.  **Every number was taken with `ROCBLAS_USE_HIPBLASLT=0`** (the per-start
hipBLASLt solution pick, issue #67 — `MEASUREMENTS-2026-10-21d.md`).  One GPU job at a time; the
journal and every run log are clean (`scripts/check-gpu-faults.sh --since "4 hours ago"`).

Note on this session's comparisons: the campaign's earlier per-cell commands were not all recorded, so
the gate run below re-established its own references — text hashes are compared **between builds and
split modes measured now** (never against a number in a doc), and throughput is compared against
`build-r99` on the same box.

---

## 1. TODO #68 1a — rotated q8_0 KV under `-sm tensor` (FIXED, patch 0009)

**Repro** (unchanged before the fix): `gemma4 12B Q8_0 -sm tensor -ctk q8_0 -ctv q8_0` aborted at

```
ggml-backend-meta.cpp: unsupported mul_mat split states:
  src0=Meta#attn_inp_k_rot#0 axis=10 (MIRRORED)  src1=Kcur_pos-5 (reshaped) axis=0
```

**Diagnosis** (with a throwaway print of both operands' full split states):

```
src0 = attn_inp_k_rot   ne=[512,512,1,1]  axis=10 (MIRRORED)  n_seg=1
src1 = Kcur (reshaped)  ne=[512,2,1,1]    axis=0              n_seg=1  seg0 ne_per_dev=(0,512)
out    = node_291       ne=[512,2,1,1]
```

`n_rot_k = 512` (the Hadamard rotation of a 512-wide full-attention head).  The rotation is
`out[rot_row, i1] = sum_k rot[k, rot_row] * cur[k, i1]`, so **axis 0 of `cur` is the contraction**.
`ne_per_dev = (0, 512)` — gemma4's single-KV-head row lands **whole on one device** (the same shape 0008
established for the per-row ops), so no device straddles the contraction: the owning device holds every
`rot` column (src0 is MIRRORED) and computes the complete dot product, the other holds an empty slice and
its node is disabled by the zero-sized-source rule.  Nothing is summed across devices.

**Fix (0009).**  `handle_mul_mat` accepts (MIRRORED src0, axis-0 src1) when every per-device extent is
empty or the whole contraction **and** `out->ne[0] == src1->ne[0]` (a square contraction, so each
device's output slice maps 1:1 onto its contraction slice), and passes the split through.  A straddling
split still aborts, now with a message saying it needs a cross-device reduction.

**Gates.**

| gate | result |
|---|---|
| 12B `-sm tensor -ctk q8_0 -ctv q8_0`, 2 GPU, `-n 64` | runs; **`c740be0e131a`** (143 chars) == `-sm layer` |
| same, 3 GPU | **`c740be0e131a`** == `-sm layer` |
| 12B f16 KV, 2 GPU, `-sm tensor` (re-check that 0008 still holds) | **`c740be0e131a`** == `-sm layer` |
| prefill-logit, rotated q8_0, 1 GPU, vs the recorded base | **-0.000000 / 99.990 %** — reproduces the record |
| prefill-logit, f16, 1 GPU, vs the recorded base | **-0.000000 / 100.000 %** — reproduces the record |

**Why the prefill-logit gate runs single-device** — measured, not assumed.  The gate's recorded bases are
1-GPU `build-r99` runs, and a `-sm tensor` run is **not** numerically comparable to a 1-GPU one (the
internal all-reduce BF16-round-trips).  Same batching (`-c 512 -b 512 --chunks 40`):

| comparison | mean KLD | same top p |
|---|---:|---:|
| `-sm layer` @ `-b 512` vs the `-b 2048` base (batching artifact only) | 0.0854 | 92.1 % |
| `-sm layer` 2 GPU vs 1 GPU, both `-b 512` | 0.0854 | 92.1 % (identical: layer split is bit-identical) |
| `-sm tensor` f16 vs the `-b 2048` base | 0.2193 | 86.8 % |
| `-sm tensor` rotated q8_0 vs the `-b 2048` base | 0.2063 | 86.0 % |
| **`-sm tensor` f16 vs `-sm layer` f16, both `-b 512`** | **0.2134** | 86.7 % |
| **`-sm tensor` rotated vs `-sm layer` rotated, both `-b 512`** | **0.2041** | 86.0 % |
| `-sm tensor` f16 vs `-sm layer`, default `-b 2048` (n_seq 4) | 0.2048 | 87.1 % |
| `-sm tensor` rotated vs `-sm layer`, default `-b 2048` (n_seq 4) | 0.2137 | 86.0 % |
| rotation ON vs OFF, `-sm tensor`, same cache | 0.2275 | 85.9 % |
| rotation ON vs OFF, 1 GPU (no split at all) | 0.2516 | 85.3 % |

Read: the ~0.21 KLD is the **tensor split itself** (it is identical for f16, which has no rotation), the
rotation's own effect (its cache quantization) is ~0.25 and **split-mode independent**, and the rotated
`-sm tensor` path is indistinguishable from the f16 one (0.2041 vs 0.2134).  So the rotation adds no
divergence beyond what `-sm tensor` costs for this model, and 0009 does not change the single-device
reference (the recorded base reproduces to 6 digits).

## 2. New defect found via 1a: a multi-sequence batch under `-sm tensor` (FIXED, patch 0010)

With 1a fixed, the rotated-cache **prefill** path hit a second abort — and so did the plain f16 cache, so
it is not a rotation problem:

```
ggml-backend-meta.cpp: GGML_ASSERT(kv_split || kv_mirrored) failed     (handle_flash_attn_ext)
  q axis=2 | K=cache_k_l5 (view)(permuted) ne=[512,256,1,4] axis=0 | V=cache_v_l5 … axis=0 | mask MIRRORED
```

**Trigger: an ubatch that carries more than one sequence** (`n_stream > 1`).  Fast repro:

```bash
llama-batched-bench -m gemma4-12B-Q8_0 -c 2048 -b 2048 -ub 512 \
  -npp 512 -ntg 0 -npl 1,4 -ngl 99 -sm tensor -fa on -t 8      # -npl 4 aborts
llama-server … -sm tensor -np 2                                 # two overlapping requests abort
llama-perplexity … --kl-divergence                              # n_batch 2048 / -c 512 -> n_seq 4
```

`-npl 1` (and every `llama-cli` run so far, and decode) is fine: with one stream the FA K/V cache view
maps the cache's axis-0 split onto the **kv-head** axis (`[0,1]`, accepted); with `n_stream > 1` the view
spans the cache's stream dimension and the state arrives on the cache's **embd** axis (`[0,512]`), which
the handler rejected.  Qwen3.8 27B under the same `-npl 1,4` `-sm tensor` config is unaffected (its KV
granularity is a real head partition).  The code is pre-existing; 0008/0009 only made it *reachable*
(gemma4 + `-sm tensor` aborted earlier before them).

**Fix (0010).**  `handle_flash_attn_ext` also accepts a KV slice that is **empty-or-whole on every
device**, provided the query is empty-or-whole **on the same devices** (a device holding queries but no
K/V would compute rows nothing serves — rejected).  Nothing is reduced; the owning device computes the
whole op and the others have a zero-sized source (node disabled), the same situation the accepted axis-2
`[0,1]` case already is.  Patch size: ~40 lines, one handler.

**Gates.**

| gate | result |
|---|---|
| `llama-batched-bench -npp 512 -npl 1,4`, 12B f16, `-sm tensor` | runs: B=1 3067 t/s, **B=4 3525 t/s** (`-sm layer`: 2897 / 2896) |
| same with `-ctk/-ctv q8_0` (rotated) | runs: B=1 3015, **B=4 3377** |
| `llama-server -np 2`, two concurrent 400-token requests, 12B `-sm tensor` | **both HTTP 200**, 1955 / 1911 chars (previously the server aborted) |
| KLD, `-sm tensor` vs `-sm layer`, same batching, f16, n_seq 4 | 0.2048 / 87.1 % (the tensor artifact, section 1) |
| KLD, same, rotated q8_0, n_seq 4 | 0.2137 / 86.0 % |
| 27B, `-npl 1,4` `-sm tensor` q8_0 (control) | unchanged, B=4 2031 t/s |
| 35B-A3B 2 GPU tensor `-ncmoe 40` text (reference = 0001..0008 build) | **`0cf4a62aa63f`** both |

## 3. Config nits (decided: all three fixed)

1. **`-md <gemma4-assistant> --spec-type none` aborted** (`llama-graph.cpp:228 GGML_ASSERT(ubatch->embd_state)`)
   — `has_dft()` only means "`-md` was given", so the unused drafter still had its model loaded and its
   context created, and the drafter's warm-up decode needs the target's hidden state.  **Fixed (0011):**
   `uses_dft()`/`has_aux()` gate the draft-model load, the context creation (server +
   `common_speculative_init_result`), the MoE draft reserve and the preload; `common_init_from_params`
   prints one WRN.  Verified: `-md X --spec-type none` now runs as a plain baseline (exit 0, WRN at
   `-lv 2`), and gemma4 26B-A4B MTP is unaffected (**135.5 t/s**, `--fit false -c 16384`; the slab
   reserve still reports `draft CONTEXT 704 MiB (kv 0 + compute 512 + slack 192)` — the 0007 behaviour).
2. **The `--fit` probe's `E … requires ctx_other to be set`** is expected (the probe has no target
   context yet).  **Fixed (0012):** reported as a **W**; any other init failure stays an **E**.
   Verified at `-lv 2`: the E line is gone, fit still takes the derived 1129 MiB reserve.
3. **`(file*).moe-heatmap`** is a tracked artifact in block 09 that `test-llama-archs` rewrites — dropped
   at the promotion cut (maintainer decision, `HANDOVER-2026-10-21.md` §0.1), together with the fold.

## 4. The campaign gate set on the 12-patch tree (item 3 of the session)

All with `ROCBLAS_USE_HIPBLASLT=0`, one job at a time.  `ref` = the same tree with 0009+0010 reverted
(`972f3b757` + 0001..0008), i.e. a within-build A/B that isolates this session's split-state changes.

### 4.1 Non-regression cells (`../README.md` §2.4)

| cell | result |
|---|---|
| Qwen3.8 27B Q8_0, 1 GPU, q8_0 KV, `-n 128` | `da2e2d192e21` |
| Qwen3.8 27B Q8_0, 3 GPU `-sm tensor`, q8_0 KV, `-n 128` | **`da2e2d192e21`** (== 1 GPU) |
| Qwen3.5 4B Q8_0, 1 GPU / 3 GPU `-sm tensor`, q8_0 KV, `-n 128` | `9bf26f362e01` / **`9bf26f362e01`** |
| Gemma4 12B Q8_0, f16 and rotated q8_0, 1/2/3 GPU | `c740be0e131a` in every arm (section 1) |
| Qwen3.6 35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 40`, `-n 200` | **`0cf4a62aa63f`** == the 0001..0008 build |
| same, `llama-bench -p 512 -n 128 -r 3`, tg128 | **118.47 ± 1.62** vs ref **118.64 ± 2.01** (flat) |
| **35B-A3B cache canary** (`-v`) | **240 tables / 61440 slots / 18662.0 MiB**, 100.0 % residency — **identical** on ref and new |
| `test-llama-archs -s 42` | **644/644 passed** |

### 4.2 Prefill-logit KLD gates (issue #113)

| arm | base | result |
|---|---|---|
| Qwen3.8 27B Q8_0 (script, 1 GPU) | Oct 8 base | **0.000707 / 98.755 %** PASS (== the r100 record) |
| Gemma4 12B f16 KV | r99 base | **-0.000000 / 100.000 %** PASS |
| Gemma4 12B `-ctk/-ctv q8_0` (rotated) | r99 base | **-0.000000 / 99.990 %** PASS |

### 4.3 Four-axis adaptive-MTP gate, `-n 3000` (`mtp-adaptive-methodology.md` rules 0-5)

Config (both models): `-ngl 99 -sm tensor --fit false -c 16384 -n 3000 -t 8 --seed 42 --temp 0
--ignore-eos -f prompts/<axis>.txt --reasoning on|off --single-turn`, MTP = the GGUF's embedded head
(`--spec-type draft-mtp`, no `-md`); 27B = 2 GPU dense, 35B-A3B = 2 GPU `-ncmoe 40`.

| axis | 27B plain | 27B `draft-mtp` | 35B-A3B plain | 35B-A3B `draft-mtp` |
|---|---:|---:|---:|---:|
| R (reasoning on) | 34.5 | **72.0** | 114.3 | **240.1** |
| P (prose) | 34.1 | **87.3** | 115.3 | **247.7** |
| C (code) | 34.5 | **94.4** | 115.1 | **256.6** |
| K (recall) | 34.5 | **101.4** | 115.5 | **275.8** |

* **Rule 1 (acceptance, `-lv 4`)**: 27B **0.75273** (207/275), 35B-A3B **0.71579** (204/285) — healthy,
  no verify/logit divergence.
* **Rule 2 (MTP >= plain)**: +109…+194 % (dense), +110…+139 % (MoE) on all four axes.
* **Rule 4 (purity)**: `none` vs `draft-mtp` text is **byte-identical on all eight pairs** at `-n 3000`
  (hashes: 27B `0db96aec29a8` R / `f8e8ccc5d8cd` P / `f7256a7b1380` C / `7f6ac30740b3` K; 35B-A3B
  `22321d319221` / `0aa71be031b3` / `291de441b84e` / `ff9c59d04e11`).
* **Rule 3 (same-seed determinism vs the known-good build, dense)**: `build-r99` gives the **same eight
  hashes** and the same throughputs within 0.2 % (72.0/87.4/94.5/101.5 vs 72.0/87.3/94.4/101.4).
* **Rule 5 (stock-relative batched decode)**: r99 vs new, dense 27B UD-Q4_K_XL, 2 GPU `-sm tensor`,
  q8_0 KV, `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`: TG t/s B=1 **40.15 vs 39.95**, B=4
  **135.43 vs 135.56**, B=8 **215.44 vs 215.81** → +0.5 % / -0.1 % / -0.2 %: within noise at B=1, not
  worse at B=4/8.  (PP 237.6/666.6/911.5 vs 213.4/582.7/877.8.)

### 4.4 Hygiene

`scripts/check-gpu-faults.sh --since "4 hours ago"` over the journal and the representative logs (4-axis
27B R, 4-axis 35B K, the 35B canary, the gemma4 batched-bench, the n_seq-4 KLD): **clean, no GPU faults**.

## 5. What this session did NOT do

* The **folded** tree (`patches/` 09/10) has not been generated or gate-run yet — that is the promotion
  cut, which needs the maintainer's go-ahead (it touches `patches/` and `release.json`).
* The `(file*).moe-heatmap` deletion rides with that cut (nit 3).
* Block-09/10 docs (`patches/README.md`, `ENVIRONMENT.md`), the `WORKLOG.md` entry, the `TODO.md` updates
  and the campaign archive all belong to the cut.
