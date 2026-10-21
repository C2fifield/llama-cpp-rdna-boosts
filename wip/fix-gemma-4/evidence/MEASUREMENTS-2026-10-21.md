# Measurements — 2026-10-21 session (gfx1201, 3x R9700, ROCm 10.1.0, 16-core)

Tree: `~/llama.cpp` @ `rdna-boosts` `972f3b757` (`v12-097cf33d9-r100`) + the patch set in
`../patch-state/`.  Reference binary for every "prefix" column: `build-r99`
(tree `6adf0b802`, `v12-097cf33d9-r99`), which is behaviourally identical to r100 for these paths.
Single job at a time; `-t 8` (GPU IRQ cores 13-15 left free); no parallel benches.

## 1. #76 — iSWA KV-write fusion (already complete; recorded here for one place)

`rocprofv3 --kernel-trace`, Gemma4 12B Q8_0, `-ctk/-ctv q8_0`, `-n 8`, 1 GPU:

| kernel | prefix | fix | fix + `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` |
|---|---:|---:|---:|
| `rms_norm_mul_rope_f32<256,false,float,256,true>` (K+V write fold) | 0 | **480** | 0 |
| `rms_norm_mul_rope_f32<256,false,float,256,false>` (rotation only) | 960 | **480** | 960 |
| `k_set_rows_q8_0_group` | 1152 | **192** | 1152 |
| total kernel launches | 13500 | **12540** | 13500 |

2-GPU `-sm layer`: identical counts (480 / 480 / 192 / 192).
`llama-bench -p 512 -n 128 -r 3`, 3 interleaved rounds:

| config | prefix tg128 | fix tg128 | delta |
|---|---:|---:|---:|
| 1 GPU, q8_0 KV | 40.50 | **40.76** | +0.6 % |
| 2 GPU `-sm layer`, q8_0 KV | 37.29 | **37.52** | +0.6 % |

`llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (total t/s): B=1 40.14 -> 40.19, B=4 148.37 ->
148.70, B=8 251.99 -> 253.11.

## 2. #77 — Gemma4 26B-A4B Q8_0 expert cache

### 2.1 The defect, measured with temporary counters (removed before the patch set)

`-ncmoe 40 -sm tensor`, `-n 8`, before the fix:

```
scheduler: prune_branch=716  copy_experts_calls=8174  copy_experts_bytes=28.5 GiB (≈2.5 GB/token)
meta:      segmented SYNC set 4087 calls, 19.0 GiB, 3.80 s of host-blocking copy in one 8-token run
cache:     enabled=1 sized=0 n_tables=60 registered=7713.8 MiB expected=23141.3 MiB complete=0
```

`-sm layer`, same model: `registered=24265359360` vs `expected=24265374720` — a **15360 B** shortfall
(30 layers x 512 B = gemma4's per-layer `ffn_down_exps.scale`), so `moe_cache_tables_complete_locked()`
could never be satisfied, `alloc_all_locked()` deferred forever and **no arena was ever created**.

### 2.2 Effect of a sized arena (proof the sizing was the whole story)

| gemma4 26B-A4B Q8_0 `-ncmoe 40`, `-n 32`, 2 GPU | gen t/s |
|---|---:|
| `-sm layer`, cache auto, before (inert) | 7.9 |
| `-sm layer`, `MOE_EXPERT_CACHE_SLOTS=32` (forced arena) | 24.3 |
| `-sm layer`, `MOE_EXPERT_CACHE_SLOTS=64` | 46.9 |
| `-sm tensor`, cache auto, before (inert) | 5.5 |
| `-sm tensor`, `MOE_EXPERT_CACHE_SLOTS=64` (`down` only cacheable) | 6.5 |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=0` (CPU MoE) | 22.9 |

### 2.3 After the patch set (A + B + C)

`-n 120 --ignore-eos`, prose prompt, 2 GPU, `Generation t/s`:

| config | prefix (r99) | patched | text |
|---|---:|---:|---|
| `-sm layer -ncmoe 40` | 7.9 | **79.0** | `34475f215443` — **identical to prefix** |
| `-sm tensor -ncmoe 40` | 5.0 | **74.7** | `fd8f9380246c` — **identical to prefix** |
| `-sm tensor -ncmoe 40`, `GGML_META_SEG_ASYNC=0` | — | 74.5 | `fd8f9380246c` (B is neutral here: no misses) |
| `-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=0` (CPU MoE) | — | 23.7 | `1e00af4c4c17` (different kernel: CPU MoE) |
| `-sm tensor`, `-n 512` long-context run | — | **71.9** (pp 172.3) | cache `h=1.0000`, identity path, arena 22.9 GiB, 0 faults |
| wikitext PPL, 20 chunks, `-sm tensor -ncmoe 40` | 22602.1245 (175 s) | 22602.1245 (107 s) | **identical PPL** |

`-sm tensor` cache report on the long run: `tables=119/120 registered=23141.2 MiB expected=23141.3 MiB`,
128 slots/table, `h=1.0000 (261016/261016)` — the whole expert set resident, no cold reads.

### 2.4 Non-regression (the gate contract in `../README.md` §2.4)

| model / config | prefix | patched | verdict |
|---|---|---|---|
| **Qwen3.8 27B Q8_0**, 1 GPU, q8_0 KV, `-n 128` | `28d76f560425` | `28d76f560425` | text identical |
| **Qwen3.8 27B Q8_0**, 3 GPU `-sm tensor`, q8_0 KV, `-n 128` | `28d76f560425` | `28d76f560425` | text identical |
| same, `llama-bench -p 512 -n 128 -r 3`, 3 rounds | tg 40.71 | tg 40.74 | flat (+0.07 %) |
| **Qwen3.5 4B Q8_0**, 3 GPU `-sm tensor`, q8_0 KV | `b0b32d140499` | `b0b32d140499` | text identical |
| **Gemma4 12B Q8_0** (dense, cache inert), 1 GPU, q8_0 KV | `9f1097128c92` | `9f1097128c92` | text identical |
| **Qwen3.6 35B-A3B UD-Q4_K_M**, 2 GPU tensor `-ncmoe 40`, `-n 200` | `6802a3091045` | `6802a3091045` | text identical |
| same, bench 3 rounds | tg 119.31 / 119.13 / 119.20 | tg 119.42 / 119.18 / 119.24 | flat |
| same, cache tables/arena | 240 tables, 61440 slots, 18662.0 MiB | **same** | unchanged |
| `test-llama-archs -s 42` | 375/375 | **375/375** | unchanged |
| `test-backend-ops` MUL_MAT_ID / SET_ROWS / UPSET_ROWS / RMS_NORM / CPY | — | all 2/2 backends | OK |
| prefill-logit KLD, Gemma4 12B f16 | — | **-0.000000 / 100.000 %** | PASS |
| prefill-logit KLD, Qwen3.8 27B standard gate | — | **0.000000 / 100.000 %** | PASS |
| GPU faults (journal + all run logs) | — | none | clean |

**One intermediate regression, found and fixed within the session** — recorded because it is the trap this
change invites: an earlier draft of the completeness fix used a `max(1 MiB, expected/128)` slack, which on
35B-A3B (one table = 36.4 MiB, 18.6 GiB total) silently accepted a sizing pass with **one table still
unregistered** (240 -> 239 tables) and cost **-3.6 % tg128** (119.3 -> 115.0).  The shipped form filters
non-bank tensors out of the loader's account (`ne[2] < 2`, i.e. `ffn_down_exps.scale`) and keeps the slack
at `min(max(1 MiB, expected/4096), 16 MiB)`.

## 3. MTP with the new Gemma4 drafter (`mtp-gemma-4-26B-A4B-it.gguf`)

`/llm/models/Gemma4/26B-A4B/Q8_0/mtp-gemma-4-26B-A4B-it.gguf` (461 MB, arch `gemma4-assistant`),
`-ncmoe 40 -sm layer --fit false -c 16384`, prose prompt, seed 42, temp 0.

| arm | `-n 300` gen t/s | text |
|---|---:|---|
| plain (`--spec-type none`) | 75.1 | `5e1cbda9f5e1` |
| `draft-mtp` n_max **2** | **86.2 / 86.3** (2/2 identical) | `8241a94d2f86` (2/2) |
| `draft-mtp` n_max 3 | 85.3 | `afa48a6ea28a` (but see §4) |
| `draft-mtp-adaptive` n_max 7, `-n 3000` | 81.8 | — |
| plain, `-n 3000` | 60.0 (2/2 identical) | `21cfadff21d7` |
| `draft-mtp` n3, `-n 3000` | 82.8 (**+38 %**) | — |

At `-n 200`, `--ignore-eos`, a short prompt: plain == `draft-mtp` n3 == `draft-mtp-adaptive` n7 =
`b47fca2a451e` (purity holds).  With `-ncmoe 0` (no host experts, no cache): plain == MTP n3 =
`5e1cbda9f5e1`, deterministic 2/2, MTP +9 % (87.5 vs 79.8).

**The drafter is rejected for `-sm tensor`** (`gemma4-assistant` is not in `llm_arch_supports_sm_tensor`) —
intended, and the reason the MTP numbers above are `-sm layer`.

**`--fit` starves the arena under MTP.**  At the default (max) context the draft + target contexts make
`--fit` reserve 19.16 GiB/device and the slab only gets 9.31 / 7.50 GiB, so the arena is 13.8 GiB instead of
23.1: `h=0.907`, 8030 cold reaches and MTP falls to **50.2 t/s, below plain (79.3)**.  Pinning the context
(`-c 16384`) restores full residency and MTP to +15 % (n2) / +38 % (-n 3000, n3).  Documented, not a code
defect.

## 4. MTP determinism (a separate, pre-existing class — handover item)

With host-resident experts (`-ncmoe 40`) **and** an active cache **and** MTP **and** captured HIP graphs,
the greedy text is **run-to-run nondeterministic**:

| arm (MTP n3, cache on, default graphs) | run 1 | run 2 |
|---|---|---|
| as shipped today | `8241a94d2f86` | `afa48a6ea28a` |

Deterministic with any one of:

| control | result |
|---|---|
| `GGML_CUDA_DISABLE_GRAPHS=1` | `5e1cbda9f5e1` twice — and **== plain** (purity restored) |
| `GGML_CUDA_ASYNC_INPUT=0` | `8241a94d2f86` twice |
| `-ncmoe 0` (no host experts, cache inert) | `5e1cbda9f5e1` twice, MTP == plain |
| `MOE_EXPERT_CACHE_MIB=0` (CPU MoE) | `d0892c28b07f` twice |
| `--spec-draft-n-max 2` instead of 3 | `8241a94d2f86` twice |
| r99 build (cache inert, GPU per-op upload) | `5e1cbda9f5e1` twice |

So it needs the host-expert upload path *and* a captured graph to co-occur; it is the r69/`TODO #75`
family ("a run is not start-to-start deterministic when the consumer is a captured HIP graph"), not
introduced by the cache — the cache is what finally makes the path fast enough to be used.  `-lv 4`
changes the text too (a timing-sensitive race).  Not fixed here; see `../HANDOVER-2026-10-21.md`.
