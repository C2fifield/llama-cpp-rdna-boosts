# qwen4exp (Qwen3.8-Flash-Next) multi-GPU MTP 4-axis gate — r97

Dated record, 2026-10-20.  This is the qwen4exp / `-sm tensor` cell of the adaptive-MTP gate
([`mtp-adaptive-methodology.md`](mtp-adaptive-methodology.md)) re-run to validate **r97's
split-mode-dependent `DSV4_HC_POST` default** (`patches/README.md` block 10,
`WORKLOG.md` 261020.008): under a Meta (tensor-split) device the deepseek4/glm5/qwen4exp HC-combine arm
is the fused op instead of the primitive `hc_combine_norm` chain, which is numerically wrong across a
tensor split.

## Provenance

| | |
|---|---|
| build | r97, `version: 0.6.0-dev (build 11583, commit b37a3b5ad)` — canonical tip `b37a3b5ad`, tree `f81213c4dcb057636ef6afdfef7e4c4907a4d83f` |
| runtime | ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), 3x Radeon AI PRO R9700 (gfx1201), 16-core 9950X3D |
| target | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-0000{1..9}-of-00009.gguf` (96 GiB total) |
| drafter | `-md /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` — **this quant's MTP head is a sidecar**, not embedded (`--spec-type draft-mtp` without `-md` fails with "model doesn't contain MTP layers") |
| prompts | `prompts/reasoning.txt` (`242f5e6f…`), `prompts/prose-rdna-boosts.txt` (`fabdec65…`), `prompts/code-python.txt` (`53da7f23…`), `prompts/recall.txt` (`4e11ce3c…`) |
| axes | R = reasoning (`--reasoning on`), P = prose, C = code, K = recall (`--reasoning off`) |

Common flags (all runs):

```sh
OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 HIP_VISIBLE_DEVICES=0,1,2 \
<llama-cli> -m <IQ4_NL 1/9> [-md <mtp shared Q8_0>] \
  -ngl 99 -sm tensor -ncmoe 48 -c 32768 -b 4096 -ub 4096 \
  --load-mode none --lazy-mode off -fa auto -ctk f16 -ctv f16 -t 8 \
  --seed 42 --temp 0 --single-turn --no-display-prompt -n 3000 \
  -f <axis prompt> --reasoning <on|off> --spec-type <none|draft-mtp --spec-draft-n-max 3> -lv 4
```

**`--load-mode none --lazy-mode off` is mandatory for qwen4exp.**  Its 26.8 GiB
`per_layer_token_embd` is mmap-lazy by default; with the default load mode and `-ub 2048` the same
prose run read 105.5 t/s and the code run 22.6 t/s (wild spread, PLE re-reads).  With the PLE resident
and `-ub 4096` the same cells read 124.6 and 139.4 t/s.  `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` is
set for the documented host-mapped-input MTP confound (methodology, 2026-09-24).

## Results — 4 axes, `-n 3000`, r97 default arm (the fused op under the Meta split)

| axis | plain t/s | `draft-mtp n3` t/s | MTP / plain | draft acceptance | acc per pos |
|---|---:|---:|---:|---:|---|
| P prose | 60.6 | **124.6** | 2.06x | 0.832 | 0.913 / 0.834 / 0.751 |
| C code | 62.6 | **139.4** | 2.23x | 0.932 | 0.956 / 0.925 / 0.868 |
| K recall | 57.9 | **120.0** | 2.07x | 0.992 | — |
| R reasoning | 62.1 | **108.3** | 1.74x | 0.610 | — |

Gate rules: acceptance >= ~0.45 on prose (0.832 ✓), MTP not slower than plain on any axis ✓ (1.74-2.23x
faster), `-n 3000` and per-axis reasoning ✓.

## Purity — `none` vs `draft-mtp n3` vs `n7`, `-n 64`, hashed with `scripts/extract-generated.py`

| axis | plain | n3 | n7 |
|---|---|---|---|
| P | `471d102e7b7d` | `471d102e7b7d` | `471d102e7b7d` |
| C | `a4f29c5ebd50` | `a4f29c5ebd50` | `a4f29c5ebd50` |
| K | `ef6e78e4a495` | `ef6e78e4a495` | `ef6e78e4a495` |
| R | `febe0a3c7911` | `febe0a3c7911` | `febe0a3c7911` |

Byte-identical across the whole purity range on all four axes.  (qwen4exp is MoE, where the methodology
relaxes byte-identity to "sane output"; it holds anyway.)

## A/B — the arm r97 replaced (`LLAMA_FUSED_DSV4_HC_POST=0`, the primitive chain)

| axis | arm | plain t/s | n3 t/s | acceptance |
|---|---|---:|---:|---:|
| P | chain (pre-r97 under `-sm tensor`) | 60.7 | 124.1 | 0.826 |
| P | r97 default (op) | 60.6 | 124.6 | 0.832 |
| C | chain | 62.4 | 138.4 | 0.917 |
| C | r97 default (op) | 62.6 | 139.4 | 0.932 |

Throughput is the same within noise and acceptance is marginally higher with the op — so the change is
free.  The chain is **not** numerically equivalent: on the code axis it emits different greedy text from
the first generated line (it writes
`"""Production-quality Python module implementing various data structures and algorithms."""`
where the op arm writes
`"""Production-quality implementations of common data structures and algorithms."""`, 850 vs 795 chars),
while both arms are internally pure (plain == n3 == n7).  The prose axis happens to agree
(`471d102e7b7d` in both arms).  This is the real-model counterpart of the synthetic
`test-llama-archs -a deepseek4` finding (NMSE 1.74e-01 through the chain, 6.5e-06 through the op).

## Notes / side findings

* The plain arm must **not** pass `-md`: `-md <mtp gguf> --spec-type none` aborts with
  `GGML_ASSERT(!suffix_fallback.empty())` at `src/llama-model.cpp:520`.  **Pre-existing** (reproduced on
  the r96 build), not introduced by r97; the assert is in the meta split-state callback's tensor-suffix
  lookup, so it fires when a draft model is loaded but no drafting is requested.  A warning + ignore
  would be friendlier than an abort.
* The expert arena reports 512/512 fully resident slots/table on this config
  (`alloc_all_locked: 144 tables / 42.2 MiB per expert, arena 64800 MiB`), i.e. this cell runs the
  "identity fast path" on the routed decode band.  The recorded numbers are therefore arena-resident
  decode, not streaming.
