# upstream: let RDNA4 use the WMMA flash-attention kernel for head 512 (Gemma 4 global layers)

**Candidate, not filed.** Prepared 2026-10-19 from the r92 campaign for
[`stew675/llama-cpp-rdna-boosts` issue #188](https://github.com/stew675/llama-cpp-rdna-boosts/issues/188)
(reported by `@DanoPTT`, upstream
[ggml-org/llama.cpp#30838](https://github.com/ggml-org/llama.cpp/issues/30838)).

Patch: `UPSTREAM-PR-fa-512-rdna-head-cap.patch` (2 files, +14/-2), cut against master **`097cf33d9`**.
The patch applies clean to that base and **builds and runs** there (gfx1201, ROCm 10.1.0).

## What

Two small edits, no new configuration rows:

1. `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the `AMD_WMMA_AVAILABLE` guard: `DKQ > 256` -> `DKQ > 576`.
   Upstream already has a `(512, 512, 32)` RDNA config row and an instantiated
   `flash_attn_ext_f16<512, 512, 4, 8, ...>` kernel for it, but this guard compiles the body out, so
   the head-512 WMMA path is dead code today (the first attempt to reach it fails at runtime with
   "no device code compatible with HIP arch").
2. `ggml/src/ggml-cuda/fattn.cu`, the WMMA chooser: allow `Q->ne[0] == 512` on RDNA4, **prefill only**
   (`Q->ne[1] > 8`). Prefill-only because at decode/verify widths a 512-wide head is memory bound, and
   because letting `n_q` pick a different kernel family makes decode and verify logits disagree.
   RDNA3 keeps the 256 cap (there the same head measured 3.5-10 % slower than the tile kernel at deep
   prefill, see the note already in that file for issue #30-style tuning).

`GGML_CUDA_FA_WMMA_MAX_HEAD` does not exist upstream, so the escape hatch is `-DGGML_HIP_ROCWMMA_FATTN`
off / reverting the patch.

## Measured (gfx1201, 1x R9700, ROCm 10.1.0, base `097cf33d9`)

Gemma-4-31B QAT Q4_K_XL, `llama-bench -p 512 -n 0 -ub 512 -b 1024 -fa on -ctk q8_0 -ctv q8_0`,
`-t 8`, `-r 2` (d0 `-r 3` for the patched arm):

| arm | pp512 d0 | pp512 d32768 | pp512 d98304 |
|---|---|---|---|
| upstream (cap 256, tile kernel) | 1131.4 | 378.8 | 167.6 |
| upstream + this patch | 1138.4 | **406.1 (+7 %)** | **170.5 (+1.7 %)** |
| fork r92, for reference (not this patch) | 1179.4 | 588.7 | 299.2 |

`test-backend-ops -o FLASH_ATTN_EXT` on the patched build: **3993/3993 OK, 0 FAIL**.

## Caveats (read before filing)

* **The win is small on clean upstream.** Most of the fork's much larger head-512 gain comes from work
  that is *not* in this patch (the fork's band/launch machinery, its native quantized and bf16 K/V
  reads, and the head-512 row it ships). This patch only makes an existing upstream row reachable.
  Validate end-to-end on the target box before proposing it; the reporter offered A/B builds.
* **Do not judge it with the isolated `--test-file` case from #30838 alone.** That case puts the query
  block at the *start* of the KV slot, so the kernel loads and discards the whole KV cache: upstream
  tile measures 84/252/249 ms there and the patched build 86/289/293 ms, i.e. the patch looks 15-18 %
  *slower*, while a real deep prefill (the table above) is faster. The end-to-end table is the
  representative measurement.
* Only q8_0 K/V and f16/bf16 K/V were exercised, single GPU. No NVIDIA hardware was available; the
  change is `GGML_CUDA_CC_IS_RDNA4`-gated, so other backends keep their current behaviour. Head 320
  (Mistral 4 MLA) and 576 (DeepSeek/GLM MLA) are deliberately *not* included: the cap is an equality
  check on 512, and their WMMA rows were not validated here on upstream's code.
* If upstream takes this, drop the corresponding hunks from the delivery (block 05 keeps the row
  tuning and the fork-only machinery; the chooser and guard hunks would then be upstream-owned).

## Reproduce

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9 && git apply .../UPSTREAM-PR-fa-512-rdna-head-cap.patch
~/bin/build-llama-rocm            # or the equivalent -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 build
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench \
  -m gemma-4-31B-it-qat-Q4_K_XL.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -ub 512 -b 1024 -p 512 -n 0 -d 0,32768,98304 -t 8 -r 2
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT
```

A/B the other way by building the same tree without the patch (head 512 then falls back to the tile
kernel, which is what upstream does today).
