# q8-decode-small-kernels: q8_0 set_rows as one 32-lane group per block, rms_norm_q8_1 from registers

Thank you for the fused `rms_norm_q8_1` kernel and its quantize-cache handoff; 0002 is a small follow-up to it.
Two bit-identical patches for two short decode kernels on a quantized-KV decode.

Two format-patches on top of `v12-097cf33d9-r71` (tree `dc2cc150`, applied tree `f05e294f`), for `git am`.
2 files (`set-rows.cu`, `norm.cu`), +94 / -25.  Not arch-specific: both kernels are shared by every target, and the
output should be unchanged elsewhere too (same arithmetic, same index math; checked on gfx1100 and gfx1201).

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (labelled there).

## 0001: set_rows q8_0, one 32-lane group per block

With a q8_0 KV cache, every layer's K and V write is a `SET_ROWS` of one short row per token.  The q8_0 path ran one
thread per block, looping over 32 values with 32 byte stores, so that loop was the latency of the whole launch.  The
new `k_set_rows_q8_0_group` gives each block a 32-lane group: an xor-shuffle `fmaxf` for amax (seeded with 0 like
`quantize_f32_q8_0_block`), then the same `d`, `id = d ? 1/d : 0` and `roundf(v*id)`, and coalesced stores.  The index
math is the same as `k_set_rows_quant`'s.  Other quant types keep the existing kernel.

## 0002: rms_norm_q8_1 quantizes from registers

Each thread quantizes exactly the columns it normalizes, so the dst write loop folds into the quantize loop: the
scaled value is quantized from a register, and the `__syncthreads()` and the dst reload go away.  The `has_add`
early-out (quantize cache already filled) becomes a uniform `continue`.  Same expressions; bit-identical.

## Tests

Qwen3-30B-A3B Q4_K_M (MoE) and Qwen3.8-27B Q4_K_M (dense), your README build (`-DGPU_TARGETS=gfx1100`), no tuning
env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.  Plain r71 against r71 + 0001..0002.

**Kernel time** (rocprofv3, one 400-token greedy MoE request per build), per call:

| | r71 | + 0001..0002 |
|---|---|---|
| set_rows q8_0 (K and V, 44.5k calls) | 5.1 us | 3.7 us |
| rms_norm_q8_1 (1024 threads, 44.8k calls) | 6.1 us | 5.8 us |
| total GPU time per request | 4111 ms | 4048 ms (-1.5 %) |

**Decode** (`llama-bench -p 0 -n 256 -r 10`, two alternating rounds): MoE r71 131.38 / 129.44 t/s, + 0001..0002
131.81 / 133.44 t/s (+1.7 %, about the size of the noise on this box; the per-call kernel times, from one trace per build, point the
same way).
Dense 38.39 -> 38.64 t/s (+0.7 %, one round, within noise).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r71: MoE plain decode prose `0116af00d69e` / code `8fd18d2f0b83`; dense with
  MTP (draft-mtp, n-max 3; this exercises the `has_add` path) prose `8345c992d1fa` / code `47114faa2042`.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops -o SET_ROWS` and `-o RMS_NORM`: pass.

## RDNA4

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201)**, native Linux, ROCm 10.0; Qwen3-30B-A3B Q4_K_M and Qwen3.8-27B Q4_K_M,
llama-server `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`, plain r71 against r71 + 0001..0002.

- Greedy hashes identical to plain r71 on both models (MoE prose `744db56ea1c8` / code `6bb5966d061f`, dense prose
  `d12cdeede326` / code `e8a93c816237`); width probe PASS (maxdiff 0) at 256 and 3584; `test-backend-ops` SET_ROWS
  159/159, RMS_NORM 53/53.
- Kernel time per call: set_rows q8_0 1.51 -> 1.33 us, rms_norm_q8_1 2.72 -> 2.47 us.
- MoE decode (two alternated rounds): 133.1 / 133.0 -> 133.9 / 133.9 t/s (+0.6 %); dense 31.2 t/s both; 31.5k prefill
  unchanged.

## Limitations

- Measured on gfx1100 (WSL2) and gfx1201 (native Linux).  Both kernels are generic, so other targets get the same
  code; we did not run them (on wave64 targets the shuffles stay within 32-lane segments, as in `quantize_q8_1`).
- set_rows: only the q8_0 destination takes the new kernel.

No urgency on our side.  Happy to re-run anything on your config.
