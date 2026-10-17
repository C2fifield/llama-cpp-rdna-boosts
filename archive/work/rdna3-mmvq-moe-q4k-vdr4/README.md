# rdna3-mmvq-moe-q4k-vdr4: the 32-element Q4_K dot product for MoE experts on gfx1100 (RDNA4 opt-in)

Thank you for folding #162 into r69.  These two patches extend its `vec_dot_q4_K_q8_1_vdr4_dense` to your MoE
expert kernel, whose `vec_dot_q4_K_q8_1_vdr4` layout it already shares.

Two format-patches on top of `v12-097cf33d9-r71` (tree `dc2cc150`, applied tree `56bd086b`), for `git am`.  The
measurements below were taken on r70 (tree `592a097c`, applied tree `e2aa5c97`); r71 does not touch the patched files
or the MoE kernel, and on r71 the greedy hashes below are unchanged (plain and patched, MoE and dense) and
`test-backend-ops -o MUL_MAT_ID` q4_K passes.  2 files (`mmvq-vecdot.cuh`, `vecdotq.cuh`), +19 / -3.  0001 is RDNA3_0 only; 0002 is an RDNA4 build-time
opt-in, off by default.  Dense models are unaffected (no `MUL_MAT_ID`); RDNA4 without the flag, RDNA3_5, CUDA and the
other targets are untouched by construction.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (a separate machine, labelled there).

## 0001: MoE experts take the 32-element Q4_K dot product on gfx1100

On RDNA3_0 the MoE expert kernel (`mul_mat_vec_q_moe`) now calls `vec_dot_q4_K_q8_1_vdr4_dense` instead of
`vec_dot_q4_K_q8_1_vdr4`.  Same 32-element layout and VDR (4), so only the summation changes: each sub-block's 16
products go through one integer `dp4a` chain before a single scale / convert / fma, instead of a float step per
8-element chunk.  One selector line in `get_vec_dot_q_cuda()`.

## 0002: RDNA4 build-time opt-in (default off)

`-DGGML_HIP_RDNA4_Q4K_MOE_VDR4` (e.g. `-DCMAKE_HIP_FLAGS=-DGGML_HIP_RDNA4_Q4K_MOE_VDR4`) does the same for RDNA4's
MoE experts.  It is separate from #162's `GGML_HIP_RDNA4_Q4K_DENSE_VDR4`, so dense and MoE can be A/B'd independently.
The MoE VDR is 4 either way, so only the device-side choice changes.

In case it's useful: ccache keys HIP objects on the host-side preprocessing, where `RDNA4` is never defined, so a
flag that only changes device code would come back from the cache unchanged.  We hit exactly that (the gfx1201
`mmvq-moe.cu.o` was byte-identical with and without the flag), so 0002 also defines a host-visible
`[[maybe_unused]]` constant under the flag.  #162's dense flag isn't affected, since it also changes host code.

gfx1201 compile check (`mmvq-moe.cu`, your README flags with `-DGPU_TARGETS=gfx1201`): with the flag, the Q4_K MoE
kernels go from 70 to 67 VGPRs (8 rows, fused) and 57 to 54 (2 rows), the same pattern as gfx1100; 0 scratch with no
flag, the MoE flag, or both flags.  Timings on an R9700 are in the RDNA4 section below.

## Tests

Qwen3-30B-A3B Q4_K_M (`Qwen/Qwen3-30B-A3B-GGUF`, 48 layers; Q4_K and Q6_K experts), your README build
(`-DGPU_TARGETS=gfx1100`), no tuning env, `-ngl 99 -ctk q8_0 -ctv q8_0`, plain decode (this model has no MTP head).
Plain r70 against r70 + 0001..0002.

**Kernel time** (rocprofv3, one 400-token greedy request per build), per call:

| | r70 | + 0001..0002 | change |
|---|---|---|---|
| fused gate+up+GLU, Q4_K experts | 24.8 us | 22.3 us | -10 % |
| down projection, Q4_K experts | 19.4 us | 16.0 us | -18 % |
| down projection, Q6_K experts (control, untouched) | 18.0 us | 18.1 us | flat |
| total GPU time per request | 4167 ms | 4087 ms | -1.9 % |

**Decode** (`llama-bench -p 0 -n 256 -r 10`, two alternating rounds): r70 131.86 / 131.51 t/s, + 0001..0002
133.13 / 134.03 t/s, +1.4 %.  Both "on" rounds above both "off" rounds, but MoE decode at ~130 t/s is noisy here
(about ±2 % within a round), so the kernel times above are the firmer evidence.  These runs used the revision just
before 0002's ccache line, which only adds a constant under the RDNA4 flag.

**Correctness.**
- Greedy hashes, plain decode, 400 tokens: r70 prose `0116af00d69e` / code `8fd18d2f0b83`; with 0001
  `bf70e0f571ea` / `6b19db0bf7d8` (decoded: a hash-table explanation and a thread-safe Python LRU cache in both
  builds).
- Dense control, Qwen3.8-27B Q4_K_M: `8345c992d1fa` in both builds (no `MUL_MAT_ID`, so unchanged).
- Your `test-logits-width-probe` on the MoE model: PASS at prefix 256 and 3584.
- `test-backend-ops -o MUL_MAT_ID`: q4_K 84/84 OK (the remaining q4_K cases are "not supported" on this backend).
- How much the output moves (`llama-perplexity -ub 4`, so the MoE path goes through mmvq, 24 x 512 tokens of the
  llama.cpp README / docs `*.md`), measured with this same change applied to r67 + #162 and not re-run on r70 (the
  MoE kernel and dot-product files are the same there, apart from 0002's opt-in lines):

  | vs the parent at `-ub 4` | parent + 0001, `-ub 4` | parent, `-ub 512` (MMQ), for scale |
  |---|---|---|
  | mean KLD | 0.0044 ± 0.0002 | 0.0065 ± 0.0005 |
  | same top token | 97.04 ± 0.22 % | 95.88 ± 0.25 % |
  | PPL ratio | 1.0013 ± 0.0021 | 1.0024 ± 0.0024 |

  Below the existing prefill-vs-decode difference, with perplexity unchanged within its error.  The headroom is
  smaller than in #162's dense case (there 0.0006 against 0.0035); expert routing may amplify rounding differences.

## RDNA4 (0002, opt-in)

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201), native Linux, ROCm 10.0**; same model, r70 + 0001..0002 (applied tree
`e2aa5c97`), built with and without `-DGGML_HIP_RDNA4_Q4K_MOE_VDR4`; llama-server, `-ngl 99 -ctk q8_0 -ctv q8_0`.

- Without the flag the build matches plain r70 (greedy hashes identical), as expected.
- With it: decode +0.7 % (132.5-132.8 -> 133.4-133.7 t/s over two alternated pairs of servers; the ranges do not
  overlap, but the gain is under 1 %), 4 concurrent +1.4 %, 31.5k prefill unchanged (MMQ).  Kernel time per call: fused 2-row Q4_K experts 26.09 -> 25.16 us (-3.6 %);
  the 8-row Q4_K kernel is unchanged within drift (+1.0 %, against +0.7 % for the untouched Q6_K control).
- Greedy text changes (prose `744db56ea1c8` -> `21e80a3158e7`, code `6bb5966d061f` -> `183ec9844654`, each stable
  across 3 servers); width probe PASS at 256 and 3584; `test-backend-ops` MUL_MAT_ID q4_K 84/84; dense 27B unchanged.
- `mmvq-moe.cu.o` differs with and without the flag, so the host-visible constant does reach the ccache key.

A gain of under 1 % on one box and one model, with changed greedy text, which is why it stays a build-time opt-in.
KLD was not measured on gfx1201, so how far the output moves there is not quantified.

## Limitations

- One MoE model; Q4_K experts only (the Q5_K MoE path has the same pattern and is left alone).
- The gfx1100 numbers are under WSL2; #162's gain was smaller on your native box, and this one may be too.
- RDNA3_5 and CUDA not built; untouched by construction (everything is under RDNA3_0 or the RDNA4 flag).

No urgency on our side.  Happy to re-run anything on your config.
