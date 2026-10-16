# FINDINGS (2026-10-16) -- r69 revalidation of the cell-offset campaign, and the issue #151 "graphs-on" flip

**Date:** 2026-10-16 (revalidation of a campaign parked 2026-10-11; the trigger is the issue #151
follow-up that seeded TODO #67).  **WIP, NOT for delivery.**
**Build:** the r69 tree (`d210c73b7`; gfx1201, ROCm 7.14.1, one R9700 unless noted).  Nothing in this
file is a delivery change.

> **Report to the maintainer before any `patches/`/`release.json`/docs-header change.  This is WIP.**

---

## 0. Why this re-run exists

The campaign was parked on the finding "*`-np > 1 --kv-unified` is not reproducible; `--no-kv-unified`
is fully green*" (issue #134 / TODO #52 part C).  On the issue #151 follow-up the reporter
(`@DanoPTT`, R9700 / Windows / ROCm 10) then reported that with **HIP graphs on** the greedy output
also differs **start to start** with `--no-kv-unified`, on the **first request of a fresh server**.
That would be a different root from the cell-offset effect, so it was seeded as TODO #67.

Before acting on #67 the claim has to be verified.  Two questions:

1. Does the campaign's unified-KV effect still reproduce on the current (r69) tree?  (Revalidate.)
2. Does a **decode-side** start-to-start nondeterminism exist here with graphs on, `--no-kv-unified`?
   (The reporter's claim; `llama-perplexity` cannot see this because it only runs the forward axis.)

## 1. The campaign effect still reproduces on r69 (`msq_plain.sh`)

GSQ Flash-Next IQ3_XXS, `-sm tensor -ncmoe 48`, 2 GPU, `-ub 6144`, `-c 204800`, `-np 2`,
`--spec-type none`, graphs default (on).  Request A is the 33.7k-token `mixed30k-v2.txt` (256 tokens
generated), B is the 5k `msqB.txt` (96 tokens).  A is run solo, then the A+B pair is run concurrently
three times.

| arm | A_solo | A_conc1 | A_conc2 | A_conc3 | verdict |
|---|---|---|---|---|---|
| `--kv-unified` | `8630d276db97` | `0d408c30b5ba` | `8c3d3fe03c60` | `1a566838a4f1` | **red** (each repeat different) |
| `--no-kv-unified`, start 1 | `90156eb6d2e4` | `5c051c43b20f` | `5c051c43b20f` | `5c051c43b20f` | concurrent repeats **green**; solo != concurrent |
| `--no-kv-unified`, start 2 | `90156eb6d2e4` | `5c051c43b20f` | `5c051c43b20f` | `5c051c43b20f` | identical to start 1 |

B (`msqB.txt`, 96 tokens): unified `cfd188d8cc01` solo vs `a918d22a0b74` / `469d57f12969` /
`6f19424e35f8` concurrent; non-unified `468f27a8ae44` solo vs `65b504db9255` concurrent (identical
across both starts).

**So:**
- The **cell-offset effect is alive** on r69: under `--kv-unified` every concurrent repeat produces a
  different greedy text (the documented quantum-64/whole-cell-set dependence).
- Under `--no-kv-unified` the **concurrent repeats are reproducible** (`5c051c43b20f` twice, across two
  fresh server starts), so the non-unified path has no cell-offset nondeterminism.  **But "fully green"
  was too strong**: solo (`90156eb6d2e4`) differs from concurrent (`5c051c43b20f`) deterministically, at
  token 59.  That is the separate cross-sequence **batching** effect (A alone vs A+B batched into one
  ubatch changes the batch shape), not the KV cell set.  The campaign README's "solo == conc1..3"
  wording is corrected to "concurrent repeats are reproducible; solo vs concurrent differ".

## 2. No graphs-on start-to-start decode nondeterminism here

`test-logits-width-probe` (prefill P tokens, then hash the W=1..8 decode rows) run in **separate
processes** (a fresh context each).  The W=1 row-0 hash is the decode fingerprint.

| model / KV | P | graphs | starts | W=1 row-0 hash | vary? |
|---|---|---:|---:|---|---|
| Qwen3.5-4B Q8_0, f16 | 256 | on | 3 | `433d2e19e6f7bcac` | no |
| Qwen3.5-4B Q8_0, f16 | 256 | off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 6 | `433d2e19e6f7bcac` | no |
| Swift 27B Q5_K_M, q8_0 | 256 | on | 3 | `cec25c2aa3cc3e21` | no |
| Swift 27B Q5_K_M, q8_0 | 256 | off | 8 | `cec25c2aa3cc3e21` | no |
| Swift 27B Q5_K_M, q8_0 | 512 | on | 4 | `143a4ba08e1e20fe` | no |

Every start, both graph modes, both models, at both the 256- and the 512-column prefill shape (the
latter is the shape the *known* Windows flip needs): **bit-identical**.  W=1..8 did not vary within a
run either (`width_purity=PASS`).

The forward/prefill axis is also flat: `llama-perplexity --kl-divergence-base` self-comparison (the
r69 `GGML_CUDA_ASYNC_INPUT` instrument, 4B, `-ub 8`, `-c 512 --chunks 24`, graphs on) gives
`RMS Δp 0.001 %` / `same top p 99.984 %`.

**Conclusion for #67:** the reporter's graphs-on / `--no-kv-unified` start-to-start flip **does not
reproduce on Linux / ROCm 7.14**, at the decode-logit level or at the text level.  Their box is
Windows / ROCm 10, where two start-to-start flips of exactly this kind are already on record:

- **rope -> view -> set_rows** fused-vs-unfused clang contraction (`#58` item D; fixed bit-transparent
  in r25) and
- the residual **hipBLASLt solution selection** per process (issue **#67**, the *old* one, closed as
  external; workaround `ROCBLAS_USE_HIPBLASLT=0`, `ROCm/rocm-libraries#12126`).

Both are **prefill-driven** (the F32 `ssm_alpha`/`ssm_beta` `mul_mat` at 512 columns) and
**Windows/ROCm-10-only**; neither involves HIP graphs.  The reporter's graphs-off control was a single
run (and in their first follow-up a graphs-off run gave the alternative text once), so "graphs on is
the trigger" is not established by their data.  Most likely #67 is the same known Windows per-start
flip, not a new graphs or unified-KV root.

**Ask for the reporter (if #67 is pursued):** across fresh starts, with their usual greedy profile,
toggle `ROCBLAS_USE_HIPBLASLT=0` and `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` /
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1` (both already shipped for bisection), and run the graphs axis
at the **decode-logit** level rather than on one text sample.  On Linux those switches change the
logits only via the switching itself; on Windows they should show whether the flip is the fusion /
hipBLASLt one.

## 3. Status

- The campaign stays **PARKED**; its core finding is revalidated on r69 and the "fully green"
  wording is corrected.  Add the **solo-vs-concurrent** axis (batching) and the **graphs** axis to any
  future re-run.
- **TODO #67 is reclassified: not reproduced on Linux; most likely the known Windows-only
  prefill/fusion/hipBLASLt per-start flip.**  Keep it open only as "needs the reporter's Windows
  bisection"; do not treat it as a new delivery defect.
- No delivery code path is implicated by anything found here.

## 4. Tools added

- `tools/decodestart.sh` -- N fresh processes of `test-logits-width-probe`, prints the distinct
  W=1..8 hashes (the decode-start instrument used in §2).
- `tools/selfkld.sh` -- `llama-perplexity --kl-divergence-base` self-comparison across two processes
  (the **forward/prefill** axis only; it does not see decode nondeterminism).

Raw logs: `/tmp/kvoffset/` (`kvuni1.seq`, `kvuni0.seq`, `kvuni0b.seq`, `decode-*.log`, `*.kld`).
