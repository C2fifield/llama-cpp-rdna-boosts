# Measurements 2026-10-21 (d) — MTP determinism: ROOT CAUSE = hipBLASLt per-start solution selection

Box: gfx1201, 3x R9700, **ROCm 10.1.0**, 16 cores, Linux.  Tree: `~/llama.cpp` @ `972f3b757` + this
campaign's patch set (`patch-state/0001..0006`), `build-rocm`.  Instrument: `../tools/fp.sh` (draft
candidate-logit fingerprint, N fresh processes per arm; §1.1 of the handover).

Config for every run below unless stated:

```
HIP_VISIBLE_DEVICES=0,1  gemma4 26B-A4B Q8_0 + mtp-gemma-4-26B-A4B-it.gguf
-ngl 99 -ncmoe 40 -sm layer --fit false -c 16384 --spec-type draft-mtp --spec-draft-n-max 3
-n 300 -t 8 --seed 42 --temp 0 --ignore-eos -f prompts/prose-rdna-boosts.txt
```

## 1. The answer

**The residual run-to-run divergence is not in our code.**  It is ROCm's **hipBLASLt solution selection
being random per process start** for an untuned FP32 GEMM size — issue #67 / 
[ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126).  `ROCBLAS_USE_HIPBLASLT=0`
removes it completely on this box.

| arm (N = fresh processes) | draft-logit fingerprints |
|---|---|
| **default** (graphs on) | `6f6923ef36` ×12, `51274cb0f2` ×4, `10e1f21f4c` ×2 — **3 classes, 6/18 deviate** |
| **`ROCBLAS_USE_HIPBLASLT=0`**, `-n 300` | `6f6923ef36` **×10 — 10/10** |
| **`ROCBLAS_USE_HIPBLASLT=0`**, `-n 3000` | `19eed3da33` **×3 (904 draft calls) — 3/3** |
| `ROCBLAS_USE_HIPBLASLT=0`, target graphs off | `6f6923ef36` ×2 |
| target graphs off, LT **on** (§4) | `8f795fbaf0` ×5 |

`call0_pos0` top-1 probability of the three classes: **0.458 / 0.352 / 0.439** (`call0_hist = 5507` in all).

## 2. Direct library evidence — the pick is visible and it is per process

`HIPBLASLT_LOG_MASK=160` on three fresh starts of the default arm.  Exactly **one** GEMM shape is
dispatched to hipBLASLt in the whole run, **30 times** (= one per target layer):

```
hipblaslt-bench --api_method c -m 128 -n 375 -k 2816 --lda 2816 --ldb 2816 --ldc 128 --ldd 128
  --transA T --transB N --a_type f32_r --b_type f32_r --compute_type f32_r
  --algo_method index --solution_index <140231 | 140232>
```

`m = 128` is Gemma4 26B-A4B's `n_expert`, `k = 2816` its `n_embd` ⇒ the **F32 MoE router** (`ffn_gate_inp`),
at prefill width `n = 375` (one width lands in the untuned-FP32 window; other widths take mmvf/MMQ).

| run | solution_index | fingerprint |
|---|---|---|
| ltlog_1 | 140231 | `6f6923ef36` |
| ltlog_2 | 140231 | `6f6923ef36` |
| ltlog_3 | 140232 | `51274cb0f2` |

**On this shape the two ids are NOT bit-identical** (the README §issue #67 wording "bit-identical
reference pair" describes the reporter's `ssm_alpha`/`ssm_beta` shape, not this one).

Upstream status (from discussion #196): fixed in rocm-libraries `c50ea60130f` (develop 2026-10-02, TheRock
nightlies 2026-10-06) but **not in ROCm 10.1.0**, whose pin branched 2026-09-09.  It therefore *does*
reproduce on **Linux** with 10.x — the README's "Linux / 7.14 does not reproduce" is true only of 7.14.

## 3. Cost of the workaround: none — it is slightly faster

Final `timings` of every run above:

| arm | prompt t/s | gen t/s |
|---|---:|---:|
| LT on (n=17) | 327.3 (sd 0.2) | 91.86 (sd **1.65**) |
| LT off (n=10) | **340.1** (sd 0.2) | **93.02** (sd **0.06**) |
| LT off, `-n 3000` (n=3) | 340.1 | 102.29 |

The gen-rate sd collapsing from 1.65 to 0.06 is the same finding seen through timing.

## 4. Retraction — the graph-capture arms were confounded by the pick

Earlier this session the arms below were read as "the target's captured graph is the defect".  They are
not; **graph mode only changes which solution gets picked**:

| arm | LT on | LT off |
|---|---|---|
| target graphs off (`GGML_CUDA_GRAPH_OFF_NON_DRAFT=1`) | `8f795fbaf0` ×5 | **`6f6923ef36` ×2** |
| draft graphs off (`GGML_CUDA_GRAPH_OFF_DRAFT=1`) | `6f6923ef36` ×1 | — |
| target 1-token graphs off | `6f6923ef36` ×3 | — |
| target verify (4-token) graphs off | `51274cb0f2`/`6f6923ef36`/`10e1f21f4c` | — |
| forced fresh capture on every replay | `6f6923ef36` ×4 | — |
| all graphs off | `8f795fbaf0` | — |

With the library path off there is **exactly one** class, graphs on or off.  So: `8f795fbaf0` was a
per-start pick that correlated with the graphs-off arm (5/5), not a graph effect, and "graphs off is
stable" was a coincidence of the same lottery.  **Nothing about capture/replay is implicated.**

Other negatives established on the way (all with the instrument, each ≥1 arm):

* The **draft's** graph replay is not the cause (draft-only graphs off keeps the same class).
* Not stale baked pointers: forcing a fresh capture on every replay is deterministic (one class).
* Not the cross-context embedding path: the draft's input embedding is bit-identical across classes —
  `SPCHDBG` hashes of `pending_h`/`verify_h` for the three prompt ubatches agree exactly
  (`f0d2bae99caa65f3`, `e513b78ddd81c4ff`, `ca86774c130519a2`) between a class-A run and a graphs-off run.
* The async-input ring arms re-tested **after** `0005` (the handover's table was measured before it):
  `ASYNC_INPUT_BATCH=0` is itself a *stable* class (`51274cb0f2` ×4 — a pick), `ASYNC_INPUT=0` is still a
  lottery (6f69/10e1/5127) — i.e. the ring is not the residual cause either.

**Open detail (not a claim).**  The only differing computation is the router GEMM, yet the last prompt
token's `h_nextn` hash is identical across the classes compared.  The likely channel is the draft's
256-position SWA window over the target's KV rows written by the affected prefill *chunk* (the draft is
the only consumer of those rows that differs).  Nothing in the fix depends on it.

## 5. What this means for the campaign

* The determinism item is **closed as external** (issue #67), not fixed in the patch set.  Bar met:
  10/10 at `-n 300`, 3/3 at `-n 3000`, single class with graphs on **and** off.
* `0005` (per-origin async-input ring) is still a **real bug fix** (it took the arm from *unstable in
  every sample* to 8/10 by removing a genuine cross-context ordering defect) and should ship, but it was
  not the last 20 %.
* **Every fresh-process gate in this repo is exposed** to this lottery whenever a prefill width lands an
  F32 GEMM in the untuned hipBLASLt window — including cross-build text comparisons whose reference was
  recorded in a different process.  The delivery's single-context `W=1..8` probe at P=512 stayed
  bit-stable 6/6 (`c97d5e9d8bf6e0d4`), so the exposure is prefill-width, not decode.
* This also explains the two "discrepancies" recorded in `HANDOVER-2026-10-21b.md` §6 (non-reproducible
  gemma4 text hashes, and gen 74.9/72.8 vs the recorded 79.0/74.7).
