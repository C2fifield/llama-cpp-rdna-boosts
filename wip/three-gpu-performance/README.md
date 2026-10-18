# `three-gpu-performance` — 3-GPU decode is slower than 2-GPU (MTP and plain, partial and fully resident)

**Status: REVIVED (2026-10-16) from `archive/work/mtp-3gpu-decode/` (the r55 closure).**  This is the same
item as the former **TODO #51**, now scoped wider: it is **not MTP-only** and it reproduces in the
**fully-resident** case.  Not part of the delivery; nothing here may be applied to the fork without the
maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).

> **Revival note (2026-10-16).**  The campaign was closed at r55 with no code change because a fresh A/B
> then found 3-GPU MTP **faster** than 2-GPU (74.3 vs 65.0 t/s, 32K prompt, IQ4_NL).  A 2026-10-16
> measurement on r73 with the GSQ model and a 13.8K prompt found the **opposite** (3-GPU 56.9 vs 2-GPU
> 84.4 t/s) — and present at 100 % arena residency.  Either the r55 arm is not comparable (model/prompt/
> length) or a regression landed between r54 and r73; **step 0 is to re-run the r55 harness on r73.**  The
> closure record is kept as [`RESOLUTION-2026-10-13-closed.md`](RESOLUTION-2026-10-13-closed.md).

## 0. Current symptom and findings (2026-10-16, r73)

Measured on this box: Qwen3.8-Flash-Next GSQ-IQ3_XXS, `-ncmoe 48 -sm tensor`, 13.8K-token prompt, 200
generated, `--spec-type draft-mtp-adaptive --spec-draft-n-max 7`, WITH the `first-prefill-latency`
ring+wide reservation so the arena is correctly sized (and not evicted) in every arm:

| GPUs (env) | arena at load | prefill t/s | decode t/s |
|---|---|---:|---:|
| 1 (`HIP_VISIBLE_DEVICES=2`) | 5355 MiB (13 %) | 1065 | 37.1 |
| 2 (`0,1`) | 23340 MiB (57 %) | 1246 | **84.4** |
| 3 (`0,1,2`) | 40925 MiB (100 %) | **1258** | **56.9** |

Two facts to build the campaign on:

1. **3-GPU decode is ~33 % slower than 2-GPU** (56.9 vs 84.4) while its **prefill is the fastest** — the
   same shape as the r38 finding.  This arm is MTP; the campaign must establish whether plain decode has
   the same deficit.
2. **It is present in the fully-resident scenario.**  On 3 GPUs the whole host-expert set fits
   (100 % residency, identity fast path); on 2 it does not (57 %).  So it is **not** an arena-sizing or
   cache-occupancy artifact — the `first-prefill-latency` work removed that confound (arena sized at load,
   only 124-246 MiB evicted on the first prefill).  The gap is in the compute/communication path.

The r55 closure found the opposite (3-GPU MTP 74.3 > 2-GPU 65.0), so the model, the prompt length, or a
regression between r54 and r73 matters.  Re-measure before theorising.

### 0.1 Related observation (r79): the P-axis MTP drift is not a 3-GPU effect

The r79 PR #167 acceptance pass measured a reproducible prose (P) MTP loss from a rounding-level MoE
dot-product change.  It is **not** the 3-GPU decode deficit, and it is not specific to 3 GPUs:

| build / topology | R | P | C | K | X (phase-switching) |
|---|---:|---:|---:|---:|---:|
| gfx1100 (1 GPU), PR #167 0001 default-on, r78 -> r79 | +1.8 % | **-2.7 %** | +1.2 % | +1.4 % | +3.0 % |
| gfx1201 (2 GPU tensor), PR #167 0002 opt-in vs default | +3.4 % | **-1.65 %** | -0.5 % | +0.8 % | not run |
| gfx1201 (3 GPU tensor), PR #167 0002 opt-in vs default | -0.2 % | **-1.4 %** | +0.5 % | +0.4 % | not run |

The P loss tracks drafter acceptance on the new greedy continuation (gfx1100: 0.772 -> 0.718, mean
accepted length 3.32 -> 3.15), while **plain** decode on the same prompt is faster (gfx1100 +0.7 %,
gfx1201 flat).  It reproduces at `-n 2000`, `-n 3000` and fixed `n_max 3`, so it is content-dependent
rounding drift, not a 3-GPU cost.  Keep it out of the 3-GPU diagnosis; if it needs a fix, it belongs to
the adaptive controller's response to a changed continuation, not to the split/all-reduce path.
Full numbers: [`../../benchmarks/2026-10-17-pr167-r79-acceptance.md`](../../benchmarks/2026-10-17-pr167-r79-acceptance.md).

### 0.2 The host link is PCIe 5.0 **x4** per card, so multi-GPU prefill is link-bound (2026-10-17)

Every 3-GPU number in this campaign (and every prefill scaling claim) has to be read against the host
topology: each R9700 hangs off its own PCIe switch, the GPU-side link is x16 @ 32 GT/s, and the switch
**uplink to the host is x4 @ 32 GT/s** (`/sys/bus/pci/devices/0000:01:00.0/current_link_width` = 4, root
ports `00:01.1` and `00:01.2` also x4).  That is ~16 GB/s raw per card, half of a PCIe 5.0 x8 host and a
quarter of x16.  The contributor boxes this campaign compares against are x8 (R9700 pair) and x16
(gfx1100 7900 XTX on PCIe 4.0), so their cross-device bandwidth is 2-4x ours.

Measured scaling on 35B-A3B Q4_K_M, `q8_0` KV, `llama-bench -p 31500 -n 0 -r 3` (r81):

| GPUs | split | pp31500 t/s | vs 1 GPU |
|---|---|---:|---:|
| 1 | layer | 3613.8 | - |
| 2 | tensor | **4481.5** | **+24 %** |
| 3 | tensor | **4347.0** | **+20 %** (and **-3 %** vs 2 GPU) |

The section-0 field config shows the same shape (prefill 1065 / 1246 / 1258 t/s for 1 / 2 / 3 GPUs, so
+17 % then +1 %).  A healthy x8/x16 host would put the tensor-split prefill at +80-95 % for the second
card.  So:

* **The link is a first-order suspect for the 3-GPU decode deficit too** (per-token all-reduce traffic is
  small but the round-trip latency and the split/gather frequency are not), and it must be controlled
  for before blaming the AR implementation.
* **Cross-device prefill claims from an x8/x16 host can only be under-measured here.**  Record them as
  "not measurable on our link" rather than "does not reproduce": PR #175's 2-GPU tensor prefill
  +11.7 % is the worked example (see `../../WORKLOG.md` 261017.008 and
  `../../archive/work/mrope-rope-fusion/RESOLUTION.md`), while its decode gain did reproduce at 2-GPU
  tensor because the per-token AR traffic is negligible.

## 1. The work matrix this campaign must fill
Do not chase one number.  Measure the full grid, each with the arena resident and the exact prompt hash:

| axis | arms | why |
|---|---|---|
| device count | 2, 3 | the headline |
| speculation | `--spec-type none`, `draft-mtp`, `draft-mtp-adaptive` | is it MTP-only or the plain forward? |
| cache residency | partially resident, **fully resident** (`MOE_EXPERT_CACHE_MIB` large enough for 100 %) | the 3-GPU run is fully resident; the 2-GPU control is not |
| split mode | `-sm tensor`, `-sm layer` | does the cost track the tensor split or the device count? |

Only after the grid is filled does the r38/r55 discrepancy get resolved and the suspect list get ordered.

## 2. Historical symptom (r38 anchor)

<!-- The pre-revival §0, kept as the historical anchor. -->

On 3 GPUs MTP was a **loss**, not a win, on clean r38 — and it hits the production config (`AGENTS.md`:
servers run unpinned, 3-GPU, `HIP_VISIBLE_DEVICES=0,1,2`, hybrid all-reduce default). Measured
2026-10-10 on the field config (Qwen3.8-Flash-Next IQ4_NL + shared Q8_0 MTP head, ~32k-token prompt,
1000 generated, `--spec-draft-n-max 3`, `-sm tensor -ncmoe 48`):

| config | prefill | decode | acceptance |
|---|---:|---:|---:|
| 2-GPU plain (`--spec-type none`) | 546 | 41.9 | — |
| 2-GPU MTP | 961 | **64.8** | 0.91 |
| 3-GPU plain | — | **52.2** | — |
| 3-GPU MTP (ring-in-slab, r38 WIP) | 1043 | **19.5** | 0.90 |
| 3-GPU MTP, ring off (`GGML_CUDA_SLAB_RING_MIB=0`) | 759 | 21.4 | 0.91 |
| **3-GPU MTP, clean r38 (no WIP patch)** | 768 | **20.1** | — |
| **3-GPU plain, clean r38** | 927 | **53.0** | — |

The verify step costs **~190 ms on 3 GPUs** against **~57 ms on 2**. It reproduces on **clean r38**, so it
is **not** the H2D-ring WIP; it is a pre-existing 3-GPU property. The all-reduce choice
(`GGML_CUDA_ALLREDUCE=ce` vs the hybrid default) makes no difference; `ce` is 2-device-only anyway
(`ENVIRONMENT.md` §6). It is also orthogonal to cache residency (3-GPU logs 99.1 % residency).

## 3. Re-measure on r73 FIRST — several relevant things changed after the finding

The table above is **r38**. The following landed between r38 and the current r73 and can all move the
3-GPU MTP number, so treat the table as the historical anchor, not the current truth:

- **r41 (block 12)** made the hybrid NCCL init **eager** again (the r34 lazy mid-run init was the
  `-sm tensor -ncmoe` MTP-impurity cause). That changes when RCCL is brought up relative to the first
  prefill/verify.
- **r45/r46 (block 16, folded into block 12)** added the small-message all-reduce knobs
  `GGML_CUDA_AR_SMALL_BLOCKS` (default **4**) and `GGML_CUDA_AR_SMALL_BYTES` (default 64 KiB), plus the
  opt-in 2-device P2P path. 3-GPU verify messages are "small" (see §5) so the new default launch already
  applies to this item — but it was only validated on **2 GPUs**.
- **r39** moved the H2D staging ring into the slab; **r47/r48** changed the CE all-reduce to fail soft and
  fixed the `--load-mode none` meta event proxy. Neither is expected to matter, but the tree is not the
  tree the table was measured on.

**Step 0 is therefore a fresh 3-GPU vs 2-GPU A/B on `~/llama.cpp` (r73 tree) before touching anything.**
If the gap is gone or much smaller, close #51 and record why.

## 4. Reproducer (field harness)

The harness is preserved in the repo (the `/tmp/srr` copies are transient):

- [`archive/work/moe-verify-fusions/tools/field3.sh`](../../archive/work/moe-verify-fusions/tools/field3.sh)
  — MTP arm (`--spec-draft-model`, `--spec-type draft-mtp --spec-draft-n-max 3`).
- [`archive/work/moe-verify-fusions/tools/field_nospec.sh`](../../archive/work/moe-verify-fusions/tools/field_nospec.sh)
  — plain arm (`--spec-type none`, no draft model).

Both default to the field model and head (present on this box):

```
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
D=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

Prompt (must not be regenerated; a result is valid only against the hash it names):

```
wip/kv-cell-offset-nondeterminism/prompts/mixed30k.txt
sha256 69624f4d207f40bdf7357e74ec978af5bfb4725a0968faf4eea4aab46c2ae241
```

Run (server on `127.0.0.1:8942`, `LOGD=/tmp/srr`; the scripts accept `GPUS`, `M`, `D`, `PROMPT`):

```bash
mkdir -p /tmp/srr
cp wip/kv-cell-offset-nondeterminism/prompts/mixed30k.txt /tmp/srr/mixed30k.txt
cd archive/work/moe-verify-fusions/tools

# MTP arm, 3 GPUs (hybrid AR default), then the same on 2 GPUs as the control
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1,2 ./field3.sh      ~/llama.cpp/build-rocm-hybrid t3  run30kfit
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1   ./field3.sh      ~/llama.cpp/build-rocm-hybrid t2  run30kfit

# plain arm (no MTP) as the reference for the "is MTP a loss?" question
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1,2 ./field_nospec.sh ~/llama.cpp/build-rocm-hybrid t3n run30kfit
```

Read `prompt_tps` / `predicted_tps` from `/tmp/srr/<tag>.timing`; the acceptance line and the cache
eviction tail are printed by the script. Do not run parallel/background benches.

## 5. What is known about the 3-GPU all-reduce path

`GGML_CUDA_ALLREDUCE=hybrid` (the default) routes **small** tensors to the internal host-staged pipeline
and large tensors to RCCL:

- `ggml_backend_cuda_comm_is_small()` (`ggml-cuda.cu:2598`): `n_backends == 3 && ne < 131072` → internal.
  A 1-token decode (ne ≈ hidden) and a 4-token verify (ne ≈ 4×hidden) are both far below 131072, so
  **the verify all-reduce is served by the internal pipeline on 3 GPUs.**
- In `ggml_cuda_ar_allreduce_impl` (`allreduce-hip.cu`) the fast paths are **2-device only**:
  `use_copy_engine` requires `p->n_devices == 2`, and the P2P push path requires exactly 2 devices
  (`GGML_CUDA_AR_P2P`; `ENVIRONMENT.md` §6). On 3 devices the call falls to the **chunked kernel over the
  GTT host wire ring**, where each device writes its shard, does an all-thread `__threadfence_system()`,
  then waits for **every** peer (two spins) before phase 3.
- `GGML_CUDA_AR_SMALL_BLOCKS` (default 4) and `GGML_CUDA_AR_SMALL_BYTES` (64 KiB) apply to the chunked
  launch (`allreduce-hip.cu:1607`). The r45 tuning found 4 blocks best on **2** devices; 3 devices were
  never measured.
- `GGML_CUDA_AR_PROFILE=1` prints per-call `min/p50/p90/p99/max` (µs) and per-device
  `phase1(entry→signal)` / `spin` percentiles at pipeline teardown. This is the first instrument to run.

## 6. Suspects and the investigation plan

Work them in order; each has an A/B and a cheap first measurement.

1. **Internal all-reduce on 3 devices (highest prior).** Profile with `GGML_CUDA_AR_PROFILE=1` and compare
   the per-call p50/p90 and the spin split between 2 and 3 GPUs. A/B
   `GGML_CUDA_ALLREDUCE=internal`, `=hybrid`, `=nccl`; sweep `GGML_CUDA_AR_SMALL_BLOCKS=1,2,4,8` and
   `GGML_CUDA_AR_SMALL_BYTES`; try the WIP `GGML_CUDA_AR_PACE=1` lockstep (default off). If the 3-device
   chunked kernel is the cost, the fix space is: a 3-device-aware reduction shape (ring vs the current
   all-peers read), a copy-engine/CE-style path extended past 2 ranks, or a block-count/`small_bytes`
   retune for 3 ranks. Note `use_copy_engine` and P2P are deliberately 2-device; extending either is a
   real design item, not a one-liner.
2. **Verify-width tensor-split gather.** A 4-token verify under `-sm tensor` reads expert weights from
   three devices; check the per-op upload/copy machinery and the expert-cache band. Compare the 3-GPU
   verify cost with `-sm layer` at the same width (does the cost track the tensor split or the device
   count?), and with the cache off (`MOE_EXPERT_CACHE_MIB=0`).
3. **Per-op graph capture re-firing.** The finding noted `graphs reused` is 267 on 3 GPUs vs 246/270 on 2.
   Since r25 a graph is **recaptured when the pool / FA-staging / H2D-ring memory it captured is freed**
   (`GGML_CUDA_GRAPH_MEM_GEN=0` disables); more splits → more captures. Check the graph-reuse counts and
   whether the 3-GPU verify path recaptures each step (`GGML_CUDA_DISABLE_GRAPHS=1` as the A/B).
4. **Draft/target device placement.** The MTP head and target share one process; check where the draft
   lands under `-sm tensor` on 3 devices and whether the draft steps serialize on one device. The
   `-sm layer` control from suspect 2 also speaks to this.
5. **Small-message AR per-call latency vs kernel-dispatch floor.** The `allreduce-small-msg-latency` note
   found that on 2 GPUs with MTP the all-reduce is *not* the limiter (a ~2.6–2.9 µs dependent-kernel
   dispatch floor dominates). If the 3-GPU profile shows the AR is not the cost, look at the number of
   kernels/graphs per verify step on 3 devices instead.
6. **Fully-resident identity path + tensor-split per-op traffic (NEW, 2026-10-16).**  The 2026-10-16 3-GPU
   arm is 100 % resident (identity fast path); the 2-GPU control is 57 %.  First control the residency:
   compare 2 vs 3 GPUs at the SAME occupancy (force it with `MOE_EXPERT_CACHE_MIB`, or pick a model whose
   host set fits both).  Then compare `-sm tensor` vs `-sm layer` at 100 % residency — if the deficit
   tracks the tensor split (per-op gathers / cross-device reads of the expert band) it is not the
   all-reduce.  `GGML_CUDA_AR_PROFILE=1` plus per-op timing of the MUL_MAT_ID / verify band is the first
   measurement.

## 7. Gates and discipline

- This is a **performance** campaign: no purity contract changes. Any candidate must still pass the
  coherence gate (same-seed `llama-cli`, 2- and 3-GPU `-sm tensor`) and the adaptive-MTP gate
  (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`, `--reasoning on/off` as appropriate).
- Record every A/B with the exact prompt hash, GPUs, tree/tip and acceptance. The field prompt is not
  degenerate but acceptance is a weak signal here — it was 0.90–0.91 in every arm — so report prefill,
  decode and the per-call AR profile, not just t/s.
- If the root cause is in an all-reduce shape or the scheduler, keep any change behind an env
  kill-switch and re-validate the 2-GPU path (the shipping default) before promotion.

## 8. Pointers

- `TODO.md` #51; `WORKLOG.md` 2026-10-10 (r39, the `slab-ring-region` promotion).
- [`archive/work/slab-ring-region/HANDOVER.md`](../../archive/work/slab-ring-region/HANDOVER.md) §4.3.1
  (the original measurement) and `WORKLOG.md` around the r39 entry.
- [`archive/work/allreduce-small-msg-latency/README.md`](../../archive/work/allreduce-small-msg-latency/README.md)
  + `RESOLUTION.md` (the r45/r46 small-message AR work; "Not measured: more than 2 GPUs" is this item).
- `ENVIRONMENT.md` §6 and the `GGML_CUDA_AR_*` rows (all-reduce knobs and their defaults).
- `ggml/src/ggml-cuda/allreduce-hip.cu` (chunked kernel, dispatch), `ggml-cuda.cu`
  (`ggml_backend_cuda_comm_is_small`, `*_comm_init_hybrid`, `*_comm_allreduce_tensor`).
- `benchmarks/mtp-adaptive-methodology.md` (the MTP gate).
