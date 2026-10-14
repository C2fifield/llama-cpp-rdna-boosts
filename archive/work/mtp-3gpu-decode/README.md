# `mtp-3gpu-decode` — 3-GPU MTP decode is slower than plain decode (TODO #51)

**Status: OPEN (seeded 2026-10-11).** Not part of the delivery; nothing here may be applied to the
fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the
same item as **TODO #51**. It has no predecessor campaign of its own: it was *found and parked* during
`archive/work/slab-ring-region/` (now archived) and `TODO.md` names the **archive** directory
`archive/work/moe-verify-fusions/` as its "working home" — that directory is a record, not a live
campaign. **This directory is the campaign.**

> **Maintainer note (2026-10-11).**  3-GPU MTP is **still a bit slower than plain decode, but no longer
> 2.6x slower.**  So the r38 table in §0 is a **historical anchor**: the first task is to re-measure the
> current gap on r48 and re-scope from there.  Do not treat the 2.6x figure as the present symptom.

> **Cold start.** Read this file in full, then `AGENTS.md` ("Report findings before shipping", "Pushing
> policy", "WIP and promotion rules", "Default-on policy", "Critical facts"), the `GGML_CUDA_ALLREDUCE` /
> `GGML_CUDA_AR_*` rows in `ENVIRONMENT.md` §6, and the two predecessor records
> [`archive/work/slab-ring-region/HANDOVER.md`](../../archive/work/slab-ring-region/HANDOVER.md) §4.3.1
> and [`archive/work/allreduce-small-msg-latency/README.md`](../../archive/work/allreduce-small-msg-latency/README.md)
> (whose "Not measured: More than 2 GPUs" **is** this item). Report findings to the maintainer before any
> delivery change and wait.

## 0. The symptom

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

## 1. Re-measure on r48 FIRST — several relevant things changed after the finding

The table above is **r38**. The following landed between r38 and the current r48 and can all move the
3-GPU MTP number, so treat the table as the historical anchor, not the current truth:

- **r41 (block 12)** made the hybrid NCCL init **eager** again (the r34 lazy mid-run init was the
  `-sm tensor -ncmoe` MTP-impurity cause). That changes when RCCL is brought up relative to the first
  prefill/verify.
- **r45/r46 (block 16, folded into block 12)** added the small-message all-reduce knobs
  `GGML_CUDA_AR_SMALL_BLOCKS` (default **4**) and `GGML_CUDA_AR_SMALL_BYTES` (default 64 KiB), plus the
  opt-in 2-device P2P path. 3-GPU verify messages are "small" (see §3) so the new default launch already
  applies to this item — but it was only validated on **2 GPUs**.
- **r39** moved the H2D staging ring into the slab; **r47/r48** changed the CE all-reduce to fail soft and
  fixed the `--load-mode none` meta event proxy. Neither is expected to matter, but the tree is not the
  tree the table was measured on.

**Step 0 is therefore a fresh 3-GPU vs 2-GPU A/B on `~/llama.cpp` (r48 tree) before touching anything.**
If the gap is gone or much smaller, close #51 and record why.

## 2. Reproducer (field harness)

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

## 3. What is known about the 3-GPU all-reduce path

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

## 4. Suspects and the investigation plan

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

## 5. Gates and discipline

- This is a **performance** campaign: no purity contract changes. Any candidate must still pass the
  coherence gate (same-seed `llama-cli`, 2- and 3-GPU `-sm tensor`) and the adaptive-MTP gate
  (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`, `--reasoning on/off` as appropriate).
- Record every A/B with the exact prompt hash, GPUs, tree/tip and acceptance. The field prompt is not
  degenerate but acceptance is a weak signal here — it was 0.90–0.91 in every arm — so report prefill,
  decode and the per-call AR profile, not just t/s.
- If the root cause is in an all-reduce shape or the scheduler, keep any change behind an env
  kill-switch and re-validate the 2-GPU path (the shipping default) before promotion.

## 6. Pointers

- `TODO.md` #51; `WORKLOG.md` 2026-10-10 (r39, the `slab-ring-region` promotion).
- [`archive/work/slab-ring-region/HANDOVER.md`](../../archive/work/slab-ring-region/HANDOVER.md) §4.3.1
  (the original measurement) and `WORKLOG.md` around the r39 entry.
- [`archive/work/allreduce-small-msg-latency/README.md`](../../archive/work/allreduce-small-msg-latency/README.md)
  + `RESOLUTION.md` (the r45/r46 small-message AR work; "Not measured: more than 2 GPUs" is this item).
- `ENVIRONMENT.md` §6 and the `GGML_CUDA_AR_*` rows (all-reduce knobs and their defaults).
- `ggml/src/ggml-cuda/allreduce-hip.cu` (chunked kernel, dispatch), `ggml-cuda.cu`
  (`ggml_backend_cuda_comm_is_small`, `*_comm_init_hybrid`, `*_comm_allreduce_tensor`).
- `benchmarks/mtp-adaptive-methodology.md` (the MTP gate).
