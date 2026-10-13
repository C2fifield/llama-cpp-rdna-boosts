# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.  **Forward-looking only**: this
file lists what is still open (active work, waiting items, accepted limitations, parked ideas);
closed and retired work lives in `WORKLOG.md` and the dated records it points to.  Details never
live here — they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `wip/*` and `benchmarks/`.

**Current state (release `v12-097cf33d9-r51`, 2026-10-13):** the delivery is the **12-block V2.0 set**
against fork point **`097cf33d9`**, canonical tip `15d62da2f`, net tree
**`48cf3e8107a3e58e3bc5c10a9b5087f73a88d1f1`** (`validate-set.sh` green; `apply-all.sh` on a fresh clone
reproduces the tree).  V2.0 re-organized the former 16-block set by concern and moved each concern into a
self-contained function/file; the applied tree is behaviour-preserving (all same-seed gates byte-identical
to r50).  See `WORKLOG.md` 261013.001 (r51).  **Previous (16-block) state:** r49 added the mmap
host-memory reclaim (block 06): the loader no longer faults the
whole model file in up front and it unmaps each tensor's mapping range right after the copy to its device
buffer, so a `-ncmoe` load never holds the full mapping and the pinned host master at once
(`LLAMA_LOAD_RELEASE_MMAP=0` is the kill-switch; peak VmRSS on Qwen3.8-Flash-Next-IQ4_NL `-ncmoe 40`,
2 GPUs: 122.5 -> 55.4 GiB, equal to `--load-mode none`), and folds two AMD consistency fixes into block
15 (the Q1_0 `__byte_perm` -> `v_perm_b32` port and the `cpy-batch.cu` device idiom; both bit-identical
or no-ops -- TODO #54 and #32 closed).  Before it, r48 fixes **issue #137** (the `--load-mode none` + `-sm tensor` + `-ncmoe` startup
GPU page fault): the Meta device now reports `host_buffer`/`events` and proxies its event iface, so the
loader's pinned async-upload pipeline engages and the pageable `read_buf` SDMA path is gone; adapted from
upstream PR #22466 and folded into **block 06** (see `WORKLOG.md` 261011.006 (r48)).  Before it, r46
**folds the r45 block 16 back into the 0000-0015 set** (host-side/system work
into block 06, the small-message all-reduce into block 12, and the decode/latency remainder into block
15) and lands **issue #133** in block 15: the RDNA4 decode/verify band-WMMA predicate now folds GQA
ratios up to `GGML_HIP_FA_BAND_WMMA_GQA_MAX` (default 12), so qwen4exp GQA-12 no longer falls to the
tile kernel (2x R9700 deep MTP +36 %).  Before it, r41 made block 12's hybrid NCCL init **eager** again (the r34 lazy mid-run init was
what made `-sm tensor -ncmoe` MTP-impure under the auto cache; the internal AR is bit-exact), so the 35B-A3B
`-ncmoe 99` + auto-cache field is `none == n1 == n3 == n7` and `none == n3` on the long field, and
qwen4exp stays pure (**#53 closed**).  The previous paragraph records the r32 state.  `release.json` is the source of truth.  r32 fixes the two bugs found during the
PR #115 review: **#48** the post-prefill re-reserve now uses the current ubatch's sequence count so a
multi-sequence decode does not build a zero-token attention graph (block 06), and **#49** the meta
split-state computation is pre-warmed bottom-up so a >1200-node `src` chain no longer overflows the
stack (block 15); both `TODO` items are closed.  Before it, r31 folded PR #115 into block 06 (the MoE
expert-cache decode/verify band follows the routed-expert MMVQ band, 16 tokens on RDNA4, 8 on
RDNA3/NVIDIA, via a single-owner `moe_cache_band` hook and a device-side admission fill-list fix;
`GGML_MOE_CACHE_MAX_TOK=8` restores the old band).  Before that, r30 root-caused issue #113: the r29 bf16
divergence was an `A_sc` stride aliasing bug for `n_seqs > 1`, so `GGML_CUDA_GDN_CHUNKED_BF16` is
default-on again (near-lossless).  Before it, r29 flipped it off as a workaround; r28 folded the
`--host-experts` flag into block 06, r27 PR #114's four bit-identical
qwen4exp decode fusions into block 15 (default-on `=0` kill-switches).  Before it, r26 folded PR #107's
DFlash F1 default-on and PR #110's DPP butterflies, and r25 folded eight of the nine PR #106 (issue #105)
patches into blocks 06/08/14/15 plus `TODO #45` (patch 0002, the data-pointer graph key, is dropped, see
#46).  The release-by-release record (r24's cache-floor fix + repack, r22's movable-boundary slab,
movable-boundary slab, r21's `llama-cli` drop, r20's compute-buffer slack, r19/r18/r17, ...) is in
`WORKLOG.md` and the campaign records under `archive/work/`; the closed tracker items (#37/#40/#41/#43,
#42, #44/#39/#38, and now #35) are in `WORKLOG.md` as well.  See `AGENTS.md` for the policy layer.

## Active (kept compact: only what this repo will work on next)

### 57. `test-llama-archs` CPU mixed-batch FAIL is a delivery regression — CORRECTNESS

**Opened 2026-10-12 (found during the `patch-block-reorg` Phase-1 block-10 gate; recorded as a known
issue).**  `test-llama-archs`'s **CPU** row (`AMD Ryzen … | MoE | ... | Mixed batch`) FAILs for the
published delivery but **passes at the upstream base `097cf33d9`**:

| build | `qwen4exp` CPU Mixed batch | `deepseek2` CPU Mixed batch |
|---|---|---|
| base `097cf33d9` | **OK (3.32e-14)** | **OK (6.28e-10)** |
| reference tip `84058162b` (r50) | **FAIL (3.52e-01)** | **FAIL (9.37e-01)** |
| v2 block 10 `27f159a7a` | **FAIL (3.52e-01)** (bit-identical to r50) | (same) |

So it is a **pre-existing delivery defect in the published tip**, not a reorg artifact — the same class
as #56.  It was missed because the release gates do not run `test-llama-archs`.  Reproduce:
`build/bin/test-llama-archs -a qwen4exp -s 42` (CPU row) vs the upstream base (passes).

### 56. `test-llama-archs -a deepseek2` GPU memory fault with host experts + `--moe-cache-mib` — CORRECTNESS

**Opened 2026-10-12 (found during the `patch-block-reorg` Phase-1 block-09 gate; recorded as a known
issue, maintainer decision 2026-10-12).**  `test-llama-archs -a deepseek2` (the only arch the upstream
test gives a host-expert / MoE-cache config) aborts with a GPU memory access fault during the
`deepseek2 / MoE cache` config's first decode:

```
kernel: void mul_mat_vec_f<float, float, 1, 64, false, true>(...)   # multi-token MUL_MAT_ID
Memory access fault ... faulting addr: 0x7f…  (a HOST/pinned range)
```

The MUL_MAT_ID kernel reads the pinned host expert table (`ROCm_Host`) past its mapping.  Bisected with
fixed seed `-s 42`: upstream base `097cf33d9` **PASS 7/7**; v2 reorg block 08 `a32c95a6e` **PASS**;
v2 reorg block 09 `4463f18c7` (`slab-moe`) **FAULT**; the **published reference tip `84058162b` also
FAULTS** with the identical signature.  So it is a **pre-existing delivery defect inside the block-09
MoE stack** (pinned host-expert loader / Meta / scheduler / cache hooks), **not** a reorg or block-10
artifact; it escaped the release gates because none runs `test-llama-archs`.

The test's cache is the **upstream** engine (`ctx_params.moe_cache_size` / `--moe-cache-mib`), not the
delivery's `MOE_EXPERT_CACHE_MIB` engine — which is why `GGML_CUDA_SLAB=0`, `MOE_EXPERT_CACHE_MIB=0`,
`MOE_EXPERT_CACHE_FILL=0`, `GGML_CUDA_DISABLE_FUSION=1` and `GGML_CUDA_GRID_STALL_PAD=0` are all inert
here (the grid-pad / `nrows`-guard and `batched_reduce` hypotheses are excluded).

**Fix direction:** root-cause the delivery's backend plumbing that the upstream `llama_moe_cache` engine
rides on (scheduler routed-expert rebalance / merged per-layer MoE split, Meta delegation, the pinned
`ROCm_Host` loader, `stage_*`/preflight), under the `--moe-cache-mib` + `--host-experts` path.
Reproduce: `build/bin/test-llama-archs -a deepseek2 -s 42` vs the upstream base (passes).
Full diagnosis: `archive/work/patch-block-reorg/PHASE1-STATUS.md` §15.

### 55. Remove the full pinned `ROCm_Host` master when `--host-experts pool` is on

**Opened 2026-10-11; scoping.**  The r49 mmap reclaim (block 06) removed the mmap half of the `-ncmoe`
host peak, but `--host-experts pool` still keeps the **full pinned master** (r35 shipped the pool as a
bounce tier; the source logs "Phase 2b; full pinned master kept").  `RssShmem` is identical with and
without the pool: 18.7 GiB on 35B-A3B `-ncmoe 40`, 54.1 GiB on Flash-Next IQ4_NL `-ncmoe 40`.
Dropping the master is the strongest lever for issue #139 (the amdgpu driver stops all of a process's
GPU queues when the kernel migrates a GPU-mapped host page).  **Working home:
`wip/host-expert-master-drop/`.**  Gates and the reader enumeration are in that README.

### 50. The cache-aware fusion guard is global while the slab's eviction is per-table (code `OPEN 2`)

**RESOLVED (2026-10-10), DELIVERED in `v16-a55e952b8-r40`.**  The guard is now **per-table**
(`moe_cache_table_serves`, mirroring the take-over's `identity` decision) and the **device-remap path that
could disagree with it has been deleted from the code base** (block 06 no longer creates it).  The eager
host-routing path is the only `-sm tensor` split policy: MTP-pure on qwen4exp (`none == n3`),
budget-sweep-pure (MIB8000 vs MIB14000 `None`), and within run-to-run noise of the old `DEVMAP=0`
performance (delivery MTP prefill 1585.2 vs 1587.0, decode 72.36 vs 72.56; plain prefill 1471.8 vs 1471.9,
decode 41.73 vs 41.80).  See `archive/work/moe-verify-fusions/FINDINGS-devmap-removal.md`.

**Working home: `archive/work/moe-verify-fusions/`** (tracked together with #51 — the maintainer suspects a shared root cause on the cache-band routed-MMVQ verify path).

**Opened 2026-10-10 (WIP finding, `archive/work/slab-ring-region/`).**  `ggml_cuda_cache_blocks_fusion`
(`ggml-cuda.cu:5804`) is gated on the **global** `moe_cache_has_arena()`, but the movable-boundary slab
evicts cache tables **per table** — `moe-expert-cache.cu` says *"the movable-boundary slab evicts the
tables in the chunks it hands to the work pool, so a PARTIAL cache is the normal state"*.  Any arena
change therefore flips the cache-aware fusions for the whole model *mid-run*: the H2D-ring region (item
49), the G4 refusal, `MOE_EXPERT_CACHE_RESERVE_MIB`, a `GGML_CUDA_SLAB_RESERVE_MIB` change.  The fused and
non-fused paths are not bit-identical (the code says the stand-down "changes the arithmetic"), so the
greedy output then depends on the **timing** of the eviction.  The code already carries the marker:
*"OPEN 2: the wholesale-fallback invariant … must hold for EVERY consumer, not just the fusion guard and
the take-over hook."*

Evidence + the A/B matrix: `archive/work/slab-ring-region/FINDINGS-numerics.md` F3.  On the field model
(ring-in-slab vs ring-outside) the placements differ; with **every** individual fusion switch off but the
`GGML_CUDA_DISABLE_FUSION`-gated alloc-deps pass still ON they agree, which also **excludes PR #27301's
alloc dependencies** as the cause (the deps do cost +512 MiB of compute buffer, charged to the arena).  Not
one bad kernel either — either the qwen4exp HC group or the idx/concat/DSV4 group alone removes it, so it
is a cumulative rounding shift.  No corruption: 0 NaN, 0 `////`, prefill logits bit-identical; the visible
symptom is one flipped near-tie at the first novel decode token.

**Fix direction:** make the guard per-table (as the marker asks) so a partial cache keeps serving, or make
the fused and non-fused paths bit-identical.

**Default decision (maintainer, 2026-10-10): fusions stay default-ON, unchanged.**  The divergence is a
pre-existing property of the arena, not of the ring, and disabling the fusions would cost more than it
fixes — see the measured cost and the re-enable recipe in `ENVIRONMENT.md` §7.

### 51. 3-GPU MTP decode is slower than plain decode (pre-existing)

**Working home: `wip/mtp-3gpu-decode/`.**  (Originally tracked with #50; re-seeded as its own campaign 2026-10-11.)

**Opened 2026-10-10 (found during the `archive/work/slab-ring-region/` 3-GPU gate).**  On 3 GPUs MTP is a *loss*,
not a win — and it hits the production config (AGENTS.md: servers run unpinned, 3-GPU,
`HIP_VISIBLE_DEVICES=0,1,2`, hybrid default).

**Maintainer note (2026-10-11):** 3-GPU MTP is **still a bit slower** than plain decode, but **no longer
2.6x slower**.  Re-measure the current gap on r48 first; the table below is the r38 anchor (several AR/
slab/scheduler changes landed after it).

Field config, Qwen3.8-Flash-Next IQ4_NL + the shared Q8_0 MTP head, 32 358-token prompt, 1000 generated,
`--spec-draft-n-max 3`, same tree:

| config | prefill | decode | acceptance |
|---|---:|---:|---:|
| 2-GPU plain (`--spec-type none`) | 546 | 41.9 | — |
| 2-GPU MTP | 961 | **64.8** | 0.91 |
| 3-GPU plain | — | **52.2** | — |
| 3-GPU MTP | 1043 | **19.5** | 0.90 |
| 3-GPU MTP, ring off (`GGML_CUDA_SLAB_RING_MIB=0`) | 759 | 21.4 | 0.91 |
| **3-GPU MTP, clean r38 (no WIP patch)** | 768 | **20.1** | — |
| **3-GPU plain, clean r38** | 927 | **53.0** | — |

The verify step costs ~190 ms on 3 GPUs against ~57 ms on 2.  **Reproduces on clean r38**, so it is not the
H2D-ring WIP (and `GGML_CUDA_ALLREDUCE=ce` vs the hybrid default makes no difference; `ce` is 2-GPU-only
anyway, ENVIRONMENT §6).  Also orthogonal to the cache residency (3-GPU logs 99.1 % residency).

**Reproducer:**

```bash
# the harness is preserved in the repo (the /tmp/srr copies are transient):
#   archive/work/moe-verify-fusions/tools/field3.sh       (MTP arm)
#   archive/work/moe-verify-fusions/tools/field_nospec.sh (plain arm)
# prompt: wip/kv-cell-offset-nondeterminism/prompts/mixed30k.txt
cd archive/work/moe-verify-fusions/tools
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1,2 ./field3.sh      ~/llama.cpp/build-rocm-hybrid t3  run30kfit
PROMPT=/tmp/srr/mixed30k.txt GPUS=0,1,2 ./field_nospec.sh ~/llama.cpp/build-rocm-hybrid t3n run30kfit
```

Suspects to start from: the 3-way tensor-split verify gather (the 4-token verify reads experts from three
devices), the draft/target device placement, and whether the per-op graph capture re-fires on 3 GPUs
(`graphs reused` is 267 on 3 GPUs vs 246/270 on 2).  Details: `archive/work/slab-ring-region/HANDOVER.md` §4.3.1.

### 52. Multi-sequence MTP is non-deterministic: a rollback reads a recurrent snapshot the last batch did not write — CORRECTNESS

**PARTIALLY RESOLVED, DELIVERED in `v16-a55e952b8-r42` (2026-10-10).** Part 1 is fixed in **block 02**:
`ggml_cuda_op_gated_delta_net_impl` took the whole-batch chunked kernel for `n_seqs > 1 && K == 1`
unconditionally while the solo path took the sequential kernel, so a sequence's GDN result depended on its
co-resident sequences; the multi-sequence branch now shares the single-sequence `n_tokens >
GDN_CHUNKED_MIN_TOKENS` gate. **Part B (the held multi-token batching change) is DELIVERED in `v16-a55e952b8-r43` (2026-10-11, block 06).**
It was correct; it only appeared to break `llama-perplexity` at `n_seq >= 4` because of a **logits-readback
ordering hazard** (the pinned async D2H copy was not ordered against the next ubatch's compute). The readback
is now completed before the next ubatch runs; PPL `8.5100`, the probe bit-identical, prefill-logit
`0.000707 / 98.755 %`. **The remaining component C** -- the `--kv-unified` server non-determinism (a
sequence's attention result depends on the physical offset of its KV cells modulo the FA reduction tile, and
it is upstream-inherent) -- is **PARKED (2026-10-11) in `wip/kv-cell-offset-nondeterminism/`** (handover +
cold-start prompt; refined findings in `FINDINGS-offset-mechanism.md`): the dependence is on the sequence's
whole physical *cell set*, a fixed alignment is insufficient, and a proper fix needs per-sequence
contiguous tile-aligned regions (an upstream redesign). Documented as a gotcha (README / `GREEDY-PURITY.md`
§38); issue #134 closed as upstream-inherent. Reopen the campaign if the maintainer or @DanoPTT decides it
is worth the redesign. It is a separate root from the recurrent rollback. The `seq_rm` boundary warning
(Issue #131) is a false positive in the cross-sequence case: the probe is bit-exact at the production
verify width, and **r44 block 02** now tracks the last-batch shape **per sequence** so the diagnostic
reports the right sequence. The deeper "large prefill as the last prepared batch before a rollback"
ordering is **OPEN and being resumed** in `wip/rs-rollback-boundary/` (seeded, not parked: see
`wip/CAMPAIGNS.md`). The predecessor record is
`archive/work/moe-verify-fusions/`. The text below is the original finding.

**Working home: `wip/kv-cell-offset-nondeterminism/` (part C); predecessor record `archive/work/moe-verify-fusions/`** (found 2026-10-10 while investigating the single-sequence
divergence — see [`HANDOVER-single-seq-rollback.md`](archive/work/moe-verify-fusions/HANDOVER-single-seq-rollback.md)).

**Symptom.**  `llama-server -np 2 --kv-unified --spec-type draft-mtp`, two concurrent greedy requests (A
long keeps decoding after B short finishes): the output is **not reproducible across identical
repetitions**, and the long request differs when concurrent vs solo.  This violates the per-sequence
independence the delivery already asserts (`test-recurrent-state-rollback`'s "seq-1-only decode
independent of seq 0", max diff 0).

**Reproduced on our r39 + narrow-2 build with NO PR #124 skip**, so it is **not** caused by PR #124
(whose reporter's "reference" is our tree's shape).  The run logs the invalid-rollback condition, verbatim:

```
seq_rm: rollback crossed a batch boundary: seq 1 rollback=2 but the last batch decoded 6136 tokens
        (last pos 18419, n_rs_seq=3, n_rs_batch=4). ... the restored state is wrong
```

i.e. a verify rejection rollback lands while the last decoded batch was a large prefill batch, and the
recurrent snapshot path reads a snapshot that batch never wrote (§27's invariant).

**Controls run (all on 2 GPU, GSQ + shared Q8_0 MTP):** identical under `GGML_CUDA_ALLREDUCE=ce` (so not
the all-reduce); `GGML_CUDA_GDN_CHUNKED=0` removes the snapshot substance but the instability **survives**
(so there is a second, batching-level component); concurrent repetitions still differ.  Note the warning
is emitted **once per process** (a `static` flag), so a later occurrence is silent — do not rely on the
log alone.

**Reproducer:** `archive/work/moe-verify-fusions/tools/multiseq.sh <build_dir> <tag>` (env `GPUS`/`M`/`D`), e.g.
`GPUS=0,1 ./multiseq.sh ~/llama.cpp/build-rocm-hybrid msq`.

**Fix direction (not yet chosen):** the §27 invariant — *every batch that can be rolled back into must
run the kernel that writes the snapshots it will read*.  Either (a) never present a verify rollback whose
`last_ubatch_nseq_tokens > n_rs_batch` (server/batching), (b) write the pre-batch plane for the chunked
prefill when a rollback can follow it, or (c) pin the rollback into the batch's own snapshot.  Confirm
with a matrix (2-seq × prefill-in-flight × `GGML_CUDA_GDN_CHUNKED`) and re-check the reporter's PR #124
scenario before that PR is ever merged.

**Re-check done 2026-10-11 (r45).**  PR #124 was **merged in block 16** on the maintainer's decision.  The
reporter's concurrent-sequence variation is the unified-KV/attention-offset non-determinism above (issue
#134) — reproduced on this tree with the skip absent — so it is not the skip.  Documented as the
`-np > 1 --kv-unified` gotcha (`--no-kv-unified` is deterministic); the skip itself is bit-identical in
the single-request and 2-GPU MTP gates (`archive/work/gdn-state-gather-skip/RESOLUTION.md`).

### 53. 35B-A3B (`qwen35moe`) is not bit-pure under `-sm tensor` + `-ncmoe` (MTP and plain forward) — CORRECTNESS

**RESOLVED (2026-10-10), DELIVERED in `v16-a55e952b8-r41`.**  The cause was **not** the internal
host-staged all-reduce (the AR is bit-exact: `out == local0 + local1` on every call) but block 12's
**lazy `ncclCommInitAll`** (r34): the mid-run init perturbs the layout/state and flips a near-tie in the
upstream split compute, so the AR *input* differs (`b2ee…` lazy vs `86cb…` eager/RCCL) while the output is
identical.  Making the hybrid init **eager** again makes it bit-identical to pure RCCL and restores the
product path: 35B-A3B `-ncmoe 99` + auto cache `none == n1 == n3 == n7` (short) and `none == n3` (long),
`-sm layer` pure, qwen4exp pure; perf is neutral (auto-cache long field MTP n3 172.1 t/s, plain 83.5).
Full record: `WORKLOG.md` 2026-10-10 (r41); `patches/README.md` block 12.  The cache-off / `-ncmoe 0`
(upstream-shaped) paths are pure too, but are not the product.  The text below is the original finding.

**Working home: `archive/work/moe-verify-fusions/`** (opened 2026-10-10; handover:
[`HANDOVER-35b-tensor-impurity.md`](archive/work/moe-verify-fusions/HANDOVER-35b-tensor-impurity.md)).

**Symptom.**  `Qwen3.6-35B-A3B-Q8_0` (embedded `qwen35moe.nextn` MTP head), `-sm tensor -ncmoe 99`, 2 GPU:
`--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3` **diverges at token 8**; the same model
under **`-sm layer` is bit-identical**.  Cache-independent (`MOE_EXPERT_CACHE_MIB=0` still diverges).

**The plain forward pass is width-impure** (no MTP, no cache): `test-logits-width-probe` on the 35B with
`NCMOE=99 SPLIT=tensor` gives `width_purity=FAIL (worst maxdiff 0.888169)` (W=1 hash `705d290898cf0699`,
W=2..8 `9577d960c75cb4bb`).  It needs the **Meta split of the host-resident expert weights**: `-sm layer`,
all-GPU, 1 GPU and `GGML_META_SPLIT_COPY=0` (mirrored) are all PASS.

**Narrowed:** `GGML_CUDA_DISABLE_MWR=1` and `GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1` each make the **probe**
PASS (the width selects MWR at W=1 vs the mmvq down fold at W>=2).  But the **field** is not fixed by
`DISABLE_MWR` (still first-diff 8) and only *moves* with `DISABLE_FUSION=1` (8 -> 72) — so expect two
overlapping `-sm tensor`+`-ncmoe` effects.  `qwen4exp` is unaffected and stays fixed.

**Not the cause:** the expert cache, the MTP machinery per se, chunked GDN, sparse QSA, the shared-expert
down gate alone, the mmvq MoE band / MMQ routed / weighted-down / GLU->Q8_1 / norm->Q8_1 paths.

### 48. The MoE expert cache corrupts a wide MTP-export consumer (G7) — CORRECTNESS

**ROOT-CAUSED, FIXED, GATED 2026-10-09 (WIP patch).**  The handover's alias/take-over hypothesis was
wrong: the skip is in the SAME split.  On the 1-GPU reproducer, `blk.47`'s weight has **two `MUL_MAT_ID`
consumers in one split** (the 0-row gathered logits tail and the 15-row unmasked MTP export); the scheduler
picked the first (0-row), hit `ggml_nelements(ids) == 0 -> continue`, and never filled the copy the export
reads.  The wide export never redirects to the arena (`moe_cache_get_table` declines every `ne[2] > band`), so
it read the unfilled copy -> NaN in `t_h_nextn`.

**Fix:** in `ggml_backend_sched_compute_splits`, count the `MUL_MAT_ID` consumers of the input copy; when
there is more than one, skip both the cache takeover and the pruned fill and copy the whole table.
Patch `archive/work/fit-slab-accounting/g7-multi-consumer-fill.patch` (46 lines, `ggml/src/ggml-backend.cpp`, sha256
`a041a4d8b4fc81c83ce2b0101e9336a088a3493e04ce4b3a9da815a81b3b9b84`).  Full handover, evidence and gates:
`archive/work/fit-slab-accounting/README.md` §13.6.

**Gates PASS:** acceptance **0.57353** for `-ncmoe {44,48,99}` (0 NaN); dense 4B `1c5d32ac537d`;
prefill-logit KLD **0.000707 / 98.755 %**; `MUL_MAT_ID` 931/931; 2-GPU GSQ r5 repro acceptance **0.72727**,
0 NaN.  **Next:** promote through block 06 (the scheduler half of the cache) — `patches/` is untouched by
this campaign.

**DELIVERED in `v16-a55e952b8-r38` (block 06).**  Original handover: `--spec-type draft-mtp` on Flash-Next (separate `-md`
head) can collapse draft acceptance to **0** with **NaN** draft logits while the target text stays
coherent.  Isolated to the cache: `MOE_EXPERT_CACHE_MIB=0` gives acceptance 0.588 and no NaN;
`GGML_OP_OFFLOAD_MIN_BATCH=100000` does not help; `-ncmoe 44` is fine, `-ncmoe 48` NaNs.  The qwen4exp
unmasked MTP export recomputes the last trunk layer's FFN on every row, so `blk.47.ffn_*_exps` has a
1-row decode-band consumer (whose copy `moe_cache_take_over` aliases to the arena and never fills) and a
wide export consumer that reads the same unfilled copy -> NaN in `t_h_nextn`.  The r5 `sched-moe-restage`
fix (`ggml-backend.cpp:1826`) is present but insufficient for this 1-GPU/`-sm layer`/`n_copies==1` config.
**Next:** prove the exact skip (ids-not-ready at staging time vs the copy-pointer alias map surviving
across graphs), then pick a fix from the four candidates.  Full handover, reproducer and gates:
`archive/work/fit-slab-accounting/README.md` §13.

### 47. Bring the MoE arena budget and slab headroom into `--fit` (G1 + G2)

**RETESTED 2026-10-09 (post-G7, r37 + revival).**  The Phase 1 patch applies to the current tree with
offsets; **the auto floor under `-sm tensor` no longer corrupts** (2/2 auto runs coherent, no arena
thrash — the Phase 1 `////` + `moe_cache_rearm` is gone).  Three re-cut fixes: the G2 CUDA getter was
**inert on r37** (`ggml_cuda_slab_enabled()` is `env_on && g_slab_armed`, false at fit time); and the
host-expert map attributed all 56762 MiB to device 0, so the MIB/auto floor reserved device 0 only — the
loader now distributes each `exps` tensor across the layer Meta device's simple devices, so G1 reserves
**both** devices (`20108 / 20110` MiB) and the post-prefill drop/rearm loop covers device 1.  Decisions:
the MIB is a **cap** (built arena must be 95-100 % of it), **G2 gate = b1**, and the **full §5.1-§5.7
matrix must pass before release** (no-abort MIB sweep, dense golden `1c5d32ac537d`, `GGML_CUDA_SLAB=0`
zeroes G2 are green).  Record + arms: `archive/work/fit-slab-accounting/README.md` §14.4; combined current-state
patch `fit-slab-r37-all-wip.patch`.  **Ordered release plan: README §15** (purity/determinism first, then
`ENVIRONMENT.md`, the full §5.1-§5.7 matrix, `fingon`, and promotion).

**DELIVERED in `v16-a55e952b8-r38` (block 06).**  `--fit` now reserves the MoE-arena budget and the
slab headroom; the auto floor under `-sm tensor` no longer corrupts, and per-device host-expert
accounting covers every device.  Record: `archive/work/fit-slab-accounting/README.md` §14.4/§15; `WORKLOG.md`
2026-10-09 (r38).  *(Phase 1 history, kept for context.)*

**Opened 2026-10-07; Phase 1 implemented then PARKED 2026-10-07.**  `--fit` is not a single VRAM planner for
ROCm: (G1) an explicit `MOE_EXPERT_CACHE_MIB` is invisible to the fit margin, so the context is sized as
if the arena did not exist; (G2) the slab's `GGML_CUDA_SLAB_HEADROOM_MIB` (4096, a HARD floor because
hipBLASLt loads Tensile code objects + workspace behind the app's back and cannot be routed into the slab)
is not modelled, and the `--fit-target` default (1 GiB) is smaller than it — the `exit 134` class.
Phase 1 (G1 + G2 + the tensor-split plumbing they need) was **implemented, built warning-free and tested**:
G2 (headroom) and G1 (explicit MIB) are safe, but newly enabling the **auto floor** under `-sm tensor`
reproducibly corrupts (2/2; `!!!!!!!!` output, arena thrash) — a latent layout-dependent bug the fix
exposes.  **PARKED** by the maintainer; the patch + full matrix + the open root-cause are in
`archive/work/fit-slab-accounting/PHASE1-ATTEMPT.md`.  `patches/` was never touched.  **Do NOT** recommend
disabling hipBLASLt (corrupt at a thin headroom, stunted at 4096).  To revisit: ship the safe subset
(headroom + explicit MIB, auto floor off under `-sm tensor`) or root-cause the corruption first.
Full handover: `archive/work/fit-slab-accounting/README.md`.

### 36. Genuine CPU/GPU overlap for the MoE misses (Strata's pipeline shape)

**Opened 2026-10-05; scoping only.**  The delivered cache-on path computes every expert on the GPU
(resident from VRAM, misses via UVA over PCIe); the cache-off path computes them on the CPU but
**serialised** with the GPU splits.  Our own CPU-computes-misses arm was correct and negative above a
~1.2 GiB arena (the loss is ~600 scheduler dispatches + ~3 cross-backend copies per layer/token, not
CPU compute; see `archive/work/moe-expert-cache/WORKLOG.md`).  Strata's custom engine runs the CPU pool
*because* it is not under llama.cpp's serial scheduler.  **Next:** O0 - measure how much of a 52 t/s
token is the UVA cold fraction at 43 % residency; if it is <20 %, park it.  Design space and gates:
`wip/moe-cpu-overlap/README.md`.

### 30. Default-flip the DFlash device-resident layer features (PR #73) — **CLOSED 2026-10-07 (r26)**

**Opened 2026-09-30** with r27.  PR #73's device path (`GGML_LF_DFLASH_DEV=1`) is now validated on
gfx1201 as output-identical to the host path and faster (`Qwen3.8-27B-DFlash2-Q4_K_M`, 27B
UD-Q4_K_XL, q8_0 KV, 5246-token prompt: `GGML_LF_DFLASH_DEV=0` and `=1` both give
`487 chars sha=dad22c4270ab`; prefill 1144.2 -> 1232.5 t/s, generation 63.8 -> 65.4 t/s).  It stays
opt-in because the device buffer allocation (`llama_context::extract_layer_inputs`) is a hard
`GGML_ASSERT` on failure: a nearly-full card aborts instead of falling back.  **Next:** give the
allocation the same warn-once-and-fall-back treatment as the FA staging arena (issue #33), then flip
the default on with `GGML_LF_DFLASH_DEV=0` as the kill switch.  Record: `WORKLOG.md` 2026-09-30 (r27).

**Maintainer note 2026-10-04:** DFlash2 was compared against the delivery's adaptive MTP across the
broad matrix and adaptive MTP came out ahead, so this is a **support** item (get the DFlash2 path
working correctly, without the hard assert), not a performance priority.  The device-resident default
flip is therefore lower value than the original r27 note implies; keep it for completeness.

**CLOSED 2026-10-07 (r26):** contributor PR #107 (@overdoingism) gives the allocation the
warn-once-and-fall-back treatment (the host buffers are always reserved, so the fallback needs no
extra memory) and flips F1 default-on for single-sequence DFlash, with the runtime
`GGML_LF_DFLASH_DEV=0` kill switch.  Folded into block 15; `WORKLOG.md` 2026-10-07 (r26).

### 28. Retune the RDNA4 dense mmvq rows-per-block table for `q5_1` / `iq4_xs` on ROCm 7.14

**Opened 2026-09-28** while integrating PR #57 (now in `v16-84e76d8a2-r21`, blocks 10+13).  The per-type
`calc_rows_per_block_weight` table was swept by the author on **ROCm 10.0**; on the maintainer's **ROCm
7.14** build it transfers for the main types (q3_K +7..+24 %, q4_K +8..+38 %, q6_K +3..+27 %, q2_K +14 %
at n=1) but carries two small dips: **`q5_1` -3..-4 %** and **`iq4_xs` -3..-4 %** at n=2/3 -- both
**+10..+12 %** at n=8, and the common MTP/DFlash verify widths are 4 and 8.  Kept as tuned (the aggregate
band is a large net win); a per-type retune (e.g. 1 row for these two at 2..3 columns) needs its own
verify-width A/B.  **Working home: `wip/mmvq-rpb-retune/`.**  Record:
`archive/work/mmvq-verify-rows/VERIFICATION-r21.md`, `WORKLOG.md` 2026-09-28 (r21, PR #57).

### 27. Extend the RDNA4 GQA-6 FA band's 64-wide K/V row to the 2-byte (f16/bf16) arm

**Opened 2026-09-28** while integrating PR #62 (now in `v16-84e76d8a2-r21`, block 15).  The new band-only
MMA config row (`fattn-mma-f16.cuh`: `is_rdna4 && ncols2 == 8 && DKQ == DV == 256 && ncols == 32`) matches
only the **native-quantized** arm (`ncols1 = 4`).  The 2-byte f16/bf16 arm uses `ncols1 = 2`
(`ggml_cuda_fattn_band_wmma_ncols1`) -> `ncols = 16`, so it keeps the old 128-half2 row.  The PR README's
"f16 3-4 %" is therefore **misattributed** (measured f16 KV `tg128 @ d16384` 27.50 -> 27.57, noise).  A
64-wide row for the 2-byte arm is worth a sweep: the arm has no dequantisation to hide the unused
columns, and r6 made native bf16 default-on so the arm is live.  Record:
`archive/work/rdna4-fa-band/VERIFICATION-r21.md`, `WORKLOG.md` 2026-09-28 (r21, PR #62).

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.  Upstream
  ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the block-13
  rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480
  (4B, q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute +
  −799 MiB host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.

- **Mixed K/V cache types fall off the GPU attention path.**  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`)
  gives `graph splits = 18`, a ~1.5 GiB host compute buffer and pp2048 7924 → 640–1049 t/s on the 4B.
  Maintainer policy (2026-09-11): **reject differing K/V types** — every mixed pair is 1.7–3.6× slower
  and never smaller; upstream already enforces same-K/V for DeepSeek V4 (#25871).  **Decided and
  implemented 2026-09-11 (12): hard-rejected at context creation** — `params.type_k != params.type_v`
  now fails `llama_init_from_model` with a message naming both types and telling the user to set
  `--cache-type-v` to match (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `--cache-type-k`/`-v` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter (`ggml-backend-meta.cpp:1177`)
  because its 2 KV heads are fewer than the 3 devices (one device gets a zero-extent share).  Works on
  1/2 GPUs and on 3 GPUs with `-sm layer`; maintainer's call: no fix.  Every other model is unaffected
  (a future block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).

## Parked (not planned now)
- **Restore the block-13 RDNA3_5 single-token fusion perf (item 16).**  The ~0.9 % `tg128` the purity
  skip costs; the proposed "pin `nwarps`/`rps`/item-split" fix is **invalid** (the two arms are already
  launch-identical).  Live candidates: codegen (`has_fusion` register pressure / FMA contraction) and the
  Q8_1 cache.  Low priority — the skip is the accepted purity trade.  `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9.
- **`-Wshadow` cleanup for `src/` (item 15).**  Audited: 128 warnings / 27 files, 46 in the risky
  "shadows a local variable" class.  Wants a dedicated cleanup commit (~20 upstream files) to avoid
  colliding on every re-base.  `archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`.
- **MoE topk fusion adoption (item 5(a)).**  ~0.5 % prefill for a ~188 MB arena cost and a numerics fork
  (fused top1 logit 18.424 -> 18.690 vs the unfused reference), so it needs a quality gate before it can
  be trusted.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`.
- **`ssm_alpha` + `ssm_beta` single-walk fusion (item 5(b)).**  ~0.3-0.6 % prefill, blocked by the graph
  expansion order (the two MMs are non-adjacent) -> needs a graph restructure or load-time stacked
  weights.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`.

## Where the current lists live

- Remaining gfx1151 work + the 2026-09-12 TODO audit: `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- QSA sparse-regime width purity (items 4/7 disposition): `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- Environment, instruments, reference hashes and the landing procedure for KV/FA work:
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its items 1/5
  and F3 are **superseded** — see `WORKLOG.md`).
- Delivery verification contract + dated records: `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- Purity/invariant analysis and the instrument rules: `GREEDY-PURITY.md`.
- Benchmarks + gates: `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- Memory campaign (wins, V3/V4 plans, Block 15 record — now promoted to `patches/0015`): `archive/work/block-15-campaign-wins/HANDOVER.md`,
  `README.md`, `BETA-TESTING.md`; upstream PR candidates: `upstream/README.md`.

## Parked: finish the meta segmented expert upload (host-resident experts under `-sm tensor`)

`llm_arch_supports_sm_tensor()` no longer rejects `LLM_ARCH_GEMMA4` outright (r9, issue #99): an
all-resident or `-ngl`-offloaded gemma4 splits under `-sm tensor`, and only a host-resident expert
table (`-ncmoe`/`-cmoe`, detected in `llama_model_create`) and the MTP head (`gemma4-assistant`) are
rejected.  The remaining work is the segmented per-ubatch upload for a host-resident expert table:
gemma4's fused expert tensor (`ffn_gate_up_exps`) has a segmented split layout (`n_segments`/`nr`),
and `ggml_backend_meta_set_tensor_async` only handles a contiguous slice, so the MMQ tail pad that
`copy_experts` appends makes the range non-row-aligned.

A partial fix exists but is **not** landed (it hung non-deterministically, one GPU spinning): port
`ggml_backend_meta_buffer_set_tensor`'s segmented handling into the async setter, copy the
row-aligned part through the segments and the <=512-byte pad as a flat per-device range, and use
per-row 1-D async sets instead of the 2-D pageable `hipMemcpy2DAsync` (which faults on this shape).
Give it a deterministic repro (e.g. `compute-sanitizer`, or the `GGML_CUDA_SPLICE_GATHER` /
`GGML_META_PINHOST` force-modes) before retrying.  All of it is in
`archive/work/moe-expert-cache/item3-findings-session20.md`; the issue-#99 gate change is in
`archive/work/issue-99/README.md`.

## Merge hygiene — finer block-06 dispatch split (r50 follow-up)

**SUPERSEDED by `archive/work/patch-block-reorg/` (seeded 2026-10-12).**  The r50 finer split is now one part of a
larger, fundamental re-organisation of the whole block taxonomy (the r50 lateral move into block 15 is
reversed by the target map in `MIGRATION-MAP.md`).  The spec below is retained as the hunk-level detail
for the `ggml_cuda_try_fuse` / `ggml-backend-meta.cpp` moves; the live campaign and the decisions it
needs are in that directory (`OPEN-QUESTIONS.md`).

The r50 per-block sweep is **16/16**, but block 06's fusion dispatch (`ggml_cuda_try_fuse` additions + the
six `ggml_cuda_match_*` matchers for hc-mix / combine-norm / idx-relu / cache-blocks) now lives in
**block 15** as a lateral move, so block 15 grew ~+1.4k lines and the features are not dispatched at their
owning blocks (08/13/14).  The finer split needs the **block-14-era** matcher bodies (the block-15 state
references block-15 struct members such as `out_xn_bf16` / `store_xn_f32`), and the
`ggml-backend-meta.cpp` move is blocked on the block-15 `ggml_backend_device_i` / `ggml_backend_i`
fields.  Spec: `archive/work/r50-merge-hygiene/REDISTRIBUTION.md`.
