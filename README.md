# llama-cpp-rdna-boosts

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

**Everything this delivery adds is default-on — you should not need to set any environment variable.**
[`ENVIRONMENT.md`](ENVIRONMENT.md) documents every variable the set adds or repoints: its default, and
whether it is a kill-switch (default-on, set to disable), an opt-in, a tuning value, or a diagnostic.

It ships as **12 patches** (block 00 + blocks 01-11) for a clean llama.cpp checkout at the fork point
**`097cf33d9`** (upstream master).  Each block is a self-contained `git am` commit, so you can apply the
whole set or pick the ones you want.  The current release is **`v12-097cf33d9-r75`**.

r75 carries the r73/r74 work plus a release-blocker fix.  r74 sizes the MoE expert-cache arena at the
**warmup** (not the first real decode) and stops the prefill seed at the band a wide prefill immediately
reclaims (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` is the kill switch); r73 unifies the heat-map sidecar to one
`<base>.moe-heatmap` per model.  r75 fixes a placement bug those two exposed: the wide (prefill) compute
view was placed from the live H2D-ring *request* instead of the *carved* region, so on a host-expert model
with MTP the wide buffer aliased the staging ring and the first prefill faulted (`k_get_rows_float` on
Q4_K_M, `rope_multi` on Q8_0); `work_alloc`/`seed_floor` now follow the carve, and the MTP draft context
(fully resident, never stages a table) no longer sizes the ring at all.  Verified: 35B-A3B Q4_K_M/Q8_0
`-ncmoe 40` + `draft-mtp` run clean and are byte-identical to r73; the GSQ field arena is unchanged
(15666.6 MiB at load, 1 GPU); dense coherence, the prefill-logit gate (0.000707 / 98.755 %) and the
adaptive-MTP gate are green (`WORKLOG.md` 261016.010).  **r73 and r74 were internal pre-release betas and
were never published** — r74 did not pass validation (the alias above) and is superseded by r75.

Before it, the r68-r72 amendments (container work, `-ncmoe` host placement, the ring-region guarded prefill
drop, the ring auto-size, the H2D-ring largest-slice fix) are recorded in [`patches/README.md`](patches/README.md)
and `WORKLOG.md`.

Before it, r67 fixes the host-resident-expert regressions the r63..r66 window introduced.  It restores the
MUL_MAT_ID router helper's batched template path (the r66 duplicate-ids fix had made it ~20 % slower on
prefill while remaining duplicate-correct), so the 35B-A3B fully-resident prefill is back to **8243**
t/s (r65 8357); it sizes the expert cache from the **total** registered host bytes so the uneven
`-sm tensor` split no longer leaves the arena unsized (GSQ decode 14.8 -> 62 t/s); it bounces any
pageable source on the weight upload so `--load-mode none` + `-sm tensor` no longer faults; and it pins
the narrow work floor before the first prefill and places a prefill ubatch **above** the ring, so the
carved H2D staging ring actually arms (GSQ `-ncmoe 48` prefill 645 -> 687 t/s).  All three DoD models
load and run with 0 faults in both `--host-experts pinned` and `pool`, the prefill ring is active on all
three, and the standing gates are green (`WORKLOG.md` 261015.005).  One residual is parked for later:
the IQ4_NL field (MTP) config is very slow in `pool` mode (TODO #65).

Before it, r66 folds the upstream MUL_MAT_ID duplicate-ids fix into block 00, caches the hot-path
environment reads in their owner blocks, and completes the -Wshadow sweep.

Before it, r65 merges five contributor PRs from `@briansp2020`, each folded into the block that owns its concern
(blocks 06/07/08/09/10, no new block): a pinned output buffer under
`-sm tensor` (`LLAMA_OUTPUT_PINNED=0`), a second scheduler for the MTP draft context's graphs with
outputs (`LLAMA_DRAFT_SCHED2=0`), fewer instructions in the Q4_K / Q6_K mmvq dot products, and two
gfx1100 ports (the Q4_K-only 128x256 MMB dense tile, and 4 rows per block for a Q4_K verify batch).
Measured on gfx1201: +5.3 % MTP decode on Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (2 GPUs, `-sm tensor`,
`-n 3000`) from the output/scheduler pair, and +0.8 % on the dense 27B; outputs and acceptance are
unchanged, the prefill-logit gate is 0.000707 / 98.755 %, and `validate-set.sh` is green.  The two
gfx1100 changes are no-ops on gfx1201.  Records: `WORKLOG.md` 261015.002, `archive/work/{output-pinned,
draft-sched2,rdna3-mmvq-vecdot-alu,rdna3-mmb-tile-spill,rdna3-mmvq-q4k-rpb}/`.

r64 rounds out the r63 host-expert upload work: block 09 now accepts an arbitrary segmented Meta
write (head partial row + full rows + tail partial row), and block 10 owns the CUDA splice-gather
ring in the context so it is torn down with its streams.  It also drops the qwen4exp MTP
`ubatch.token` assert, so mtmd image requests work under `--spec-type draft-mtp` (PR #160).  Gates:
`gate-prefill-logits.sh` 0.000707 / 98.755 %, `gate-qwen4exp-quant-coherence.sh` PASS, same-seed
coherence unchanged on 4B/35B/Gemma4, adaptive-MTP `none == n3` byte-identical, 35B `-ncmoe 20` hang
loop 10/10 clean (`-lm auto` and `-lm none`).

r63 flips the `-ncmoe` host placement to the **last N** trunk layers (the tail is the colder, more
cache-friendly set and has preceding resident compute to overlap the expert upload with; measured
`pp4096` 4264 vs 4120 and `tg500` 87.5 vs 82.6 on 35B-A3B).  `LLAMA_HOST_EXPERTS_FROM_END=0` restores
the old first-N placement.

The dated release history — every block amendment, issue fix and measurement, r1 through the
current release — is in [`WORKLOG.md`](WORKLOG.md), newest first.  `release.json` is the single
source of truth for the current release (fork point, canonical tip/tree, block count, per-artifact
hashes).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts (12 commits)
```

- One-line summary of each block: [The 12 blocks](#the-12-blocks-v20)
- The folded `mmb`/QSA campaign: [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery)
- Apply details, env knobs, server config: [`patches/README.md`](patches/README.md)
- **Running a model bigger than your VRAM on one card** (`-ncmoe` + the MoE expert cache, with the
  measured config tables and the measurement method): [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md)
- Current status and validation: [`release.json`](release.json) (source of truth) and [`MANIFESTS.md`](MANIFESTS.md)

## Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v12-<fork-point>-r<N>`, e.g. **`v12-097cf33d9-r59`**; the `v16-*`
tags are the historical 16-block set.  `r1` is the
re-base of the set onto that fork point and every later `rN` is a dated change recorded in
[`WORKLOG.md`](WORKLOG.md), newest first).

## Supported architectures

The set targets the **RDNA3 / RDNA3.5 / RDNA4** GPU families, with RDNA2 as best-effort:

| family | arches | example parts | status |
|--------|--------|---------------|--------|
| RDNA 3 | `gfx1100` | RX 7900 XTX/XT, RX 7800 XT, ... | supported |
| RDNA 3.5 | `gfx1150`/`gfx1151` | Strix Point / Strix Halo APUs | supported |
| RDNA 4 | `gfx1200`/`gfx1201` | RX 9060 XT; RX 9070 / 9070 XT | supported |
| RDNA 2 | `gfx1030` | RX 6800/6800 XT/6900 XT, Radeon Pro V620 | best-effort, **not validated** |

> **Strix Point (`gfx1150`) is in the target set as of 2026-10-05.**  Images
> published before then (for example the `server-rocm-10.0` build at `2dbece4`)
> carry no `gfx1150` code object and fail on the first kernel launch (`ROCm
> error: device kernel image is invalid`).  For those, build from source with
> `gfx1150` in the target list, or set `HSA_OVERRIDE_GFX_VERSION=11.5.1` at
> runtime to select the `gfx1151` code objects.  See [Strix Point
> (gfx1150)](CONTAINERS.md#strix-point-gfx1150) for the validation and both
> paths.

> **RDNA2 (`gfx1030`) is best-effort and NOT validated on hardware.**  It is
> compiled into the image so RDNA2 owners pick up the generic, runtime-dispatched
> boosts instead of a first-kernel `ROCm error: invalid kernel file` (issue
> #158), and the RDNA3/4-only paths fall back to the upstream non-WMMA kernels at
> runtime.  There are no performance claims and no purity gates for it, and
> `-fa on` may hit upstream RDNA2 tile-kernel defects — use `-fa off` if a
> long-context prompt aborts.  See [RDNA2
> (gfx1030)](CONTAINERS.md#rdna2-gfx1030--best-effort-not-validated-on-hardware).

**RDNA4 (gfx120x) sees the most benefit** — the WMMA flash-attn path, the
chunked-GDN kernel, the k-quant VDR boosts and block 12's internal
all-reduce were all first built and validated there. As much of that work
as possible is back-ported to the RDNA3/3.5 families instead of being
gated off:

- block 02's **chunked gated-delta-net** bf16/WMMA prefill ships as two
  arch-segregated kernels: a dedicated first-gen WMMA port for gfx11
  (`gated_delta_net_chunked_bf16_gfx11.cu`) next to the RDNA4 kernel;
- block 04's **WMMA flash-attn** is *not* RDNA4-only despite the block
  name — RDNA3.0 runs it with the same 576-head limit as RDNA4, RDNA3.5
  with a tuned 320-head limit;
- block 10 adds a **dedicated RDNA3.5 mmvq parameter table** (previously
  folded into the RDNA2 fallback) on top of the RDNA4 k-quant boosts.

Arch selection is **runtime** everywhere in the set (device `cc` / `gcnArchName`),
so a multi-arch build such as `GPU_TARGETS="gfx1030;gfx1100;gfx1151;gfx1201"`
yields one binary that picks the right path on whichever of these it runs on.
The WMMA kernel *bodies* are compile-gated by `RDNA3`/`RDNA4`; on RDNA2 the
non-WMMA arms compile and the runtime predicates keep the WMMA kernels from
being launched. The one genuine exception is **block 12** — its internal all-reduce is RDNA4-only
(gfx1200/gfx1201) and falls back to RCCL elsewhere (see
`patches/README.md` for the gate and env knobs).  Block 13's fused
MoE MMQ gate now covers RDNA4 + RDNA3_5 + RDNA3_0 (gfx1151 validated
2026-09-05, gfx1100 validated 2026-09-05 — see
[`WORKLOG.md`](WORKLOG.md)).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # purity rulebook: index, invariants, per-finding claims (read before shipping)
│                          #   narratives/evidence for the closed cases: archive/docs/GREEDY-PURITY-FINDINGS.md
├── WORKLOG.md             # dated delivery records (newest first; README points here)
├── rdna-boosts-all.patch  # convenience: the entire 12-patch net as ONE patch
├── patches/               # the delivery set: 0000-0015
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift)
│   ├── gate-prefill-logits.sh  # prefill-logit KLD release gate (issue #113)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── prompts/               # versioned, hash-stable test prompts (sha256-recorded; never edited in place)
├── wiki/                  # source for the GitHub wiki (Home, MTP & Adaptive MTP, Quick Reference); see wiki/README.md
├── wip/                   # ACTIVE exploration docs / handoffs (currently only: nwarps/)
├── beta/                  # promotion-candidate patches pending a beta test (NOT part of the delivery)
├── upstream/              # upstream-PR candidates (UPSTREAM-PR-*.md + .patch) + their index
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 12 blocks (V2.0)

`v12-097cf33d9-r52` (net tree `8b177a7b…`).  Each row is one `git am` commit.

| patch | subject | concern |
|-------|---------|---------|
| `0000` | **foundation** — band-uniformity, Meta-wrapper, allocator fixes | old-00 + old-05 + old-07 + old-09a + the multi-ubatch logits readback + the `ggml-alloc` view release |
| `0001` | **hybrid HIP all-reduce** (RDNA4-gated) | the old block 12; + no per-AR `ev.ker` and the opt-in P2P-large path (r52) |
| `0002` | **adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) | old-01; + opt-in MTP prompt-tail ingest and the last-row prompt copy (r52) |
| `0003` | **chunked gated-delta-net prefill** (bf16/WMMA GDN) | old-02 |
| `0004` | **BF16 KV cache + native-BF16 flash attention** | old-03 |
| `0005` | **flash-attention band / native quantized K-V / derived kq mask** | old-04 FA + the old-15 FA work |
| `0006` | **MMB quantized prefill GEMM** | old-08 MMB |
| `0007` | **k-quant boosts** (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) | old-10 + old-04 MMQ + old-15 Q1_0/grid/PR57 |
| `0008` | **fused kernels** (conv1d/RMS/top-k MoE/verify fusions + fused MoE gate+up+GLU) | old-08 fusions + old-13 fused MoE + old-15 PR63/114; + `topk_moe` sorted-key argmax (r52) |
| `0009` | **host-resident MoE** (iface, scheduler, Meta, movable-boundary slab, cache engine) | old-06 MoE + old-09b + old-11 + old-13 engine + old-15 hooks/arena |
| `0010` | **qwen4exp** (model, hyper-connections, lazy reader, QSA/indexer) | old-14 + old-15 HC16/policy; + short-K MoE rows-per-block (r52) |
| `0011` | **campaign remainder** (CPU wins, backend iface fills, DFlash, FA build grouping, mmap reclaim) | old-14 QSA + old-15 W1-W3 + old-06/15 mmap reclaim + old-16 |

Phase-2 additionally moved each concern into a self-contained file: the `ggml_cuda_try_fuse` arms, the
MMVQ vec-dot/VDR selectors + MoE kernel (`mmvq-vecdot.cuh`, `mmvq-moe.cu`), the FA native-KV/band
predicates (`fattn-rdna.cuh`), and the four `ggml-cuda.cu` TUs (`ggml-cuda-try-fuse.cu`,
`ggml-cuda-slab.cu`, `ggml-cuda-moe-cache-hooks.cu`, `ggml-cuda-arena.cu`; `ggml-cuda.cu` 9819 → 6370
lines).

<details>
<summary>Historical: the pre-V2.0 16-block notes (provenance/edit history only)</summary>

### The 16 blocks

| patch | what |
|-------|------|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25) + Vulkan masked-V/freed-cell fixes (dead columns never read V). The base every later block applies on top of. |
| `0001` | adaptive MTP draft depth (`--spec-type draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn (+ the HIP masked-V/freed-cell fixes since 2026-09-10) |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf (WMMA path also runs on RDNA3.0/3.5, tuned head limits) |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | **general system-operations bucket** — the delivery's **catch-all** for changes that fit no other block: the FA instance build-time work, `--fit` under `-sm tensor`, the host-buffer input layer, the tiny-CPU-split single-thread fix and (r12) the op-offload **H2D staging ring** + tensor-split op-offload.  Named for what it is since r12; the original host-buffer revert content is long gone (upstream reverted #24233 in #28604).  Amended r13 (issue #52): the tiny-CPU-split heuristic now counts what a graph *reads* — not only its node outputs — so a CPU-offloaded FFN chunk keeps the full thread pool |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04; amended 2026-09-07 with the mul_mat+add through-view shape guard, PR #15). **Now folds the `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning, and the GDN/PLE conv1d + narrow-row RMS-norm prefill fusions** (absorbed from the former `archive/work/mmb-general` campaign). |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4 + RDNA3_5 + RDNA3_0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes). The masked-V/freed-cell fixes it once carried now live in blocks 00 (Vulkan) and 03 (HIP). |
| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 + **V5** native bf16 K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert in `archive/work/block-15-campaign-wins/ab/`).  ~3.4 GiB/GPU + ~1.2 GiB host saved on qwen4exp, ~800 MiB/GPU + ~800 MiB host on dense models, at ~1.3 % prefill / ~0.3 % decode. **Now also folds the qwen4exp/QSA/HC/indexer campaign** (`qsa3` packed-block WMMA attention, fused indexer top-k + prefill score fusions, HC16 native-BF16 producers, `hc_gate_mix`, sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the host-buffer/CPU/meta fixes) — the former `archive/work/mmb-general` work.  Also carries the r14-r21 **MoE expert-cache** work: auto sizing, host-pinned expert buffers, the op-offload H2D staging path, the compute-buffer slack, the fail-soft arena yield, the per-layer uniform slot re-size (`t.slots` fix), the `MTP_DRAFT_N_UBATCH` cap, and the cli-only wide-prefill drop.  **r22 adds the movable-boundary slab allocator** (ONE slab per device, reserved and mapped once, split by a movable boundary: compute buffer below, MoE arena above, HIP never called at runtime) which SUPERSEDES the per-allocation VMM pool (removed) and the fail-soft arena yield, sizes the arena on a decode-band pass, right-sizes the compute reserve to the observed workload, and makes the wide-prefill drop default-on for servers too — the TODO #42 abort is gone. **r27 folds PR #114's four qwen4exp decode fusions:** the latency-scheduled BF16 `hc_mix` up/collapse (`GGML_HC_UP_V2=0`), `HC_COMBINE` fused into that norm (`GGML_CUDA_FUSE_HC_COMBINE_MIX=0`), the shared-expert `sigmoid`-`mul`-`add` gate (`GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0`), and the GDN `beta` sigmoid folded into the sequential kernel (`GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0`); +2.4 % decode on qwen4exp, byte-identical text. |

> **The delivery is `0000`–`0015` (16 patches).**  Block `0016` was added in
> `v16-a55e952b8-r45` (the contributor decode/latency PRs) and **folded back into the 0000-0015 set in
> `v16-a55e952b8-r46`**: host-side/system work into `0006`, small-message all-reduce into `0012`, the
> decode/latency remainder (and issue #133) into `0015`.  (`v16-a55e952b8-r16` had briefly shipped a
> *different* `0016`/`0017` pair — op-offload staging redirect and per-device tensor-split guard — which
> r17 folded into `0006`/`0013`/`0015`; that old pair is unrelated.)

> **Block 15 (attention-memory wins) is part of the delivery since
> 2026-09-12** (`patches/0015`, promoted from
> `archive/work/block-15-campaign-wins/`; a fresh set is now **16 patches**,
> blocks 00-15).

> **Greedy-purity note (read before shipping):** on the K-split decode
> paths, block 10 (`0010`) is the only patch that changes decode numerics on
> ANY architecture — its VDR kernels reorder the fp32 reduction. Compute
> outputs are not bit-identical to a build without it (max logit diff 0.184
> vs 0.203 for flash-attn on/off; greedy streams are deterministic within a
> build but can flip across configs). This is a different rounding path, not
> a correctness change. If you require 100% greedy purity across builds, do
> not install `0010-…k-quant-boosts…patch` — it is one line to drop from
> `scripts/apply-all.sh`. Full discussion:
> [`GREEDY-PURITY.md`](GREEDY-PURITY.md). **Block-13 caveat (2026-09-02):**
> block 13 rewrites the small-batch mmvq decode kernel and is a second
> decode-numerics source on the rows that run it (short-K K<4096 ncols==1
> rows, MoE projections; ncols 2..8 and long-K rows were restored to the
> pre-block-13 K-split kernel by the 2026-09-02 fix). Excluding block 10 no
> longer reproduces stock bits exactly on those rows — see
> GREEDY-PURITY.md §9.

</details>

## Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 12/12 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0011  (one commit per block on a fresh `rdna-boosts` branch)

# 3. build + verify (trim -DGPU_TARGETS to your GPU arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# coherence gate (same-seed output must match a known-good build):
./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p "The capital of France is" \
  -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

> **Speed up rebuilds with ccache.**  The set's flash-attention template instances are the build's
> critical path, and their native-KV loader arms are deliberately force-inlined — the optimiser's
> cross-inlining is what makes them fast at runtime *and* slow to compile.  With `ccache` on PATH,
> a wiped rebuild of *unchanged* sources is a full cache hit: measured **282 s -> 4.2 s** on a
> 16-core gfx1201 box, **321.8 -> 5.2 s** on gfx1151 and **383.9 -> 4.8 s** on gfx1100 (657/657
> compile steps hit on each).  Add
> `-DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
> (the launcher form works with ROCm clang HIP device compilation; ccache 4.12.3 tested, on CMake
> 4.3).  ccache
> replays the compiler's own objects, so the cached build is the same code — verified with
> same-seed greedy text (identical hash on every host before and after enabling it),
> `llama-bench` (within noise) and `test-backend-ops`.  Any header change (e.g. `fattn-mma-f16.cuh`)
> invalidates its dependents, i.e. the whole FA group.
>
> **Do not use `git apply` on the concatenated 1-11 series** — it silently
> drops hunks (30 files / 2483 lines vs the correct 35 / 6094, verified
> 2026-08-29). `git am` (or `scripts/apply-all.sh`) is the required flow.

### Manual equivalent

```bash
git am patches/000[1-9]-*.patch patches/001[0-1]-*.patch   # blocks 01-11
```

### The `mmb` campaign is in the delivery

On the **`beta-integration`** branch the `mmb` (bf16-WMMA dequant weight GEMM) / QSA / indexer
campaign — the former 28-patch `archive/work/mmb-general/` set — is **folded directly into the 12 delivery
blocks**, so the normal workflow above is all there is to apply.  There is **no separate beta layer
any more**:

* **block 08** absorbs the `mmb` core, the RDNA4 fragment port and per-arch tuning, and the
  GDN/PLE conv1d + narrow-row RMS-norm prefill fusions;
* **block 06** (the catch-all system-operations bucket) absorbs the host-buffer input layer and the tiny-CPU-split
  single-thread fix;
* **blocks 13/14** absorb the `mmb` fusion stand-downs and the extended MMVQ routed band;
* **block 15** absorbs `qsa3`, the fused indexer top-k + prefill score fusions, HC16, `hc_gate_mix`,
  sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the meta/CPU backend fixes.

Applying the 12 patches to the fork point in [`release.json`](release.json) therefore reproduces the
**full campaign tree** in one pass:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9          # the base recorded in release.json
bash <path-to-this-repo>/scripts/apply-all.sh .   # 12/12 strict, tree checked against release.json
```

The per-patch fold mapping and validation record are in
[`archive/work/beta-integration/integration.md`](archive/work/beta-integration/integration.md).  Much of the MMB
kernel work is heavily adapted from **[pwilkin](https://github.com/pwilkin)**'s
[`strix-halo` fork](https://github.com/pwilkin/llama.cpp/commits/strix-halo/), with thanks.

> **Historical record only.**  [`archive/work/mmb-general/`](archive/work/mmb-general/) (the 28 patch files, the
> `mmb-general.patch`, `BETA-TESTING.md`, the gfx1201/gfx1100 records) is kept as the campaign's
> verification record; its patches are **no longer applied separately** and the `apply-beta.sh`
> helper has been **removed** (the delivery itself now contains the campaign).  Its gfx1151
> beta-window re-validation was **GREEN**
> (2026-09-25 — the four gates + the recurrent rollback; see `BETA-TESTING.md` §8), which is what
> the fold relies on.

## Recommended configuration — adaptive MTP + `ngram-mod`

The best general-purpose speculative-decoding configuration measured on this delivery combines the
**adaptive MTP controller** with the draftless **`ngram-mod`** speculator:

```bash
--spec-type draft-mtp-adaptive,ngram-mod \
  --spec-ngram-mod-n-match 45 \
  --spec-draft-n-max 9 --spec-draft-n-start 9
```

`ngram-mod` supplies the long verbatim-recall drafts the MTP head cannot match, while the adaptive
controller keeps the MTP depth right for everything else.  Measured against plain `draft-mtp-adaptive`
on the Q8_0 2-GPU reference cell (`-n 3000`, a 4-prompts-per-axis corpus): **recall +67.5 %,
code +0.8 %, prose +0.5 %, reasoning −1.9 %, overall +13.6 %**; on the dense 1-card cell it is
code/prose-neutral with the same recall win.  The small reasoning cost is the price of the deeper
`n-max 9`; a workload with no verbatim recall is marginally better served by plain
`draft-mtp-adaptive`.

> **Do not combine `ngram-mod` with host-resident MoE experts.**  When MoE experts are partially or
> fully host-resident (`-ncmoe`/`-cmoe`, i.e. the MoE expert cache or the pinned `ROCm_Host` path is
> active), adding `ngram-mod` to `draft-mtp-adaptive` **roughly halves decode throughput**.  Dropping
> `ngram-mod` and keeping plain `--spec-type draft-mtp-adaptive` restores the expected decode rate.
> The adaptive controller is itself a good fallback for replayed/verbatim content: it drafts deeper
> when the content is predictable, so it handles replay quickly (not as fast as `ngram-mod`, but far
> ahead of a halved decode).  On a host-expert config, plain `--spec-type draft-mtp-adaptive` is
> therefore the recommended pick.  The cause of the interaction is not yet understood; it is tracked
> in [`wip/ngram-mod-host-experts/`](wip/ngram-mod-host-experts/README.md).  This applies only to
> host-resident experts: on a fully device-resident model the combo above is still the best general
> pick.

Guidance:

* **Cap.**  `9` is the general-purpose pick and the optimum on a **single card** (a cap of `6` costs
  11-14 % on code there).  On a **multi-card tensor split** `6-7` is ~1 % better.  A lower cap saves
  only a small verify-batch scratch, not the model/KV memory.
* **`n_match`.**  Keep it `>= 40` **and an integer multiple of the cap** — `9/45`, `8/48`, `6/42`.
  Too short (`nm24`) makes ngram fire on incidental code repeats and lose code throughput; a
  non-multiple (e.g. `nm42` at cap 12) degrades acceptance.
* **Host-resident experts.**  On a config with host-resident MoE experts (`-ncmoe`/`-cmoe`), skip
  `ngram-mod` entirely (see the caveat above): the combination roughly halves decode, and plain
  `--spec-type draft-mtp-adaptive` is the better pick.
* The combo changes the draft strategy, so it is **opt-in** — the controller default stays plain
  `--spec-type draft-mtp-adaptive`.

Full derivation (the 4×4 corpus, all four cells, and the rejected controller alternatives):
**[`archive/work/mtp-journey-2026-09-17/SUMMARY.md`](archive/work/mtp-journey-2026-09-17/SUMMARY.md)** (narrative in its
[`README.md`](archive/work/mtp-journey-2026-09-17/README.md)); the dated controller records are in
[`benchmarks/`](benchmarks/README.md), newest `2026-09-15-adaptive-mtp-tuning.md`.

## VRAM vs prefill — the derived KQ mask (`LLAMA_KQ_MASK_DERIVED`)

**On by default, deliberately.**  The delivery derives the attention mask inside the FA kernel from
compact per-cell state instead of materialising the `n_kv x n_q` f16 mask.  That removes
`n_ubatch x n_ctx x 2` bytes of compute-buffer VRAM **plus the same again on the host** — measured on
a 9B at `-c 98304` / ub 512: **184.0 -> 88.4 MiB** device and **112.0 -> 16.4 MiB** host.  The saving
scales linearly with the ubatch, which is the point: a deep-context **MoE** or **qwen4exp /
Qwen3.8-Flash-Next** workload wants a large ubatch, and that is exactly the configuration where the
mask is biggest (~800 MiB/GPU at ub 2048 / 196k) and where the VRAM the feature frees is the
difference between fitting the context and not.

The cost is **prefill only** — decode is untouched, because the derived path only fires for batches
larger than 8 tokens (speculative verify keeps the packed mask, so `n_max <= 7` stays bit-identical).
Measured PP512, mask on vs off, `-r 3` ("+" = the mask helps):

| config | d0 | 32k | 64k | 98k |
|---|---|---|---|---|
| gfx1201 9B dense 1 GPU | — | +1.9 % | — | **+3.5 %** |
| gfx1201 27B 2 GPU **tensor** | −1.3 % | −0.4 % | **+1.3 %** | **+1.7 %** |
| gfx1201 27B 2 GPU layer | — | — | — | −1.6 % |
| gfx1201 27B 3 GPU tensor | −3.4 % | −0.6 % | — | **+2.0 %** |
| gfx1151 9B dense 1 GPU | +0.4 % | −0.2 % | −0.9 % | −1.8 % |
| gfx1151 35B-A3B MoE 1 GPU | −0.3 % | −0.3 % | −0.8 % | −1.6 % |
| gfx1100 9B dense 1 GPU | −0.4 % | −0.5 % | −0.4 % | **−0.2 %** |

The tensor-split shape is the one to understand: the packed mask grows with `n_kv`, so on a
**tensor split at shallow depth the mask is a small loss (−1.3 % at d0, crossing zero near 48k) and
becomes a win by 64k+**; on the maintainer's 3-GPU tensor serving setup it is a win at depth.  gfx1100
and gfx1151 pay a depth-growing ~1–2 % (they did not recover as much from the r7 kernel fix as
gfx1201 — the iGPU shares host bandwidth and the 7900 XTX has more of its own).  gfx1100 on a
dual-card **`-sm tensor`** split is the one cell we still cannot measure here (only a single 7900 XTX
is available); a community report on 2x RX 7900 XTX is pending.

**Turning it off.**  `LLAMA_KQ_MASK_DERIVED=0` restores the packed mask (upstream's behaviour).
Worth doing if you are on **gfx1100/gfx1151** and want the last ~1–2 % of deep prefill, or on a
**tensor split at shallow depth** and prefill latency matters more than the VRAM.  For a deep-context
MoE / qwen4exp workload the default is the right side of the trade.

**Not a correctness knob:** same-seed output is byte-identical either way (the derived mask produces
the same values; only the memory layout and prefill cost differ).

**It works on both prefill kernels (r9).**  The derived mask is implemented by the **MMA** and the
**tile** flash-attention kernels, so it is no longer tied to the chooser picking MMA: a head above the
per-arch WMMA cap (RDNA4 576, RDNA3_5 320, RDNA3_0 256) used to lose the mask entirely, and that is
every **Gemma4** (head 512) on gfx1100/gfx1151 — the two arches that live on the tile kernel.  There
the mask is a *win*, not a tax (PP512, mask on vs off):

| config (tile kernel, natural selection) | cell | delta |
|---|---|---|
| gfx1100 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.2 %** / **+0.9 %** |
| gfx1151 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.6 %** / +0.6 % |
| gfx1201 Gemma4 E4B (tile forced) | @ d0 | **+2.3 %** |

and **decode pays nothing for it.**  Decode and the spec verify batch always take the tile kernel (the
chooser's WMMA branch requires `ne[1] > 8`), so the derived branch there is a cost on every arch; it is
hoisted out of the KV loop so the packed path's code generation is unchanged.  Measured r8 vs r9 at
that kernel: `tg128` deltas of **+0.01 %** (gfx1201 9B @ d16384), **+0.02 %** (gfx1100 9B @ d16384),
and flat on gfx1151 — the earlier per-iteration form cost −0.5..−0.8 % at depth before the hoist.

**The vec kernel has no derived arm**, but it is decode/verify-only (`n_tps <= 2`) while the derived
form only exists for prefill-shaped batches (`kq_mask_derivable()` rejects `n_tokens <= 8`), so it
cannot be selected for one.  If the launch log *does* print `derived kq mask flash attention not
supported, set to disabled` on a CUDA/HIP backend, the FA node did not reach the GPU at all — check
which kernel serves that head (the log adds a note pointing there).

One performance caveat with nothing to do with this knob: forcing the **tile** kernel for a head it
would not normally serve (a stale `GGML_CUDA_FA_WMMA_256=0`, a fixed env in the September qwen4exp
gates, is the usual cause) makes a head-256 model on gfx1201 **~3x slower at deep prefill** (9B,
`-d 98304`: 2104 -> 710 t/s).  That env is worth removing regardless of the derived mask.  One
exception worth knowing: **qwen4exp / Qwen3.8-Flash-Next gets its deep-context mask elision from the
QSA path's own derived visibility** (`GGML_QSA_DERIVED_VIS`, the code's "-800 MiB win"), which is
independent of this knob; `LLAMA_KQ_MASK_DERIVED` only serves that model's dense shortcut
(`n_kv <= 2051`), where the mask is tiny.

Full matrix, raw CSVs and the A/B harness: [`archive/work/kq-mask-derived-ab/`](archive/work/kq-mask-derived-ab/); the
2026-09-19 block-15 (r7) amendment in [`patches/README.md`](patches/README.md).

## Cross-start determinism on ROCm (issue #67)

Issue [#67](https://github.com/stew675/llama-cpp-rdna-boosts/issues/67) (from #58 item D, reported by
[@DanoPTT](https://github.com/DanoPTT) on Windows / ROCm 10 / gfx1201) tracked Q6_K greedy output that
intermittently swapped one near-tie across fresh `llama-server` starts while staying deterministic within
a start.  It is **closed as an external ROCm issue — nothing to fix in this patch set.**  Two separate
defects were in play:

* **A real bit-transparency bug in the fused rope path, fixed in r25.**  The address-gated
  `ROPE -> VIEW -> SET_ROWS` fusion did not reproduce the unfused chain bit-for-bit: clang contracted the
  fused `<float,__half>` and unfused `<float,float>` template instantiations differently, and one element
  of the prefill K-cache write landed on opposite sides of an f16 rounding boundary.  r25's
  `#pragma clang fp contract(off)` in `rope.cu` makes every instantiation round identically, verified at
  the logits level on Linux (`W=1` hash `60e77916673db071` with the fusion on and off).  See
  [`WORKLOG.md`](WORKLOG.md) for the r25 record.

* **The residual per-start flip is hipBLASLt solution selection in ROCm, not ggml** — and it reproduces
  on a stock upstream build with none of these patches.  The F32 `mul_mat` for `ssm_alpha`/`ssm_beta`
  (5120 -> 48) at 512 prefill columns misses the mmvf/mmf kernels, goes to `hipblasSgemm`, and rocBLAS
  routes it to hipBLASLt (`rocblaslt_matmul`, T,N, m=48 n=512 k=5120).  `HIPBLASLT_LOG_MASK=160` shows a
  per-process pick between a bit-identical reference solution pair (`140231`/`140232`) and a deviating
  pair (`140216`/`140217`); it is random from process to process, so layer 0's prefill output — and with
  it the whole run — differs at startup.  This is
  [ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126).

**Workaround: `ROCBLAS_USE_HIPBLASLT=0`.**  On the reporter's setup, 150 fresh starts per arm: default
**5/150** deviate, `ROCBLAS_USE_HIPBLASLT=0` **0/150** (bit-identical to the reference class), with no
measurable prefill/decode cost on the current build (the earlier text-hash runs measured higher flip
rates, up to ~14 %).  `ROCBLAS_DEFAULT_ATOMICS_MODE=0` does not help.  The selection is **prefill-only**
(decode at n=1 uses mmvf), so decode throughput is unaffected.

**Linux / ROCm 7.14 does not reproduce it** — the reporter's box (Windows / ROCm 10) is where the
per-process selection actually varies.  The r24 rope-fusion kill switches
(`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`, `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`) and r25's fix remain
shipped for bisection and for the real defect above.

## Concurrent serving with `--kv-unified` is not deterministic (issue #134)

Mixing `-np > 1` with `--kv-unified` does **not** give run-to-run reproducible greedy output: a
sequence's attention result depends on the physical **set** of KV cells it occupies in the shared
unified stream, and that set is a function of allocator/scheduling history (the flash-attention kernel
reduces in fixed tiles anchored to the buffer base).  Identical concurrent repetitions can differ, and a
concurrent run can differ from the same request run solo, at the rounding level (a greedy near-tie can
flip).  **Use `--no-kv-unified` when strict reproducibility is required.**  This is inherent to
upstream's unified-KV design — it reproduces on unmodified upstream `097cf33d9` — and not a defect
introduced by this patch set; the analysis and reproducers are preserved (parked) at
`wip/kv-cell-offset-nondeterminism/`.  Single-request and non-unified runs are unaffected by the
unified-KV effect, and the delivery's op-level purity invariants (`W = 1..8`, `plain == draft-mtp`,
`native == staging`) are scoped to a fixed batch and are unchanged.

**Separate from the unified-KV case (2026-10-16, issue #151 follow-up).**  The reporter (R9700 /
Windows) also sees a start-to-start greedy flip with HIP graphs on, and reports that `--no-kv-unified`
does not remove it.  We could **not** reproduce it on Linux/ROCm 7.14: fresh-process decode-logit
probes (W=1..8) are bit-identical across starts, graphs on and off, including on the reporter's 27B
model with q8_0 KV (`wip/kv-cell-offset-nondeterminism/FINDINGS-2026-10-16-revalidation.md`).  It is
most likely the known Windows-only per-start flip (the r25 `rope -> view -> set_rows` bit-transparency
fix, and the external hipBLASLt solution-selection issue, workaround `ROCBLAS_USE_HIPBLASLT=0`), both
prefill-driven and independent of graphs.  Tracked as TODO #67.

## When upstream master moves

The patches are static against the fork point in `release.json.base`. When upstream
drifts and hunks no longer apply, re-base the block commits (the fork checkout carries
them), regenerate the whole set with `scripts/make-patches.sh`, then refresh
`release.json` (`scripts/make-release.sh --base … --tip … --tree …`) and update the
current-state headers. The old `baseline/<sha>`-branch-per-upstream-range workflow
was retired when the delivery moved to the flat patch set on `main`.


## Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. Block 12's internal all-reduce is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for per-block verification and `BASELINE.md` for provenance.


## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18)
- https://github.com/eoprede
- https://github.com/overdoingism  (issue #45: the RDNA4 head-256 GQA-6 decode/verify flash-attention band, reported with the diagnosis, op-level data, the round-robin KV split idea and a working opt-in patch; the r4 block-15 band is built on that submission)
- https://github.com/pwilkin  (the `strix-halo` fork at https://github.com/pwilkin/llama.cpp/commits/strix-halo/, heavily adapted for the MMB bf16-WMMA dequant-weight GEMM work, now folded into the delivery — formerly `archive/work/mmb-general/`)
- https://github.com/tungel
- https://github.com/DanoPTT  (block-08 mul_mat+add through-view shape guard, PR #15; and issue #45 follow-up: the f16 verify-width diagnosis / the f16 + bf16 band coverage folded into block 15 in r5/r6, measured on their R9700)

I, and everyone else who benefits from this work, really appreciate you!

## Inspirational Works

While most of the work in this repository are original works of my own, there are
some significant portions, most notably around the prefill tuning, inspired by the
excellent work performed by the community of: https://github.com/halo-box/strix-llama.cpp

Thank you to all the maintainers of the Strix Halo Llama.cpp project

Of course none of this would be possible without the baseline that all of this rests
on, and that is the huge community over at https://github.com/ggml-org/llama.cpp

Many thanks to the llama.cpp team

## License

Same as llama.cpp (MIT).
