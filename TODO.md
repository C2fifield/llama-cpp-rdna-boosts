# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **12-block V2.0 set** (block 00 + blocks 01-11) at fork point **`097cf33d9`**,
release **`v12-097cf33d9-r95`**.  `release.json` is the source of truth for the tip/tree and hashes; the
top of `WORKLOG.md` is the newest record.

| # | subject | kind | status |
|---|---|---|---|
| **79** | RDNA3_0 verify-band fusions: no multi-device measurement, and gfx1101/gfx1102 untested | platform validation | open (shipped default-on in r95; needs a 2+ card gfx1100 box) |
| **78** | DFlash2 `-sm tensor`: the real Meta distributed `TOP_K` fix | correctness | open, scoping (`wip/dflash-meta-split-impl/`) |
| **76** | the q8_0 K/V-write fusion is inert on iSWA (`build_attn` iswa ordering) | correctness-adjacent perf | open |
| **72** | the unchecked `ncols2 > 1` tile load can over-read a natively-read K/V operand | correctness | open (RDNA2 only) |
| **71** | share the tile K/V type selection between the chooser and the allocator | correctness | open, small |
| **68** | `gemma4` dense + `-sm tensor` + a rotated K cache (`-ctk q8_0`) aborts | correctness | open |
| **67** | reported HIP-graphs start-to-start decode nondeterminism (issue #151 follow-up) | correctness | open — **not reproduced on Linux**; waiting on the reporter |
| **57** | `test-llama-archs` CPU mixed-batch FAIL | correctness | open (pre-existing) |
| **56** | `test-llama-archs -a deepseek2` GPU fault with host experts + `--moe-cache-mib` | correctness | open (pre-existing) |
| **75** | 35B-A3B greedy text is not run-to-run deterministic with async **off** | correctness | open (filed 2026-10-19 when #73 closed) |
| **70** | harden the expert-cache re-arm / wide-aux growth against a starved arena | robustness | open |
| **66** | RDNA2 (`gfx1030`) best-effort bring-up — issue #158 | platform validation | open (no RDNA2 hardware here) |
| **61** | MoE expert-cache coverage: offline trace replay, per-layer capacity quota | performance | open (`wip/moe-cache-coverage/`) |
| **59** | `ngram-mod` + `draft-mtp-adaptive` halves decode with host-resident experts | performance | open (`wip/ngram-mod-host-experts/`) |
| **51** | 3-GPU decode is slower than 2-GPU | performance | open (`wip/three-gpu-performance/`) |
| **36** | genuine CPU/GPU overlap for the MoE misses | performance | open, scoping (`wip/moe-cpu-overlap/`) |
| **77** | segmented per-op expert upload is synchronous (gemma4 ~9 t/s) | performance | open (`wip/meta-segmented-upload-async/`) |
| **65** | IQ4_NL field (MTP) + `--host-experts pool` decode collapses | memory | parked (debug after the next release) |
| **64** | MoE heat-map residuals | minor | open |
| **55** | drop the full pinned `ROCm_Host` master when `--host-experts pool` is on | memory | open (`wip/host-expert-master-drop/`) |

Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 79. RDNA3_0 verify-band fusions: the multi-device arm and gfx1101/gfx1102 are unmeasured — PLATFORM VALIDATION

**Status: open, accepted limitation.  Shipped default-on in r95 on the maintainer's decision**
(`WORKLOG.md` 261020.006, record `archive/work/rdna3-verify-fusions/RESOLUTION.md`).  Not a suspected
defect: the change is arch-identical code behind a host predicate and every single-GPU gate is green
(gfx1100 `W=1..8` maxdiff 0 on six model/prefix cells, `none == draft-mtp`, same-binary kill-switch
identity, prefill-logit KLD 0.000000).

Two gaps, both needing hardware the fleet does not have:

1. **RDNA3_0 multi-device.**  The batched f32 `GGML_OP_CPY` fusion is **not** single-device restricted and
   it is an actively used multi-device path: on 3x gfx1201 with the GDN model + MTP,
   `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1` shows **450 runs under `-sm layer` and 1350 under `-sm tensor`**.
   The RDNA4 multi-device path is therefore heavily exercised, but the same code on a 2+ card gfx1100
   box was never run.  What to run: a GDN model with MTP (`draft-mtp`, n-max 3) under both `-sm tensor`
   and `-sm layer`, plus `test-logits-width-probe` with `SPLIT=tensor`, comparing default against
   `GGML_CUDA_FUSE_CPY_BATCH=0`.  `fingon` is one gfx1100 plus a gfx1036 iGPU and `halo` is gfx1151, so
   neither is a valid host.  Community members running multiple gfx1100 cards are expected to report.
   If a multi-device failure does appear, `GGML_CUDA_FUSE_CPY_BATCH=0` is the immediate workaround and
   restricting that fusion to `get_device_count() <= 1` on RDNA3_0 is the local fix.
2. **gfx1101 / gfx1102.**  `GGML_CUDA_CC_IS_RDNA3_0` covers `0x1100 <= cc < 0x1150`, so gfx1101 (RX
   7800 XT / 7700) and gfx1102 (RX 7600) take the same fusions, but neither the PR author nor the r95
   release ran them.  There is no such card in the fleet.  The same `W=1..8` probe and kill-switch A/B
   is the check if one becomes available.

**Do not re-report as a bug without one of those runs.**

## 78. DFlash2 `-sm tensor`: the real Meta distributed `TOP_K` fix

r54 shipped the honest interim: a `-sm tensor` + DFlash2 combination is **rejected cleanly at model load**
instead of aborting inside the Meta split-state build (the selector's `ggml_top_k` over vocab-split logits
is split on axis 0, which `handle_top_k` does not implement).  The real fix — a Meta distributed top-k, or
a split→mirrored gather before the selector — is unstarted scoping in
[`wip/dflash-meta-split-impl/`](wip/dflash-meta-split-impl/README.md).  `-sm layer` works today.

## 77. Segmented per-op expert upload is synchronous (gemma4 ~9 t/s) — PERFORMANCE

The **correctness** half is done (r54, block 09): `-sm tensor` + a host-resident gemma4 expert table
(`-ncmoe`/`-cmoe`) is no longer rejected and generates coherently (the old accumulate fallback dropped the
scheduler's partial segmented upload; the sync segmented setter asserted on the `copy_experts` MMQ tail
pad).  Record: `archive/work/meta-segmented-expert-upload/RESOLUTION.md`.

The **performance** half is open in
[`wip/meta-segmented-upload-async/`](wip/meta-segmented-upload-async/README.md): the fix routes the per-op
segmented upload through the **synchronous** setter, so gemma4 + `-ncmoe` + `-sm tensor` runs at only ~9
t/s (it was an abort/rejection before, so no working config regresses).  The follow-up is to make that
upload async again (per-row 1-D `set_async` + the tail), with the sync arm as the A/B reference and
`cache-on == cache-off` byte-identity as the correctness oracle.

## 76. The q8_0 K/V-write fusion is inert on iSWA models

The r81 rope + FWHT fusion folds the two q8_0 K/V `SET_ROWS` launches into the fused rope kernel, which
fires on qwen35/qwen35moe because they build attention through the **non-iSWA** `build_attn` (K expands
last).  `build_attn(llm_graph_input_attn_kv_iswa*)` instead expands `q_cur, k_cur, v_cur` and only then the
two cache writes, so the V block sits between the K rotation and the K cache write and the matcher
declines — **inert on iSWA models (Gemma4)**.  Two options: a bounded-window match that tolerates the
interposed V block, or reorder the iSWA expansion and measure it on its own (graph splits), with a trace
showing the two `set_rows` launches are gone.  Record: `archive/work/kv-cache-q8-write/RESOLUTION.md`.

## 75. 35B-A3B greedy text is not run-to-run deterministic with async **off**

Filed 2026-10-19 while closing #73, then explicitly **not** that item: 35B-A3B (1 GPU and 2-GPU tensor)
produces 2-3 different greedy texts run-to-run **with `GGML_CUDA_ASYNC_INPUT=0`**, i.e. on the plain
default path.  Nothing here has isolated it yet.  Start from a fresh-process decode-logit probe
(`test-logits-width-probe` / `decodestart.sh` in `archive/work/kv-cell-offset-nondeterminism/tools/`) to
separate "different binary state" from "different kernel selection", and keep it away from the #67
reporter's Windows-only flip.

## 72. The unchecked `ncols2 > 1` tile load can over-read a natively-read K/V operand — CORRECTNESS (RDNA2)

The r88 guard tail reserves 511 rows after each **converted** F16 operand, which covers the last-tile
over-read for the staged path.  A natively-read operand (an f16 KV cache) in the same branch
(`oob_check = false` for the last tile on every backend) still reads up to `nbatch_fa - 1` rows past the
KV **view**.  In a real model that stays inside the KV cache allocation (one tensor per cache), so it is
not a field issue; it is exposed only by an exact-size tensor as in `test-backend-ops`, where it can read
into the neighbouring allocation.  Options: extend the guard to the native path (pad the cache view or mask
the tail), or take the checked tail for the last tile on RDNA2 and measure the register cost the original
change avoided.  Block 05.  Reported to `@slavap` for follow-up.

## 71. Share the tile K/V type selection between the chooser and the allocator — CORRECTNESS (small)

The tile kernel's K/V type is decided twice by hand: `ggml_cuda_flash_attn_ext_tile_case_type`
(`fattn-tile.cu`, the instantiation) and `ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu`, the F16
scratch the launcher may stage into).  They have drifted three times: the missing q4_0 arm (wasted
whole-cache scratch), the mixed-K/V NaNs (the kernel read a staging buffer `need_f16_K == false` had left
unwritten), and the RDNA2 bf16 case (scratch missing entirely, the whole-cache converter wrote past the
node).  Extract one `ggml_cuda_fattn_tile_type(K, V, cc)` (bf16 under `bf16_mma_hardware_available`, else
the native dequant arm, else F16) and have both call it, so "the allocator and the launcher agree" is a
property of the code instead of a comment.  Block 05 only; needs the FA coherence + prefill-logit gates.

## 70. Harden the expert-cache re-arm / wide-aux growth against a starved arena — ROBUSTNESS

Opened 2026-10-19 while fixing r87.  r87 forces the post-prefill drop, so the r86 trigger (the wide layout
was never released and the arena decayed) is gone.  But the failure shape was: a re-arm that had to shrink
one layer to a partial slot count (`layer=47 ... 344 requested, only 202 fit`), a starved wide-aux region
(thousands of `SLAB_WIDE_AUX` attempts), then a GPU page fault in `k_get_rows_float` on the next graph.
The re-arm is supposed to be layer-uniform and safe; make a partial/refused re-arm leave the cache
**wholesale stood down** (the fuse-guard invariant) instead of running a mixed-slot layer, and make the
wide-aux growth refuse once per pass instead of once per allocation.  Repro: the 10-prompt series against
r86 in `WORKLOG.md` 261019.002; validation arm `MOE_EXPERT_CACHE_VALIDATE=1`.

**r94 note:** the issue #187 fix adds the missing *graph* half of this stale-pointer family — a captured
graph is now invalidated when the slab work boundary shrinks (block 10, `WORKLOG.md` 261020.004).  That
covers the observed #187 fault; it does **not** discharge this item, because a re-arm can still change a
table's arena placement under a graph that is not being dropped.  Keep #70 open.

## 68. `gemma4` dense + `-sm tensor` on 2 devices aborts with a rotated K cache (`-ctk q8_0`) — CORRECTNESS

Opened 2026-10-18.  A plain dense `gemma4` target (`Gemma4/12B/Q8_0`, no draft, `-ngl 99 -sm tensor -fa on
-ctk q8_0 -ctv q8_0 -c 16384 -t 8 -np 1`, `HIP_VISIBLE_DEVICES=0,1`) aborts at the warmup decode:

```
ggml/src/ggml-backend-meta.cpp:900: unsupported mul_mat split states:
  node=node_286 src0=Meta(ROCm0,ROCm1)#attn_inp_k_rot#0 axis=10 src1=Kcur_pos-5 (reshaped) axis=0
```

`attn_inp_k_rot` is the K rotation matrix (`llama_kv_cache::build_input_k_rot`, built only when
`n_rot_k > 0`, i.e. for a rotated-cache type), so the aborting op is `ggml_mul_mat(rot, Kcur)` and its
`src1` (Kcur) is split on **axis 0** — a combination `handle_mul_mat` does not implement.  `-ctk f16` and
`-ctk bf16` load and answer (`n_rot_k == 0`), so the trigger is the rotated cache type, not gemma4 as such.
Independent of `-np`, `--kv-unified`, `-fa` and of `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` /
`GGML_CUDA_DISABLE_FUSION` (the rotation is a meta-side graph op, not a CUDA fusion).
`gemma4-assistant` (the MTP head) is separately and correctly rejected for `-sm tensor` by
`llm_arch_supports_sm_tensor` — that half is intended.

**Fix direction:** teach `handle_mul_mat` the (MIRRORED, axis-0-split) rotation pair; or express the
rotation so Kcur is not split on its hidden axis (rotate **after** the split gather, or per shard with a
full-size matrix); or reject a rotated cache type under `-sm tensor` at the params level the way DFlash is
rejected (#78) — the honest interim, since the current outcome is a crash rather than a clear error.
Repro is a 2-minute server start; the surrounding investigation is
`archive/work/issue-184-meta-r80r82/README.md` §6.

## 67. Reported HIP-graphs start-to-start decode nondeterminism (issue #151 follow-up) — CORRECTNESS (NOT REPRODUCED here)

Opened 2026-10-16 from the issue #151 follow-up; reclassified the same day after revalidation.  The
reporter (`@DanoPTT`, R9700 / Windows / ROCm 10) reports that with HIP graphs on, greedy output differs
start to start under `--no-kv-unified`, on the first request of a fresh server (`-np 2`, draft-mtp n3,
q8_0 KV).

**Not reproduced (2026-10-16, r69 tree, gfx1201 / Linux / ROCm 7.14).**  Fresh-process decode-logit probes
(W=1..8) are bit-identical across starts, graphs on and off, on the 4B Q8_0 and on the reporter's 27B
Swift Q5_K_M with q8_0 KV at P=256 and P=512 (27B W=1 hash `cec25c2aa3cc3e21` both ways), and the
forward/prefill `llama-perplexity --kl-divergence-base` self-comparison is `RMS Δp 0.001 %`.  Data:
`archive/work/kv-cell-offset-nondeterminism/FINDINGS-2026-10-16-revalidation.md`; tools
(`decodestart.sh`, `selfkld.sh`) in that campaign's `tools/`.

**Working hypothesis: the known Windows-only per-start flip, not a graphs or unified-KV root.**  Two such
flips are on record from this reporter: the `rope -> view -> set_rows` fused-vs-unfused clang contraction
(fixed bit-transparent in r25, `#pragma clang fp contract(off)`) and the residual **hipBLASLt solution
selection** (the old issue #67, closed as external; `ROCBLAS_USE_HIPBLASLT=0`,
`ROCm/rocm-libraries#12126`).  Both are prefill-driven and Windows/ROCm-10-only; neither involves graphs,
and the reporter's graphs-off control was a single run.

**Status:** open only as "needs the reporter's Windows bisection": across fresh starts, toggle
`ROCBLAS_USE_HIPBLASLT=0`, `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` and
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`, and compare **decode logits** rather than one text sample.  Do not
treat it as a new delivery defect.

## 66. RDNA2 (`gfx1030`) best-effort bring-up — issue #158 — PLATFORM VALIDATION

Opened 2026-10-16; the code shipped in **r70**, the tuned add-on in **r76**; the platform validation is
open.  r70 folded the two `gfx1030` WMMA compile guards into blocks 06/10 and added `gfx1030` to
`ROCM_DOCKER_ARCH`, so the image now runs on RDNA2 instead of aborting every kernel launch with
`invalid kernel file`.  r76 folds `@slavap`'s tuned RDNA2 add-on (PR #165, from
`sixvolts/llama-halo-hybrid`): per-DKQ tile config, gfx1030 launch caps + WGP-aware occupancy, the
partial-batch tile-load branch (kv=113 fault class), the wide-kernel Q-from-global path, and the Navi21
MMVQ table / MMQ float scale.  All of it is gated to RDNA2 (`#if defined(RDNA2)` / `GGML_CUDA_CC_IS_RDNA2`),
so gfx1100/gfx1151/gfx1201 keep their previous numerics.

No RDNA2 hardware is available here, so it stays **best-effort / unvalidated** (no perf claims, no purity
gates) and `-fa on` still carries the known upstream RDNA2 tile-kernel risk.  Open: external confirmation
on a real `gfx1030` (issue #158), a re-run of `test-backend-ops -o FLASH_ATTN_EXT` and `-o MUL_MAT` on the
current tree, and the deferred halo sparse-tile walk (`60ecdf8b + e53cee3e`).  Record and tester checklist:
`archive/work/rdna2/`.

## 65. IQ4_NL field (MTP) + `--host-experts pool` decode collapses — MEMORY

The Qwen3.8-Flash-Next IQ4_NL 70K field config (`--n-cpu-moe 48`, MTP n7, `--cache-ram 16384`) decodes at
**2.1 t/s** in `--host-experts pool` against the handover's recorded 13.7, while prefill is fine (**497**),
the H2D ring is armed and there are **0 faults**.  The non-MTP `-ncmoe 36/48` pool decode (18-40) matches
the handover's ~2.4x pool penalty, so this reads as a **pool memory shortfall** specific to the 70K/MTP
config.  Parked by the maintainer (2026-10-15) to debug after r67; `WORKLOG.md` 261015.005.

## 64. MoE heat-map residuals (pool warm order, decode-identity tally, dead `moe_cache_observe`) — MINOR

The heat map itself is promoted (r60, `WORKLOG.md` 261014.005).  Three residuals were deliberately left out:

1. **`MOE_EXPERT_CACHE_PREFILL_SEED` is still dead (F1/F6).**  The in-run `prefill_count` tally is gated
   on `!g_sized` and the staged prefill path never reaches the host hook, so `pool_prepopulate_locked`
   warms the `--host-experts pool` pinned tier in **expert-id order** and `pool_rerank_from_tally_locked`
   remains an **uncalled function**.  The heat vector now covers the arena seed, so this is only the
   host-pool warm order (and the dead function) — a small fix.
2. **Decode-identity heat gap.**  A fully-resident table's decode traffic is still not tallied (the
   identity take-over short-circuits `access_locked`); the prefill tally is a good proxy (measured), so a
   sampled decode tally is optional polish.
3. **`moe_cache_observe` is dead code** (superseded by `moe_cache_heat_observe`): remove it or wire it.

Record: `archive/work/moe-heatmap-persistence/` (`RESOLUTION.md` lists these).

## 61. MoE expert-cache coverage: heat-map correctness, capacity allocation, offline replay — PERFORMANCE

Retargeted 2026-10-17 (was "hotness-aware arena placement + idle repack").  Three results of the
2026-10-17 review closed the old premise and opened this one
([`wip/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md`](wip/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md)):

1. **The heat map doubled per flush** — fixed in r80 (block 09: delta merge + a degenerate-map guard);
   the poisoned sidecars were deleted.
2. **Defrag/repack is not the lever**: the arena is ~100 % allocated in slots (0.51 % truncation) and
   eviction churn is 3.0 % of reaches, so an idle repack recovers ≤ 0.5 % of bytes while invalidating
   captured graphs and the arena address in kernel launch params (old phases 3-4 are closed).
3. **The Strata hit-rate gap is accounting + arena size**, not fragmentation: on equal accounting 91.9 %
   vs our 90.6 % on the same model, and the residual is 25.46 vs 15.30 GiB of arena.

**Open work** (`wip/moe-cache-coverage/`): (a) an **offline routing-trace replayer** (`MOE_EXPERT_TRACE` +
a policy scorer — we have only an aggregate heat vector, which cannot answer thrash/prefetch/window
questions); (b) a **per-layer capacity quota** from heat (uniform slots per table give up only 0.3-0.5 pts
on a uniformly-tallied map; the 1.5-1.9 pts seen on the 35B-A3B sidecars was the #64.2 tally-gap artifact)
— the only strictly-inside-the-arena win, and a small one; (c) metric parity (report covered mass +
expert MiB/token beside `h`).  The **arena size** lever (the slab's 8 GiB compute reserve + 4 GiB VA
spare, worth +3-6 pts of coverage) belongs to `archive/work/slab-transient-absorption/`, not here.

## 59. `ngram-mod` + `draft-mtp-adaptive` halves decode with host-resident MoE experts — PERFORMANCE

Opened 2026-10-14; scoping.  With partially or fully host-resident MoE experts (`-ncmoe`/`-cmoe`, the
expert cache or op-offload path active), the README's recommended
`--spec-type draft-mtp-adaptive,ngram-mod` combo roughly **halves decode throughput**; dropping
`ngram-mod` and using plain `draft-mtp-adaptive` restores the expected rate.  The combo is the measured
best on a fully device-resident model, so the interaction is with the host-expert serve path.  Documented
in the README "Recommended configuration" section.  Working home
[`wip/ngram-mod-host-experts/`](wip/ngram-mod-host-experts/README.md) (hypotheses: verify width against the
per-op host-expert upload, the MoE cache band and verify graph recapture, the staging-ring width; bisect
plan in that README).

## 55. Remove the full pinned `ROCm_Host` master when `--host-experts pool` is on — MEMORY

Opened 2026-10-11; scoping.  The r49 mmap reclaim (block 06) removed the mmap half of the `-ncmoe` host
peak, but `--host-experts pool` still keeps the **full pinned master** (r35 shipped the pool as a bounce
tier; the source logs "Phase 2b; full pinned master kept").  `RssShmem` is identical with and without the
pool: 18.7 GiB on 35B-A3B `-ncmoe 40`, 54.1 GiB on Flash-Next IQ4_NL `-ncmoe 40`.  Dropping the master is
the strongest lever for the **memory-pressure** half of issue #139 (the amdgpu driver stops all of a
process's GPU queues when the kernel migrates a GPU-mapped host page); the AR timeout itself was fixed in
**r59** (block 01, wall-clock soft/hard spin deadlines), so the remaining value here is fewer stalls, not a
correctness fix.  Working home [`wip/host-expert-master-drop/`](wip/host-expert-master-drop/README.md)
(gates and the reader enumeration are in that README).

## 51. 3-GPU decode is slower than 2-GPU (MTP and plain, partial and fully resident) — PERFORMANCE

Reopened 2026-10-16; measured.  Formerly closed at r55 (a fresh A/B then found 3-GPU MTP 74.3 vs 2-GPU
65.0 t/s).  A 2026-10-16 measurement on r73 found the opposite: 2-GPU **84.4** vs 3-GPU **56.9** t/s
(13.8K prompt, GSQ-IQ3_XXS, `-ncmoe 48 -sm tensor`, adaptive MTP), and it is present at **100 % arena
residency** (so not an arena-sizing artifact).  3-GPU prefill is the fastest (1258 t/s).  Broader than MTP
(the fresh arm is MTP; the matrix must cover plain and fully-resident too).

Working home [`wip/three-gpu-performance/`](wip/three-gpu-performance/README.md) (revived from
`archive/work/mtp-3gpu-decode/`; the r55 closure record is kept there as
`RESOLUTION-2026-10-13-closed.md`).  Step 0: re-run the r55 harness on the current tree, then the full
`{2,3}` x `{plain,MTP}` x `{partial,fully-resident}` x `{tensor,layer}` grid, then the all-reduce /
identity-path / tensor-split-gather profile.

## 36. Genuine CPU/GPU overlap for the MoE misses (Strata's pipeline shape) — PERFORMANCE (scoping)

Opened 2026-10-05.  The delivered cache-on path computes every expert on the GPU (resident from VRAM,
misses via UVA over PCIe); the cache-off path computes them on the CPU but **serialised** with the GPU
splits.  Our own CPU-computes-misses arm was correct and negative above a ~1.2 GiB arena (the loss is
~600 scheduler dispatches + ~3 cross-backend copies per layer/token, not CPU compute; see
`archive/work/moe-expert-cache/WORKLOG.md`).  Strata's custom engine runs the CPU pool *because* it is not
under llama.cpp's serial scheduler.  **Next:** O0 — measure how much of a 52 t/s token is the UVA cold
fraction at 43 % residency; if it is <20 %, park it.  Design space and gates:
[`wip/moe-cpu-overlap/`](wip/moe-cpu-overlap/README.md).

---

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
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

## Where the current lists live

- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.
