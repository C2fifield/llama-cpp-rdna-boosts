# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md` (there is no `wip/README.md`).

**A closed or promoted campaign does not live here — not even as a redirect stub.**  Its record moves to
`archive/work/<campaign>/` and `WORKLOG.md`; the history in dated records refers to the path that was
correct at the time.  Keeping finished work under `wip/` (or leaving a "MOVED — see archive" placeholder)
makes the directory lie about what is in flight, so the stub habit is retired.  See
`archive/work/lightning-indexer-fusion/README.md` for the resolution pattern
(investigate -> fold into the owning block -> regenerate + validate -> record -> archive -> ship).

An entry below is one of exactly two things: a campaign with **work in flight**, or a **scoping note for
work not yet started** (a plan, not a result).  Both stay.  The distinction matters when tidying: a
"MOVED — archived/closed/promoted" redirect is finished work and is deleted, while an unstarted scoping
note is open work and stays.  When in doubt, look for the word MOVED at the top of the `README.md` — that
is the signature of a stub, not of a campaign.

| directory | what | status |
|---|---|---|
| [`issue-170-meta-alloc/`](issue-170-meta-alloc/README.md) | **issue #170**: `-sm tensor` on a 24 GiB-class split aborts at the warmup decode.  Root cause is *not* a failed allocation: `--fit` reduces `n_gpu_layers` -> a CPU-offloaded layer's ops are op-offloaded onto the Meta backend (block 09's `offload_op` hook; upstream's meta iface has none) -> the meta split-state walk recurses into `build_attn_mha`'s intermediate `ggml_view_4d`->`ggml_permute` alias, which the gallocr never initialises (`buffer == NULL`) -> `buffer_n_bufs(NULL)` abort.  Second, independent defect on the same path: the meta wrapper **aborts** on a failed constituent allocation instead of returning `NULL` (so a genuine KV/weight/compute OOM is a core dump, not an error).  Both fixed in the fork tree (staging branch `fix-170`, `[TAG_META_UNALLOC_ALIAS]` + `[TAG_META_ALLOC_FAIL]`), reproduced on 2x R9700 by holding 8192 MiB/device so the fit behaves like a 7900 XTX box; validated: the reporter's minimal and full command lines listen + serve `HTTP 200`, the three OOM classes exit 1 with clean errors, coherence text identical (`sha=383655591392`) | **FIXED in the fork tree (2026-10-17) — not in `patches/`; needs the maintainer's call (propose amend block 09 -> r82; `upstream/` candidate)** |
| [`three-gpu-performance/`](three-gpu-performance/README.md) | **3-GPU decode is slower than 2-GPU** — MTP and plain, partial and **fully resident** (the former `archive/work/mtp-3gpu-decode/`, closed at r55, **revived** 2026-10-16).  Fresh measurement: 2-GPU 84.4 vs 3-GPU 56.9 t/s at 100 % residency, while 3-GPU prefill is fastest.  Step 0 is to re-run the r55 harness on r73; then the full {2,3} x {plain,MTP} x {partial,fully-resident} x {tensor,layer} grid, then profile the all-reduce / identity path / tensor-split gather | **OPEN (revived 2026-10-16); measurement first** |
| [`dflash-meta-split-impl/`](dflash-meta-split-impl/README.md) | the **real fix** behind the r54 clean rejection: a Meta distributed `TOP_K` (or a split→mirrored gather) so the DFlash2 selector's `ggml_top_k` can reduce a vocab-split input under `-sm tensor` (TODO #58 follow-up) | **OPEN (seeded 2026-10-13), scoping + design** |
| [`kv-cell-offset-nondeterminism/`](kv-cell-offset-nondeterminism/README.md) | **the unified-KV residual (TODO #52 part C)**: a sequence's attention result depends on the physical **set** of KV cells it occupies in the shared stream, not just its start offset; under `--kv-unified` the allocator lets that set depend on scheduling history, so greedy repetitions differ | **PARKED (2026-10-11)** — root cause pinned (cell-set vs the FA reduction/split grid; a fixed alignment is insufficient, and the grid is model/shape-dependent), but a proper fix is an upstream unified-KV redesign (per-sequence contiguous, tile-aligned regions). Documented as a gotcha (`-np > 1` + `--kv-unified` is not deterministic; use `--no-kv-unified`); issue #134 closed as upstream-inherent. Record: [`FINDINGS-offset-mechanism.md`](kv-cell-offset-nondeterminism/FINDINGS-offset-mechanism.md), probe + repro in `kv-cell-offset-nondeterminism/tools/`.  **Revalidated 2026-10-16 (r69):** the effect still reproduces (unified red); `--no-kv-unified` concurrent repeats are green but solo != concurrent (cross-sequence batching, deterministic).  The issue #151 graphs-on flip (TODO #67) was **not reproduced** on Linux/ROCm 7.14: [`FINDINGS-2026-10-16-revalidation.md`](kv-cell-offset-nondeterminism/FINDINGS-2026-10-16-revalidation.md) |
| [`rs-rollback-boundary/`](rs-rollback-boundary/README.md) | the recurrent rollback-boundary guard (issue #131): the global `last_ubatch_nseq_tokens` tracker reports the wrong sequence; the per-sequence guard is folded into block 02, but the **"large prefill as the last prepared batch before a rollback"** ordering and whether the restored snapshot is stale are open | **OPEN / ACTIVE (resuming)** — the dense 27B does not reproduce the warning on r43 (0/106 rollbacks) and `--no-kv-unified` does not change that; the MoE reproducer fires it and shows the rolling sequence's own last prepare is a 4571-token prefill. Open questions, reproducers and instrumentation: [`README.md`](rs-rollback-boundary/README.md) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`host-expert-master-drop/`](host-expert-master-drop/README.md) | remove the full pinned `ROCm_Host` master when `--host-experts pool` is on (the pool's Phase 3): every cold expert served from the bounded pool over the page cache | **OPEN (seeded 2026-10-11)** — the r49 reclaim removed the mmap half of the `-ncmoe` host peak, but `--host-experts pool` still keeps the full pinned master (measured `RssShmem` identical pinned vs pool: 18.7 GiB on 35B-A3B, 54.1 GiB on Flash-Next IQ4_NL).  Also the strongest lever for issue #139's `evict_process_queues` stops |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (the Strata shape) | OPEN / scoping (TODO #36) — **work in flight**: the note refutes the earlier "cannot be done" verdict for the interleaved shape and sets out the pipeline to try |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`fa-band-2byte-row/`](fa-band-2byte-row/README.md) | extend the RDNA4 GQA band's 64-wide K/V config row to the 2-byte f16/bf16 arm (`ncols == 16`; TODO #27) | **OPEN (seeded 2026-10-11), small tuning** |
| [`rdna35-mmvq-fusion-perf/`](rdna35-mmvq-fusion-perf/README.md) | restore the block-13 RDNA3_5 single-token dense-GLU fusion perf (~0.9 % `tg128`; TODO item 16) | **PARKED** — the original "pin the params" plan is dead; live candidates are codegen FMA contraction and the Q8_1 cache |
| [`meta-segmented-upload-async/`](meta-segmented-upload-async/README.md) | make the segmented per-op expert upload **async** again (per-row 1-D `set_async` + the MMQ tail) — r54 fixed it synchronously, so gemma4 + `-ncmoe` + `-sm tensor` is correct but only ~9 t/s | **OPEN (seeded 2026-10-13), perf follow-up** |
| [`ngram-mod-host-experts/`](ngram-mod-host-experts/README.md) | `ngram-mod` + `draft-mtp-adaptive` roughly halves decode when the MoE experts are host-resident (`-ncmoe`/`-cmoe`, expert cache active); plain `draft-mtp-adaptive` restores the expected rate | **OPEN (seeded 2026-10-14), scoping + measurement** |
| [`moe-cache-coverage/`](moe-cache-coverage/README.md) | **how much of the routing distribution the arena can hold and how well admission uses it** (renamed from `moe-arena-defrag` 2026-10-17).  P0 (heat-map correctness) shipped in r80; next is the offline routing-trace replayer, then a per-layer capacity quota | **OPEN (renamed + retargeted 2026-10-17)** — the defrag premise was refuted by measurement: the arena is ~100 % allocated in slots (0.51 % truncation) and eviction churn is 3.0 % of reaches, so compaction wins ≤ 0.5 % of bytes.  The same review found the **MoE heat map doubling per flush** (fixed in r80) and showed the Strata hit-rate gap is accounting + arena size, not fragmentation (equal basis: 91.9 % vs 90.6 %).  Plan: P1 offline routing-trace replayer, P2 per-layer capacity quota (0.3-0.5 pts on a clean map, the only strictly-in-arena win), P3 metric parity; arena size handed to `slab-transient-absorption/`.  See [`FINDINGS-2026-10-17-coverage-vs-defrag.md`](moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md) |
| [`slab-transient-absorption/`](slab-transient-absorption/README.md) | move the controlled raw-`cudaMalloc` transients (FA staging, Q8_1, MMB shadows, AR scratch, per-table remap, pool high-water) inside the slab so `GGML_CUDA_SLAB_HEADROOM_MIB` can be safely lowered | **WIP COMPLETE in the fork tree (2026-10-17) — not promoted.**  Landed in the fork tree (not in `patches/`): **step 0** (H2D ring slab-only + migration-leak fix) and a **wide-phase auxiliary region** that now hosts the **FA prefill staging arena** and the **QSA3 prefill workspace** (the QSA3 redirect is measured: pool high-water 372.3 → 271.4 MiB, free at the 15.7k-prefill drop 1842 → 1942 MiB, greedy text identical).  §8 has the decomposition of the remaining ~628 MiB peak (pool 372 → 271, hipBLASLt ~104, MMB ~40) and the headroom sweep (2048 and 1024 both clean on the field config; 1024 leaves ~1.8 GiB at the peak).  Next: MMB activation cache, then the remaining prefill-only pool consumers, then the `-ub 6144` wide matrix before any default change.  §9: aux 4096 -> 3584 (+512 MiB arena) and the 512 MiB load-time H2D calibration probe now come from the slab; `GGML_CUDA_POOL_CACHE_MB` is a measured dead end (327 vs 271 MiB); headroom 1024 + aux 3584 = arena 19 263.5 MiB (47.1 %), 408 MiB left at the 15.7k prefill drop.  §10: chunked-GDN prefill scratch (78 MiB/ubatch) is now in the band too; an **80K-token prefill at headroom 1024** runs 1 165 t/s with **331.5 MiB** left at the drop; the generic pool redirect was deferred at that point (MMB/allreduce keep pool blocks ACROSS graphs) and is landed in §10.6.  Only `headroom + aux >= ~3.9 GiB` matters, so the next step is headroom 512 / aux 3584, not aux 1536.  §10.2 (real seesaw, distinct prompts): short/long/short/long bottoms out at **204 MiB outside** at headroom 1024, clean, bit-stable across the two 80K prefills; a 512 headroom would be ~-300 MiB, so the deferred generic pool redirect (pool high-water 337 MiB) is the only route to 512.  §10.3: the **wide `-ub 6144 -c 204800` gate passes at 2048 / 3584** on IQ4_NL (629 MiB left at the drop, no `hipblaslt.cpp:164`); 1024 would be ~-42 MiB there (pool 671 MiB at ub 6144), so the recommendation is **2048 / 3584 default + 1024 / 3584 opt-in for `-ub <= 4096`**.  §10.4: **2 GPUs clean in both `-sm tensor` (40 725 MiB arena, 99.5 %) and `-sm layer` (40 199 MiB, 98.2 %)** at 2048/3584, identical greedy text, no errors; the only remaining gate is a long-context `-sm layer` run (the r36 6144 floor was a 255k `-ts 59,41` config).  §10.5: the wide gate is also clean at **1536** (arena 18 225.0 MiB / 28.1 %, 459 MiB left at the drop, same greedy sha as 2048).  **§10.6 (step 2b, landed WIP `3f4497d14`): the generic workspace-pool wide redirect is in** — `ggml_cuda_pool::set_wide_routing` + `ggml_cuda_pool_wide_pin` (the pinned escape hatch), `pool_leg::alloc` serving large allocations from the armed wide band and `free` returning them to it (never cached, never `cudaFree`d), plus a **`POOL_WIDE_LEAK` tripwire** at the drop and `SLAB_STATS pool_wide=.../refused=...`.  Pinned: MMB cache + slots, the allreduce/NCCL BF16 scratch, and **`bf16_src1_cache`** (NOT on the handover's list — the tripwire caught it as a real use-after-free).  `mmf.cu:51-53` checked and left unpinned (function-local RAII).  Also fixed a pre-existing `wide_aux_alloc` grow check that refused every growth once the base retained a chunk gap (it silently pushed the aux back outside the slab: 80K arm free 577.5 -> 349.5, and an OOM at wide headroom 1536).  Measured: seesaw @1024 floor **204.5 -> 286.5 MiB**, long drop 333.5 -> 390.5, pool 337.4 -> 262.9, prefill 1 165 -> 1 247 t/s, all four greedy shas identical; wide gate @1536 459 -> 477 (pool 335 -> 316, kill switch reproduces 459/335 exactly) but **@1024 it still OOMs** (an ~80 MiB pinned cache alloc with ~80 MiB free), so **1536 stays the `-ub 6144` floor**; 2-GPU tensor 2 118.5 -> 2 238.5 and layer 1 097.5 -> 1 349.5 MiB/device, same sha; prefill-logit KLD gate **0.000000 KLD / 100 % same-top-p** against a base recorded by the pre-change build on a slab-active config; 35B-A3B greedy identical.  **Finding (decision needed before the DoD):** the residual pool is the MMB activation cache (~168-170 MiB live at the drop, measured by the `GGML_CUDA_MMB=0` A/B: pool 262.9 -> 95.2, drop 390.5 -> 589.5, floor 286.5 -> 452.5) plus `bf16_src1_cache` (~80 MiB) — both pure caches, so the route to pool <= 80 MiB / floor >= 500 is a **yield-on-reclaim** (flush them in the same device callback that disarms the ring), deferred pending the maintainer's call and its own prefill-cost measurement.  **No delivery default changed, nothing promoted.**  **§10.8 (step 2b-2, WIP `f639cb646`): the per-graph cache yield is LANDED** — `graph_compute` releases the MMB activation cache + slots and `bf16_src1_cache` on scope exit (both are per-graph caches, so no hit is lost), the three wide pins are gone, and the wide-aux region now tracks its LIVE blocks.  That last part was the real work: putting graph-lifetime blocks in a bump stack made out-of-order frees the norm and every one left a permanent hole (measured 7 392 MiB of `wide_aux` and 149 holes in ONE 6.6k prefill, arena 20 096 -> 4 992 MiB, 417 GiB refused); `SLAB_STATS` now reports `wide[n_pop hole inval rebase]` and `hole=0` is the acceptance signal.  Measured: seesaw @1024/3584 floor 286.5 -> **522.5** MiB, pool 262.9 -> **23.6**, prefill 1 238 t/s, all shas identical, decode unchanged; wide gate @1536 drop 477 -> **791** (pool 315.6 -> 6.4) and **@1024 now PASSES** (279 MiB, OOM before); 2-GPU tensor 2 238.5 -> **2 384.5** and layer 1 349.5 -> **1 489.5** MiB/device; 35B greedy identical; prefill-logit KLD 0.000000 / 100 %.  **`aux 1536 / headroom 2560` (sum 4096) now completes the seesaw with identical shas** but with only **52.5 MiB** of outside margin, so it is reachable, not shippable on one run — the scope note's reserve matrix (§6) plus the ~600 MiB steady-state decomposition is the remaining work before any default.  **No delivery default changed, nothing promoted.**  **§10.10 (step 1, WIP `d6938507d`): the draft reserve is DERIVED** from the draft's own GGUF instead of the flat 3584 constant — weights (sum of tensor bytes, exact) + KV for n_ctx at the real `-ctk/-ctv` element size (+ a declared indexer cache at f32) + a measured compute tier (512 MiB at `-ub <= 4096`, 768 above) + 192 MiB slack; falls back to the constant for a shared/embedded head or missing metadata, and `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` still overrides.  Calibrated against the measured draft footprint: 3782 vs 3584 (field) and 4281 vs 4000 (wide), i.e. +5.5 % / +7 % (the safe direction).  Effect: field seesaw @1024/3584 outside 522.5 -> 714.5, wide @1536/3584 drop 791 -> 1495, 2-GPU tensor 2384.5 -> 2576.5 / layer 1489.5 -> 1681.5 MiB/device, all shas identical, 35B coherence identical.  **This is a correctness step, not an arena win**: with the current headroom default a correct aux grows the reserve by the same amount, so the arena loses 192/704 MiB until step 2 lowers the headroom (`reserve = D + excess`; `aux = D` only pays off with `headroom ~= T + 300`).  Also exposed: `--fit`'s margin (`max(base, slab_headroom) + arena_budget`) does NOT include the draft at all, so the draft is accounted for only by shrinking the arena — the next piece to design, with the 2/3-GPU per-device split of D.  **No delivery default changed, nothing promoted.**  **§10.11 (step 2 / fit, WIP `aec734be8`): `--fit` now reserves the MTP draft.**  Correcting §10.10: fit DOES have an exact per-device draft accounting, but an MTP draft cannot be measured during fitting (its context needs `ctx_other`, and fitting runs before the target context exists) and the catch path set it to ZERO — so fit sized the target as if the draft were free, the draft then came out of the slab reserve and the outside cushion collapsed.  Fixed with `common_fit_extra_model::reserve_bytes` (the derived reserve) used on both failure paths (per-device and the Meta/`-sm tensor` one), the reserve derived once before both fit and the preflight, and `n_ctx == 0` resolving to the draft GGUF's `context_length` instead of under-estimating.  Also fixed a segfault on that path (the fallback must return; the code below dereferences the empty `measured`).  Measured: fit reserves **4469 MiB** (wide, `-ub 6144`) / **4213 MiB** (2-GPU, `-ub 4096`) instead of 0; wide `--fit on -c 0` at headroom 1024 -> outside 1007 MiB at the drop, identical sha, 618 t/s; 2-GPU `-sm tensor` -> 2596.5 MiB/device, same sha.  **The remaining `--fit` work is only the numbers**: `reserve = D + headroom` per device, so the matrix (scope note §6) is now the only thing between this and a lowered default.  **§10.14 (step 3c, WIP `156f32652`): the draft reserve is split-aware under `-sm tensor`.**  The headroom did not need split awareness — the AUX did: the slab reserve is per device, but it charged the whole draft footprint to every device, and under `-sm tensor` the weights+KV are split, giving a constant **+895 MiB/device** of unused outside VRAM at every headroom.  For an equal tensor split each device is now charged `(D - tier)/n + tier + 128` (2 GPUs 2371 instead of 3782): **2-GPU tensor excess 1296.5 -> 300.0 MiB, arena 17 344 -> 18 752 MiB/device (+2.8 GiB on the box)**, 3-GPU tensor 408.0 MiB with arena 20 608, both with the unchanged sha.  `-sm layer` is untouched (the head's layer lands on ONE device — the split value was measured to abort at draft load with a 512 MiB alloc), and an explicit `-ts` keeps the full reserve (shares not plumbed; documented follow-up).  The finished matrix has every measured topology at or above the 300 MiB target (300.0-599.0) with the arena maximal for that cushion.  **§10.13 (step 3b, WIP `5d461d67f`): the headroom default scales per GPU — `512 + 128 x n` (1 GPU 640, 2 GPU 768, 3 GPU 896 MiB).**  A flat 600 was measured and REJECTED (2-GPU `-sm layer` drops to 209.5 MiB, under the 300 target, while the 1-GPU field seesaw keeps 346.5): `T` is per DEVICE (~300 / ~390 / ~500 MiB for 1/2/3 cards), so a flat value is either too generous on one card or too thin on several.  Verified with NO env: 1 GPU field seesaw floor 346.5 MiB and arena **19 776** MiB (+4.0 GiB vs the r80 default 15 666.6), wide drop 599.0 (sha 746954a1038b), 2-GPU layer 401.5 / tensor 1296.5 MiB per device, 3-GPU layer 519.5 on all three, 35B coherence identical.  An explicit `GGML_CUDA_SLAB_HEADROOM_MIB` stays verbatim and the `at_least_mib` floor stays raise-only/unset-env, so the 6144 `-sm layer` floor is unaffected; a fully resident model has no slab at all.  **§10.12 (step 3, WIP `9c46d14df`): the targets are set — `GGML_CUDA_SLAB_HEADROOM_MIB` default 4096 -> 1024**, verified with no env: field seesaw floor **718.5 MiB** and arena **19 136** MiB (was 15 667 at the r80 default; reserve 7 680 -> 4 806 MiB, **+3.4 GiB of arena**), wide IQ4_NL drop **983**, 2-GPU tensor 1 552.5 / layer 657.5 MiB per device, 35B coherence identical.  512 measured and rejected (field floor 214.5, under the 300 target); 1024 keeps ~2x the worst measured transient demand T=502.  A fully resident model has no slab/arena at all, and the 6144 `-sm layer` floor is raise-only + unset-env only, so the lower default cannot weaken it.  New finding: **the MoE expert-cache hit/miss changes the numerics** (the 2-GPU `-sm layer` text returns to the reference sha when the arena is pinned to the reference size), so sha checks across reserves are only valid with the arena held constant.  **Nothing is in `patches/`; promotion (re-cut against `release.json.base`, release gates, new block) is the remaining step and needs the maintainer's go-ahead.**  **No delivery default changed, nothing promoted.**  **§10.7 + [`SCOPE-2026-10-17-mmb-yield-aux1536.md`](slab-transient-absorption/SCOPE-2026-10-17-mmb-yield-aux1536.md) (scoping, 2026-10-17):** the follow-up (absorb the MMB activation cache + `bf16_src1_cache`, ~250-315 MiB) is designed but not implemented — ~40 lines / 3 files, an end-of-graph RAII yield of the two *per-graph* caches (no registry, no `ring_set` coupling), with `POOL_WIDE_LEAK == 0` as the acceptance criterion.  Measuring the `aux = 1536` target first changed the plan: **only `headroom + aux` matters, not the split** (2048/2048 and 2560/1536 (sum 4096) are byte-identical, and sum 3584 cannot load the draft at all), so `aux 1536` implies `headroom ~= 2560`; and at sum 4096 the abort is precisely an 84 MiB (80 MiB + 5 %) MMB cache entry with 93.5 MiB free at the wide arm — i.e. the MMB cache *is* the blocker at that reserve.  Note the headroom is permanently outside the slab, so `aux 3584 / headroom 512` at the same sum dominates `aux 1536 / headroom 2560` on arena; the choice is safety, not arena (decision D1 in the scope note).  §1a is the current table (line numbers, measured sizes, three previously missing rows); §1b corrects the r60 findings (`slab_extend` is NOT dead when an MTP draft is present: `reserve = headroom + aux`, field `va_spare = 4.00 GiB`).  The payoff is now measured: ~1 pt of hit rate per GiB moved from headroom to arena, and 2048 headroom is already clean on the field config.  Phase map §3.0, hybrid routing §3.1 (wide→ring, narrow→narrow-2-style pin) |

**`slab-transient-absorption/` re-verified at r80 (2026-10-17).**  Before the next session's pickup the
campaign was audited against the current tree: its §1 table was three releases stale, so **§1a** is now the
current one (refreshed line numbers, measured sizes, and three rows it was missing — the cuBLAS workspace,
the `ggml_cuda_pool_leg` high-water, and the 512 MiB H2D calibration peak) and **§1b** corrects F1
(`slab_extend` is **not** dead when an MTP draft is present: `reserve = headroom + aux`, field
`va_spare = 4.00 GiB`).  **F5 was also rewritten**: the H2D ring's `cudaMalloc` fallback is now dead weight
in host-expert mode (the region is sized `slots × largest staged slice` and the outside pool is only
~0.3-0.6 GiB free), and it hides two defects — an unarmed request bypasses the "never read a disarmed slot"
guard, and a slot that migrates to the slab leaks its old block.  **Step 0 of the campaign is therefore:
make the ring slab-only (refuse instead of `cudaMalloc`) when the slab is active.**  The campaign's payoff
is now measured rather than estimated (~1 pt of hit rate per GiB moved from headroom to arena); see
`wip/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md` §6.1.

**Promoted (r79, 2026-10-17).**  PR #167 (`rdna3-mmvq-moe-q4k-vdr4`) was promoted into **blocks
07/10**: the MoE expert kernel takes `vec_dot_q4_K_q8_1_vdr4_dense`, **default on RDNA3_0**
(+0.83 % tg256, +1.4/+1.9/+2.7 % batched B=1/4/8, 4-axis MTP R/C/K/phase-switching
+1.2-3.0 %; the P axis is -2.7 % acceptance-driven content drift, plain decode on the same prompt
+0.7 %) and an **RDNA4 build-time opt-in** (`-DGGML_HIP_RDNA4_Q4K_MOE_VDR4`, default off: 2-GPU
tg flat, +0.5-1.5 % at the verify widths, mixed MTP).  The same release flips the PR #168 KV prefill
pad to **default-on (8192) on a multi-device split** and inactive on one GPU (3-GPU tensor +2.5 % MoE
/ +0.8 % dense, 3-GPU layer +42 % / +49 %, 1-GPU unchanged).  The PR was never a `wip/` entry on
`main` (it arrived as a PR branch), so the open set was not touched; the record is
`archive/work/rdna3-mmvq-moe-q4k-vdr4/` with a `RESOLUTION.md`.  The P-axis finding is cross-referenced
in `three-gpu-performance/README.md` §0.1 (it is not 3-GPU-specific).  See `WORKLOG.md` 261017.002.

**Promoted and rejected (r78, 2026-10-17).**  Three contributor PRs were reviewed; **none of them was
ever a `wip/` entry on `main`** (they arrived as PR branches under their own `wip/<name>/`, so the open
set here was not touched and no stub was created or removed).  **#172 `kv-rot-fwht-fusion`** promoted
into **block 10** (the KV rotation's FWHT runs in the fused rope kernel, and the attention-output FWHT
also fills the projection's Q8_1 cache -- both bit-identical; +1.5 % Gemma4 12B decode).  **#174
`add-rms-norm-q8-decode`** promoted into **block 10** (the residual ADD + `rms_norm_q8_1` fusion now
covers one token, +1.7 % dense 27B decode; bit-identical).  Both are reachable only where the graph
matches: #172 needs NORMAL/NEOX rope, so it is **inert on every Qwen3.x model here** (`rope_multi`), and
#174 is inert where the shared-expert epilogue already eats the residual add.  **#173
`kv-cache-q8-write` rejected**: it folds the q8_0 K/V cache `SET_ROWS` into the #172 kernel, but its
matcher cannot fire on this tree (the ISWA builder interposes the V block between the K rotation and the
K write, and every other model here is `rope_multi`), so it would ship unmeasurable -- re-entry
conditions in `archive/work/kv-cache-q8-write/RESOLUTION.md`.  All three records are under
`archive/work/` with a `RESOLUTION.md`; see `WORKLOG.md` 261017.001 (r78).

**Promoted and rejected (r77, 2026-10-16).**  Three contributor PRs were reviewed; **none of them was ever
a `wip/` entry on `main`** (they arrived as PR branches under their own `wip/<name>/`, so the open set
here was not touched and no stub was created or removed).  **#171 `q8-decode-small-kernels`** promoted
into **block 08** (q8_0 `SET_ROWS` one 32-lane group per block; `rms_norm_q8_1` quantizes from
registers -- bit-identical, MTP decode +0.6-0.7 %, no regression at the verify widths).  **#168
`kv-pad-prefill`** promoted into **block 10** as an **opt-in** tuning value (`LLAMA_KV_N_PAD_PREFILL`,
default off: +2.0 % prefill on 35B-A3B at 2 GPUs and +1.0 % on the dense 27B at 3 GPUs, but -0.35 % at
1 GPU, so it is not default-on).  **#169 `spec-batch-budget` rejected** (its cap-to-1 arm is -16 % on the
MoE at 3 GPUs, -2.7 % in the reporter's own 2-GPU topology, -12 % on the dense 27B; plus the adaptive
controller is fed the uncapped depth, so a burst drains it).  All three records are under
`archive/work/` with a `RESOLUTION.md`; see `WORKLOG.md` 261016.012 (r77).

**Promoted (r75, 2026-10-16).**  `first-prefill-latency/` left `wip/`: the MoE arena is now sized at the
warmup (the priming pass), with the arena sized to the **full** region and the seed capped at the prefill
reclaim boundary (`narrow_floor + ring + wide`) via the shared `ggml_cuda_slab_ring_layout` helper.  Folded
into **blocks 09/10/11** and default-on (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` is the kill switch); measured
on the maintainer's GSQ command (1 GPU, 13.8K prompt): arena 15666.6 MiB at load, 97/144 tables seeded,
first-prefill eviction ~6.1 GiB (the band, reclaimed cheaply and re-armed to full), decode 75.0 t/s.  An
earlier persistent-reservation draft (eviction 84 MiB) shrank the arena permanently and was reverted.  r75
then fixed a placement regression this work exposed (the wide compute view aliasing the carved H2D ring;
blocks 10/11).  The `keep-survivors` prototype was inert in this config and stays un-promoted.  Record
moved to `archive/work/first-prefill-latency/` (with a `RESOLUTION.md`).  The 3-GPU decode deficit it
surfaced is revived as `wip/three-gpu-performance/`.  See `WORKLOG.md` 261016.009 (r74) and 261016.010 (r75).

**Promoted (r69, 2026-10-16).**  Two contributor PRs from `@briansp2020` (#162, #163) were merged and
folded into their owner blocks (07/08/10; the delivery stays a 12-block set), then archived with a
`RESOLUTION.md` each under `archive/work/`: `rdna3-mmvq-q4k-vdr4/` (gfx1100 dense Q4_K VDR-4 plus the
Q6_K/Q4_K verify row rules; +2.1 % MTP on the 7900 XTX; the RDNA4 arm is a build-time opt-in
`-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4`, default off) and `q8-quantize-fusion/` (fewer `quantize_q8_1`
launches in `-sm tensor` decode; output bit-identical, accepted as a no-regression cleanup).  The same
release reverts `GGML_CUDA_ASYNC_INPUT` to opt-in (`=1`, default off) after a `-ub 8` perplexity
self-comparison exposed r68 run-to-run logits divergence.  See `WORKLOG.md` 261016.001.

**Promoted (r68, 2026-10-15).**  `issue-151-host-path-regression/` left `wip/`: both candidate patches
plus a log-hygiene change shipped in `v12-097cf33d9-r68` (canonical tip `fc6dcad9f`, net tree
`ffe832696698786865f4adf18216a019b61b160e`).  `GGML_CUDA_ASYNC_INPUT` is now **default-on** (`=0`
restores the synchronous path; the reporter measured +1.4 % decode on their slow host), the `O(gathers x
nodes)` GDN state-gather walk is replaced by a per-graph index built lazily on the first candidate
(`GGML_CUDA_GDN_GATHER_INDEX`; local **-0.63 ms host/graph, -13 %**, +0.23 % tg128, reporter +0.3 %), and
the two per-table arena seed notices (`apply_prefill_seed_locked`, `apply_heat_seed_locked`) moved from
`WARN` to the `-lv 4`-only `INFO` level.  Folded into **blocks 09/10**; the record moved to
`archive/work/issue-151-host-path-regression/` (with a `RESOLUTION.md`).  See `WORKLOG.md` 261015.006.

**Promoted (r67, 2026-10-15).**  `block09-cache-ring-arming/` left `wip/`: all three r63..r66 host-expert
regressions (the cache completeness gate on an uneven split, the pageable `-lm none` weight source, and
the carved-but-unarmed H2D ring) are fixed and folded into blocks 09/10/11, together with a fourth,
larger regression the ring work surfaced -- the r66 MUL_MAT_ID helper had regressed the prefill by
~20-23 % (fixed in block 10 `mmid.cu`).  Released as `v12-097cf33d9-r67`; the record moved to
`archive/work/block09-cache-ring-arming/` (with a `RESOLUTION.md`).  The one residual (IQ4_NL field
`--host-experts pool` decode) is parked as TODO #65.  See `WORKLOG.md` 261015.005.

**Closed as refuted (2026-10-15).**  `moe-layer-stagger/` (TODO #63) — with the page-cache confound removed, four disjoint host-layer sets (`0..23`, even, odd, `24..47`) are identical to 0.1 % at `-ncmoe 24`; partial-offload prefill is H2D-bound, so the upload schedule cannot matter.  The wished-for speedup is already the delivery's tail-set host residency (r63 rear-fill: `-ncmoe N` hosts the last N trunk layers).  The record moved to `archive/work/moe-layer-stagger/` (with a `RESOLUTION.md`); the `wip/` entry is deleted with no stub.  Nothing was promoted; no release change.

**Promoted (r65, 2026-10-15).**  Five contributor PRs from `@briansp2020` (#154, #155, #156, #157,
#159) were merged and folded into their owner blocks (06/07/08/09/10; the delivery stays a 12-block
set), then archived with a
`RESOLUTION.md` each under `archive/work/`: `draft-sched2/` (second scheduler for the MTP draft's output
graphs, `LLAMA_DRAFT_SCHED2=0`), `output-pinned/` (pinned output buffer under `-sm tensor`,
`LLAMA_OUTPUT_PINNED=0`), `rdna3-mmvq-vecdot-alu/` (Q4_K/Q6_K mmvq vecdot instruction reductions),
`rdna3-mmb-tile-spill/` (Q4_K-only 128x256 MMB dense tile on gfx1100) and `rdna3-mmvq-q4k-rpb/` (4 rows
per block for a Q4_K verify batch on gfx1100).  None was ever listed in this index (they arrived as
PRs, not as `wip/` seeds).  See `WORKLOG.md` 261015.002 (r65).

**Promoted (r63, 2026-10-14).**  `moe-upload-overlap/` left `wip/`: the campaign's premise (make the
host-expert H2D upload overlap compute) was refuted, but the pivot finding — `-ncmoe N` should host the
**last N** trunk layers, not the first N — is folded into **blocks 09 and 10** and shipped as
`v12-097cf33d9-r63` (real `llama_model_params.n_cpu_moe_from_end`; `LLAMA_HOST_EXPERTS_FROM_END=0` kill
switch).  The same release carries the block-09 cache/preflight fixes (universal preflight, robust
deferred arena sizing, whole-tensor Meta upload for `-lm none`) that fixed the tail-placement decode
collapse.  The record moved to `archive/work/moe-upload-overlap/` (with a `RESOLUTION.md`); the two
residuals (Gemma4 front/tail near-tie drift, the `ggml_backend_meta_set_tensor_async` segmented mid-row
fast path) were investigated and resolved in r64 (see below).

**Promoted / closed (r64, 2026-10-15).**  `gemma4-drift-and-meta-async/` (the r63 residuals) is moved
to `archive/work/gemma4-drift-and-meta-async/` (`FINDINGS-2026-10-15.md`): the Gemma4 front/tail drift
is a proven logits near-tie (accepted as MoE drift, no code), the segmented `set_tensor_async` write is
folded into **block 09**, and the CUDA splice-gather ring ownership bug into **block 10**.  The external
`qwen4exp-mtmd-mtp-assert` candidate (PR #160) is folded into **block 10** (the PR was merged to
`main` for authorship) and its record archived at `archive/work/qwen4exp-mtmd-mtp-assert/`.  Released as
`v12-097cf33d9-r64`.

**Promoted (r62, 2026-10-14).**  `meta-split-copy-splice-race/` (seeded and fixed the same day) was
folded into **block 10** and released as `v12-097cf33d9-r62`: the CUDA compact splice-gather issued its
1-D H2D on `copy_stream()`, but the scheduler's `wait_before_overwrite()` (which guards the graph
allocator's reuse of the split-input buffer across splits) is on the **main** stream, so the H2D raced
the previous split's read and corrupted the `-sm tensor` + `-ncmoe` below-gate path (garbage/NaN on
35B-A3B, gemma4 26B-A4B, Flash-Next GSQ).  One line: gather H2D on the main stream.  The record moved to
`archive/work/meta-split-copy-splice-race/` (with a `RESOLUTION.md`); the residual is the pre-existing
upstream `MUL_MAT_ID` `f16 dup=1` fails/spin already tracked in `MANIFESTS.md`.

**Promoted (r60, 2026-10-14).**  `moe-heatmap-persistence/` left `wip/`: the persistent cross-run expert
heat map (device-side routing tally + on-disk sidecar + seed-at-sizing + heat-ordered layer placement)
was folded into **block 09** and is **default-on** (`MOE_EXPERT_HEATMAP=0` is the kill switch).  Measured
on `MOE_EXPERT_CACHE_MIB=4096`: **+34 % decode over a 48-token turn** (38.15 -> 51.30 t/s, hit 0.767 ->
0.940, fills 11748 -> 2976) and **+10.3 % over 256 tokens** (48.53 -> 53.55 t/s); prefill unchanged.
Gates: same-seed coherence byte-identical, prefill-logit KLD 0.000707 / 98.755 %, MTP `none == n3`,
`MUL_MAT_ID` no regression, `validate-set.sh` green.  The record moved to
`archive/work/moe-heatmap-persistence/` (with a `RESOLUTION.md`); the residuals
(`MOE_EXPERT_CACHE_PREFILL_SEED` repair, the decode-identity tally gap, the dead `moe_cache_observe`)
are in `TODO.md`.  See `WORKLOG.md` 261014.005.

**Resolved (r59, 2026-10-14).**  Issue #139 (`@briansp2020`, the internal all-reduce spin timeout) was
fixed in **block 01**: the spin budget is now wall-clock time (`wall_clock64()` sized by
`hipDeviceAttributeWallClockRate`, because the `clock64()` counter steps backwards on gfx1201 while a
queue is stopped), `GGML_CUDA_AR_SPIN_TIMEOUT_MS` (default 2000 ms) is a soft deadline that reports a
late peer and keeps waiting, and `GGML_CUDA_AR_SPIN_HARD_MS` (default 20000 ms) alone poisons and falls
back to the butterfly.  A same-call butterfly re-run was measured at -20-25 % decode and rejected.  The
record moved to `archive/work/issue-139-ar-spin-timeout/` with a `RESOLUTION.md`; the memory-pressure
half stays in `wip/host-expert-master-drop/`.  See `WORKLOG.md` 261014.004.

**Resolved (r58, 2026-10-14).**  Contributor PR #150 (`@overdoingism`, `dflash-layer-inp-preset`) was
merged and folded into the blocks that own each concern, then archived with a `RESOLUTION.md` under
`archive/work/dflash-layer-inp-preset/`: the draft-setup half reads the draft GGUF's
`<arch>.target_layers` and presets them before the target context exists (**block 02**), and the
context half adds the preset API/storage, applies it before the first reserve and skips a no-op
`set_embeddings_layer_inp()` reserve (**block 11**).  A DFlash draft no longer frees and reallocates the
target's compute buffers at the first request (5 -> 4 reserves).  See `WORKLOG.md` 261014.003.

**Resolved (r56, 2026-10-13).**  Four contributor PRs (#146-#149) were merged and folded into the
blocks that own each concern, then archived with a `RESOLUTION.md` each under `archive/work/`:
`rdna4-mmq-select-j/` (block 08), `pageable-bounce/` (block 09), `hc-mix-bf16-16/` (block 10) and
`state-save-merged-reads/` (block 11).  None was ever listed in the table below (they arrived as PRs, not as `wip/`
seeds).  See `WORKLOG.md` 261013.007.

**Resolved (2026-10-13, no code change).**  `moe-topk-fusion/` (TODO item 5(a)) was investigated and
closed: the fused `topk_moe` router already fires at every prefill width on r55 — on gfx1201 (3x R9700)
**and** gfx1151 (Strix Halo, the discovery's own device) — so the campaign's ~188 MB logits-pin unlock
is a no-op and the ~0.5-1.2 % prefill is already in the tree.  Nothing was promoted and no release was
cut.  The record moved to `archive/work/moe-topk-fusion/` with a `RESOLUTION.md`; TODO #5(a) is removed.
See `WORKLOG.md` 261013.006.

**Resolved (r55, 2026-10-13).**  Three campaigns left `wip/`, none of them as a new block -- the
changes were folded into the blocks that own the code (see `WORKLOG.md` 261013.005):
`wshadow-cleanup/` (all 139 `src/` `-Wshadow` sites fixed and `-Wshadow` enabled on the `llama` C++
target; 138 upstream sites -> block 00, the one feature-introduced site -> block 05; two real upstream
bugs found) and `mmvq-rpb-retune/` (`q5_1` rows-per-block now width-dependent -> block 08; `iq4_xs`
closed as a no-win) moved to `archive/work/` with a `RESOLUTION.md` each; `mtp-3gpu-decode/` was
**closed with no code change** (the r38 3-GPU MTP deficit is gone on r54: MTP 74.3 vs plain 55.2 t/s),
its record also moved to `archive/work/`.  TODO #15, #28 and #51 are closed.

> **Reopened (2026-10-16).**  `mtp-3gpu-decode/` was **revived** from `archive/work/` as
> [`three-gpu-performance/`](three-gpu-performance/README.md) after a fresh measurement found 3-GPU
> decode *slower* than 2-GPU (56.9 vs 84.4 t/s) at 100 % residency.  The r55 closure is kept as
> `three-gpu-performance/RESOLUTION-2026-10-13-closed.md`; TODO #51 is live again.


**Resolved (r54, 2026-10-13).**  Three campaigns left `wip/`, all with a `RESOLUTION.md` under
`archive/work/`: `dflash-meta-split/` (root-caused; a clean `-sm layer` rejection folded into **block
11**, the real fix re-seeded as `wip/dflash-meta-split-impl/`), `meta-segmented-expert-upload/`
(fixed and folded into **block 09**; rejection lifted; perf follow-up re-seeded as
`wip/meta-segmented-upload-async/`) and `ssm-alpha-beta-fusion/` (re-measured and **closed below the
adoption bar**).  The four-campaign cleanup handover moved to
`archive/docs/HANDOVER-2026-10-13-wip-cleanup.md`.  See `WORKLOG.md` 261013.004 (r54).

**TODO #29 is closed and needs no campaign.** The address-selected `ROPE -> VIEW -> SET_ROWS` fusion was made
bit-transparent in **r25** (`#pragma clang fp contract(off)` in `ggml/src/ggml-cuda/rope.cu`): the W=1 hash is now
the same with and without `GGML_CUDA_DISABLE_ROPE_SET_ROWS`, and issue #67's residual cross-start flip was
localized to hipBLASLt solution selection (external, ROCm/rocm-libraries#12126). The stale `TODO.md` #29 entry was
removed 2026-10-11 (see `GREEDY-PURITY.md` §41 and `WORKLOG.md` 261011.015).

**Promoted (r49, 2026-10-11).**  Three campaigns were folded into the delivery and their records moved to
`archive/work/`: `gdn-conv-device-idiom/` and `q1-0-byte-perm/` into **block 15**, and
`host-memory-reclaim/` into **block 06** (TODO #32 and #54 closed).  The reclaim's fix differs from its
plan: the loader now unmaps each tensor's mapping right after its device upload instead of the moot
steady-state `madvise` (measured: 122.5 -> 55.4 GiB peak on the 100 GB Flash-Next IQ4_NL).  Its
remaining `--host-experts pool` half is re-seeded as the new
[`host-expert-master-drop/`](host-expert-master-drop/README.md).  Record: `WORKLOG.md` 261011.017 (r49).

**Closed on seeding (2026-10-11).** Two campaigns seeded earlier the same day were closed at the maintainer's
call and removed from `wip/`: `sched-cross-backend-events/` (TODO #33 — dropped; the reporter files PRs often and
would have raised it if it still mattered) and `lfru-host-weight-migration/` (delivered/integrated in a different
form) plus `rdna-boosts-all-patch-hygiene/` (already resolved). The last two have a resolution record under
`archive/work/<name>/`; the first was dropped outright. `TODO.md` #33 and the parked LFRU / patch-hygiene entries
are removed.

**Note on the `PARKED` rows.** These are open TODO items that the maintainer has explicitly marked "not planned
now". They are seeded here as scoping notes (per the `TODO.md` header, which counts parked ideas as still open)
so a cold session can pick one up without re-deriving it — they are **not** active work.

**Recently archived (2026-10-11 wip audit).**  `mmvq-verify-rows/` (integrated in r21 as PR #57 —
block-10 + block-13; only the `q5_1`/`iq4_xs` retune remains, now `mmvq-rpb-retune/`),
`cache-split-admission/` (superseded — r35 folded the pool-aware device policy and r40 removed the whole
`-sm tensor` device-policy/remap family, so `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT` no longer exists), and
`host-memory-footprint/` (partially overtaken by r33/r35; remainder re-seeded as `host-memory-reclaim/`, promoted in r49 -- record now `archive/work/host-memory-reclaim/`).
All three now live under `archive/work/`.

**Closed as refuted (2026-10-11).**  `strata-amd-kernels/` (TODO #35) — K0 (six-candidate source diff +
K1 measurement) found nothing to port: `#262` is already delivered, the router / carveout /
IQ-grid-staging / verify items are no-deltas, and the one survivor (arena THP) was measured and refuted.
The record moved to `archive/work/strata-amd-kernels/` (with a resolution banner) and the `wip/` entry was
deleted with no stub; its lone residual was re-seeded as `q1-0-byte-perm/` (promoted in r49 -- record now
`archive/work/q1-0-byte-perm/`).

**Promoted (r53, 2026-10-13).** Three campaigns left `wip/`/`beta/`:
`dflash-multiseq/` (PR #144) was folded into the delivery (blocks 00/10/11) and `fa-stage-prealloc/`
(issue #138, from `beta/`) into blocks 09/10; their records now live under
`archive/work/dflash-multiseq/` and `archive/work/fa-stage-prealloc/`. The upstream `MUL_MAT_ID`
duplicate-ids fix was completed and published as `upstream/UPSTREAM-PR-mul-mat-id-dup.{md,patch}`
(the delivery ships pure on it), with its record under `archive/work/upstream-mul-mat-id-dup/`.
See `WORKLOG.md` 261013.003 (r53).
