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
| [`dflash-meta-split-impl/`](dflash-meta-split-impl/README.md) | the **real fix** behind the r54 clean rejection: a Meta distributed `TOP_K` (or a split→mirrored gather) so the DFlash2 selector's `ggml_top_k` can reduce a vocab-split input under `-sm tensor` (TODO #58 follow-up) | **OPEN (seeded 2026-10-13), scoping + design** |
| [`kv-cell-offset-nondeterminism/`](kv-cell-offset-nondeterminism/README.md) | **the unified-KV residual (TODO #52 part C)**: a sequence's attention result depends on the physical **set** of KV cells it occupies in the shared stream, not just its start offset; under `--kv-unified` the allocator lets that set depend on scheduling history, so greedy repetitions differ | **PARKED (2026-10-11)** — root cause pinned (cell-set vs the FA reduction/split grid; a fixed alignment is insufficient, and the grid is model/shape-dependent), but a proper fix is an upstream unified-KV redesign (per-sequence contiguous, tile-aligned regions). Documented as a gotcha (`-np > 1` + `--kv-unified` is not deterministic; use `--no-kv-unified`); issue #134 closed as upstream-inherent. Record: [`FINDINGS-offset-mechanism.md`](kv-cell-offset-nondeterminism/FINDINGS-offset-mechanism.md), probe + repro in `kv-cell-offset-nondeterminism/tools/` |
| [`rs-rollback-boundary/`](rs-rollback-boundary/README.md) | the recurrent rollback-boundary guard (issue #131): the global `last_ubatch_nseq_tokens` tracker reports the wrong sequence; the per-sequence guard is folded into block 02, but the **"large prefill as the last prepared batch before a rollback"** ordering and whether the restored snapshot is stale are open | **OPEN / ACTIVE (resuming)** — the dense 27B does not reproduce the warning on r43 (0/106 rollbacks) and `--no-kv-unified` does not change that; the MoE reproducer fires it and shows the rolling sequence's own last prepare is a 4571-token prefill. Open questions, reproducers and instrumentation: [`README.md`](rs-rollback-boundary/README.md) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`host-expert-master-drop/`](host-expert-master-drop/README.md) | remove the full pinned `ROCm_Host` master when `--host-experts pool` is on (the pool's Phase 3): every cold expert served from the bounded pool over the page cache | **OPEN (seeded 2026-10-11)** — the r49 reclaim removed the mmap half of the `-ncmoe` host peak, but `--host-experts pool` still keeps the full pinned master (measured `RssShmem` identical pinned vs pool: 18.7 GiB on 35B-A3B, 54.1 GiB on Flash-Next IQ4_NL).  Also the strongest lever for issue #139's `evict_process_queues` stops |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (the Strata shape) | OPEN / scoping (TODO #36) — **work in flight**: the note refutes the earlier "cannot be done" verdict for the interleaved shape and sets out the pipeline to try |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`fa-band-2byte-row/`](fa-band-2byte-row/README.md) | extend the RDNA4 GQA band's 64-wide K/V config row to the 2-byte f16/bf16 arm (`ncols == 16`; TODO #27) | **OPEN (seeded 2026-10-11), small tuning** |
| [`rdna35-mmvq-fusion-perf/`](rdna35-mmvq-fusion-perf/README.md) | restore the block-13 RDNA3_5 single-token dense-GLU fusion perf (~0.9 % `tg128`; TODO item 16) | **PARKED** — the original "pin the params" plan is dead; live candidates are codegen FMA contraction and the Q8_1 cache |
| [`meta-segmented-upload-async/`](meta-segmented-upload-async/README.md) | make the segmented per-op expert upload **async** again (per-row 1-D `set_async` + the MMQ tail) — r54 fixed it synchronously, so gemma4 + `-ncmoe` + `-sm tensor` is correct but only ~9 t/s | **OPEN (seeded 2026-10-13), perf follow-up** |
| [`ngram-mod-host-experts/`](ngram-mod-host-experts/README.md) | `ngram-mod` + `draft-mtp-adaptive` roughly halves decode when the MoE experts are host-resident (`-ncmoe`/`-cmoe`, expert cache active); plain `draft-mtp-adaptive` restores the expected rate | **OPEN (seeded 2026-10-14), scoping + measurement** |
| [`moe-arena-defrag/`](moe-arena-defrag/README.md) | hotness-aware slab **table placement** (the boundary move currently evicts the last-allocated tables regardless of heat) + heterogeneous per-table expert sizes + a phase-3 idle repack/relocate | **OPEN (seeded 2026-10-14); phase 1 (placement) SHIPPED in r60 as part of the heat map, phases 3-4 parked** |
| [`slab-transient-absorption/`](slab-transient-absorption/README.md) | move the controlled raw-`cudaMalloc` transients (FA staging, Q8_1, MMB shadows, AR scratch, per-table remap) inside the slab so `GGML_CUDA_SLAB_HEADROOM_MIB` can be safely lowered (today the headroom is permanently outside the slab and `slab_extend` is dead at the default) | **OPEN (seeded 2026-10-14); phase map in §3.0 + hybrid routing in §3.1 (wide→ring, narrow→narrow-2-style pin), measurement first** |
| [`issue-151-host-path-regression/`](issue-151-host-path-regression/README.md) | issue #151: dense decode with graphs **off** is 1.3–2.4 % slower on r56 than r30 on the reporter's slow Windows host (Ryzen 5 PRO 4650G).  Two candidate patches are now implemented and validated on r67: `GGML_CUDA_ASYNC_INPUT` default-on (the reporter's +1.4 %) and a per-graph GDN state-gather index replacing the `O(gathers × nodes)` walk (local **−0.63 ms host/graph, +0.24 % tg**, reporter +0.3 %). | **OPEN (seeded 2026-10-15, candidate fix validated 2026-10-15)** — does **not** reproduce on the maintainer's box as a version regression (GPU-bound: host is ~4.3 ms/graph vs the reporter's ~10), so the win is verified by host time and the reporter's box, not by local tg.  `=2` (index vs walk) clean on dense GDN / MoE / non-GDN / MTP / 2-GPU; prefill-logit gate PASS.  **Awaiting the promote / hold / beta decision** (§8 of the README). |

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
