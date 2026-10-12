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
| [`r50-rebase/`](r50-rebase/README.md) | **re-base the 16-block delivery onto upstream `097cf33d9`** (+195 commits from `a55e952b8`); absorbs the upstream MoE cache (#29887/#30112), the #29943 expert-copy refactor, the #29953 MMQ `J_best` rework and the qwen4exp/U0→U1 delta | **IN FLIGHT (2026-10-12)** — blocks 00–14 resolved on the `~/llama-rebase` worktree (backups `wip-r50-blocks-12/14`); block 15 has 24 mapped conflicts, no build yet; resolutions in [`README.md`](r50-rebase/README.md) |
| [`kv-cell-offset-nondeterminism/`](kv-cell-offset-nondeterminism/README.md) | **the unified-KV residual (TODO #52 part C)**: a sequence's attention result depends on the physical **set** of KV cells it occupies in the shared stream, not just its start offset; under `--kv-unified` the allocator lets that set depend on scheduling history, so greedy repetitions differ | **PARKED (2026-10-11)** — root cause pinned (cell-set vs the FA reduction/split grid; a fixed alignment is insufficient, and the grid is model/shape-dependent), but a proper fix is an upstream unified-KV redesign (per-sequence contiguous, tile-aligned regions). Documented as a gotcha (`-np > 1` + `--kv-unified` is not deterministic; use `--no-kv-unified`); issue #134 closed as upstream-inherent. Record: [`FINDINGS-offset-mechanism.md`](kv-cell-offset-nondeterminism/FINDINGS-offset-mechanism.md), probe + repro in `kv-cell-offset-nondeterminism/tools/` |
| [`rs-rollback-boundary/`](rs-rollback-boundary/README.md) | the recurrent rollback-boundary guard (issue #131): the global `last_ubatch_nseq_tokens` tracker reports the wrong sequence; the per-sequence guard is folded into block 02, but the **"large prefill as the last prepared batch before a rollback"** ordering and whether the restored snapshot is stale are open | **OPEN / ACTIVE (resuming)** — the dense 27B does not reproduce the warning on r43 (0/106 rollbacks) and `--no-kv-unified` does not change that; the MoE reproducer fires it and shows the rolling sequence's own last prepare is a 4571-token prefill. Open questions, reproducers and instrumentation: [`README.md`](rs-rollback-boundary/README.md) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`host-expert-master-drop/`](host-expert-master-drop/README.md) | remove the full pinned `ROCm_Host` master when `--host-experts pool` is on (the pool's Phase 3): every cold expert served from the bounded pool over the page cache | **OPEN (seeded 2026-10-11)** — the r49 reclaim removed the mmap half of the `-ncmoe` host peak, but `--host-experts pool` still keeps the full pinned master (measured `RssShmem` identical pinned vs pool: 18.7 GiB on 35B-A3B, 54.1 GiB on Flash-Next IQ4_NL).  Also the strongest lever for issue #139's `evict_process_queues` stops |
| [`mmvq-rpb-retune/`](mmvq-rpb-retune/README.md) | retune the RDNA4 mmvq `calc_rows_per_block_weight()` table for `q5_1` / `iq4_xs` on ROCm 7.14 — the small residual of the **integrated** PR #57 work | **OPEN (seeded 2026-10-11), small** — `-3..-4 %` at n=2/3 (both `+10..12 %` at n=8); candidate per-type n-dependent rows (TODO #28) |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (the Strata shape) | OPEN / scoping (TODO #36) — **work in flight**: the note refutes the earlier "cannot be done" verdict for the interleaved shape and sets out the pipeline to try |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`mtp-3gpu-decode/`](mtp-3gpu-decode/README.md) | 3-GPU MTP decode is slower than plain decode (TODO #51): quantify the current gap on r49, then profile the 3-device internal all-reduce (the fast paths are 2-device-only), the tensor-split verify gather, and graph recapture | **OPEN (seeded 2026-10-11)** — re-seeded from the archived `slab-ring-region` finding; has never had a live campaign |
| [`fa-band-2byte-row/`](fa-band-2byte-row/README.md) | extend the RDNA4 GQA band's 64-wide K/V config row to the 2-byte f16/bf16 arm (`ncols == 16`; TODO #27) | **OPEN (seeded 2026-10-11), small tuning** |
| [`rdna35-mmvq-fusion-perf/`](rdna35-mmvq-fusion-perf/README.md) | restore the block-13 RDNA3_5 single-token dense-GLU fusion perf (~0.9 % `tg128`; TODO item 16) | **PARKED** — the original "pin the params" plan is dead; live candidates are codegen FMA contraction and the Q8_1 cache |
| [`wshadow-cleanup/`](wshadow-cleanup/README.md) | enable `-Wshadow` for `src/` and fix the 46 "shadows a local variable" sites (TODO item 15) | **PARKED** — needs its own cleanup commit; touches ~20 upstream files |
| [`moe-topk-fusion/`](moe-topk-fusion/README.md) | adopt the fused `topk_moe` (TODO item 5(a)) — ~0.5 % prefill for ~188 MB pinned logits and a numerics fork | **PARKED** — blocked on a CPU-reference quality gate |
| [`ssm-alpha-beta-fusion/`](ssm-alpha-beta-fusion/README.md) | single-walk `ssm_alpha` + `ssm_beta` (TODO item 5(b)) — ~0.3-0.6 % prefill | **PARKED** — the two MUL_MATs are non-adjacent in the DFS graph; needs loader plumbing or a graph restructure |
| [`meta-segmented-expert-upload/`](meta-segmented-expert-upload/README.md) | finish the segmented host-resident expert upload so gemma4 + `-ncmoe` + `-sm tensor` stops being rejected (TODO, parked) | **PARKED** — a partial fix hung non-deterministically; needs a deterministic repro first |

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
