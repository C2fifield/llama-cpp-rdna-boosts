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
| [`kv-cell-offset-nondeterminism/`](kv-cell-offset-nondeterminism/README.md) | **the unified-KV residual (TODO #52 part C)**: a sequence's attention result depends on the physical **set** of KV cells it occupies in the shared stream, not just its start offset; under `--kv-unified` the allocator lets that set depend on scheduling history, so greedy repetitions differ | **PARKED (2026-10-11)** — root cause pinned (cell-set vs the FA reduction/split grid; a fixed alignment is insufficient, and the grid is model/shape-dependent), but a proper fix is an upstream unified-KV redesign (per-sequence contiguous, tile-aligned regions). Documented as a gotcha (`-np > 1` + `--kv-unified` is not deterministic; use `--no-kv-unified`); issue #134 closed as upstream-inherent. Record: [`FINDINGS-offset-mechanism.md`](kv-cell-offset-nondeterminism/FINDINGS-offset-mechanism.md), probe + repro in `kv-cell-offset-nondeterminism/tools/` |
| [`rs-rollback-boundary/`](rs-rollback-boundary/README.md) | the recurrent rollback-boundary guard (issue #131): the global `last_ubatch_nseq_tokens` tracker reports the wrong sequence; the per-sequence guard is folded into block 02, but the **"large prefill as the last prepared batch before a rollback"** ordering and whether the restored snapshot is stale are open | **OPEN / ACTIVE (resuming)** — the dense 27B does not reproduce the warning on r43 (0/106 rollbacks) and `--no-kv-unified` does not change that; the MoE reproducer fires it and shows the rolling sequence's own last prepare is a 4571-token prefill. Open questions, reproducers and instrumentation: [`README.md`](rs-rollback-boundary/README.md) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`host-memory-reclaim/`](host-memory-reclaim/README.md) | host-memory footprint of GPU-resident weights (gfx1100 / `fingon`): reclaim the `RssFile` mmap pages of GPU-resident tensors, and re-measure the `-ncmoe` field now that the pageable master is gone and the bounded pinned pool exists | **OPEN (re-seeded 2026-10-11)** — the open remainder of the archived `host-memory-footprint/` after r33 (pageable `--host-experts mmap` removed) and r35 (`--host-experts pool` added); hypotheses 1/3/4 unmeasured |
| [`mmvq-rpb-retune/`](mmvq-rpb-retune/README.md) | retune the RDNA4 mmvq `calc_rows_per_block_weight()` table for `q5_1` / `iq4_xs` on ROCm 7.14 — the small residual of the **integrated** PR #57 work | **OPEN (seeded 2026-10-11), small** — `-3..-4 %` at n=2/3 (both `+10..12 %` at n=8); candidate per-type n-dependent rows (TODO #28) |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (the Strata shape) | OPEN / scoping (TODO #36) — **work in flight**: the note refutes the earlier "cannot be done" verdict for the interleaved shape and sets out the pipeline to try |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels against block-10/13/15 (TODO #35) | **scoping note — not yet started**: no implementation yet, just the candidate list and the comparison method |
| [`mtp-3gpu-decode/`](mtp-3gpu-decode/README.md) | 3-GPU MTP decode is 2.6x slower than plain decode (TODO #51): re-measure on r48, then profile the 3-device internal all-reduce (the fast paths are 2-device-only), the tensor-split verify gather, and graph recapture | **OPEN (seeded 2026-10-11)** — re-seeded from the archived `slab-ring-region` finding; has never had a live campaign |
| [`sched-cross-backend-events/`](sched-cross-backend-events/README.md) | PR #104 follow-ups (TODO #33): a foreign-event guard for the scheduler's first cross-backend `ggml_backend_event_wait`, per-change kill-switches, and the reporter-side FAIL→PASS for the alias guard | **OPEN (seeded 2026-10-11), non-blocking** — dropped follow-ups from the archived `2gpu-sched-fixes` |
| [`fa-band-2byte-row/`](fa-band-2byte-row/README.md) | extend the RDNA4 GQA band's 64-wide K/V config row to the 2-byte f16/bf16 arm (`ncols == 16`; TODO #27) | **OPEN (seeded 2026-10-11), small tuning** |
| [`gdn-conv-device-idiom/`](gdn-conv-device-idiom/README.md) | unify the device idiom in `gdn-conv.cu` vs `cpy-batch.cu` (TODO #32) | **OPEN (seeded 2026-10-11), cosmetic** — one-line fix at the next block-15 touch |
| [`rdna-boosts-all-patch-hygiene/`](rdna-boosts-all-patch-hygiene/README.md) | the net-patch artifact's git cost (3.19 MiB × 194 revisions; `.git` 96 MiB) — keep / untrack / rewrite | **PARKED** (TODO "not planned now") — a decision item, not a kernel campaign |
| [`rdna35-mmvq-fusion-perf/`](rdna35-mmvq-fusion-perf/README.md) | restore the block-13 RDNA3_5 single-token dense-GLU fusion perf (~0.9 % `tg128`; TODO item 16) | **PARKED** — the original "pin the params" plan is dead; live candidates are codegen FMA contraction and the Q8_1 cache |
| [`wshadow-cleanup/`](wshadow-cleanup/README.md) | enable `-Wshadow` for `src/` and fix the 46 "shadows a local variable" sites (TODO item 15) | **PARKED** — needs its own cleanup commit; touches ~20 upstream files |
| [`moe-topk-fusion/`](moe-topk-fusion/README.md) | adopt the fused `topk_moe` (TODO item 5(a)) — ~0.5 % prefill for ~188 MB pinned logits and a numerics fork | **PARKED** — blocked on a CPU-reference quality gate |
| [`ssm-alpha-beta-fusion/`](ssm-alpha-beta-fusion/README.md) | single-walk `ssm_alpha` + `ssm_beta` (TODO item 5(b)) — ~0.3-0.6 % prefill | **PARKED** — the two MUL_MATs are non-adjacent in the DFS graph; needs loader plumbing or a graph restructure |
| [`lfru-host-weight-migration/`](lfru-host-weight-migration/README.md) | LFRU host→GPU slow migration of hot expert weights (TODO, parked) | **PARKED** — most of the premise is now covered by `moe-expert-cache` + `--host-experts pool`; re-scope before coding, and do not duplicate `moe-cpu-overlap` |
| [`meta-segmented-expert-upload/`](meta-segmented-expert-upload/README.md) | finish the segmented host-resident expert upload so gemma4 + `-ncmoe` + `-sm tensor` stops being rejected (TODO, parked) | **PARKED** — a partial fix hung non-deterministically; needs a deterministic repro first |

**TODO #29 is resolved and needs no campaign.** The address-selected `ROPE -> VIEW -> SET_ROWS` fusion was made
bit-transparent in **r25** (`#pragma clang fp contract(off)` in `ggml/src/ggml-cuda/rope.cu`): the W=1 hash is now
the same with and without `GGML_CUDA_DISABLE_ROPE_SET_ROWS`, and issue #67's residual cross-start flip was
localized to hipBLASLt solution selection (external, ROCm/rocm-libraries#12126). The `TODO.md` #29 entry still
reads as open (it was written 2026-09-29, before the r25 fix) — **propose closing it**, see `GREEDY-PURITY.md` §41.

**Note on the `PARKED` rows.** These are open TODO items that the maintainer has explicitly marked "not planned
now". They are seeded here as scoping notes (per the `TODO.md` header, which counts parked ideas as still open)
so a cold session can pick one up without re-deriving it — they are **not** active work. If the maintainer
would rather they not appear here, remove the rows and keep the directories as archive-style records.

**Recently archived (2026-10-11 wip audit).**  `mmvq-verify-rows/` (integrated in r21 as PR #57 —
block-10 + block-13; only the `q5_1`/`iq4_xs` retune remains, now `mmvq-rpb-retune/`),
`cache-split-admission/` (superseded — r35 folded the pool-aware device policy and r40 removed the whole
`-sm tensor` device-policy/remap family, so `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT` no longer exists), and
`host-memory-footprint/` (partially overtaken by r33/r35; remainder re-seeded as `host-memory-reclaim/`).
All three now live under `archive/work/`.
