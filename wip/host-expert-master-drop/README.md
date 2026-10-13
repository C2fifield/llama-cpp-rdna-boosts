# `host-expert-master-drop` — remove the full pinned `ROCm_Host` master when `--host-experts pool` is on

**Status: OPEN (seeded 2026-10-11), scoping.**  Not part of the delivery; nothing here may be applied to
the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).  Seeded from
the r49 `host-memory-reclaim` finding (campaign now in `archive/work/host-memory-reclaim/`) and from issue
#139 (@briansp2020).

## 0. Why this exists

The host-memory-reclaim work (r49, block 06) removed the mmap page-cache half of the `-ncmoe` host peak,
but it explicitly does **not** touch the pinned host expert master.  `--host-experts pool` was shipped in
r35 as the budgeted alternative to the full-master allocation, yet on r49 it still keeps the full pinned
master: `ggml/src/ggml-cuda/moe-expert-cache.cu` logs *"Phase 2b; full pinned master kept"* and the
`RssShmem` peak is identical with and without it.

Measured (r49):

| model (2x R9700) | config | peak `RssShmem` |
|---|---|---:|
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `-ncmoe 40`, default `pinned` | 18.7 GiB |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `-ncmoe 40 --host-experts pool` | 18.7 GiB |
| `Qwen3.8-Flash-Next-IQ4_NL` (100 GB) | `-ncmoe 40`, default `pinned` | 54.1 GiB |
| `Qwen3.8-Flash-Next-IQ4_NL` (100 GB) | `-ncmoe 40 --host-experts pool` | 54.1 GiB |

So the pool is an **option that adds a bounce tier** today, not a memory win.  Phase 3 (the "full pinned
master kept" TODO in the source) is to remove the master and serve every cold expert from the pool (over
the page cache, optionally `O_DIRECT`), with the existing scheduler fallback.

## 1. Why it matters beyond footprint

Issue #139 (Brian) shows the amdgpu driver **stops all of a process's GPU queues** whenever the kernel
migrates or reclaims a page of host memory that is mapped for the GPU (`hipHostMalloc` /
`hipHostRegister`, including mlocked).  bpftrace measured ~2.4 `evict_process_queues_cpsch`/s on his box,
p50 5 ms, p99 178 ms, max 517 ms; a stop over the all-reduce spin budget trips the timeout and leaves a
call unreduced.  A smaller pinned footprint means fewer pages the kernel can migrate, hence fewer queue
stops.  The r49 reclaim already reduces *overall* pressure (it gave back ~67 GiB of mmap on the 100 GB
model); dropping the master would remove the remaining ~54 GiB of GPU-mapped host memory for a pool-mode
run.  That is the strongest candidate for making pool mode genuinely small.

## 2. Current reality check (r49)

- **The master is load-bearing.**  The per-ubatch op-offload H2D upload and the expert-cache fill both
  read the pinned master in place; under `-sm tensor` a pageable master would land in the non-`is_host`
  `CPU_REPACK` (no device access, cache inert).  `--host-experts mmap` was removed in r33 (no-XNACK GPU,
  issue #116), so "read the master from the pageable mmap" is dead.
- **The pool already exists and is gated:** `MOE_HOST_POOL_MIB` (unset = `MOE_HOST_POOL_FRAC` 25 % of the
  MoE host expert bytes), `MOE_HOST_POOL_PREWARM=1`, `MOE_HOST_POOL_BGFETCH=1`, `MOE_HOST_POOL_DIO=0`,
  `MOE_HOST_POOL_PREFETCH=0`.  See `ENVIRONMENT.md` §1.2.1.
- **What the cold path does now:** a cache/op-offload miss serves the cold id by reading the pinned host
  slice in place (`g_cold_uva`, `MOE_EXPERT_CACHE_COLD_UVA=1`).  Removing the master forces every cold
  serve through the pool (fill-on-miss) or a fresh page-cache/DIO read.
- **The r35 record:** `archive/work/host-expert-dio-cache/README.md` (the promotion record) — read its
  Phase 3 section before starting.

## 3. Hypotheses / work to do

1. **Quantify the pool-only read cost.**  With the master dropped, every cold expert is a pool fill.  The
   pool is `MOE_HOST_POOL_FRAC` % of the host set — measure hit rate and the added cold-serve latency at
   25 % / explicit / 100 % on the 35B-A3B and Flash-Next IQ4_NL.  Does the fraction need to rise to keep
   decode/MTP flat?
2. **Prove the numerics.**  The host bytes are the same expert data, so the result must be bit-identical
   whether served from the master or the pool.  Gate: same-seed coherence, `--spec-type none ==
   draft-mtp` where the current field promises it, and the prefill-logit KLD gate (the cold path is not a
   prefill kernel, but the pool changes which code path dequantises).
3. **Scheduler fallback.**  The source says the master is also the scheduler fallback when a pool is not
   used.  Enumerate every reader of `cur->data` for a host expert tensor (`llama-model-loader.cpp`'s
   `moe_cache_set_host_source`, the op-offload upload, the cache fill) and make each pool-aware under
   `--host-experts pool`, leaving the default `pinned` path byte-identical.
4. **Issue #139 measurement.**  Re-run Brian's field config (2x R9700, Flash-Next IQ3_S, `-sm tensor
   -ncmoe 48`, MTP n-max 3) with the master dropped and count `evict_process_queues_cpsch` stops
   (bpftrace) and AR spin timeouts.  The goal is fewer stops, not just less RSS.

## 4. Gates

- `scripts/validate-set.sh` if promoted; coherence `1c5d32ac537d`; `scripts/gate-prefill-logits.sh`
  (KLD ≤ 0.005 / same-top-p ≥ 98 %); `MUL_MAT_ID` 931/931; `test-backend-ops -o MUL_MAT -b ROCm0`.
- Field MTP: 35B-A3B `-ncmoe 99` + auto cache `none == n1 == n3 == n7` (short) and `none == n3` (long),
  qwen4exp pure; `llama-batched-bench -npl 1,4,8` decode not regressed.
- gfx1100 (`fingon`) is the memory-pressure target; gfx1201 is the portability check.

## 5. Pointers

- r35 promotion record: [`../../archive/work/host-expert-dio-cache/`](../../archive/work/host-expert-dio-cache/README.md).
- r49 finding: [`../../archive/work/host-memory-reclaim/`](../../archive/work/host-memory-reclaim/README.md);
  `WORKLOG.md` 261011.017 (r49).
- `ENVIRONMENT.md` §1.2.1; `patches/README.md` block 06; issue #139.
- `ggml/src/ggml-cuda/moe-expert-cache.cu` (`g_pool_*`, `moe_host_pool_on`, the "full pinned master kept" log).

## r53 re-confirmation (2026-10-13, WIP-cleanup session)

Re-confirmed on the r54 parent tree (r53, canonical tip `4b27ac644`), 2x R9700, 35B-A3B UD-Q4_K_M
`-ncmoe 40`, `memwatch.sh` peak `RssShmem`:

| config | peak `RssShmem` |
|---|---:|
| default `pinned` | **18776 MiB** |
| `--host-experts pool` (MIB unset = 25 %) | **18776 MiB** (identical) |

The source still logs `Phase 2b; full pinned master kept`.  So the r49 finding holds on r53: the pool
is a bounce tier, not a memory win, and the full pinned master is the remaining footprint.  This
campaign was **deferred to a dedicated session** in the 2026-10-13 cleanup (the four-campaign
handover, `archive/docs/HANDOVER-2026-10-13-wip-cleanup.md`); the reader enumeration, hypotheses and
gates above are the starting point.
