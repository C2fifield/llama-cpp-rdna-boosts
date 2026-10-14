# RESOLUTION — issue #139 (internal all-reduce spin timeout)

**Status:** FOLDED into the delivery as **block 01** (`ggml/src/ggml-cuda/allreduce-hip.cu`), release
**`v12-097cf33d9-r59`** (2026-10-14).  **Reporter:** `@briansp2020`
(<https://github.com/stew675/llama-cpp-rdna-boosts/issues/139>).

## What was adopted

1. **Wall-clock spin budget.**  The kernel's `wait_t0` and the budget check use `wall_clock64()`, and
   the budget is `GGML_CUDA_AR_SPIN_TIMEOUT_MS * hipDeviceAttributeWallClockRate` (kHz, fallback
   100000) rather than `ms * 2e6` `clock64()` shader cycles.  On gfx1201 `clock64()` reads
   `HW_REG_SHADER_CYCLES_LO/HI` and steps backwards while the amdgpu driver stops the process's queues
   to migrate/reclaim a GPU-mapped host page (`evict_process_queues_cpsch`; reported p50 5 ms, p99
   178 ms, max 517 ms), so `(clock64() - wait_t0)` wrapped to ~2^64 and passed any budget: a
   slightly-late peer was treated as lost.  The `GGML_CUDA_AR_PROFILE` `prof_start` / `prof_spin`
   fields and their tick-to-us conversion use `wall_clock64()` and `hipDeviceAttributeWallClockRate`
   too.
2. **A soft timeout keeps waiting (B2).**  `GGML_CUDA_AR_SPIN_TIMEOUT_MS` (default raised 20 ms ->
   **2000 ms**) is a soft deadline.  A peer wait that exceeds it is a late peer from a queue stop, not
   a lost one: the kernel reports it once (a host-mapped `slow` flag, logged at the next call) and
   keeps waiting, then completes the normal phase-3 reduction, so no consumer ever sees a local shard.
3. **A hard cap remains the last resort.**  `GGML_CUDA_AR_SPIN_HARD_MS` (default **20000 ms**) sets the
   poison and falls back to the meta butterfly, preserving the bound that keeps an unbounded spin from
   making the compute queue non-removable (the MES `REMOVE_QUEUE` hazard).  `0` = never poison; a hard
   value below the soft one is clamped up.

## Why not the same-call butterfly re-run

The reporter's suggested (and my initially preferred) fix was to observe the poison in the **same**
call and re-run that tensor through the butterfly before returning.  It was implemented and measured:
correct, but it costs about **20-25 %** of decode because the internal pipeline runs 4096+ ARs over a
~69-step run (roughly 59 small ARs per token) and a host event sync per call removes the host's
run-ahead pipelining.

| model | same-call re-run | deferred (old) | B2 (shipped) |
|---|---:|---:|---:|
| 4B Q8_0 `tg128` (2 GPU, `-sm tensor`) | 89-90 t/s | 118 t/s | 117.6 t/s |
| 27B Q4_K_XL `tg64` (2 GPU, `-sm tensor`) | 31-34 t/s | 42-43 t/s | 43.0 t/s |

B2 never involves the host on the hot path, so it matches the old path exactly.

## Validation on the delivery tree (gfx1201, ROCm 7.14.1)

- `AR_PROFILE` (real config) shows the wall clock in use (`clock 100000 kHz`), spin p50 2.5 us, p99
  115 us, **max 1.3 ms** against the 2000 ms soft budget, and **0** soft/hard warnings.
- 4B `Qwen3.5-Q8_0` 3-GPU `-sm tensor` same-seed greedy **byte-identical** to r58.
- `llama-bench` 2-GPU `-sm tensor`: no regression (table above).
- `scripts/gate-prefill-logits.sh` **PASS** (mean KLD 0.000707, same-top-p 98.755 %, the exact r58
  values).
- **Forced soft-timeout test:** with the soft budget temporarily set to 1 tick and the check decimation
  to 1 (so every AR soft-times-out while the hard cap stays at 20 s), the 4B `-sm tensor` greedy
  continuation is **byte-identical** to the no-timeout reference, and the soft warnings fire.
- `scripts/validate-set.sh` strict 12/12, tree `59b9a7faeb2c975994fabdb2cfb6761a6beda560`;
  `git am --whitespace=error-all` clean.

## Not part of this fix

The queue stops themselves (host-memory pressure) are a separate deliverable, tracked in
`wip/host-expert-master-drop/` (TODO #55).  Kernel sysctl tuning to reduce reclaim pressure was offered
to the reporter in the issue response.
