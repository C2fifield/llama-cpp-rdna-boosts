# issue-139-ar-spin-timeout — the internal all-reduce spin budget (wall clock + a soft deadline that keeps waiting)

**Resolved (r59, 2026-10-14).**  Folded into the delivery as **block 01**
(`ggml/src/ggml-cuda/allreduce-hip.cu`).  Reporter: `@briansp2020`
(<https://github.com/stew675/llama-cpp-rdna-boosts/issues/139>).  See
[`RESOLUTION.md`](RESOLUTION.md).

## The report

On 2x R9700 (gfx1201), Qwen3.8-Flash-Next IQ3_S, `-sm tensor -ncmoe 48`, MTP n-max 3, roughly 1-5 in
100 fresh-start runs produced a different greedy output, and every run with a flip logged

```
internal AR: peer arrival not observed within 20 ms (call N); falling back to butterfly AllReduce to re-sync devices
```

The reporter also instrumented the timed-out wait and found the clear bug: `clock64()` on gfx1201 reads
`HW_REG_SHADER_CYCLES_LO/HI` and can go **backwards** while the amdgpu driver stops the process's GPU
queues to migrate/reclaim a GPU-mapped host page (`evict_process_queues_cpsch`; p50 5 ms, p99 178 ms,
max 517 ms).  One wait read `1089107279 -> 1075226717` over 1.6 ms of wall time, so
`(uint64_t)(clock64() - wait_t0)` wrapped to ~1.8e19 and passed any budget: a slightly-late peer was
treated as lost.

## The two root causes

1. **`clock64()` step-back (the clear bug).**  The spin budget was a `clock64()` cycle count, so a
   step-back wrapped it and false-timed-out.
2. **The correctness gap.**  A timed-out call returns with its local shard; the poison flag was only
   read at the *next* allreduce, so the timed-out step was consumed unreduced (0/1191 draft acceptance,
   a wrong greedy sha).

The queue stops themselves (memory pressure) are a separate concern, tracked in
`wip/host-expert-master-drop/` (TODO #55) and not part of this fix.

## The fix (block 01, r59)

- The budget is wall-clock time: `wall_clock64()` (`s_sendmsg_rtn GET_REALTIME`, a constant-rate counter,
  100 MHz on gfx12) and `GGML_CUDA_AR_SPIN_TIMEOUT_MS * hipDeviceAttributeWallClockRate` (kHz, fallback
  100000).  The `AR_PROFILE` fields use the same clock and rate.
- `GGML_CUDA_AR_SPIN_TIMEOUT_MS` (default 20 ms -> **2000 ms**) is a **soft** deadline: exceeding it
  reports a late peer once and **keeps waiting**, so the reduction still completes.  A new
  `GGML_CUDA_AR_SPIN_HARD_MS` (default **20000 ms**) alone sets the poison and falls back to the meta
  butterfly, preserving the bound that keeps an unbounded spin from making the queue non-removable.

## The rejected alternative

The same-call butterfly re-run (observe the poison in the current call, return `false`) was implemented
and measured: it is correct but costs about **20-25 %** of decode (4B `tg128` 118 -> 89 t/s, 27B `tg64`
43 -> 31-34 t/s) because `AR_PROFILE` counts 4096+ internal ARs over a ~69-step run, roughly 59 small
ARs per token, and a host sync per call removes the host's run-ahead.  It was dropped in favour of the
device-only soft/hard wait, which matches B1-off exactly.
