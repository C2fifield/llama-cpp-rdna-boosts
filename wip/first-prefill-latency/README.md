# `first-prefill-latency` — where the first request's "before prefill" and "after prefill" time goes

**Status: OPEN (seeded 2026-10-16), measured on the real server config.**  **Not part of the delivery**;
nothing here may be applied to the fork without the maintainer's go-ahead (WIP and promotion rules in
`AGENTS.md`).

## 0. TL;DR

Measured on the actual `Qwen3.8-Flash-Next` server (`GSQ-IQ3_XXS`, `--n-cpu-moe 48`,
`HIP_VISIBLE_DEVICES=2`, `-ub 4096`, `-c 131072`, adaptive MTP), with a small temporary phase-profiling
patch (`instrumentation.patch`, `RDNA_PROF=1`):

* **There is no idle "before the prefill" gap.**  The first prefill ubatch starts ~75 ms after
  `launch_slot_` and computes immediately.  What looks like a "6 s delay before prefill" is the first
  ubatch's *own* compute time, reported late: the server does not print `prompt processing` until an
  ubatch finishes.
* **The first-prefill ubatch is only ~5–17 % slower than steady state**, and only on the very first
  request (graph build / wide-layout reserve); request 2's first prefill ubatch is at steady rate.  It is
  not a recurring per-request cost in this config.
* **The "after the prefill" cost is the one-shot MoE arena sizing + heat/prefill seed**, which lands on
  the *first real decode* because the warmup does **not** manage to size the arena.  It is one-shot
  (`g_sized` latches), which is exactly why the second prefill+decode has no such delay.  It is the heat
  map being applied — the maintainer's original hypothesis was right.
* **The re-arm path never fired** in this workload: the only cache event in request 2 was an 84 MiB slab
  eviction with no `re-armed`/re-seed.  So the `keep-survivors.patch` prototype (below) is **inert here**
  and was not worth a before/after number in this config.

## 1. Measured timeline

Config: `llama-server` (build `r73` + instrumentation), model
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, `--n-cpu-moe 48`, `--ctx-size 131072`,
`--batch-size 4096 --ubatch-size 4096`, `--split-mode tensor`, draft-mtp-adaptive, single GPU.

| phase | request 1 (13 800 tok) | request 2 (75 600 tok) |
|---|---|---|
| load + warmup | warmup decode `tokens=2 ms=5.5`; **no `alloc_all_locked`** | — |
| first prefill ubatch | `tokens=4096` **3586 ms** (1142 t/s) | `tokens=4096` **3091 ms** (1325 t/s) |
| steady prefill ubatch | `4096` 3067 ms (1335 t/s) | `4096` ~3065 ms (1336 t/s) |
| prefill total | 13 431 ms / 13 800 (1027 t/s) | 61 492 ms / 75 600 (1229 t/s) |
| `[PROF] drop+rearm` | **11 ms** | not fired |
| `[PROF] alloc_all_locked` | **446 ms** (arena 5355 MiB, 13.1 %) | not fired |
| seed | 144 tables, 5355 MiB @ **~12.2 GB/s** | not fired |
| `[PROF] slab_evict` | — | 84 MiB, 0 ms, **no re-arm** |
| first decode ubatch | `tokens=4` **654 ms** | — (steady decode only) |
| steady decode ubatch | `tokens=4–7`, ~120–180 ms (MTP) | same |

**First-token latency, request 1 ≈ 11 + 446 + 654 ≈ 1.1 s**; request 2 ≈ first decode only (no seed).
The seed bandwidth (~12.2 GB/s) means the maintainer's larger 15 666 MiB arena seeds in ~1.3 s — the
rest of the "4 s" in their log is the ~143 tokens generated before the first `n_gen` progress print
(143 / 47 t/s ≈ 3.0 s), i.e. a reporting interval, not first-token latency.

## 2. Why the arena sizing lands on the first real decode

* `heat_map_open_locked` reads the sidecar at model load into `g_heat_loaded`; no arena exists yet.
* `alloc_all_locked` is gated to a decode-band pass (`n_tok <= 8`) **and** to
  `moe_cache_tables_complete_locked()` (total registered host bytes >= the preflight total).
* The warmup decode (`tokens=2`) enters the hook but the completeness check is false, so sizing defers;
  the first real request's prefill registers the remaining tables and the following decode latches it.
* Result: the whole arena is allocated and seeded synchronously (`cudaMemcpyAsync` + a
  `cudaDeviceSynchronize` per table) on the first generated token.

**The real lever is therefore to make the arena size + seed before the first user request** — i.e. get
the warmup (or an explicit load-time step) to satisfy the completeness check and latch the sizing, so
the seed is paid at load.  This is the open question to investigate next (why does the warmup not
register every host table? staged/gather path? the draft model? a geometry mismatch?).  It is far more
valuable than the re-arm micro-optimisation below.

## 3. `keep-survivors.patch` (the requested prototype)

`moe_cache_rearm` rebuilds a whole layer whenever the eviction boundary cut through it.  Today it frees
even the tables that survived (they lie outside the taken slab range, their bytes are valid, and their
slot count already equals the per-layer target) and then re-seeds them from `t.heat`.  The prototype
keeps those survivors and only re-allocates the stood-down tables:

* opt-in: `MOE_EXPERT_CACHE_REARM_KEEP_SURVIVORS=1`;
* if a survivor is not at the target, or a stood-down table cannot reach it, it falls through to the
  existing whole-layer rebuild (the shrink retry needs every table free);
* on success it prints `layer N re-armed X stood-down table(s), kept Y survivor(s) (no re-seed)`.

It is **inert in the measured config** because the re-arm never ran (the wide prefills did not stand
down any tables beyond an 84 MiB, apparently non-straddling, eviction).  A config that does fire it is a
tight arena plus a wide prefill (`MOE_EXPERT_CACHE_MIB` large, `--ctx-size` large, long prompt); that A/B
was not run.  If a later workload shows stood-down tables, this removes the redundant H2D of the cut
layer's survivors.

## 4. Reproduce

```bash
# 1. instrumentation (temporary; not delivery)
cd ~/llama.cpp && git apply <this-dir>/instrumentation.patch
cmake --build build-rocm-hybrid --target llama-server -j 16

# 2. real config, single GPU
RDNA_PROF=1 HIP_VISIBLE_DEVICES=2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 \
  build-rocm-hybrid/bin/llama-server --host 127.0.0.1 --port 8033 \
  --n-gpu-layers all --n-cpu-moe 48 --ctx-size 131072 \
  --batch-size 4096 --ubatch-size 4096 --split-mode tensor --lazy-mode off \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 7 --spec-draft-p-min 0.001 \
  --spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -m /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf

# 3. two long /completion requests, then inspect [PROF] lines
grep -ao '\[PROF\] [a-z_]*[^[]*ms=[0-9]*\.[0-9]' server.log
```

Artifacts in this directory: `instrumentation.patch`, `keep-survivors.patch`, `server-prof.log`
(the full request-1/2 run with the `[PROF]` lines).

## 5. Open items

1. **Make the warmup latch the sizing** (the actual first-token lever).  Instrument
   `moe_cache_tables_complete_locked` (registered vs expected at the end of the warmup) to find why the
   warmup is incomplete.
2. If (1) is not viable, seed the arena **asynchronously** so the first decode overlaps the H2D.
3. Re-cut `keep-survivors.patch` A/B on a config where the re-arm actually fires.
