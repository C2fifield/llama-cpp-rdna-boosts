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
* **Warmup sizing works but over-commits.**  The warmup only ever *primes* each table (`!t.primed`
  returns before the sizing check), so `alloc_all_locked` is never called and the arena cannot be sized at
  load.  Making the priming pass also call `alloc_all_locked` *does* size the arena at load
  (`registered == expected` reaches 100 % in ~0.2 s, arena 15107 MiB, 36.9 %), but a subsequent long
  prefill then **OOMs in HIP graph capture** because at warmup the slab `extend_all` reclaims the
  model's unused reserve and the empty-context arena takes it.  The deferred sizing avoids this by
  running after the wide layout/KV exist.  See §6 — this needs a reserve/boundary rule before it can
  ship.

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

Artifacts in this directory: `instrumentation.patch`, `keep-survivors.patch`, `warmup-sizing.patch`
(the full size-at-load prototype: priming sizing, skip-extend, the ring+wide reservation and the shared
slab layout helper), `server-prof.log` (the first measurement run), `warmup-arena-at-load.log`,
`warmup-sizing-oom.log`, `warmup-size-at-load-noextend.log` and `warmup-size-at-load-ring-wide.log`
(the final working run).

## 5. Open items

1. **Make the warmup latch the sizing safely** (the actual first-token lever): the primary sizing must
   not take the VRAM the widest prefill will need.  The deferred path gets that for free by sizing
   after the wide layout has existed and been dropped; the warmup path must reproduce it (e.g. size
   against a reserved wide-layout boundary, or suppress `ggml_cuda_slab_extend_all` for the priming
   sizing so the arena cannot consume the reserve).  Repro + OOM trace: §6.
2. If (1) is not viable, seed the arena **asynchronously** so the first decode overlaps the H2D.
3. Re-cut `keep-survivors.patch` A/B on a config where the re-arm actually fires.

## 6. Warmup sizing prototype — what worked and how it failed

The diagnosis (instrumented `moe_cache_tables_complete_locked`, `warmup-sizing.patch`):

* During the warmup every host table is seen exactly once.  `moe_cache_table`'s `!t.primed` branch sets
  `t.primed` and **returns before** the `!g_sized` sizing check, so the warmup can never size.  The first
  real prefill re-sights the tables but is gated off (`n_tok > 8`); the first decode then sizes.
* The completeness gate itself is **not** the blocker: the instrumented warmup reached
  `registered=42912972800 expected=42912972800 tables=144 (100.0%)` (i.e. all 48×3 host tables).
* Consequence: the arena size depends on the **first request** — a 5-token request sized it to
  11750 MiB, the 13.8K request to 5355 MiB (the prompt's KV/wide layout occupies VRAM before sizing).

Prototype: call `alloc_all_locked()` from the priming branch too
(`MOE_EXPERT_CACHE_SIZE_AT_PRIME=1`).  Result:

* **Good:** the arena is sized at load, before `listening` — `alloc_all_locked: ... arena 15107.1 MiB
  (36.9%)` at t≈0.09 s, and the first request logs **no** `alloc_all_locked`/seed at all.
* **Bad:** the next long prefill+decode **aborts with `ROCm error: out of memory` in
  `ggml_cuda_graph_evaluate_and_capture`**.  At warmup the context is empty, so
  `ggml_cuda_slab_extend_all()` grows the slab into the reserve the model "did not need" and the arena
  takes 15 GiB; the wide prefill/decode cannot then be placed.  Raising
  `MOE_EXPERT_CACHE_RESERVE_MIB` does **not** help — `extend_all` reclaims it (`warmup-sizing-oom.log`).

So the prototype proves the mechanism but needs a matching reserve/boundary rule before it can ship
(§5 item 1).  The likely shape: for the priming sizing, either reserve the widest work layout up front or
skip `slab_extend_all` so the arena is sized against a boundary the later wide view cannot exceed.

## 7. Skip `slab_extend_all` at the priming sizing — works

The first fix attempt at the boundary rule: when the sizing is the priming one
(`g_alloc_priming`), do **not** call `ggml_cuda_slab_extend_all()` — the extension is what maps the reserve
at the empty-context moment.  Result (`warmup-size-at-load-noextend.log`):

* arena sized at load to **11110.5 MiB** (27.1 %), **zero `slab grew`** lines;
* the long 13.8K-token request **succeeds** (prompt 13040 ms, 200 tok, no abort), zero request-path
  `alloc_all_locked`;
* the wide prefill still evicts 5810 MiB (arena → 5361.8 MiB) and `moe_cache_rearm` rebuilds 12 tables
  cleanly.

**Corrected diagnosis of the OOM.**  It was *not* the prefill being allocated outside the slab — the
prefill work is placed in the slab (`WORK_ALLOC ret_off=5760`, boundary → 7808).  Two separate things:

1. `alloc_table_locked: arena alloc failed … cache disabled for this table` is **fail-soft and normal
   (the success run has MORE of them: 863 vs 535)** — it is the cache declining tables that no longer
   fit a fragmented arena, and the layer falls back to the host path.
2. The terminal abort is **`hipGraphInstantiate`** (`ggml-cuda.cu:4410`) — a HIP-runtime allocation for
   the executable graph, outside the slab's control.  It failed because `slab_extend` had over-mapped the
   slab to 16.81 GiB (only 4.02 GiB free on the card); with extend skipped ~19 GiB is free and it
   succeeds.

**Residual:** the arena is still sized above the wide view (11110 MiB) and the first wide prefill evicts
5810 MiB of it.  That is the design's moveable boundary doing its job (and heat-ordered, so the reclaimed
tables are the coldest), but it is exactly the "filled past the reclaim point" the maintainer flagged — to
avoid it entirely the priming sizing must first reserve `ring + wide` (raise the slab work boundary to the
wide layout) and then size the arena above it (`warmup-sizing.patch` is the first half; the boundary
reservation is the next step).

## 8. Reserve ring + wide at load (the ordering fix) — works

Extracts the ring+wide layout into ONE helper (`ggml_cuda_slab_ring_layout`) and adds a
`ggml_cuda_slab_reserve_wide` hook the context calls after `sched_reserve` (the widest compute size is
`backend_buf_exp_size`).  The reservation is stored until the narrow floor exists, then realised by the
cache immediately before it sizes (`ggml_cuda_slab_realize_wide`) and kept as a persistent `wide_floor` so
later boundary shrinks cannot reclaim it.  `work_alloc` uses the same helper, so the setup phase and the
real wide placement can never disagree.

Result (`warmup-size-at-load-ring-wide.log`): arena **5355 MiB at load** (was 11110); the first wide
prefill's boundary is already at `7552 = narrow_floor 1536 + ring 4224 + wide 1792`, so it evicts only
**84 MiB** (was 5810); no OOM; decode **37.1 t/s vs 29.2**.  This is the boundary ordering the maintainer
asked for: the wide region is reserved before the arena is sized, so the arena is never filled past the
reclaim point.

Gotcha (found and fixed): the first version deadlocked — `reserve_wide` held `g_slab_mutex` and called a
`realize_wide` that relocked the same non-recursive mutex.  The fix is a `_locked` inner helper shared by
the public entry point and `reserve_wide`.

## 9. Multi-GPU `-sm tensor` — works (2 and 3 GPUs)

The hook is per host-expert device: the context loops `model.moe_host_expert_bytes` (the real devices
under `-sm tensor`, not the Meta wrapper), each device reserves its own ring+wide, and the cache sizes
each arena above it.  Same 13.8K-token request:

| | 1 GPU | 2 GPUs (`0,1`) | 3 GPUs (`0,1,2`) |
|---|---|---|---|
| ring region (per device) | 4.12 GiB | 2.12 GiB | 1.44 GiB |
| arena at load | 5355 MiB (13 %) | 23340 MiB (57 %) | 40925 MiB (100 %) |
| first-prefill eviction | 84 MiB | 246 MiB | 124 MiB |
| decode | 37.1 t/s | 84.4 t/s | 56.9 t/s |
| OOM | none | none | none |

Every device logged the reservation, none over-committed, and the only request-path evictions are the
chunk-rounding remainder (84-246 MiB total).  On 3 GPUs the whole host-expert set fits, so the arena is
100 % resident and the identity path fills it at load (off the request path).  Logs:
`warmup-size-at-load-2gpu-tensor.log`, `warmup-size-at-load-3gpu-tensor.log`.
