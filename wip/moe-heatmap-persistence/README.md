# `moe-heatmap-persistence` — remember the hot expert set across runs and pre-populate the arena

**Status: OPEN (seeded 2026-10-15), scoping + findings.**  Not part of the delivery; nothing here may be
applied to the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`).
Seeded from a code review of the host-resident MoE expert system
(`ggml/src/ggml-cuda/moe-expert-cache.{h,cu}`, `ggml/src/ggml-cuda/ggml-cuda-slab.cu`,
`ggml/src/ggml-backend.cpp`, `tools/server/server-context.cpp`).

This is the campaign for the maintainer's Q1: *"what could be done to remember the set of hot MoE
experts in the VRAM cache from run to run … an ordered list of hot experts … to pre-populate the expert
cache arena in order of hotness?"*

## 0. TL;DR

The cache already has the *machinery* for a hot-set prior — an LRU/LFRU policy with per-expert counters
and a one-shot bulk-admission path (`apply_prefill_seed_locked`) that loads ranked experts as
**provisional** slots (evictable until first hit).  What it does **not** have is a prior to feed it:

1. The only long-horizon ranking source (`prefill_count`, the prompt-routing tally) is **dead in the
   default configuration** — two independent reasons, both measured below.
2. Nothing is persisted: `count`/`ghost`/`last` are per-process and deliberately decayed every
   `MOE_EXPERT_CACHE_PERIOD` (32) steps, so they describe the last few hundred tokens, not a workload.
3. The arena is sized from free VRAM at a point where the prompt's routing is **not yet known** (a
   server warmup decode, or the first decode after the prefill), so even a repaired in-run tally would
   be too late to seed the *initial* sizing.  The seed has to come from a previous run.

So the smallest useful change is: keep a long-horizon per-expert heat accumulator, persist it to a file
keyed by the model geometry, and load it as the `apply_prefill_seed_locked` ranking source at sizing.
A dedicated manager thread is optional (a low-priority write at decode-idle or process exit is enough);
the feed is the existing per-op `access_locked` path, which is already under `g_mutex`.

## 0.1 Decisions (maintainer, 2026-10-15)

1. **On-disk heat map, default-on when the slab is active.**  Acceptable and desirable.  Enabled by
   default when the MoE expert cache is running against host-expert tables (the movable-boundary slab is
   the practical proxy for that state); inert otherwise.  `MOE_EXPERT_HEATMAP=0` is the kill-switch.
2. **Location: a sidecar in the model directory, with a writable fallback.**  `<first-shard>.moe-heatmap`;
   if the model directory is not writable (a read-only container mount), fall back to a well-defined
   cache directory (see the Q1a decision).
3. **Keyed per geometry** (the geometry is stored in the file and mismatches are rejected by matching,
   not trusted from the filename).
4. **Failures are non-fatal.**  Any create/update failure warns/errors; after the soft-fails persist,
   persistence deactivates for the session (the run continues with no prior and no further writes).
5. **Staged (atomic) writes.**  Write a new object, then move it over the old one (tmp + fsync +
   `rename`), so a failed write cannot leave a partial file.
6. **Debounced ~60 s.**  No more than one flush per 60 s, plus a final flush at a safe boundary.
7. **Phase 1 of the arena work is the deliverable** (hotness-ordered placement); the relocate/repack
   phases are parked for a later session.

## 0.2 Does persistence depend on the prefill tally? (Q2)

**Yes, in a stronger sense than first framed — the maintainer's instinct is right: repair the tally
first, and make the device-side tally (R2) the map's data source.**

The reason is a second dead seam beyond F1 (now F7): the **identity fast path**
(`moe_cache_take_over`, `moe-expert-cache.cu:3645`) serves a fully-resident table **without calling
`access_locked`** — that is its whole purpose (no per-token routing readback).  So the LFRU/eager-path
counters record heat **only when the arena is constrained enough to take the eager path**.  Measured:

| run | arena | `takeover` (eager ops) | `expert access: hits` |
|---|---|---:|---:|
| run1, 35B-A3B auto | fully resident (identity, 256/256 slots) | **1** | **16** |
| run2, 35B-A3B `MOE_EXPERT_CACHE_MIB=4096` | partial (161-184 slots) | 30840 | 226923 |

A map persisted from run1 would be **all zeros** — useless if the same model is later run with a tighter
budget.  So "build the map from the decode counters" only works in the regime that already needs the
map; it cannot seed a *change* of regime, and a fully-resident run teaches nothing.

The durable source is therefore a **device-side per-expert tally (R2)** that runs regardless of which
path took the op over (identity, eager, staged prefill, host prefill).  That also makes the map
independent of `MOE_EXPERT_CACHE_MIB`, `-ub` and the staging gate, and it is the same substrate the
maintainer described as "a stream of hit rates".

Cost note: a naive per-op tally kernel is one extra launch per `MUL_MAT_ID` op (~120/token on a 41-layer
MoE).  Prefer folding the increment into an existing router/kernel (or a per-split post-op launch that
covers all of the split's MoE ops) so the decode loop gains no new launch; that is part of the R2 design
work.

**Revised sequencing:** (1) the device-side tally (R2) — single data source, also repairs the host-pool
warm order (F6); (2) persist it and load it as the arena-seed/placement prior; (3) the first-run
in-run seed (`apply_prefill_seed_locked` with the decoupled post-sizing pass, R1-as-policy) becomes
optional polish once (1)+(2) exist.  The heat-map *format/lifecycle* (§2.2) can be built in parallel
with R2; only the feed depends on it.

### Why not just the LFRU counters (for the record)

The `count`/`ghost` vectors are decayed every 32 steps and are eviction state, not a workload prior;
using them would (a) miss every identity table and (b) be short-horizon.  Keep them for eviction;
feed the persisted `heat` from the device tally.

## 1. Findings (evidence)

### F1 — the in-run prefill seed is inert, for two reasons

`MOE_EXPERT_CACHE_PREFILL_SEED` defaults **on** (`moe-expert-cache.cu:518`), and the header/README
present it as the mechanism that warms the arena from "the prompt's hottest experts before the first
decode token".  It does not fire.

* **F1a — sizing happens first (server / warmup).**  The tally is gated on `!g_sized`
  (`moe-expert-cache.cu:3308`), but `alloc_all_locked()` is triggered by *any* decode-band pass
  (`n_tok <= 8`, `moe-expert-cache.cu:3402`), and llama-server runs a **warmup decode during load**.
  Instrumented run (35B-A3B UD-Q4_K_M, `-sm layer -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`,
  `-ub 2048`, 5256-token prompt, 3x R9700): the sizing log fires at `0.06.76 s`, and all **152**
  prefill-tally entries carry `sized=1`.  The tally is never accumulated → `prefill_count` stays empty →
  `apply_prefill_seed_locked` returns at its `prefill_count.size() != n_experts` guard
  (`moe-expert-cache.cu:1420`) → no `prefill-seeded` log, ever.
* **F1b — the default staging path bypasses the hook (llama-cli / no warmup).**  Above the calibrated
  staging width gate (`sched_stage_min_tokens`, measured **1542 tokens at 14.5 GB/s** on this box) the
  scheduler stages the whole host-weight table in the ring / backend `stage_input` and returns before
  the input loop reaches `moe_cache_update` (`ggml-backend.cpp:2640` `return;` in the staged branch;
  `:2538` `stage_consumed`).  A 2048-token prefill ubatch is always above the gate, so the tally branch
  is unreachable even in a run with no warmup.

Measured A/B: `MOE_EXPERT_CACHE_MIB=4096` with staging on (`run2`) and with `GGML_SCHED_STAGE=0`
(`run3`) give **identical** end state — `h=0.9060` (226923/250464), `fills=19452`, `evictions=504`,
`cold_reaches=4089`.  Disabling staging does not repair the seed; nothing changes.

The source comment at `moe-expert-cache.cu:3304` already suspects the staging half ("the tally is inert
there - see the campaign README").  This campaign is that README.  The sizing-order half (F1a) is new.

### F2 — no persistence, and the live counters are short-horizon by design

`table_t` (`moe-expert-cache.cu:98-108`) holds `count` (uses while resident), `ghost` (demand while
non-resident) and `last`, all decayed by `decay_locked` (`:580`): `count /= 2` and `ghost /= 2` every
`g_period` decode steps (`MOE_EXPERT_CACHE_PERIOD`, default 32).  These are correct for LFRU eviction
and useless as a cross-run workload prior — after ~32 steps the history is halved, after a few hundred
it is gone.

There is **no file I/O** for expert state anywhere in the module.  The only long-horizon accumulator is
`prefill_count` (a monotone tally), and it is only read once, by `apply_prefill_seed_locked`, and only
while `!g_sized`.

### F3 — the "live hit-rate" seam is dead code

`moe_cache_observe()` (`moe-expert-cache.cu:3046`) is defined and the header (`:42`, `:202`) says "the
`mul_mat_id` dispatch already calls `moe_cache_observe()` (live hit-rate measurement)" — but a
`grep` of `ggml/src` finds **no call site**.  It is the only per-routing device-feed entry point and it
is unreachable.  Whatever feeds a heat map must either revive this or use the scheduler-side
`moe_cache_update_host` path.

### F4 — the server already has a per-turn seam, but only aggregate counters

`tools/server/server-context.cpp:737` snapshots `llama_moe_cache_stats()` before/after each turn and logs
`h`, hits, misses, arena MiB.  `moe_cache_get_stats` (`moe-expert-cache.cu:2295`) aggregates only
`hits`/`misses`/`arena_bytes`; there is no per-expert readout.  This is the natural place to trigger a
heat-map dump at turn end (or at slot idle), and it already exists on the hot path, so it does not add a
new thread to the server.

### F5 — the arena's *table placement* is also hotness-blind (folded out to campaign 2)

Nothing here ranks **tables**; campaign [`../moe-arena-defrag/`](../moe-arena-defrag/README.md) covers
the slab placement / boundary-move side.  The two share the heat accumulator, so they should be designed
together even though they can ship separately.

### F6 — the dead tally also silently degrades the host-pool warm order

`prefill_count` has three consumers, and all three are affected:

| consumer | site | state |
|---|---|---|
| arena prefill seed | `apply_prefill_seed_locked` (`:1414`) | inert: `prefill_count` is empty (F1) |
| host-pool prewarm order | `pool_prepopulate_locked` (`:1085`) | inert: falls back to **expert-id order** |
| host-pool re-rank from the prompt | `pool_rerank_from_tally_locked` (`:1156`) | **never called** (dead function) |

So a `--host-experts pool` run DIO/buffered-fills its pinned pool in id order (a sequential warm), not
in the prompt's hot order — the exact thing the prewarm comment at `:1082` says it is for.  Repairing
the tally therefore has value **beyond** the arena seed, and it is the reason the pool prewarm cannot be
A/B'd meaningfully today.

### F7 — the identity fast path records no per-expert heat at all

`moe_cache_take_over` (`moe-expert-cache.cu:3645`) serves a fully-resident (`identity`) table directly and
returns before the scheduler ever calls `moe_cache_update`, so `access_locked` — the only place the
per-expert counters move — is **never reached** for such a table.  Measured on the fully-resident run1:
`takeover=1`, `expert access: hits=16` for the whole run; on the partial run2: `takeover=30840`,
`hits=226923`.  So any map built from the LFRU/eager counters is populated **only** when the arena is
already constrained, and is **all zeros** for a fully-resident run.  This is why the device-side tally
(R2) must be the persistence source (see §0.2), and it is also why the current `moe_cache_report`
"expert access" line reads empty on a big-arena run.

## 2. Design sketch

### 2.1 The accumulator and its feed

Add a **non-decayed, saturating** per-table counter next to the LFRU counters, e.g.:

```cpp
std::vector<uint32_t> heat;   // [n_experts] long-window use count (persistence source)
```

* **Feed it from the device-side tally (R2), not from `access_locked`.**  F7 shows the eager path is
  skipped for identity tables, so `access_locked` cannot be the source.  The tally kernel increments a
  per-`(table, expert)` device counter for every `MUL_MAT_ID` the backend runs (identity, eager, staged
  prefill), and the host `heat` vector is updated by a periodic readback (at the debounce/idle flush)
  rather than per op — this also keeps the hot path free of host work.
* A device counter per `(table, expert)` is `n_tables × n_experts × 4 B` ≈ 123 KB — negligible, and it
  can live in the same slab arena as the remap (campaign 3).
* Saturate (`min(heat+1, UINT32_MAX)`) so a long-lived server does not overflow and a very old run does
  not dominate forever.
* Optionally apply a slow decay (e.g. `>>=1` every `256` decode steps via the existing `last_decay`
  clock) so a workload shift is followed over minutes, not seconds.  This is a tuning knob; the default
  should be "long".
* Under `-sm tensor` a layer's tables live on several devices; the heat map is per `(layer, role)` so it
  is device-agnostic.  Keep one accumulator per table and later merge by `(layer, sem_role)`.

Note the **per-table expert-size heterogeneity** (see campaign 2 §F1): a persisted entry must carry
`expert_bytes` and the quant type so a re-keyed/re-quantised model is not loaded onto the wrong actor.

### 2.2 The on-disk format and lifecycle (decided)

**Path.**  A sidecar next to the model: `<first-shard-path>.moe-heatmap` (e.g.
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.moe-heatmap`).  The loader already hands the backend the GGUF path for
the host pool (`moe_cache_set_host_source`, `src/llama-model-loader.cpp:1968`); mirror that with a new
device-iface hook `moe_cache_set_heatmap_source(model_path, ...)` so the CUDA module never hard-codes a
location.  `MOE_EXPERT_HEATMAP_DIR` overrides the directory (for a read-only model mount); an explicit
`MOE_EXPERT_HEATMAP=0` disables, `=save`/`=load` give a one-way A/B.

**Key (geometry, not runtime).**  The file header stores the model identity and every table's geometry;
a table's row is matched at registration time by
`(layer, sem_role, n_experts, host_bytes, ggml_type)`.  The header identity is
`(gguf stem, total shard bytes, n_tensors, per-table geometry digest)` — cheap, no 100 GB data hash.  The
geometry is the **host** geometry (device-independent), so the same map serves `-sm layer`, `-sm tensor`
and any `-ncmoe`.  A re-quantised model with byte-identical per-expert geometry would collide; that is
accepted because a wrong heat vector is harmless (provisional slots, §2.4), and the digest makes it very
unlikely.

**Body.**  `magic MHT1`, u32 version, u32 n_tables, header identity; per table a fixed-size record
`{ i32 layer; u8 sem_role[64]; u32 n_experts; u64 host_bytes; u32 ggml_type; u32 heat[n_experts]; }`.
~130 KB for a 120-table / 256-expert model.  Little-endian; a version/geometry mismatch is a clean
"ignore, start fresh" (never a fatal).

**Accumulator and merge.**  Per table keep a `std::vector<uint32_t> heat` next to the LFRU counters,
incremented in `access_locked()` on both the hit and miss paths (i.e. `count` + `ghost` observations),
saturating.  On load, blend the persisted vector with the session vector (a configurable decay, default
target: a long EWMA so a workload shift is followed over minutes, not seconds) and write the blend.
Raw `uint32_t` at 4 B × 256 × 120 = 123 KB — trivial.

**Write policy (staged + debounced).**

* Mark dirty on increment (cheap, under `g_mutex`).
* A **low-priority flush** (reuse the `pool_bg_worker` pattern, `moe-expert-cache.cu:969`, already
  `setpriority(..., 19)`): snapshot the heat under `g_mutex`, do all file I/O **unlocked**.
* At most every **60 s** (`MOE_EXPERT_HEATMAP_DEBOUNCE_S`), plus a final synchronous flush at a safe
  boundary (process exit, and the existing server decode-idle seam at `server-context.cpp:737`).
* **Atomic:** write `<path>.tmp.<pid>`, `fsync`, `rename` over `<path>`, then `fsync` the directory.
  A failure leaves the previous file intact.
* **Fail-soft:** warn/error on any failure and increment a consecutive-failure count; after
  `MOE_EXPERT_HEATMAP_MAX_FAILS` (default 3) the session **deactivates persistence**
  (one final warning; the run continues with the in-memory map, no further writes).  Load failure (no
  file, unreadable, corrupt, geometry mismatch) is not a "failure" — it is a cold start with one INFO.
* **No writes from the hottest path.**  The snapshot is O(tables × n_experts) (~100 K reads) and only
  happens on the debounce/idle boundary.

### 2.3 The device-side tally (R2) — the feed, and the F1/F7 repair

R2 is now the **primary** work item, not a follow-up: it is the map's data source (§2.1) and it repairs
the prefill seed (F1) and the identity gap (F7) at once.  Shape:

* Launch a per-`(table, expert)` counter increment for the `MUL_MAT_ID` the backend runs, from the CUDA
  `mul_mat_id` dispatch.  `moe_cache_observe()` was designed for this and is dead; revive it (adapted to
  operate on device data) rather than inventing a second seam.
* **Cover every path**: identity take-over, the eager host-routing path, the staged prefill, and the
  host prefill copy.  Because it reads the routing where it already lives (device, `src[2]`), the width
  gate and the staging branch stop mattering.
* **Prefer no new decode launch**: fold the increment into an existing kernel where possible, or issue
  one post-op launch per split covering its MoE ops.  A naive per-op launch is ~120 extra launches/token
  on a 41-layer MoE and must be measured.
* Publish the counter to the host `heat` vector at the flush boundary, not per op.

R1 (`apply_prefill_seed_locked` used as a *post-sizing* re-seed policy, so a mid-run arena fix-up can
seed from this run's own tally) becomes optional polish once R2 exists.  On its own R1 fixes nothing
when staging is on.

R2 is the durable fix; it also repairs the host-pool prewarm (F6).  R1 remains useful as the
post-sizing re-seed policy once R2 exists.

### 2.4 Loading the prior at sizing

`apply_prefill_seed_locked` (`moe-expert-cache.cu:1414`) already does everything needed; only two
changes:

1. **Ranking source:** if a persisted heat vector exists for the table, rank by it; otherwise fall back
   to `prefill_count` (when repaired) or expert-id order.
2. **Timing:** call it from `alloc_table_locked` as today (it is a one-shot at sizing) — the persisted
   ranking is available *before* the warmup decode, so F1a is moot for it.

Because the seed path already marks entries **provisional** (`t.slot_prov[i] = 1`,
`moe-expert-cache.cu:1476`) and the admission rule treats a never-hit provisional entry as evictable
(`MOE_EXPERT_CACHE_PROVISIONAL=1`, default on), a stale or wrong heat map degrades to "an empty arena
plus some bonus hits" — the exact property the session-14 measurement established.  That makes a
persisted prior low-risk by construction, which is the strongest argument for doing it.

### 2.5 "Manager thread"?

The maintainer's phrasing ("a small MoE expert manager thread … fed a stream of hit rates … record this
information to a file") describes a design where the cache has its own writer.  The recommendation:

* **No new hot-path thread.**  The counters are already updated under `g_mutex` in `access_locked`; a
  second consumer of the same state needs no separate feed.
* A **low-priority writer** (like the existing `pool_bg_worker`, `moe-expert-cache.cu:969`, which is
  already `setpriority(PRIO_PROCESS, 0, 19)`) *is* reasonable for the flush: it takes a snapshot under
  `g_mutex` and does the file I/O off the decode path.  The existing worker is a fine model (leaked,
  process-lifetime, condition-variable-driven).
* Trigger the flush at **decode idle** (the server's "all slots are idle" / turn end) rather than a
  timer, so a busy server never pays disk I/O.

### 2.6 R2 prototype — implemented and validated (2026-10-15)

Patch: [`r2-device-tally.patch`](r2-device-tally.patch) (3 files, +219 lines; `ggml-cuda.cu`,
`moe-expert-cache.{cu,h}`).  Env-gated **OFF** as a WIP prototype: `MOE_EXPERT_HEATMAP=1` (the
promotion flips this to default-on when the cache is active on host experts).

**What it does.**

* A per-table `std::vector<uint32_t> heat` (saturating, non-decayed) plus a lazy device mirror
  `heat_dev` (one raw `cudaMalloc` per table, ~1 KiB — deliberately NOT via `ggml_cuda_device_malloc`,
  so a 1 KiB alloc can never trigger the arena yield ladder).
* **Host feed (decode band):** `moe_cache_update_host` tallies `heat` from the true host routing it
  already iterates.  Free, exact, no extra launch.
* **Device feed (prefill band):** `moe_cache_heat_observe` (declared in the header, implemented in the
  module) launches `moe_cache_heat_kernel` from the CUDA `mul_mat_id` dispatch.  It reads the raw
  device ids and `atomicAdd`s per expert.
* **Readback:** `heat_pull_locked` folds every device mirror into `heat` and zeroes it (at the report,
  later at a persistence flush); `free_heat_dev_locked` does the pull+free on stand-down / migration /
  re-size so the tally survives a boundary move.
* **Report:** `moe_cache_report` prints the total, the nonzero/total expert count, and the global top-5
  `(layer, role, expert)`, so a run shows whether the prior is populated and how peaked it is.

**The enabling invariant (why the outer entry is always the true routing).**  The cache's slot-remap
redirect is a *local* shallow-copy + re-entry inside `ggml_cuda_mul_mat_id`; the recursive call passes a
stack `src0` with `buffer == nullptr` and `dst->src[2]` set to the remap.  So the **outer** entry always
carries the graph's true raw ids, for every path — identity, eager, staged prefill.  The tally is
launched at the outer entry only (`src0->buffer != nullptr`), and the two feeds are band-disjoint
(host = decode, device = prefill) so nothing is double-counted and the decode latency path gains no
launch.

**Measured (35B-A3B UD-Q4_K_M, `-sm layer -ncmoe 40`, `-ub 2048`, 5256-token prompt).**

| run | arena | `takeover` | `expert access hits` | heat total | heat nonzero | heat top |
|---|---|---:|---:|---:|---:|---|
| `MOE_EXPERT_CACHE_MIB=4096` | partial | 7800 | 52968 | 4 982 016 | 28398/30720 | L38 e209 = 4571 |
| auto | fully resident (identity) | **1** | **16** | 4 915 888 | 27873/29952 | L38 e209 = 4554 |

The second row is the **F7 gap closed**: before R2 a fully-resident run recorded nothing (`hits=16`);
now it yields a full map from the device prefill tally alone.  The two rows nearly agree (L38 e209 4571
vs 4554), i.e. **the prefill routing is an excellent proxy for the decode hot set** on this workload —
which also justifies making the prefill feed the primary one.

**Cost.**  Prefill `1049.99` vs `1049.23` t/s (`MIB=4096`), auto decode `75.77` vs `75.68` t/s — no
measurable cost.  The device kernel is a few launches per prefill ubatch; the decode band adds none.

**Known gaps / next.**

1. The **decode identity** path still contributes no decode heat (it is exactly the fully-resident case,
   which needs no map *for that run*; the prefill feed covers portability).  If wanted, a sampled
   device tally on the decode band (every N tokens) closes it; measure the launch cost first.
2. **Persistence** (the file, geometry key, atomic write, 60 s debounce, read-only fallback) is §2.2 and
   is the next increment; the accumulator is ready for it.
3. `moe_cache_get_stats` does not expose the heat; the per-turn server seam (`server-context.cpp:737`)
   is the natural flush trigger for the debounced write.

### 2.7 Persistence — implemented and validated (2026-10-15)

Patch: [`heatmap-persistence.patch`](heatmap-persistence.patch) (**R2 + persistence**, +592 lines over
the three files; it is the canonical prototype — [`r2-device-tally.patch`](r2-device-tally.patch) is
the earlier R2-only snapshot and is a subset).  Env: `MOE_EXPERT_HEATMAP=1` (the WIP gate).

**What it does.**

* **Sidecar path:** `<gguf>.moe-heatmap`, derived in `moe_cache_set_host_source` from the GGUF path the
  loader already passes for the host pool — no new iface/loader plumbing.  If the model directory is
  not writable (probe = create+unlink a `.probe`), fall back to `MOE_EXPERT_HEATMAP_DIR`, else
  `$XDG_CACHE_HOME/llama-moe-heatmap/<stem>.moe-heatmap` (`$HOME/.cache` then `/tmp`).
* **Format:** `MHT1` + u32 version + u32 n_tables, then fixed-size records
  `{i32 layer; char role[64]; u32 n_experts; u64 host_bytes; u32 heat[n_experts]}`.  A table is matched
  at registration by `(layer, sem_role, n_experts, host_bytes)`; a mismatch is skipped (a cold table),
  never fatal.  ~150 KB for a 120-table / 256-expert model.
* **Atomic staged write:** `<path>.tmp.<pid>` → `fsync` → `rename` over `<path>` (the old file survives a
  failed write).  A failure increments a counter; after `MOE_EXPERT_HEATMAP_MAX_FAILS` (default 3)
  persistence is **disabled for the session** with one final warning.
* **Debounce:** `MOE_EXPERT_HEATMAP_DEBOUNCE_S` (default 60 s), checked at the server's per-turn
  `moe_cache_get_stats` seam; `moe_cache_report` (process exit) is the mandatory backstop.
* **Merge:** at save, the session heat is added into the loaded prior (saturating), so the map is a long
  EWMA across runs rather than last-run-wins.
* **Seed:** `apply_heat_seed_locked(t)` runs at sizing *before* the (inert) prefill seed and bulk-admits
  the top-`slots` experts by persisted heat as **provisional** slots (count 0, so they yield to a real
  hot expert immediately).  `apply_heat_seed_locked` is silent about identity tables and no-ops when no
  prior exists.
* **Placement (campaign 2 phase 1):** `alloc_all_locked` now allocates layers in **descending heat**
  order, so the hottest layers sit highest in VA and are the last the work-pool boundary reaches.
  Frozen at sizing; the order falls back to ascending layer when no heat is present.

**Measured (35B-A3B UD-Q4_K_M, `-sm layer -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`, 5256-token prompt,
same seed).**

| turn length | run | hit rate | fills | colds | decode t/s |
|---|---|---:|---:|---:|---:|
| 48 tok | cold (no map) | 0.7674 | 11748 | 63 | **38.15** |
| 48 tok | warm (seeded) | **0.9402** | **2976** | 63 | **51.30** (+34 %) |
| 256 tok | cold (no map) | 0.9060 | 19452 | 4089 | **48.53** |
| 256 tok | warm (seeded) | **0.9561** | **6912** | 4089 | **53.55** (+10.3 %) |

The gain is the cold-start transient: the warm arena begins with the previous run's hot set resident,
so the first tokens are hits instead of fills (fills down 4× over 48 tokens, 2.8× over 256).  Prefill is
unchanged (1050 vs 1059 t/s) and the seed cost lands in the warmup/sizing phase, not the measured turn.

This is the first observable end-to-end win from the campaign, and it is the direct answer to the
original question ("stop evicting hot experts at start-up"): the arena no longer starts cold.

## 3. Hypotheses / work to do

1. **H1 (first-turn win).**  With a persisted heat map, a cold server start on a repeated workload
   raises the hit rate over the first N decode tokens versus today.  Metric: a new
   "first-T-token hit rate" counter (snapshot `moe_cache_get_stats` at token 1/8/32/128 and log the
   delta), plus `fills` and H2D MiB over the first turn.  Current baseline: first turn
   `h=0.9060`, `fills=19452`, `55.6 MiB/token` (MIB=4096, 35B-A3B, 257 tokens).
2. **H2 (prefill-reclaim survival).**  With the hot tables pre-seeded (and campaign 2's placement), a
   wide-prefill boundary move evicts fewer *hot* tables.  Metric: log per-table resident bytes before
   and after a `moe_cache_evict_slab_range`, weighted by heat.
3. **H3 (no harm).**  A wrong/stale heat map is no worse than today (the provisional mechanism), and a
   correct one is a strict improvement.  Test by loading a deliberately shuffled map and diffing the
   same-seed output (must be byte-identical: placement must not change arithmetic) and the hit curve.
4. **H4 (workload generality).**  Measure the persistence on the adaptive-MTP field config
   (`-ncmoe 99` + auto cache, `none == n1 == n3 == n7` purity) and on a second, dissimilar prompt, to
   confirm the map is not overfit to one prompt's routing.

## 4. Resolved and remaining questions

**Resolved by the maintainer (2026-10-15):** on-disk is acceptable and default-on when the slab is
active; sidecar in the model directory; keyed per geometry; non-fatal failures with session
deactivation after repeated soft-fails; staged writes; 60 s debounce; phase 1 (placement) is the arena
deliverable.

**Still open:**

* **Q1a (RESOLVED 2026-10-15).**  Read-only model directories (the delivery's container case) **fall back
  to a well-defined writable directory** rather than deactivating.  Chosen: try the model-dir sidecar
  first; if it cannot be opened for write, fall back to `$MOE_EXPERT_HEATMAP_DIR` if set, else
  `$XDG_CACHE_HOME/llama-moe-heatmap/<geometry-digest>` (`$HOME/.cache/...` when `XDG_CACHE_HOME` is
  unset).  Log one INFO/WARN naming the chosen path.  The geometry is in the filename so a cache dir can
  hold many models.  If *both* locations fail, that is a real soft-fail and the session deactivates after
  the failure count.  A read-only model dir is a stable property, so it must not be treated as a
  soft-fail (else every container user loses persistence silently).
* **Q1b — merge vs last-run.**  Should the loaded prior be blended with the session heat (long EWMA, so a
  workload shift is followed) or replaced (last run wins)?  **Recommendation: blend**, with the decay a
  knob; the flagging question is the default half-life.
* **Q1c — flush trigger.**  Strictly the 60 s debounce, or also the server decode-idle seam?  The
  maintainer asked for debounce; a final flush at `atexit` is mandatory (a 60 s window must not lose the
  last run).  **Proposal:** debounce + `atexit` + optional idle.
* **Q1d — scope split (REVISED 2026-10-15).**  The heat map, the device-side tally (R2) and the placement
  (campaign 2) are one dependency chain: **R2 is the feed**, the map is the storage, placement is the
  consumer.  **Recommendation: R2 first**, then the map, then placement.  The map's format/lifecycle can
  be developed in parallel with R2 (only the feed depends on it).

## 5. Gates

* Same-seed coherence (current recorded hash) on the field MoE model — the prior is a *cache placement*
  change, so the output must be **byte-identical** for the same prompt and seed.
* `scripts/gate-prefill-logits.sh` — the seed path touches fills, not prefill kernels, so KLD should be
  0/identical; run it anyway because the seed can change which experts are read from host vs VRAM.
* Adaptive-MTP gate (`benchmarks/mtp-adaptive-methodology.md`), field config `none == n1 == n3 == n7`
  (short) / `none == n3` (long) on 35B-A3B `-ncmoe 99` + auto cache.
* A new "cold-start" metric (H1) recorded in `benchmarks/`.
* Greedy purity (`GREEDY-PURITY.md`): a placement change must not move any arithmetic; assert the
  8-type x 5-length grid is unchanged.

## 6. Pointers

* `ggml/src/ggml-cuda/moe-expert-cache.cu` — `table_t`, `access_locked`, `decay_locked`,
  `apply_prefill_seed_locked`, `alloc_all_locked`, `moe_cache_update_host`, `moe_cache_observe`,
  `moe_cache_get_stats`, `pool_bg_worker`.
* `ggml/src/ggml-cuda/moe-expert-cache.h` — the env contract and the (stale) `moe_cache_observe` claim.
* `ggml/src/ggml-backend.cpp` — `sched_stage_issue`, `ggml_backend_sched_copy_input` (the staging /
  host-path split and the `moe_cache_update` call site).
* `tools/server/server-context.cpp:737` — the per-turn stats seam.
* `ENVIRONMENT.md` §1.2.1 — `MOE_EXPERT_CACHE_*`, `MOE_HOST_POOL_*`.
* `archive/work/moe-expert-cache/` — the feature's own history (session 14 = the seed measurement).
* Campaigns: [`../moe-arena-defrag/`](../moe-arena-defrag/README.md) (table placement, shares the heat
  accumulator), [`../host-expert-master-drop/`](../host-expert-master-drop/README.md) (the pool tier the
  heat map could pre-warm too).
