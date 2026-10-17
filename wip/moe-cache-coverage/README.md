# `moe-cache-coverage` — how much of the routing distribution the expert-cache arena can hold, and how well the admission policy uses it

*(Formerly `moe-arena-defrag`.  Renamed 2026-10-17: the "defrag" premise was refuted by measurement —
see [`FINDINGS-2026-10-17-coverage-vs-defrag.md`](FINDINGS-2026-10-17-coverage-vs-defrag.md).  The old
title and body below are kept as the campaign's own history; read §2.5 and the findings first.)*

**Status: OPEN, RETARGETED and RENAMED 2026-10-17 (maintainer-approved).  Phase 1 (heat-ordered
placement) shipped in r60; the P0 heat-map correctness fix shipped in r80; the "defrag" phases are
closed.**  Not part of the delivery; nothing here may be applied to the fork without the
maintainer's go-ahead (see `AGENTS.md`).  Seeded from a code review of the movable-boundary slab and the
expert-cache arena (`ggml/src/ggml-cuda/ggml-cuda-slab.cu`, `ggml/src/ggml-cuda/moe-expert-cache.cu`).

> **Read the 2026-10-17 findings before using anything below.**  Three results change the plan: (1) the
> MoE **heat map was broken** — a doubling per flush saturated it, so every heat-driven feature was
> ranking by expert id; **fixed in r80** (block 09), and the poisoned sidecars were deleted, so
> measurements taken before r80 against a heat-seeded or heat-ordered layout measured nothing;
> (2) the hit-rate gap against Strata is **accounting + arena size**, not fragmentation — equal-basis
> numbers are 91.9 % vs 90.6 % on the same model; (3) a repack at idle can recover at most ~0.5 % of
> arena *bytes*, because the arena is already ~100 % allocated in slots (the free VRAM is the slab's
> 8 GiB compute reserve, which is `../slab-transient-absorption/`).

**Maintainer decision (2026-10-14): phase 1 only for now.**  The relocate/repack (3/4) "sounds like it
may take multiple sessions to get right"; it is kept below as a scoping note, not as this session's work.

This is the campaign for the maintainer's Q2: *"not all experts are of equal size. How are we handling
the potential arena fragmentation that would occur today with the eviction + admission of experts over
time? At the end of a decode phase where all decodes are idle, should we do something like a hotness
ordered re-pack of the VRAM resident experts?"*

## 0. TL;DR

* The maintainer is **right that expert size varies** — but the variation is *per table*
  `(layer, role)`, not within a table.  It is a **quantization** effect: a mixed/UD quant promotes some
  layers, so their expert blobs are larger.  Verified on the box below.
* There is **no byte-level fragmentation inside a table**: a table's arena is one contiguous
  `slots * expert_bytes` block and eviction frees a logical slot, not bytes.  The real fragmentation is
  **between tables** in the slab's arena region.
* Today the slab is **(a) hotness-blind** and **(b) evicts whole tables**:
  `ggml_cuda_slab_arena_alloc` fills top-down, the boundary grows from the bottom, and
  `moe_cache_evict_slab_range` drops *every table that overlaps* the reclaimed range.  So the reclaim
  victims are set by **allocation order** (currently ascending layer order), not by hotness.
* The cheapest, highest-value fix is a **placement reorder** at sizing (allocate the hottest tables
  first, i.e. highest in VA and furthest from the boundary).  It needs no copy and no new slab API.
* A true "repack at idle" (relocate a table's arena and/or shrink its slot count so the boundary can
  move without dropping it) is a larger, phase-2 change: it moves a live arena base, which invalidates
  captured graphs and the kernel launch parameters that hold the arena address (`moe_cache_redirect_fused`).
  Scoped below; the design exists (`ggml_cuda_graph_mem_freed`) but the ordering hazards are real.

## 1. Findings (evidence)

### F1 — expert blobs are not the same size; the variation is per table

Measured from the GGUF (35B-A3B **UD**-Q4_K_M, `gguf` reader), bytes **per expert**:

| tensor | type | bytes/expert |
|---|---|---:|
| `ffn_gate_exps` (all layers) | Q4_K | 589 824 |
| `ffn_up_exps` (all layers) | Q4_K | 589 824 |
| `ffn_down_exps` (layers 0-33, 35-37, 40) | Q5_K | 720 896 |
| `ffn_down_exps` (**34, 38, 39**) | **Q6_K** | **860 160** |

So the per-table arena size is genuinely heterogeneous: a layer's `down` table is +22 % (Q5_K) or
+46 % (Q6_K) the size of its `gate`/`up` tables, and 3 of 41 MoE layers are a further +19 % again.
The cache already accounts for this correctly (it sizes each table as `slots * expert_bytes` and the
per-device budget divides by `Σ expert_bytes`), so the *capacity* is right; what is missing is any
awareness of it in **placement**.

Also relevant to Q3 (campaign 3): the `head_pad` is `min(expert_bytes, 512)` **per table** and each
table's slab chunk is rounded up to the VMM unit (2 MiB default).  Neither is a correctness problem —
`alloc_all_locked` holds back `dev_n_tables * unit` (`moe-expert-cache.cu:1939`) — but `head_pad` is not
counted and `table_arena_bytes()` (`moe-expert-cache.cu:1595`) is **dead code**, so the accounting is
slightly optimistic and there is no single place that reports the true per-table footprint.

### F2 — within a table there is no fragmentation, and "hotness ordering of experts" is a no-op

`table_t` stores `slots` fixed-size slots in one contiguous arena (`arena_reserved`, a slab chunk);
`slot_expert`/`expert_slot` map ids to slots.  Eviction picks a victim slot and clears the mapping —
the bytes stay allocated.  There are therefore **no holes to compact** inside a table: reordering the
slot map cannot reduce the table's footprint or move its boundary.  Any "partition hot experts above a
reclaim pointer" must operate at **table granularity**, not expert-slot granularity.

(The one intra-table lever that *can* reduce bytes is **shrinking the slot count** so the tail of the
arena can be returned to the slab — see §2.3.)

### F3 — the slab fills top-down and reclaims bottom-up, so placement is the eviction policy

`ggml_cuda_slab_arena_alloc` (`ggml-cuda-slab.cu:993`) takes the **top of the highest run that fits**,
so the first allocated table sits at the very top of `[boundary, mapped)` and later tables stack
downward.  A boundary move takes the **lowest** chunks (`ggml_cuda_slab_work_alloc`,
`ggml-cuda-slab.cu:820`) and `moe_cache_evict_slab_range` (`moe-expert-cache.cu:2675`) drops every table
whose arena intersects `[lo, hi)`.

Consequence: **the last-allocated table is the first evicted.**  Allocation order comes from
`std::map<int, std::vector<int>> by_layer` (`moe-expert-cache.cu:2046`) — ascending layer, larger-slice
first within a layer.  There is no hotness term anywhere.  If the later layers are the hotter ones (a
common MoE pattern), the cache is systematically evicting its best residents first, and, worse, the
victim set is frozen at sizing time: a workload shift cannot move a table away from the boundary.

The code's own comment at `ggml-cuda-slab.cu:987` states the design intent correctly ("PLACEMENT IS THE
EVICTION POLICY … keep the band just above the boundary free"); it just has nothing to place *by*.

### F4 — eviction is all-or-nothing per table

`moe_cache_evict_slab_range` drops a whole table on any overlap, and `stand_down_table_locked`
(`moe-expert-cache.cu:2353`) resets `slots` to 0 and frees the whole chunk.  A table whose resident set
is 90 % cold still pays the full table to release a few chunks, and a table that is 90 % hot is dropped
just as completely as a cold one.  `moe_cache_shrink_step`/`moe_cache_shrink_arena`
(`:2397`/`:2441`) are device/whole-arena fallbacks, not selective.

So the "hotness-ordered re-pack" the maintainer describes is worth doing, and the right primitive is a
**relocate-and-maybe-resize** of individual tables, not a sort.

### F5 — nothing is re-packed at idle today

The only arena mutation after sizing is the boundary-driven eviction (`moe_cache_evict_slab_range`), a
wholesale stand-down/re-arm (`stand_down_table_locked` / `moe_cache_rearm`, `:2580`), and the fail-soft
yield (`moe_cache_release_arena`/`moe_cache_shrink_*`).  There is no idle-time pass and no heat-driven
reordering.  llama-context already has a decode-idle hook where a repack could run safely (it re-arms
the arena there, `src/llama-context.cpp:1968`).

## 2. Design sketch

### 2.1 Phase 1 — hotness-ordered placement (no copy, no new API) — **DELIVERABLE**

At `alloc_all_locked`, replace the `by_layer` allocation order with a global order over **all** tables
on a device (still honouring the "a layer's tables are allocated as a unit / largest-slice-first"
requirement for uniformity retries), sorted so the **hottest tables allocate first**.  Because
`arena_alloc` stacks top-down, hot tables then sit highest in VA and are the *last* the boundary reaches.

* Heat source: the accumulator from [`../moe-heatmap-persistence/`](../moe-heatmap-persistence/README.md),
  falling back to `prefill_count` (once repaired) or expert-id order when cold.  A table's heat is
  `Σ_e heat[e]` (and maybe `hits+t` for a run that has already been going).
* Ordering must be **stable and deterministic** for a given heat vector (tie-break by `(layer, role)`)
  so the same run reproduces the same layout (important for reproducibility and for the graph-mem
  generation hash).
* Keep the layer-unit invariant: the slab layout only changes *where* a layer's three tables sit, not
  their slot counts.  Group the three tables of a layer adjacently and order groups by group heat.

This is a small, low-risk change and should be phase 1: it converts the boundary move from "drops
whatever was allocated last" to "drops the coldest first" with no copies.

### 2.2 Phase 2 — heat-aware eviction order within a boundary move (still no copy) — *secondary, measure first*

`moe_cache_evict_slab_range` currently iterates `g_tables` in id order.  When the range must drop `k`
tables it is forced to drop all overlapping ones, but a boundary move that would drop a hot table could
instead be *refused* (return 0, leave the slab's free-run accounting to take a smaller range) — i.e. a
boundary move may not need all the chunks it asks for if the only way to get them is a hot table.
That is a policy question for the slab (it currently assumes eviction frees everything it needs); worth
a measurement before coding.

### 2.3 Phase 3 — repack/relocate at decode idle (the real "defrag") — **PARKED (maintainer, 2026-10-14)**

A `moe_cache_repack_idle()` pass, run where `moe_cache_rearm` already runs (decode idle), that:

1. Computes each table's **retained slot count** from its heat (e.g. the smallest `k` such that the top-`k`
   heat covers ≥ X % of the table's total, floored at `n_expert_used` and the current `slots`).
2. For a table that can shrink and sits near the boundary: allocates a new, smaller arena from a free
   run **above** the boundary, copies the retained resident experts into it (host master or the old
   arena as source), rebuilds `slot_expert`/`expert_slot`, frees the old chunk, and repoints the alias
   and `remap_dev`.
3. Optionally also relocates a hot table upward for free (a fresh allocation already lands at the top).

Hazards to solve before this is shippable:

* **Live arena addresses.**  `moe_cache_redirect_fused` writes the arena base into the kernel launch
  parameters, so a repack must be done with no graph in flight and must invalidate captured graphs
  (`ggml_cuda_graph_mem_freed(device)`, cf. the Q8_1-arena retirement at `common.cuh:1802`).  The
  slab's VMM arena keeps its reserved VA across a stand-down, which helps for a like-for-like re-arm but
  not for a *move*.
* **Consumers' cached state.**  `g_alias_to_id` is re-pointed per hook call; the `identity` fast path
  (`slots == n_experts`) must be invalidated; the per-op `remap_dev` must be rebuilt; the fused
  gate+up path shares the up table's remap (`moe_cache_table_serves`), so a layer's three tables must
  be repacked together or not at all.
* **Cost.**  Copying a warm table is `retained_slots * expert_bytes` H2D or D2D; at 25 MiB/slot and ~160
  slots that is ~4 GiB per table, i.e. only worth it for a table that will otherwise be evicted.  The
  trigger should be "a boundary move is imminent and would drop a hot table", not a periodic sweep.
* **Purity.**  A repack changes which bytes a kernel reads (VRAM vs host) but must not change any
  arithmetic.  Since the LFRU policy already depends on placement (a resident hit vs a cold read has
  different admission bookkeeping), a repack **is** a state change and must be gated like any other
  cache-policy change (same-seed output must be identical because the expert *values* are identical;
  admission decisions may differ but the math must not).

### 2.4 Phase 4 — selective (partial) eviction — **PARKED (maintainer, 2026-10-14)**

If phase 3 shows a table is only barely over the boundary, a cheaper primitive is to **shrink it in
place**: because the arena is unit-rounded, the tail (highest offset) may contain free units; move any
resident slot out of the tail, then return those units to `arena_free`.  This helps only when the table
is *above* the boundary and the work region needs to grow into its **low** end — which is not what a
bottom-up boundary move does.  So partial eviction helps the *arena's own* free-run fragmentation, not
the boundary.  Keep it on the list but expect phase 1/3 to dominate.

## 2.5 Retarget (2026-10-17) — phases 2-4 superseded

`FINDINGS-2026-10-17-coverage-vs-defrag.md` §6-§8 replaces §2.2-§2.4 with a value-ordered plan:

| phase | what | status |
|---|---|---|
| P0 | heat-map correctness (stop the per-flush doubling; ignore a degenerate map; re-record) | **SHIPPED in r80** (block 09) |
| P1 | offline routing-trace replayer (the instrument phases 2-4 needed and never had) | **next** |
| P2 | per-*layer* capacity quota from heat (our uniform slots/table give up 0.3-0.5 pts on a uniformly-tallied map; the 35B-A3B 1.5-1.9 was a tally-gap artifact) | proposed |
| P3 | metric parity: report covered mass and expert MiB/token beside `h` | proposed |
| P4 | arena size (8 GiB reserve + 4 GiB VA spare) | hand-off to `../slab-transient-absorption/` |

The old phase 2 (heat-aware eviction order), phase 3 (relocate/resize at idle) and phase 4 (selective
eviction) are kept below as history only: they defend against a fragmentation tax the arena does not
pay (0.51 % truncation, 3.0 % eviction churn over a field turn), and each carries live-arena-address
hazards.  Do not start them without a measurement from P1 that shows the churn they would fix.

## 3. Hypotheses / work to do

1. **H1.**  Hotness-ordered placement reduces the number of hot tables lost to a given boundary move,
   and raises the post-prefill hit rate on an arena-constrained config.  Metric: resident-bytes weighted
   by heat before/after `moe_cache_evict_slab_range`, and the per-turn `h` after a wide prefill.
2. **H2.**  The effect is largest when the hot tables are the high layers (current order evicts them
   first).  Test by logging the table->layer order at sizing and the eviction set on a boundary move.
3. **H3.**  A repack at idle is only worth it when a boundary move actually happens; a sweep-based repack
   is a net loss.  Measure on a config that drives a boundary move (a wide `-ub` prefill after a large
   arena), and on one that does not (no-op).
4. **H4.**  Per-table slot shrink (adaptive `slots` from heat) can raise the arena's effective residency
   for the same budget by giving cold tables fewer slots and hot tables more — *without* breaking the
   per-layer uniformity (gate/up/down must still agree).  Note this changes `slots` per layer, which the
   re-arm path already tracks (`g_rearm_slots`).

## 4. Open questions for the maintainer

* **Q2a (RESOLVED 2026-10-14).**  Keep per-layer slot uniformity; phase 1 does not touch slot counts.
  A per-table slot count is parked with phases 3-4.
* **Q2b (RESOLVED 2026-10-14).**  Phase 1 ships on top of campaign 1's persisted map; the fallback is
  expert-id order, so it is independently testable.  The two are developed together but ship separately.
* **Q2c (RESOLVED 2026-10-14).**  Phase 3 (repack) is **parked** — multiple sessions, real purity/state
  hazards.  Phase 1 (placement reorder) is this session's deliverable.
* **Q2d (RESOLVED 2026-10-14).**  The placement order is **frozen at sizing** (as today).  A re-arm
  restores `g_rearm_slots` and does not reshuffle addresses.

## 5. Gates

* Same-seed coherence — output must be byte-identical (placement only).
* `scripts/gate-prefill-logits.sh` — a placement change must not touch prefill kernels; run it because
  a moved arena can change which path (VRAM vs cold host read) serves a prefill-band op.
* Adaptive-MTP field gate (35B-A3B `-ncmoe 99` + auto cache), `none == n1 == n3 == n7` short /
  `none == n3` long.
* `test-backend-ops -o MUL_MAT_ID` (931/931) and the greedy-purity 8-type x 5-length grid.
* A boundary-move survival metric (H1) added to the server per-turn log.
* `scripts/validate-set.sh` if promoted; whitespace-clean under `git am`.

## 6. Pointers

* `ggml/src/ggml-cuda/ggml-cuda-slab.cu` — `ggml_cuda_slab_arena_alloc` (:993, top-down placement),
  `ggml_cuda_slab_arena_free`, `ggml_cuda_slab_arena_alloc_transient`, `ggml_cuda_slab_work_alloc`
  (boundary move), `ggml_cuda_slab_extend`.
* `ggml/src/ggml-cuda/moe-expert-cache.cu` — `alloc_table_locked`/`alloc_all_locked` (sizing + the
  layer-unit retry), `moe_cache_evict_slab_range` (:2675), `stand_down_table_locked`,
  `moe_cache_rearm`, `moe_cache_shrink_step`/`_arena`, `table_arena_bytes` (dead), `table_t`.
* `src/llama-context.cpp:1968` — the decode-idle re-arm site a repack would share.
* Campaign: [`../moe-heatmap-persistence/`](../moe-heatmap-persistence/README.md) (the heat accumulator
  phase 1 and 4 need).
* `archive/work/moe-cache-autosize/`, `archive/work/fit-slab-accounting/`,
  `archive/work/slab-ring-region/` — the slab's history and the boundary-move invariants.
