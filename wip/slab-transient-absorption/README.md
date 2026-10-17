# `slab-transient-absorption` — move the optional device transients inside the slab so the headroom (and the arena) can grow

**Status: OPEN, re-verified at r80 (2026-10-17) — step 0 + the first two absorptions are implemented and
measured, see §8.**  Not part of the delivery; nothing here may be applied to the fork without the
maintainer's go-ahead (see `AGENTS.md`).
Seeded from a code review of the movable-boundary slab and the CUDA backend's raw device allocations
(`ggml/src/ggml-cuda/ggml-cuda-slab.cu`, `ggml/src/ggml-cuda/common.cuh`,
`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/allreduce.cu`, `ggml/src/ggml-cuda/mmb.cu`).

> **Re-verified 2026-10-17 against r80 (the r60-era table was stale).**  What changed: the line numbers
> all moved; the FA staging arena gained the `GGML_CUDA_FA_STAGE_PREALLOC` hint path (r53); the H2D ring
> fallback now refuses when the carved hole is unarmed (issue #164/r67); **three rows were missing** (the
> cuBLAS workspace, the `ggml_cuda_pool_leg` workspace high-water, and the 512 MiB H2D calibration peak);
> the MTP draft row has measured numbers; and **F1 is now wrong for the field config** -- with the MTP
> draft present `reserve = headroom + aux`, so `mapped < size` and `ggml_cuda_slab_extend` IS live (the
> field log shows `va_spare=4.00 GiB`; F1's "extend is dead" only holds when `aux == 0`).  See §1a for
> the r80 table and §1b for the corrections.  The payoff this campaign targets is now measured: **~1 pt
> of expert-cache hit rate per GiB moved from the headroom into the arena** on the field config
> (`wip/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md` §6.1).

This is the campaign for the maintainer's Q3: *"what opportunities exist to either minimise the use of
VRAM outside of the slab when operating in host-expert mode? Are there any transient items that could be
moved to within the slab when the slab is enabled, such that the slab could potentially be grown
larger?"*

## 0. TL;DR

The slab is created with `size(VA) = mapped = free - reserve`, where the default reserve **is** the
headroom (`GGML_CUDA_SLAB_RESERVE_MIB` defaults to `headroom + aux`), measured on this box as
`free=31.04 GiB, reserve=4.00, headroom=4.00, mapped=27.00 GiB, va_spare=0.00` (run1).  So the arena
gets `mapped - boundary` and nothing more, and **the only lever that enlarges the arena at slab creation
is lowering the headroom** (later it can also grow when the wide view is dropped and the boundary
moves down).  The headroom exists because several transients are allocated *outside* the slab (raw
`cudaMalloc`); the `fit-slab-accounting` campaign proved that a thin headroom aborts or silently
corrupts.  Several of those transients are under our control and **prefill-only or process-lifetime**,
so they can be re-homed into the slab.  Each GiB absorbed is ~+1 GiB per device of arena, i.e. more
resident experts at the same budget.

The external floor (hipBLASLt Tensile code objects + workspace) is not controllable and must be
**measured**, not assumed, before the headroom default can move.

**Phase map (2026-10-14).**  The controlled transients split cleanly by operational phase, which is the
deciding question the maintainer asked: the wide-phase ones (FA staging `Q->ne[1] > 8`, H2D-ring
fallback, MMB shadows `nrows >= 512`) can be allocated/dropped with the wide view exactly like the ring;
the narrow-phase ones (Q8_1 from mmvq/norm/unary, the per-table remap, the AR scratch) are live on every
graph and join the **pinned narrow** mechanism (narrow-2's shape).  See §3.0 for the gates and §3.1 for
the region decisions.

## 1. What lives outside the slab today

*(The r60-era seed table, kept as the record.  **Superseded by §1a** — line numbers, sizes and three
missing rows are corrected there; read §1a/§1b as the current state.)*

Measured/derived from the code (all are raw `cudaMalloc` unless noted; "transient" = not the model
weights or the KV cache, which are allocated before the slab by design):

| # | allocation | where | lifetime | size | controller |
|---|---|---|---|---|---|
| 1 | FA prefill staging arena (`fattn_stage[stream]`) | `common.cuh:1869` | per-stream, prefill-only, retained | up to `GGML_CUDA_FA_STAGE_MAX_MB` (hint path) | **ours** |
| 2 | Q8_1 cache arena (`q8_1_arena`) | `common.cuh:1764` | multi-token graph, retired-not-freed (r25) | tens of MiB, grows | **ours** |
| 3 | op-offload H2D ring fallback (`h2d_stage[slot]`) | `common.cuh:2080` | when the slab ring hole is too small/exhausted | up to `GGML_SCHED_STAGE_MAX_MB` | **ours** |
| 4 | MMB shadow weights | `mmb.cu:2480`, `:2495` | process-lifetime | size of the shadowed weights | **ours** |
| 5 | all-reduce scratch (`p->dev_tmp[i]`) | `allreduce.cu:527` | process-lifetime, per device | `copy_bytes` (small) | **ours** |
| 6 | expert-cache per-table `remap_dev` | `moe-expert-cache.cu:1765`, `:3430` | table lifetime | `n_experts*8*4` B × tables (~1 MB/model) | **ours** |
| 7 | MTP draft compute buffer | slab narrow-2, else fallback | context lifetime | ~3.7 GiB when a draft is present | ours (already slabbed when narrow-2 fits) |
| 8 | hipBLASLt Tensile code objects + workspace | external | process-lifetime | **unknown; this is the real floor** | not ours |
| — | model weights, KV cache | before the slab | — | — | out of scope (not transient) |

The G4 guard already bounds #1 and #3 when the slab is active
(`optional_alloc_within_free_cap`, `common.cuh:2033`; default `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT=50`)
so they cannot eat the headroom without a warning and a fallback — but a *refusal* degrades prefill
(the native K/V read) or staging, so their worst case is still a performance cliff, and their normal
size still comes out of the same 4 GiB.

## 1a. The table at r80 (re-verified 2026-10-17, `ggml_cuda_device_malloc` never consults the slab)

Everything that allocates through `ggml_cuda_device_malloc` / a raw `cudaMalloc` is outside the slab by
construction; the slab serves only the scheduler's compute buffers and the expert arena.  Line numbers
are r80.

| # | allocation | where (r80) | lifetime | size (measured) | controller |
|---|---|---|---|---|---|
| 1 | FA prefill staging arena (`fattn_stage[stream]`) | `common.cuh:1953` (`fattn_stage_try_get`) | per-stream, **retained across prefills**, freed only when it grows | hint = the K/V view roots capped by `GGML_CUDA_FA_STAGE_MAX_MB`; **762 MiB** on the GSQ config (`ENVIRONMENT.md`) | **ours** |
| 2 | Q8_1 cache arena (`q8_1_arena`) | `common.cuh:1793` decl, `:1848` (`q8_1_cache_get`) | multi-token graph, retired-not-freed (r25; captured graphs point into it) | "tens of MiB", grows 1.25x | **ours** |
| 3 | op-offload H2D ring fallback (`h2d_stage[slot]`) | `common.cuh:2168` | per-slot, slab-hole-first; `cudaMalloc` only when there is no hole / the hole is exhausted / the hole is unarmed | up to a slot of `GGML_SCHED_STAGE_MAX_MB` | **ours** |
| 4 | MMB shadow weights (`g_mmb_shadow`) | `mmb.cu:2495`, `:2510` | process-lifetime (freed only at cleanup) | size of the shadowed BF16 weights | **ours** |
| 5 | all-reduce scratch (`p->dev_tmp[i]`) | `allreduce.cu:527` | process-lifetime, per device, `-sm tensor` only | `copy_bytes` (small) | **ours** |
| 6 | expert-cache per-table `remap_dev` | `moe-expert-cache.cu:2357`, `:4263` | table lifetime | `n_experts*8*4` B per table (~1 MB / 144 tables) | **ours** |
| 7 | expert-cache `heat_dev` tally mirror | `moe-expert-cache.cu:3848` | table lifetime | **2 KiB/table** (deliberately raw: it must not be able to yield the arena) | ours, out of scope |
| 8 | MTP draft compute | slab **narrow-2** when it fits (`slab.cu:782`), else fallback | context lifetime | **512 MiB** (field log), draft weights 2 647 MiB + 288 KV | ours (already slabbed) |
| 9 | **cuBLAS/hipBLAS workspace** (new row) | `common.cuh:2364` | per device × per stream, process-lifetime | 4 MiB (32 MiB on Hopper+), so 4 × streams per device | **ours**, tiny |
| 10 | **workspace-pool high-water** (`ggml_cuda_pool_leg`, new row) | `ggml-cuda.cu` (`ggml_cuda_device_malloc` → raw) | cached blocks, grows with prefill width | part of the measured ~3.7 GiB; the operator's `GGML_CUDA_POOL_CACHE_MB` caps the cache | **ours** |
| 11 | **H2D bandwidth calibration probe** (new row) | `ggml-cuda-moe-cache-hooks.cu:139` | once per device, **freed** immediately | **512 MiB peak** (runs during ring auto-sizing, i.e. at load) | ours, peak only |
| 12 | hipBLASLt Tensile code objects + internal workspace | external driver | process-lifetime | **unknown; this is the real floor** | not ours |

`ENVIRONMENT.md` measured the post-slab consumption of rows 1/8/9/10/12 together at **~3.7 GiB** on a
GSQ-era config (MTP draft buffer 762 MiB, FA-QSA workspace 762 MiB, hipBLASLt Tensile + workspace, pool
high-water) — that is what the 4096 MiB headroom is really paying for, and the ceiling on what absorbing
rows 1-7 can buy back.

## 1b. Corrections to the r60-era findings

* **F1 ("`extend` is dead at the default") — only true with no draft.**  `reserve = max(headroom + aux,
  1024 MiB)` (`slab.cu:321`), and `aux` is the MTP/DFlash draft VRAM the preflight measures
  (`ggml_cuda_slab_set_aux_reserve`).  With a draft, `mapped = free - reserve < size = free - headroom`,
  so `ggml_cuda_slab_extend` (`:559`) is live: the field GSQ log shows `reserve=8.00 GiB` (4096 headroom
  + 4096 aux), `size(VA)=21.38 GiB`, `mapped=17.38 GiB`, **`va_spare=4.00 GiB`**.  Without a draft they
  are equal and extend cannot fire, as F1 said.
* **#1's controller note is now "hint + optional cap"** — `GGML_CUDA_FA_STAGE_PREALLOC` (default on, r53)
  allocates once at the hinted size instead of climbing the 1.25x ladder, and the hint is only honoured
  inside `optional_alloc_within_free_cap` (`common.cuh:2121`, default 50 % of free VRAM).
* **#3 is stronger than the table said**: the slot is served from the slab's transient hole first and only
  falls back to `cudaMalloc` when there is no hole, the request exceeds a slot, or **the hole is unarmed**
  (`common.cuh:2235`, the r67 carved-but-unarmed fix).
* **The headroom is a per-config tuning value, not only a floor to reduce** — measured 2026-10-17 on the
  field GSQ config: 4096 → arena 15 666.6 MiB / hit 0.9056, 3072 → 16 705.7 / 0.9147, 2048 → 17 744.8 /
  0.9223, and a 16 384-token prompt 0.9165 → 0.9380 at 2048 with identical text.  So the campaign's
  premise is confirmed *and* the first ~2 GiB can be taken today by an operator who accepts a wide-ubatch
  that is not the rejected `-ub 6144 -c 204800` case.  Absorbing rows 1-7 is what makes lowering the
  *default* safe.

## 2. Findings (evidence)

### F1 — the headroom does not "grow back": `extend` is dead at the default

`ggml_cuda_slab_reserve_bytes` (`ggml-cuda-slab.cu:258`) defaults the reserve to
`max(headroom + aux, 1024 MiB)`.  `ggml_cuda_slab_init_locked` then maps
`want = (free - reserve)` and reserves `va = (free - headroom)`.  With `aux == 0` those are **equal**, so
`mapped == size` and `ggml_cuda_slab_extend` (`:531`) — the mechanism that is supposed to "map the part
of the reserve the model turned out not to need" — can never fire.  Confirmed by the run1 `SLAB_INIT`
line (`va_spare=0.00 GiB`).  **So the 4 GiB headroom is permanently outside the slab**, and the whole
"slab grows into the unused reserve" story only applies when an operator sets
`GGML_CUDA_SLAB_RESERVE_MIB` above the headroom explicitly.

Therefore the way to give the arena more VRAM is to make the headroom unnecessary, i.e. absorb the
transients that need it.

### F2 — the largest controlled transient is prefill-only and shares the wide-view window

The FA staging arena (#1) is only touched by multi-token graphs, which are **never graph-captured**
(`common.cuh:1826` says so explicitly).  Its natural home is therefore the slab's **ring/wide work
region** (already reserved only while the wide view is live, `ggml-cuda-slab.cu:1068`
`arena_alloc_transient`).  It would then be reclaimed to the arena at the prefill→decode transition
exactly like the ring, instead of being a permanent cudaMalloc.  This is the single biggest controlled
win and the ring region already exists as the precedent (block 06 promoted the ring *into* the slab for
the same reason: it freed ~1 GiB of headroom; recorded in `ggml-cuda-slab.cu:325`).

**Caveat (the ring's lifetime contract applies).**  The ring region is reclaimed to the arena when the
wide view drops and re-armed on the next wide reserve (`ggml-cuda-slab.cu:132`, `:1052-1060`); a ring
slot is only ever *read* while armed.  The FA staging arena has the same "live only while the wide view
is" property, but unlike a ring slot it is **retained across prefills** (`fattn_stage[stream]`, freed
only when it grows).  So serving it from the transient region requires (a) a *persistent* bump
allocation in that region (not a per-arm slot), (b) the region sized for `max(ring, FA staging)` rather
than the ring alone, and (c) the same discipline the ring has: never read it while disarmed.  That is
viable but is more than "point the FA arena at the ring"; it is the first thing to validate in the
measurement.  If the retained-across-prefills requirement makes it awkward, the FA arena is the best
candidate for the Option-A small region (§3.1).

### F3 — the Q8_1 arena and the MMB shadows are long-lived (but in different phases)

Neither is transient in the reclaim sense, but they differ by phase (see §3.0): the Q8_1 arena serves the
mmvq/norm/unary **narrow** (decode/verify) paths and is deliberately **retired, not freed** (r25,
`common.cuh:1802`) because captured graphs hold pointers into it; the MMB shadows are created during
**prefill** (`nrows >= 512`) but cached across prefills for the life of the weights.  So Q8_1 belongs in
a **pinned narrow** region (narrow-2's mechanism), and MMB needs a capped pinned home that yields under
pressure (the dequant re-pay is the cost of dropping it).  Neither is a good fit for the *work* region
(which must be reclaimable).  The Q8_1 arena's growth already copies old→new, so a slab-backed arena is
no worse.

### F4 — the per-table `remap_dev` buffers are one cudaMalloc per table

120 tables × `n_experts*8*4` B = one small `cudaMalloc` each (`moe-expert-cache.cu:1765`, `:3430`),
never freed until stand-down.  Individually negligible, but they are 120 opportunities for the driver to
fragment the region outside the slab.  A single per-device remap arena (bump-allocated per table) would
remove all of them and is a prerequisite for any phase-3 arena repack (a repack must not have to
`cudaMalloc` a new remap while the arena is short).

### F5 — the H2D ring's `cudaMalloc` fallback is dead weight in host-expert mode, and it hides two defects
**(rewritten 2026-10-17 at r80; the r60 text above it was `H2D staging shortfall` from run1)**

The r60 finding said the ring "still falls back outside the slab".  At r80 it should not, and the fallback
is worse than useless when it fires.  Sizing first (`h2d_stage_region_bytes`, `common.cuh:2040`; the
context's `slice_floor` in `llama-context.cpp`):

* `ring_req = min(slots × (largest staged slice + 512), cap)`, with `cap = max(auto, slice_floor)` capped
  at **a quarter of the card** (`total_b / 4`), or an explicit `GGML_SCHED_STAGE_MAX_MB`;
* the field configs land exactly on `slots × slice`: GSQ 1-GPU `ring_req = 4.11 GiB` = 8 × 526 MiB;
  35B-A3B 2-GPU `2.11 GiB` = 8 × 270 MiB — i.e. the region is sized to hold every slot at its real
  slice size, and `ring_slot_capacity = ring_region / slots` therefore always fits.

So a shortfall needs either **(a)** the carve to decline (`narrow_floor + ring + chunk > mapped`,
`slab.cu:215`), which needs a slab whose mapped region cannot hold the narrow view plus a quarter of the
card — a config where the slab barely exists; or **(b)** the cap to undercut `slots × slice` (a small
card, or an explicit `GGML_SCHED_STAGE_MAX_MB` below it).  Both are "not enough room to justify the slab".
Meanwhile the pool it falls back to is the **outside** VRAM: headroom 4096 MiB minus the measured ~3.7 GiB
of post-slab allocations, i.e. **~0.3-0.6 GiB free** at steady state — an order of magnitude smaller than
the region that just failed.  A `cudaMalloc` there either fails (and the caller correctly takes the
non-staged path) or eats the reserve FA staging / hipBLASLt / the pool need, which is the abort-or-silently-
corrupt mode this whole campaign exists to prevent.

**Two defects in the fallback path (`common.cuh:2176-2225`):**

1. **The unarmed case bypasses the refusal guard.**  The "never read a slab slot while disarmed" check
   (`:2235`) applies only to a slot already slab-backed.  A request that arrives while `ring_armed == false`
   (or before the hole is carved) takes `cap == 0` -> skips the slab attempt -> `cudaMalloc`s, and the slot
   then **stays** outside the slab (the slab is only retried on a later *growth*).
2. **A slot that migrates slab-ward leaks its old block.**  When a previously-`cudaMalloc`ed slot grows
   while armed, the slab branch returns at `:2201` **before** the `else if (h2d_stage[slot] != nullptr)
   { cudaFree }` cleanup at `:2210`, so the old allocation is neither freed nor counted
   (`h2d_stage_total` still subtracts `old_size`).

**Recommended step 0 for this campaign (small, ~10 lines + a kill switch):** when the slab is *active*
(host-expert mode), a ring shortfall is a **refusal** (return `nullptr` -> the scheduler's non-staged path,
which is correct, only slower), never a `cudaMalloc`.  Keep the `cudaMalloc` path only for slab-inactive
devices (dense model / no VMM / `GGML_CUDA_SLAB_RING_MIB=0`), which is exactly the pre-slab behaviour.
That closes the unarmed path and the leak, and removes the last *unbounded* outside-slab transient from
the headroom budget — a prerequisite for lowering the default headroom.

## 3. Design sketch

0. **Step 0 — make the H2D ring slab-only in host-expert mode** (see F5): a ring shortfall becomes a
   refusal, never a `cudaMalloc`.  Removes the unarmed bypass and the migration leak, and takes the ring
   out of the headroom budget.  Independent of the rest and small enough to land first.
1. **A slab "optional" allocator** — `ggml_cuda_slab_optional_alloc(device, size)` that serves from the
   ring/wide transient region first and falls back to the arena's free list; every caller is already
   fail-soft (returns null → caller degrades), so a refusal is never fatal.  Reuse
   `ggml_cuda_slab_arena_alloc_transient` for the prefill-only cases (#1, #3) and a new no-evict arena
   sub-list for the lifetime cases (#2, #4, #5, #6).
2. **Re-home the call sites one at a time, each behind an env kill-switch** (default on per the
   default-on policy, once gated):
   * `fattn_stage_try_get` (`common.cuh:1869`) → transient region (prefill-only).
   * `h2d_stage_buffer` fallback (`common.cuh:2080`) → transient region.
   * `q8_1_cache_get` (`common.cuh:1780`) → no-evict arena region.
   * MMB shadow allocs (`mmb.cu:2480/2495`) → no-evict arena region.
   * `dev_tmp` (`allreduce.cu:527`) → no-evict arena region.
   * `remap_dev` (`moe-expert-cache.cu:1765/3430`) → a per-device remap arena.
3. **Measure the residual floor.**  With #1-#6 absorbed, bisect `GGML_CUDA_SLAB_HEADROOM_MIB` down to
   failure on the `fit-slab-accounting` wide-prefill matrix (the 2048-table abort and the silent-corrupt
   case).  The largest safe value becomes the new default; document it as "hipBLASLt + the widest graph's
   un-routable workspace".
4. **Only then** consider whether the KV cache or other pre-slab allocations could move inside — those
   are invasive and not "transients"; keep them out of scope.

Do **not** simply lower the default headroom without absorbing the transients: the campaign record
(`ggml-cuda-slab.cu:325-331`) already trialled 3072 and rejected it on a wide-prefill OOM
(`hipblaslt.cpp:164`).  The absorption is what buys a *safe* reduction.

## 3.0 Phase map — WHEN each transient is live (evidence)

The maintainer's question ("do they map to operational phases, like the ring/wide buffers?") is the
right way to decide the region.  Traced from the gates:

| transient | live when | phase | drop/recreate cost | region |
|---|---|---|---|---|
| FA staging arena (`fattn_stage`) | `Q->ne[1] > 8` (`native_width`, `fattn-common.cuh:1526`); on RDNA4 `prefill_stages = true` | **wide** (prefill, and a wide verify 9-16) | cheap: it re-stages each pass, and a refusal falls back to the native K/V read (correctness-neutral) | **wide/ring transient** |
| MMB shadow weights (`g_mmb_shadow`) | `nrows >= mmb_min_t()` (default **512**, `mmb.cu:1693/2040`) | **prefill** | a BF16 dequant re-pay on the next prefill | wide-pinned, or arena no-evict with pressure-drop |
| H2D ring cudaMalloc fallback | staging width gate (measured 1542 tok) | **prefill** | cheap | **wide/ring transient** |
| Q8_1 arena (`q8_1_arena`) | mmvq / norm / unary → `q8_1_cache_get` (`mmvq.cu`, `norm.cu`, `unary.cu`) | **narrow** (decode/verify) | rebuilds per graph anyway, but the arena survives for captured reuse | **narrow-pinned** (a narrow-3) |
| expert-cache `remap_dev` | decode/verify band per-op | **narrow** | cheap | narrow-pinned |
| all-reduce scratch (`dev_tmp`) | every graph, `-sm tensor` only | **both** | never freed | narrow-pinned |
| MTP draft compute | draft context | both | — | narrow-2 (already done) |

**Consequences.**

1. **The wide-phase transients (FA staging, H2D ring fallback) map exactly onto the existing ring/wide
   mechanism** — the maintainer's suggestion is right.  They are live only while the wide view is, and
   the ring region already returns to the arena when the wide view drops (`ggml-cuda-slab.cu:132`,
   `:1052`).  So they become persistent bump allocations in that region (sized `max(ring, FA staging, …)`),
   reclaimed to the arena on the wide drop, with the same discipline the ring has: **never read them
   while disarmed** (`h2d_stage_buffer` already refuses when `ring_slot_capacity == 0`,
   `common.cuh:2080`; `fattn_stage_try_get` must gain the same check).
2. **The narrow-phase transients cannot be treated as transient** — they are live on essentially every
   graph, and the Q8_1 arena is read by captured decode graphs, so its VA must stay stable.  The right
   home is the mechanism narrow-2 already uses for the MTP draft: a **pinned narrow region**
   (`ggml_cuda_slab_narrow2_floor_locked`).  Generalising it to a "narrow-3" (or extending the narrow
   floor with a pinned-optional sub-region) absorbs Q8_1 + remap + AR scratch.
3. **MMB shadows are the one awkward case**: created during prefill but *cached across* prefills
   (`g_mmb_shadow`, freed only at cleanup), so a per-wide-drop would re-pay the dequant every prefill.
   They want a pinned/capped home that is only yielded under pressure (a no-evict flag with a
   yield-on-OOM, i.e. the Option-B exemption), or a sized pinned region.

So the hybrid is not just "prefill vs lifetime" — it is **route each transient to the region matching its
phase**, using the two mechanisms the slab already has.  Only MMB needs a new (small) policy.

## 3.1 Q4 — a third "optional" slab region vs folding into the arena region (pros/cons)

The maintainer asked for the trade-off explicitly.  There are three shapes; note that "third region"
is not novel — the ring hole is already a third region (`narrow | ring | wide`), so the precedent
exists.

### Option A — a dedicated `optional` region in the slab

Carve `[narrow_floor + ring, wide)`-style space, or a new pinned region like narrow-2, sized from the
measured optional peak.

**Pros**

* The arena's allocator and eviction logic are untouched: `arena_free`/`arena_alloc` and
  `moe_cache_evict_slab_range` never see these bytes, so there is no new exemption to audit.
* Independent, explicit accounting (`slab_optional_total()`) and an easy "is it big enough?" report.
* A stable VA by construction, which is what the graph-captured Q8_1 arena needs.
* Lifetime is process-scoped, so stand-down/re-arm logic cannot accidentally free it.

**Cons**

* **Reserved whether used or not** — the exact waste `slab_extend` was supposed to fix.  Sizing needs the
  worst case (Q8_1 + FA staging + MMB peaks rarely coincide), so a fixed region re-creates the headroom
  *inside* the slab; the only gain over today is that the *unused* headroom can then be reclaimed to the
  arena — which we can also get by just lowering the headroom once the transients are routed.
* A third boundary/extent to carry through every layout decision (`work_alloc`, `extend`, `arena_total`,
  narrow-2).  More surface in the trickiest file.

### Option B — fold the process-lifetime allocations into the arena region (on-demand, no-evict)

Serve Q8_1 / MMB / AR scratch / remap from `ggml_cuda_slab_arena_alloc` at first use, flagged no-evict.

**Pros**

* No new region or boundary; reuses the existing free-run allocator.
* **Adapts to the actual need** — you pay only for what the run really allocates, which is strictly
  better VRAM-wise than a worst-case reservation.
* Pays off "the slab could be grown larger" directly: with the headroom lowered, the arena owns the
  VRAM until a consumer actually asks.

**Cons**

* `moe_cache_evict_slab_range` frees *everything* in the reclaimed range; it must learn a no-evict flag
  or it will hand a Q8_1 arena (that a captured graph points into) to the work pool.  That is a new
  invariant on the most dangerous path.
* The expert cache sizes itself against `ggml_cuda_slab_arena_total()` **before** these consumers appear
  (all at sizing; Q8_1 grows at the first deep prefill).  So a no-evict allocation competes with the
  arena's own post-sizing slack, and a pinned block in the middle of the arena can strand a contiguous
  run the allocator needs (it takes the top of the highest run).  Fragmentation risk.
* The reported arena total/capacity would over-count unless the no-evict bytes are subtracted from the
  cache's sizing basis.

### Option C — hybrid (recommended), refined by the phase map (§3.0)

Route each transient to the region that matches its **operational phase**, using the two mechanisms the
slab already has.  This is Option C, but derived from the gates rather than from lifetime guesswork:

* **Wide-phase** (FA staging arena, H2D-ring fallback) → the **existing transient work/ring region**
  (`ggml_cuda_slab_arena_alloc_transient`).  They are live only while the wide view is, the region is
  already reclaimed at the prefill→decode transition, and their refusal is fail-soft.  This is exactly
  the ring mechanism the maintainer suggested.  Needed: a *persistent* bump allocation (not a per-arm
  slot), the region sized `max(ring, FA staging, …)`, and the same "refuse when disarmed" check the ring
  has (`common.cuh:2080`) added to `fattn_stage_try_get`.
* **Narrow-phase** (Q8_1, remap, AR scratch) → a **pinned narrow region**, i.e. generalise the narrow-2
  mechanism (`ggml_cuda_slab_narrow2_floor_locked`) that already pins the MTP draft's compute.  They are
  live on every graph and the Q8_1 arena is read by captured graphs, so a stable VA is required; this is
  mechanically closer to narrow-2 than to the ring.
* **MMB shadows** → the only genuinely new policy: wide-created but cached across prefills, so a
  per-wide-drop re-pays the dequant.  Give them a **capped pinned home that yields only under pressure**
  (a no-evict flag with a yield-on-OOM), which is the Option-B exemption scoped to one consumer.  If their
  measured peak is small, a sized pinned region (Option A) is simpler.

A/B is therefore not a global either/or: each consumer takes the mechanism its phase implies.

### What the measurement must produce

1. The per-consumer peak (FA staging, Q8_1, MMB, AR, remap) on the wide-prefill matrix.
2. The residual external floor (hipBLASLt Tensile + workspace) with all controlled transients
   slab-resident — this is the new `GGML_CUDA_SLAB_HEADROOM_MIB`.
3. Whether any process-lifetime consumer's peak is a function of prefill width/context (drives A vs B).

## 4. Hypotheses / work to do

1. **H1.**  Absorbing #1 (FA staging) alone allows a measurable headroom reduction with no regression on
   the wide-prefill matrix, because it is the largest transient that is *not* process-lifetime.
2. **H2.**  The arena residency gain is proportional to the headroom reduction: `Δarena ≈ Δheadroom`
   per device.  Quantify on a model whose arena is the binding constraint (a ~100 GiB MoE, not the
   fully-resident 35B-A3B).
3. **H3.**  The external floor is hipBLASLt, and it is stable across prefill widths / batch sizes; measure
   it directly (e.g. a run with every controlled transient slabbed, bisect the headroom).
4. **H4.**  Removing the 120 per-table `remap_dev` cudaMallocs measurably reduces allocation jitter /
   fragmentation outside the slab (small, but it is also a prerequisite for the campaign-2 repack).

## 5. Open questions for the maintainer

* **Q3a (RESOLVED 2026-10-14, pending measurement).**  Hybrid (C), **refined by the phase map (§3.0)**:
  wide-phase → the existing ring/wide transient region; narrow-phase → a pinned narrow region
  (narrow-2's mechanism); MMB shadows → a capped pinned home that yields only under pressure.  The
  measurement then fixes the region sizes and the new headroom default.  No A-vs-B either/or remains.
* **Q3b (DEFERRED 2026-10-14).**  Keep `GGML_CUDA_SLAB_HEADROOM_MIB` as is for now, **monitor the slack**
  as the absorption work lands, and set the default last — once the other campaigns' sizes have stopped
  moving.  (The maintainer: "determining the headroom seems like it's the last thing that should be done
  once everything else is secure".)
* **Q3c.**  The Q8_1/MMB absorption interacts with the r25 `GGML_CUDA_GRAPH_MEM_GEN` recapture logic and
  the retired-arena list; confirm the graph-mem generation must be bumped when a slab-backed arena is
  retired (it should not need to be, since the VA is stable, but it must be checked).

## 6. Gates

* The `archive/work/fit-slab-accounting/` matrix: wide-prefill configs (`-ub 4096/8192`, deep context)
  must not abort and must not silently corrupt (the `!!!!!!!!` / 475 t/s signature).
* `--fit` interaction: `llama-server --fit` must still size the context around the arena (the preflight
  reserve path, `src/llama-model.cpp:3899`).
* Same-seed coherence; `scripts/gate-prefill-logits.sh` (the FA staging arena feeds a prefill kernel —
  this one genuinely touches the prefill path, so the KLD gate is mandatory).
* Adaptive-MTP field gate; `test-backend-ops`; `scripts/validate-set.sh` if promoted.

## 7. Pointers

* `ggml/src/ggml-cuda/ggml-cuda-slab.cu` — `ggml_cuda_slab_reserve_bytes` (:321),
  `GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT` (:395), `ggml_cuda_slab_init_locked` (:638),
  `ggml_cuda_slab_extend` (:559), `ggml_cuda_slab_arena_total` (:534),
  `ggml_cuda_slab_arena_alloc_transient` (:1144), `ggml_cuda_slab_ring_slot_capacity` (:1123),
  `ggml_cuda_slab_narrow2_floor_locked` (:782), `ggml_cuda_slab_wide_offset` (:234),
  `ggml_cuda_slab_seed_floor` (:292).
* `ggml/src/ggml-cuda/common.cuh` — `fattn_stage_try_get` (:1953), `q8_1_cache_get` (:1848),
  `q8_1_arena` (:1793), `h2d_stage_buffer` (:2168), `optional_alloc_within_free_cap` (:2121),
  `h2d_stage_region_bytes` (:2040), the cuBLAS workspace (:2364).
* `ggml/src/ggml-cuda/moe-expert-cache.cu` — `remap_dev` (:2357, :4263), `heat_dev` (:3848).
* `ggml/src/ggml-cuda/mmb.cu:2495`/`:2510`; `ggml/src/ggml-cuda/allreduce.cu:527`;
  `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu:139` (the 512 MiB calibration peak).
* `wip/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md` §6.1 — the measured
  headroom -> arena -> hit-rate conversion on the field config.
* Campaigns: `archive/work/fit-slab-accounting/`, `archive/work/fa-stage-prealloc/`,
  `archive/work/slab-ring-region/`, `archive/work/moe-cache-autosize/ARENA-UB-TENSION.md`.
* `ENVIRONMENT.md` — `GGML_CUDA_SLAB_*`, `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT`,
  `GGML_CUDA_FA_STAGE_PREALLOC`, `GGML_CUDA_FA_STAGE_MAX_MB`, `GGML_SCHED_STAGE_MAX_MB`.

---

## 8. 2026-10-17 session — what landed, measured (r80 tree, 1× R9700)

Work is in the **fork working tree** (`~/llama.cpp`, uncommitted at the time of writing); nothing here is
in `patches/` yet.  The changes:

1. **Step 0 — the H2D ring is slab-only in host-expert mode** (F5's recommended first move).
   `h2d_stage_buffer` now refuses a shortfall (`ggml_cuda_slab_active() && ggml_cuda_slab_ring_requested()`)
   instead of `cudaMalloc`-ing it; the two defects F5 named are closed: the unarmed request no longer
   bypasses the refusal guard, and a slot that MIGRATES into the slab now frees its old raw block (the
   slab branch used to return before the `cudaFree`).  Kill switches: `GGML_CUDA_SLAB_RING_FALLBACK=1`
   restores the old fallback; `GGML_CUDA_SLAB_RING_MIB=0` keeps the whole ring pre-slab (unchanged).
2. **The FA prefill staging arena is now a slab WIDE-PHASE transient** (`ggml_cuda_slab_wide_aux_alloc`).
   It is bump-allocated just above the live work boundary (so it sits in the band a wide prefill already
   claims), grown on demand, and **released at the prefill → decode transition** together with the ring
   (`ggml_cuda_slab_ring_set(..., false)`); the arena gets its bytes back.  A cached pointer is validated
   against `ggml_cuda_slab_wide_aux_gen()`.  The refusal path is fail-soft (the native K/V read), and the
   old raw path remains for slab-inactive devices and the 9..16-row wide-verify case below the ring's
   arming gate.
3. **The QSA3 prefill workspace is now in that same wide band** (`fattn-qsa3.cu`, a small RAII
   `qsa3_scratch<T>`).  The workspace pool (`ggml_cuda_pool_leg`) keeps every freed block, so its
   high-water (372 MiB on the 15.7k field prefill) was permanently outside the slab; QSA3 is 100 MiB of
   that.  It is safe to move because the launcher is **prefill-only** (`q->ne[1] >= 128`), i.e. never in a
   captured decode/verify graph.  The release is LIFO (the seven blocks are destroyed in reverse
   declaration order); a non-LIFO caller leaves a hole that the wide drop reclaims.  The base is retained
   when the stack empties so each layer reuses the same region (without that the boundary walked
   8064 → 11136 MiB, ~3 GiB of arena churn per prefill; with it, one 64 MiB chunk).
4. **Diagnostic:** `GGML_CUDA_SLAB_STATS=1` logs `SLAB_STATS` at every wide arm/drop and at context exit
   (free/mapped/work/narrow/ring/wide-aux/narrow-2/arena/**pool** mib, headroom and aux config), plus
   `SLAB_WIDE_AUX` per growth.  The pool figure is the `ggml_cuda_pool_leg` retained bytes per device.

### 8.1 The field config's outside-slab peak, decomposed (15 748-token prefill, headroom 2048)

`free` is VRAM outside the slab, sampled at the wide drop (i.e. after the prefill, before the reclaim).

| run | free at wide-drop | pool high-water | arena | note |
|---|---:|---:|---:|---|
| pre-change (r80) | 1842 MiB | 372.3 MiB | 11776 MiB | — |
| QSA3 off (`GGML_CUDA_QSA3=0`) | 1942 | — | 11776 | QSA3 = ~100 MiB |
| MMB off (`GGML_CUDA_MMB=0`) | 1882 | — | 11776 | MMB = ~40 MiB |
| no hipBLASLt (`ROCBLAS_USE_HIPBLASLT=0`) | 1946 | — | 11776 | hipBLASLt = ~104 MiB |
| **QSA3 in the slab (this work)** | **1942** | **271.4** | 11712 | boundary churn 64 MiB; greedy text identical |

So the ~628 MiB prefill peak is **pool 372 (QSA3 ~100, now absorbed, + ~270 of other prefill pool
temporaries) + hipBLASLt ~104 + MMB ~40 + QSA3-off delta noise**, not one thing.  The remaining 271 MiB
is the next target, but it is *not* a blanket pool redirect: some of those temporary blocks belong to
graphs that ARE captured at decode, and a captured pointer cannot live in a reclaimed band.  Route them
per consumer, as QSA3 was routed.

### 8.2 Headroom sweep (same config, 15 748-token prefill, greedy text identical where compared)

| `GGML_CUDA_SLAB_HEADROOM_MIB` | cache arena (reported) | mapped / boundary at the drop | free at wide-drop | prefill |
|---:|---:|---:|---:|---:|
| 4096 | 15 666.6 MiB | — (5.2k-token probe: 3925 MiB free) | 3925 | 787 t/s |
| 2048 | 17 744.8 MiB | 19840 / 8128 MiB | 1842 → **1942** (post-QSA3) | 1156 → 1104 t/s |
| 1024 | 18 783.9 MiB | 20864 / 8128 MiB | 818 → **918** (post-QSA3) | 1157 → 1105 t/s |

All runs used the same 15 748-token prompt (except the 4096 row's free figure, from the 5 256-token
probe — the deep 4096 run was not repeated this session).  The deep 1024 run's greedy output
(`sha256=df10d0b5f19eccb2`) is identical to the pre-QSA3 build, and the 35B-A3B A/B is identical to r80.

The field config therefore has **1.8-1.9 GiB free at the prefill peak even at headroom 1024**, i.e. the
remaining floor is not the headroom any more; it is the ~270 MiB pool remainder + hipBLASLt + MMB.  The
`-ub 6144 -c 204800` wide config (which rejected 3072 on 2026-10-10) was **not** re-run this session
(the IQ4_XS model is not on this box) — that is the gate that must pass before any default change.

### 8.3 Coherence

* 35B-A3B Q4_K_M, `-ncmoe 40`, `code-reasoning-mixed.txt`, seed 42, temp 0, `-n 64`: **greedy text
  identical** to the r80 baseline build (stash/rebuild A/B).
* Flash-Next GSQ, 15 748-token prompt, seed 0/temp 0, `-n 16`: **sha256 identical** before and after the
  QSA3 redirect (`df10d0b5f19eccb2`).

### 8.4 Next (in order)

1. Route the **MMB activation cache** (`mmb_state().cache`, a `ggml_cuda_pool_alloc` retained for the
   process) into the wide band.  It is prefill-only and ~40 MiB here; it needs a yield-on-reclaim rule
   because the cache is not RAII-scoped (or simply drop the cache at the wide drop and re-convert).
2. Identify the remaining ~270 MiB of pool high-water (`GGML_CUDA_SLAB_STATS` + `-ub 512` bisect), and
   route the prefill-only consumers one by one.  Anything that can run in a captured decode/verify graph
   stays out.
3. Re-run the wide-prefill matrix on a box with the IQ4_XS model (or the field config at `-ub 6144`), then
   decide the headroom default.  With the wide-phase transients absorbed, 2048 is the candidate; the data
   here says 1024 still has ~1.8 GiB at the peak on the 15.7k field prefill.
4. Consider the aux reserve: it is a flat 4096 MiB per draft, and the draft's *compute* is now narrow-2
   (inside the slab), so the estimate is ~512 MiB larger than the measured post-slab draft cost.

### 8.5 The WIP diff

`slab-transient-absorption-wip.patch` in this directory is `git diff <release.json.tip>..<wip commit>` from
`~/llama.cpp` (the tree that produced §8's numbers).  It is **exploration only**: it is cut against the
r80 tree, it is not part of `patches/`, and `apply-all.sh` does not read it.  Re-cut it against a fresh
`release.json.base` checkout before evaluating it as a promotion candidate, and re-run the §6 gates.

Recovery coordinates (2026-10-17 session, all reproduced before this line was written):

* fork working tree: `~/llama.cpp`, branch `rdna-boosts`, commit **`256f589df`**
  (`256f589df454e3dd699eac04094c1b5d0316a3b7`), parent `00f4b1ef8` = `release.json.tip`.
  The branch tree deliberately no longer equals `release.json.tree` (it carries this WIP commit).
  This commit is **local only** (the fork is never pushed except the release-time `rdna-boosts` refresh).
* patch: `slab-transient-absorption-wip.patch`, **852 lines, sha256 `513da9e0…36feb1`**
  (`513da9e053a0a22740e1456fa92f19f58583735703ace3e24002f6530336feb1`), byte-identical to
  `git diff 00f4b1ef8..256f589df` in the fork.
* to restore: `git clone` llama.cpp, `git checkout 097cf33d9` (or `00f4b1ef8` if the block chain is already
  applied), `git apply` the patch, then `~/bin/build-llama-rocm` (`build-rocm`, ROCm 10.1.0).


---

## 9. 2026-10-17 (later) — step 2 (partial) + step 4: aux and the load-time probe

Same box/config, 15 748-token prefill, `GGML_CUDA_SLAB_STATS=1`.

1. **Step 4 — `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` 4096 → 3584** (`include/llama.h`).  The old value
   predates narrow-2; the draft's measured post-slab cost is **~3447 MiB** (weights 2647 + KV 288 +
   narrow-2 compute 512), so ~650 MiB of the reserve was permanently unmapped and outside the arena.
   3584 keeps a ~137 MiB margin; a larger draft spills into the headroom, which is the budget for it.
2. **Step 2a — the H2D 512 MiB bandwidth calibration probe now comes from the slab arena**
   (`ggml-cuda-moe-cache-hooks.cu`).  It runs once at load, after the slab exists and before the cache
   sizes itself, and is freed immediately; it was the largest single outside spike (and a hard blocker
   for a low headroom).  Falls back to `cudaMalloc` when the slab is inactive.
3. **Negative result: `GGML_CUDA_POOL_CACHE_MB` is not a lever for the pool high-water.**  Capping it at
   64 MiB on the 15.7k prefill made the high-water *worse* (327.0 vs 271.4 MiB): the pool flushes and
   re-allocates, and the figure is dominated by **live** prefill temporaries, not cached freed blocks.
   The remaining pool bytes must be routed per consumer (QSA3 is done; the rest is step 2b).

Measured (cache arena is the slot budget; free is outside the slab):

| headroom / aux | cache arena | residency | free at wide-arm | free at wide-drop |
|---|---:|---:|---:|---:|
| 4096 / 4096 (r80 default) | 15 666.6 MiB | 38.3 % | ~2 470 | 1 842 |
| 2048 / 3584 | 18 224.4 MiB | 44.5 % | 1 958 | 1 430 |
| **1024 / 3584** | **19 263.5 MiB** | **47.1 %** | **934** | **408** |

The 1024 / 3584 row is the re-assessment of step 1: it survives the 15.7k prefill (1 103 t/s, no errors,
406-408 MiB left at the drop) and is **+3.6 GiB of arena vs the r80 default**, i.e. ~+2.5-3.5 pts of hit
rate once the decode hit rate is measured.  The prefill's own consumption is ~526 MiB (pool 271 + hipBLASLt
~104 + MMB ~40 + q8_1/etc ~110), so the remaining floor is that 526 MiB, not the headroom.  **Do not set a
default below 2048 until the `-ub 6144 -c 204800` wide matrix passes**; this box has no IQ4_XS model to
run it.  Remaining step 2b: route the MMB activation cache (~40) and the live prefill pool blocks (~270)
into the wide band, which would take the prefill demand to ~150-200 MiB and make 512 a candidate.

---

## 10. 2026-10-17 (later still) — 2b (targeted) + an 80K seesaw at headroom 1024

**2b, targeted rather than generic.**  The generic pool redirect was investigated and *deferred*: the pool is
not only RAII scratch -- `mmb.cu:1610`/`:1943` (activation cache + slots) and `allreduce*.cu` hold pool
allocations **across** graphs, so a generic redirect would serve them from a band that is reclaimed at the
prefill -> decode drop (a silent use-after-free).  It needs a "pinned / never-wide" escape hatch and those
sites converted first.  Instead the **chunked-GDN prefill scratch** (`gated_delta_net_chunked.cu`,
`A_sc`/`KQ_sc`) now uses the shared `ggml_cuda_wide_scratch<T>` helper (same guard as QSA3: slab live AND
the wide band armed, else the pool).  Measured: a **78.2 MiB** request per 4096-token ubatch now lands in
the band (it was pool high-water before).

**An 80K-token prefill at `GGML_CUDA_SLAB_HEADROOM_MIB=1024` + aux 3584** (the aggressive candidate):

| | value |
|---|---:|
| arena | 19 263.5 MiB (47.1 % residency) |
| prefill | 80 094 tokens in 68.7 s = **1 165 t/s** |
| free outside at wide-arm | 575.5 MiB |
| free outside at wide-drop | **331.5 MiB** |
| pool at wide-drop | 337.4 MiB (other prefill temps; GDN's 78 MiB is now in the band) |
| errors / aborts / corruption | 0 |
| greedy output | `sha=df10d0b5f19e` (identical to the earlier reference) |

So an 80K prefill at a 1 GiB headroom still leaves ~330 MiB outside.  **Caveat on the seesaw harness:**
short/long/short/long against one server did *not* re-run the second prefill -- the prompt cache served the
repeated prompts (only 4 arm/drop transitions, and 4 tokens of "prefill" on the repeats).  A real seesaw
needs distinct text per request or `--cache-ram 0`.

### 10.1 aux vs headroom: only their SUM is a resource

`reserve = headroom + aux`, and the draft allocates **after** the slab: weights 2 647 + KV 288 + narrow-2
512 = **3 447 MiB**, plus the post-slab transient peak (~350-500 MiB).  So:

  headroom + aux  >=  ~3 900-4 000 MiB     (the draft cannot be evicted)

`aux` alone can be 1536 (or anything) only if the headroom is raised to keep that sum -- the split changes
*which* part `slab_extend` can later map into the arena (it reclaims down to `headroom` only), so the best
split is **aux ~= the draft footprint (3584) and the headroom ~= the transient floor**, i.e. the next step
is headroom 512 / aux 3584 (sum 4 096), not aux 1536 / headroom 2048 (sum 3 584, which is *below* the
measured draft + transient need).

### 10.2 The real seesaw (distinct prompt per request), headroom 1024 / aux 3584

Each request now carries a unique `[variant N]` prefix, so the prompt cache cannot serve it — 4 genuine
arm/drop pairs.  Requests: short 251 tok, long 80 107 tok, short 249 tok, long 80 105 tok.

| transition | free outside | pool | note |
|---|---:|---:|---|
| short-1 arm / drop | 934 / 700 MiB | 6 → 21 | first prefill |
| long-1 arm / drop | 577 / **333** MiB | 38 → 337 | 1 165 t/s; pool high-water established |
| short-2 arm / drop | **208** / 208 MiB | 337 | short prefill reuses the pool |
| long-2 arm / drop | **204** / 308 MiB | 337 | 80K again, clean |

`long-1` and `long-2` produced the **same greedy continuation** (`sha=7194bf85ff0f`), i.e. the seesaw is
bit-stable.  No errors, no aborts, no corruption.  The minimum outside headroom across the seesaw is
**~205 MiB** (at the long-2 arm).

**Consequence for the headroom:** 1024 is the floor with today's transient set — a 512 headroom would put
that arm at **−307 MiB** and fail.  The remaining outside consumers are, in order: the **pool high-water
337 MiB** (the dominant one, and it is *live+cached* — see the `GGML_CUDA_POOL_CACHE_MB` negative result),
hipBLASLt ~104, MMB ~40.  So **the deferred generic pool redirect (with the pinned escape hatch) is worth
~300 MiB and is the only thing that makes 512 possible.**

### 10.3 The wide-prefill gate, run on IQ4_NL in place of IQ4_XS (2026-10-17)

The 2026-10-07 record rejected 3072 and corrupted at 2048 on `--fit off -sm tensor -ncmoe 48 -ub 6144
-c 204800` (IQ4_XS).  IQ4_XS is not on this box, so the gate was re-run on the same-architecture 4-bit
**IQ4_NL** (the 9-shard PROJFIX build) with the shared Q8_0 MTP draft, 6 657-token prefill.

| | headroom 2048 / aux 3584 |
|---|---:|
| SLAB_INIT | free 25.23 GiB, reserve 5.50 GiB, mapped 19.69 GiB, boundary 2.00, `va_spare` 3.50 GiB |
| arena slots | 17 718.8 MiB of 64 800 MiB host experts (**27.3 %**) |
| narrow-2 (ub 6144) | 768 MiB |
| free outside at wide-arm | 1 562 MiB |
| free outside at wide-drop | **629 MiB** |
| pool at wide-drop | **671 MiB** (grows with the ubatch; twice the GSQ figure) |
| prefill | 614 t/s (6 657 tokens), coherent output |
| errors / OOM / `hipblaslt.cpp:164` / corruption | **none** |

**So the gate passes at 2048 / 3584 on IQ4_NL**, which is the first time this wide config runs below 4096
without the failure mode the 2026-10-07 record described.  But it is also the ceiling for the wide config:
with 629 MiB left at 2048 and the pool alone at 671 MiB, **headroom 1024 would put the drop at ~-42 MiB**
— the generic pool redirect (the deferred piece) is a prerequisite for 512/1024 on wide ubatches, not just
a nicety.

**Recommendation from the data:** ship **2048 / 3584** as the default (safe on both the field `-ub 4096`
config and the wide `-ub 6144` config, +2.5 GiB arena vs the r80 default) and keep 1024 / 3584 as a
documented opt-in for `-ub <= 4096` field use, where the seesaw bottoms out at ~205 MiB outside.

### 10.4 Two GPUs, both split modes (2026-10-17)

Same GSQ + MTP field command, `HIP_VISIBLE_DEVICES=0,1`, headroom **2048 / aux 3584**, 5 262-token prefill,
24 greedy tokens.  `GGML_CUDA_SLAB_HEADROOM_MIB` is explicit, so it wins over the r36 `-sm layer` floor of
6144 (that floor was measured on a 255k `-ts 59,41` run -- not covered here).

| | `-sm tensor` | `-sm layer` |
|---|---|---|
| SLAB_INIT (both devices) | mapped 21.94 GiB, `va_spare` 3.50 GiB | mapped 23.12 / 22.81 GiB, `va_spare` 3.50 GiB |
| arena (total) | **40 725.1 MiB (99.5 %)** | **40 199.1 MiB (98.2 %)** |
| free outside at wide-drop | 2 118.5 MiB per device | 1 097.5 MiB per device |
| pool at wide-drop | 294 MiB | 279 / 393 MiB |
| prefill | 810.6 t/s | 763.0 t/s |
| greedy output | `sha=394193591d4d` | **same `sha=394193591d4d`** |
| errors / aborts | none | none |

Both modes run clean at the low headroom, and they produced **identical greedy text** on this prompt.
Caveat: the layer-mode headroom floor (r36, 6144) was for a 255k `-ts 59,41` config; a long-context
`-sm layer` run is the remaining gate before an explicit low headroom is blessed for that topology.

### 10.5 The wide gate at headroom 1536 (IQ4_NL, measured)

`-ub 6144 -c 204800`, 6 657-token prefill, 24 greedy tokens, aux 3584:

| | 2048 | **1536** | 1024 |
|---|---:|---:|---:|
| mapped | 20 160 MiB | **20 672 MiB** | (not run) |
| arena slots | 17 718.8 MiB (27.3 %) | **18 225.0 MiB (28.1 %)** | (not run) |
| free at wide-arm | 1 562 | **1 050** | — |
| free at wide-drop | 629 | **459** | — |
| pool at wide-drop | 671 | 335 | — |
| prefill | 614 t/s | 610 t/s | — |
| greedy sha | `746954a1038b` | **`746954a1038b`** | — |

**1536 is clean and comfortable on the widest config (459 MiB left), and the greedy output is identical to
the 2048 run.**  Note the pool figure is not repeatable (671 vs 335 MiB for the same config), so 1024 on
the wide config must be *measured*, not extrapolated (−43 MiB was the arithmetic from the 671 run; the 335
run would leave ~+120).  The next-session plan is `HANDOVER-2026-10-17-pool-redirect.md`.

---

## 10.6 2026-10-17 — step 2b: the generic workspace-pool redirect (landed, WIP)

The deferred half of step 2 (`HANDOVER-2026-10-17-pool-redirect.md`).  The workspace pool
(`ggml_cuda_pool_leg`) was the binding outside-slab consumer: its high-water is permanent for the run
(the pool caches every freed block), so ~335 MiB of it sat outside the slab on every prefill.

**What landed (fork WIP commit `3f4497d14`, still not in `patches/`):**

1. **Escape hatch first** — `ggml_cuda_pool::set_wide_routing(bool)` (base no-op, `pool_leg` overrides,
   returns the previous state) plus a tiny RAII guard `ggml_cuda_pool_wide_pin`.
2. **`pool_leg::alloc` serves a LARGE allocation from the wide band** when *all* of: the pool is not
   pinned, `size >= GGML_CUDA_POOL_WIDE_MIN_MIB` (default 1), `GGML_CUDA_POOL_WIDE` is on (default),
   no graph capture is active, and the band is available (slab live **and** the wide view armed).
   It is tried **before** the cached-block scan (reusing a cached block during a wide phase would keep
   the outside high-water alive for no gain).  The band block returns the *requested* size as
   `actual_size`, so `free` pops exactly what `alloc` pushed.
3. **`pool_leg::free` recognises its own band blocks** (a 64-entry side table, no slab call on the hot
   path): given back to the band (LIFO), never cached, never counted in `pool_size`/`cached_size`, never
   `cudaFree`d.  `clear_pool` deliberately leaves that table alone — it is called from the OOM retry
   *mid-graph*, where those entries describe LIVE blocks.
4. **Pinned sites** (the handover's list + one it missed):
   `mmb.cu` activation cache (`mmb_cache_insert`) and persistent slots (`ggml_cuda_mmb_slot_reserve`);
   `allreduce.cu` / `allreduce-hip.cu` `bf16_tmp[]`; the NCCL `bf16_tmp[]` in `ggml-cuda.cu`;
   and **`ggml_backend_cuda_context::bf16_src1_cache`** (`common.cuh`), a per-context BF16 copy kept
   across graphs — the leak the tripwire caught on the first run (`POOL_WIDE_LEAK blocks=2 bytes=2.4 MiB`).
   `mmf.cu:51-53` checked: function-local RAII around one kernel launch, **not** retained — left
   unpinned (it is exactly the kind of large prefill temporary the redirect is for).
5. **Tripwires + diagnostics.** `POOL_WIDE_LEAK` (unconditional warning) when a pool band block is still
   live at the prefill -> decode drop — that is the moment the bytes go back to the arena, so a non-zero
   count is a silent use-after-free waiting to happen; a one-shot warning when such a block is later freed
   from a dead band generation; `SLAB_STATS` now reports `pool_wide=<MiB>/<blocks>` and
   `refused=<MiB>` (bytes a large pool allocation asked for that the armed band refused — those stay
   outside as high-water).  `GGML_CUDA_POOL_WIDE_DEBUG=1` logs every band alloc/free and every
   `OUTSIDE` allocation with the reason.

**Two bugs this exposed, both fixed (the second was pre-existing and blocked the step):**

* **The pinned list was incomplete** (item 4): without the tripwire the `bf16_src1_cache` block would
  have been reused by the arena after the drop.  That is the whole reason the hatch exists.
* **`ggml_cuda_slab_wide_aux_alloc` refused every growth once the base retained a chunk gap.**
  Its post-eviction sanity check compared the *live boundary* against the aux **end**
  (`s.boundary > off`).  The base is set to the boundary *before* a move and the move then rounds the
  boundary up to `chunk`, so a retained base normally sits a whole chunk below it and every later growth
  was refused.  The aux then silently fell back to raw `cudaMalloc` outside the slab (measured: outside
  free VRAM at the 80K arm 577.5 -> 349.5 MiB, and the wide config OOMed at headroom 1536 on a 118 MiB
  pool allocation).  The check now compares against the **boundary snapshot** (`s.boundary != lo`).
  (A first attempt at "re-base when the boundary moved past the base" was wrong the other way — it
  ratcheted the boundary a chunk per allocation and ate the whole arena (measured: `arena=0.0 MiB`,
  boundary == mapped).  The retained base below the boundary is legitimate; the invariant is that it must
  stay above every *live view's* top, which `work_alloc` already enforces.)

### Measured (same box, 1x R9700 gfx1201; `GGML_CUDA_SLAB_STATS=1`)

**Field GSQ seesaw, headroom 1024 / aux 3584** (4 genuine arm/drop pairs: distinct `[variant N]` prefix
per request, 251 / 80 107 / 249 / 80 105 tokens, one server):

| transition | before (§10.2) | with 2b |
|---|---:|---:|
| short-1 arm / drop | 934 / 700 | 934 / 709.5 |
| long-1 arm / drop | 577 / 333 | 587.5 / **415.5** |
| short-2 arm / drop | 208 / 208 | 290.5 / 290.5 |
| long-2 arm / drop | **204** / 308 | **286.5** / 390.5 |
| pool at the long drop | 337.4 | **262.9** |
| prefill | 1 165 t/s | **1 247 t/s** |
| greedy shas (short-1/long-1/short-2/long-2) | 130d51af8d12 / 7194bf85ff0f / 8db6010f8533 / 7194bf85ff0f | **identical** |

So the outside floor at headroom 1024 moves **204.5 -> 286.5 MiB** (+82) and the drop **333.5 -> 390.5**
(+57).  No `POOL_WIDE_LEAK`, no stale-free warning, `refused=0`.

**Wide gate (IQ4_NL PROJFIX + shared Q8_0 MTP draft, `-ub 6144 -c 204800`, 6 657-token prefill, aux 3584):**

| headroom | before | with 2b | pool | sha |
|---:|---|---|---:|---|
| 1536 | free 459.0 | **477.0** | 335.2 -> **315.6** | `746954a1038b` (same) |
| 1536, `GGML_CUDA_POOL_WIDE=0` | 459.0 | 459.0 / 335.2 | (kill switch reproduces the baseline exactly) | same |
| 1024 | (predicted ~-43) | **OOM** | — | — |

Headroom **1024 fails on the wide config**: a pinned ~80 MiB cache allocation could not be placed
outside the slab (free VRAM was ~80 MiB by then) and the arena, being inside the slab, cannot yield
physical back.  `ggml_cuda_device_malloc` reported `83.92 MiB allocation failed`.  So **1536 remains the
floor for `-ub 6144`**, exactly as §10.5's arithmetic warned — the redirect softens it, it does not fix it.

**Two GPUs (GSQ + MTP, 5 262-token prefill, headroom 2048 / aux 3584):**

| | before (free/device, pool) | with 2b | greedy |
|---|---|---|---|
| `-sm tensor` | 2 118.5, 294 | **2 238.5**, 176.2 | `394193591d4d` (same) |
| `-sm layer` | 1 097.5, 279 / 393 | **1 349.5**, 249.9 / 146.5 | `394193591d4d` (same) |

Both clean, arena unchanged (40 725.1 / 40 199.1 MiB), prefill 829 / 769 t/s.

**Prefill-logit KLD gate (issue #113): PASS.**  Base recorded by the **pre-change** build on a
*slab-active* config (35B-A3B Q4_K_M, `-sm tensor -ncmoe 40`, so the redirect actually fires; the
script's dense auto-model has no slab and would be a vacuous run): **mean KLD 0.000000**, same-top-p
**100.0 %**.  `test-backend-ops` not re-run: no op math, dispatch or instance set is touched.

**35B-A3B coherence** (`-ncmoe 40`, `code-reasoning-mixed.txt`, seed 42, `-n 64`): greedy text
**identical** to the pre-change build (the printed `Generation:` differs by ~1 % in both directions across
repeats — the redirect is inert whenever the band is disarmed, i.e. all of decode).

### The finding: the pool high-water is now the MMB / BF16 caches, not prefill temporaries

Everything above is what the *pinned* design buys, and it is not the definition of done (§5 of the
handover: pool <= ~80 MiB, seesaw floor >= ~500 MiB at 1024).  The remainder is retained **caches**:

| arm | pool at drop | free at drop | floor (seesaw) |
|---|---:|---:|---:|
| GSQ seesaw @1024, 2b | 262.9 | 390.5 | 286.5 |
| GSQ seesaw @1024, 2b + `GGML_CUDA_MMB=0` | **95.2** | **589.5** | **452.5** |
| IQ4_NL wide @1536, 2b | 315.6 | 477.0 | — |
| IQ4_NL wide @1536, 2b + `GGML_CUDA_MMB=0` | **145.2** | **655.0** | — |

(`GGML_CUDA_MMB=0` is a **diagnostic A/B**, not the proposal: it changes the GEMM path and the greedy
sha, and it costs prefill.  It measures `MMB's retained activation cache` = **~168-170 MiB** live at the
drop; the rest of the residual is the `bf16_src1_cache` (~80 MiB) plus the sub-1 MiB pool tail.)

Both of those are **pure caches** — the MMB activation cache is rebuilt on demand and the bf16 src1 cache
is already dropped at every graph start.  So the safe extension is a **yield on reclaim**: let both use
the band normally and flush them in the same device callback that disarms the ring
(`ggml_backend_cuda_device_slab_ring_set(dev, false)` -> `ggml_cuda_mmb_cache_yield(device)` /
`bf16_src1_cache_clear()`), *before* `ggml_cuda_slab_ring_set(device, false)` releases the bytes.  That
is worth **~+200 MiB at the drop / ~+170 MiB of floor**, i.e. it is the only route to the DoD — and it is
**deferred for the maintainer's decision** (it changes cache lifetime across a delivery block, and it
needs its own prefill-cost measurement: every prefill re-converts the evicted activations).

**Recommendation unchanged:** keep **2048 / 3584** as the default and 1024 / 3584 as the documented
opt-in for `-ub <= 4096` field use (now with a 286 MiB floor instead of 205, still the measured floor).
`-ub 6144` needs 1536 and nothing here changes that.  **No delivery default was changed and nothing was
promoted** — this is still a WIP campaign.

**Kill switches / knobs:** `GGML_CUDA_POOL_WIDE=0` (whole redirect off), `GGML_CUDA_POOL_WIDE_MIN_MIB`
(size floor, default 1), `GGML_CUDA_POOL_WIDE_DEBUG=1` (per-allocation trace).

**Recovery coordinates:** fork `~/llama.cpp`, branch `rdna-boosts`, WIP commit
**`3f4497d14`** (`3f4497d142ee5c957e8cddb76f5df411c2473177`, tree `97df52a6ab0f`), parent `256f589df`;
local only (never pushed).  `slab-transient-absorption-wip.patch` is `git diff 00f4b1ef8..3f4497d14`,
**1 225 lines, sha256 `e1a6b843b3e43cb51d04d516e6a6a344ec7bcca41a7d6f373d49fa57071c13a9`**.  Restore as
in §8.5 (`git checkout 00f4b1ef8`, apply the patch, `~/bin/build-llama-rocm`).

---

## 10.7 2026-10-17 — what `aux = 1536` actually costs (measured), and the MMB/BF16 yield scope

Scoped the follow-up ("absorb the MMB activation cache + `bf16_src1_cache` so a lower reserve default
becomes safe") — plan + design + gates in
[`SCOPE-2026-10-17-mmb-yield-aux1536.md`](SCOPE-2026-10-17-mmb-yield-aux1536.md).  Measuring the *target*
first changed the plan, so the numbers are here.

Same field GSQ seesaw (distinct prefixes, one server, 251/80 107/249/80 105 tokens), `GGML_CUDA_SLAB_STATS=1`:

| reserve (`headroom`/`aux`) | sum | draft load | arena (2nd arm) | free arm-1 / drop-1 / arm-2 | outcome |
|---|---:|---|---:|---|---|
| 1024 / 3584 | 4608 | ok | 19 328 MiB | 934 / 709.5 / 286.5 | **clean** (§10.6) |
| 2048 / 2048 | 4096 | ok (barely) | **20 096 MiB** | 422 / 203.5 / **93.5** | OOM `84.00 MiB` |
| 2560 / 1536 | 4096 | ok (barely) | **20 096 MiB** | 422 / 203.5 / **93.5** | OOM `84.00 MiB` |
| 2048 / 1536 | 3584 | **FAIL `512.00 MiB`** | (20 302.6 MiB) | — | draft-load abort |

1. **Only the SUM matters, not the split.**  `2048/2048` and `2560/1536` are identical to the byte
   (422.0 / 203.5 / 93.5, arena 20 096, same eviction pattern): `reserve = headroom + aux` is the only
   place either number enters (`ggml_cuda_slab_reserve_bytes`), and `slab_extend` cannot fire while the
   draft is consuming the reserve.  So **`aux = 1536` means `headroom ~= 2560`** (sum 4096); sum 3584
   does not even load the draft.
2. **At sum 4096, the blocker IS the MMB activation cache** — the abort is `84.00 MiB`, i.e. a pool
   request of `80 MiB + 5 %` (one MMB cache entry, the same 80 MiB the §10.6 `GGML_CUDA_MMB=0` A/B
   isolated) with 93.5 MiB free at the wide arm.  Absorbing ~250 MiB is exactly what makes 4096 viable.
3. **But `aux 1536 / headroom 2560` is not the arena-maximising direction**: the headroom is permanently
   outside the slab, so the campaign's original shape (`aux` ~= draft, `headroom` ~= the transient floor,
   i.e. `3584/512` at the same sum 4096) is strictly better *if* 512 is safe.  `aux 1536` is the
   "keep a 2.5 GiB transient cushion" variant — +1.5 GiB arena vs the r80 default, same arena as
   `3584/512`.  Decision D1 in the scope note.
4. Also recorded there: the outside free **degrades across the seesaw** (934 -> 587 -> 290 -> 286) and
   then stabilises, i.e. the minimum is a steady state ~650 MiB below the first arm, not a prefill peak;
   the steady-state outside demand at reserve 4608 is ~875 MiB, of which the two caches are ~250-315.
   The unattributed ~300-500 MiB is the next decomposition if the yield alone is not enough (decision D2).

No code landed for this step: it is a scoping note plus the measurements that shape it.

---

## 10.8 2026-10-17 — step 2b-2: the per-graph cache yield landed (MMB + BF16 src1)

The yield scoped in [`SCOPE-2026-10-17-mmb-yield-aux1536.md`](SCOPE-2026-10-17-mmb-yield-aux1536.md) §3-§4,
**implemented** (fork WIP commit `f639cb646`, still not in `patches/`).

**What landed:**

1. **`per_graph_cache_yield`** — an RAII object in `ggml_backend_cuda_graph_compute`: on scope exit (every
   return path) it calls the new `ggml_cuda_mmb_yield_caches(ctx)` and `bf16_src1_cache_clear()`.  Both
   caches were *already* cleared at the start of the next compute, so this loses no hit — it just moves the
   free to before the graph boundary, so **nothing retained is live across a graph**, which is what makes the
   wide band safe for them.  The MMB yield is scoped to the calling context (it looks the context up in
   `g_mmb_state` directly, not through `g_mmb_active_ctx`), so a concurrent backend context's live slots are
   never freed from another thread.  The shadow weights are untouched.
2. **The three `ggml_cuda_pool_wide_pin` guards are gone** (`mmb.cu` cache + slots, `common.cuh` BF16 src1):
   the two caches are band-eligible again and are yielded before the prefill -> decode drop.
3. **The wide-aux region tracks its LIVE blocks** (`ggml-cuda-slab.cu`).  This is the bug the first attempt
   exposed and it is worth recording: a bump stack that only pops its *exact top* is fine for per-op scratch
   (strictly LIFO), but the MMB/BF16 blocks live for a whole graph and are released *as a group* — so
   out-of-order frees become the norm and every one of them leaves a permanent hole.  Measured on **one**
   6 657-token prefill: **7 392 MiB of `wide_aux`, 149 of 2 135 frees unable to pop**, the region walked to
   the mapped top (`work == mapped`, **arena 20 096 -> 4 992 MiB**, 417 GiB of `refused` re-falls outside the
   slab).  A "free the group highest-address-first" batch did not help (the hole is *below* a still-live FA
   staging block).  The fix is to keep `off -> size` for every live block: `wide_aux_size` is then the extent
   of the **highest live block**, so holes below it are bounded and cost nothing, and any free order returns
   the region to the live high-water.  A free also takes the block's size from the map, so a mismatched free
   can no longer corrupt the stack.  **`SLAB_STATS` now reports `wide[n_pop= hole= inval= rebase=]`** — that
   diagnostic is what found it, and `hole=0` is the acceptance signal.

### Measured (same box; `GGML_CUDA_SLAB_STATS=1`; all greedy shas identical to the r80 baseline)

**Field GSQ seesaw, 1024 / 3584** (4 arm/drop pairs, distinct prefixes):

| | pre-2b (§10.2) | 2b (§10.6) | **2b-2** |
|---|---:|---:|---:|
| min outside (long-2 arm) | 204.5 | 286.5 | **522.5** |
| long-1 drop / long-2 drop | 333 / 308 | 415.5 / 390.5 | **651.5 / 626.5** |
| pool at the drop | 337.4 | 262.9 | **23.6** |
| `pool_wide` / `refused` / `hole` | — | 0 / 0 / — | **0 / 0 / 0** |
| prefill (80 105 tok) | 1 165 t/s | 1 247 | 1 238 |
| decode (16 tok, per request) | 31.7 / 43.8 / 49.6 | — | 31.9 / 42.8 / 49.5 |
| greedy shas | (baseline) | identical | **identical** (`130d51af8d12` / `7194bf85ff0f` / `8db6010f8533` / `7194bf85ff0f`) |

**Wide gate (IQ4_NL PROJFIX, `-ub 6144 -c 204800`, 6 657-token prefill, aux 3584):**

| headroom | pre-2b | 2b | **2b-2** |
|---|---:|---:|---:|
| 1536: free at drop / pool | 459.0 / 335.2 | 477.0 / 315.6 | **791.0 / 6.4** |
| 1024: free at drop | **OOM** | **OOM** | **279.0** (repeat: 279.0) |
| greedy sha | `746954a1038b` | same | **same** (both headrooms) |
| prefill | 610 t/s | 618 | 619 |

**2 GPUs (GSQ + MTP, 5 262-token prefill, 2048 / 3584), free at the drop per device / pool:**

| | pre-2b | 2b | **2b-2** |
|---|---|---|---|
| `-sm tensor` | 2 118.5 / 294 | 2 238.5 / 176.2 | **2 384.5 / 32.8** |
| `-sm layer` | 1 097.5 / 279,393 | 1 349.5 / 249.9,146.5 | **1 489.5 / 4.5,7.7** |

Both clean, same greedy sha `394193591d4d`, no `POOL_WIDE_LEAK`, no dead-generation warning.

**35B-A3B coherence**: text identical to the pre-change build (decode 83.3 t/s, noise).
**Prefill-logit KLD gate**: **mean KLD 0.000000, same-top-p 100 %** against the base recorded by the
pre-change build (slab-active 35B + `-ncmoe 40` config).

### What this changes for the `aux = 1536` target (§10.7)

The MMB cache *was* the blocker, and it is gone.  `aux 1536 / headroom 2560` (sum 4096) now **completes the
full seesaw** (pre-yield: OOM on an 84 MiB = 80 MiB + 5 % MMB request):

| reserve | min outside | drops | pool | shas |
|---|---:|---|---:|---|
| **2560 / 1536** (sum 4096) | **52.5** | 151.5 / 52.5 / 126.5 | 14.2 | identical |

So `aux 1536` is **technically reachable**, but that floor is **52.5 MiB** — one more ~50 MiB outside
allocation from failing, and the pre-yield `refused`/`free=0` failures show how close that is.  It is **not**
a default anyone should ship on one run: it needs the reserve matrix in the scope note (§6.1-6.3: repeats,
the wide config, 2 GPUs, and the never-run long-context `-sm layer`), and the "steady-state outside demand"
decomposition (§10.7 finding 4) is still ~600 MiB at sum 4096, i.e. the cushion is thin by measurement, not
by luck.  **No default was changed; nothing was promoted.**

**Recovery coordinates:** fork WIP commit **`f639cb646`** (`f639cb646afbbf8b6b964cf9d05d30a4b55c7591`, tree `51e176e04b47`), parent `3f4497d14`;
`slab-transient-absorption-wip.patch` = `git diff 00f4b1ef8..f639cb646`, **1 353 lines, sha256
`7807792c977b7ccfb9fca74ffdb02e1016d007bf30adde480e424760a3f7b1d1`**.

---

## 10.9 2026-10-17 — the reserve is now computable: `reserve = D + T + excess`

With the transients absorbed (§10.6-§10.8) the outside demand separates into **two measured terms** plus a
margin.  Measured directly this time: the same wide config **with and without the MTP draft**, reserve
forced to 4608 (`GGML_CUDA_SLAB_RESERVE_MIB`), `-ub 6144 -c 204800`, 1 GPU:

| arm (reserve 4608) | free at the wide-arm | free at the drop | pool | sha |
|---|---:|---:|---:|---|
| with the draft | 538.0 | 279.0 | 6.4 | `746954a1038b` |
| **no draft** | **4538.0** | **4398.0** | 4.6 | `196f24c1e76d` (expected: no draft) |

So **D = the draft's footprint = 4000 MiB** (`-c 204800`, `-ub 6144`: weights 2 647 + KV ~450 + narrow-2
768 + ~135 transients).  That is **+416 MiB over the 3584 MiB `aux` default** — i.e. the "unattributed
400-600 MiB of outside demand" chased in §10.7/§10.8 is the *draft itself*, not a stray transient.  The
3584 default is exactly the **field** draft's footprint (2 647 + 288 + 512 + ~135 = 3 582), which is
presumably how it was derived; it is too small for the same draft at a longer context / wider ubatch.

**T = our own steady-state outside demand**, computed as `(reserve - D) - min_free`:

| config | reserve | min free | T |
|---|---:|---:|---:|
| field seesaw (`-ub 4096`) | 4 096 | 52.5 | 460 |
| field seesaw | 4 608 | 522.5 | 502 |
| field seesaw (default aux 1536 + headroom 4096) | 5 632 | 1 546.5 | 502 |
| wide, single prefill (`-ub 6144 -c 204800`) | 4 608 | 279 | 329 |
| wide, single prefill | 6 144 | 1 815 | 329 |
| wide, **no draft**, single prefill | 4 608 | 4 398 | 210 |

T is **~500 MiB for a 4-request session** and **~210-330 MiB for a single prefill**, and it is *stable
across reserves* (502 at three different reserves) — so it is a real demand, not an artefact of the split.
Its measured components: **hipBLASLt 104 MiB** (`ROCBLAS_USE_HIPBLASLT=0` -> drop 279 -> 383, identical
sha; the only external, library-chosen term), the Q8_1 arena (32 MiB floor, doubling ladder, **0 measured
growth** — `GGML_CUDA_Q8_1_ARENA_FREE_OLD=1` changed nothing), the cuBLAS workspace 4 MiB, the pool's
decode-phase high-water 4.6-33 MiB, and the remainder (~350 MiB) unenumerated but bounded by the
measurement above.

### The formula, and what it says about `aux`

```
D      = draft weights + KV(n_ctx) + narrow-2(ubatch) + ~135 MiB      (computable at load; measured 3584 / 4000)
T      = 512 MiB                                                     (measured ceiling 502; 104 of it rocBLAS)
excess = the cushion we want outside the slab                         (e.g. 300)
reserve = headroom + aux  =  D + T + excess
```

| config | D | computed reserve | today |
|---|---:|---:|---|
| field `-ub 4096 -c 131072` | 3 584 | **4 396 -> 4 608** | 4 608 opt-in / 7 680 default |
| wide `-ub 6144 -c 204800` | 4 000 | **4 812 -> 4 864** | 4 608 opt-in / 7 680 default |

Three consequences:

1. **The wide config needs ~4.9 GiB, not 4.6** — which is exactly why it showed 279 MiB at 1024/3584 (just
   under a 300 MiB excess) and OOMed there before §10.8.  The field config at 4 608 already carries a
   **522 MiB** excess, i.e. the 1024/3584 opt-in is *not* at the edge for `-ub <= 4096`; it is at the edge
   *for the wide config only*.
2. **`aux` is not the right variable** — `D` scales with (n_ctx, ubatch) through the draft's KV and its
   narrow-2 compute (512 -> 768 between `-ub 4096` and 6144), so if a rule keyed on the ubatch is wanted at
   all, the defensible one is **`aux >= D(n_ctx, ubatch) - headroom + excess`**, computed from the draft's
   own GGUF/params — not the earlier "1536 for `-ub <= 4096`, 2048 above" heuristic, which happens to point
   the right way only because a wider ubatch makes narrow-2 bigger.
3. **The excess can be *verified*, not just estimated**: `ggml_cuda_slab_extend` already measures the exact
   post-draft free VRAM.  A one-line check there (`free_b < excess_floor` -> warn, naming D/T/excess) turns
   the guarantee into something every run reports.  Note the asymmetry that makes this the *only* guarantee
   available: the failure mode of a thin outside is an **abort** (`cudaMalloc` fails and the slab cannot
   unmap sub-ranges to help), so there is no graceful degrade to fall back on.

**Not implemented, not proposed as a default** — this section is the model + the measurements; the reserve
matrix (scope note §6) decides the numbers, and the draft-derived `D` is the piece worth building first
(it is also the fix for the `-c 204800` under-reserve above).

---

## 10.10 2026-10-17 — step 1 of the computable reserve: the draft reserve is DERIVED

`LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` is no longer the source of truth for the slab's auxiliary
reserve: it is derived from the draft's own GGUF (fork WIP `d6938507d`; `common/draft-reserve.{h,cpp}` +
the wiring in `common_init_from_params`).  Nothing is in `patches/`; the constant remains the fallback and
`MOE_EXPERT_CACHE_AUX_RESERVE_MIB` remains the override.

**Terms** (each documented in the header, all measured/derived, none guessed):

| term | how | notes |
|---|---|---|
| weights | Σ GGUF tensor bytes, exact | the dominant term (2 647 MiB of the 3 584 MiB measured) and the part that differs per draft model |
| kv | `n_ctx` × (k + v) per KV layer at the **real** element size of `-ctk`/`-ctv`, ×1.10 for row/mask padding | KV layers = `nextn_predict_layers` when the GGUF declares an MTP head, else `block_count`; a declared **indexer** key cache is added at f32 (the QSA/indexer cache is not covered by `-ctk`) |
| compute | measured tier keyed on the ubatch: **512** MiB at `-ub <= 4096`, **768** above | the draft's compute buffer (pinned into the slab's narrow-2) still draws its *mapping* from the outside pool -- this is the one term that genuinely scales with the ubatch |
| slack | 192 MiB | the draft's own load/steady transients (measured ~135) |

**Calibration** against the measured draft footprint (§10.9: `D` = 3 584 field / 4 000 wide):

| config | weights | kv | compute | slack | **estimate** | measured D | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| field `-ub 4096 -c 131072` | 2 647 | 431 | 512 | 192 | **3 782** | 3 584 | **+198 (5.5 %)** |
| wide `-ub 6144 -c 204800` | 2 647 | 674 | 768 | 192 | **4 281** | 4 000 | **+281 (7 %)** |

The bias is deliberately upward (an under-estimate is the dangerous direction: the draft eats the cushion
and an outside `cudaMalloc` aborts).  The constant was exactly the *field* draft, so the same draft at
`-c 204800` used 3 584 and ran with a 3 680 MiB cushion where the headroom said 4 096 -- i.e. it silently
over-committed the arena.  That is now fixed by construction.

**Fallbacks** (all → the constant, logged): empty draft path (an embedded / shared MTP head, whose weights
are the target's), unreadable GGUF, or a required key missing.  `-c 0` estimates with the model's *trained*
context rather than 0 (a 0 here would under-estimate the KV term).

### Measured effect (aux 3 584 → derived; same headroom, same shas)

| run | reserved aux | outside at the arm/drop | arena | sha |
|---|---:|---|---:|---|
| field seesaw `1024/3584` | 3 584 → **3 782** | 522.5 → **714.5** / 626.5 → 818.5 | 19 328 → 19 136 | identical (4/4) |
| wide IQ4_NL `1536/3584` | 3 584 → **4 281** | 1 754 / 791 → **1 495** | 12 224 → 11 520 | identical |
| 2-GPU `-sm tensor` `2048/3584` | 3 584 → **3 782** | 2 384.5 → **2 576.5** | 16 256 → 16 064 | identical |
| 2-GPU `-sm layer` `2048/3584` | 3 584 → **3 782** | 1 489.5 → **1 681.5** | 16 256 → 16 064 | identical |

Prefill 617-621 t/s (unchanged), 35B-A3B coherence identical (that config has no draft, so it takes the
constant path).

**Read this as a correctness step, not an arena win.**  With the *current* headroom default (4 096) a
correct aux makes the reserve grow by the same amount, so the arena *shrinks* by 192 MiB (field) / 704 MiB
(wide) until step 2 lowers the headroom: the formula is `reserve = D + excess`, so `aux = D` is only a gain
in arena *together with* `headroom = excess` (≈ T + 300 ≈ 800-1 024).  That pairing is what the reserve
matrix (scope note §6) has to bless — and it is where the ~2.7-3.3 GiB of arena is.

### The `--fit` hole this exposed (for step 2)

`common_fit_cache_reserve::margin()` reserves `max(base, slab_headroom) + arena_budget` per device and
**does not include the draft at all** (no aux term).  So `--fit` sizes the context as if the draft did not
exist, and the draft is accounted for only by the slab's reserve — i.e. by silently shrinking the arena,
never by shrinking the context.  A derived `D` is the prerequisite for putting it into `--fit`'s margin
properly; that is the next thing to design, together with the multi-device (2/3-GPU `-sm layer`/`tensor`)
per-device split of `D` (the draft's weights are per-device tensor-split, its KV is per-device, its
narrow-2 is pinned per device).

**Recovery:** fork WIP `d6938507dd5446f0687609d5d6eb28f96ab7050b`, patch **1 691 lines, sha256
`bfa96cd4eb02c7a5f8be1cd870bd5bd1dac4789438e40d3ae888661ac154ab50`** (`git diff 00f4b1ef8..d6938507d`).

---

## 10.11 2026-10-17 — step 2 (fit): `--fit` now reserves the MTP draft

**Correction to §10.10.**  The claim there ("`--fit` does not include the draft at all") was right about the
*effect* and wrong about the *mechanism*: `--fit` does have an exact per-device draft accounting
(`common_fit_extra_model` -> `common_get_device_memory_data` measures the draft's model + context + compute,
with `shares_model` zeroing the weights for an embedded head).  But for an **MTP** draft that measurement
cannot work, and the failure is silent:

```
operator(): getting device memory data for the extra model at a context size of 262144:
llama_init_from_model: failed to initialize the context: qwen4exp requires ctx_other to be set
operator(): failed to measure the memory of the extra model, fitting without it
```

The draft context needs `ctx_other`, and `--fit` runs *before* the target context exists — so the throw is
unavoidable, the catch path set the draft's memory to **zero**, and fit sized the target as if the draft were
free.  The draft then had to come out of the slab's reserve, i.e. the outside cushion collapsed by `D` — and
with a lowered headroom that is the `cudaMalloc` abort.  That is the real `--fit` hole, and it is the one
"gnarly" interaction: *the reserve and `--fit` must agree on `D` or the projection is fiction.*

**Fix** (fork WIP `aec734be8`, ~60 lines, 3 files):

1. `common_fit_extra_model` gains `reserve_bytes` — the derived reserve from `draft-reserve.h` (§10.10), used
   **when the extra model cannot be measured**; `0` keeps the old behaviour.
2. Both failure paths use it: the per-device one (`add_extra_memory`) splits the weights + KV by the tensor
   split and adds the compute/transient tier to **every** device (the draft's narrow-2 exists per device);
   the Meta-device one (`extra_estimate`, the `-sm tensor` case) returns the total via a shared helper.
3. The draft reserve is derived **once** in `common_init_from_params`, *before* both `--fit` and the MoE
   preflight, so the projection and the slab reserve use the same number (the preflight no longer re-derives).
4. `n_ctx == 0` ("from the model") now resolves inside the estimator to the draft GGUF's declared
   `context_length` — the previous `0` would have under-estimated the KV term, the dangerous direction.

**Crash found on the way:** the fallback must `return` after filling in the estimate — the pre-existing code
below it dereferences `measured.back()` on the (empty) failed measurement.  It segfaulted inside `--fit`
before the `return` was added.

**Measured** (`--fit on`, `-c 0`, no user `-ngl` so fit actually runs):

| run | fit's draft reserve | outside at the drop | arena | sha |
|---|---:|---|---:|---|
| wide IQ4_NL, `-ub 6144`, headroom 1024 | **4 469 MiB** (was 0) | 1 007 | 11 072 | `746954a1038b` (same) |
| 2-GPU `-sm tensor`, `-ub 4096`, headroom 2048 | **4 213 MiB** (was 0) | 2 596.5 / device | 14 592 | `394193591d4d` (same) |

Both paths (per-device and Meta) fire and both are now accounted; prefill 618 t/s (wide).  The 2-GPU value is
lower than the wide one by exactly the compute-tier difference (512 vs 768) — the ubatch-keyed term from
§10.10 doing its job.

**What is left for `--fit` is now only the *numbers*, not the plumbing:** with `D` derived and reserved in
both places, the remaining decision is machine-wide — `reserve = D + headroom` per device, so the headroom is
the outside cushion and `--fit` sizes `n_ctx` around it.  That is the reserve matrix (scope note §6), and it
is now the only thing standing between this work and a lowered default.

**Recovery:** fork WIP `aec734be88c1187305371c67a3c5ae4908c90340`, patch **1 856 lines, sha256
`ecfcb97c4503a2bcad3445fe8719b1b2aa941173e291f15eba4eb3299540a8e0`** (`git diff 00f4b1ef8..aec734be8`).

---

## 10.12 2026-10-17 — step 3: the targets are set (`headroom` 4096 -> 1024) — campaign complete

With the derived draft reserve (§10.10) and `--fit` reserving the draft (§10.11), the headroom is the **only**
knob left, and it *is* the outside cushion: `reserve = D + headroom`, `arena = free - reserve` (while the
draft consumes the reserve), and `--fit` sizes `n_ctx` around `D + max(base, headroom)`.

### The measured matrix (derived aux, `GGML_CUDA_SLAB_STATS=1`; "excess" = free outside the slab at the worst arm/drop of the run)

| headroom | field GSQ seesaw (`-ub 4096`, 4 req) | wide IQ4_NL (`-ub 6144`) | 2-GPU `-sm tensor` | 2-GPU `-sm layer` |
|---:|---|---|---|---|
| 512 | floor **214.5** MiB | drop **471** MiB | — | — |
| **1024 (shipped)** | floor **718.5** MiB | drop **983** MiB | **1 552.5** MiB/dev | **657.5** MiB/dev |
| 2048 | — | — | 2 576.5 MiB/dev | 1 681.5 MiB/dev |

* **1024 is the default.**  Every measured config keeps **>= ~657 MiB** of excess (the target was ~300) and
  the field reserve drops **7 680 -> 4 806 MiB**, i.e. **+3.4 GiB of arena** (SLAB_STATS `arena` at the
  wide-arm: 15 666.6 -> **19 136** MiB; the r80 default was 4096 headroom + 3584 aux).
* **512 was measured and rejected:** the field seesaw bottoms out at 214.5 MiB, under the 300 MiB target.
  1024 keeps ~2x the worst measured `T` (§10.9: 502 MiB over a 4-request session, 210-330 single-prefill),
  which is the margin that matters because a thin headroom can fail *silently* (the fit-slab-accounting
  §2.2 `!!!!!!!!` case) rather than loudly.

### Verified at the new default, with **no env set**

| run | excess | arena | prefill / decode | greedy |
|---|---:|---:|---|---|
| field GSQ seesaw (distinct prompts, 4 arm/drop pairs) | **718.5** MiB | 19 136 MiB | 1 238 t/s | 4/4 shas identical |
| wide IQ4_NL `-ub 6144 -c 204800` (6 657-token prefill) | **983** MiB | 12 032 MiB | 619 t/s | `746954a1038b` |
| 2-GPU `-sm tensor` | 1 552.5 MiB/dev | 17 088 MiB | — | `394193591d4d` |
| 2-GPU `-sm layer` | 657.5 MiB/dev | 17 088 MiB | — | see the finding below |
| 35B-A3B coherence (no draft) | — | — | gen 91.8 t/s | **identical** |
| `--fit on -c 0`, wide formula | fit reserves `D`, outside 1 007 MiB | 11 072 MiB | 618 t/s | `746954a1038b` |

**Two notes on what did *not* need doing:**

* **A fully resident model (no `-ncmoe`) has no slab and no arena at all** — `moe_host_expert_bytes` is
  empty, so the preflight never runs (`-ncmoe 0` on 2 GPUs: not one `SLAB_*` line).  There is nothing to size
  in that mode.
* **The 6144 `-sm layer` floor is raise-only and unset-env only** (`eff = (!env_set && min > mib) ? min : mib`),
  so lowering the default cannot weaken it.  It lives in the *host-expert* path (`alloc_all_locked`), so it
  applies to a partially host-resident multi-device `-sm layer` run, not to the fully resident one.

### Finding (new, pre-existing, worth knowing)

**The MoE expert-cache hit/miss changes the numerics.**  The 2-GPU `-sm layer` greedy text changed when the
*arena size* changed (a lower headroom -> a bigger arena -> more cache hits), and it returns to the reference
sha `394193591d4d` when `MOE_EXPERT_CACHE_MIB` is pinned to the arena the reference run had.  So "identical
greedy sha across reserve values" is only a valid gate when the **arena is held constant** — a cached expert
and a streamed expert do not round identically.  This predates the campaign (any VRAM difference already moved
the arena), but it is the first time it was isolated, and it means the reserve matrix's sha checks are
*arena-controlled*, not reserve-controlled.

### State of the campaign

All four steps are landed in the fork WIP tree (`9c46d14df`, patch `70dd2283…`):
step 0 (ring slab-only) + wide-aux region + FA/QSA3/GDN/probe/pool redirects (§8-§10.6), the per-graph cache
yield (§10.8), the derived draft reserve (§10.10), `--fit` reserving the draft (§10.11), and now the headroom
default (§10.12).  **Nothing is in `patches/` and nothing is promoted** — promotion is a separate step: re-cut
the WIP against `release.json.base`, run the release gates (`gate-prefill-logits.sh`, MTP 4-axis, the
`llama-batched-bench -npl 1,4,8` decode gate), then cut the block with the maintainer's go-ahead.

**Recovery:** fork WIP `9c46d14dfc4b08be2cd0feb8582ed3df705bc264`, patch **1 916 lines, sha256
`70dd2283d27077caac14bb7d05a673bb2ab82abd4f5865d652fcfaa76ff550cf`**.

---

## 10.13 2026-10-17 — step 3b: the default scales per GPU (`512 + 128 x n`), not flat

§10.12 shipped a flat 1024.  Two better-aimed questions followed ("why not 600?" then "why not
`512 + 128/GPU`?"), and the measurements answered them: **`T` is per DEVICE**, so a flat value is either too
generous on one card or too thin on several.

| headroom | 1-GPU field GSQ seesaw | 2-GPU `-sm layer` | 3-GPU `-sm layer` | wide IQ4_NL (-ub 6144) | 2-GPU `-sm tensor` |
|---:|---|---|---|---|---|
| 512 (flat) | 214.5 | — | — | 471 | — |
| **600 (flat)** | 346.5 | **209.5** | — | — | — |
| 640 | **346.5** | — | — | **599.0** | — |
| 768 | — | **401.5** | — | — | **1 296.5** |
| 896 | — | — | **519.5** | — | — |
| 1024 (flat) | 714.5 | 657.5 | — | 983 | 1 552.5 |
| 2048 (flat) | — | 1 681.5 | — | — | 2 576.5 |

**Flat 600 is rejected**: the 2-GPU `-sm layer` case drops to **209.5 MiB**, under the 300 MiB target, while
the 1-GPU field seesaw is comfortable at 346.5.  `T` ≈ 300 MiB on one card, ≈ 390 on two, ≈ 500 on three —
each device runs its own pools, Q8_1 arena and rocBLAS workspace, so the cushion must be per card.

### Shipped default

```
GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT   512   (base)
GGML_CUDA_SLAB_HEADROOM_MIB_PER_GPU   128   (per device)
default headroom = 512 + 128 x n_gpu   ->  1 GPU 640, 2 GPU 768, 3 GPU 896 MiB
```

Verified with **no env set**, per device:

| config | effective headroom | excess | arena at the wide-arm | greedy |
|---|---:|---:|---:|---|
| 1 GPU, field GSQ seesaw (4 reqs) | **640** | **346.5** | **19 776** MiB | 4/4 identical |
| 1 GPU, wide IQ4_NL (-ub 6144) | 640 | 599.0 | 12 416 MiB | `746954a1038b` |
| 2 GPU, `-sm layer` | **768** | 401.5 /device | 17 344 MiB | `394193591d4d` (arena-pinned) |
| 2 GPU, `-sm tensor` | 768 | 1 296.5 /device | 17 344 MiB | `394193591d4d` |
| 3 GPU, `-sm layer` | **896** | 519.5 /device (all 3) | 17 664-18 176 MiB | clean |
| 35B-A3B coherence (no draft) | — | — | — | **identical** |

Every case lands **>= the 300 MiB target** with a 15-70 MiB cushion, and the field arena is **19 776 MiB**
vs 15 666.6 at the r80 default — **+4.0 GiB of arena** (reserve 7 680 -> 4 222 MiB).

**Semantics that were kept:** an explicit `GGML_CUDA_SLAB_HEADROOM_MIB` is taken **verbatim** (a total, not a
per-card value), matching every other explicit override in this tree;
`ggml_cuda_slab_headroom_at_least_mib()` still only **raises** and only when the env is unset (so the 6144
`-sm layer` floor for a partially host-resident multi-device MoE is unaffected); and a **fully resident**
model (no `-ncmoe`) has no slab and no arena, so nothing is sized there.

**Note the quantum:** the reserve is `headroom + aux` mapped in slab **chunk** units (64 MiB), so values
inside one chunk of each other measure identically — 600 and 640 are the same on one GPU.

**Recovery:** fork WIP `5d461d67f`, patch **1 946 lines, sha256
`ff2035ed69c1e7bcfbd264d6f5265bf41b640ed883cd08961b1701264527f41d`**.

---

## 10.14 2026-10-17 — step 3c: the draft reserve is split-aware (`-sm tensor`) — the arena win on multi-GPU

The headroom/excess did **not** need split awareness; the **aux** did.  The slab's reserve is *per device*, but
it was derived as the whole draft footprint on every device — and under `-sm tensor` the draft's weights and KV
are **split**, so each device reserved the other devices' share too.  The signature was unmistakable: a
**constant +895 MiB/device** of unused outside VRAM at *every* headroom.

| headroom | 2-GPU `-sm layer` | 2-GPU `-sm tensor` | difference |
|---:|---|---|---:|
| 768 | 401.5 | 1 296.5 | +895 |
| 1024 | 657.5 | 1 552.5 | +895 |
| 2048 | 1 681.5 | 2 576.5 | +895 |

**Fix** (fork WIP `156f32652`): for an **equal** tensor split, each device is charged its share plus the tier
that really is per device, plus a small margin:

```
aux_per_device = (D - tier) / n + tier + 128 MiB        tier = compute + slack
2 GPUs: 2371 MiB instead of 3782
```

| topology | excess before | excess after | arena/device before -> after | sha |
|---|---:|---:|---|---|
| 2-GPU `-sm tensor` | 1 296.5 | **300.0** | 17 344 -> **18 752** (+1.4 GiB/device, **+2.8 GiB** on the box) | `394193591d4d` |
| 3-GPU `-sm tensor` | — | **408.0** (all 3) | **20 608** | `394193591d4d` |
| 2-GPU `-sm layer` | 401.5 | 401.5 (unchanged) | 17 344 | `394193591d4d` |
| 3-GPU `-sm layer` | 519.5 | 519.5 (unchanged) | 17 664-18 176 | clean |

**Two measured failure modes defined the guards:**

* `-sm layer` with the *split* value → **aborts at draft load** (`512.00 MiB allocation failed on device 1`):
  the MTP head's layer lands on ONE device, so `-sm layer` keeps the **full** reserve.
* The margin is load-bearing: without it the 2-GPU tensor reserve came out 2 243 MiB, which was **113 MiB
  short** of a `113.87 MiB` capture allocation — a *loud* OOM abort (never a silent one, which is what makes a
  small margin + gates acceptable).  2 371 MiB measured exactly the 300 MiB target.
* An explicit `-ts` (unequal shares) keeps the full reserve: the per-device shares are not available at that
  point.  That is the one documented follow-up (it needs the shares plumbed through the preflight API, and a
  `-ts 59,41` gate — the r36 floor case).

### The finished matrix (all verified with **no env**)

| topology / config | effective headroom | excess/device | arena/device | greedy |
|---|---:|---:|---:|---|
| 1 GPU, field GSQ seesaw (4 reqs) | 640 | 346.5 | 19 776 | 4/4 identical |
| 1 GPU, wide IQ4_NL `-ub 6144` | 640 | 599.0 | 12 416 | `746954a1038b` |
| 2 GPU `-sm layer` | 768 | 401.5 | 17 344 | `394193591d4d` |
| 2 GPU `-sm tensor` | 768 | **300.0** | **18 752** | `394193591d4d` |
| 3 GPU `-sm layer` | 896 | 519.5 | 17 664-18 176 | clean |
| 3 GPU `-sm tensor` | 896 | **408.0** | **20 608** | `394193591d4d` |
| 35B-A3B coherence (no draft) | — | — | — | identical |

Every measured configuration now sits **at or above the 300 MiB target** (300.0-599.0), and the arena is
maximal for that cushion: **19 776 MiB** on the field config (vs 15 666.6 at the r80 default, **+4.0 GiB**)
and **+2.8 GiB** on a 2-GPU tensor box from this step alone.

**Recovery:** fork WIP `156f3265213b8aec99dc453acff20e6578a73fda`, patch **1 991 lines, sha256
`478a8e9f395d4467219db3f928c97ddb8c6e613f50fd435b4269e916dc660add`**.

---

## 10.15 2026-10-17 — why the cache budget changes the *numerics* (the mechanism, proven)

§10.13/§10.14 noted that greedy text differs between runs whose expert-cache budget differs.  It is **not** a
wrong expert and **not** an ordering change, and it is **not** the hit *rate*.  The mechanism, from the code:

1. **A miss is arithmetic-neutral.**  A miss inside a cache-managed table serves the *same bytes* to the *same
   kernel* — either in place through the pinned host master's device alias (`table_cold_ok` / `g_cold_uva`) or
   by copying the expert into its slot (`expert_slot` -> `arena + slot * expert_bytes`, no repack).  So
   hit vs miss cannot change the text by itself.
2. **What changes is *table residency*.**  `moe_cache_take_over()` requires
   `identity && t.arena != nullptr && t.slots >= t.n_experts` — a table is cache-managed only if the arena holds
   a slot for **every** expert in it, and `slots = per_host / t.host_bytes` is a continuous function of the
   per-device budget.  A budget change therefore *crosses that threshold* for individual tables.
3. **And a cache-managed table is offloaded to the GPU even at one token** —
   `ggml_cuda.cu` `ggml_backend_cuda_device_supports_op`/`offload_op`: `moe_cache_enabled() && op == MUL_MAT_ID
   && op->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK -> true` — whereas a **non**-cache-managed host-expert table is
   offloaded only when `get_op_batch_size(op) >= op_offload_min_batch_size` (**32** by default, block 09's
   op-offload).  Below that width (decode, and the MTP verify band) the op runs **on the CPU**.

So the difference is **CPU vs GPU arithmetic for a whole table in the narrow decode band** (or, for a table that
*is* cache-managed, the cache's compact-arena kernel path instead of the plain one) — a different code path for
the same mathematics, not a different expert.  A second, *global* switch rides along: the cache-aware fusions
stand down whenever any table is unserveable (`moe_cache_has_arena()`), so fused vs separate kernels can change
too.

**Proof.**  The 2-GPU `-sm layer` config that flipped between headroom 1024 and 768 was re-run with
`GGML_OP_OFFLOAD_MIN_BATCH=1` (force *every* host-expert op onto the GPU, including at 1 token):

| arm | default offload | `GGML_OP_OFFLOAD_MIN_BATCH=1` |
|---|---|---|
| headroom 1024 | `c5b7ff695660` | **`c5b7ff695660`** |
| headroom 768 | `394193591d4d` | **`c5b7ff695660`** |

With the offload decision made independent of cache residency the two arms agree exactly, which pins the cause.

**Consequences (this is a purity rule, not a bug):**
* greedy text is comparable across runs only when the **cache budget / offload decision** is the same — pin
  `MOE_EXPERT_CACHE_MIB` (same value on both arms) or neutralise with `GGML_OP_OFFLOAD_MIN_BATCH=1`;
* a *hit-rate* improvement alone can never change the text; a *budget* change can;
* the beta's coherence gate therefore compares against a build with the same cache budget, or compares logits
  (`gate-prefill-logits.sh` is unaffected: it is a prefill-path comparison and measured 0.000000 across this
  whole campaign).  Worth a `GREEDY-PURITY.md` entry when this is promoted.
