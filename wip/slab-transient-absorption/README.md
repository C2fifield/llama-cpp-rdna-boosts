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

| `GGML_CUDA_SLAB_HEADROOM_MIB` | arena | free at wide-drop | prefill | text |
|---:|---:|---:|---:|---|
| 4096 | 15 666.6 MiB | ~3.9 GiB (5.2k-token probe) | 787-1164 t/s | coherent |
| 2048 | 17 744.8 MiB | 1842 → **1942** (post-QSA3) | 1156 t/s | coherent |
| 1024 | 18 783.9 → **~19 000** MiB mapped 20864 | 818 → **918** (post-QSA3) | 1105 t/s | greedy sha256 identical vs pre-QSA3 |

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
