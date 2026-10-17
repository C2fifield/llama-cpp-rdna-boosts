# SCOPE — the MMB/BF16 cache yield, and what "aux = 1536" actually means

Date: 2026-10-17 (after step 2b, §10.6).  Status: **items A+B IMPLEMENTED (step 2b-2, §10.8); the reserve
matrix (§6) and the default decision (D1/D2) are still open.**  See §10 below for what changed when it was
built.
Fork WIP tree at `3f4497d14` (§10.6), delivery `v12-097cf33d9-r80`, 1× R9700 gfx1201 + 2× (gfx1201 pair)
on the same box.

This note answers two questions:

1. what does it take to let the two retained caches (**MMB activation cache/slots**, **`bf16_src1_cache`**)
   be served from the slab's wide band and flushed at the prefill -> decode drop?  (~250 MiB)
2. does that get us to `aux = 1536`?  **Partly — and the aux number is not the lever it looks like.**
   The measured blocker at the reserve size `aux 1536` implies is the MMB cache, but the *split* between
   `aux` and `headroom` is irrelevant; only their **sum** is.  See §2.

---

## 0. TL;DR

* **Measured: only `headroom + aux` matters, not the split.**  `2048 / 2048` and `2560 / 1536`
  (both sum 4096) produce **byte-identical** SLAB_STATS numbers and die at the same allocation.
* **Measured: `aux = 1536` forces `headroom >= ~2560`** (sum 4096).  Sum 3584 (`2048/1536`) cannot even
  *load* the draft: a 512 MiB allocation fails during draft creation.
* **Measured: at sum 4096 the thing that fails IS the MMB activation cache.**  The abort is an
  `84.00 MiB` device_malloc — i.e. a pool request of **80 MiB + 5 % look-ahead**, the exact size of one
  MMB activation-cache entry — with 93.5 MiB free at the wide arm.  So yes: the MMB cache is *the*
  blocker at that reserve.
* **The MMB cache is a *per-graph* cache** (`ggml_cuda_mmb_begin_graph()` clears it at the start of every
  `graph_compute`), and `bf16_src1_cache` likewise.  Their only cross-graph lifetime is the *delay* between
  "last graph finished" and "next graph starts".  So the yield is **semantically free** — no cache hit is
  lost that would not have been lost anyway — and the implementation is small (§3, §4).
* **But `aux 1536 / headroom 2560` is not the arena-maximising direction.**  In the regime that matters
  (the draft eats through the reserve) the arena is `free - reserve`, and when the draft does *not* eat it
  all, `slab_extend` reclaims down to `headroom` only — so a **large headroom permanently costs arena**.
  The campaign's own target (headroom ≈ the transient floor, aux ≈ the draft footprint) stays the better
  shape; `aux 1536` is the "keep a 2.5 GiB transient cushion" variant (§2.3).
* Recommendation: do the yield work (§3, §4) — it is ~40 lines, two files, plus one hook — and **then**
  run the reserve matrix in §6.  If it holds, the choices are
  `aux 1536 / headroom 2560` (reserve 4096, +1.5 GiB arena vs the r80 default, 2.5 GiB cushion) **or**
  `aux 3584 / headroom 512` (reserve 4096, same arena, 512 MiB cushion) — the same sum, so pick on safety,
  not on arena.

---

## 1. The measurements that shape the plan (all 2026-10-17, this box)

Field GSQ + shared Q8_0 MTP draft, the §10.2 seesaw (short 251 / long 80 107 / short 249 / long 80 105
tokens, distinct prompt prefix per request, one server), `GGML_CUDA_SLAB_STATS=1`:

| reserve (`headroom`/`aux`) | sum | draft load | arena at 2nd arm | free at arm-1 / drop-1 / arm-2 | outcome |
|---|---:|---|---:|---|---|
| 1024 / 3584 | 4608 | ok | 19 328 MiB | 934 / 709.5 / 286.5 | **clean**, min 286.5 (§10.6) |
| 2048 / 2048 | 4096 | ok (barely) | **20 096 MiB** | 422 / 203.5 / **93.5** | OOM: `84.00 MiB` |
| 2560 / 1536 | 4096 | ok (barely) | **20 096 MiB** | 422 / 203.5 / **93.5** | OOM: `84.00 MiB` |
| 2048 / 1536 | 3584 | **FAIL: `512.00 MiB`** | (arena 20 302.6 MiB) | — | draft load abort |
| 3584 / 4096 (r80 default) | 7680 | ok | 15 666.6 MiB | — | reference (§9) |

Three facts fall out:

1. **Split-irrelevance.**  Rows 2 and 3 are identical to the byte (422.0 / 203.5 / 93.5, arena 20 096,
   same eviction pattern).  `reserve_mib = headroom + aux` in `ggml_cuda_slab_reserve_bytes()` is the only
   place either number is used, and `slab_extend` can only map when `free > headroom + chunk` — with the
   draft consuming the reserve, that never fires.  **Treat `aux + headroom` as ONE knob.**
2. **A hard load-time floor around 4 GiB.**  Sum 3584 dies during draft creation (512 MiB alloc).  This
   matches §10.1's `draft 3 447 + post-slab transient peak` estimate.  Any `aux = 1536` therefore needs
   `headroom ~= 2560`.
3. **The 80 MiB entry is the blocker.**  `84.00 MiB` = `1.05 × 80 MiB`: the pool's look-ahead of an
   **80 MiB** request — one MMB activation-cache entry — while 93.5 MiB was free at the wide arm.  The
   prefill then kept ~10 MiB and the request could not be placed.  (The 80 MiB size is the same in the
   §10.6 `GGML_CUDA_MMB=0` A/B: pool 262.9 -> 95.2 MiB, seesaw floor 286.5 -> 452.5 MiB.)

Also worth recording, because it bounds what the yield can buy: the outside free **degrades across the
seesaw** (934 -> 587 -> 290 -> 286 MiB with 2b; 934 -> 577 -> 208 -> 204 pre-change) and then stabilises
(long-1 vs long-2).  So the minimum is a *steady state*, ~650 MiB below the first arm, not a one-off
prefill peak.  The pre-yield steady-state demand outside the slab at reserve 4608 is ~875 MiB; the yield
**plus everything already absorbed** has to fit under the cushion the chosen sum leaves
(`sum - draft`): 4608 -> 1 161 MiB, 4096 -> 649 MiB, 3584 -> 137 MiB.

## 2. The reserve model (read from the code, 2026-10-17)

`ggml_cuda_slab_reserve_bytes()` returns `max(headroom_mib + aux_mib, 1024)`; the slab maps
`free - reserve` at creation.  `ggml_cuda_slab_extend()` later maps `free_b - headroom` (chunk-aligned,
bounded by the reserved VA), i.e. it **leaves exactly `headroom` outside** — but only if
`free_b > headroom + chunk`.

`aux` is set by `LLAMA_MOE_CACHE_AUX_RESERVE_MIB` (default **3584**, `include/llama.h`) and is a
*projection* of the MTP draft context, which is created **after** the slab.  Consequences:

* while the draft is loading, the reserve is the only outside VRAM it can use: `reserve >= draft load
  peak` is a hard floor (measured: sum 3584 fails, sum 4096 passes with 422 MiB to spare);
* after the draft is resident, whatever the reserve did **not** need is reclaimed by `slab_extend()` down
  to `headroom`, so a **large `headroom` permanently withholds arena** and a large `aux` does not;
* the steady-state outside budget equals `reserve - draft_steady` (if that is <= headroom, `extend`
  cannot fire), or exactly `headroom` (if it is > headroom).

### 2.3 What to aim for

```
reserve  = headroom + aux  >=  draft_load_peak + T        (T = steady-state transient demand)
arena    = free - max(reserve, headroom + draft_steady)
outside  = reserve - draft_steady   (or exactly headroom once extend fires)
```

> **Measured 2026-10-17 (README §10.9)**: `D` and `T` are now separated by a with/without-draft run at a
> forced reserve.  `D` = **3 584 MiB** (field draft) / **4 000 MiB** (same draft at `-c 204800`, `-ub 6144`),
> and `T` = **502 MiB** (4-request session) / 210-330 (single prefill), stable across reserves.  So the
> `aux` default (3584) is exactly the *field* draft and ~416 MiB short for the long-context one, and the
> formula below is `reserve = D + 512 + excess`.

So: **minimise `reserve` (that is the arena) and pick `headroom = T` (that is the safety margin)**;
`aux = reserve - headroom` follows.  Two corollaries:

* "aux 1536" is only meaningful together with a headroom; `1536 + 2560` is a *conservative* setting whose
  benefit over the r80 default is +1.5 GiB of arena with a 2.5 GiB transient cushion (vs 4 GiB today);
* the arena-maximising end state is `aux = reserve - T` with `T` as small as the transient decomposition
  allows — i.e. `aux 3584 / headroom 512` at reserve 4096, **not** `aux 1536`.  Reaching it needs the same
  work as `aux 1536` (below), plus a *measured* `T <= 512`

Both end states need the same prerequisite: **`T` must stop being dominated by the two retained caches.**

---

## 3. Work item A — the MMB activation cache + slots (~168-170 MiB)

### 3.1 What holds memory (`mmb.cu`)

| state | type | lifetime today |
|---|---|---|
| `mmb_ctx_state::cache` (<= `mmb_cache_max()` = **4** entries) | `ggml_cuda_pool_alloc<uint16_t>` each, ~80 MiB on a 4096-token ubatch (GSQ), ~120 MiB at `-ub 6144` | cleared by `ggml_cuda_mmb_begin_graph()` at the **start of every `graph_compute`** |
| `mmb_ctx_state::slots[MMB_SLOT_COUNT = 5]` | same, with `slot_cap[]` so the buffer is reused | `begin_graph` resets only `root/data/n`; **the buffers are never freed** until `ggml_cuda_mmb_release_all()` at context destruction |
| `g_mmb_shadow` / `g_mmb_shadow_pair` | **raw `cudaMalloc`**, dense-weight BF16 shadows | process lifetime — **NOT part of this work, and must never be flushed by the yield** (a re-dequant of every dense table) |

Key point: **the cache is already per-graph.**  Its blocks are live at the drop purely because the drop
happens between the last prefill compute and the next decode compute — i.e. *before* the `begin_graph`
that would free them.  The slots are the only genuine cross-graph retention, and they are a pure cache
(`slot_cap` is a realloc optimisation; `optimize_begin` rebuilds the marks per graph).

### 3.2 Recommended design — flush at the drop, then un-pin

1. `void ggml_cuda_mmb_yield_caches()` in `mmb.cu` (+ `mmb.cuh`): for **every** entry of `g_mmb_state`
   (all contexts — the state carries no device, and flushing a non-active context's cache is free),
   `delete e.buf` the cache entries, clear the vector, and for the slots `delete e.buf`, `e.buf = nullptr`,
   `e.slot_cap = 0`; **keep** `bf16_only` / `bf16_copy` / `bf16_slot` / `first_split` (they describe the
   graph being optimised, and the drop is between graphs — but keeping them is strictly safer).
   **Do not touch the shadow maps.**
2. Call it from `ggml_backend_cuda_device_slab_ring_set()` (`ggml-cuda.cu`) **before**
   `ggml_cuda_slab_ring_set(ctx->device, false)`.  That keeps the slab module free of an MMB dependency
   and puts the flush at the only point where the band is about to be released.  (The alternative —
   flushing at the *end* of `graph_compute` — is item B's shape; see §4.2.  Either is fine; do not do
   both.)
3. Delete the two `ggml_cuda_pool_wide_pin` guards added in `mmb.cu` (§10.6), so the cache and the payload
   become band-eligible again.
4. Keep `ggml_cuda_mmb_release_all()` as it is (destruction path, shadows included).

Size: ~25 lines in `mmb.cu`/`mmb.cuh`, ~4 in `ggml-cuda.cu`.  No change to `apply-all`-facing behaviour.

### 3.3 Gotchas

* **Do not flush the shadows.**  `release_all()` does; a yield must not.
* **`g_mmb_active_ctx`**: flushing the active context's cache is fine *between* graphs (the next
  `begin_graph` would do it anyway), but assert/verify the drop is never taken mid-graph — today it is
  driven by the context's ubatch classification, so it is between graphs.  Worth a comment.
* **The draft context's state** is in the same map and will be flushed too.  That is desirable (its cache
  is per-graph as well) and costs nothing.
* **`slot_cap = 0` means the next graph re-`new`s the slot buffers** (one `ggml_cuda_pool_alloc` per
  used slot).  On the band that is a bump; on the pool it is a cached-block hit.  Measure decode
  (captured graphs allocate once at capture, so the expectation is no measurable cost).
* Threading: `g_mmb_state` is touched from the backend compute thread(s); the drop is on the same thread
  for a given context.  The yield iterates all contexts, so guard it like the existing accessors
  (`mmb.cu` currently relies on "one active context, ops run synchronously inside a backend compute").
  If the implementation touches another context's vector while that context computes, that is a data
  race — the safe form is to flush only the requesting context **plus** the idle ones, or to take a
  small mutex (the existing code has none, so a mutex is a behaviour change: prefer "flush the active
  context only, and record that the draft's cache is flushed by its own drop").

## 4. Work item B — `bf16_src1_cache` (~80 MiB)

### 4.1 What it is

`ggml_backend_cuda_context::bf16_src1_cache` (`common.cuh`): one `ggml_cuda_pool_alloc<nv_bfloat16>`,
keyed by `(root, data, stream_no, n)`, holding the F32 -> BF16 conversion of an activation that two
adjacent BF16 GEMMs share (`hc_down` and the 4-wide `hc_inject`; also `hyperconn.cu`).  Cleared by
`bf16_src1_cache_clear()` at the **start of every `graph_compute`**, next to `q8_1_cache_clear()`.

Note the sibling: the Q8_1 cache is *not* pool memory (a raw arena whose cursor resets, so the pointer
stays stable).  Only the BF16 cache is a pool allocation.

### 4.2 Design — free it at the END of the graph (preferred), not via a registry

`ggml_backend_cuda_context` has no back-pointer to its device context and there is no registry of
contexts, so "flush it from the device callback" would need a new global registry (construction /
destruction hooks, lifetime care).  Not worth it: the cache is already dropped at every graph start, so
moving the drop to the graph *end* is behaviour-preserving and needs no cross-module plumbing:

```cpp
// in ggml_backend_cuda_graph_compute, right after the existing cache clears:
// wip/slab-transient-absorption (step 2b-2): the per-graph caches must not outlive the graph, or they
// pin a pool block across the prefill -> decode drop.  Free at scope exit (all return paths) instead of
// at the next graph's start; `bf16_src1_cache_clear()` is exactly what that start does today.
struct per_graph_bf16_yield {
    ggml_backend_cuda_context * c;
    ~per_graph_bf16_yield() { c->bf16_src1_cache_clear(); }
} bf16_yield{cuda_ctx};
```

then delete its `ggml_cuda_pool_wide_pin` in `bf16_src1_cache_put()`.

Trade-off vs a registry + drop-flush: freeing at the graph end also frees it for **decode** graphs between
captures, so the decode path re-allocates one ~small buffer per graph where today it is reused.  Decode
graphs are captured (the dispatch path runs once per capture), so the expectation is no cost — but it
must be measured (MTP tg, not just "it compiles").

The same RAII shape can carry item A if we prefer one hook to two (§3.2 item 2 alternative): a single
`per_graph_cache_yield` that releases the MMB cache/slots **and** clears the BF16 cache.  That is the
smallest diff overall (one hook, no `ring_set` coupling, no cross-module call) and it makes the
"per-graph caches never outlive their graph" invariant explicit.  **Recommended**, with the drop-flush
kept as the fallback if the end-of-graph free shows a decode cost.

## 5. Safety argument + the acceptance criteria the tripwires give us

The band is released at the prefill -> decode drop and its bytes go back to the arena.  A block that
survives the drop is a use-after-free.  After this work the invariant is: **every band block is released
before its graph returns** (end-of-graph hook) or **before the band is released** (drop flush).

Acceptance criteria, all already instrumented (§10.6) — no new tooling needed:

* `POOL_WIDE_LEAK` **never** fires (it is unconditional, not `SLAB_STATS`-gated);
* the one-shot `freeing a wide-band block from a dead generation` warning never fires;
* `SLAB_STATS ... pool_wide=0.0 MiB/0 refused=0.0 MiB` at every `wide-drop`;
* `pool=` at the drop falls to the sub-1-MiB tail (~15-65 MiB) and `free` at the drop rises by the
  absorbed bytes.

The capture guard stays exactly as it is: a captured graph's pointers must never come from the band, and
that is why decode is unaffected by construction.

## 6. Gates (all mandatory; the campaign's own list plus what the reserve change adds)

1. **The reserve matrix**, field GSQ seesaw (distinct prefixes, one server, 4 arm/drop pairs), stats on:
   `2560/1536`, `2048/2048`, `3072/1536`, `1024/3584`, `3584/512`, `3584/1024`
   — the runs that decide the default.  Record free at each arm/drop, `pool`, `pool_wide`, `refused`,
   and the greedy shas (must be unchanged).
2. **The wide config** (IQ4_NL `-ub 6144 -c 204800`, 6 657-token prefill): every candidate reserve.
   This is the config that needs the most outside room (§10.6: 1536/3584 leaves 477 MiB; 1024/3584 OOMs).
   `hipblaslt.cpp:164` is the failure to watch for at the low end.
3. **2 GPUs**, `-sm tensor` **and** `-sm layer`, at each candidate reserve — and the **long-context
   `-sm layer`** run that has never been done (the r36 floor is
   `ggml_cuda_slab_headroom_at_least_mib(6144)`, set by `alloc_all_locked` for a multi-device unsplit
   table: a 255k `-ts 59,41` needle).  An explicit `GGML_CUDA_SLAB_HEADROOM_MIB` overrides that floor, so
   a low-headroom default must be gated against that config, not assumed.
4. **Coherence**: 35B-A3B (`-ncmoe 40`, `code-reasoning-mixed.txt`, seed 42, `-n 64`) text identical to
   the pre-change build; Flash-Next GSQ greedy sha; the seesaw shas.
5. **`scripts/gate-prefill-logits.sh`** (mean KLD <= 0.005, same-top-p >= 98 %) on a *slab-active* config
   — the §10.6 recipe (base recorded by the pre-change build, `-sm tensor -ncmoe 40`, `ppl` wrapper).
6. **Decode cost of the yield**: `llama-batched-bench -npl 1,4,8` (the stock-relative gate,
   `benchmarks/mtp-adaptive-methodology.md` rule 5) plus one MTP long run (`-n 3000`) before/after, since
   item B frees the BF16 cache at every graph end.
7. `test-backend-ops` only if any op code is touched (it should not be: this is allocation lifetime).

## 7. Sequencing and decision points for the maintainer

**Step 1 (small, self-contained):** item B — the end-of-graph `per_graph_cache_yield` RAII hook that
releases the MMB cache/slots **and** clears the BF16 cache, plus removing the three
`ggml_cuda_pool_wide_pin` guards.  ~40 lines, 3 files.  Gates 4, 5, 6, and the tripwire criteria.

**Step 2 (measurement, no code):** the reserve matrix (§6.1-6.3).  This is where `aux` is actually
decided, and it is ~8 seesaw runs + 4 wide runs + 4 two-GPU runs.

**Step 3 (only if the matrix says a low headroom is safe):** propose the reserve default, re-cut the WIP
against `release.json.base`, run the release gates, promote.

**Decision needed now:**

* **D1 — which reserve shape is the target?**  `aux 1536 / headroom 2560` (reserve 4096, +1.5 GiB arena
  vs the r80 default, 2.5 GiB transient cushion) **or** `aux 3584 / headroom 512` (same arena, 512 MiB
  cushion).  Same sum, so this is a safety choice, not an arena choice — and it is worth being explicit
  that "aux 1536" alone does not buy arena.
* **D2 — is the ~875 MiB steady-state outside demand at reserve 4608 acceptable to leave in place?**  The
  caches are ~250-315 MiB of it; the rest (hipBLASLt ~104, the Q8_1 arena ~110, the pool tail, and
  ~300-500 MiB not yet attributed) is *not* in this work.  If the answer to `aux 1536 / headroom 2560`
  turns out to be OOM after the yield, this is the next thing to decompose (the same
  one-env-A/B-at-a-time method as §8.1, but sampled **at the seesaw minimum** — the arm, not the drop).
* **D3 — end-of-graph free vs drop flush** (§4.2).  Recommended: end-of-graph (smaller, no plumbing);
  fall back to the drop flush if gate 6 shows a decode cost.

## 8. Sizing

| item | files | lines (est.) | risk |
|---|---|---|---|
| A: `ggml_cuda_mmb_yield_caches()` + call from the ring_set device hook | `mmb.cu`, `mmb.cuh`, `ggml-cuda.cu` | ~30 | low — pure cache drop, shadows untouched |
| B: end-of-graph RAII yield (+ the same hook can cover A) | `ggml-cuda.cu`, `mmb.cu/cuh`, `common.cuh` (remove 2 pins) | ~25 | low — same operation the graph start already does |
| pin removal (`mmb.cu` x2, `common.cuh` x1) | 2 | ~-8 | none (they become band-eligible) |
| reserve matrix + gates | — | — | this is the session |

## 9. What NOT to do

* Do not flush `g_mmb_shadow` / `g_mmb_shadow_pair` in a yield (raw `cudaMalloc`, permanent, expensive).
* Do not route anything through the band while a graph capture is active (§10.6's guard) — a captured
  graph's pointer must never be a band block.
* Do not raise `headroom` to make a small `aux` work and call it a win: the headroom is permanently
  outside the slab, so that trade *shrinks* the arena.  The sum is the dial; the split is safety.
* Do not lower the reserve default on the field config alone — the wide (`-ub 6144`) and multi-device
  `-sm layer` configs are the ones that set the floor, and the `-sm layer` 6144 headroom floor has never
  been gated at a long context.
* Do not enable tail effects: `GGML_CUDA_MMB=0`, `ROCBLAS_USE_HIPBLASLT=0`, `GGML_CUDA_QSA3=0` are
  **diagnostic** arms (they change numerics and the greedy sha); a shipped default must be measured with
  the full feature set.


---

## 10. What changed when it was built (step 2b-2, README §10.8)

Items A+B are implemented as designed (one RAII `per_graph_cache_yield` in `ggml_backend_cuda_graph_compute`,
the three pins removed, `ggml_cuda_mmb_yield_caches(ctx)` scoped to the calling context).  Two things the
scope underestimated:

* **The band allocator, not the caches, was the real work.**  Dropping blocks in it is only free when the
  frees are strictly LIFO.  The MMB/BF16 blocks live for a whole graph and are released as a *group*, which
  makes out-of-order frees the norm — measured **7 392 MiB of `wide_aux` and 149 holes in ONE 6.6k prefill**,
  arena 20 096 -> 4 992 MiB, 417 GiB of allocations refused and re-fallen outside the slab.  A
  "highest-address-first batch" did **not** fix it (the hole sits below a still-live FA staging block); the
  fix is the live block set in `ggml-cuda-slab.cu` (§10.8 item 3).  Budget for touching the slab whenever a
  band user's lifetime stops being per-op.
* **The target works, the margin does not.**  `aux 1536 / headroom 2560` (sum 4096) now completes the seesaw
  with identical greedy shas, but the **minimum outside is 52.5 MiB** (§10.8).  Treat §6.1's matrix as the
  gate for any default, with repeats, and expect to land above sum 4096 (e.g. `aux 1536 / headroom 3072`,
  sum 4608 — the same reserve as today's tested `1024 / 3584`) unless §6 shows the ~600 MiB steady-state
  demand can be cut further.
* Side effect worth knowing: **the wide gate now passes at headroom 1024** (it OOMed at 2b), with 279 MiB at
  the drop and the same greedy sha — so the 1536 floor from §10.6 was the caches, not the ubatch width.
