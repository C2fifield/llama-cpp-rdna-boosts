# HANDOVER — next session: the generic workspace-pool redirect (step 2b, deferred half)

**State of the campaign first:** read `README.md` §8–§10.5.  Everything below is already in the fork
working tree (WIP commit `git log -1` in `~/llama.cpp`, never pushed; the exploration diff is
`slab-transient-absorption-wip.patch` here).  Nothing is in `patches/`.

## 0. Why this is the next piece (measured, not assumed)

The outside-slab consumers left after step 0 / FA / QSA3 / GDN / calibration / aux-3584 are, in order:
**the workspace pool** (`ggml_cuda_pool_leg`, 335–671 MiB on a prefill), hipBLASLt ~104 MiB, MMB ~40 MiB.
The headroom floor is now set by the pool:

| config | headroom/aux | arena | free at the prefill drop |
|---|---|---:|---:|
| 1 GPU GSQ `-ub 4096`, seesaw short/long/short/long | 1024/3584 | 19 263.5 MiB | **204.5 MiB** (long-2 arm) |
| 1 GPU IQ4_NL `-ub 6144 -c 204800` | 2048/3584 | 17 718.8 MiB | 629 MiB (pool 671) |
| same, wide gate | **1536/3584** | **18 225.0 MiB** | **459 MiB** (pool 335) |
| 2 GPU `-sm tensor` (GSQ) | 2048/3584 | 40 725.1 MiB (99.5 %) | 2 118.5 / device |
| 2 GPU `-sm layer` (GSQ) | 2048/3584 | 40 199.1 MiB (98.2 %) | 1 097.5 / device |

The pool figure is **not repeatable** (671 vs 335 MiB for the same wide config on different runs), so the
margin at 1024/1536 has that variance in it.  Absorbing ~300 MiB is what makes 1024/1536 a default rather
than an opt-in.  **Recommendation until then: 2048 / 3584 default, 1024–1536 / 3584 opt-in for `-ub <= 4096`.**

## 1. The hazard that stopped the first attempt (do not skip)

A generic `pool_leg::alloc -> wide_aux` redirect is unsafe because the pool is **not only RAII per-op
scratch**.  These retained allocations are made during prefill and outlive the graph:

| site | what | action |
|---|---|---|
| `mmb.cu:1610` `mmb_cache_insert()` | MMB BF16 activation cache (≤ `mmb_cache_max()` = 4 entries) | must be **pinned** (never wide) |
| `mmb.cu:1943` `mmb_state().slots` | MMB persistent activation slots | pinned |
| `allreduce.cu:809`, `allreduce-hip.cu:1604` | `bf16_tmp[devices]` held by the comm object (`-sm tensor`) | pinned |
| `mmf.cu:51-53` | `ids_src_compact_dev` / `ids_dst_compact_dev` / `expert_bounds_dev` | **check** whether struct members (retained) or RAII; pin if retained |

If any of these were served from the band they would be dangling after the prefill -> decode reclaim: a
silent use-after-free.  `ggml_cuda_slab_wide_aux_free` only pops the TOP of the band stack (a non-LIFO
free leaves a hole, reclaimed at the drop) — that is fine for correctness, but it means a pointer must
never survive the drop.

## 2. Recommended design

1. **Escape hatch on the pool interface** (`common.cuh`, `struct ggml_cuda_pool`):
   ```cpp
   virtual void set_wide_routing(bool /*on*/) {}   // no-op base; pool_leg overrides
   ```
   `pool_leg` gets `bool wide_routing = true;` and the override sets it.  Convert the retained sites above
   with a tiny RAII guard that sets it false around their allocation (they allocate rarely).
2. **Route in `pool_leg::alloc`** (same TU as `ggml_cuda_graph_capture_active()`, which is static there):
   ```cpp
   if (wide_routing && ggml_cuda_slab_wide_available(device) && !ggml_cuda_graph_capture_active()) {
       void * p = ggml_cuda_slab_wide_aux_alloc(device, size, &gen);
       if (p) { *actual_size = size; return p; }   // requested size, so free() pops the same block
   }
   // else: existing cache scan + cudaMalloc path
   ```
   `ggml_cuda_slab_wide_available()` already means "slab live AND the wide view armed" (prefill).  The
   capture guard is belt-and-braces (a captured graph's pointers must stay valid forever).
3. **`pool_leg::free`**: if the pointer is inside the slab (`ggml_cuda_slab_owns_ptr(device, ptr)`, new
   range check `[base, base+mapped)`), call `ggml_cuda_slab_wide_aux_free(device, ptr, size)` and do
   **not** put it in the free list / do not touch `pool_size`.  A band pointer must never be cached.
4. **Do not count band bytes in `pool_size`/`cached_size`** (they are not pool-owned).  `clear_pool()` must
   never see a band pointer.
5. Keep the QSA3/GDN/QSA-style *static* routes as they are (they work and do not depend on this).

## 3. Gates (all mandatory)

1. **35B-A3B coherence** (slab active, `-ncmoe 40`, `code-reasoning-mixed.txt`, seed 42, `-n 64`) vs the
   r80 baseline text — must be identical.  Harness shape: `/tmp/coh.sh` from this session (rebuild baseline
   via `git stash` once).
2. **Field GSQ seesaw** short(≈256)/long(≈80K)/short/long against ONE server with **distinct prompt prefixes**
   (the prompt cache otherwise serves the repeat — measured), headroom 1024/3584, `GGML_CUDA_SLAB_STATS=1`:
   compare the wide-drop free and the pool figure with §0.  Expect the pool to fall to ~40–80 MiB.
3. **Wide gate**: IQ4_NL `-ub 6144 -c 204800`, 6.6K prefill, headroom **1536** then **1024** (1024 is
   unmeasured; the pool variance means it must be run, not extrapolated).
4. **2 GPUs**: `-sm tensor` and `-sm layer`, headroom 2048/3584 (and `-sm layer` at a long context if
   possible — the r36 6144 floor was a 255k `-ts 59,41` config).
5. `scripts/gate-prefill-logits.sh` (mean KLD <= 0.005) — this changes a **prefill path** allocation, and
   the release rule requires it.  Also `test-backend-ops` if any op code is touched.
6. Coherence on Flash-Next (`sha` of a fixed greedy continuation) before/after.

## 4. Gotchas learned this session

* The band's alloc is **per device**, the pool is **per context/stream**: the draft (MTP) context also owns
  pools.  During a wide prefill the arm state is device-global, so a *draft* allocation could land in the
  band — fine (not captured), but it must be freed LIFO or left as a hole.
* `actual_size` from the pool normally includes the 5 % + 256 B look-ahead.  For a band block return the
  **requested** size (the band rounds to `unit`), otherwise `free()` pops the wrong amount.
* `GGML_CUDA_POOL_CACHE_MB` is a **dead end** for this (measured: 64 MiB -> pool 327 vs 271 MiB, i.e.
  worse) — the high-water is live temporaries, not the cache.
* `-sm layer` with host experts has a 6144 headroom floor (`ggml_cuda_slab_headroom_at_least_mib`) unless
  `GGML_CUDA_SLAB_HEADROOM_MIB` is explicit; an explicit value wins (used for the §0 layer run).
* Always filter the server log at `-lv 4` or use `GGML_LOG_WARN` for diagnostics: `GGML_LOG_INFO` from the
  ggml lib is suppressed at the field `--verbosity 3`.
* Never run two of these benches at once, and never run the container build beside them.

## 5. Definition of done

Pool high-water <= ~80 MiB on both the field and the wide config; the seesaw floor >= ~500 MiB at headroom
1024 (1 GPU) and >= ~300 MiB on the wide config at 1536; every gate green; then propose
**1024 or 1536 / 3584** as the default, re-cut the WIP against `release.json.base`, and promote.
