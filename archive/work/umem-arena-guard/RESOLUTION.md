# `umem-arena-guard` — RESOLUTION (closed in `v12-097cf33d9-r91`, 2026-10-19)

**Status: CLOSED, promoted into block 09.**  The defect, the guard and both boxes' gates are below; the
campaign's own record is this directory's `README.md` + `umem-arena-guard.patch` (the patch is the same
change that is now in the delivery, kept as the provenance copy).

## The defect

`MOE_EXPERT_CACHE_MIB` unset means **AUTO**: size each device's arena from its free VRAM.  The
movable-boundary slab sizes itself as `free VRAM - reserve`.  Both assume device memory is a separate
physical pool.

On an integrated GPU it is not (gfx1151: a 2 GiB VRAM carve-out over ~121 GiB of GTT over the same host
RAM; llama.cpp reports "Total VRAM: 124000 MiB" on a 123 GiB box).  Measured on 35B-A3B UD-Q3_K_M
`-ncmoe 40`, one device:

* AUTO: **121 GiB resident + 13 GiB swap**, still loading after ~15 min;
* `MOE_EXPERT_CACHE_MIB=4096` (arena bounded): the **slab** still mapped **98.19 GiB** — bounding the
  arena alone just hands the overcommit to the slab.

There is also nothing to gain: an expert in the host master is already device-reachable through the same
physical memory, so an arena slot is a copy tier over bytes the device can read anyway.

## The fix (block 09)

`moe_cache_preflight` argues the cache **off** on `prop.integrated`, keyed *before* any table registers, so
the whole cache stack stays inert and the run is the documented host-expert path; the wrapper disarms the
slab.  `common/fit.cpp` mirrors the predicate so `--fit` stops reserving slab headroom + arena there.
`MOE_EXPERT_CACHE_UNIFIED=1` is the only way back in and requires **both** an explicit
`MOE_EXPERT_CACHE_MIB` **and** an explicit `GGML_CUDA_SLAB_RESERVE_MIB` (the Windows fixed-carve-out
case), which bounds `mapped = free - reserve` by construction.  Detection uses the live
`prop.integrated`, not `ggml_cuda_info().devices[].integrated` (forced false on CUDA builds, and
`GGML_FORCE_NO_INTEGRATED` on HIP).

## Gates

gfx1151 (the box that found it; README §3): three refusal arms single-message and safe (15.4 GiB peak
RSS), guarded default **byte-identical** to `MOE_EXPERT_CACHE_MIB=0`, two-knob override maps 19.19 GiB as
asked, prefill-logit `0.000000 / 100.000 %`, dense coherence identical, `test-recurrent-state-depth`
153 == 153, `test-recurrent-state-rollback` 0.

gfx1201 (primary box, r91 tree; the required no-op check): 35B-A3B Q3_K_M `-ncmoe 40`, MIB unset, the
cache/slab lines are **identical** to the unpatched build:

```
moe_cache_preflight: auto cache preflight: device 0 projected arena 31524.0 MiB (free 32548.0 - reserve 1024.0 - aux 0.0) >= floor 2444.0 MiB (host experts 13578.0 MiB); cache stays on
SLAB_INIT dev=0 free=24.70 GiB reserve=1.00 GiB headroom=0.75 GiB aux=0.00 GiB size(VA)=23.94 GiB mapped=23.69 GiB ...
alloc_all_locked: MoE expert cache (auto): arena 13578.0 MiB of 13578.0 MiB host experts (100.0% residency)
```

## Follow-ups (documented, not tickets)

1. **Root cause**: `device_get_memory` is UMA-blind on HIP — the `/proc/meminfo` branch in
   `ggml-cuda.cu` is `#if defined(__linux__) && !defined(GGML_USE_HIP)`, so HIP reports the GTT window as
   "VRAM", which is what feeds AUTO, `--fit` and the arena floor.  The guard is the safety net.
2. The slab should refuse to map a shared pool itself (defence in depth), not rely on the cache preflight
   having declined first.
3. A real "is the VRAM aperture a partition?" detector (so the Windows carve-out needs no manual reserve).
4. `MOE_EXPERT_CACHE_MIN_RES_PCT`'s auto floor is computed from `host_expert_bytes`, the wrong basis on a
   shared pool; inert while the arena is refused, but it must not be resurrected by (1) without re-deriving.
