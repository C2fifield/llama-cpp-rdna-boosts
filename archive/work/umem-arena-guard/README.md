# `umem-arena-guard` — never auto-size the MoE expert arena on a shared host/device memory pool

**Status: OPEN, fix implemented and measured on the box that found it (2026-10-19); not part of the
delivery.**  Nothing here may be applied to the fork or folded into `patches/` without the maintainer's
go-ahead (see the WIP and promotion rules in `AGENTS.md`).  Patch:
[`umem-arena-guard.patch`](umem-arena-guard.patch) (sha256 `b05afe53…`), cut against the `release.json`
r89 tree.  Working branch: `wip/umem-arena-guard` in this repo and in `~/llama.cpp`.

> **Cold start.**  Read this file, then `AGENTS.md` ("Critical facts" → the `-ncmoe` / arena bullets, and
> "WIP and promotion rules"), `ENVIRONMENT.md` §1.2.1/§4 (`MOE_EXPERT_CACHE_*`, `GGML_CUDA_SLAB_*`), and
> `patches/README.md` block 09 (the slab and the expert cache).  Report findings to the maintainer before
> any delivery change and wait.

## 0. The defect

`MOE_EXPERT_CACHE_MIB` unset means **AUTO: size each device's arena from its free VRAM**
(`moe-expert-cache.cu`, `g_auto`), and the movable-boundary slab sizes itself as
`free VRAM - <reserve>`.  Both assume device memory is a **separate physical pool** from host RAM.

On an integrated GPU it is not.  On this box (gfx1151 Strix Halo, Ryzen AI MAX+ 395, 16 cores, 123 GiB
RAM) the APU carves out 2 GiB of VRAM and puts everything else behind GTT:

```
/sys/class/drm/card1/device/mem_info_vram_total = 2147483648        # 2 GiB carve-out
/sys/class/drm/card1/device/mem_info_gtt_total  = 130023424000      # ~121 GiB, i.e. host RAM
llama-bench -p 0 -n 1  ->  "ggml_cuda_init: found 1 ROCm devices (Total VRAM: 124000 MiB)"
```

So "free VRAM" is the whole machine, and the weights, the pinned `ROCm_Host` master and the
slab/arena all come out of one budget.  Two measured consequences:

| arm (35B-A3B, `-ncmoe 40`, 1 device) | result |
|---|---|
| `MOE_EXPERT_CACHE_MIB` unset (AUTO) | 121 GiB resident **+ 13 GiB of swap**, still loading/decoding after ~15 min; killed |
| `MOE_EXPERT_CACHE_MIB=4096` + `GGML_CUDA_SLAB=1` (arena bounded!) | `SLAB_INIT dev=0 free=99.24 GiB reserve=1.00 GiB … size(VA)=98.44 GiB **mapped=98.19 GiB**` → same overcommit |

The second row is the important one: **bounding the arena is not enough.** The arena cannot be served
without the slab, and the slab maps `free - reserve`, so on a shared pool it maps the machine.  Any
guard that only caps the arena silently hands the problem to the slab.

There is also nothing to gain.  An expert that lives in the host master is already device-reachable
through the same physical memory, so an arena slot is a copy tier over bytes the device can read anyway
-- which is exactly the maintainer's call: *"we should simply block people from trying to set the arena
up on a shared memory host.  It's slower, and it makes zero logical sense.  If the weights can fit on
the host, then they can fit on the GPU since the memory is shared."*

## 1. The guard

In `moe_cache_preflight` (the model-load seam that also arms the slab), keyed on
`cudaDeviceProp_t::integrated`:

| device | env | outcome |
|---|---|---|
| discrete | anything | unchanged |
| shared pool | default | **cache off**, one WARN; no arena, no pool, no heat map, no slab (the wrapper disarms it), no cache-band fusions — the run is the documented host-expert path |
| shared pool | `MOE_EXPERT_CACHE_UNIFIED=1`, AUTO | off, WARN: AUTO cannot size a shared pool |
| shared pool | `MOE_EXPERT_CACHE_UNIFIED=1` + explicit MIB, no slab reserve | off, WARN: the slab would map the machine |
| shared pool | `MOE_EXPERT_CACHE_UNIFIED=1` + explicit MIB + explicit `GGML_CUDA_SLAB_RESERVE_MIB` | **on**, WARN once; the slab is bounded by construction (mapped = free − reserve) |

Notes on the design:

* The refusal runs **before any table registers**, so `g_enabled == false` leaves the entire cache stack
  inert and the run is byte-identical to `MOE_EXPERT_CACHE_MIB=0` (verified below).  A second preflight
  (the `common` draft/MTP aux call) returns early, so the message appears once.
* The override deliberately requires **two** knobs.  Whether an integrated GPU's VRAM aperture is a real
  partition (Windows can cap an APU to a fixed carve-out — the one case where the arena is defensible) or
  the whole machine is not detectable from `prop.integrated` alone, so the operator asserts the bound
  instead of the guard guessing.  A proper detector is follow-up (c) below.
* Detection uses the **live** `prop.integrated`, not `ggml_cuda_info().devices[].integrated`: that flag
  is forced false on CUDA builds (upstream #15034) and honours `GGML_FORCE_NO_INTEGRATED` on HIP, and both
  are about the host-buffer/UMA weight path.  `prop.integrated` is also what
  `ggml_backend_cuda_device_get_type()` reports, so the fit reserve (below) sees the same topology.
* `common/fit.cpp` keeps the same predicate: `--fit` no longer reserves slab headroom + arena for a
  device where the arena will be refused.  Measured saving on this box: the 2444 MiB reservation over
  13578 MiB of host experts is gone.

## 2. Measured on the box that found it (2026-10-19, gfx1151, ROCm 10.1.0, Qwen3.6-35B-A3B UD-Q3_K_M)

| arm | log | peak RSS | notes |
|---|---|---:|---|
| default (guard) | `expert cache OFF on device 0: integrated GPU whose "VRAM" is the same pool as host RAM…` | 15.4 GiB | + `--fit` "shared host/device memory pool … no slab headroom or arena is reserved" |
| `MOE_EXPERT_CACHE_UNIFIED=1` (AUTO) | `… needs an explicit MOE_EXPERT_CACHE_MIB …` | 15.4 GiB | |
| `MOE_EXPERT_CACHE_UNIFIED=1 MOE_EXPERT_CACHE_MIB=4096` | `… also needs an explicit GGML_CUDA_SLAB_RESERVE_MIB …` | 15.4 GiB | the arm that over-committed before the second condition existed |
| `MOE_EXPERT_CACHE_UNIFIED=1 MOE_EXPERT_CACHE_MIB=2048 GGML_CUDA_SLAB_RESERVE_MIB=81920` | `SLAB_INIT … reserve=80.00 GiB … mapped=19.19 GiB` | 15.3 GiB | override works, bounded by construction |

Coherence: the guarded default and `MOE_EXPERT_CACHE_MIB=0` produce **identical** generated text
(seed 42, greedy, 64 tokens, 35B-A3B `-ncmoe 40`, diff clean apart from the timing line) -- i.e. the
guard's only behavioural change is that the arena does not exist.

## 3. Gates — **run on this box 2026-10-19** (baseline build = `rdna-boosts` tip `15552e758`, r89 tree)

| gate | baseline | with the patch | verdict |
|---|---|---|---|
| `scripts/gate-prefill-logits.sh` (9B Q6_K, gfx1151, 40 x 512, no explicit MIB needed) | base recorded, PPL **7.2099** (`06d1b8e5…`), self-compare 0.000000 / 100.000 % | mean KLD **0.000000**, same-top-p **100.000 %** | PASS |
| dense coherence, 4B Q8_0 seed 42 `--reasoning off`, 32 tokens | `b0e99b1137b4…` (generated text) | **identical** | PASS |
| MoE `-ncmoe 40` 35B-A3B Q3_K_M, seed 42 greedy 64 tokens | `MOE_EXPERT_CACHE_MIB=0` arm | **identical** (guard == explicit cache-off) | PASS |
| `test-recurrent-state-depth` (35B-A3B, gfx1151) | **153** failures | **153** | unchanged |
| `test-recurrent-state-rollback` | 0 failures | **0** | PASS |

> `test-recurrent-state-depth`'s baseline is recorded in `AGENTS.md` as 130 failures; on **gfx1151 with this model**
the baseline is **153**.  Compare like for like (same box/model), not to the 130.

Still to run before promotion:

* `scripts/validate-set.sh` — only after the change is folded into a block (the WIP patch is not in `patches/`).
* **gfx1201 / dGPU regression**: the guard is a no-op there (`integrated == false`), so verify the cache still
  arms with AUTO on the primary box (one 35B-A3B `-ncmoe 40` load: "auto cache preflight … cache stays on"
  + `SLAB_INIT`).
* The adaptive-MTP four-axis gate on gfx1201 unchanged; on gfx1151 the arena is now off by default, so the
  host-expert field that gate exists for does not exist there.
* `ENVIRONMENT.md`: the `MOE_EXPERT_CACHE_UNIFIED` row + a note in the `MOE_EXPERT_CACHE_MIB` row that AUTO
  is refused on a shared pool.  `README.md`'s `-ncmoe` guidance gets one sentence.

## 4. Follow-ups (do not fold into this change)

1. **`device_get_memory` is UMA-blind on HIP.**  The CUDA-only `/proc/meminfo` branch
   (`ggml-cuda.cu`, `#if defined(__linux__) && !defined(GGML_USE_HIP)`) means HIP reports the GTT window
   as "VRAM", which is what feeds AUTO, `--fit` and the arena floor.  Fixing it upstream-style is the real
   root cause; the guard here is the safety net.
2. **The slab should refuse to map a shared pool itself** (defence in depth), rather than relying on the
   cache preflight having declined first.  Today `GGML_CUDA_SLAB=1` + any later arming path would still
   map `free - reserve` on an APU.
3. **A real "is the VRAM aperture a partition?" detector** so the Windows carve-out case does not need
   `GGML_CUDA_SLAB_RESERVE_MIB`: compare `totalGlobalMem` against host RAM, or read the driver's
   `mem_info_vram_total` (2 GiB here vs the GTT window) where the platform exposes it.
4. `MOE_EXPERT_CACHE_MIN_RES_PCT`'s auto floor is computed from `host_expert_bytes`, which is also the
   wrong basis on a shared pool — inert now that the arena is refused, but it must not be resurrected by
   (1) without re-deriving it.
