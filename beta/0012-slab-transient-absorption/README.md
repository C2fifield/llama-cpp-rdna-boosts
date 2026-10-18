# `0012-slab-transient-absorption` — BETA candidate (NOT part of the delivery)

**Status: BETA 2 — 2026-10-18.  Awaiting independent validation; nothing here is promoted.**
BETA 1 was 2026-10-17.  BETA 2 adds the step-4 *measurement* fix (the draft model is loaded before the
slab), the graph-capture safety fix, the 896 MiB 2-GPU headroom default and the new diagnostics — see
"BETA 2" below.  Nothing in this directory is applied by `scripts/apply-all.sh` unless `--include-beta`
is given.

| | |
|---|---|
| patch | `0012-rdna-boosts-beta-12-slab-transient-absorption.patch` |
| sha256 | `9facba49301c0f4f6efa880692bbb48d06bde3e40f4a4c45adc97db98595c5d8` |
| cut against | `release.json`: `v12-097cf33d9-r82`, base `097cf33d9`, tree `dc48b4a9…` |
| applies on | a fresh `scripts/apply-all.sh` checkout (blocks 0000-0011), one commit |
| applied tree (this box, gfx1201) | `591528bd69930ff8a38c25ccab971dcae011e131` |
| apply | `scripts/apply-all.sh --include-beta .` (or `git am -3 <patch>` on a clean release tree) |
| campaign record | [`../../wip/slab-transient-absorption/README.md`](../../wip/slab-transient-absorption/README.md) §8-§10.16 |
| BETA 2 validation | 2x R9700 gfx1201, `Qwen3.8-Flash-Next` IQ4_NL `-ncmoe 48`, 131k ctx, MTP draft, 73k-token prefill |

## BETA 2 — 2026-10-18

### The failure

On the 2-GPU MTP config above the server aborted during the first long prefill with:

```
E ROCm error: operation not permitted when stream is capturing
E   current device: 0, in function alloc at ggml-cuda.cu:844
E   hipDeviceSynchronize()
```

and, just before it, a 400 MiB allocation failed outside the slab.  Two independent defects:

1. **The capture was started while the slab's wide band was armed.**  The MTP draft context does not call
   `slab_ring_set` (`llama-context.cpp`: `ctx_type != LLAMA_CONTEXT_TYPE_MTP`), so the target prefill's
   band stayed armed for the draft's small graph.  A capture cannot use band memory (it is reclaimed at
   the prefill -> decode drop), so `pool_leg::alloc` deliberately skips the band during capture — but the
   pre-capture warmup had been served *by* the band, so the pool held nothing for the capture and the
   capture allocated its temporaries outside the slab.  On device 1 there was nothing left, and the
   pool's OOM path then called `cudaDeviceSynchronize()` (illegal inside a capture).  The immediate
   trigger was the pool's 512 MiB free-VRAM floor refusing the allocation; the floor's only recovery is
   flush + sync, which is exactly what a capture forbids.
2. **The outside reserve was short on the heavy device.**  The failing 400 MiB allocation is the Meta
   backend's all-reduce temp (`ggml_backend_meta_graph_compute` -> `ggml_backend_alloc_buffer`), which is
   a plain (non-COMPUTE) buffer and therefore outside the slab by design.  The beta charged the draft
   reserve equally per device (`3078/2 + 704 + 128 = 2371 MiB`), but under `-sm tensor` a 1-layer draft is
   placed as a **whole layer on ONE device**: measured 1601.8 MiB on device 1 vs 1091.8 MiB on device 0.
   Device 1 was therefore ~510 MiB under-reserved, leaving 384.5 MiB free where device 0 had 890.5.

The beta's own `SLAB_STATS free=` diagnostic hid this: it called `cudaMemGetInfo()` without selecting the
device, so every device's line repeated device 0's figure (both showed 890.5; the corrected code showed
890.5 / 384.5).  That is why the 2-GPU "300 MiB excess per device" measurement looked safe.

### The fix

* **Measure, don't predict: the draft model is loaded before the target context.**  `common_init_result`
  loads the draft *model* (weights) with `common_speculative_load_model()` before `llama_init_from_model()`
  and hands ownership to the speculative init.  The slab is created inside the target context and sizes
  itself from the free VRAM it measures, so the draft's per-device (asymmetric) weight footprint is
  measured exactly; device 0's surplus becomes arena and device 1's deficit becomes a smaller arena.
  The draft *context* still comes after the target context (it needs `ctx_other`).
* **The reserve carries only the draft context**: `aux = kv + compute + slack` (whole KV per device, the
  safe bound, because the KV follows the layer placement too).  It is no longer charged the weights,
  which are already resident.  `--fit` still reserves the full draft footprint (conservative, safe).
* **Capture is never started while the wide band is armed.**  Replay of an already-captured graph is
  unaffected; a deferred capture does one direct run on the first disarmed pass (so the pool really
  holds its temporaries) and captures on the next.  The pool's floor/retry path is also capture-safe
  now: no floor and no `cudaDeviceSynchronize()`/`clear_pool()` inside a capture.
* **Headroom default 512+128*n -> 640+128*n** (768 / 896 / 1024 MiB).  The 2-GPU MTP decode steady state
  left only **148 MiB** free at 768 (the decode-phase graph captures' pool temporaries are outside the
  slab by design and cached for the run); the same run settles at **~270 MiB** at 896.
* **Diagnostics**: per-device `free=` fixed; `outside_max` / `outside_failed` / `free_min` in
  `SLAB_STATS`; `GGML_CUDA_GRAPH_TRACE=1`.

### Measured (this box, 2x R9700, default env)

73 527-token prefill, default env (`GGML_CUDA_SLAB_STATS=1`):

| | dev0 | dev1 |
|---|---:|---:|
| `SLAB_INIT free` (draft weights resident) | 27.21 GiB | 26.71 GiB |
| reserve (headroom 896 + aux 1135) | 1.98 GiB | 1.98 GiB |
| free at wide-arm | 1012.5 | 1014.5 |
| 400 MiB reduce temp | fits | fits (`outside_failed=0`) |
| `free_min` during the prefill | 412.5 | 414.5 |
| free at wide-drop | 436.5 | 438.5 |
| **free in the decode steady state** | **270** | **268** |
| arena at the wide-drop | 18 880 | 18 368 |

The arena is asymmetric by exactly the measured draft split (dev0 +512 MiB), the run has no allocation
failure, and the prefill runs at 945-947 t/s.  At the old 768 default the same run left 148 MiB in the
decode steady state (no failure, but very thin).

**Not yet re-gated on this tree.**  The r80 campaign's coherence, adaptive-MTP and prefill-logit results
do not automatically carry over: the arena size changes expert-cache residency, which can flip greedy
near-ties.  Run the coherence gate, the adaptive-MTP gate and `scripts/gate-prefill-logits.sh` on the
applied tree before promotion.

## Why this is a beta and not a block

It changes **when memory lives and dies** across the whole ROCm backend, and the failure mode of getting
that wrong is a hard VRAM abort (or worse, a silently corrupt graph) on a configuration we have not seen.
The original work was developed and measured on one box (1x R9700 gfx1201, plus 2- and 3-GPU pairs of the
same card); BETA 2 adds the 2-GPU Flash-Next MTP measurement above.  The campaign's own gates have been
run on the BETA 1 tree (greedy coherence, the field GSQ seesaw, the wide `-ub 6144` gate, 2/3-GPU both
split modes, the prefill-logit KLD gate) but **not** on other RDNA families (gfx1100/gfx1151), other ROCm
lines (7.14 was not re-run), other model families, or third-party configurations.  That is exactly what
this beta window is for.

## What it changes

1. **H2D staging ring is slab-only** in host-expert mode (a shortfall refuses instead of `cudaMalloc`-ing
   outside), and a slot that migrates into the slab frees its old block.
2. **Wide-phase auxiliary region**: bump-allocated above the live work boundary, released with the ring at
   the prefill -> decode drop.  It hosts the **FA prefill staging arena**, the **QSA3 prefill workspace**
   and the **chunked-GDN prefill scratch**.  It tracks its **live blocks** (a plain bump stack ratchets:
   7.4 GiB of holes in one 6.6k prefill).
3. **The workspace pool** (`ggml_cuda_pool_leg`) serves its **large** prefill temporaries from that band
   (`set_wide_routing` + `ggml_cuda_pool_wide_pin` for the retained sites), and a **`POOL_WIDE_LEAK`
   tripwire** fires if a pool block is still live when the band is released.
4. **The per-graph caches** (MMB activation cache + producer slots, BF16 src1) are released at the **end**
   of a compute instead of the start of the next one, so nothing retained crosses a graph boundary.
5. **The load-time H2D calibration probe** (512 MiB) comes from the slab arena.
6. **The MTP draft model is loaded before the target context** (BETA 2), so the slab measures the draft's
   per-device weight footprint from free VRAM instead of estimating it; the reserve carries only the draft
   context (KV + narrow-2 compute + slack).  `--fit` reserves the full draft (conservative).
7. **`GGML_CUDA_SLAB_HEADROOM_MIB` default 512+128*n -> 640+128*n** (768 / 896 / 1024 MiB).
8. **Capture safety** (BETA 2): no capture/recapture while the wide band is armed; the pool's floor/OOM
   path never synchronizes or flushes inside a capture.
9. **Diagnostics** (BETA 2): per-device `free=`, `outside_max`, `outside_failed`, `free_min`,
   `GGML_CUDA_GRAPH_TRACE`.

## Measured on the development box (BETA 1 history)

Excess = free VRAM outside the slab at the worst arm/drop of the run.  Target: >= 300 MiB.

| topology / config | headroom | excess/device | arena/device | vs r80 |
|---|---:|---:|---:|---|
| 1 GPU, field GSQ seesaw (4 requests) | 640 | 346.5 | **19 776** MiB | 15 666.6 → **+4.0 GiB** |
| 1 GPU, wide IQ4_NL `-ub 6144 -c 204800` | 1024 | 599.0 | 12 416 MiB | — |
| 2 GPU `-sm layer` | 768 | 401.5 | 17 344 MiB | — |
| 2 GPU `-sm tensor` | 768 | 300.0 (dev0 only — see BETA 2) | **18 752** MiB | — |
| 3 GPU `-sm layer` | 896 | 519.5 | 17 664-18 176 MiB | — |
| 3 GPU `-sm tensor` | 896 | 408.0 | **20 608** MiB | — |

Prefill 1 238 t/s on the field config (was 1 165), wide gate 619 t/s (was 610), decode unchanged within noise,
35B-A3B greedy text identical, prefill-logit KLD 0.000000 / 100 % same-top-p (BETA 1 tree).

## Known risks / what to watch for

* **A thin outside cushion is a hard abort.** The slab cannot unmap to help a `cudaMalloc` that cannot be
  satisfied.  If you see `ggml_cuda_device_malloc: ... allocation failed ... the arena cannot yield physical
  back to the driver` (often during context creation, graph capture or the first request), **your
  configuration needs more headroom**: raise `GGML_CUDA_SLAB_HEADROOM_MIB` (e.g. 1024, or 2048) and please
  report the config + the message.
* **`outside_failed=` in `SLAB_STATS` is the direct signal.**  A non-zero value is the largest allocation
  the outside cushion refused; report it with the config.  `free_min=` is the low-water mark.
* **`MOE_EXPERT_CACHE_AUX_RESERVE_MIB` is now the **context** reserve** (the draft weights are loaded
  before the slab), not the whole draft footprint.  It is taken verbatim per device.
* **The arena is deliberately asymmetric under `-sm tensor`** (BETA 2): each device's slab is sized from
  its own measured free VRAM after the draft weights, so a 1-layer draft's whole layer on one device
  shows up as a smaller arena there, not as a short outside cushion.
* **Unequal `-ts` with `-sm tensor`**: the weights are measured too, so the same rule applies; no
  speed/VRAM projection is promised there.
* **The expert-cache *budget* changes the numerics** (pre-existing, isolated while developing this) — via
  **table residency**, not the hit rate.  A cache-managed table is offloaded to the GPU even at one token
  (it must, to read the compact arena); a non-cache-managed host-expert table is offloaded only at
  `>= GGML_OP_OFFLOAD_MIN_BATCH` (**32** by default) and otherwise runs **on the CPU**.  So changing the cache
  budget can flip a whole table between CPU and GPU arithmetic in decode, which flips greedy text at
  near-ties.  A hit vs a miss *inside* a cache-managed table is arithmetic-neutral (same bytes, same kernel).
  For an A/B comparison, pin `MOE_EXPERT_CACHE_MIB` to the same value on both arms (or set
  `GGML_OP_OFFLOAD_MIN_BATCH=1`); the prefill-logit KLD gate is unaffected.
* `GGML_CUDA_SLAB=0` restores the pre-slab allocation path entirely; `GGML_CUDA_SLAB_HEADROOM_MIB=<verbatim MiB>`
  pins the cushion; `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` pins the context reserve; `GGML_CUDA_POOL_WIDE=0` turns
  off the workspace-pool redirect; `GGML_CUDA_GRAPH_TRACE=1` traces capture/deferral.  Any of these is a
  valid A/B arm.

## Promotion

If the beta comes back clean (or with issues fixed), this candidate is folded into the blocks that own the
code it touches (block 09 is the likely home once it is settled), the block set is regenerated and
re-validated, and this directory moves to `archive/work/slab-transient-absorption/` with a `RESOLUTION.md`
and a `WORKLOG.md` entry.  If it is refuted, the same move records why and what would re-open it.

**Open follow-up (not in this beta):** the draft's per-device KV is charged whole to every device (safe,
bounded by `(n-1)/n` of the KV).  If the draft's layer placement is later available from the loaded model,
the KV can be measured the same way as the weights; a future move to streaming KV weights may make the
whole term moot, so this is deliberately deferred.
