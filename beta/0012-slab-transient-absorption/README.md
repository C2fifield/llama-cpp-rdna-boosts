# `0012-slab-transient-absorption` — BETA candidate (NOT part of the delivery)

**Status: BETA 1 — 2026-10-17.  Awaiting independent validation; nothing here is promoted.**

| | |
|---|---|
| patch | `0012-rdna-boosts-beta-12-slab-transient-absorption.patch` |
| sha256 | `5f5110659ed2274e1d48942188331da187507c8e6143d75169ef3d6eed882640` |
| cut against | `release.json`: `v12-097cf33d9-r80`, base `097cf33d9`, **tip `00f4b1ef8`**, tree `873d0335…` |
| applies on | a fresh `scripts/apply-all.sh` checkout (blocks 0000-0011), one commit |
| applied tree (this box, gfx1201) | `18bb050f3ff2d47c54352c8ac04a986d4a4c4349` |
| apply | `scripts/apply-all.sh --include-beta .` (or `git am -3 <patch>` on a clean release tree) |
| campaign record | [`../../wip/slab-transient-absorption/README.md`](../../wip/slab-transient-absorption/README.md) §8-§10.14 |

## Why this is a beta and not a block

It changes **when memory lives and dies** across the whole ROCm backend, and the failure mode of getting
that wrong is a hard VRAM abort (or worse, a silently corrupt graph) on a configuration we have not seen.
It was developed and measured on one box (1x R9700 gfx1201, plus 2- and 3-GPU pairs of the same card); every
measurement below is from that box.  The patch has been through the campaign's own gates (greedy coherence,
the field GSQ seesaw, the wide `-ub 6144` gate, 2/3-GPU both split modes, the prefill-logit KLD gate) but
**not** on other RDNA families (gfx1100/gfx1151), other ROCm lines (7.14 was not re-run), other model
families, or third-party configurations.  That is exactly what this beta window is for.

## What it changes

The movable-boundary slab (block 09) lets a wide prefill and a large expert-cache arena coexist, but anything
that could not live *inside* the slab was paid for out of `GGML_CUDA_SLAB_HEADROOM_MIB` — permanently, because
the slab maps once and ROCm cannot unmap sub-ranges.  This block moves those transients into the slab and then
sizes the outside cushion from measurements:

1. **H2D staging ring is slab-only** in host-expert mode (a shortfall refuses instead of `cudaMalloc`-ing
   outside), and a slot that migrates into the slab frees its old block.
2. **Wide-phase auxiliary region**: bump-allocated above the live work boundary, released with the ring at the
   prefill -> decode drop.  It hosts the **FA prefill staging arena**, the **QSA3 prefill workspace** and the
   **chunked-GDN prefill scratch**.  It tracks its **live blocks** (a plain bump stack ratchets: 7.4 GiB of
   holes in one 6.6k prefill).
3. **The workspace pool** (`ggml_cuda_pool_leg`) serves its **large** prefill temporaries from that band
   (`set_wide_routing` + `ggml_cuda_pool_wide_pin` for the retained sites), and a **`POOL_WIDE_LEAK` tripwire**
   fires if a pool block is still live when the band is released.
4. **The per-graph caches** (MMB activation cache + producer slots, BF16 src1) are released at the **end** of a
   compute instead of the start of the next one, so nothing retained crosses a graph boundary.
5. **The load-time H2D calibration probe** (512 MiB) comes from the slab arena.
6. **The MTP draft reserve is derived** from the draft's own GGUF (weights + KV(n_ctx) at the real `-ctk/-ctv`
   element size + a measured compute tier keyed on the ubatch + slack) instead of the flat
   `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT`, and **`--fit` reserves it** (an MTP draft cannot be measured
   during fitting — its context needs `ctx_other`, which does not exist yet — and the old code silently fell
   back to zero).
7. **`GGML_CUDA_SLAB_HEADROOM_MIB` default 4096 -> `512 + 128 x n_gpu`** (640 / 768 / 896 MiB), and under an
   **equal `-sm tensor` split** the draft reserve is charged per device (its weights and KV are split, so the
   whole footprint on every device over-reserved by ~1.4 GiB/device).

## Measured on the development box (what it buys)

Excess = free VRAM outside the slab at the worst arm/drop of the run.  Target: >= 300 MiB.

| topology / config | headroom | excess/device | arena/device | vs r80 |
|---|---:|---:|---:|---|
| 1 GPU, field GSQ seesaw (4 requests) | 640 | 346.5 | **19 776** MiB | 15 666.6 → **+4.0 GiB** |
| 1 GPU, wide IQ4_NL `-ub 6144 -c 204800` | 640 | 599.0 | 12 416 MiB | — |
| 2 GPU `-sm layer` | 768 | 401.5 | 17 344 MiB | — |
| 2 GPU `-sm tensor` | 768 | 300.0 | **18 752** MiB | — |
| 3 GPU `-sm layer` | 896 | 519.5 | 17 664-18 176 MiB | — |
| 3 GPU `-sm tensor` | 896 | 408.0 | **20 608** MiB | — |

Prefill 1 238 t/s on the field config (was 1 165), wide gate 619 t/s (was 610), decode unchanged within noise,
35B-A3B greedy text identical, prefill-logit KLD 0.000000 / 100 % same-top-p.

## Known risks / what to watch for

* **A thin outside cushion is a hard abort.** The slab cannot unmap to help a `cudaMalloc` that cannot be
  satisfied.  If you see `ggml_cuda_device_malloc: ... allocation failed ... the arena cannot yield physical
  back to the driver` (often during context creation, graph capture or the first request), **your
  configuration needs more headroom**: raise `GGML_CUDA_SLAB_HEADROOM_MIB` (e.g. 1024, or 4096 for the old
  behaviour) and please report the config + the message.
* **Unequal `-ts` with `-sm tensor` keeps the full (conservative) reserve** — no speed/VRAM benefit there yet,
  and no regression expected.
* **The expert-cache *budget* changes the numerics** (pre-existing, isolated while developing this) — via
  **table residency**, not the hit rate.  A cache-managed table is offloaded to the GPU even at one token
  (it must, to read the compact arena); a non-cache-managed host-expert table is offloaded only at
  `>= GGML_OP_OFFLOAD_MIN_BATCH` (**32** by default) and otherwise runs **on the CPU**.  So changing the cache
  budget can flip a whole table between CPU and GPU arithmetic in decode, which flips greedy text at
  near-ties.  A hit vs a miss *inside* a cache-managed table is arithmetic-neutral (same bytes, same kernel).
  **This is proven**, not assumed: with `GGML_OP_OFFLOAD_MIN_BATCH=1` two arms that differ by budget produce
  *identical* text.  For an A/B comparison, pin `MOE_EXPERT_CACHE_MIB` to the same value on both arms (or set
  `GGML_OP_OFFLOAD_MIN_BATCH=1`); the prefill-logit KLD gate is unaffected.  See
  `../../wip/slab-transient-absorption/README.md` §10.15.
* `GGML_CUDA_SLAB=0` restores the pre-slab allocation path entirely; `GGML_CUDA_SLAB_HEADROOM_MIB=<verbatim MiB>`
  pins the cushion; `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` pins the draft reserve; `GGML_CUDA_POOL_WIDE=0` turns off
  the workspace-pool redirect.  Any of these is a valid A/B arm.

## Promotion

If the beta comes back clean (or with issues fixed), this candidate is folded into the blocks that own the
code it touches, the block set is regenerated and re-validated, and this directory moves to
`archive/work/slab-transient-absorption/` with a `RESOLUTION.md` and a `WORKLOG.md` entry.  If it is refuted,
the same move records why and what would re-open it.
