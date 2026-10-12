# Strata's AMD decode kernels vs our MMB/HIP kernels

> **RESOLVED — CLOSED AS REFUTED (2026-10-11, r48).**  Archive record; not part of the delivery and not a
> live campaign.  Maintainer's decision (2026-10-11): *"Close as refuted."*
>
> **Outcome:** K0 (line-by-line source diff) and K1 (the one survivor, measured) are both done.  Nothing
> ported.  **#262 is already delivered** (upstream `vecdotq.cuh` AMD perm arm + SWAR `vendors/hip.h`, plus
> our own `apply_ksigns`/`mmb_pack2`); the `router_top10`, `fused_gr` carveout, `#646` IQ-grid-staging and
> `verify` items are no-deltas; and **arena THP** — the only candidate that survived K0 — was measured and
> refuted (the GPU's read bandwidth out of the pinned host arena is identical whether the buffer is
> 4 KiB-backed or 100 % 2 MiB-backed).  The lone residual, the **`Q1_0` `__byte_perm` sites**, was split
> out into the new campaign `wip/q1-0-byte-perm/` (TODO #54).  The two THP probes were deliberately **not**
> preserved ("no point adding something that provides no benefit") — their method and numbers are recorded
> below.  `TODO #35` is closed; see `WORKLOG.md` 261011.016.

**Original status line (historical): K0 COMPLETE + K1 REFUTED (2026-10-11, r48).** Maintainer:
"worthwhile if they're better than our MMB style kernels we have today."  The comparison was done against
the **r48** tree (`~/llama.cpp`, branch `rdna-boosts`, tip `0bc12586413e7fe4547916dc3ed43453b3eb09b8`).

Related: `patches/README.md` block 10 (`mmb.cu` IQ dequant, `__mul24` exact scales), block 13 (`mmvq.cu`
multi-row verify blocks, `mul_mat_vec_q_moe`), block 15 (`topk-moe`, MoE gate+up+GLU), block 11 (verify
graphs), block 06 (pinned host expert arena), `archive/work/mmvq-verify-rows/`, Strata
`src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/router_top10.cu`, `src/kernels/cuda/fused_gr.cu`,
`include/strata/hip_compat/intrinsics.hpp`, `src/core/pinned.cu`, and the gfx1201 findings in Strata
`docs/AMD_HIP.md`.

## K0 findings — candidate by candidate (source-level, r48)

| candidate | verdict | evidence |
|---|---|---|
| **#262 packed byte ops** | **ALREADY DELIVERED — no delta.** | Strata's #262 = `__byte_perm` → `v_perm_b32` + SWAR `__vsub4`/`__vsubss4`/`__vcmpne4` (Strata `87fbc000`, `600d7307`). Our tree already has **all** of it, from two independent sources: (a) upstream base `a55e952b8` already ships the AMD `__builtin_amdgcn_perm` arm in `vecdotq.cuh::get_int_from_table_16` (8 hits, incl. `vec_dot_q2_0_q8_1`) and the SWAR `__vsub4`/`__vcmpne4` in `vendors/hip.h` (4 hits of `0x80808080`); (b) our block 10/15 add `apply_ksigns` (packed `(g ^ m) + lsb`, `vecdotq.cuh:119`) and `mmb.cu`'s `mmb_pack2`/`mmb_pack2_so` (v_perm vs shift/or, both present, chosen per-quant by measurement).  Residual: `Q1_0` only — `vecdotq.cuh:833-844` and `mmq-load-tiles.cuh:50-61` still call HIP's `__byte_perm`, whose ROCm 7.14 body is a private-memory union of 4 indexed byte loads (`amd_device_functions.h:246`).  Q1_0 is a rare 1-bit type (not on any local model); **not worth chasing**. |
| **`router_top10` fast path** | **NO DELTA — our router is a different, already-fast design.** | Strata's gain came from removing *their* router's serial FP64 sum, 10 selection passes with 20 block barriers, and a re-read of weights from global (`648ab7fa`).  Our MoE router is upstream's fused `topk_moe_cuda` (`topk-moe.cu`, present in the base; block 08 touched it): it is **warp-shuffle** based, **float** (`wt_sum`), with **no** FP64 and **no** per-selection `__syncthreads`.  The fusion is on by default (`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` is the only opt-out, `ggml-cuda.cu:5494`).  There is no serial FP64 router in our tree to remove.  (Our HIP-only `top_k_radix` path, `top-k.cu`, only engages at `ncols > 1024`; MoE routing is 256/512 experts → the bitonic path, which we do not claim as a win.) |
| **`fused_gr` LDS carveout** | **RDNA-INAPPLICABLE — the hint is a no-op.** | Strata's commit only added `#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout` to their `hip_compat` shim (`f1665640`); no AMD gain is claimed.  On AMD, LDS is a dedicated per-CU resource, not carved out of L1/L2, so `hipFuncAttributePreferredSharedMemoryCarveout` is ignored (ROCm documents the CUDA cache/carveout hints as ignored on AMD, `hip_runtime_api.h:1354`).  Our kernels set only `cudaFuncAttributeMaxDynamicSharedMemorySize`, which is the load-bearing one.  Nothing to port. |
| **arena transparent huge pages** | **GAP CONFIRMED on the host, REFUTED for the GPU — no delta.** | Strata `6d51272`: back the Linux expert arena with THP; without it "every 3 MB expert costs ~750 TLB misses".  Our pinned host arena is `cudaMallocHost` (`ggml-cuda.cu:3288`) and the bounded `--host-experts pool` is `cudaHostAlloc(Portable\|Mapped)` (`moe-expert-cache.cu:776`).  **Probe (`tools/thp-pinned-probe.cpp`, gfx1201, kernel 7.2.9):** ROCr backs both with a **shmem** mapping (`/dev/zero (deleted)`), and with the host's `shmem_enabled=never` it lands on **4 KiB pages** (`ShmemPmdMapped 0` of 4 GiB); `MADV_HUGEPAGE` alone does **not** retroactively collapse it, `MADV_COLLAPSE` does (`ShmemPmdMapped = 4194304 kB`, 100 %).  The CPU-side gap is real.  **K1 (`tools/thp-host-read-bench.cpp`) shows it does not reach the GPU:** the same buffer measured 4 KiB-backed vs 100 % 2 MiB-backed gives identical GPU read bandwidth to <0.1 % (sequential 13.49 GiB/s both; random-page scatter 0.85 GiB/s both; `PmdMapped` verified 0 % vs 100 %).  The amdgpu GMMU page tables for the userptr mapping are built at registration, independently of the host page size, so a later host-side collapse cannot change what the GPU walks.  **Drop.** |
| **#646 IQ-grid staging** | **NO ACTIONABLE DELTA — our state matches their measured-best.** | Strata's #646 finding (`f9455154`) is that *staging* the IQ grids is 10-20 % **slower** for the single-matrix mmvq at 3-8 columns on an RTX 5070, so they **do not** stage.  Our `vecdotq.cuh` mmvq reads `iq2xxs_grid`/`iq3xxs_grid`/`iq1s_grid_gpu` directly; they are `static const __device__` (`ggml-common.h:493`, i.e. global/L1, not `__constant__`), and we never stage them into LDS.  So we already sit on the side Strata measured as faster.  (Their grouped *expert* kernels keep staging and gain; our grouped MoE path is `mul_mat_vec_q_moe` in block 13, not a grid-staging design.)  RDNA4-specific staging was not measured and the mechanism (L1 hits vs LDS) does not obviously favour it; **park unless a microbench says otherwise.** |
| **`verify` zero-doorbell + sub-warp packing** | **NOT A KERNEL CANDIDATE — engine-protocol change.** | `cfd3b722` is a **CUDA** commit in Strata's own verify/window graph (zero-doorbell verify graph, sub-warp expert packing, shared-mem staging, batched PLE/MTP): "doorbell" is *their* host↔GPU handshake, not a kernel we share.  Our counterparts are block 11 (verify graphs), block 06 (op-offload H2D ring + MoE verify fusions) and block 13 (verify batches), designed independently.  Porting their window protocol is not a kernel transcription; treat it as out of scope for this campaign. |

**Not in the original list (found while diffing Strata's post-2026-10-05 work; all engine features, not kernels):**
`--pipeline-windows 2` for gfx12 layer splits (`b03304d9`), `STRATA_ROUTE_TAIL_SKIP` (`ad2aef76`),
`route-resident` one-warp-per-token swap (`248dbee5`/`33fd01b4` — a *quality-changing* speculative
cache swap tied to their 8083-slot cache, not a kernel), and the gfx12 HC decode node cuts
(`a9917a55`, `7fc02cb5`).  None is a drop-in kernel for our delivery; none is pursued here.

## Shortlist (K1)

**No candidate survives.  S1 (arena THP) was measured and refuted.**

**S1 — `MADV_HUGEPAGE` on the pinned host expert arena — REFUTED.**
The original idea: add a Linux-only `madvise` after each pinned host allocation (`ggml_cuda_host_malloc`
in `ggml-cuda.cu` and the pool's `cudaHostAlloc` in `moe-expert-cache.cu`) so the `ROCm_Host` master and
the `--host-experts pool` land on 2 MiB instead of 4 KiB pages, cutting GPU TLB misses.

- *Why it looked real:* the pinned arena is shmem-backed and **does** sit on 4 KiB pages by default
  (`shmem_enabled=never`); `MADV_COLLAPSE` collapses it to 100 % 2 MiB pages.
- *Why it is dead:* `tools/thp-host-read-bench.cpp` measures the GPU reading that same buffer over UVA,
  4 KiB-backed vs 100 % 2 MiB-backed, and the bandwidth is **identical** — sequential 13.49 GiB/s both
  arms, random-page scatter 0.85 GiB/s both arms, run-to-run spread <0.1 %.  The GMMU page tables for a
  KFD userptr mapping are built by amdgpu when the buffer is registered (ROCr pins it inside
  `hipHostMalloc`, before any `madvise` we could issue), so the host page size never reaches the GPU
  walker.  A host-side collapse after allocation cannot help; collapsing *before* registration is not
  reachable through the HIP API.
- *Residual:* the CPU side (the r35 pool's `pread`, the H2D staging memcpy) would see fewer TLB misses,
  but that is a memcpy-bound path whose cost is dwarfed by the copy itself — not worth a change.
- *If ever revisited:* it would have to be a **CPU** benchmark (host-side expert streaming), not a GPU
  one, and it would need a non-`hipHostMalloc` allocator to get huge pages at registration time.

Parked (revisit only with a reason): **P2** RDNA4 IQ-grid LDS staging microbench (Strata's own data says
don't stage for mmvq).  **P1** (Q1_0 `__byte_perm`) is no longer parked — it became its own campaign,
`wip/q1-0-byte-perm/` (TODO #54).

**Resolution (2026-10-11).**  The maintainer closed this campaign as refuted: the candidate list is
exhausted (`#262` delivered upstream + in our blocks; router, carveout, IQ-grid staging, verify all
no-deltas; arena THP measured and refuted).  This record was moved here from `wip/strata-amd-kernels/`
per the WIP rule; `TODO #35` is closed; `wip/CAMPAIGNS.md` and `WORKLOG.md` 261011.016 carry the result.
The two THP probes were **not** preserved (the maintainer: "no point adding something that provides no
benefit") — their method and their measured numbers are recorded in the Tools note below so the negative
result is reproducible without the files.

## Method (unchanged)

1. **Read both kernels side by side** and classify: instruction mix (perm/packed vs shift/or, FP64 vs
   FP32 router reduction), grid/block mapping, LDS usage, and whether the difference is already in our
   block 10/13/15 work. Only the ones with a real delta go forward. — **done in K0 above.**
2. **Microbench** the qwen4exp geometry (48 layers, 512 experts, top-10, `n_embd 2560`, `head_dim 256`)
   and the dense 27B geometry, on gfx1201 / ROCm 7.14.1, against the r48 build. A/B with env switches
   where the block already has them (`GGML_CUDA_*`).
3. **End-to-end**: single R9700, IQ3_XXS + MTP head, `MOE_EXPERT_CACHE_MIB=20480` (the auto-size
   campaign's target), `-ncmoe 48`, MTP n3, shallow + 32K. Also the fully-resident / multi-GPU case,
   where our decode is not PCIe-bound and a kernel win should translate more directly.
4. **Purity**: any change must keep `W = 1..8` bit-identical (band-uniform knobs, `GREEDY-PURITY.md`
   §19/§24/§25) and pass `llama-batched-bench -npl 1,4,8` stock-relative (`mtp-adaptive-methodology.md`
   rule 5). A kernel that changes the reduction order is not shippable here.

## Tools (NOT preserved — method recorded)

Both probes were standalone HIP programs, deleted with the close-out per the maintainer's call.  They are
trivial to recreate; this note records exactly what they did and what they measured.

- **`thp-pinned-probe`** — `hipHostMalloc`s a region, touches it, and reads `AnonHugePages` /
  `ShmemPmdMapped` / `VmFlags` / `KernelPageSize` / `Rss` from `/proc/self/smaps`, before and after
  `MADV_HUGEPAGE` / `MADV_COLLAPSE`.  Build: `clang++ -x hip --offload-arch=gfx1201` against
  `/opt/rocm-7.14.1-gfx120X`, link `-lamdhip64`.  **Measured (gfx1201, kernel 7.2.9, 4096 MiB):** the
  mapping is `rw-s` shmem (`/dev/zero (deleted)`); after first touch `ShmemPmdMapped = 0` (4 KiB pages);
  `MADV_HUGEPAGE` returns 0 but does not collapse; `MADV_COLLAPSE` returns 0 and collapses to
  `ShmemPmdMapped = 4194304 kB` (100 % 2 MiB).
- **`thp-host-read-bench`** — allocates pinned memory, optionally `MADV_HUGEPAGE`+`MADV_COLLAPSE`s it, then
  times the **GPU** reading it over UVA with a grid-stride `uint4` kernel (`read_host_bw`, sequential) and
  a xorshift index kernel (`read_host_scatter`, random-page), reporting the verified `ShmemPmdMapped`
  ratio alongside the bandwidth.  **Measured (gfx1201):** 2048 MiB sequential = **13.49 GiB/s in both
  arms**; 1024 MiB random-page scatter = **0.85 GiB/s in both arms**; `PmdMapped` verified 0 % vs 100 %
  per run; run-to-run spread <0.1 %.  (The scatter arm collapsing to 0.85 GiB/s is what confirms these
  are real PCIe/UVA host reads, not a cached path.)

## Caveat on the end-to-end ceiling

At 43 % residency our single-card decode is partly **UVA-cold-read bound** (the other campaign's
finding), so a 15 % kernel win will not fully show end-to-end there. The honest targets are the
fully-resident and multi-GPU/short-K cases. Still worth it: these are HIP/RDNA4 kernels, exactly this
repo's scope, and several are shaped like upstream PRs (`upstream/`).

## Milestones

* **K0 — Kernel inventory + diff** (no build): the table above filled in with our source line-by-line,
  and a shortlist with expected deltas. — **DONE (2026-10-11, r48).**
* **K1 — Microbench the shortlist** on gfx1201. — **DONE (2026-10-11, r48): S1 (arena THP) refuted.**
* **K2 — Port the winners** (RDNA-gated, per-change kill switch, bit-identical where the reduction order
  allows). — **no winner to port.**
* **K3 — Gates + promotion decision** (delivery block amendment, and/or `upstream/` copy). — **n/a.**
* **Close-out** — **DONE (2026-10-11, r48):** the record moved to `archive/work/strata-amd-kernels/`,
  the `wip/` entry was deleted (no stub), the probes were discarded, and `WORKLOG.md` 261011.016 records
  the result.  **Campaign closed as refuted; TODO #35 closed.**

## Attribution

Strata is MIT. Any transcribed kernel should carry the upstream attribution the delivery already uses
for the i-quant dequantizers in `~Strata third_party/ggml/LICENSE`.
