# `slab-ring-wide-alias` — the wide compute view was placed inside the carved H2D staging ring

**Status: FIXED in `v12-097cf33d9-r75` (2026-10-16, blocks 10/11).**  Found while running the r74 gate
plan past the dense/short checks the r74 handover ran.  This is the record of the root cause, the
evidence and the fix.

## 0. Summary

r74 moved the wide (prefill) compute-buffer placement onto a new `ggml_cuda_slab_ring_layout()` helper
that recomputed the ring from the mutable global `g_slab_ring_region_bytes` (the *requested* ring size)
instead of the region the slab had actually **carved** (`s.ring_region`).  The context re-applies its
auto ring cap *after* the carve, so the request changed from 1201.2 MiB to 975.3 MiB and the recomputed
ring became 1024 MiB while the carved ring stayed 1216 MiB.

`ggml_cuda_slab_work_alloc()` therefore placed the wide view at `768 + 1024 = 1792 MiB` instead of
`768 + 1216 = 1984 MiB` — **inside** the carved H2D staging ring `[768, 1984)`.  Both sub-ranges map the
same slab VRAM, so the wide compute buffer and the staging ring aliased each other.  On the 35B-A3B +
host experts + MTP path the first wide prefill corrupted live tensors and the GPU faulted.

## 1. Layout (the invariant that was broken)

```
[0, narrow_floor)                    narrow compute (always resident)
[narrow_floor, +ring_region)          H2D staging ring        (the carved s.ring_region)
[+ring_region, boundary)              wide (prefill) compute  <- must start AT the ring's end
[boundary, mapped)                    arena
[narrow2_off, size)                   narrow-2 (MTP draft compute), pinned at the top
```

The wide view base is **`narrow_floor + ring_region`**.  It must follow the *carve*, not the *request*.

## 2. Evidence

`GGML_CUDA_WORK_ALLOC_DEBUG=1` on the 35B-A3B + `-ncmoe 40` + `draft-mtp` config:

```
r73  WORK_ALLOC dev=0 need=1024.0 MiB narrow=0 ret_off=1984.0 MiB narrow_floor=768.0 ring=1216.0 boundary=1024.0
r74  WORK_ALLOC dev=0 need=1024.0 MiB narrow=0 ret_off=1792.0 MiB narrow_floor=768.0 ring=1216.0 boundary=1024.0
     RING_LAYOUT global=975.3 nf=768.0 mapped=22208.0 ring=1024.0 sring=1216.0 wide_off=1792.0
```

`global` (`g_slab_ring_region_bytes`) = 975.3 MiB (the re-armed request) while `sring`
(`s.ring_region`) = 1216 MiB (the carve).  r74's `wide_off` came from the request; r73's from the carve.

The fault (pre-fix):

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION ... kernel: void k_get_rows_float<float,float>, grid=[256,48,1]   (Q4_K_M)
Memory Fault Error ... kernel: void rope_multi<true,false,float,__hip_bfloat16>(...), faulting addr 0x7f7152042000 (Q8_0, GPU 0 and 1)
```

The corrupted tensor differs by model; the alias is the same.

## 3. Why the request changed after the carve

The ring is one global region shared by all contexts.  The two writers:

| context | `n_ubatch` | auto = `1800/n + ub/(1.7 n²)` MiB | effect |
|---|---:|---:|---|
| target (created at 0.07 s) | 2048 | **1201.2** | carves `s.ring_region` = 1216 |
| MTP draft (created at 0.28 s) | 512 | **975.3** | `set_ring_cap` overwrites the global |

The MTP draft context is built against the target model, so `model.moe_host_expert_bytes` is non-empty
and it ran the ring auto-size block — even though its weights are fully resident and it never stages a
host table through the ring.  The guard fix (block 11) gives that block the same
`cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP` exclusion the narrow-floor pin and ring-arm blocks already
had, so the drafter no longer sizes the ring at all.

## 4. The fix

**Block 10 — `ggml/src/ggml-cuda/ggml-cuda-slab.cu`.**  Split the helper so placement/seed use the carved
region; the request is read only when the region is carved:

```cpp
// carve-time only (from the request)
static size_t ggml_cuda_slab_ring_candidate(const ggml_cuda_slab & s, size_t narrow_floor);
// every placement/seed decision (the carve)
static size_t ggml_cuda_slab_wide_offset(const ggml_cuda_slab & s) {
    return s.narrow_floor + s.ring_region;
}
```

`work_alloc` and `seed_floor` now call `ggml_cuda_slab_wide_offset()`.

**Block 11 — `src/llama-context.cpp`.**  The ring auto-size block now requires
`cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP`.

## 5. Reproduce / verify

```sh
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # or .../Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0,1 build-rocm-hybrid/bin/llama-cli -m "$M" -ngl 99 -sm tensor -mg 0 -ncmoe 40 \
  -c 32768 -b 2048 -ub 2048 -ctk bf16 -ctv bf16 -fa auto -t 8 \
  --seed 42 --temp 0 --single-turn --no-display-prompt -lv 4 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -f prompts/prose-rdna-boosts.txt -n 3000
```

Pre-fix r74 (commit `9450cbb89`) aborts at the first prefill; r75 (and r73) run clean and the generated
text is byte-identical to r73.  Add `GGML_CUDA_WORK_ALLOC_DEBUG=1` to print the `WORK_ALLOC` /
`RING_LAYOUT` lines above.

## 6. Verified results

| config | pre-fix r74 | r75 | r73 |
|---|---|---|---|
| 35B-A3B Q4_K_M, 2 GPU, `-ncmoe 40`, draft-mtp n3, 2K prefill | SIGABRT `k_get_rows_float` | rc=0 | rc=0 |
| 35B-A3B Q4_K_M `-n 3000` | crash | `sha 0225e8352231`, acc 0.71471, len 3.14 | identical |
| 35B-A3B Q8_0 `-ncmoe 40`, 2 GPU, same | SIGABRT `rope_multi` | rc=0 | rc=0 |
| 35B-A3B Q8_0 `-n 3000` | crash | `sha 69e8ecfffcef`, acc 0.67643, len 3.03 | identical |

POST-fix variants also clean: `MOE_EXPERT_CACHE_SIZE_AT_PRIME=0`, `GGML_CUDA_DISABLE_GRAPHS=1`,
`GGML_CUDA_GDN_CHUNKED=0`, `draft-mtp-adaptive`.  GSQ field arena (q8_0 KV, `--n-cpu-moe 48`) unchanged:
1-GPU 15666.6 MiB at load, evict ~6.1 GiB, re-arm full, hit ~90.3 %; 2-GPU 35569.6 MiB (97.8 %); 3-GPU
40925 MiB (100 %).
