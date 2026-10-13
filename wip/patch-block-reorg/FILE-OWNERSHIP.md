# File ownership and coupling

Companion to `README.md` and `REBASE-RESILIENCE.md`. Goal (G2): a target block owns the files/regions it
modifies, and where a shared file is unavoidable, one block owns each region and later blocks are
append-only. Target numbers are the **18-block (00–17)** scheme from `TARGET-BLOCKS.md`
(qwen4exp 12/13 precede host MoE 14/15/16 per Q1; getenv folded).

---

## 1. Pairwise coupling matrix (old blocks)

Shared-file counts (template instances collapsed to one synthetic file). Read as "these two old blocks
both modify N files".

```
       00   01   02   03   04   05   06   07   08   09   10   11   12   13   14   15
 00     0    1    1    0    0    0    1    0    1    0    0    0    0    0    0    2
 01     1    0    4    0    1    0    7    0    1    0    0    0    0    1    6    8
 02     1    4    0    1    4    0    4    0    1    0    1    0    0    2    8   14
 03     0    0    1    0    4    0    2    0    5    1    2    2    1    2    2    8
 04     0    1    4    4    0    0    2    0    3    1    3    1    1    4    5    9
 05     0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
 06     1    7    4    2    2    0    0    1    4    4    1    2    1    5   16   42
 07     0    0    0    0    0    0    1    0    0    0    0    0    0    0    1    1
 08     1    1    1    5    3    0    4    0    0    1    3    2    1    5    6   18
 09     0    0    0    1    1    0    4    0    1    0    1    1    1    1    0    4
 10     0    0    1    2    3    0    1    0    3    1    0    1    1    4    2    5
 11     0    0    0    2    1    0    2    0    2    1    1    0    1    1    1    2
 12     0    0    0    1    1    0    1    0    1    1    1    1    0    1    0    1
 13     0    1    2    2    4    0    5    0    5    1    4    1    1    0    7   12
 14     0    6    8    2    5    0   16    1    6    0    2    1    0    7    0   41
 15     2    8   14    8    9    0   42    1   18    4    5    2    1   12   41    0
```

Notes: block 05 shares no file with any other old block (a pure CPU one-liner) → trivial absorb into 00.
Block 15 shares 42 files with old-06, 41 with old-14, 18 with old-08, 14 with old-02, 12 with old-13 — the
coupling centre. Old-06 shares 42 with old-15, 16 with old-14, 7 with old-01.

## 2. Hub files and their target owner/region

| hub file | primary owner(s) | later appenders | region notes |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 01 (comm init/dispatch), 04/05 (FA dispatch), 08/09 (MMQ/MMVQ dispatch + J selection), 10/11 (try_fuse arms), 14 (slab/device alloc), 15 (cache hooks), 16 (graph), 12/13 (QSA/HC dispatch) | — | biggest hub; see §3 |
| `ggml/src/ggml-cuda/common.cuh` | 04 (type-KV), 14 (h2d staging helpers), 12 (arch predicates) | 05/06, 09, 11, 15, 16 | append-only helpers |
| `ggml/src/ggml-cuda/mmvq.cu` / `mmvq.cuh` | 09 (VDR, nwarps, multi-row) | 11 (item-split/ksplit/fusion), 15 (cache-aware kernels) | 09/11 are the hardest seam — keep adjacent |
| `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` / `mmq-vec-dot.cuh` | 08 (config/policy), 09 (Q6_K/J cap/PR57) | 11 (routed/pair/gate), 13 (indexer mmq) | config rows should be feature-local (Phase-2 #2) |
| `src/llama-context.cpp` | 00 (readback), 02 (MTP), 03 (GDN), 05 (FA hint) | 14 (MoE/loader), 12/13 (qwen4exp, QSA) | keep each hunk a standalone block |
| `tests/test-backend-ops.cpp` | every block appends its own cases | all | permanent hub; **no trailing tests block** |
| `common/common.cpp` / `common/common.h` | 02 (MTP) | 03, 10, 11, 14, 12 | flags/config plumbing |
| `common/arg.cpp` | 02 (MTP) | 14, 12 | CLI flags |
| `common/speculative.cpp` | 00 (shared-NextN) | 02, 03, 14 | small hunks |
| `ggml/src/ggml-backend-meta.cpp` | 14 (Meta delegation + slab iface) | 00 (compute headroom), 15 (stage_gather), 12 (qwen4exp meta) | 00's headroom fix is a standalone hunk |
| `ggml/src/ggml-alloc.c` / `ggml/include/ggml-alloc.h` | 00 (view release) | 14 (drop buffers), 12 | 00's fix is a pure bugfix |
| `include/llama.h` | 02 (MTP) | 14, 12 | public API additions |
| `src/llama-model-loader.cpp` / `.h` | 14 (host-expert policy) | 11 (split-load), 15, 17 (mmap), 12 | loader is upstream-active |
| `src/llama-model.cpp` / `.h` | 14 (MoE) | 03 (GDN), 12 (qwen4exp) | |
| `ggml/src/ggml.c` / `ggml/include/ggml.h` | 04/05 (ops), 08 (MMB op) | 03 (GDN op), 12/13 (QSA/HC ops), 06/07 (derived inputs) | op definitions are upstream-active |
| `ggml/src/ggml-cuda/quantize.cu` / `quantize.cuh` | 09 (K-quant) | 11 (MoE), 15 (cache) | |
| `src/llama-memory-*.cpp` / `.h` | 03 (GDN/recurrent) | 12 (qwen4exp hybrid-idx), 13 | |

## 3. `ggml-cuda.cu` region map (the key hub)

| region | target | content |
|---|---|---|
| includes / top-of-file macros | 00, then append-only | `GGML_ENV_STR`, graph helpers |
| device/comm init + `ggml_cuda_get_device` | 01 | hybrid/NCCL init, comm dispatch |
| slab / `ggml_cuda_device_malloc` / pool | 14 | movable-boundary slab, OOM yield, pool cap |
| FA dispatch / chooser | 04/05, 06 (native), 07 (derived) | type-KV, band, native reads, derived mask |
| MMQ dispatch + `J` selection + config | 08/09 | MMB/J cap/PR57 |
| MMVQ dispatch + VDR/nwarps | 09/11 | band-uniform knobs, item-split |
| `ggml_cuda_try_fuse` arms | 10/11, 15 (cache), 12/13 (HC/QSA) | one-line call to a feature-local matcher |
| cache hooks / `MUL_MAT_ID` takeover | 15 | cache-aware folds, iface assignment |
| graph capture / recapture | 16 | skip, recapture generation |
| QSA / HC op dispatch | 12/13 | fused ops |
| `device_supports_op` predicates | per feature | each block adds its own predicate line |

**Rule:** a target block may only add/remove lines in its own region. If two target blocks need the same
line, the later block factors a helper into its own file and calls it (Phase 2, `REBASE-RESILIENCE.md`
§5.1).

## 4. Generated instantiations rule (G6)

`ggml/src/ggml-cuda/template-instances/*.cu` and `generate_cu_files.py` are **build artifacts of a type
axis**. They belong to the block that introduces the axis:

| axis | owning block | examples |
|---|---|---|
| BF16 / `type_KV` tile + mma | 04 | `fattn-tile-instance-*`, `fattn-mma-f16-instance-*` |
| FA band rows / wide ncols | 05 | band instance TUs |
| native KV type arms | 06 | per-type dequant instantiation |
| MMB / dense GEMM | 08 | `mmb-*` if generated |
| MoE gate + routed compact | 11 | `SOURCE_MMQ_GATE` cases, `mmq-instance-*` |
| QSA indexer / HC | 12/13 | QSA/HC instances |

Never put an instance TU in a block other than the one that dispatches to it (that was the block-06
build failure). `generate_cu_files.py` is edited by the block that adds the case (04/05/11), each in its
own `SOURCE_*` region.

## 5. Per-target-block file surfaces

The authoritative per-block owner-file lists are in `TARGET-BLOCKS.md` ("Owner files"). When redistributing
a hunk, check the destination block's file surface first: if the file is not listed, either the target
list is wrong or the hunk belongs to another block. Update this document when the target list changes.

---

## 6. Phase-2 region map updates (V2.0, 2026-10-13)

Phase 2 split delivery concerns out of upstream hub files. New/changed regions and the block that
owns each:

| file / region | owner | note |
|---|---|---|
| `mmvq-vecdot.cuh` | 10 (moved from mmvq.cu) | the `vec_dot_q_cuda_t` typedef, `GGML_CUDA_MMVQ_MOE_DENSE` guard, `get_vec_dot_q_cuda`, `get_vdr_mmvq` (shared by the dense dispatch and the MoE kernel) |
| `mmvq-moe.cu` / `mmvq-moe.cuh` | 10 | `mul_mat_vec_q_moe` + `mul_mat_vec_q_moe_launch` (23 explicit launch instantiations); `mmvq.cu` keeps only `mul_mat_vec_q_moe_launch<type>(...)` at the `has_ids` branch |
| `fattn-rdna.cuh` | 05 | the FA native-KV policy/selectors and the band-WMMA predicates; `fattn-common.cuh` keeps only `#include "fattn-rdna.cuh"` + the predicate calls |
| `ggml-cuda-try-fuse.cu` / `.cuh` | 10 | `ggml_cuda_try_fuse`, `ggml_cuda_can_fuse`, the fusion matchers/range helpers, `ggml_cuda_should_fuse_rope_set_rows`…`find_*`; `ggml-cuda.cu` keeps the single `ggml_cuda_try_fuse()` call |
| `llama_context_set_fa_tensor_parallel_hint()` | 05 | `src/llama-context.cpp`; one call in the constructor |
| `llama_host_experts_mode_apply()` | 10 | `src/llama-model-loader.cpp`; one call in the constructor, caller's `__func__` preserved |

`ggml-cuda.cu` region map (§3) still holds for the remaining regions; the `try-fuse arms` row is now
satisfied by `ggml-cuda-try-fuse.cu`. **Residual (item 6b–d):** the movable-boundary slab
(`ggml-cuda-slab.cu`), the MoE-cache hooks (`ggml-cuda-moe-cache-hooks.cu`) and the arena
(`ggml-cuda-arena.cu`) are **not yet split** — they are state-coupled (file-static `g_slab_*` /
`g_slab_armed` are set by the MoE preflight and the narrow2 wrapper) and need a shared
`ggml-cuda-internal.cuh` or setters; `ggml-cuda.cu` is **7722** lines.

### Per-block modified-vs-added metric (V2.0 chain, 12 blocks)

`git show --name-status` per block (`add` = new files, `mod` = modified upstream files — the conflict
surface). Compare `REBASE-RESILIENCE.md` §2 (old 16-block chain).

| block | add | mod | subject |
|---|---:|---:|---|
| 00 | 0 | 9 | foundation |
| 01 | 1 | 3 | hybrid HIP all-reduce |
| 02 | 3 | 16 | adaptive MTP |
| 03 | 4 | 18 | chunked GDN |
| 04 | 0 | 8 | BF16 KV / native BF16 FA |
| 05 | 229 | 58 | FA band / native KV / derived mask |
| 06 | 3 | 2 | MMB |
| 07 | 0 | 9 | k-quant boosts |
| 08 | 8 | 27 | fused kernels |
| 09 | 6 | 51 | host-resident MoE |
| 10 | 19 | 74 | qwen4exp / HC / QSA / indexer |
| 11 | 0 | 24 | campaign remainder |

Block 05's 229 added files are almost entirely generated `template-instances/*.cu` (the axis travels
with its block).  The modified-file surface is now concentrated in blocks 05 (58), 09 (51) and 10 (74),
which is the expected residual (FA/quant/MoE hubs); the pre-Phase-2 `ggml-cuda.cu` interleave is now
split across the concern files and `ggml-cuda-try-fuse.cu`.
