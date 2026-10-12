# File ownership and coupling

Companion to `README.md` and `REBASE-RESILIENCE.md`. Goal (G2): a target block owns the files/regions it
modifies, and where a shared file is unavoidable, one block owns each region and later blocks are
append-only.

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

Notes: block 05 shares no file with any other old block (it is a pure CPU one-liner) → trivial absorb into
00. Block 15 shares 42 files with old-06, 41 with old-14, 18 with old-08, 14 with old-02 and 12 with
old-13 — it is the coupling centre. Old-06 shares 42 with old-15, 16 with old-14, 7 with old-01. Old-01
and old-14 share 6 (the userland MTP/flag files).

## 2. Hub files and their target owner/region

"Primary owner" is the block that should introduce the region and is expected to touch it; "later
appenders" list the other target blocks that must add to the same file. A later appender should keep its
edit contiguous and away from the owner's region.

| hub file | primary owner(s) | later appenders | region notes |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 01 (comm init/dispatch), 04/05 (FA dispatch), 08/09 (MMQ/MMVQ dispatch + J selection), 10/11 (try_fuse arms), 12 (slab/device alloc), 13 (cache hooks), 14 (graph), 16/17 (QSA/HC dispatch) | — | the biggest hub; see §3 for the region map |
| `ggml/src/ggml-cuda/common.cuh` | 04 (type-KV), 12 (h2d staging helpers), 16 (arch predicates) | 05/06, 09, 11, 13, 14, 18 | append-only helpers; each helper is a distinct block |
| `ggml/src/ggml-cuda/mmvq.cu` / `mmvq.cuh` | 09 (VDR, nwarps, multi-row) | 11 (item-split/ksplit/fusion), 13 (cache-aware kernels) | 09 and 11 are the hardest seam — keep adjacent |
| `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` / `mmq-vec-dot.cuh` | 08 (config/policy), 09 (Q6_K/J cap/PR57) | 11 (routed/pair/gate), 17 (indexer mmq) | config rows should be feature-local (Phase-2 #2) |
| `src/llama-context.cpp` | 00 (readback), 02 (MTP), 03 (GDN), 05 (FA hint) | 12 (MoE/loader), 16/17 (qwen4exp, QSA) | keep each hunk a standalone block |
| `tests/test-backend-ops.cpp` | every block appends its own cases | all | permanent hub; keep each block's addition contiguous; **no trailing tests block** |
| `common/common.cpp` / `common/common.h` | 02 (MTP) | 03, 10, 11, 12, 16, 18 | flags/config plumbing |
| `common/arg.cpp` | 02 (MTP) | 12, 16, 18 | CLI flags |
| `common/speculative.cpp` | 00 (shared-NextN) | 02, 03, 12 | small hunks |
| `ggml/src/ggml-backend-meta.cpp` | 12 (Meta delegation + slab iface) | 00 (compute headroom), 13 (stage_gather), 16 (qwen4exp meta) | 00's headroom fix is a standalone hunk |
| `ggml/src/ggml-alloc.c` / `ggml/include/ggml-alloc.h` | 00 (view release) | 12 (drop buffers), 16 | 00's fix is a pure bugfix |
| `include/llama.h` | 02 (MTP) | 12, 16, 18 | public API additions |
| `src/llama-model-loader.cpp` / `.h` | 12 (host-expert policy) | 11 (split-load), 13, 15 (mmap), 16 | loader is upstream-active |
| `src/llama-model.cpp` / `.h` | 12 (MoE) | 03 (GDN), 16 (qwen4exp) | |
| `ggml/src/ggml.c` / `ggml/include/ggml.h` | 04/05 (ops), 08 (MMB op) | 03 (GDN op), 16/17 (QSA/HC ops), 06/07 (derived inputs) | op definitions are upstream-active |
| `ggml/src/ggml-cuda/quantize.cu` / `quantize.cuh` | 09 (K-quant) | 11 (MoE), 13 (cache) | |
| `src/llama-memory-*.cpp` / `.h` | 03 (GDN/recurrent) | 16 (qwen4exp hybrid-idx), 17 | |

## 3. `ggml-cuda.cu` region map (the key hub)

Ordered by where the region sits in the file; the target block that owns each region:

| region | target | content |
|---|---|---|
| includes / top-of-file macros | 00, then append-only | `GGML_ENV_STR`, graph helpers |
| device/comm init + `ggml_cuda_get_device` | 01 | hybrid/NCCL init, comm dispatch |
| slab / `ggml_cuda_device_malloc` / pool | 12 | movable-boundary slab, OOM yield, pool cap |
| FA dispatch (`ggml_cuda_flash_attn_ext*`, FA chooser) | 04/05, 06 (native), 07 (derived) | type-KV, band, native reads, derived mask |
| MMQ dispatch + `J` selection + config | 08/09 | MMB/J cap/PR57 |
| MMVQ dispatch + VDR/nwarps | 09/11 | band-uniform knobs, item-split |
| `ggml_cuda_try_fuse` arms | 10/11, 13 (cache), 16/17 (HC/QSA) | one-line call to a feature-local matcher |
| cache hooks / `MUL_MAT_ID` takeover | 13 | cache-aware folds, iface assignment |
| graph capture / recapture | 14 | skip, recapture generation |
| QSA / HC op dispatch | 16/17 | fused ops |
| `device_supports_op` predicates | per feature | each block adds its own predicate line |

**Rule:** a target block may only add/remove lines in its own region. If two target blocks need the same
line, the later block must factor a helper into its own file and call it (Phase 2, `REBASE-RESILIENCE.md`
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
| QSA indexer / HC | 16/17 | QSA/HC instances |

Never put an instance TU in a block other than the one that dispatches to it (that was the block-06
build failure). `generate_cu_files.py` is edited by the block that adds the case (04/05/11), each in its
own `SOURCE_*` region.

## 5. Per-target-block file surfaces

The authoritative per-block owner-file lists are in `TARGET-BLOCKS.md` ("Owner files"). When redistributing
a hunk, check the destination block's file surface first: if the file is not listed, either the target
list is wrong or the hunk belongs to another block. Update this document when the target list changes.
