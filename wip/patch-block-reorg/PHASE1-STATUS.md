# Phase 1 — execution status and blocking finding (2026-10-12)

Companion to `README.md` §6. This is a **work-in-progress checkpoint**: Phase 1 has started and the
method is proven for the cleanly-separable blocks, but the flash-attention split has hit a structural
problem that needs the maintainer's decision (per `AGENTS.md` "Report findings before shipping").

---

## 1. Workspace

| ref | value |
|---|---|
| worktree | `~/llama-v2` (git worktree of `~/llama.cpp`) |
| branch | `v2-phase1` |
| base | `097cf33d9` (tree `30c400ccea2e2fb47b663036a6cee08a4dc542d8`) |
| reference tip | `84058162b` (pin `pin/r50-tip`, tree `f102245dfb116f02f6b3f572df7fb413a5c65cf8`) |
| partial-safe tag | `pin/v2-phase1-00-04` = `2a1502311` (the 00–04 checkpoint) |

Phase-1 tooling (in this campaign): `tools/select_hunks.py` (extract hunks from a source-block diff by
regex) and `tools/hunk_summary.py` (tag every hunk with the concerns its added lines mention, for
triage).

## 2. Done: target blocks 00–04 (committed on `v2-phase1`)

| target | subject | source | files / lines | status |
|---|---|---|---|---|
| 00 | foundation (band-uniformity, Meta-wrapper, allocator, readback) | old 00+05+07+09a+06-readback+15-W4 | 9 files | applied clean |
| 01 | hybrid HIP all-reduce | old 12 | 4 files | applied clean |
| 02 | adaptive MTP draft depth | old 01 | 19 files | applied clean |
| 03 | chunked gated-delta-net prefill | old 02 | 22 files | applied clean |
| 04 | BF16 KV cache + native-BF16 FA | old 03 | 8 files | applied clean |

Method that worked: start at base, and for each target `git apply -3` the union of its source hunks
(whole source blocks where the source block maps 1:1 to the target; `tools/select_hunks.py` for the
parts). Every target so far replayed with `-3` and zero conflicts, and the per-target file/line counts
match the source blocks exactly (e.g. target 03 = 22 files, +3141/−15 = old 02).

**Correction to `MIGRATION-MAP.md`:** `ggml/src/ggml-cuda/cpy-batch.cu`/`.cuh` are **new files added by
old-15** (part of the cache-blocks/`cpy_batch` work), not a target-00 device idiom. They belong to target
15. The r49 "device idiom" was a change inside that block-15-introduced file. Target 00 therefore does
**not** include cpy-batch (it was 9 files, not 11).

## 3. BLOCKING FINDING — the flash-attention stack (targets 05/06/07) cannot be hunk-split

The plan assumed the old-15 v4/v5 (native KV), v3 (derived kq-mask) and band work could be separated by
hunk. **They cannot.** Old-15's FA diffs were authored as a single evolution and squashed; a single hunk
mixes concerns. Measured hunk counts and samples:

- `fattn-mma-f16.cuh`: **44 hunks**, e.g. `[1]` mixes `ncols | band | GQA | native`; `[9]` mixes
  `derived | band | cell_pos | ncols`; `[10]`/`[11]` mix `kv_native | derived`.
- `fattn.cu`: 10 hunks, e.g. `[3]` is band/wide, `[9]` is native+alloc, `[10]` is derived.
- `fattn-tile.cuh`: 25 hunks, `[1]`/`[6]`/`[14]` mix native/derived/wide.
- `fattn-common.cuh`: 17 hunks, `[1]`–`[4]` mix derived/native.
- `common.cuh`: 16 hunks (band context, native, slab, graph-mem) — several unrelated.

`tools/select_hunks.py` can only select whole hunks; it cannot split a hunk that carries two concerns.
So `git apply`-based extraction cannot produce "band only", then "native only", then "derived only".

**The intermediate states do exist as archive patches** (authored against older bases):

| concern | archive patch |
|---|---|
| band 64-wide K/V batches | `archive/work/rdna4-fa-band/0001-*.patch` |
| band 64-column verify block | `archive/work/rdna4-fa-band-wide/0001-*.patch` |
| native q8_0 K/V (v4) | `archive/work/arch-independent-memory/patches/0006-v4-native-q8-kv.patch` |
| native bf16 K/V (v5) | `archive/work/arch-independent-memory/patches/0007-v5-native-bf16-kv.patch` |
| derived kq-mask | `archive/work/kq-derived-tile/`, `archive/work/kq-mask-derived-ab/` |
| the whole old-15 super-block | `archive/work/block-15-campaign-wins/block-15-campaign-wins.patch` |

But they are against different bases and do not include the later amendments (last-four native arms,
prefill band split, stage arena, issue-#53, the GQA-12 cap), so replaying them will not reproduce the
tip state.

### Options

- **A — reconstruct intermediate states from the archive patches, then forward-port the amendments.**
  Faithful to Phase 1 (mostly tree-invariant). Cost: high and fragile — the patches' bases differ, and
  the later amendments would have to be re-partitioned by hand anyway.
- **B — treat the FA stack as a Phase-2 refactor now (recommended).** V2.0 already includes Phase 2 and
  Q7 says self-contained functions/files are the goal. Define target 05/06/07 by **writing** the band,
  native-KV and derived-mask logic as self-contained modules (`fattn-band.cuh`, `fattn-native-kv.cuh`,
  `fattn-derived-mask.cuh` or equivalent), leaving only call points in `fattn.cu`/`launch_fattn`. This
  skips the un-splittable squashed diff entirely and lands the desired end state. Cost: design + coding,
  and it changes the net tree (allowed by Q3/Q8); gates are the FA matrix + prefill-logit + width purity.
- **C — go backwards from the tip.** target 07 = tip; target 06 = tip minus derived; target 05 = tip
  minus derived minus native, using reverse-applies of the archive patches and hand edits. Same fragility
  as A; useful only as an aid to B.
- **D — keep the FA stack as one "flash-attention" block** (05+06+07 merged). Cheapest, but leaves the
  largest remaining super-block and contradicts the goal.

**Recommendation: B.** It is the only option that both respects the Q7/G8 direction and avoids fighting
a squashed, interwoven diff. It means the FA blocks are completed in Phase 2 rather than Phase 1.

## 4. Preliminary assessment of the other splits (to be confirmed)

- **old-06 → many:** the worst forward-deps were already separated at r50 (matcher functions). The rest
  is largely file-level (MoE cache engine, slab, loader, meta) and should be tractable. **Medium.**
- **old-08 → 08/10/11:** mostly file-level (`mmb.cu` vs `gdn-conv.cu`/`norm*.cu` vs `mmvq.cu`); the
  `ggml-cuda.cu` `try_fuse` arms interleave but each is a matcher call → Phase-2 extraction. **Medium.**
- **old-13 → 11/15/09:** engine file separable; the cache-aware `try_fuse` arms interleave with the
  fusion arms → Phase-2 extraction. **Medium/high.**
- **old-14 → 12/13:** QSA/indexer files vs model/HC/lazy files are largely separable; the shared
  `qwen4exp.cpp`/`llama-graph.cpp` hunks need care. **Medium.**
- **old-15 → many:** the non-FA parts (MoE hooks, build-time, qwen4exp policy) are similar to the above;
  the FA stack is the blocker (above). **High.**

The emerging conclusion: for the **interwoven `ggml-cuda.cu`/FA dispatch** concerns, Phase 1 and Phase 2
should be done together (extract self-contained functions/files), exactly as Q7/G8 anticipated. The
purely file-level splits can stay mechanical.

## 5. Requested decision

Confirm the approach for the flash-attention stack (targets 05/06/07):

- **B (recommended):** split them as a Phase-2 self-contained refactor rather than a hunk move.
- Or A/C if you want the intermediate states reconstructed from the archive first.
- Or D if you would rather keep the FA stack as one block for now.

Once decided, the same "self-contained extraction" pattern can be applied to the other interleaved
`ggml-cuda.cu` concerns (fusions, MoE hooks), while the file-level splits proceed mechanically.

## 6. Update — closer inspection of the FA kernel (2026-10-12, after choosing B)

Option B was chosen. Inspecting the actual kernel call chain shows why the 3-way split is a genuine
kernel-layer refactor, not a header move:

- Both concerns are threaded as **function parameters through the whole call chain**:
  `flash_attn_ext_f16_iter(..., const kq_derived_t derived, const fattn_kv_native_t kv_native)`,
  `_process_tile(..., derived, kv_native, ...)`, the `flash_attn_ext_f16` kernel, and the `load_tile`
  helpers (`fattn-mma-f16.cuh` lines ~736, ~898, ~1550, ~2207, ~2321, ~2384, ~2449). The concern
  **implementations** are already fairly self-contained (dequant chunk helpers; `kq_derived_t` +
  `flash_attn_kq_derived_blocks`), but the **interface** carries both.
- `fattn-common.cuh`'s `fattn_kernel_t` typedef and struct area carry both the derived parameters
  (`cell_pos`/`tok_lo`/`tok_hi`) and the native parameters (`kv_native_K`/`kv_native_V`), side by side
  (lines ~78-96).
- `load_tile` branches on `kv_native_type` with a `switch` calling the per-type chunk dequantizers;
  `iter`/`process_tile` branch on `derived.cell_pos != nullptr` in ~10 places; both are inline in the
  kernel bodies, not helper calls.

A clean 05/06/07 split therefore requires **layering the kernel interface across the targets**: target
05 (band) must ship the kernel *without* the `derived`/`kv_native` parameters and branches, target 06
adds the native parameter + enum/policy + load switch, target 07 adds the derived parameter + struct +
classification + branches. That is ~1-2k lines of careful, gate-sensitive churn across `fattn-mma-f16.cuh`,
`fattn-tile.cuh`, `fattn-vec.cuh`, `fattn.cu`, `fattn-common.cuh`, and it must clear the full FA matrix
(`FLASH_ATTN_EXT` 6000+), width purity, prefill-logit and same-seed gates at each intermediate.

### Refined recommendation

**Keep targets 05/06/07 as ONE "flash-attention optimizations" block for Phase 1, but still do the G8
module extraction inside it**: move the band predicates to `fattn-band.cuh`, the native enum/policy/chunk
dequantizers to `fattn-native-kv.cuh`, and `kq_derived_t` + `flash_attn_kq_derived_blocks` to
`fattn-derived-mask.cuh`, leaving the shared kernel interface and call points in `fattn-common.cuh` /
`fattn.cu`. This:

- satisfies G8 (concern bodies in self-contained files; only the shared kernel interface/call points
  remain interleaved, and the interface genuinely is shared plumbing),
- avoids the risky cross-target kernel layering,
- still removes the FA code from the old-15 super-block, and
- is independently gateable (the refactor is meant to be bit-identical, so the existing FA gates are the
  proof) before any per-concern behavioural change.

The three modules can then be promoted into separate blocks later (Phase 3) **if** the FA gates make the
layering worth it. Net target count becomes **16** (00-04, 05-fa, 06 mmb, 07 kquant, 08 fused-prefill,
09 fused-moe, 10 qwen4exp, 11 qsa, 12 moe-iface, 13 moe-cache, 14 graph-offload, 15 mmap-reclaim), or a
renumber of the same set.

**Requested:** confirm this refinement (one FA block + module extraction) or insist on the full 3-way
kernel layering (B, larger/riskier).

## 7. Update — confirmed refinement, then a second structural finding (2026-10-12)

The refinement is **confirmed**: one `05-fa` block with the band/native/derived modules inside it.

Building `05-fa` then hit a **forward-dependency**, saved on scratch branch `v2-fa-wip` (`116ff4ccf`;
`v2-phase1` is restored to the clean `00-04` tag `pin/v2-phase1-00-04` = `2a1502311`):

- The `fattn_stage` prefill arena (native-KV, old-15 hunk in `common.cuh`) calls
  `optional_alloc_within_free_cap(...)` (slab, target `moe-iface`) and `ggml_cuda_graph_mem_freed(...)`
  (graph, target `graph-offload`). Both are owned by **later** targets, so `05-fa` cannot build before
  them.
- Re-applying old-15's FA core files succeeded (110 files; the two conflicts in `fattn.cu` /
  `fattn-tile.cuh` were the band/verify additions and resolved by taking them). But the shared-file
  hunks are not cleanly FA: e.g. `common.cuh` hunk [9] that a "band" keyword matches is actually the
  **GLU->Q8_1 verify-band fusion** (target `fused-moe`), not the FA band. **Keyword triage over the
  squashed old-15 is unreliable**; only file ownership or function-level extraction is.

### Recommendation — an early "memory infrastructure" block

The root cause is that generic plumbing (the movable-boundary slab, the graph-memory generation, the
optional-allocation cap, the compute-margin helpers) is currently owned by the late MoE/graph blocks,
but the FA (and dense) paths need it much earlier. Move that generic plumbing into an **early block**
right after foundation:

| new | alias | contents | sources |
|---|---|---|---|
| 00 | `foundation` | as now | 00,05,07,09a,06-readback,15-W4 |
| 01 | `mem-infra` | movable-boundary slab + `optional_alloc_within_free_cap` + `ggml_cuda_graph_mem_freed`/graph-mem generation + `GGML_COMPUTE_BUFFER_*` helpers | split out of old-06/15 (the generic allocation/graph-memory half) |
| 02 | `allreduce` | old 12 | |
| 03 | `mtp-adaptive` | old 01 | |
| 04 | `gdn` | old 02 | |
| 05 | `bf16-kv` | old 03 | |
| 06 | `fa` | old-04 FA + old-15 band/native/derived (modules inside) | |
| 07 | `mmb` | old-08 MMB | |
| 08 | `kquant` | old 10 + old-04 MMQ + old-15 Q1_0/grid/PR57 | |
| 09 | `fused-prefill` | old-08 fusions + PR #63 | |
| 10 | `fused-moe` | old 13 + old-08 mmID + PR #114 | |
| 11 | `qwen4exp` | old-14 model/HC/lazy + old-15 HC16 | |
| 12 | `qsa` | old-14 QSA + old-15 W1-W3 | |
| 13 | `moe-iface` | the remaining (cache-specific) iface/scheduler/Meta + loader host-expert policy | old-06 remainder, 09b |
| 14 | `moe-cache` | engine + hooks + arena | old-06 engine, 13 engine, 15 hooks | |
| 15 | `mmap-reclaim` | old-06/15 r49 reclaim | |

(`graph-offload` mostly folds into `mem-infra`; its remaining cache-arm half stays in `moe-cache`.)
This removes the forward dependency, gives the FA/dense/MoE paths one infra owner, and keeps the
self-contained direction. It does renumber again, so confirm before I rebuild.

**Requested:** confirm the `mem-infra` block (or an alternative: stub the slab/graph calls in `05-fa`
and re-add them later — more layering, less desirable).

## 8. Update — mem-infra scoped; the reorg is a multi-session refactor (2026-10-12)

`mem-infra` is confirmed. Scoping it:

- The generic infra lives in **old-06** and **old-15**: `old-06`'s `ggml-cuda.cu` hunk
  `@@ -408,9 +488,906` is ~897 added lines that are almost entirely the movable-boundary slab
  (`ggml_cuda_slab_*`, `g_slabs`, `ggml_cuda_device_malloc` OOM path, `ggml_cuda_vmm_map_phys`, the
  pool-floor) with only **2** `moe_cache` references; the graph-memory generation
  (`ggml_cuda_graph_mem_gen`/`_freed`) is **old-15**; `get_compute_margin_pct`,
  `ggml_gallocr_drop_buffers` and the `GGML_COMPUTE_BUFFER_*` plumbing are `old-06`.
- **Entanglement (deeper than arming):** the slab implementation **calls the MoE cache directly** at
  ~10 sites in old-06's hunk [6] (`moe_cache_validate`, `moe_cache_shrink_step`,
  `moe_cache_release_arena`, `moe_cache_evict_slab_range`, `moe_cache_floor_active`,
  `moe_cache_disable_streaming`, `moe_cache_is_sized`), so the slab **cannot compile without the cache
  engine**. A standalone `mem-infra` before FA is therefore not viable as-is. (Tried it: the selected
  11-hunk patch applies cleanly but does not link without `moe-expert-cache.cu`.)

The pattern is now clear and consistent across the hard blocks: the delivery's 50 revisions have
interleaved generic infrastructure with feature policy at the function level (slab↔MoE arming,
FA kernels band↔native↔derived, `try_fuse` arms). **A fundamental orthogonal reorg of this set is a
code refactor (Phase 2 fused with Phase 1), not a commit-boundary redistribution.** It is a multi-session
effort that needs a fresh build + the relevant gates at each intermediate step.

### Staged execution runbook (for the next session)

1. **Checkpoint** — `v2-phase1` at `pin/v2-phase1-00-04` (`2a1502311`): blocks 00-04 rebuilt and clean.
2. **`01 mem-infra`** — extract the generic infra as a self-contained module set:
   `slab.cu`/`slab.cuh` (slab + vmm + device_malloc OOM + optional-alloc cap), `graph-mem` helpers in
   `common.cuh`, `ggml-alloc.c` `drop_buffers`/compute-margin. Arm the slab from the env in the dense
   path; `moe-cache` later narrows it. Build + `GGML_CUDA_SLAB=0`/`--fit off`/dense coherence gates.
3. **Rebase 02-05** (allreduce, mtp, gdn, bf16-kv) onto `mem-infra` (cherry-pick; expected conflict only
   in `ggml-cuda.cu`/`common.cuh`).
4. **`06 fa`** — the three modules (`fattn-band.cuh`, `fattn-native-kv.cuh`, `fattn-derived-mask.cuh`) +
   the stage arena (now that slab/graph-mem exist). Full FA gates + prefill-logit + width purity.
5. **07 mmb, 08 kquant, 09 fused-prefill, 10 fused-moe** — file-level splits with `try_fuse`/dispatch
   arms extracted to self-contained functions; adjacent (mmvq dispatch) done together.
6. **11 qwen4exp, 12 qsa** — model/HC/lazy vs QSA/indexer; the HC matchers go to `qwen4exp`.
7. **13 moe-iface, 14 moe-cache, 15 mmap-reclaim** — the remaining cache-specific iface/scheduler/Meta,
   the engine/hooks/arena, and the reclaim.
8. At every step: `scripts/gate-per-block-build.sh` (fresh `all`), then the release gates; the final tree
   is the new V2.0 baseline (`release.json` via `make-release.sh`).

### Recommended correction: keep the `fattn_stage` arena OUT of the FA block

The only reason FA needs the slab is the `fattn_stage` prefill arena (a **memory optimisation** added for
issue #30), not the native-KV feature itself. The native read works with the pre-arena mechanism (staging
into the compute-graph reserve). Therefore:

- **`06 fa`** ships band + native-KV + derived-mask **without** the `fattn_stage` arena; prefill stages
  into the reserved scratch (the pre-arena behaviour). No slab/graph dependency.
- The `fattn_stage` arena + slab + cache land later in the memory/MoE blocks (`moe-cache` /
  `mmap-reclaim`), where the slab already lives.
- **`mem-infra` is then unnecessary** (or shrinks to the small self-contained `ggml_cuda_graph_mem_gen`
  helper, if FA wants it).

This avoids the slab↔cache entanglement entirely and is far less work than either merging the whole
host-MoE stack early or refactoring the slab behind an interface. It is another small, deliberate code
layer (the arena appears in a later block), which is exactly the Phase-2 pattern.

### Final approved order (2026-10-12, maintainer)

The maintainer approved the corrected plan and grouped **Slab + MoE together, ahead of qwen4exp**. With
`mem-infra` dropped, the arena deferred out of FA, and graph-offload folded into the MoE block, the
final V2.0 set is **14 blocks** (this supersedes the 16/18-block tables in `README.md` / `TARGET-BLOCKS.md`;
those will be rewritten when the chain is rebuilt):

| # | alias | contents | sources |
|---|---|---|---|
| 00 | `foundation` | band-uniformity, Meta-wrapper, allocator W4, logits readback | old 00,05,07,09a,06-readback,15-W4 |
| 01 | `allreduce` | hybrid HIP all-reduce | old 12 |
| 02 | `mtp-adaptive` | adaptive MTP | old 01 |
| 03 | `gdn` | chunked gated-delta-net | old 02 |
| 04 | `bf16-kv` | BF16 KV cache + native-BF16 FA | old 03 |
| 05 | `fa` | band + native-KV + derived-mask (arena deferred) | old-04 FA, old-15 FA |
| 06 | `mmb` | MMB quantized prefill GEMM | old-08 MMB |
| 07 | `kquant` | mmvq VDR, Q6_K MMQ, Q1_0, grid pad | old 10, old-04 MMQ, old-15 Q1_0/grid/PR57 |
| 08 | `fused-prefill` | conv1d/RMS/top-k MoE/verify fusions | old-08 fusions, old-15 PR63 |
| 09 | `fused-moe` | gate+up+GLU MMQ, item-split, shared-expert, MWR | old 13, old-08 mmID, old-15 PR114 |
| 10 | `slab-moe` | host-resident MoE (iface+scheduler+Meta+slab+cache engine+hooks) + `fattn_stage` arena + graph-offload | old-06 MoE, old-09b, old-11, old-13 engine, old-15 hooks/arena/r48, old-06 async |
| 11 | `qwen4exp` | model/HC/lazy/MTP head + qwen4exp policy | old-14 model/HC, old-15 HC16/policy |
| 12 | `qsa` | QSA/indexer + QSA memory wins | old-14 QSA, old-15 W1-W3 |
| 13 | `mmap-reclaim` | host memory reclaim | old-06/15 r49 |

Dependency check: 05 `fa` needs no slab (arena deferred to 10); 09 `fused-moe` precedes 10 so the cache
hooks can reference the fusions; 10 `slab-moe` precedes 11 `qwen4exp`, which consumes it; 12 `qsa` after
`qwen4exp`; 13 last. The getenv caching is folded into each owner.

### Honest status

The **audit is complete** and the **method is proven** (00-04 rebuilt, tooling written, three structural
findings recorded with decisions). The remaining execution is a genuine refactor of the hub files; I
recommend it be done as a focused multi-step effort with builds between steps, not squeezed into the tail
of this session, where a broken intermediate would be hard to verify.

---

## 9. Block 05 `fa` rebuilt and gated (2026-10-12, execution session 2)

Target **05 fa** is committed on `v2-phase1` as **`05e88da52`** (checkpoint parent
`pin/v2-phase1-00-04` = `2a1502311`; base `097cf33d9`).  Per-block fresh `all` build: **6/6**.
The delivery `patches/` and `release.json` are untouched.

### What actually moved into 05 (HANDOFF §5's list was incomplete)

- **FA-owned kernels** (reconstructed at the reference tip, minus the arena):
  `fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.{cu,cuh}`, `fattn-vec.cuh`, `fattn.cu`, and all
  **286** `template-instances/fattn-*` (the per-type TILE/MMA/VEC axis; `generate_cu_files.py` follows).
- **V3 derived kq-mask** (a *graph/infrastructure* concern, not just the kernel):
  `ggml.h`/`ggml.c` (`ggml_flash_attn_ext_add_kq_derived`), `llama-cparams.h` (`kq_mask_derived`),
  `llama-kv-cache.{cpp,h}` (`kq_mask_derivable` / `set_input_kq_derived`),
  `llama-graph.{cpp,h}` (`llm_graph_input_kq_derived`, `build_attn_inp_kq_mask`, the
  `build_attn_inp_*_impl` method refactor, the `kq_derived` map, the **null-mask guards**),
  `llama-context.{cpp,h}` (the CUDA/HIP probe + `kq_mask_packed_reachable` +
  `graph_reserve(packed_kq_mask)` reserve sizing), and the **CPU oracle** in
  `ggml-cpu/ops.cpp` (`kq_derived_mask_value`).
- **Band** (old-04 FA): the band/`ncols2=8` additions in `fattn.cu`/`fattn-mma-f16.cuh`, the
  `ggml_set_fa_tensor_parallel`/`ggml_get_fa_tensor_parallel` hint, the `src/llama-context.cpp` hint,
  and the `GGML_CUDA_OP_TIMING` diagnostic (see finding 2).
- **FA quants enablement** (old-08, beyond its `fattn*` files): `ggml/CMakeLists.txt` +
  `ggml/cmake/common.cmake` (`GGML_CUDA_FA_QUANTS` full type list), `convert.cu` + `dequantize.cuh`
  (the `iq4_nl` `to_*_nc_cuda` arms and `dequantize_q4_nl`), `docs/build.md`, and the
  `ggml-hip/CMakeLists.txt` `-Wno-pass-failed` build-time fix.

### Arena deferred to 10 (as approved)

`launch_fattn` ships the **pre-arena** behaviour: a native-capable operand is read natively at every
width and a non-native operand is converted into the node's reserved scratch
(`ggml_cuda_flash_attn_ext_get_f16_extra_data`); `ggml_cuda_flash_attn_ext_get_alloc_size` keeps its tip
arm.  The prefill band split + the `fattn_stage` arena + `GGML_CUDA_FA_STAGE_MAX_MB` land with
`10 slab-moe`.  This is the actual git-recorded pre-arena state (the arena patch's `-` side), not the
node-scratch intermediate measured in the issue-30 session.

### Gates (block 05) — all green

| gate | result |
|---|---|
| per-block fresh `all` build (`gate-per-block-build.sh`) | **6/6** |
| `test-backend-ops -o FLASH_ATTN_EXT` (default) | **6389/6389** |
| `test-backend-ops -o FLASH_ATTN_EXT` `GGML_CUDA_FA_KV_NATIVE=0` (staged arm) | **6389/6389** |
| `test-backend-ops -o FLASH_ATTN_EXT` `GGML_HIP_FA_BAND_WMMA=0` | **6389/6389** |
| `gate-prefill-logits.sh` | mean KLD **0.000586**, same-top-p **98.725 %** (lim 0.005 / 98.0 %) PASS |
| same-seed coherence vs r49 known-good `f0a74f4cc` (`-sm tensor` 3-GPU and 1-GPU) | greedy text identical |

Two defects found and fixed *inside* 05 by these gates (both invisible to the build gate): the missing
CPU oracle made all 13 `derived=1` cases fail, and the missing null-mask guard crashed the first decode
(`GGML_ASSERT(buffer)` in `set_input_kq_mask`).

### Findings for the maintainer

1. **HANDOFF §5's file list understates 05.**  The derived kq-mask is a graph/context/cparams/kv-cache
   concern with a CPU oracle, and the FA-quants enablement is CMake + `convert.cu`/`dequantize.cuh` +
   docs — neither is in the §5 list.  Recorded here so the doc pass and the later blocks know.
2. **`GGML_CUDA_OP_TIMING`** (`ggml/src/ggml-cuda/ggml-cuda.cu`) is old-04's only content in that file
   and is a *generic* per-op timing diagnostic, not FA.  It is in 05 per HANDOFF §5.1; **decision
   wanted**: move it to `00 foundation` (amend the 00 commit) or leave it.  It does not change the plan.
3. **`kq_mask_packed_reachable` / `graph_reserve(packed_kq_mask)`** is introduced by old-14 in the
   canonical chain, but it is a V3-derived-reserve concern; it now lives in 05 and is reused by qwen4exp
   (11).  No action needed unless the maintainer wants it in 11.
4. **Unmapped old-15 stragglers** left out of 05 (they are not FA): `inp->s_copy`'s
   `ggml_set_output` (#124 GDN state gather — `03 gdn` or `10`) and old-04's MUL_MAT Q6_K/Q8_0 test
   perf shapes (kquant, `07`).  Both are recorded so they are not lost.
5. **`test-backend-ops` grew 6382 → 6389** (block 04's `logit_softcap` head-gate relaxation adds 7
   cases); the FA matrix baseline for the next blocks is **6389**.

---

## 10. Block 06 `mmb` rebuilt (2026-10-12, execution session 2)

Target **06 mmb** is committed on `v2-phase1` as **`298060c73`**.  Per-block fresh `all` build:
**7/7**.

### What moved into 06 (old-08's MMB half)

- `ggml/src/ggml-cuda/mmb.{cu,cuh}` — the **old-08** state of the file (not the block-15 tip, which
  extends it with the bf16-activation cache / HC slots / marks that belong to later blocks).
- `ggml-cuda.cu`: the `mmb.cuh` include, `ggml_cuda_mmb_release_all()` in the destructor,
  `ggml_cuda_mmb_begin_graph()` in `graph_compute`, the `ggml_cuda_mul_mat` /
  `ggml_cuda_mul_mat_id` MMB dispatch guards and the fused MoE gate+up+swiglu `try_fuse` arm.
- `tests/test-logits-width-probe.cpp` + `tests/CMakeLists.txt`.
- The **MMVF/MMF verify-batch dispatch uniformity** (`ne11_mmvf`) for `n_q <= 8` (bundled in
  old-08's `ggml-cuda.cu`).

### Gate result and the gate-boundary finding

- `test-backend-ops -o MUL_MAT` — **2444/2444**.  MMB is confirmed **prefill-only**: on both a dense
  4B and the 35B-A3B Q4_K_M, MMB on/off produce identical *decode/verify* behaviour (the 4B has no MMB
  shape; on the 35B the routed path is taken).
- `test-logits-width-probe` — **`width_purity` FAILS at 06, and this is expected**: the W=1 vs W>=2
  difference is present with `GGML_CUDA_MMB=0` too, and the full delivery (`~/llama.cpp` r49 build)
  passes with the same probe.  **The `width_purity` check is a `07 kquant` gate** (band-uniform mmvq
  `nwarps`/VDR + the multi-row verify blocks), not a 06 one; 06 only proves MMB adds no width
  dependence.  Recorded as a G3 *gate*-only dependency (the per-block **build** is unaffected).
- `test-backend-ops -o MUL_MAT_ID` — 4 `f16 / dup=1` cases fail and the run ends in
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, **identically with `GGML_CUDA_MMB=0`** and fixed in the
  full delivery (r49: 931/931).  Not a 06 regression; the MUL_MAT_ID dispatch is owned by the fused-MoE
  blocks (09/13).  Flagged so it is not mistaken for an MMB defect when 09/13 land.

### To carry forward

- **Re-run `test-logits-width-probe` at `07 kquant`; it must PASS there** (that is the checkpoint that
  closes the 06 gate dependency).
- The `GGML_CUDA_OP_TIMING` placement question from §9 finding 2 is still open.

---

## 11. Block 07 `kquant` rebuilt — with a plan-affecting finding (2026-10-12)

Target **07 kquant** is committed on `v2-phase1` as **`95d15bad9`** (blocks 00–07).  Per-block fresh
`all` build: **8/8**; `test-backend-ops -o MUL_MAT` **2444/2444**.

### What moved into 07

- **old-04 MMQ**: `GGML_CUDA_MMQ_J_MAX` in `ggml_cuda_mul_mat_q` + the `mmq-vec-dot.cuh`
  `x_scale`/`x_df` register hoist.
- **old-10 (whole)**: per-type mmvq VDR/nwarps (Q4_K/Q5_K/Q6_K/Q8_0), dense Q6_K `VDR=2`, the
  MoE-expert VDR split, `vecdotq.cuh` + `ggml-cuda.cu` + tests.
- **old-15 kquant bits**: the Q1_0 `ggml_cuda_byte_perm` / `apply_ksigns` port (`vecdotq.cuh`,
  `mmq-load-tiles.cuh`), the gfx1201 grid-stall pad (`common.cuh` + `mmvf.cu`), the `mmvf.cu`
  multi-token cross-warp reduction and the `mmq-vec-dot.cuh` `#pragma unroll 1` (gfx12 VGPR spills).
- old-04's MUL_MAT Q6_K/Q8_0 test shapes (left out of 05).

### FINDING — the mmvq multi-row/row-loop hunks are not hunk-separable from fused-MoE

The `mmvq.cu` side of old-15's PR #57 (`GGML_MMVQ_RDNA4_DECODE_GRID`, the `nrows_loop` row loop, the
`lf_nval` recursive-halving reduction and the two-rows-per-block sweep) is **interleaved inside the same
functions as the fused-MoE item-split/ksplit machinery** (`mul_mat_vec_q_ksplit`,
`mul_mat_vec_q_switch_ncols_dst`, `mul_mat_vec_q_moe`).  A `git apply -3` of those hunks onto the 06
tree conflicts at 6 sites (the merge pulls the fused-MoE code in), because old-15's `mmvq.cu` base
already contains old-13/old-08's item-split.  **They are deferred to `09 fused-moe`**, which owns the
mmvq item-split and already needs the same file.

**Consequence:** `test-logits-width-probe`'s `width_purity` will PASS at **09**, not 07.  The 07 build
is otherwise clean (MUL_MAT 2444/2444; the grid pad and old-10 VDR are in).  This is the clearest
Phase-2 extraction candidate so far: the mmvq dispatch (`calc_nwarps` / `calc_rows_per_block*` /
row-loop selection) should become a feature-local `mmvq-rdna.cuh` so the kquant and fused-MoE concerns
stop sharing a function.

**Decision requested (maintainer):**
- **A (what was done):** keep the deferral — 07 ships the VDR/Q1_0/MMQ/grid parts, 09 ships the mmvq
  row-loop/`lf_nval` and the width-probe PASS.
- **B:** make 07 absorb the mmvq hunks now via a Phase-2 extraction (a day-scale refactor of
  `mmvq.cu`), keeping the width-probe gate on 07.
- **C:** something else.

---

## 12. Block 08 `fused-prefill` — BLOCKED on a second entanglement (2026-10-12)

The old-08 fusion set was assembled on a scratch branch **`v2-prefill-wip` = `17ce09fa7`**
(`gdn-conv`, `ple-conv`, `norm-gated`, `topk-moe`, `norm`, `unary`, `getrows`, `argsort`,
`ggml-cpu/ops.cpp`, `common.cuh`, the `ggml-cuda.cu` `try_fuse` fusion arms).  `v2-phase1` is left at
**`95d15bad9`** (block 07) — clean.

### Finding

With the old-08 fusions applied, the **fusion-on greedy text is wrong**: on the 4B Q8_0 (single GPU,
`-n 20 --seed 42 --temp 0`) the default build emits `[Start thinking] Thinking`, while
`GGML_CUDA_DISABLE_FUSION=1` emits `[Start thinking] Thinking Process:\n1. **Analyze the Request:** ...`
— and the fusion-off text matches the r49 known-good (`f0a74f4cc`) **both** with its fusions on and
off.  So the delivery's fusions are bit-identical, and the block-08 assembly is incomplete/faulty.

The missing pieces are **old-15's PR #63 corrections**: the `rms_norm_q8_1_f32` refactor
(`norm.cu`), `ggml_cuda_op_add_rms_norm_q8_1` / `ggml_cuda_op_rms_norm_q8_1` (`norm.cu` +
`ggml-cuda.cu` `try_fuse` arms) and the `ssm_gate_beta` ncols arm (**`mmvq.cu`**).  The old-08
norm→Q8_1 activation fold is only correct with that correction.  Since the `ssm_gate_beta` half lives
in `mmvq.cu` — the file already deferred to 09 — **08 and 09 are interdependent**, exactly like the
07/09 mmvq split in §11.

### Decision requested (maintainer)

- **A:** merge `08 fused-prefill` + `09 fused-moe` into one **fusions** block (old-08 fusions + PR #63 +
  old-13/old-08 mmvq + MWR).  This is the honest unit of the current code: the fusions and the mmvq
  dispatch share `ggml-cuda.cu` `try_fuse`, `mmvq.cu`, `norm.cu`, `common.cuh` and the
  `ggml_cuda_mm_fusion_args` plumbing.
- **B:** do the Phase-2 extraction first (feature-local `*_try_fuse` matchers + `mmvq-rdna.cuh`), then
  keep the planned 08/09 split.
- **C:** leave 08 broken and continue to 09–13, fixing the fusions at the end (the final tree is what
  matters; the per-block gates for 08 would be red).

`v2-phase1` is clean at block 07; nothing is half-applied.  The scratch branch preserves the 08 attempt.
