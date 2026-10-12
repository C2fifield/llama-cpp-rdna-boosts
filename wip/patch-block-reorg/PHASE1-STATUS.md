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
