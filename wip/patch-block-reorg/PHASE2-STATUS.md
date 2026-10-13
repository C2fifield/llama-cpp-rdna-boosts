# Phase 2 — execution status (self-contained refactor, G8)

Companion to `PHASE2-REFACTOR.md` (the work plan) and `HANDOFF-PHASE2.md` (the cold start).
Phase 1 is complete at the checkpoint tag `v2-phase1` = `a40841e1e` (12 blocks, tree == the
published delivery).  Phase 2 works on top of it on branch **`v2-phase2`**.

## 0. Refs (current)

| ref | value |
|---|---|
| Phase-1 checkpoint | `v2-phase1` = `a40841e1e` (12 blocks, tree == `af0595360`) |
| Phase-2 branch | `v2-phase2` = **`d8ad56737`** (block 11 rebased) |
| block 10 (amended) | **`1fa36b3ea`** |
| delivery repo | `~/llama-cpp-rdna-boosts` (`main`) |

## 1. Method (as executed)

- **One extraction per commit, inside the owning block.**  The extraction is applied to the owning
  block's commit (`git commit --amend`) and the later block(s) are rebased (`git rebase --onto`).
  Block-10-owned arms are the cheapest: only block 11 has to be rebased, and block 11 touches none of
  the concern files, so it replays clean.  An amend cascades the block-10 hash; the rebased block 11
  gets a new hash too.
- **Validate before re-homing.**  The extraction is first made on the tip, built fresh
  (`~/bin/build-llama-rocm-714`, `build-phase2`), gated, and only then applied to the owning block.
  The re-homed tree equals the validated tip tree, so the gate carries over.
- **Gates per extraction:** the feature's own gate, a same-seed/byte-identity check, the full per-block
  fresh `all` build (`gate-per-block-build.sh`, 12/12), and — because the extracted arm lives in a
  prefill fusion path — `gate-prefill-logits.sh`.
- **Nothing in `patches/` / `release.json` is touched.**  The Phase-2 tree is the new baseline only
  after V2.0's final gates; earlier work is recoverable at `v2-phase1`.

## 2. Extraction 1 — HC_COMBINE → HC_MIX arm → `hc-mix.cu` (owner block 10)

`ggml_cuda_try_fuse`'s `HC_COMBINE -> BF16 HC_MIX` arm (matcher call + op dispatch) moved into
`ggml_cuda_hc_combine_mix_try_fuse()` in `hc-mix.cu` (declared in `hc-mix.cuh`).  `try_fuse` now has a
single guarded call.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| `test-backend-ops -o HC_MIX` | **30/30** |
| qwen4exp `Qwen3.8-Flash-Next-Q4_K_M` `-sm tensor` dense + sparse vs r50 | **identical** |
| `GGML_CUDA_FUSE_HC_COMBINE_MIX=0` vs default (dense) | **identical** (bit-identical fusion) |
| `GGML_CUDA_DISABLE_FUSION=1` vs r50 `GGML_CUDA_DISABLE_FUSION=1` | **identical** (pre-existing non-bit-identical fusions, not this arm) |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 3. Extraction 2 — indexer relu-sum matcher+arm → `indexer-score.cu` (owner block 10)

`ggml_cuda_match_idx_relu_sum` (~83 lines, previously a `static` in `ggml-cuda.cu`) moved to
`indexer-score.cu` (now non-static, declared in `indexer-score.cuh`; the graph-optimizer call site in
`ggml-cuda.cu` keeps working off the header).  The `try_fuse` arm became
`ggml_cuda_indexer_relu_sum_try_fuse()` in `indexer-score.cu` (it carries the `RDNA3_5` cc guard), so
`try_fuse` is a single guarded call.

Note: the arm is `RDNA3_5`-only, so on this gfx1201 box it is inert — the gate is the build plus the
unchanged-path coherence.  The move is a pure relocation.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| qwen4exp dense + sparse vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 4. Extraction 3 — HC_MIX arm + matchers → `hyperconn.cu` (owner block 10)

`ggml_cuda_match_hc_mix` + `ggml_cuda_hc_mix_closed` (previously `static` in `ggml-cuda.cu`) and the
`hc_mix_on` prefill arm moved into `hyperconn.cu` (`ggml_cuda_hc_mix_try_fuse()`; the kill-switch
`GGML_CUDA_DISABLE_HC_FUSION`/`GGML_CUDA_DISABLE_HC_MIX` semantics are reproduced exactly).  The
matchers are declared in `hyperconn.cuh` (the gate-mix arm in `ggml-cuda.cu` still calls
`ggml_cuda_hc_mix_closed`).  `try_fuse`'s `hc_mix_on` block is now a single guarded call; the
`hc_comb_on` block follows it in the same order.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| `test-backend-ops -o HC_MIX` | **30/30** |
| qwen4exp dense + sparse vs r50 | **identical** |
| `GGML_CUDA_DISABLE_HC_MIX=1` vs default (dense) | **identical** |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 5. Extraction 4 — sigmoid-mul-add arm → `unary.cu` (owner block 10)

The `UNARY(sigmoid) -> MUL -> ADD` per-row-gate arm moved into
`ggml_cuda_sigmoid_mul_add_try_fuse()` in `unary.cu` (declared in `unary.cuh`); the matcher and op
were already there.  `try_fuse` is a single guarded call.

| gate | result |
|---|---|
| fresh `all` build (tip) | OK |
| qwen4exp dense + sparse vs r50 | **identical** |
| `GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0` vs default (dense) | **identical** |
| 4B Q8_0 same-seed greedy vs r50 | **identical** |
| per-block fresh `all` build | **12/12** |
| `gate-prefill-logits.sh` | mean KLD **0.000707**, same-top-p **98.755 %** PASS |

## 6. Next (remaining `try_fuse` arms, highest value next)

The big remaining interleaves, in rough value order:

- **hc-comb cluster** (`hc_comb_on`: the REPEAT-anchored combine+norm matcher and the SCALE A-chain,
  plus `ggml_cuda_hc_combine_norm_identify` and the `set_bf16`/`set_bf16_src1` helpers) →
  `hyperconn.cu`.  Owner block 10, the largest remaining piece (~480 lines), but it needs two
  `ggml-cuda.cu` statics hoisted first: `ggml_cuda_is_view_or_noop` and `ggml_cuda_bf16_src1_cache_on`
  (both used widely in `ggml-cuda.cu`), and `hyperconn.cu` must include `mmb.cuh`.  Deferred to a
  focused step.
- **hc gate-mix arm** → `hyperconn.cu` (needs `mmb.cuh` for `ggml_cuda_hc_gate_mix` /
  `ggml_cuda_mmb_gatemix`).
- **conv1d** (`ple_conv`, `gdn_conv`) → `ple-conv.cu` / `gdn-conv.cu`.  Owner block 08 (needs rebasing
  09/10/11; expect `ggml-cuda.cu` conflicts).
- **cpy_batch** → `cpy-batch.cu`.  Owner block 09 (rebase 10/11).
- **norm-gated / norm_rows** → `norm-gated.cu` (uses `ggml_cuda_compute_forward`: needs a hoist).
- **shared-expert / topk-moe / MWR / mmb / q_pair / shexp / weighted-down** matcher relocations.

Then the maintainer's `ggml-cuda.cu` split (`ggml-cuda-try-fuse.cu`, `-slab.cu`,
`-moe-cache-hooks.cu`, `-arena.cu`).

## 7. Process decision and checkpoint

**Resolved (maintainer, 2026-10-12):** amend-owning-block + rebase is correct; hoisting more per change
(batching a whole concern rather than one tiny arm) is encouraged.  The earlier `git diff <commit>
<delivery-tip>` empty-check does not apply after Phase 2 changes the tree; the correct check is that the
tree delta is exactly the intended extraction.

**Checkpoint 1** (`checkpoints/v2-phase2-ckpt1.bundle`, tag `pin/v2-phase2-ckpt1`): `v2-phase2` =
`d8ad56737`, block 10 = `1fa36b3ea`.  Restore instructions in `checkpoints/README.md`.

## 8. Scale of the remaining work

The `try_fuse` split is the bulk.  `ggml_cuda_try_fuse` is **2212 lines** now (upstream base: 792), and
the delivery matchers *before* it (`topk_moe`, `MWR`, `shared_expert`, the `hc_combine_norm` helpers)
add ~700 more lines in `ggml-cuda.cu` (11307 lines total).  Roughly **25+ delivery arms / ~2300 lines**
remain inside `try_fuse`, plus: `mmvq.cu` dispatch (3257 lines; ~700 lines of `calc_nwarps`/
`calc_rows_per_block`/item-split/ksplit), the FA chooser (`fattn.cu` 919 + the MMA/tile kernels), the
`llama-context.cpp`/`llama-model-loader.cpp` glue, and finally the maintainer's `ggml-cuda.cu` split
(`-try-fuse.cu`, `-slab.cu`, `-moe-cache-hooks.cu`, `-arena.cu`).  The MMQ `J` selection of plan item 2
is **already** a self-contained `ggml_cuda_mmq_select_J` (3 call sites) — largely done.

Batching plan (more per change, per the maintainer): one commit per concern cluster, not per arm —
(a) conv1d + norm arms (block 08); (b) MoE/MMVQ decode cluster: cpy_batch, glu_q8_1, shared_expert,
weighted-down, q_pair (blocks 08/09/10); (c) the ssm/gdn cluster: ssm_conv_in, ssm_prescan, gdn_gate,
gdn_beta_sigmoid, gdn/ssm cpy, gate_beta_verify, l2_norm_pair (block 08/10); (d) the MWR matcher;
(e) the hc_comb cluster (needs `ggml_cuda_is_view_or_noop` / `ggml_cuda_bf16_src1_cache_on` hoists);
(f) the remaining upstream-shared arms.  Each batch still gets a fresh per-block build N/N + its
gate(s).
