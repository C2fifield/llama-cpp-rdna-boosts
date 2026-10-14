# `moe-topk-fusion` — the fused router already fires at prefill widths on r55 (TODO item 5(a))

**Status: RESOLVED — no delivery change (2026-10-13), archived as `archive/work/moe-topk-fusion/`.**
The premise of this campaign (the address-overlap guard refuses the fused `topk_moe` router at prefill
widths, so a ~188 MB logits pin would unlock ~0.5 % prefill) **does not reproduce on the current
delivery r55**, on gfx1201 (3x R9700) **or gfx1151** (Strix Halo, the discovery's own device): the
fusion already fires at every prefill width tested and the pin is a no-op. The ~0.5-1.2 % is already in
the tree. See [`RESOLUTION.md`](RESOLUTION.md); nothing was promoted.

> **Cold start.** This supersedes the 2026-10-11 seeding, which was copied from a 2026-09-06 discovery
> doc and carried two stale claims: the numerics fork (retired 2026-09-13, §31) and the alias refusal
> (does not reproduce, §1). Evidence below; diagnostic patch in
> [`topk-moe-diagnostics.patch`](topk-moe-diagnostics.patch).

## 0. The original item

The fused `topk_moe` router (block 08, `ggml/src/ggml-cuda/topk-moe.cu`, selected by
`ggml_cuda_topk_moe_fusion()` in `ggml-cuda-try-fuse.cu`) replaces
`soft_max -> argsort -> get_rows -> sum/clamp/div` with a partial top-k. In the 2026-09-06 discovery
the gallocr placed the router norm output (`ffn_moe_weights_norm`, tens of KB) into the just-freed
`ffn_moe_logits` buffer (`[512 x n_tokens x 4 B]`), and the guard
`ggml_cuda_check_fusion_memory_ranges(..., is_topk_moe=true)` **correctly** refused to fuse for
`nrows > TOPK_MOE_ROWS_PER_BLOCK` (`= 8`) because a multi-block read/write race would result. qwen4exp
then fell back to a full 512-expert argsort per token (~0.5 % prefill on gfx1151). Pinning the logits
(`ggml_set_output`) keeps their buffer alive (`ggml-alloc.c:749`), removes the alias, and lets the
fusion fire.

## 1. What r55 actually does (the finding)

Instrumented r55 (3x R9700 / gfx1201 / ROCm 7.14.1, `--log-verbosity 2`) with a one-line trace at the
`ggml_cuda_op_topk_moe()` call and a match/guard trace in the fusion dispatcher
(`topk-moe-diagnostics.patch`, reverted from `~/llama.cpp` after use):

| model | split | ubatch | fusion fires at (max n_tokens) | refusals (`mem=0` on `nrows>8`) | pin effect |
|---|---|---:|---:|---|---|
| Flash-Next GSQ-IQ3_XXS | `tensor` | 512 | 512 | none | none |
| Flash-Next GSQ-IQ3_XXS | `layer` | 512 | 512 | none | none |
| Flash-Next GSQ-IQ3_XXS | `tensor` | 2048 | **2048** | none | none |
| Flash-Next GSQ-IQ3_XXS (margin `GGML_COMPUTE_BUFFER_MARGIN_PCT=0`) | `tensor` | 2048 | 2048 | none | none |
| Flash-Next IQ4_NL (the discovery's quant, `-ncmoe 12` to fit) | `tensor` | 512 | 280 | none | n/a |

Key observations:

- **The fusion fires, and there are no guard refusals.** The fusion returns before the diagnostic block
  after it can run, so a refusal would print `TOPK_MOE_DBG b1 ... mem=0`; there were **zero** such
  lines at any width. The guard passes at `nrows = 2048` — i.e. **no logits/norm-output address
  overlap exists** on r55.
- **The pin changes nothing.** With `LLAMA_MOE_TOPK_PIN_LOGITS=1` the fired-width set is byte-identical
  to the pin-off run (verified for `tensor` and `layer`, ubatch 512 and 2048). `-v` confirms the pin
  *is* applied (`probs_in=(nil)`, `ggml_set_output(logits)` reached).
- **The win is already banked.** Forcing the fusion off (`GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1`) costs
  prefill: pp2048 on GSQ-IQ3_XXS is **1575.7 vs 1563.8 t/s** (fusion on vs off, `-r 3`), i.e. ~+0.8 %
  — the same order as the discovery's 0.5 %. Since the fusion fires by default, that is in the tree.
- The earlier ubatch-512 runs are not a confound: the ubatch-2048 run reaches `nrows = 2048`, well past
  the `8`-row alias threshold, and still fires.

## 2. The numerics half is also settled

The 2026-09-06 discovery recorded a real numerics fork (fusion on moved the qwen4exp top1 logit
`18.424 -> 18.690`). That was fixed a week later — `WORKLOG.md` 2026-09-13 item 19,
`GREEDY-PURITY.md` §31, block-08 amendment (seventh): the fused router now reproduces the generic
softmax reduction order, the `sum_rows` order with a true divide, and the argsort index tie-break. Its
validation used a temporary force-fuse probe (`GGML_CUDA_TOPK_MOE_IGNORE_ALIAS`) precisely to cover the
call sites the address guard skips. So even if the alias *were* present, the pin is no longer a
correctness question.

## 3. Why the pin is a no-op here (and what that implies)

`ggml_set_output` only prevents `ggml_gallocr_free_node()` from freeing a *live* output; it does not
change where an unrelated tensor is placed. The pin is therefore useful **only if the logits buffer is
actually reused for the norm output** — which on r55 it is not. The mechanism is unchanged (still
`TOPK_MOE_ROWS_PER_BLOCK = 8`, still the same `is_topk_moe` exemption); the **allocation plan** is what
differs from the r48-era discovery. Determining exactly which r51-r55 change (V2.0 re-org, the r20
compute margin, a graph reorder) removed the reuse is not necessary for the decision — the fusion
already fires, so there is nothing to unlock.

## 4. Caveat — resolved: gfx1151 checks out too

The discovery was made on **gfx1151 (Strix Halo), single GPU**. That device was checked on 2026-10-13
(Flash-Next IQ4_NL — the discovery's own quant — `-b 2048 -ub 2048 -c 16384`): the fusion fires at
`n_tokens = 1, 2, 4, 42, 1160, 2044, 2048`, there are **zero** guard refusals, and the pin produces an
identical fired set. Prefill pp2048 is **1030.1 t/s** with the fusion vs **1018.0** with it disabled —
the win is already in the tree there as well. See [`RESOLUTION.md`](RESOLUTION.md).

## 5. Recommendation — adopted (closed, no release)

Close `moe-topk-fusion` as **already delivered** (done 2026-10-13): the fused router fires at all
prefill widths on r55 (gfx1201 **and** gfx1151), the ~0.5-1.2 % is in the tree, and the ~188 MB pin
adds nothing. No `patches/` change, no new env knob, no `GGML_CUDA_TOPK_MOE_*` delivery surface, no
release. The diagnostic patch is kept here for reproducibility; nothing from it was promoted.

## 6. Pointers

- [`topk-moe-diagnostics.patch`](topk-moe-diagnostics.patch) — the temporary trace + env-gated pin used
  for this investigation (against the r55 tree; **not** part of the delivery, reverted from the fork).
- `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` (matcher + `ggml_cuda_check_fusion_memory_ranges` and its
  `is_topk_moe` exemption), `ggml/src/ggml-cuda/topk-moe.cu` / `topk-moe.cuh`
  (`TOPK_MOE_ROWS_PER_BLOCK = 8`), `src/llama-graph.cpp` (`llm_graph_context::build_moe_ffn`,
  `ffn_moe_logits`).
- `GREEDY-PURITY.md` §30/§31 (address-selected fusions must be bit-transparent); `WORKLOG.md`
  2026-09-13 item 19.
- [`archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`](../../archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md)
  — the original kernel-count ledger and alias analysis.
