# `moe-topk-fusion` — adopt the fused `topk_moe` (quality gate first) (TODO item 5(a))

**Status: PARKED (seeded 2026-10-11).** Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the parked TODO
item *"MoE topk fusion adoption (item 5(a))"*.

> **Cold start.** Read this file, then
> [`archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`](../../archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md)
> (the discovery + numerics fork), and `GREEDY-PURITY.md` §30/§31 (address-selected fusions). Report
> findings to the maintainer before any delivery change and wait.

## 0. The item

The qwen4exp (and MoE generally) router runs a **full 512-expert argsort per token** to take the top-10.
The CUDA/HIP `topk_moe` fusion replaces `softmax → argsort → get_rows → sum/clamp/div` with a partial
top-k, saving ~**0.5 % prefill** on Strix Halo (gfx1151) — but the fusion's memory-ranges guard
(`ggml_cuda_check_fusion_memory_ranges(..., is_topk_moe=true)`) refuses to fire when the fused
`ffn_moe_weights_norm` output aliases the dead `ffn_moe_logits` buffer, which is the normal layout at
`pp2048` (2048 rows ≫ `TOPK_MOE_ROWS_PER_BLOCK`, so the refusal is **correct** — multi-block reads would
race the writes).

## 1. The two ways to unlock it

- **Pin the logits** (`ggml_set_output(logits)` in `build_ffn_moe_outer`, `llama-graph.cpp`) so the
  allocator cannot reuse the buffer → the fusion fires. Measured: `k_argsort 94 → 0`, `topk_moe_cuda`
  96 calls, capture 4.852 → 4.826 s (**−26 ms ≈ −0.5 %**). **Cost: all 47 logits `[512×2048×4 B]` pinned =
  ~188 MB held for the graph duration.**
- Find a layout-only way to avoid the alias so the **unfused** chain numerics are untouched — but the
  fusion *is* the fix for the sort waste, so this mostly means not using it.

## 2. The blocker: a numerics fork (this is the whole campaign)

The fused topk is **not transparent** for qwen4exp. On the fixed 838-token prompt, top1 logit:

| arm | top1 logit |
|---|---:|
| A unfused | **18.424** |
| A + fused | **18.690** (0.27 scale — a real behavioural change, not a ULP) |
| B (the reference shape) | 18.086 |

The whole token stream diverges from step 0, and **neither A variant matches B**. The fused kernel's
internal softmax/top-k/weights differ from the plain-op chain for this router shape. Adopting the fusion
is therefore a **numerics re-baseline with an unknown correctness direction** (the kernel may be subtly
wrong for this config, or the unfused chain may be the deviant).

## 3. What to do

1. **Quality-gate the fused kernel against the plain-op chain**: build a CPU reference for the
   softmax→top-k→weights chain at the qwen4exp router shape, and compare both GPU arms to it.
2. If the fused behaviour validates as equivalent-or-better, adopt the pin (scoped ~188 MB) behind an env
   kill-switch and re-baseline (prefill-logit KLD gate + MTP + PPL).
3. If it does not validate, the ~0.5 % is not worth a numerics fork; document and drop. The discovery doc
   lists two smaller open leads (the `scale_f32`/`unary_op` excess windows and the GDN +72 launches) that
   do not require a numerics decision.

## 4. Current reality / code

- The fusion and its guard are live in the r48 tree: `ggml_cuda_topk_moe_fusion` (`ggml-cuda.cu:5503`),
  `ggml_cuda_topk_moe_fusion_disabled` (`:5494`, env `GGML_CUDA_DISABLE_TOPK_MOE_FUSION`), and the
  `ggml_cuda_check_fusion_memory_ranges(..., is_topk_moe=true)` call sites (`:6907`, `:6923`). The
  `logits_may_alias` exemption (`:5681`) permits logits/output aliasing only when
  `nrows <= TOPK_MOE_ROWS_PER_BLOCK`.
- The fusion is a prefill win only; the `W=1..8` decode band must keep its per-op reduction order.

## 5. Gates

- Prefill-logit KLD gate (`scripts/gate-prefill-logits.sh`): mean KLD ≤ 0.005 and same-top-p ≥ 98 %.
- `test-logits-width-probe` `W=1..8` one hash; adaptive-MTP gate (`-n 3000`).
- Perplexity vs the unfused chain on wikitext-2, in addition to the CPU reference, if the fusion is
  adopted.
- `scripts/validate-set.sh` if promoted; new env knob documented in `ENVIRONMENT.md`.

## 6. Pointers

- `TODO.md` → "Parked (not planned now)" → item 5(a).
- [`archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`](../../archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md)
  (the kernel-count ledger, the alias analysis, the numeric fork, and the follow-up leads).
- `ggml/src/ggml-cuda/ggml-cuda.cu:5494-5700` (the fusion + the `is_topk_moe` guard),
  `src/llama-graph.cpp` (`build_ffn_moe_outer`).
- `GREEDY-PURITY.md` §30/§31 (address-selected fusions must be bit-transparent); §20 (cross-backend
  consistency).
