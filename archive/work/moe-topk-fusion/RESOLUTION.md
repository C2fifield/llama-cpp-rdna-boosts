# RESOLUTION — moe-topk-fusion (TODO item 5(a))

**Status: CLOSED — no delivery change (2026-10-13).** The campaign's premise does **not** reproduce on
the current delivery **r55**: the fused `topk_moe` router already fires at every prefill width, on both
the maintainer's devices (gfx1201 and gfx1151), and the ~188 MB logits pin is a **no-op**. The
~0.5-1.2 % prefill the campaign was created to unlock is already in the tree. **Nothing was promoted;
no release was cut.**

## What the campaign was

The 2026-09-06 discovery (qwen4exp IQ4_NL, gfx1151) found that the gallocr placed the router norm output
`ffn_moe_weights_norm` (tens of KB) into the just-freed router-logits buffer `ffn_moe_logits`
(`[512 x n_tokens x 4 B]`), and the fused-router guard
`ggml_cuda_check_fusion_memory_ranges(..., is_topk_moe=true)` correctly refused to fuse for
`nrows > TOPK_MOE_ROWS_PER_BLOCK` (`= 8`) because a multi-block read/write race would result. qwen4exp
then ran a full 512-expert argsort per token (~0.5 % prefill). The proposed unlock was
`ggml_set_output(logits)` in `build_moe_ffn` (keeps the buffer alive, `ggml-alloc.c:749`), at a cost of
~188 MB held for the graph — **only if the alias actually exists**.

## What was checked

Instrumented r55 (a trace at the `ggml_cuda_op_topk_moe()` call and a match/guard trace, both in the
reverted diagnostic patch) and read the guard outcome and the fused-kernel coverage directly. The key
signal: the fusion `return`s before the post-decision diagnostic, so a refusal prints
`TOPK_MOE_DBG ... mem=0`; **zero such lines appeared at any width on either device**.

### gfx1201 (3x R9700, ROCm 7.14.1)

| model | split | ubatch | fusion fires up to | refusals (`nrows>8`) | pin effect |
|---|---|---:|---:|---|---|
| Flash-Next GSQ-IQ3_XXS | tensor | 512 | 512 | none | none |
| Flash-Next GSQ-IQ3_XXS | layer | 512 | 512 | none | none |
| Flash-Next GSQ-IQ3_XXS | tensor | 2048 | **2048** | none | none |
| Flash-Next GSQ-IQ3_XXS, `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` | tensor | 2048 | 2048 | none | none |
| Flash-Next IQ4_NL (`-ncmoe 12`) | tensor | 512 | 280 | none | n/a |

Prefill pp2048, GSQ-IQ3_XXS: **1575.7 t/s** with the fusion vs **1563.8 t/s** with
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` (`-r 3`) — the win is already banked.

### gfx1151 (Strix Halo, single GPU — the discovery's own device)

Flash-Next **IQ4_NL**, `-b 2048 -ub 2048 -c 16384`, prose prompt:

- Fusion fires at `n_tokens = 1, 2, 4, 42, 1160, 2044, 2048`; `match op=SOFT_MAX ... can_fuse=1` at
  every width; **0 refusals**.
- `LLAMA_MOE_TOPK_PIN_LOGITS=1` produces an **identical** fired set (byte-for-byte `diff` clean).
- Prefill pp2048: **1030.10 ± 4.50 t/s** with the fusion vs **1018.03 ± 2.86 t/s** with it disabled
  (`-r 3`) — ~+1.2 %, already in the tree.

## Why it is moot

Two independent reasons:

1. **Numerics** (retired 2026-09-13, `WORKLOG.md` item 19 / `GREEDY-PURITY.md` §31): the fused router
   was made bit-identical to the generic chain (softmax reduction order, `sum_rows`+divide, index
   tie-break), validated with a force-fuse probe that covered exactly the guard-skipped call sites.
2. **The alias itself is gone on r55.** The mechanism is unchanged (`TOPK_MOE_ROWS_PER_BLOCK = 8`, same
   `is_topk_moe` exemption); what changed is the **allocation plan** between the r48-era discovery and
   r55 (V2.0 re-org, the r20 compute-buffer margin, graph reorder), so the gallocr no longer reuses the
   freed logits buffer. With no overlap, the guard passes and the fusion fires. Pinning a logits tensor
   that was never reused cannot help — hence the no-op on both devices.

## Disposition

- No change to `patches/`, `release.json`, `rdna-boosts-all.patch`, or any env var; no release.
- `TODO.md` item **5(a) removed**; the `wip/CAMPAIGNS.md` entry deleted (no stub).
- `wip/moe-topk-fusion/` moved here (`README.md` + `topk-moe-diagnostics.patch`).
- Recorded in `WORKLOG.md` 261013.006.

## Files

- `README.md` — the campaign record (finding, mechanism, pointers).
- `topk-moe-diagnostics.patch` — the temporary trace + env-gated pin used for the investigation
  (against the r55 tree; **not** part of the delivery, reverted from both `~/llama.cpp` checkouts).
