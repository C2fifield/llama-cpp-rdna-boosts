# RESOLUTION -- `add-rms-norm-q8-decode` (PR #174, @briansp2020)

**Status: RESOLVED (2026-10-17), promoted into `v12-097cf33d9-r78` as block 10.**
The contributor's format-patch and their write-up are preserved in this directory; nothing was left
behind as a `wip/` stub (the campaign arrived as a PR branch, never on `main`'s `wip/`).

## What landed (block 10)

One line: `ggml_cuda_add_rms_q8_1_try_fuse` no longer requires `node->ne[1] > 1`.  The residual `ADD` in
front of a narrow-band `RMS_NORM` whose product feeds an mmvq matmul is now folded into
`rms_norm_q8_1_f32<..., true>` for a single token as well.  The matcher, the kernel and the Q8_1
cache key are unchanged; the kernel's values are exactly the unfused `k_bin_bcast` + `rms_norm_q8_1`
pair (`GREEDY-PURITY.md` §39/§42).  Kill switch (pre-existing) `GGML_CUDA_FUSE_ADD_RMS_Q8=0`.

The `ne[1] <= 1` exclusion dated from the verify-band landing, on the reasoning that the attention
residual `ADD` is already folded into the output projection's mmvq epilogue at one token.  That is true
for the attention residual, but **not** for the FFN residual after `moe_weighted_reduction` (a
non-matmul producer), which ran as its own `k_bin_bcast` launch every layer.

## Where it is reachable

- **Active: dense models** (Qwen3.8-27B).  On a 40-token trace (3 GPU tensor, q8_0 KV) the standalone
  add launches fall **19,152 -> 4,158** and `rms_norm_q8_1_f32<1024, true>` rises **756 -> 15,750**.
- **Inert: the MoE models on this box** (Qwen3.6-35B-A3B).  Their FFN residual add is already consumed
  by the fused shared-expert output epilogue (`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` territory), so the
  `ADD` node never reaches this matcher: the r77 and r78 kernel histograms are identical
  (150,854 kernels each, same per-name counts).

## Validation (r78 candidate, gfx1201, 3x R9700, ROCm 7.14.1)

- **Text byte-identical to r77** and `none == draft-mtp n3`: dense 27B UD-Q4_K_XL 3 GPU tensor
  `sha 7ef45a05dcfe` (7405 chars, `-n 3000`, q8_0 KV, seed 42, temp 0) and MoE 35B-A3B UD-Q4_K_M 2 GPU
  tensor `sha e726a23ca0cc` (5500 chars).
- `test-logits-width-probe` (layer, q8_0 KV): PASS, worst maxdiff 0, row-0 hash identical to r77
  (`5f83c38e2a620d84` at P=512); `test-backend-ops -o RMS_NORM` / `-o ADD` pass.
- Prefill-logit gate: **0.000707 / 98.755 %**, exactly the r77 binary's own figure against the same
  base (the change is decode-only; the gate confirms nothing leaked into prefill).
- Speed (llama-bench `-p 512 -n 256 -r 5`, two alternating rounds): dense 27B 3 GPU tensor tg
  **53.98 -> 54.88 t/s (+1.7 %)**; with `GGML_CUDA_FUSE_ADD_RMS_Q8=0` 53.91 (the fusion accounts for all
  of it).  pp512 2063.6 -> 2063.1 (flat).  MoE 35B-A3B 2 GPU tensor 114.63 -> 114.45 (noise, as
  expected -- inert there).  `llama-batched-bench -npl 1,4,8` (dense 27B, q8_0 KV, 3 rounds): B=1
  28.52 -> 28.88, B=4 105.7 -> 105.0, B=8 184.5 -> 185.4 -- all within noise; the B=4 figure is
  reproduced (or undercut) by the switch-off control arm, so it is not a regression (rule 5 of
  `benchmarks/mtp-adaptive-methodology.md`).
