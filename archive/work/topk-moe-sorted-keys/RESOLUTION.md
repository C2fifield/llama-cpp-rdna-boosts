# RESOLUTION — topk-moe-sorted-keys (PR #141)

**Status:** FOLDED into the delivery as **block 08** (`ggml/src/ggml-cuda/topk-moe.cu`, r52, 2026-10-13),
**default-on**.  **Owner:** `@briansp2020` (PR #141).

## What was adopted

`0001-ggml-cuda-topk_moe-argmax-over-sorted-64-bit-keys-wi.patch` — the `topk_moe` argmax butterfly
exchanges lanes with `ggml_cuda_xor_lane` (DPP / `permlanex16` on RDNA) instead of
`__shfl_xor_sync`, and for the gating outputs (softmax / sigmoid / sqrt-softplus, no bias, no delayed
softmax, power-of-2 candidates per lane) each candidate becomes one 64-bit key (order-preserving float
bits above the inverted expert index) that is sorted once per lane.  Ids, weights and their order are
unchanged; ties still go to the lower expert index.

`GGML_CUDA_TOPK_SORTED=0` restores the per-round scan.  The reporter measured the argmax rounds at
7.1 -> 3.8 us per call (512 experts, 10 used, 4 tokens); the end-to-end gain is within noise, so this
is a kernel-level improvement.

## Validation on the delivery tree (gfx1201, 3x R9700)

- `test-backend-ops -o TOPK_MOE`: **448/448**, 0 FAIL (matches the reporter's 448/448).
- `test-logits-width-probe`: **PASS (worst maxdiff 0)** on 4B and 35B-A3B.
- 35B-A3B draft-MTP (`-sm tensor -c 16384 -n 3000`, bf16 KV, prose): acceptance **0.733** identical to
  r51, **160.4** vs **159.2 t/s** (within noise, as expected for a kernel-only change).
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS).
