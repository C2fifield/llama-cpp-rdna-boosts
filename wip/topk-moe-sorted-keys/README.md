# topk-moe-sorted-keys: faster `topk_moe` argmax for wide routers (same ids, weights and order)

One format-patch on top of `v16-097cf33d9-r50`, applied with `git am` (it also applies cleanly on r49).  It changes only
`ggml/src/ggml-cuda/topk-moe.cu`.  `GGML_CUDA_TOPK_SORTED=0` restores the current per-round scan.

## What it addresses

For Flash-Next (512 experts, 10 used), the fused router kernel spends most of its time in the 10 argmax rounds: 7.1 us
per call at 4 tokens, against 2.8 us for a single round.  Two changes:

- The argmax butterfly exchanges lanes with `ggml_cuda_xor_lane` (DPP / permlanex16 on RDNA, the helper the
  `warp_reduce_*` functions already use) instead of `__shfl_xor_sync`, which goes through `ds_bpermute`.
- For the gating outputs (softmax, sigmoid, sqrt-softplus; no bias, no delayed softmax, power-of-2 candidates per
  lane), each candidate becomes one 64-bit key: the order-preserving float bits above the inverted expert index.  Each
  lane sorts its keys once (bitonic, in registers).  After that, each round is a 5-stage u64 max over the lane heads
  plus a shift on the winning lane.

Keys are unique, so ties go to the lower expert index, as in the current scan.  The sorted path runs after the
existing NaN to -FLT_MAX sanitising, and the gating outputs are never -0.0, so the key order matches the float order.

## Measurements (2 x R9700 32 GB, gfx1201)

| | current scan | this patch |
|---|---:|---:|
| `test-backend-ops` shape 512 x 4, k = 10 (rocprofv3 kernel time) | 7.1 us | 3.8 us |
| in the server, per call during MTP verify | 6.6 us | 3.6 us |

The steps were DPP alone 6.1 us, then sorted float/index pairs 5.3 us, then 64-bit keys 3.8 us.

`test-backend-ops -o TOPK_MOE`: 448/448 on both GPUs (r50 + patch).  The greedy sha is unchanged on r49 and r50.  End
to end the saving (~3 us x 48 router calls per verify step) is too small to separate from run-to-run noise in our
t/s numbers, so this is offered as a kernel-level improvement only.

## Notes

- Only the `topk_moe` kernel changes; the plain `TOP_K` / argsort paths are untouched.
- The scan path is kept, and is still used for delayed softmax, biased routers, a non-power-of-2 number of
  candidates per lane, and routers with at most 32 experts.
