# `gdn-conv-device-idiom` — one device idiom in `gdn-conv.cu` (TODO #32)

**Status: OPEN (seeded 2026-10-11), cosmetic.** Not part of the delivery; nothing here may be applied to
the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). Same item as
**TODO #32**. Found during the PR #102 review; recorded in
[`archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md`](../../archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md)
(review notes). No correctness impact, no perf impact — this is a one-line consistency fix to make at the
next block-15 touch.

> **Cold start.** Read this file, then the r10 review note and `patches/README.md` block 15. Report
> findings to the maintainer before any delivery change and wait.

## 0. The two idioms

The RDNA4 decode/verify **GDN conv fusion** (`gdn_conv_check`, the 2..255-token arm added by PR #102) and
the **batched-copy** path added in the same PR gate on RDNA4 with two different ways of asking "which
device is this?":

```c
// ggml/src/ggml-cuda/gdn-conv.cu:72
const bool verify_ok = conv_verify && T >= 2 &&
    GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);

// ggml/src/ggml-cuda/cpy-batch.cu:66
if (!cb_on() || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc)) return 0;
```

`ggml_cuda_get_device()` reads the cuda/hip current device; `ctx.device` is the backend context's device.
They agree on every current graph because `ggml_cuda_try_fuse` (which calls both matchers) runs after the
device is set for the split, so this is two idioms for the same thing.

## 1. Current reality check (r48)

- **Both call sites are still present unchanged** in the r48 tree (gated above). `gdn_conv_check` is used
  by both the fused dispatch (`ggml-cuda.cu:6499`) and the alloc-deps pass (`ggml-cuda.cu:9449`);
  `ggml_cuda_cpy_batch` is called from the fusion dispatch.
- `ggml_cuda_should_fuse_mul_mat_vec_q` also uses `ggml_cuda_get_device()`, so `gdn-conv.cu` is consistent
  with the *majority* idiom — if anything, `cpy-batch.cu` is the odd one out.
- No behavioural difference has ever been observed; this is not a bug report. The r10 review flagged it as
  a nit.

## 2. Fix options

1. **One-liner (recommended):** change `cpy-batch.cu:66` to `ggml_cuda_info().devices[ggml_cuda_get_device()].cc`,
   matching `gdn-conv.cu` and `ggml_cuda_should_fuse_mul_mat_vec_q`. `ggml_cuda_cpy_batch` already has
   `ctx`, so this is trivially safe and the diff is one line.
2. **Alternative:** thread the context device into `gdn_conv_check` (it currently takes only the graph and
   index). This is invasive (the matchers are called from two places and would need a device argument) and
   not worth it for a cosmetic cleanup.

Prefer (1). Do it in the same change as any other block-15 edit to avoid a rebase-only commit (and to keep
the block-15 commit's diff reviewable), exactly as the TODO suggests.

## 3. Gates

- **Behaviour must be byte-identical** — this changes only which expression reads the same value. Verify
  the dense 4B same-seed coherence gate (`1c5d32ac537d`) and, because the touched matcher gates the GDN
  conv fusion on RDNA4 verify batches, the adaptive-MTP gate on a qwen4exp/Flash-Next model.
- Because the expression really is equivalent on the current graph, no purity re-run beyond the standard
  coherence gate is needed; but state in the commit that it is a no-op idiom change.
- `scripts/validate-set.sh` if promoted; regenerate the block-15 patch via `scripts/make-patches.sh`.

## 4. Pointers

- `TODO.md` #32.
- [`archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md`](../../archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md)
  (the review note, §"Verify-step A/B" and the bullet naming both idioms).
- `ggml/src/ggml-cuda/gdn-conv.cu:72`, `ggml/src/ggml-cuda/cpy-batch.cu:66`,
  `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_should_fuse_mul_mat_vec_q`).
- `patches/README.md` block 15; `WORKLOG.md` 2026-10-04 (r10, PR #102).
