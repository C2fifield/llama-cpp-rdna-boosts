# RESOLUTION — gdn-state-gather-skip (PR #124)

**Status:** FOLDED into the delivery as **block 16** on 2026-10-14 (release `v16-a55e952b8-r45`).
**Owner:** `stew675/llama-cpp-rdna-boosts`; contributor `@briansp2020` (PR #124).

## What was adopted

`0001-cuda-skip-the-GDN-state-gather-the-sequential-kernel.patch` — in the decode/verify band
(one sequence, `<= 16` tokens) the CUDA graph skips each GDN layer's `GET_ROWS` recurrent-state
gather; the sequential kernel reads its initial state from the cache row instead. Default-on
kill-switch `GGML_CUDA_FUSE_GDN_STATE_GATHER=0` restores the gather.

The patch brings two small non-CUDA-backend changes with it, both folded: `rs_s_copy` is flagged
`OUTPUT` in `src/llama-graph.cpp`, and the scheduler keeps that flag on a single split-input copy
(`ggml/src/ggml-backend.cpp`, `n_copies == 1`); without them the allocator could reuse the row-id
buffer before the last layer's GDN read it.

## Validation on the delivery tree (gfx1201, 3x and 2x R9700)

- `Qwen3.5-4B-Q8_0` same-seed coherence identical to the r44 baseline.
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + shared `mtp-...-Q8_0` head, 2 GPUs, `--lazy-mode off`,
  `-sm tensor --fit on`, MTP `n_max 3`: `--spec-type none == draft-mtp` byte-identical
  (sha `2b2341670f23`, 3000 tokens), and identical to the r44 baseline.
- `test-backend-ops -o GATED_DELTA_NET`: OK.
- Prefill-logit KLD gate: mean **0.000707** / same-top-p **98.755 %** (PASS, unchanged).

## The concurrent-sequence report

`@DanoPTT` reported (2026-10-10, Windows, `-np 2 --kv-unified`, r38) that the skip arm changed
between concurrent repetitions while the r30 reference did not.  That residual is the
**unified-KV / attention-offset non-determinism** root-caused in the
`wip/kv-cell-offset-nondeterminism/` campaign (issue #134): a sequence's attention result depends
on the physical set of KV cells it occupies, so `-np > 1 --kv-unified` is not deterministic
regardless of this patch (documented gotcha; use `--no-kv-unified`).  The maintainer confirmed the
non-determinism sources were resolved by the earlier campaigns (DEVMAP removal in r40 and the
r41-r44 fixes).
