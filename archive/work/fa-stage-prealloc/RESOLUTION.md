# RESOLUTION — `fa-stage-prealloc` (issue #138): PROMOTED into r53

**Status: folded into delivery `v12-097cf33d9-r53` (2026-10-13), default-on.**  Record: `WORKLOG.md`
261013.003.

## What was promoted

`fa-stage-prealloc.patch` (reported by @DanoPTT, issue
[#138](https://github.com/stew675/llama-cpp-rdna-boosts/issues/138)) was folded into the two blocks
that own the changed regions (the old README's "block 15" is pre-V2.0 numbering):

| hunk | owning block |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` `fattn_stage_try_get(size_hint)` definition | **09** |
| `ggml/src/ggml-cuda/fattn-common.cuh` `launch_fattn` hint computation | **10** |

Default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` as the kill switch (per the default-on policy).  The
hint is only tried while `optional_alloc_within_free_cap(...)` allows it, so a slab-active run cannot
have its free VRAM consumed by the prealloc; an oversized/unbounded hint can never cost more than the
old 1.25x ladder.

## Validation

- prefill-logit KLD gate on r53: mean KLD **0.000707**, same-top-p **98.755 %** PASS (unchanged from r52).
- the reporter's Windows/WDDM A/B (issue #138 comment): dedicated growth **-77 % / -76 %**, pp/tg
  unchanged, greedy hash identical.
- `GGML_CUDA_FA_STAGE_MAX_MB=0` (unbounded) passes no hint; `=1` keeps growth near zero at ~11 % pp cost.
- The `ENVIRONMENT.md` row and the `patches/README.md` note are added in r53.
