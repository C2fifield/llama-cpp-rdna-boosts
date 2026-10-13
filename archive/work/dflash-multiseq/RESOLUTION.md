# RESOLUTION — `dflash-multiseq` (PR #144): PROMOTED into r53

**Status: folded into delivery `v12-097cf33d9-r53` (2026-10-13).**  Record: `WORKLOG.md` 261013.003.

## What was promoted

PR [#144](https://github.com/stew675/llama-cpp-rdna-boosts/pull/144) (@overdoingism) had two commits,
folded into the block that owns each concern (not a trailing block):

| hunk | owning block |
|---|---|
| `common/speculative.cpp` DFlash F1 multi-sequence (`lf_dev` for `n_seq > 1`, `llama_lf_get_dev_rows`, host-experts guard) | **11** |
| `src/llama-context.{cpp,h}` + `src/llama-ext.h` F1 target API (`lf_dev_rows`, `get_lf_dev_rows`, the `output_reorder` guard) | **10** |
| `src/llama-context.cpp` `GGML_READBACK_SYNC` wait for a ubatch without outputs (the per-token DFlash layer inputs) | **00** |

The patch is preserved as `dflash-multiseq.patch` (cut against r50; the folded form is in the r53
blocks).  The host-resident-MoE-experts guard ships exactly as the PR wrote it: with `-ncmoe`/`-cmoe`
several sequences stay on the host path (the movable-boundary slab's device reads were observed to
break there; issue #127, MoE form — a follow-up, not folded).

## Validation (gfx1201, 2x R9700, `-sm layer`; `-sm tensor` + DFlash is a pre-existing abort, r52 does it too)

- single sequence, 27B UD-Q4_K_XL + DFlash2, prose, n=512: `GGML_LF_DFLASH_DEV=1` and `=0` are
  byte-identical (`834bfd5bf6b8`); F1 prefill 1181 vs 1099 t/s.
- `--parallel 2` server, two pinned slots, F1 on == off: slot0 `bd13536b7a7b`, slot1 `c8a25913427d`,
  acceptance 0.62121 / 0.48052; F1 ~5 % faster prefill (concurrent wall 10.87 vs 11.38 s).
- host-experts guard fires: 35B-A3B + `-ncmoe 40` + `--parallel 2` logs
  `F1: 2 sequences with host-resident MoE experts, using the host path` and stays coherent.
- `test-backend-ops -o MUL_MAT_ID -p 'n=1,'` 162/162; no `ggml/` source changed.
