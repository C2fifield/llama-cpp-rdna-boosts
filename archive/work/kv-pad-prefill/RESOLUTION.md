# RESOLUTION -- `kv-pad-prefill` (PR #168, @briansp2020)

**Status: RESOLVED (2026-10-16), promoted into `v12-097cf33d9-r77` as **block 10**, as an OPT-IN tuning
value (default unchanged: off).**  The contributor's format-patch and write-up are preserved in this
directory; nothing was left behind as a `wip/` stub (the campaign arrived as a PR branch, never on
`main`'s `wip/`).

## What landed (block 10)

`llama_kv_cache::get_n_kv(sinfo, n_tokens)` now takes the ubatch size.  A prompt-sized ubatch
(`n_tokens > 64`) pads `n_kv` to `LLAMA_KV_N_PAD_PREFILL`, rounded up to a power of two, once the
decode-padded occupancy (`GGML_PAD(used, LLAMA_KV_N_PAD_MIN)`) exceeds `LLAMA_KV_N_PAD_PREFILL_FROM`
(default 4096).  With `-ub 2048` and `8192`, four consecutive prompt chunks share one graph shape, so
the split state / allocation / capture is not rebuilt per chunk.  Decode and verify batches keep the
decode pad (a coarse pad there slows every step and changes its numerics), `n_kv` is still capped at the
cache size, and the extra cells are masked exactly as before.  It applies to the **unified** KV cache
only, and not to training (whose graphs assert `n_kv == n_tokens`).

## Why opt-in and not default-on

The win is shape-dependent, and default-on would regress one shipped configuration:

| config (r77 build, `-ub 2048 -b 2048`, q8_0 KV, 31.7 k-token prompt, MTP n3) | pad off | `=8192` |
|---|---|---|
| MoE 35B-A3B UD-Q4_K_M, 2 GPUs tensor | 4383 t/s | **4471 t/s (+2.0 %)** |
| dense 27B UD-Q4_K_XL, 3 GPUs tensor | 1573 t/s | **1589 t/s (+1.0 %)** |
| dense 27B UD-Q4_K_XL, **1 GPU** | 1156 t/s | 1152 t/s (**-0.35 %**) |

A single GPU rebuilds its graph cheaply, so the coarser pad only adds masked attention work there.  The
delivery's rule is that no shipped default may regress, so this stays a `tuning (opt-in)` value beside
the existing `LLAMA_KV_N_PAD_MIN` (also opt-in), documented with these numbers in `ENVIRONMENT.md`.
`8192` is the value to use on a multi-GPU, prefill-heavy setup.  Flipping the default is a one-line
change to the `getenv` default if the maintainer decides the multi-GPU config is the only one that
matters.

## Validation (r77 candidate, gfx1201)

- Output unchanged with the pad on: the 31.5 k-prompt completions hash identically to pad off (MoE
  `f1610e5940f6`, dense `01f03fc68e13`).
- Prefill-logit gate at 8 k ctx against the r76 8 k base: **0.000000 / 99.996 %**, identical to the pad-off
  arm (and to the harness's own r76-vs-r76 figure at that length), so the graph-shape change is
  numerically inert.
- The default (pad off) build is bit-identical to r76 everywhere else: dense/MoE MTP text, acceptance,
  width probe, `llama-bench`/`llama-batched-bench` within noise (see `WORKLOG.md` 261016.012).
- The contributor's r71 2x R9700 numbers (+4.8 % at 31.5 k, decode unchanged) are the same effect; this
  box measures +2.0 % on the same topology with a different model.

## Record

`WORKLOG.md` 261016.012 (r77); `ENVIRONMENT.md` (`LLAMA_KV_N_PAD_PREFILL`,
`LLAMA_KV_N_PAD_PREFILL_FROM`); `patches/README.md` (current-release block).
