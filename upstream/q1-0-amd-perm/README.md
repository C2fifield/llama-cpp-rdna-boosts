# `ggml-cuda`: use `__builtin_amdgcn_perm` for the Q1_0 table lookups

> **Upstream candidate** · status: prepared 2026-10-11; applies clean to `a55e952b8`; bit-identical (16.7M randomized cases); **no Q1_0 model locally**, so no speed claim · not filed.
> Delivery: **block 07** (Q1_0) · patch: [`UPSTREAM-PR-q1-0-amd-perm.patch`](UPSTREAM-PR-q1-0-amd-perm.patch) (this directory).

## What

`vec_dot_q1_0_q8_1` (`ggml/src/ggml-cuda/vecdotq.cuh`) and `ggml_cuda_mmq_load_tiles_q1_0`
(`ggml/src/ggml-cuda/mmq-load-tiles.cuh`) are the last i-quant table lookups that call HIP's
`__byte_perm`, whose ROCm 7.14 body is a private-memory union of four indexed byte loads
(`amd_device_functions.h`), while every other lookup already uses `__builtin_amdgcn_perm`
(`v_perm_b32`): `get_int_from_table_16`, `vec_dot_q2_0_q8_1`, `ggml_cuda_mmq_load_tiles_q2_0`, and the
SWAR helpers in `vendors/hip.h`.

A plain substitution is wrong: CUDA's `__byte_perm` builds its selector from four **nibbles** (one per
output byte), while `v_perm_b32` builds it from four full **bytes**.  The patch adds
`ggml_cuda_byte_perm(x, y, s)`, which expands the four selector nibbles into selector bytes and swaps
the operands (`v_perm_b32(a, b, s)` treats `b` as the low dword, `__byte_perm(x, y, s)` treats `x` as
the low dword), then routes all 24 Q1_0 sites through it.  Non-HIP builds keep `__byte_perm`, so the
CUDA path is unchanged.

This is a consistency/completeness fix: no Q1_0 user should land on the private-memory path simply
because Q1_0 was the one type that was skipped.

## Validation

- **Exact by construction and by test**: `ggml_cuda_byte_perm` reproduces HIP `__byte_perm` over
  16.7M randomized `(x, y, s)` cases on gfx1201 (0 mismatches), including selectors with bit 3 and the
  high bits set.
- `test-backend-ops -o MUL_MAT -b ROCm0`: **1304/1304** passed, every `q1_0` width (mmvq and MMQ)
  matching the CPU reference.
- Patch cut against upstream `a55e952b8`; `git apply --check` is clean.

## What was NOT validated

- **No Q1_0 model locally**, so there is **no speed claim**.  Strata's `#262` measured R9700 decode +15 %
  in *its own* engine; that does not transfer to a claim here.  A speed measurement needs a real Q1_0
  GGUF, which is a separate question.
- **No NVIDIA hardware** was available; the CUDA arm deliberately keeps the existing `__byte_perm`.

## Where it lives

Delivery **block 15** (the same two hunks are inline; applying this file on top of the full delivery
would double-apply them).  Record: `archive/work/q1-0-byte-perm/README.md`, `WORKLOG.md` 261011.017 (r49).
