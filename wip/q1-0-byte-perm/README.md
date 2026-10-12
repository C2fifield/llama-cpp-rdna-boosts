# Q1_0 `__byte_perm` → `v_perm_b32` (the last un-ported i-quant table lookup)

**Status: OPEN / scoping, small (2026-10-11).**  Seeded from the K0 of the now-closed
`strata-amd-kernels` campaign (its only residual), and it **replaces that campaign** in the open set
(`TODO.md` #54).  Not part of the delivery.

## What

Two sites still call HIP's `__byte_perm`, whose ROCm 7.14 body is a private-memory union with four
indexed byte loads per call (`/opt/rocm-7.14.1-gfx120X/include/hip/amd_detail/amd_device_functions.h:246`),
instead of the single `v_perm_b32` instruction (`__builtin_amdgcn_perm`):

| site | file:line | role |
|---|---|---|
| `vec_dot_q1_0_q8_1` | `ggml/src/ggml-cuda/vecdotq.cuh:833-844` (12 calls per 32-value chunk) | the Q1_0 **mmvq** dot product (decode / verify widths) |
| `ggml_cuda_mmq_load_tiles_q1_0` | `ggml/src/ggml-cuda/mmq-load-tiles.cuh:50-61` (12 calls) | the Q1_0 **MMQ** tile loader (prefill) |

Every other i-quant table lookup in the tree already uses the instruction, so this is purely
"finish the port":

- `get_int_from_table_16` — HIP arm at `vecdotq.cuh:46-62` (`__builtin_amdgcn_perm`),
- `vec_dot_q2_0_q8_1` — HIP arm at `vecdotq.cuh:879-892`,
- `ggml_cuda_mmq_load_tiles_q2_0` — HIP arm at `mmq-load-tiles.cuh:141-154`,

plus the SWAR packed-byte helpers already shipped upstream in `vendors/hip.h` and our own
`apply_ksigns` (`vecdotq.cuh:119`).  This residue is exactly what the `strata-amd-kernels` K0 diff
against Strata's `#262` (`__byte_perm` → `v_perm_b32`) turned up.

## Why it is small, and why it is still worth doing

- **Bounded:** ~24 call sites across two functions, and the target form already exists two functions
  away (the Q2_0 arms) — it is a mechanical mirror, not a new kernel.
- **Bit-identical by construction:** `v_perm_b32` *is* the hardware meaning of CUDA's `__byte_perm`
  (selector byte *i* picks byte `s.nibble[i] & 7` of the pair `{y:x}`).  The Q2_0 arm's selector
  expansion (`(q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18)`) is the idiom to
  reuse; the `0x11100100` crumb unpack, the `0x01FF` nibble unpack and the `0x5410` / `0x7632`
  unshuffles all take constant selectors that can be expanded the same way.
- **No local measurement is possible:** Q1_0 (~1.7 bpw) is a rare type and there is **no Q1_0 model** in
  `/llm/models/`.  So this carries **no speed claim** — the class of fix measured R9700 decode +15 % in
  Strata's *own* engine (`docs/AMD_HIP.md`), but that does not transfer to a claim here.  The value is
  consistency/completeness: no Q1_0 user should land on the private-memory path because it is the one
  type we skipped.
- **Upstream-shaped:** a pure AMD table-lookup port inside `vecdotq.cuh`/`mmq-load-tiles.cuh`, so it also
  belongs in `upstream/` as a candidate.

## Plan

1. Mirror the `vec_dot_q2_0_q8_1` HIP arm into `vec_dot_q1_0_q8_1` (expand each `__byte_perm` selector to
   `__builtin_amdgcn_perm`).
2. Mirror `ggml_cuda_mmq_load_tiles_q2_0` into `ggml_cuda_mmq_load_tiles_q1_0`.
3. Prove bit-identity with the in-tree oracle (below) and the coherence gate
   (`GREEDY-PURITY.md`) if a Q1_0 model can be obtained.
4. Cut an `upstream/UPSTREAM-PR-q1-0-amd-perm.*` copy of the same two hunks.

## Oracle / gate

There is no Q1_0 model on the box, so the gate is the **CPU-oracle backend-ops matrix**:

```sh
./build/bin/test-backend-ops -o MUL_MAT -b CUDA0   # Q1_0 is in all_types/base_types/other_types
```

`test-backend-ops`'s `all_types` includes `GGML_TYPE_Q1_0` and `test_mul_mat_id` exercises it, so the
mmvq **and** MMQ paths are both covered by generated quantized data (no model file needed), and the
comparison is against the CPU reference.  Run it before and after; the kernels must be bit-identical.
This is the "QSA op has a CPU oracle" discipline (`GREEDY-PURITY.md` §21) applied to a dequant/table
change.  A speed measurement would need a real Q1_0 GGUF, which is a separate question.

## Related

- `archive/work/strata-amd-kernels/README.md` — the K0/K1 that produced this residual (closed as
  refuted).
- `TODO.md` #54.
- `upstream/README.md`.
- Strata `#262` (`87fbc000`, `600d7307`) for the original `__byte_perm` → `v_perm_b32` mapping rationale.
