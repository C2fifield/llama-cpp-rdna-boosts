# upstream `MUL_MAT_ID` duplicate-ids on CUDA/HIP (r50 follow-up)

**Not part of the delivery.** Seeded 2026-10-12 while rebasing the set onto `097cf33d9`; the failure was
confirmed to be an upstream bug, so the delivery ships on `097cf33d9` with this documented and this
campaign carries the fix.

## Symptom

`test-backend-ops -o MUL_MAT_ID -p 'dup=1'` fails the F16 cases and then page-faults the GPU:

```
[MUL_MAT_ID] ERR = 1.45-1.86 > 0.0005  MUL_MAT_ID(type_a=f16, type_b=f32, n_mats=28, n_used=10,
                                            b=0, m=1024, n=9/16/33/64, k=256, dup=1)
...
Memory access fault by GPU node-3 ... kernel: void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0,false,false>(...)
```

`q8_0` and `q4_0` `dup=1` cases pass; only F16 fails. The non-`dup` `MUL_MAT_ID` matrix is green.

## Confirmed upstream (not the delivery)

A plain, unpatched `097cf33d9` build (`~/llama.cpp/build-ops-base`, `cmake --build --target
test-backend-ops`) reproduces the **identical** four F16 `dup=1` failures and the same
`quantize_mmq_q8_1` memory fault + core dump. The `test_mul_mat_id_dup` cases were added by the Vulkan
fix `966baae76` ("vulkan: handle mul_mat_id duplicates in the prepass rather than looping", #29998),
which touched only `ggml-vulkan/*` + `tests/test-backend-ops.cpp`; CUDA/HIP was never fixed. The test is
not gated per-backend, so upstream's own HIP CI hits it.

## Scope of the fix

- The F16 `MUL_MAT_ID` path needs to handle repeated expert ids within a row. The `q8_0`/`q4_0` (MMQ) path
  already does (via `ggml_cuda_launch_mm_ids_helper`'s dedup/inverse map); F16 does not.
- The page fault in `quantize_mmq_q8_1` during the run suggests the F16 failure leaves state that a later
  MMQ case then consumes out of bounds — treat the fault as a second symptom, and re-check the MMQ
  `src1_q8_1` sizing independent of the dedup fix.
- Candidate homes: `ggml/src/ggml-cuda/mmvf.cu` (`mul_mat_vec_id`) and/or the non-MMQ `MUL_MAT_ID`
  dispatch in `ggml-cuda.cu`.

## Upstream handling

This is a candidate for `upstream/` (a self-contained PR against unadulterated `ggml-org/llama.cpp`): the
fix + the `test_mul_mat_id_dup` cases already upstream. Keep it out of `patches/` unless the maintainer
decides the delivery must carry it (the delivery is pure here, so it should not).

## Validation for the fix

- `test-backend-ops -o MUL_MAT_ID` green, including `dup=1` (the full matrix, not just F16).
- `test-backend-ops -o MUL_MAT` green (the dense sibling).
- No `Memory access fault` in the run; `scripts/check-gpu-faults.sh` clean on a follow-up run.
