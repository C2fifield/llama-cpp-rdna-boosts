# RESOLUTION — upstream `MUL_MAT_ID` duplicate-ids: UPSTREAM CANDIDATE (not in the delivery)

**Status: closed 2026-10-13; the fix is an `upstream/` candidate, the delivery ships pure.**
Record: `WORKLOG.md` 261013.003.

## Outcome

The `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` failures (F16/q8_0/q4_0) and the `quantize_mmq_q8_1`
page fault were a genuine upstream CUDA/HIP bug: `mm_ids_helper` kept one compact column per
`(token, expert)` pair (dropping repeated ids and leaving `ids_src1` uninitialised), the MMQ/MMF
column-tile grids were sized from the token count (so an expanded expert's higher tiles were never
launched), and the non-compact MMF path kept a single slot per token.

The fix was written against unadulterated `097cf33d9`, validated with `test-backend-ops` on gfx1201,
and published as:

- `upstream/UPSTREAM-PR-mul-mat-id-dup.md`
- `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` (applies clean to `097cf33d9` and current master
  `1b70350c9`)

**Not filed** (awaiting the maintainer's go-ahead).  **Not in `patches/`** — the r53 delivery ships
pure on this bug.

## Validation

- `-o MUL_MAT_ID -p 'dup=1'` **16/16** (base 0/16); `-o MUL_MAT_ID` **947/947**; `-o MUL_MAT`
  **2444/2444**; `scripts/check-gpu-faults.sh` clean.
- 35B-A3B Q8_0 2x R9700 prefill pp512/pp2048 within noise of the unpatched tree (2587/2577 vs
  2588/2563 t/s).
