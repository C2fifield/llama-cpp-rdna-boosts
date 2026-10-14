# `wshadow-cleanup` — RESOLVED (r55, 2026-10-13)

**Promoted into the delivery as block 00 (upstream base) + block 05 (one feature-introduced site).**

## Result

All 139 `-Wshadow` sites in `src/` were fixed and `-Wshadow` is enabled on the `llama` C++ target.
The full plan and evidence live in `WORKLOG.md` 261013.005 (r55); the audit that scoped it is
`archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`.

- 46 "shadows a local variable" + 92 "shadows a field" + 1 "modifies constructor parameter that shadows
  a field" = **139** sites across 35 files.
- 138/139 are upstream base code -> **block 00**; the single feature-introduced site
  (`src/llama-kv-cache.cpp` `const int64_t n_stream = cell_pos->ne[1];`, added by block 05) -> **block 05**.
- `-Wshadow` enabled on the `llama` target (`src/CMakeLists.txt`, `LLAMA_ALL_WARNINGS`).
- Two real upstream bugs surfaced by the cleanup: the **duplicated nested `seq_id` loop** in
  `llama-batch.cpp` (collapsed) and `llama-memory-recurrent.cpp`'s `n_rs_seq = 0;`, which zeroed the
  constructor *parameter* instead of the *member*, so an empty memory module never actually disabled
  rollback snapshots (now `this->n_rs_seq = 0;`).

## Note on the original plan

The campaign proposed `-Wshadow -Wno-shadow-field-in-constructor`. On clang 22 that flag is a no-op for
the member-shadow class (`-Wno-shadow-field` does not control it either), so all 139 sites were fixed
rather than suppressed.

## Validation

`validate-set.sh` strict 12/12; the `llama` target builds with **0** `-Wshadow` warnings; 4B greedy
`1c5d32ac537d` unchanged; `FLASH_ATTN_EXT` 6389/6389.
