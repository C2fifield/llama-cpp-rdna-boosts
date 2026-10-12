# r50 merge hygiene — distribute the build fixes to the owning blocks

**Status: OPEN, seeded 2026-10-12.** The r50 delivery tip (`9a9e6aa6c`, release
`v16-097cf33d9-r50`) **builds and passes every release gate** (see `wip/r50-rebase/README.md`), but the
r2 **"every one of the 16 commits builds the `all` target"** standard is **not met**: the merge build
fixes are folded into block 15 instead of the blocks that own them. This campaign is the distribution,
a repeat of `archive/work/rebase-merge-hygiene/`.

## Method (do NOT use the incremental build)

The r2 lesson holds: a reused `build-check` cache hides configure/header errors and gives a **wrong**
per-block list. Use a **fresh `rm -rf` + configure + `all` build per commit**:

```bash
cd ~/llama-rebase     # the rebased rdna-boosts worktree
for c in $(git rev-list --reverse 097cf33d9..rdna-boosts); do
  git checkout -q "$c"
  rm -rf build-check && <configure> && cmake --build build-check --target all -j16 -- -k \
    || echo "FAIL $c $(git log -1 --format=%s $c)"
done
git checkout -q rdna-boosts
```

## Known distribution issues (from inspection, not yet sweep-confirmed)

1. **block 06** — `ggml/src/ggml-cuda/common.cuh`: `h2d_stage_budget()` (introduced by block 06) calls
   `h2d_stage_budget_explicit()`, but block 06's `common.cuh` **does not define it** (only the call site,
   line ~1655). The definition lives in the tip (block 15). Move the `static size_t
   h2d_stage_budget_explicit()` definition (and the `h2d_stage_slots()`/`h2d_stage_budget()` ordering) into
   block 06. This is a hard compile error at block 06 and every block after it.
2. **block 14** — `ggml/src/ggml-backend-meta.cpp`: block 14 is missing ~435 lines of its own canonical
   meta changes (they landed in block 15): `GGML_ENV_STR`, `ggml_backend_meta_device_slab_narrow2_floor` /
   `_slab_compute_narrow2` (+ the `ggml_backend_device_i` entries), `..._alloc_buffer_usage`,
   `..._get_compute_margin_pct`, and the `moe_cache_set_host_pool`/`_set_host_source` iface entries. Move
   them to block 14.  Also the host-tensor branch body fix (the `bcj.nodes[i] = node; continue; }` that
   closes the block-14 host branch in `ggml_backend_meta_graph_compute`) and the
   `ggml_cuda_mul_mat_takes_mmq` `warp_size` argument in `ggml/src/ggml-cuda/ggml-cuda.cu`.
3. **block 13** — already distributed: `get_stream_k(..., cc, prec_src1)` in `mmq.cuh` and the
   `GGML_PREC_Q8` args on `get_nthreads`/`get_I` in `mmq-vec-dot.cuh`. Verify on the sweep.
4. **block 15** — keep only its own fixes: the `ggml_cuda_try_ssm_scan_cache_fusion` restore (block 15's
   conflict had displaced it), the `fattn-mma-f16.cuh` `get_swizzled`/`load_tile` dedup, the `mmq.cuh`
   duplicate `mul_mat_q_routed_compact` template deletion, and `llama_context::sched_new()` (block 15's
   conflict had replaced it with `kq_mask_packed_reachable`). The block-15 commit should shrink by the
   block-14 subset once that moves.

## Mechanical approach that avoids the fixup-base trap

Fixups to older commits with `git rebase -i --autosquash` **do not work here**: the fixup diff's base is
the tip, so replaying it onto an earlier block conflicts or doubles hunks (this happened; the ssm fusion
was briefly duplicated). Instead: `git rebase -i`, mark the owning commits `edit`, and at each stop edit
the file directly (or `git apply` a hunk whose context matches that block), `git add` + `git commit
--amend`, then continue. For a fix that must move from block 15 to block 14, apply it at block 14 **and**
`git apply -R` the same hunk at block 15's stop so it is not committed twice.

## Definition of done

- fresh-per-commit `all` sweep: 16/16 OK.
- `scripts/validate-set.sh` still green and `release.json` tree unchanged (the final tree must not change
  — this is pure commit-boundary re-distribution, not new content).
- `WORKLOG.md` / `wip/r50-rebase/README.md` updated; this campaign deleted from `wip/` (no stub).
