# RESOLUTION — issue #187

**Fixed in `v12-097cf33d9-r94`** (block 10, `ggml/src/ggml-cuda/ggml-cuda-slab.cu`), tree
`40d43e3044051ea8168c02cd1a899877b5d8574d`.  Release record: `WORKLOG.md` 261020.004.

## What was wrong

`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `k_set_rows_q8_0_group` on a 4-stream MTP
`-sm tensor -ncmoe 10` decode (2x R9700).  It is **not** the r87 drop condition: the r87 forced drop only
un-suppresses a legitimate prefill -> decode drop.  That drop releases the wide compute view and the narrow
re-reserve **shrinks the movable-boundary slab's work region** (5568 -> 1792 MiB measured), returning the
bytes above the boundary to the MoE arena.  `ggml_cuda_slab_work_alloc`'s shrink path did not bump
`ggml_cuda_graph_mem_gen`, so a captured HIP graph was replayed with addresses in the released region (the
KV `set_rows` source indices are graph temporaries) and faulted two or three decode steps later.

The reporter's r87-hunk revert hides the crash only because it leaves `slab_narrow_boundary` recorded at
the **wide** size, so the legitimate drop never fires — which restores the r85 regression (the second
request keeps the wide layout and a small arena).  That is not the fix.

## The fix

The shrink branch in `ggml_cuda_slab_work_alloc` now calls
`ggml_cuda_graph_mem_freed(device)` before returning `[target, old)` to the arena: the next replay of a
graph captured against the old boundary is recaptured (the existing mem-gen mechanism, already used for
the pool / FA-staging / H2D-ring frees), and a `GGML_CUDA_WORK_ALLOC_DEBUG` line reports the shrink.  The
drop condition and `slab_narrow_boundary` bookkeeping are untouched, so r87's second-prefill repair stands.
It fires only on real shrinks (6 events / 5 drops over a full 3-round repro run), never per token.

## Evidence

| arm | result |
|---|---|
| r93 (unfixed) | crash **2/2**, `k_set_rows_q8_0_group`, same grid as the report |
| unfixed + `GGML_CUDA_DISABLE_GRAPHS=1` | clean 3/3 with the same drops/re-arms — the drop/re-arm machinery is sound, only a replay is unsafe |
| fixed | clean **3/3 runs** (9 rounds); `check-gpu-faults.sh` and the kernel journal clean |
| coherence (4B Q8_0, 3 GPU, seed 42) | `sha=1c5d32ac537d`, identical to r93 |
| prefill-logit | mean KLD 0.000707 / same-top-p 98.755 %, identical to the recorded base |
| `-sm layer` 1 stream, 6-turn x 2-cycle multi-turn | clean on both stock and fixed r93; the report's arm did **not** reproduce here (r90 landed after r89 — ask for an r94 retest) |

## Same release: the per-block boundary repair

Running the mandatory per-block build gate exposed that the set had not built block-by-block since
r58/r86.  Every defect was a split the tip hides; the repair is **tree-identical** (net tree unchanged):
block 02 gains the embeddings-layer-input preset API; block 06 drops the stray `}`/duplicate line that
block 10 used to remove; block 09 gains the complete device-iface fills, the `ggml_cuda_pool_wide_pin` /
`ggml_cuda_slab_seed_floor` uses, and the whole loaded-draft handoff (`load_model`, the 4-argument
ctor/`init_from_params`, `take_draft_model`).  Coverage: blocks 00-05 pass, 06-08 pass after the block-06
repair, 09-11 pass after the block-09 repairs (each block's tree is unchanged since its passing run).
`--changed-from` is not valid for this kind of repair — it compares each commit's own delta and skips
blocks whose tree changed but delta did not.

## Review notes carried out of the investigation

* **PR #189** (`wip/state-read-batch`) is orthogonal to this fault (context-checkpoint host reads; no work
  buffers, no graph capture, no drop).  It does not compile on the r93 tree as submitted: `std::find` in
  `llama_state_sync_all` needs `#include <algorithm>` in `src/llama-context.cpp`.  With that line, r94 +
  #189 + #186 built and the #187 repro ran clean 3/3.
* **PR #186** (`wip/sched-moe-dup-consumer`, the scheduler patch the report says also hit the crash) is
  clean on top of the fix; it stays WIP.
* **TODO #70 stays open**: a re-arm can still change a table's arena placement under a graph that is not
  being dropped.  This fix covers the drop-shaped staleness, not that one.
