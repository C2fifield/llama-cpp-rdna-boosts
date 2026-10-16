# `issue-151-host-path-regression` -- RESOLUTION (r68, 2026-10-15)

**Promoted.**  Both candidate fixes plus a log-hygiene change shipped in **r68**
(`v12-097cf33d9-r68`, canonical tip `fc6dcad9f`, net tree
`ffe832696698786865f4adf18216a019b61b160e`).  No new block: **block 09** owns
`moe-expert-cache.cu` and **block 10** owns `ggml-cuda.cu` + `ggml-cuda-try-fuse.{cu,cuh}`, so each
change was folded into the last block that owns the file.

## What shipped

* **Async graph-input uploads default-on** (block 10, `ggml-cuda.cu`).  `g_async_input_on` is now
  `getenv("GGML_CUDA_ASYNC_INPUT") == nullptr || atoi(...) != 0`; the first-hit notice moves from
  `WARN` to `INFO` and names the kill-switch.  This is the issue's own `0001`
  (`GGML_CUDA_ASYNC_INPUT` was opt-in since r46/PR #125 because of a single-run impurity on an early
  build; the done-event ordering hardened it, and r68 re-validated it).
* **Per-graph GDN state-gather index** (block 10, `ggml-cuda-try-fuse.{cu,cuh}` + the node loop in
  `ggml-cuda.cu`).  Replaces the `O(gathers x nodes)` forward walk in
  `ggml_cuda_gdn_state_gather_deferrable`; built **lazily on the first candidate** so a graph with no
  candidate pays only one pointer check per `GET_ROWS`.  `GGML_CUDA_GDN_GATHER_INDEX=0` selects the
  walk, `=2` runs both and `GGML_ABORT`s on a different answer.
* **Per-table arena seed notices to `INFO`** (block 09, `moe-expert-cache.cu`).  The
  `apply_prefill_seed_locked` `prefill-...` line and the `apply_heat_seed_locked` `heat-seeded ...`
  line printed once per expert table at load.  ggml has no `TRACE` level; `GGML_LOG_INFO` is the
  `-lv 4`-only level (`common_log_get_verbosity(GGML_LOG_INFO) == LOG_LEVEL_TRACE`), so they are gone
  at the default `-lv 3` and visible at `-lv 4`.

## Why the index is correctness-safe

The index computes each gather's fields over the **whole** graph (a reader before the gather still
counts as `other_reader`; every GDN that reads the gather counts toward `n_gdn`), while the walk only
sees the nodes **after** the gather.  So an index/walk disagreement is one-directional: the index can
only answer `false` where the walk answers `true` -- a dropped deferral, i.e. the ordinary un-skipped
`GET_ROWS` (bit-identical), never a skip the walk would have refused.  `=2` confirmed exact agreement
on every real graph tested.  The reporter's 1-in-12 index-arm output difference is therefore not the
index changing a decision; it is consistent with the tree's own numeric drift (their r56 vs r30 draft
acceptance already differs, and their graphs-on near-tie reproduces on r30).

## Measurements

| | host per graph | tg128 | prefill pp4096 |
|---|---:|---:|---:|
| walk + synchronous (old defaults) | 4.97 ms | 24.600 | 1077 t/s |
| index + async (r68 defaults) | 4.34 ms (index only) | 24.659 (+0.24 %) | unchanged |

5 interleaved `-r 10` rounds on gfx1201 / 27B Q6_K / graphs off; the r68 build reproduced it over 3
rounds (24.593 -> 24.650, +0.23 %).  Async removes the per-copy `cudaStreamSynchronize`, which is
outside `graph_compute` and so absent from the `GCDBG` timer; its own tg delta is +0.21 % locally.  On
the reporter's box async measured **+1.4 %** over the index arm (+0.5 % over their r30 reference) and
the index **+0.3 %**.

## Gates (r68 build, `fc6dcad9f`)

* `gate-prefill-logits.sh` **PASS** -- mean KLD `0.000707`, same-top-p `98.755 %`.
* 4B Q8_0 same-seed greedy `7386359e5dac` (identical to r67), unchanged across
  `{graphs on/off} x {r68 defaults / both kill-switches}`.
* Adaptive-MTP `none == draft-mtp n3 == n7`, prose `n 3000`, byte-identical (`1a80de87f2b6`).
* `GGML_CUDA_GDN_GATHER_INDEX=2` clean on 4B Q8_0, 27B Q6_K, 27B Swift UD-Q5_K_M, 35B-A3B
  (`-ncmoe 99` single GPU and `-sm tensor -ncmoe 20` 2 GPU), Gemma4 E4B (non-GDN), with graphs on and
  off, and with `--spec-type draft-mtp` (the 2-8-token verify band).
* 2-GPU `-sm tensor` and `-sm tensor -ncmoe` same-seed coherent.
* `validate-set.sh` green on a fresh tarball (strict 12/12 `git am`, applied tree == `release.json`).

## Residual

The report's remaining ~0.4 % is unattributed upstream drift (`a55e952b8` -> `097cf33d9`); it was not
bisected.  Also noted: `ggml_cuda_try_fuse` is now the dominant host cost (~84 %), which is **not** a
delivery regression -- a separate campaign if it is ever worth attacking.
