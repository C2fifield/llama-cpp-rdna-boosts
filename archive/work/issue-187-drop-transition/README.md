# archive/work/issue-187-drop-transition

**Status: CLOSED — fixed in `v12-097cf33d9-r94` (block 10).**  The candidate below is the patch that was
folded in; the released fix is in `patches/0010-…-qwen4exp-…` (the work-boundary shrink now calls
`ggml_cuda_graph_mem_freed`).  The release record is `WORKLOG.md` 261020.004; the resolution note is
[`RESOLUTION.md`](RESOLUTION.md).

Issue: <https://github.com/stew675/llama-cpp-rdna-boosts/issues/187> (`@briansp2020`).
Candidate patch: [`candidate-work-shrink-invalidate.patch`](candidate-work-shrink-invalidate.patch)
(15 lines, `ggml/src/ggml-cuda/ggml-cuda-slab.cu`, block 10's file).

## Symptom

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + the shared Q8_0 MTP head, 2x R9700, `-sm tensor --n-cpu-moe 10`,
MTP `--spec-draft-n-max 3`, 4 concurrent 300-token greedy chats.  After the first slot of the second
round is released the server aborts on both devices:

```
Warning: Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: ...
... [GPU index: 1, kernel: void k_set_rows_q8_0_group<long>(...) grid=[256,1,1] ...]
```

Reproduced 2/2 here on the r93 delivery tree (`v12-097cf33d9-r93`, tree `260e6f49f4`), first round
transitions; the fault lands on the second or third decode step after the drop.  `-sm layer`, one
stream, multi-turn/`cache_prompt` did **not** reproduce here on r93 (stock or fixed), although the
`slab work region has no room for 512.00 MiB ... falling back` warnings from the report do appear — see
"Not reproduced" below.

Reproduce:

```
cd archive/work/issue-187-drop-transition
./repro.sh --tag <name>              # 4 concurrent streams x 3 rounds, -sm tensor, 2 GPUs
./repro.sh --sm layer --np 1 --tag <name>   # needs DRIVE=turns, see below
DRIVE=turns ./repro.sh --sm layer --np 1 --tag <name>   # 6-turn x 2-cycle multi-turn chat
```

`repro.sh` starts the server with the issue's exact command (2 visible GPUs, `-t 8`), runs
`drive.py`/`drive_turns.py`, then `scripts/check-gpu-faults.sh --since "10 min ago" --log <log>`.
`BIN=` selects a different build (used for the #189/#186 overlay run).

## Root cause (confirmed with the logs)

The fault is **a replayed captured HIP graph whose baked compute-buffer pointers were released by the
r87 post-prefill drop**, not the drop condition itself.

1. The r87 forced-drop hunk fires on a MTP *verify -> decode* transition (measured `n_prev=12/16,
   cur=1792 MiB, was=5568 MiB`): it is not a prefill -> decode transition, but it is harmless and it
   re-records `slab_narrow_boundary` at the real narrow size.
2. That un-suppresses a later **legitimate** wide -> narrow drop (`cur=5568 MiB` vs `was=1792 MiB`,
   i.e. the width compare alone fires).  The drop calls `ggml_backend_sched_drop_buffers()`, which frees
   the wide compute view, and the narrow re-reserve then shrinks the slab work boundary
   **5568 -> 1792 MiB**.  The bytes above the new boundary go back to the MoE arena.
3. Nothing invalidates the backend's captured graphs at that point: `ggml_cuda_slab_work_release()` /
   the shrink branch of `ggml_cuda_slab_work_alloc()` never bump `ggml_cuda_graph_mem_gen`, which is the
   mechanism that recaptures "a graph whose pool temporaries or scratch buffers were freed since its
   capture".  A later replay therefore uses stale addresses for the graph's temporaries (here the KV
   `SET_ROWS` source indices, which live in the compute buffer) and faults.
4. Why the reporter's revert "works": it stops the r87 forced drop at step 1, so
   `slab_narrow_boundary` stays poisoned at the **wide** size (`max(re-reserved, drop_observed)` with
   `drop_observed` = the wide prefill region), the width compare at step 2 never fires, and no drop (and
   no stale replay) happens.  That is exactly the r85 regression the r87 fix repaired: the wide layout
   and the small arena stay around for the next request (second request at ~half speed).

Evidence:

* `GGML_CUDA_DISABLE_GRAPHS=1` on the **unfixed** tree: 3/3 rounds clean, same five drops and three
  re-arms.  So the drop/re-arm machinery itself is sound; only a graph replay is unsafe.
* Instrumented decision log (`LLAMA_DROP_DEBUG=1`, scratch build): the fault is preceded by
  `DROP_DECIDE ... n_prev=12 tok=8 ... cur=5568.0 MiB was=1792.0 MiB drop_now=1` ->
  `moe_cache_rearm: re-armed 54 ... (arena now 9475.0 MiB)` -> two clean decode steps -> aperture
  violation.  (The `was=5568` on the following steps is the r85 recorded-boundary behaviour; not the
  fault.)
* `k_set_rows_q8_0_group<long>` grid `[256,1,1]`/`[1024,1,1]` is the q8_0 K/V cache write; its
  `src0`/`src1` (Kcur and the position indices) are graph-allocated temporaries in the slab work region.

## The fix

In the `target < s.boundary` shrink branch of `ggml_cuda_slab_work_alloc()`, call
`ggml_cuda_graph_mem_freed(device)` before returning the bytes to the arena.  That is the existing
per-device invalidation: the next replay of any graph captured against the old boundary is recaptured
(its `graph->mem_gen` no longer matches), exactly as for the pool / FA-staging / H2D-ring frees.  The
`GGML_CUDA_WORK_ALLOC_DEBUG` diagnostic prints the shrink and the invalidation.

No delivery contract changes: the drop condition and `slab_narrow_boundary` bookkeeping are untouched,
so r87's second-prefill repair (and r85's narrow band) keep working.  The invalidation fires only when
the boundary actually shrinks (6 events over the whole 3-round run; 5 drops, 3 re-arms), not per token,
so steady-state graph replay is unaffected.

## Measured

| arm | result |
|---|---|
| r93 (unfixed) 4 streams | **crash** 2/2, `k_set_rows_q8_0_group`, `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` |
| r93 (unfixed) + `GGML_CUDA_DISABLE_GRAPHS=1` | clean 3/3 rounds |
| candidate fix, 4 streams | **clean 3/3 runs** (9 rounds), `check-gpu-faults.sh` and the kernel journal clean |
| candidate fix, `-sm layer` 1 stream, 6-turn x 2-cycle | clean, identical turn times to unfixed (3.66/2.01/1.88/2.05/1.06/1.41 s vs 3.70/2.02/1.90/2.06/1.08/1.44 s) |
| coherence (4B Q8_0, 3 GPU, seed 42) | `sha=1c5d32ac537d`, identical to r93 |
| prefill-logit gate | mean KLD **0.000707** / same-top-p **98.755 %** — PASS, identical to the recorded r93 base |
| overlay: candidate fix + PR #189 + PR #186 | builds, 4 streams clean 3/3 rounds, visibly faster (the PRs' TTFT/decode wins) |

## Overlay PRs (#189, #186)

* **#189 `wip/state-read-batch`** batches the *context-checkpoint host reads*; it touches
  `llama_io_write_host`, `state_stage`/`state_backend` and the meta `get_tensor_async` split arithmetic.
  It is orthogonal to this fault (no work buffers, no graph capture, no drop) but it **does not compile
  on the r93 tree as submitted**: `std::find` in `llama_state_sync_all` needs
  `#include <algorithm>` in `src/llama-context.cpp` (it presumably arrived transitively on the r91 tree
  the PR was cut against).  With that one line added, the fix + #189 + #186 build and the #187 repro is
  clean.  Finding reported to the reporter.
* **#186 `wip/sched-moe-dup-consumer`** (the scheduler patch the report says "also hit the crash") was
  carried in the same overlay run and is clean on top of the fix.  It is a WIP patch, not part of the
  delivery; nothing here folds it.

## Not reproduced / open

* `-sm layer`, one stream, multi-turn with `cache_prompt: true`: clean on **both** stock and fixed r93
  (2 cycles x 6 turns), with the report's "no room for 512.00 MiB" warnings.  r90's
  `set_tensor_async` UAF fix landed after r89, so the reporter's `-sm layer` arm may already be gone on
  r93; ask for a retest / the exact turn sequence and token counts before claiming it.
* TODO #70 (harden a partial/refused re-arm so the cache is left wholesale stood down) is the same
  family — a captured graph can also bake stale arena table addresses — but is not proven by this
  repro; this fix invalidates per drop, which should cover the observed shape.  Keep #70 open.
* Decision: **done** — folded into **block 10** (the block that introduced the shrink branch) and shipped
  in `v12-097cf33d9-r94`.  The same release repaired the per-block gate's pre-existing boundary splits
  (blocks 02/06/09/10/11); see `RESOLUTION.md` and `WORKLOG.md` 261020.004.  No scratch debug logging was
  shipped: the patch adds only the gated `GGML_CUDA_WORK_ALLOC_DEBUG` diagnostic.

## Files

* `repro.sh` — server + workload + fault check (issue config; `--sm`, `--np`, `--rounds`, `--tag`,
  `BIN=`, `DRIVE=turns CYCLES= TURNS= MAXTOK=`).
* `drive.py` — rounds of N concurrent greedy chat completions.
* `drive_turns.py` — multi-turn chat with `cache_prompt: true` (the `-sm layer` shape).
* `candidate-work-shrink-invalidate.patch` — the candidate against the r93 tree.
* `runs/` — the reproduction/fix logs (kept for the record; large).
