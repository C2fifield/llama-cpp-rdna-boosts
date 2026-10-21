# RESOLUTION (2026-10-20, r100) — the split-state pre-warm walk no longer allocates per node

**Status: FIXED and shipped in `v12-097cf33d9-r100` (block 09).**  This is the second phase of issue #184.
The first phase (2026-10-18) answered the reported r80 -> r82 3.5 % 4-stream regression: not reproducible
here (<= 0.4 %, sign unstable, 224 B of `.text`, no inlining change) — see `README.md` and
`COMMENT-2026-10-18.md`.  The reporter then took the cold-helper direction himself and, in
[#issuecomment-6291554091](https://github.com/stew675/llama-cpp-rdna-boosts/issues/184#issuecomment-6291554091),
measured the remaining cost at the right place: **the walk's containers**, not the dispatcher.

## What the reporter measured (r91 + his patches, GCC 13.3 host, 2x R9700)

* ~230 graph re-allocations per run; `ggml_backend_sched_alloc_graph` averages 21 ms (33-130 ms for the
  target graph) = ~10 % of the run's wall time, 25-40 % in the seconds where streams start or finish.
  96 % of `ggml_gallocr_alloc_graph` is `ggml_backend_meta_get_split_state` via
  `ggml_backend_meta_buffer_init_tensor_impl`.
* Per re-allocation: ~2800 pre-warm walks, ~150k tensors, ~4700 split states, ~165k cache hits,
  ~6 version switches and 1 clear.
* **94 % of the `malloc` calls under the allocation came from `wseen`'s inserts** — one heap node per
  visited tensor plus the rehash and the frees.
* `wstack`/`wseen` on a `std::pmr::monotonic_buffer_resource` over a reused 1 MiB per-thread buffer:
  re-allocation **21.1 -> 16.0 ms**, four concurrent streams **+5.0 %** (5 of 6 pairs), TTFT on short
  follow-up turns **-26 ms**, decode and prefill unchanged, greedy output identical.  Their proposal was
  +10/-2 lines.

Two things they ruled out, so we did not repeat them: shrinking `ggml_backend_meta_split_state` to a
quarter (`GGML_BACKEND_META_MAX_DEVICES` 4) and reusing `calculate_split_state`'s `src_ss` vector per
thread - both unchanged.

## What shipped

`ggml/src/ggml-backend-meta.cpp` (block 09), 22 lines: `wstack` and `wseen` are `std::pmr::vector` /
`std::pmr::unordered_set` over a reused `thread_local` 1 MiB `monotonic_buffer_resource`, released at the
top of each walk.  `GGML_META_SS_SCRATCH=0` selects the default (heap) resource for both containers, so
the two arms are one binary and one env.

Why it cannot change a computed value: only the outermost call walks (`ss_warm_active`), so the resource
is never re-entered; the walk pushes/pops the same tensors in the same order (the containers' iteration
order is never read); a walk that visits more tensors than 1 MiB holds falls back to the resource's
upstream allocator, so nothing is capped; and the split states are the same values.  Verified directly
with `GGML_META_SS_VERIFY=1` (recompute every outermost cache hit, abort on a stale entry): no abort on
27B plain, 27B MTP n3 or 35B-A3B `-sm tensor -ncmoe`.

## Verification (gfx1201, 3x R9700, ROCm 10.1.0, release tree `4775fc92`)

A/B by swapping only `libggml-base.so.0.26.0` between the r99 build and the r100 build.

| gate | result |
|---|---|
| 4B Q8_0, 3 GPU tensor, seed 42 | `1c5d32ac537d` == r99 |
| 27B UD-Q4_K_XL, 2 GPU tensor, q8_0 KV, 200 tok greedy, plain / `draft-mtp` n3 / `draft-mtp` + `GGML_META_SS_VERIFY=1` | all `5067a0089fc1` == r99 (plain == MTP, so greedy purity holds) |
| prefill-logit KLD (`gate-prefill-logits.sh`) | mean **0.000707**, same-top-p **98.755 %** (== r99, PASS) |
| `test-backend-ops -o RMS_NORM,ADD,MUL_MAT,CPY,CONCAT,SSM_CONV` on ROCm0+ROCm1 | 6262 OK, 0 FAIL, 3/3 backends |
| per-block build (blocks 09-11) | 3/3 |
| `scripts/validate-set.sh` | green, tree `4775fc92` |
| `check-gpu-faults.sh` | every r100 log clean; no journal fault during any r100 run |

Perf, `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, 27B UD-IQ3_S, 3 GPU tensor, q8_0 KV, total t/s,
3 interleaved rounds (stock r99 -> r100):

| B | stock | r100 | change |
|---|---|---|---|
| 1 | 61.37 / 61.26 / 61.35 | 62.34 / 62.40 / 62.33 | **+1.6 %** |
| 4 | 201.68 / 201.56 / 201.29 | 202.46 / 202.82 / 202.89 | **+0.6 %** |
| 8 | 318.08 / 318.01 / 317.61 | 319.38 / 320.03 / 319.21 | **+0.5 %** |

Server A/B (Flash-Next IQ3_XXS, 2 GPU tensor `-ncmoe 47`, MTP n3, the reporter's shape, `ab2.py` /
`ab_mixed.py` in this directory): pre-warmed pure decode 4 streams **+0.29 %** (3 pairs), 4 concurrent
chats with the prefills inside the window **+0.36 %** mean over 7 pairs (paired -3.5 .. +4.9 %).

**Our effect is smaller than the reporter's +5 % for a measurable reason**: his workload re-allocates
~230 times per run, this box's config reports `graph reallocs = 1-5` per request (the pre-warmed harness
sees 24/18 over a whole session), so the walk is a correspondingly smaller share of wall time.  The
change is shipped because it is neutral-to-positive on every gate here, removes ~4.7k heap nodes per
re-allocation, and is the reporter's own measurement on the shape that has the churn.

## Remaining headroom (not taken, recorded)

Their last paragraph: ~165k cache hits per re-allocation, 30+ per node, each a hash find plus a 336-byte
`memcmp`.  Hashing the tensor bytes to skip the `memcmp` would cost about as much as the `memcmp`
(336 B is ~11 vector words), so the lever is the lookup *count* (recursion/pruning structure), not the
comparison.  Not measured, not attempted.

## Repro

```bash
# the two arms are one binary: GGML_META_SS_SCRATCH=1 (default) vs =0
GGML_META_SS_VERIFY=1 <cmd>          # proves the states are identical (aborts otherwise)
python3 archive/work/issue-184-meta-r80r82/ab.py 3 '{}' pmr S,P
```
