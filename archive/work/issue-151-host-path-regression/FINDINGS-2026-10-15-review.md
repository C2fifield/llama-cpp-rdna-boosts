# Issue #151 / env-audit / test-backend-ops — analysis session (2026-10-15)

Working tree: `/home/stew675/llama-issue151` (branch `issue151-review` @ `c6c656f1c`, = the
`~/llama.cpp` `rdna-boosts` tip).  Delivery repo and `~/llama-src` left untouched (another session is
cutting a release).  Builds reused read-only from `~/llama.cpp/build-rocm-hybrid`.

## A. test-backend-ops `MUL_MAT` / `MUL_MAT_ID` (q4_K/q6_K) and the `quantize_mmq_q8_1` fault

**Confirmed on the delivery build (`c6c656f1c`), `HIP_VISIBLE_DEVICES=0`:**

| test | result |
|---|---|
| `-o MUL_MAT    -p type_a=q4_K` | 121 OK / 0 FAIL |
| `-o MUL_MAT_ID -p type_a=q4_K` |  85 OK / 0 FAIL |
| `-o MUL_MAT    -p type_a=q6_K` |  63 OK / 0 FAIL |
| `-o MUL_MAT_ID -p type_a=q6_K` |   4 OK / 0 FAIL |

So q4_K/q6_K are clean.  The failure is in the upstream `dup=1` cases
(`test_mul_mat_id_dup`, added by upstream `073b74034`, "metal : fix mul_mat_id with duplicate ids"):

- `type_a=f16` → wrong result (`ERR ~1.5–1.9`) at n=9/16/33/64 and the 1025/2050-mats cases.
- `type_a=q8_0`, `type_a=q4_0` → n=9,16 OK, then wrong result at n=33, then a GPU
  `Memory access fault` / `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.

**Root cause (already diagnosed):** the CUDA/HIP `mm_ids_helper` (`ggml/src/ggml-cuda/mmid.cu`)
deduplicates `(token, expert)` pairs — it keeps one compact column per pair, so when a token's id row
repeats an expert the un-written `ids_src1` tail is read by `quantize_mmq_q8_1_cuda`; the memory fault
is a symptom, not a separate sizing bug.  The MMQ/MMF column-tile grids were also sized from the token
count.  This is an **upstream** bug, not a delivery regression.

**A prepared fix already exists** (not filed): `upstream/UPSTREAM-PR-mul-mat-id-dup.{md,patch}`
(commit `43a3bd3a6` on `~/llama-base` branch `mul-mat-id-dup-fix`).  Verified here:

- `~/llama-base/build-base` (base `097cf33d9`): `-p 'dup=1'` fails and then faults.
- `~/llama-base/build-mmid` (base + fix): `-p 'dup=1'` **16/16 OK**.

The delivery ships "pure" on this by the documented policy (generic upstream code → `upstream/`).
Record: `archive/work/upstream-mul-mat-id-dup/RESOLUTION.md`; `WORKLOG.md` 261013.003.
Note for porting: the patch is against `097cf33d9`; the delivery's `mmq.cu`/`mmq.cuh` are heavily
modified (617/445 diff lines), `mmid.cu` 82, `mmf.cu` 0.

## B. Issue #151 — code review

### B1. Confirmed delivery-side host cost: the GDN state-gather deferral walk

`ggml_cuda_gdn_state_gather_deferrable()` (`ggml-cuda-try-fuse.cu:341`) is called from the eager node
loop for every `GET_ROWS`.  For each candidate it scans all *remaining* nodes (in the worst case the
whole graph) to find the GDN consumer and prove no other reader / cache writer intervenes.  On a
hybrid GDN model that is `O(n_gdn_layers × n_nodes)` per graph (`~48 × ~1100` on the reporter's 27B).
The walk only runs when graphs are off or a graph is being (re)captured — which is exactly why
`GGML_CUDA_DISABLE_GRAPHS=1` shows the regression and graphs-on hides it.

The contributor (`@DanoPTT`) measured ~1.3 ms/graph and a per-graph root→reader/writer index removes
it (~0.27 ms/graph residual).  That is the right shape for a fix: build one index per
`ggml_cuda_graph_evaluate_and_capture` pass (root→writers, gather→readers) and answer each gather from
it.  Note this fixes only ~0.25 % of the ~1.3 % gap on their box, so it is necessary but not
sufficient.

### B2. The dominant residual tracks the synchronous input-upload path

Only `GGML_CUDA_ASYNC_INPUT=1` moves llama-bench (~+0.6 %, ~40 % of the gap); fusions off does not
change the gap.  `ggml_cuda_async_input_wait()` and the `done_guard` are no-ops when the feature is
off, so the cost is in the synchronous `set_tensor`/copy path itself (r30 predates the v16→v12 reorg,
the pageable bounce, the slab and the extra per-graph hooks).  Attributing the rest needs a **bisect of
the upstream base** (`a55e952b8` → `097cf33d9`) *and* of the block diff — the reporter acknowledged
this.

### B3. `GGML_CUDA_ASYNC_INPUT` should probably be default-on

It is opt-in today only because of *one* historical single-run impurity seen **before** the done-event
ordering was added (`archive/work/decode-host-overhead/README.md`); the hardened ordering closed the
one overwrite window found, and 4 repeat runs (2- and 3-GPU) were byte-identical, with the
prefill-logit gate PASS.  Under the repo's default-on policy ("beneficial and gated → on; env var only
disables") it should be flipped to `GGML_CUDA_ASYNC_INPUT=0` to disable, after re-running the purity
gates on the current tree.  This is the cleanest single answer to the reporter's question 3.

### B4. No other new per-graph cost found

- `ggml_cuda_mmb_begin_graph` / cache clears are cheap when the caches are empty (decode).
- The concurrent-event machinery (`node_to_idx` map, per-node `find`) is dead by default —
  `GGML_CUDA_GRAPH_OPT` must be `1`; the code is present in both bases anyway.
- `gdn_deferred_gathers` `std::find` is `O(48²)` — negligible.
- The last-layer ADD concern was retracted by the reporter; no code path blocks on it in the delivery.

### B5. Graphs-on nondeterminism (issue's 2nd comment)

This reproduces on r30 too, so it is not delivery-specific.  **Corrected 2026-10-16 (issue #151
follow-up):** it was first filed as the unified-KV / attention-offset non-determinism, but the
reporter's follow-up shows it survives `--no-kv-unified` and appears on the first request of a fresh
server, so it is a *separate* HIP-graphs start-to-start effect (TODO #67), not `--kv-unified` cell
offset (issue #134).  Recommend `GGML_CUDA_DISABLE_GRAPHS=1` for strict reproducibility.

## C. Environment-variable read audit

Script: `/tmp/env_audit.py` (classifies every `getenv("NAME")` call site as once-cached vs per-call).
Scope `ggml/src`, `src`, `common`, `include`, `tools`, `examples`.  Result: **~452 vars / 627 sites**
read env, of which most use the good function-local-`static` idiom (and several call sites even carry
the comment "getenv() is expensive on Windows").  The remaining genuinely per-call / per-hot-path
reads:

**On a per-token path:**
| env | site | note |
|---|---|---|
| `GGML_CUDA_MMQ_J_MAX`, `GGML_Q6_COMPACT_J` | `ggml/src/ggml-cuda/mmq.cu:212,231` (`ggml_cuda_mmq_select_J`) | read for **every** MUL_MAT / MUL_MAT_ID; delivery-added |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | `ggml/src/ggml-cuda/ggml-cuda.cu:2788` (`ggml_cuda_mul_mat_cublas`) | every cublas matmul; also lowercases a `std::string` per call; upstream |
| `LLAMA_DROP_COMPUTE_BUFFERS` | `src/llama-context.cpp:1859` (`process_ubatch`) | every ubatch/decode; delivery-added |
| `LLAMA_GRAPH_INPUT_DEBUG` | `src/llama-graph.h:110` (`llm_graph_input_i` ctor) | once per graph **input object** (several per decode); upstream |
| `LLAMA_BATCH_DEBUG` | `src/llama-batch.cpp:16` (`llama_batch_allocr` ctor) | once per batch; upstream |
| `LLAMA_QSA_PLE_HOSTGATHER` | `src/models/qwen4exp.cpp:2912` (graph build) | once per qwen4exp decode; delivery-added |

**Per-allocation / per-reserve (lower frequency, still avoidable):**
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` (`ggml-cuda.cu:171`, `ggml_cuda_device_malloc`),
`GGML_SCHED_STAGE` (`common.cuh:1974`, `h2d_stage_region_bytes`),
`GGML_CUDA_WORK_ALLOC_DEBUG` / `GGML_CUDA_CACHE_GEOM_DEBUG` / `GGML_SCHED_STAGE` debug flags.

**Recommended pattern:** hoist each to a function-local `static const` (as `gdn-conv.cu` / `ple-conv.cu`
already do), or add one helper, e.g.
`static int ggml_cuda_env_int(const char * name, int def)` backed by a function-local
`static std::unordered_map<std::string,int>` — read the CRT env once per name.  For the hot-path set
above the simple `static const` hoist is enough.

## D. Next steps (once the box is quiescent)

1. Bisect the upstream base (`a55e952b8`…`097cf33d9`) at `GGML_CUDA_DISABLE_GRAPHS=1` for the tg gap;
   also A/B `GGML_CUDA_ASYNC_INPUT`, `GGML_CUDA_BOUNCE_MIN=0`, `GGML_CUDA_FA_STAGE_PREALLOC`.
2. Port `@DanoPTT`'s gather index (or an equivalent per-pass index) and re-measure.
3. Re-run the purity gates with `GGML_CUDA_ASYNC_INPUT=1` and, if clean, flip it default-on.
4. Decide on the upstream `MUL_MAT_ID` dup fix (file the PR and/or fold into the delivery).
5. Apply the env-caching hoists; measure on Windows (or at least confirm no behaviour change).
