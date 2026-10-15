# issue-151-host-path-regression

**Status: OPEN — seeding + scoping (2026-10-15).**  Nothing in this directory is part of the delivery,
and nothing here may be applied to the fork without the maintainer's explicit go-ahead (see the WIP and
promotion rules in `../../AGENTS.md`).

**Issue:** https://github.com/stew675/llama-cpp-rdna-boosts/issues/151
**Reporter:** `@DanoPTT` (R9700/gfx1201, **Windows 11**, Ryzen 5 PRO 4650G, Swift Qwen3.8-27B Q6_K /
Q5_K_M dense hybrid GDN).

---

## 1. What the issue is

With HIP graphs **off** (their production setting), dense decode on the r56 delivery is **1.3–2.4 %**
slower than their r30 reference.  The GPU does the same or *less* work; the extra cost is host-side, and
graphs-on hides nearly all of it, so it is the eager launch path.

| | P = `v16-a55e952b8-r30` | S = `v12-097cf33d9-r56` |
|---|---|---|
| `llama-bench tg128` Q6_K, graphs off | 24.277 / 24.274 | 23.979 / 23.944 (**−1.3 %**) |
| graphs on | 24.400 | 24.340 (**−0.25 %**) |
| host per `graph_compute` (`GGML_CUDA_GCDBG=1`) | 9.21 ms | **10.17 ms** |
| `ev_rest` − `fuse` − `fwd` (their 129-graph run) | ~10 ms | **~181 ms (~1.4 ms/graph)** |
| unfused last-layer `ADD attn_residual-63` | (fused) | 6 µs GPU / **256 µs host** |

Reporter's own narrowing (see the issue comments):

- `ggml_cuda_gdn_state_gather_deferrable` is the only addition to the eager node loop between r30 and
  r56, and it is `O(gathers × nodes)`: for every recurrent-state `GET_ROWS` it scans all remaining
  nodes.  48 GDN layers × ~3850 nodes ⇒ ~90k node visits/token on their box.
- Their **index patch** (2 commits, ~160 lines, `ggml-cuda-try-fuse.cu/.cuh` + the loop in
  `ggml-cuda.cu`; `GGML_CUDA_GDN_GATHER_INDEX=0` keeps the walk, `=2` runs both and aborts on
  disagreement) recovers **+0.3 %** — about a quarter of the gap; the index build itself costs
  ~0.27 ms/graph.
- `GGML_CUDA_ASYNC_INPUT=1` recovers **~40 % of the gap (+0.6 %)** — the only switch that helps.
- The remainder (~0.4 %) is possibly upstream drift (`a55e952b8` → `097cf33d9`); the reporter has not
  bisected that.
- Their graphs-on nondeterminism (issue's 2nd comment) reproduces on **r30** too, so it is not delivery
  specific; it matches the documented unified-KV / cell-offset non-determinism
  (`../kv-cell-offset-nondeterminism/`, issue #134) — test with `--no-kv-unified`.

---

## 2. What we measured on the maintainer's box (2026-10-15, quiet)

gfx1201 (R9700, 32 GiB), Linux, `/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf`, 1 GPU,
`-fa 1 -p 0 -n 128 -r 10 -ngl 99 -t 8`, `GGML_CUDA_DISABLE_GRAPHS=1`:

| arm | tg128 t/s |
|---|---|
| r65 (`build-rocm-hybrid`) | 24.45 ± 0.02 |
| r66 (`build-rocm-r66`, dup fix + env-read caching) | 24.43 ± 0.01 |
| r66 + `GGML_CUDA_ASYNC_INPUT=1` | 24.42 ± 0.00 |

**All identical.**  `GGML_CUDA_GCDBG=1`, `-n 64 -r 1`, 65 graphs / 250055 nodes:

```
CUDAGC calls=65 nodes=250055 host_total=302.8ms per_node=1.2us per_call=4658.3us
CUDAGC_OPS ADD=10.4ms/65(160.2us) MUL_MAT=7.4ms/7410(1.0us) GET_ROWS=6.8ms/3250(2.1us) ...
CUDAGC_BREAK pre=0.0ms ev_pre=0.0ms ev_rest=303.7ms | fuse=228.0ms/72277 fwd=41.0ms/18197
```

- Host per graph **4.66 ms** (their box: ~10 ms; ~2× slower host).
- `ev_rest − fuse − fwd` ≈ **0.53 ms/graph**.
- The GDN walk isolated by the kill-switch: gather-on `host_total` **302.8 ms** vs
  `GGML_CUDA_FUSE_GDN_STATE_GATHER=0` **283.0 ms** ⇒ **the walk costs ~0.30 ms/graph** (~0.7 % of a
  ~41 ms token).
- The skip still wins: gather-**on** 24.40 vs gather-**off** 23.99 t/s in the `-r 5` run.

**Conclusion so far: #151 does not reproduce on this box.**  The token is GPU-bound (~41 ms GPU vs
~4.7 ms host), so ~0.3 ms/graph of host work is invisible to tg.  The reporter's host is the bottleneck
(~10 ms host/graph, and likely much less compute-per-node), which is why the same code costs them
1.4 ms/graph.  We do **not** have `v16-a55e952b8-r30` as a binary, so the r30 baseline cannot be
bisected locally.

---

## 3. What is already shipped (r66) and what is not

Shipped in `v12-097cf33d9-r66` (see `../../WORKLOG.md` 261015.003):

- **Env-read caching** — every hot-path env read is resolved once (per matmul / malloc / graph input /
  batch).  This is the Windows-relevant half of #151 (the reporter is on Windows, where `getenv()`
  locks and rescans the environment block).  It is measured neutral on Linux.
- **MUL_MAT_ID duplicate-ids fix** — unrelated to #151 (correctness; `test-backend-ops dup=1`).
- **`-Wshadow` / `-Wmissing-field-initializers` sweep** — unrelated.

**Not shipped / not done:**
- The GDN gather **index** (the reporter's `+0.3 %`).
- `GGML_CUDA_ASYNC_INPUT` default-on (their `+0.6 %`; repo policy says a gated beneficial feature should
  be default-on, but it was deliberately kept opt-in after a historical single-run impurity — see
  `../../archive/work/decode-host-overhead/README.md`).
- Any upstream-base bisect (`a55e952b8` → `097cf33d9`).

---

## 4. Candidate work, in priority order

1. **Port the gather index.**  Replace the forward walk in
   `ggml_cuda_gdn_state_gather_deferrable` (`ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`, owner **block
   10** in the V2.0 set) with one index built per `ggml_cuda_graph_evaluate_and_capture` pass (gather →
   its GDN + other readers; cache root → writers), and add `GGML_CUDA_GDN_GATHER_INDEX` with `=0`
   (walk) / `=2` (both, abort on disagreement).  Validate correctness with `=2` on a GDN model and a
   same-seed coherence run.
   **Caveat:** on this box the walk is only 0.30 ms/graph, so the index build may net ~0 here; the win
   is on the reporter's (slow) host and on larger GDN models.  It cannot be perf-verified locally —
   ask `@DanoPTT` to re-run their matrix.
2. **Decide `GGML_CUDA_ASYNC_INPUT`.**  Re-run the purity gates on the current tree with it on
   (`scripts/gate-prefill-logits.sh`, adaptive-MTP `-n 3000`, 4B coherence) and, if clean, flip the
   default (`GGML_CUDA_ASYNC_INPUT=0` to disable).  It is neutral here; it is the single biggest lever
   on the reporter's box.
3. **Bisect the upstream base** by building `a55e952b8` and the delivery's r30-equivalent.  Note: r30 is
   `v16-a55e952b8-r30`, a *different* (pre-V2.0) tree; the old 16-block set is archived under
   `../../archive/` and the branches `baseline/*` / `block/*` in `~/llama.cpp` are historical.  This is
   the only way to attribute the residual ~0.4 %.
4. Optional: a git-history search for any other per-node/per-graph host addition between r30 and r56
   (the `ADD` at 160 µs/op host is measured here; the reporter saw 256 µs and it is a relocated wait,
   not new cost).

---

## 5. Exact reproduction (this box)

```bash
# current build (r66)
cd /home/stew675/llama-fold && export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf
HIP_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 \
  ./build-rocm-r66/bin/llama-bench -m $M -fa 1 -p 0 -n 128 -r 10 -ngl 99 -t 8

# host instrumentation
HIP_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GCDBG=1 \
  ./build-rocm-r66/bin/llama-bench -m $M -fa 1 -p 0 -n 64 -r 1 -ngl 99 -t 8 2>&1 | grep -E 'CUDAGC'

# A/B arms
GGML_CUDA_FUSE_GDN_STATE_GATHER=0   # walk off AND gather runs
GGML_CUDA_ASYNC_INPUT=1             # async graph-input uploads
```

Rebuild the fold worktree after any change: `cd /home/stew675/llama-fold && BUILD_DIR=build-rocm-r66
EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`.

**Gotchas (same as `../../AGENTS.md`):** never run benches in parallel; size `-t` to leave the GPU-IRQ
cores free (`/usr/local/bin/pin_gpu_irqs.sh`; one CCD = 8 is within 1–2 % of best); `llama-cli` needs
`--single-turn`; the reporter's box is Windows and much slower — do not equate our neutral result with
theirs.

---

## 6. Key code / records

| what | where |
|---|---|
| the walk | `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` — `ggml_cuda_gdn_state_gather_deferrable` |
| the deferral, call site | `ggml/src/ggml-cuda/ggml-cuda.cu` — `gdn_deferred_gathers`, `GGML_CUDA_FUSE_GDN_STATE_GATHER` |
| the consumer | `ggml/src/ggml-cuda/gated_delta_net.cu` — `s_ids`/`s_row_stride` arm |
| the reporter's index patch | https://github.com/DanoPTT/llama.cpp/compare/f711-rdna-r56-stack1014...f711-rdna-r56-stack1014-gathidx |
| async-input origin + the single-run impurity | `../../archive/work/decode-host-overhead/README.md`, `RESOLUTION.md` |
| the GDN state-gather skip | `../../archive/work/gdn-state-gather-skip/README.md` |
| unified-KV nondeterminism (graphs-on) | `../kv-cell-offset-nondeterminism/` (issue #134) |
| the issue's own measurements | https://github.com/stew675/llama-cpp-rdna-boosts/issues/151 |

## 7. Open questions

1. Is the index worth shipping if it is neutral on our box (slow-host/large-model win only)?  The
   contributor's patch exists and is validated on their hardware; the cheapest path is to have them
   re-test a delivery build with it folded.
2. Should `GGML_CUDA_ASYNC_INPUT` be default-on?  Needs a fresh purity pass on the current tree and the
   maintainer's risk call.
3. Does the residual ~0.4 % come from upstream `a55e952b8` → `097cf33d9`?  Requires building both bases.
4. Is the reporter's 4650G host path representative enough to gate the delivery on, or does this stay a
   documented slow-host improvement?
