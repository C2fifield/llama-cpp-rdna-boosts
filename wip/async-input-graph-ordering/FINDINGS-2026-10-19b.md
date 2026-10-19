# wip/async-input-graph-ordering — second session (2026-10-19, session 2)

Same day, second debugging session.  Read [`HANDOVER.md`](HANDOVER.md) and
[`FINDINGS-2026-10-19.md`](FINDINGS-2026-10-19.md) (session 1) first; this file supersedes session 1's
*hypothesis* section and its "next experiments" 1-3, and keeps session 1's eliminated list intact.

Build: r89 + `tools/DEBUGBUILD.patch` + `tools/DEBUGBUILD2.patch` (`~/llama.cpp` branch `ain-debug`,
commits `dceb26d38` + `dcf22dfce`, `BUILD_DIR=build-ain`).  Same instrument as session 1
(`llama-perplexity --kl-divergence-base` self-comparison, 4B Q8_0, `-c 512 --chunks 12 -ub 8 -b 8`,
`-ngl 99 -fa on -t 8 -ctk/ctv q8_0`, gfx1201).

Arms run in this session, same binary, same session (each arm = 2 passes):

| arm | env | same top p |
|---|---|---|
| async, divergent control | `ASYNC_INPUT=1 BATCH=0` | 96.6 / 98.8 (repeat) |
| async + host sleep | `+ AIN_SLEEP_US=20` | **99.967** (floor) |
| async off | `ASYNC_INPUT=0` | **99.967** |
| no env (default) | — | **99.967** |
| async + f16 KV | `-ctk/-ctv f16` | 96.2 |
| async + copies on the compute stream | `+ AIN_MAIN_COPY=1` | 95.6 / 95.7 |
| async + wait captured inside the graph | `+ AIN_CAP_WAIT=1` | 97.9 / 96.6 |
| async + per-graph cache frees disabled | `+ AIN_NO_FREE=1` | 96.8 / 98.3 |
| async + Q8_1 cache disabled | `+ AIN_Q81_NOCACHE=1` | 97.6 / 98.5 |
| async + Q8_1 verify fold off | `+ GGML_CUDA_FUSE_Q8_1_VERIFY=0` | 96.6 |
| async + GLU Q8_1 fold off | `+ GGML_CUDA_FUSE_GLU_Q8_1=0` | 96.3 |
| async + one tiny kernel before the launch | `+ AIN_VK=1` | **99.967** |
| async + 9-12 tiny hash kernels inside the graph | `+ AIN_HK=...` | 97.1-98.7 |

## What is established (new; do not re-derive)

1. **Split 2 is not a CPU-side consumer — the hand-over is at the FRONT of the graph.**  For the 4B
   reproducer `split 0 = CPU`, `nodes = [0,2)` = two `GET_ROWS` (`model.input_embed`, `node_1`) writing
   into a `ROCm_Host` buffer; `split 1 = ROCm0`, `nodes = [2, 2025)` (2023 nodes, first node `node_2
   (DUP)`, last `result_output (MUL_MAT)`), `inputs = 12`.  Every split input is a **device** copy
   (`ROCm0#<name>#0`).  So session 1's H-A ("a CPU-side split 2 reads the first split's output before the
   graph produced it") is **dead as written**: the CPU produces the graph's inputs, it does not consume
   the graph's output, and it runs synchronously on the host before the copies.
   *Aside:* `GGML_SCHED_DEBUG=1` **does** work, but the `## SPLIT #` lines are `GGML_LOG_DEBUG`, which
   needs **`-lv 5`** (not `-lv 4` — `-lv 4` prints nothing, which is why session 1 saw no output).

2. **The CUDA split contains no host (`ROCm_Host`) tensor at all.**  A whole-graph scan of the
   `ROCm0` split (all 2023 nodes, every `src`): `host_dst = 0, host_src = 0`.  So there is **no zero-copy
   host read inside the graph**, and `result_output` is an ordinary device tensor (not a pinned host
   output buffer read by the host without a synchronize).  Session 1's "next experiment 3" is negative on
   both counts.

3. **The upload path is definitively innocent, for the whole run.**  `AIN_VK` hashes every upload
   destination with **one** device kernel on the compute stream immediately after
   `ggml_cuda_async_input_wait()` (i.e. exactly at the graph's first read) and compares it with a hash of
   the bytes the host uploaded: **769 graphs x 9-10 uploads, 0 mismatches**.  Session 1's `AIN_VERIFY`
   only covered the first ~1024 snapshots (~85 graphs, i.e. *before* the first divergent window); this
   covers the entire run.  `AIN_HK` on the upload *copies* confirms the same in a **divergent** run:
   `ROCm0#out_ids#0`, `ROCm0#inp_pos#0`, `ROCm0#attn_inp_kq_mask#0` are byte-identical between two runs
   that diverge (0 differing graphs of 769).

4. **It is not the cross-stream hand-off either.**  `AIN_MAIN_COPY=1` issues every ring upload as a
   `cudaMemcpyAsync` **on the compute stream** (no second stream, no event pair, ordering by stream order
   alone) and it **still diverges** (95.6 / 95.7).  So no ordering hole between the copies and the graph
   can explain it.  Consistently, capturing the wait *inside* the graph (`AIN_CAP_WAIT=1`, a
   wait-on-upload-event node that every replay honours) changes nothing (97.9 / 96.6).

5. **Not the per-graph scratch lifetimes, not the Q8_1 machinery, not the KV quant type.**
   `AIN_NO_FREE=1` (leaks the `mmb` activation cache + producer slots + the cuBLAS BF16 src1 buffer
   instead of freeing them right after the asynchronous launch) still diverges; `AIN_Q81_NOCACHE=1`
   (forces a fresh `quantize_row_q8_1` for every mmvq) still diverges; the Q8_1 verify/GLU folds off still
   diverge; `-ctk/-ctv f16` (no q8_0 KV path at all) still diverges (96.2).

6. **The graph alternates two shapes in aligned 32-graph blocks, and the divergence is only *observable*
   in one of them.**  `AIN_TL` (a zero-perturbation per-graph record: token count, `nodes[0]`, graph
   instance, `use_cuda_graph`/`update_required`, upload count/bytes, ring wraps) shows:
   - block A: 9 uploads / 364868 B, `attn_inp_kq_mask` = **4096 B**, **no `out_ids` upload** → the
     row-pruned tail is empty (`result_output` has no data);
   - block B: 10 uploads / 368996 B, mask = **8192 B**, `out_ids` = 32 B (8 int32) → the row-pruned
     tail (last-layer FFN + final norm + output MUL_MAT) is real.
   The blocks alternate every 32 graphs (`nodes[0]` differs, so they are two distinct captured/replayed
   graph keys), so half of session 1's "bounded region that re-converges" was an **observation-window
   artifact**: in block A there is no logits tensor to differ.  The ring wraps every ~45 graphs
   (16 MiB / ~369 KB), which is *not* block-aligned, so the ring is not the block structure.

7. **Where the difference appears.**  Per-node device-side hashes (`AIN_HK`, a tiny kernel immediately
   after the named node, so no buffer-aliasing ambiguity for the tensor hashed):
   - `l_out-5`, `l_out-15`, `l_out-20`: **identical in every graph** (deep layers are fine);
   - `norm-31`, `gate-31`, `gate_sigmoid-31`, `attn_pregate-31`: **identical in every graph**;
   - `l_out-30`: differs in a subset of graphs (160/769 in one pair);
   - `attn_gated-31` / `attn_output-31` / `node_2012` / `node_2013` / `result_output`: differ in
     2-3x more graphs (256-274/769) — the difference appears *in the last-layer output path* and then
     propagates forward inside it.
   Magnitude is **not** rounding: `result_output` logits differ by 0.3-0.8 absolute (5-20 % relative),
   `attn_output-31` by up to 0.2 absolute (up to 40 % relative) — i.e. a substantive value difference.
   *Caveat:* the per-node instrument is **not yet trustworthy** for the nodes whose tensors are fused or
   aliased (it reported `attn_gated-31` differing while all of its inputs were identical — a MUL cannot do
   that, so that specific reading is an instrument artifact, see "next experiments").

8. **Instrument sensitivity, characterised.**  Anything that adds work **before the graph launch** makes
   the run deterministic (one 1-block kernel: 99.967; a host sleep anywhere: floor; a pre-launch D2H:
   floor), while **9-12 tiny kernels inside the graph do not** (97.1-98.7).  So the exposure is a
   narrow window *before* the launch, not inside the replay.

## ROOT CAUSE (found, fixed, validated this session)

`ggml_backend_cuda_set_tensor_async()` issued a **pinned-source `cudaMemcpyAsync` straight from the
caller's host pointer on the compute stream**.  A pinned-source async copy is read by the copy engine
*when the stream reaches it*, so the caller must not modify the source until the copy completes — and the
one caller that matters here cannot promise that: the scheduler's `host_src && dev_dst` branch uses
`set_tensor_async` for a host-resident **split** input, which for this graph is exactly the CPU split's
two outputs (`node_1`, `model.input_embed`, both in a `ROCm_Host` buffer — see `AIN_SPLIT_COPY`).  The next
ubatch's CPU split rewrites that buffer in place, so with `GGML_CUDA_ASYNC_INPUT=1` (host runs ahead) the
copy can read the *next* ubatch's bytes, or a torn mix.

That explains every observation in one mechanism:

- **rare and single-graph** — it needs the host to be far enough ahead that the copy has not executed when
  the next split-0 rewrites the source;
- **propagates** — the wrong embedding row enters layer 0 and every later graph in the chunk inherits it;
  the perplexity tool clears the cache per chunk, so the difference **re-converges at the next chunk**
  (that is why the runs looked "bounded and re-converging");
- **host-speed dependent** — any host delay (or a pre-launch device op that costs host time) lets the copy
  execute first;
- **graphs only** — without graph capture the host is the bottleneck (it must issue 2000+ kernels per
  graph), so it never gets ahead, which is why `GGML_CUDA_DISABLE_GRAPHS=1` is deterministic;
- **not the ring** — the ring path snapshots the source with a host `memcpy` into a pinned slot; only this
  one path copies *directly* from live host memory.

The onset probe agrees: with real (non-fused) node hashes, `node_15` (layer 0's qkv `MUL_MAT` output)
differs in exactly **5 of 769 graphs**, and `ffn_gate-0`/`ffn_out-0`/`ffn_out-30`/`result_output` differ
from each of those graphs to the end of its chunk (runs `(140,192) (207,256) (286,320) (457,512)`).

### Fix

Snapshot through the pinned upload ring instead of copying from live host memory (candidate patch:
[`r90-set-tensor-async-snapshot.patch`](r90-set-tensor-async-snapshot.patch), 19 lines, block 10 only):

```c
if (g_async_input_on && size > 0 && size <= GGML_CUDA_ASYNC_INPUT_MAX &&
    ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
    ggml_cuda_set_device(cuda_ctx->device);   // the meta backend forwards this per child device
    if (ggml_cuda_async_input_set(cuda_ctx->device, (char *) tensor->data + offset, data, size)) {
        return;
    }
}
```

Ring slots are not reused until the ring wraps, and the wrap is guarded by `cudaStreamSynchronize(upload
stream)`, so the snapshot lifetime is already safe.  `GGML_CUDA_ASYNC_INPUT=0` keeps the direct copy
(the delivery's current default) — see the open question below.

### Validation (same session, same binary; `AIN_SET_DIRECT=1` is the pre-fix control)

| arm | control (direct copy) | fix |
|---|---|---|
| 4B Q8_0, 12 chunks, `-ub 8 -b 8` | **91.0 / 90.8** | **99.967** (x5) |
| 4B, 24 chunks | **92.5** | **99.984** |
| 4B, `-ub 1 -b 1` | **95.1** | **99.967** |
| 4B, `BATCH=1` (default inside async) | **91.3** | **99.967** |
| 27B `UD-Q4_K_M`, 8 chunks | **91.6** | **100.000** |
| 35B-A3B `UD-Q4_K_M`, 2 GPU tensor, 6 chunks | 100.000 (does not reproduce here) | 100.000 |
| 4B, `-ub 512` (prefill, no capture) | 100.000 | 100.000 |
| 4B greedy `-n 64` seed 42 (off / fix / control, and `-ub 8`) | — | all **`7386359e5dac`** |
| 27B greedy `-n 48` (off vs fix) | — | both **`ad1333a99867`** |
| `test-logits-width-probe` W=1..8 (off / fix / control) | — | identical hashes, **`width_purity=PASS`** |
| `scripts/check-gpu-faults.sh` | — | clean |

The fix also holds with `AIN_SLEEP_LAUNCH_US=300` (the session-1 "must not be a delay" constraint) and is
deterministic with every `AIN_*` switch off.

### Instrument lesson (do not repeat)

A **fused** node's output is never materialised, so hashing it "via its consumer's src" reads a *stale*
buffer.  Always probe with `AIN_HK_NOSRC=1` first and keep only the names that come back non-sentinel.
Session 1's and this session's first "the tail is the only site" readings were that artefact: with real
node hashes the difference is in **layer 0** already.

## Remaining work (r90)

1. **Block ownership**: the fix is one hunk in `ggml/src/ggml-cuda/ggml-cuda.cu` = **block 10**, so no
   scheduler/block-09 change is needed.  Add the r90 stale-comment fix in the same amendment.
2. **Default flip** (the payoff): `g_async_input_on` becomes on-unless-`GGML_CUDA_ASYNC_INPUT=0`, then
   re-measure the PR #182 table (35B-A3B 2 GPU tensor 109.5 -> 115.8, 27B MTP 83.2 -> 86.2) and run the
   MTP four-axis gate.  Only after this session's fix.
3. **Full gate set** on the amended tree (build-rocm): the two per-arm greedy checks, `test-backend-ops`
   `ADD`/`GET_ROWS`/`CPY`/`MUL_MAT`, `scripts/gate-prefill-logits.sh` (and with `-ub 8` explicitly),
   `check-gpu-faults.sh`, `scripts/validate-set.sh`.
4. **Open question 1**: `ggml_backend_cuda_set_tensor_2d_async` (the meta/split splice path) has the same
   "copy straight from the caller's pointer" shape.  It is not reached by this reproducer (single GPU);
   decide whether to snapshot it in the same amendment or file it.
5. **Open question 2**: with `GGML_CUDA_ASYNC_INPUT=0` the same direct copy is used, so the latent race
   exists in principle on the shipping default too (unobserved: the host blocks repeatedly on the other
   copies).  Options: snapshot unconditionally (cost: one host memcpy per such copy) or keep it gated and
   document it.  Recommendation: gate on the async path for r90 (minimal change), and file the
   unconditional variant as a follow-up with its own measurement.
