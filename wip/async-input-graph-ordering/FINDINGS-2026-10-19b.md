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

## Revised hypothesis

The upload ring, the split hand-off, the host-resident-tensor paths and the per-graph scratch lifetimes
are all excluded.  The difference is produced **inside the last layer's output path** (row-pruned FFN +
final norm + output projection), it is **large**, it is **per-run** (not per-process), and the only thing
the async path changes is *how far ahead the host is when the graph runs*.  Two candidates remain:

- **H-D (top):** the verify-band (<= 8 row) kernels used only by that tail are themselves
  timing-sensitive — `mul_mat_vec_q_ksplit` (the dense-rows K-split mmvq: `attn_output-31`'s and the
  gate's MUL_MAT are the probes that first differ) and/or `GET_ROWS` at 8 rows.  The reading "identical
  fp32 input -> different fp32 output" is what a K-split reduction with a shared-memory/`lf_reduce`
  hazard, or a K-split scratch block that collides with a *different* pool block, would look like.
- **H-E:** something in the tail reads a *pool block* that a later host step has already recycled (the
  generic `ggml_cuda_pool` scratch, not the `mmb`/BF16 caches that `AIN_NO_FREE` covers).  The pool
  alloc/free calls are host-time, so with the host ahead a block can be handed out while an earlier
  graph's kernel still holds its address; on a single stream that is safe *in theory*, which is why this
  needs the direct test in "next experiments" 2 rather than more argument.

## Next experiments (in order)

1. **Make the per-node hash immune to fusion/aliasing** (the instrument gap above).  Hash *every* node
   output with a device kernel inserted in the node loop for all nodes (not a name list), and store the
   per-node hash for the 2-3 graphs around the first divergence (the differential is `AIN_HK` with the
   list replaced by "all nodes", 2023 hashes/graph for ~5 graphs only).  That gives the true first
   differing node in the last-layer path without a name/aliasing artefact — or shows the difference is
   already there at `attn_gated-31`'s inputs.
2. **Test H-E by poisoning the pool**: add a diagnostic that fills every `ggml_cuda_pool` block with a
   NaN/poison pattern on free (or hands out blocks only after a `cudaStreamSynchronize`) and see whether
   the divergence disappears.  If it does, the fix is a real device-side dependency on pool reuse.
3. **Test H-D by taking the verify-band matmuls off the ksplit path** (a diagnostic `GGML_CUDA_MMVQ_...`
   switch forcing the pre-block-13 path for `ncols_dst 2..8`).  Deterministic => the ksplit kernel; still
   divergent => the divergence is upstream of the matmul.
4. **Then narrow the tail**: hash `attn_gated-31`'s two inputs (`attn_pregate-31`, `gate_sigmoid-31`,
   already stable) *and* the Q8_1 buffer mmvq actually reads, in the same graph, to separate "wrong
   activation" from "wrong matmul".
5. Re-run the two controls in every arm (`ASYNC_INPUT=1 BATCH=0` must land in the band, `+AIN_SLEEP_US=20`
   on the floor) and only then read a "fix" as a fix.  Any "fix" must still be deterministic with the
   `AIN_*` switches off and must survive `AIN_SLEEP_LAUNCH_US=300` (design constraint from session 1).

## Tooling (this session's additions)

`tools/DEBUGBUILD2.patch` (applies on top of `tools/DEBUGBUILD.patch`, i.e. `dcf22dfce` on `ain-debug`)
adds, all `AIN_*`-gated and diagnostic only:

| switch | what it does |
|---|---|
| `AIN_SPLITS=N` | prints each schedule's splits: backend, node range, inputs (name/size/buffer/backend) — the `GGML_SCHED_DEBUG` print path, without the log-level dependency |
| `AIN_SPLIT_COPY=N` | prints every split-input copy: which of `sync_graph_input` / `set_async` / `cpy_async` / `copy_sync` ran, with src/dst buffer+host flags |
| `AIN_SPLITS_GRAPH=N`, `AIN_GRAPH_TAIL=1`, `AIN_HOSTTENS=1` | prints the first nodes / last 15 nodes of a split graph, and scans the whole split for host tensors |
| `AIN_TL=1` | per-graph timeline (tokens, `nodes[0]`, instance, used/upd, uploads, bytes, ring wraps), printed from `atexit` |
| `AIN_HK=a,b,c` (+`AIN_HK_FLOAT=1`) | per-node device hashes of named nodes *or their srcs*, at the point of production, into per-graph slots (device counter, tick kernel per graph) |
| `AIN_VK=1` | one device kernel hashing every upload destination right after the input wait, compared with the host hash of the uploaded bytes |
| `AIN_MAIN_COPY=1` | issues the ring uploads on the compute stream instead of the upload stream (no event pair) |
| `AIN_CAP_WAIT=1` | enqueues the wait-on-upload-event *inside* the capture, so the graph itself carries the dependency |
| `AIN_NO_FREE=1` | skips the per-graph `mmb`/BF16 cache frees (leak) |
| `AIN_Q81_NOCACHE=1` | forces a Q8_1 cache miss (fresh quantization for every mmvq) |

Note the instrument sensitivity above: `AIN_VK` and any pre-launch kernel make the run deterministic, so
`AIN_VK` can verify the uploads but cannot be used to observe a divergent run; `AIN_HK` (inside the
graph) and `AIN_TL` (host-only) do not perturb it.

## Reporting status

Review findings / open questions reported to the maintainer before any delivery-affecting change.  The
r90 comment-only fix (block 10, the withdrawn `done_ev` wording) and the default-on flip are **not**
done; nothing from this campaign is in `patches/`.
