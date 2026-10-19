# wip/async-input-graph-ordering

**The `GGML_CUDA_ASYNC_INPUT` path is run-to-run nondeterministic whenever the consumer is a captured
HIP graph.**  That is the whole reason the async graph-input ring has been opt-in since r69, and fixing
it is what makes it (and PR #182's batched uploads, r89) default-on.

Status: **OPEN (seeded 2026-10-19)** — mechanism narrowed to "the copy-completion wait does not order
the graph replay"; the exact hole is not identified yet.  Nothing here is part of the delivery.

## Payoff if fixed

Turning `GGML_CUDA_ASYNC_INPUT` on today (r89, with the batched uploads default-on inside it),
gfx1201, 400-token greedy, per-arm spread <= 0.4 %:

| config | default (async off) | `ASYNC_INPUT=1` |
|---|---|---|
| 35B-A3B Q4_K_M, 2 GPU `-sm tensor` | 109.5 t/s | **115.8 (+5.8 %)** |
| 27B Q4_K_M MTP n3, 2 GPU `-sm tensor` | 83.2 t/s | **86.2 (+3.6 %)** |
| 35B-A3B, 1 GPU | 100.0 t/s | 99.6 (−0.4 %) |
| 27B, 1 GPU | 31.7 t/s | 31.8 (flat) |

So the prize is ~+4-6 % decode on every multi-GPU config, ~flat on one card.  The r68 default-on was
reverted in r69 for exactly the defect below; nothing else blocks default-on.

## The instrument

`llama-perplexity --kl-divergence-base` **self-comparison**: the same binary run twice in separate
processes (fresh start each time), then compared to the base file it wrote.  A deterministic path
measures `RMS Δp 0.001 % / same top p 99.984 %` (that is the noise floor of the instrument); a
nondeterministic path measures `3-15 % / 85-97 %`.  `tools/selfkld.sh` from
`wip/kv-cell-offset-nondeterminism/` wraps it, and this is the same instrument r69 used.

## Reproducer

4B Q8_0, wikitext-2 test, 24 x 512-token chunks, gfx1201, one card.  `-ub 1` and `-ub 8` diverge;
`-ub 512` is clean.  **No MTP, one sequence (`n_seq = 1`), one backend** — see the hypothesis section,
because that matters.

```bash
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
C=/llm/models/wikitext-2-raw/wiki.test.raw
PPL=~/llama.cpp/build-rocm/bin/llama-perplexity      # any r69..r89 build

run() { # <tag> <env> <ub> <batch>
  rm -f /tmp/$1.kld
  env $2 HIP_VISIBLE_DEVICES=0 $PPL -m $M -f $C -c 512 --chunks 24 -ub $3 -b $4 \
      -ngl 99 -fa on -t 8 -ctk q8_0 -ctv q8_0 --kl-divergence-base /tmp/$1.kld > /tmp/$1-base.log 2>&1
  env $2 HIP_VISIBLE_DEVICES=0 $PPL -m $M -f $C -c 512 --chunks 24 -ub $3 -b $4 \
      -ngl 99 -fa on -t 8 -ctk q8_0 -ctv q8_0 --kl-divergence-base /tmp/$1.kld --kl-divergence > /tmp/$1-cmp.log 2>&1
  echo -n "$1: "; grep -E "Same top p:" /tmp/$1-cmp.log; rm -f /tmp/$1.kld
}

run off_ub8      "GGML_CUDA_ASYNC_INPUT=0"                                             8 8
run async_ub8    "GGML_CUDA_ASYNC_INPUT=1"                                             8 8
run async_ub1    "GGML_CUDA_ASYNC_INPUT=1"                                             1 1
run async_ub512  "GGML_CUDA_ASYNC_INPUT=1"                                           512 512
run async_ub8_nograph "GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_DISABLE_GRAPHS=1"             8 8
run async_ub1_nograph "GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_DISABLE_GRAPHS=1"             1 1
run async_ub8_noverify "GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_DISABLE_VERIFY_GRAPHS=1"     8 8
run batch_ub1    "GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_ASYNC_INPUT_BATCH=1"               1 1
run lazy_ub1     "GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_ASYNC_INPUT_LAZY_DONE=1"           1 1
```

## Results (r88 + PR #182 as the candidate tree; r89 reproduces the same numbers)

| arm | `-ub/-b` | graphs | same top p |
|---|---|---|---|
| `ASYNC_INPUT=0` | 8/8 | on | **99.984** (noise floor) |
| `ASYNC_INPUT=0` | 1/1 | on | **99.984** |
| `ASYNC_INPUT=1` | 512/512 | on | **100.000** |
| `ASYNC_INPUT=1` | 8/8 | on | 95.83 / 96.00 / 96.49 (3 runs) |
| `ASYNC_INPUT=1` | 8/512 | on | 95.33 |
| `ASYNC_INPUT=1` | 1/1 | on | **96.52** |
| `ASYNC_INPUT=1` | 8/8 | **off** | **99.984** (3/3) |
| `ASYNC_INPUT=1` | 1/1 | **off** | **99.984** |
| `ASYNC_INPUT=1 DISABLE_VERIFY_GRAPHS` | 8/8 | on | **99.984** |
| `ASYNC_INPUT=1 BATCH=1` | 8/512 | on | 96.01 |
| `ASYNC_INPUT=1 BATCH=1` | 8/8 | on | 90.41 |
| `ASYNC_INPUT=1 BATCH=1` | 1/1 | on | **85.77 / 84.89** |
| `ASYNC_INPUT=1 BATCH=1` | 1/1 | **off** | **99.984** |
| `ASYNC_INPUT=1 LAZY_DONE=1` | 1/1 | on | 90.25 |
| `ASYNC_INPUT=1` + forced stream sync (below) | 1/1 | on | **99.984** |
| `ASYNC_INPUT=1 BATCH=1` + forced stream sync | 1/1 | on | **99.984** |

Readings:

1. **The trigger is graph capture, not the ubatch or the sequence count.**  `-ub 512` never captures
   (multi-token prefill graphs are skipped by `ggml_cuda_graph_is_multi_token`), `-ub 8` and `-ub 1`
   do (`n_tokens <= MMVQ_MAX_BATCH_SIZE`, i.e. the verify/decode band).  `DISABLE_VERIFY_GRAPHS=1`
   turns the 8-token case into "multi-token" and fixes it, which is a second, independent control.
2. **r69's "`n_seq > 1`" was incidental.**  All of the above run `n_seq = 1` (the log line
   `kl_divergence: computing over 24 chunks, n_ctx=512, batch_size=..., n_seq=1`).  The r69 note should
   not be read as "multi-sequence only".
3. **Batching and the lazy done-event make it worse, not better** (85-90 % vs 96 %).  Both defer the
   copy later, i.e. they widen the window — which is a *useful* amplification for bisecting.
4. **The flips are per-run, not per-process-state**: the same binary, same arm, repeated → 95.8 /
   96.0 / 96.5, so it is a race, not a deterministic layout difference.
5. **A host barrier at the copy→graph boundary removes it.**  The scratch patch used (maintainer A/B
   only; not in any release):

   ```c
   // ggml_backend_cuda_graph_compute(), immediately after ggml_cuda_async_input_wait(...)
   if (getenv("PR182_FORCE_SYNC")) { CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream())); }
   ```

   `PR182_FORCE_SYNC=1` gives `99.984 %` for both `ASYNC_INPUT=1` and `ASYNC_INPUT=1 BATCH=1` at
   `-ub 1`.  So the copies themselves are byte-correct and the ring bookkeeping is sound: the missing
   piece is the **completion dependency between the transfer and the replayed graph**.

## Hypotheses to test

Ordered by how much of the evidence each explains.  Note what the reproducer rules in/out: plain
`llama-perplexity`, one sequence, one backend, no speculative decoding, no second scheduler in play.
A defect there is at the copy↔graph boundary.  Separately, the 35B-A3B and MTP configs have *their own*
run-to-run text nondeterminism even with async off (below), so an MTP-side hole is a real second axis —
a fix must satisfy both.

- **H1 (maintainer's guess) — a host/shared-state overwrite from the second scheduler, MTP-side.**
  The lead-up is r84 (issue #183): `ggml_gallocr_alloc_graph()` reused a layout for a graph with the
  same node/leaf *count* but a different node order, so the in-graph sampling graph of a newly active
  sequence read onto one recycled block.  The second scheduler (r65, block 09) is what puts two
  same-count graphs side by side, so the class is "a second graph-shaped thing invalidates an
  assumption the first one makes".  Candidates here: whatever caches per-graph *input* state across the
  two schedulers (input tensor identities, the scheduler's split-input buffer reuse, `moe_cache_take_over`,
  or a `graph_key`/`props` cache that now sees two live graphs).  Test: rerun the `-ub 1` reproducer with
  the second scheduler off (`LLAMA_DRAFT_SCHED2=0`-equivalent for the model in use) and with
  `--spec-type none`; if it is unchanged, H1 is not the plain-reproducer cause and stays an MTP-only
  axis to be bisected separately on the 35B-A3B/MTP configs that flip text run-to-run.
- **H2 — `hipGraphLaunch` does not inherit the preceding `hipStreamWaitEvent`.**  The wait is issued on
  the compute stream *before* the launch and should order the graph's first node; the forced-sync
  result is consistent with it not doing so.  Cheap discriminators: (a) keep the wait and add a trivial
  no-op kernel launch on the stream between the wait and `cudaGraphLaunch` (if that fixes it, the wait
  is fine and the graph launch path is the problem); (b) `rocprofv3` trace of one `-ub 1` run and check
  the vgpr packet/barrier order around `hipGraphLaunch`; (c) try recording the copy event on an explicit
  stream instead of `cudaStreamPerThread` (the per-thread default stream is the one unusual object in
  the chain).
- **H3 — `cudaStreamPerThread` is per *thread*, and the ring wrap sync is per thread.**  `memcpy` into
  the ring is bump-allocated and the wrap path only does `cudaStreamSynchronize(cudaStreamPerThread)` +
  `cudaEventSynchronize(flush_ev)`; if uploads are issued from more than one host thread, thread B's
  wrap does not wait for thread A's in-flight copies out of the slots it is about to overwrite.  This
  is a genuine pre-existing hazard (the ring predates the second scheduler and the multi-threaded
  split uploads).  Test: log `pthread_self()`/thread id in `ggml_cuda_async_input_set` and check for
  more than one id in the reproducer; then force a ring wrap (small `GGML_CUDA_ASYNC_INPUT_RING`) and
  see whether the divergence rate moves.
- **H4 — the destination is written *after* the queued copy by a scheduler-managed path that the
  drains do not cover.**  The batched path defers the copy to `graph_compute`, so any host-side write
  into a COMPUTE buffer that is not one of the drained entry points lands *before* the copy (harmless)
  or *between* the copy and the graph (dangerous).  Test: for a `-ub 1` run, record `(dst, size,
  first-16-bytes hash)` at upload time and compare against a device readback right before the graph
  replay (`--logits-all`-style probe or a debug `get_tensor` under `GGML_CUDA_GRAPH_TRACE=1`).
- **H5 — the forked/join streams inside a captured graph** (`try_launch_concurrent_event`, block 06)
  read the input on stream N while the copy-completion event only orders stream 0.  The join happens
  before the graph ends, so this should not matter for the *previous* graph; it would matter if a fork
  reads a graph input that the next upload overwrites.  Test: `GGML_CUDA_DISABLE_...` the concurrent
  path, or check whether the reproducer's graph has a fork/join at all (`GGML_LOG_DEBUG`) first.

## Tests to run when it is fixed

- This self-KLD matrix (all `99.984 %`), 3 starts per arm.
- Decode start-to-start (`wip/kv-cell-offset-nondeterminism/tools/decodestart.sh`) with async on, and
  `test-logits-width-probe` hashes across starts.
- 4B / 27B same-seed greedy, and the prefill-logit gate with `ASYNC_INPUT=1` (it is a *default-ubatch*
  instrument; add `-ub 8` explicitly, because that is where the defect lives).
- Then the default flip + the MTP four-axis gate (`benchmarks/mtp-adaptive-methodology.md`) on the
  configs that already flip text run-to-run without it: 35B-A3B 1 GPU / 2 GPU tensor, 27B MTP.

## Notes

- **Greedy text is a weak gate here.**  On 35B-A3B Q4_K_M (1 GPU and 2 GPU tensor) the *stock r88
  binary* already produces 2-3 different 400-token greedy texts across runs with `ASYNC_INPUT=0`
  (`722063f3e28d` / `e3b891655a80` / `27d50a6d97f5`), so a text hash cannot separate the async path's
  effect on that model.  Use the self-KLD instrument and the 4B/27B models instead.  That separate
  nondeterminism is worth its own record — it is present in the delivery default, not this campaign.
- The ring, the drain/done-event machinery and the batched upload path all live in block 10
  (`ggml/src/ggml-cuda/ggml-cuda.cu`); per the promotion rules, nothing here may be applied to the
  fork or folded into `patches/` without the maintainer's go-ahead.
- History: PR #125 (r46, opt-in ring) → r68 (default-on) → r69 (back to opt-in, this defect) →
  PR #182 / r89 (batched uploads default-on *inside* the opt-in path; amplifies the race, and that is
  why this campaign exists).  Records: `WORKLOG.md` 261016.001 (r69), 261015.006 (r68),
  `archive/work/async-input-batch/RESOLUTION.md` (r89), `archive/work/issue-151-host-path-regression/`.
