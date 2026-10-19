# HANDOVER — async-input / graph nondeterminism (2026-10-19)

Session handover for a fresh agent.  Read this file, then
[`FINDINGS-2026-10-19.md`](FINDINGS-2026-10-19.md), then `AGENTS.md` (policy layer) before touching
anything.  This campaign is **not** part of the delivery: nothing here may be applied to the fork or
folded into `patches/` without the maintainer's explicit go-ahead.

## Mission

1. Find the actual defect behind "`GGML_CUDA_ASYNC_INPUT` is run-to-run nondeterministic".
2. Fix it properly (a device→host synchronization, **not** a delay — see "wrong answers" below).
3. If the fix makes the async path bit-deterministic, flip `GGML_CUDA_ASYNC_INPUT` default-on (that is
   the payoff: **+5.8 % 35B-A3B 2-GPU tensor, +3.6 % 27B MTP 2-GPU**, flat on one card) and drive that
   to an **r90** release.
4. Regardless of (3), r90 must fix the one stale comment in block 10 (see "r90 checklist").

Do not start with a re-derivation.  The facts below cost a full session.

> **RESOLVED IN r90** (released; `GGML_CUDA_ASYNC_INPUT` is default-on).  Root cause, session 2, [`FINDINGS-2026-10-19b.md`](FINDINGS-2026-10-19b.md)).**
> `ggml_backend_cuda_set_tensor_async()` issued a pinned-source `cudaMemcpyAsync` **directly from the
> caller's live host pointer**; the scheduler uses that path for a host-resident *split* input, which here
> is the **CPU split's** output (`node_1` / `model.input_embed` in a `ROCm_Host` buffer) that the next
> ubatch rewrites.  With `ASYNC_INPUT=1` the host runs ahead, so the copy could read the next ubatch's
> bytes (or a torn mix): a rare single-graph corruption in **layer 0** that propagates to the end of the
> chunk and re-converges at the next chunk (the perplexity tool clears the cache per chunk).  Candidate
> fix: [`r90-set-tensor-async-snapshot.patch`](r90-set-tensor-async-snapshot.patch) — snapshot through the
> pinned upload ring (19 lines, block 10).  Self-KLD: control **91.0 / 90.8 / 92.5 / 95.1** -> fix
> **99.967-100.000** (4B `-ub 8`/`-ub 1`/24 chunks/BATCH=1, 27B, 35B 2-GPU tensor); greedy text and
> `width_purity` unchanged.  Everything below in this file is the *pre-fix* record; H-A/H-B/H-C are dead
> (the split hand-over is at the front of the graph and forcing the uploads onto the compute stream still
> diverged).  `GGML_SCHED_DEBUG=1` works but needs `-lv 5`, not `-lv 4`.  Remaining work for r90 is the
> "Remaining work" section of the findings file (block-10 amendment, default flip + MTP gates, the two
> open questions).

## Where things stand

- Delivery release is **`v12-097cf33d9-r89`** (published, tagged, fork refreshed, images built by the
  worker).  r89 folded PR #182's batched async uploads into **block 10**, default-on *inside* the opt-in
  `GGML_CUDA_ASYNC_INPUT` path.  `ASYNC_INPUT` itself is still opt-in, because of this campaign.
- The cause recorded at r89 ("the uploads land after the graph replay read them") is **withdrawn** and
  corrected in `AGENTS.md`, `ENVIRONMENT.md` and `WORKLOG.md` 261019.006.  The **r89 block-10 code
  comment still carries the withdrawn wording** — that is a deliberate pending item for r90.
- `TODO.md` **#73** is the tracker item; `wip/CAMPAIGNS.md` has the row.  The campaign README's
  "Results"/"Readings" are the *original* data; the FINDINGS file supersedes their interpretation.

## What is established (do not re-test)

Instrument: `llama-perplexity --kl-divergence-base` **self-comparison** (same binary, two fresh
processes), 4B Q8_0, `-c 512 --chunks 12|24 -ub 8 -b 8 -ngl 99 -fa on -t 8 -ctk/ctv q8_0`, gfx1201.
`safe top p 99.967-99.984 %` = noise floor = deterministic; 85-98 % = divergence; a divergent arm's
run-to-run spread is ~1.5 points, so only floor-vs-band is meaningful.

- **The async uploads are ordered.**  `AIN_VERIFY` (D2H of the destination on the **compute stream right
  after `ggml_cuda_async_input_wait()`**, where the graph's first read happens) matched the uploaded
  bytes in **1024/1024** snapshots with no intervening device sync.
- **No inter-graph overlap.**  `hipEventQuery(done_ev)` at upload time = complete at **14227/14227**
  uploads; `wait` sees `pending` every time; one thread, one stream pair; ~9.5 uploads/graph.
- **Every per-graph host write is ordered.**  Async ring (verified) + `set_tensor_2d_async`
  splice-gather (main stream, 2/graph).  `cpy_tensor`, `cpy_tensor_async`, `get_tensor*` are unused in
  the loop; the sync-path `set_tensor` calls are **model-weight loads at startup**.
- **HIP is fine.**  `tools/ain_hip_test.cpp` (wait-before-replay, direct kernel vs graph, explicit vs
  per-thread upload stream) and `tools/ain_hip_test2.cpp` (event-after-launch covers the replay, slow
  graph, host several graphs ahead).  Both pass; both are needed together (copy waits `done_ev`, graph
  waits `ev`).
- **Not GDN / not fusions / not graph lifetime / not the per-thread stream.**  Dense 27B diverges
  identically; `GGML_CUDA_GDN_CHUNKED=0`, `GGML_CUDA_GDN_GATHER_INDEX=0`, `GGML_CUDA_DISABLE_FUSION=1`,
  `GGML_CUDA_GRAPH_MEM_GEN=0`, `GGML_HIP_GRAPH_FORCE_UPDATE=1`, `AIN_EXPL_STREAM=1` all still diverge.
- **Only captured graphs show it.**  `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_DISABLE_VERIFY_GRAPHS=1`
  (stops capturing the ≤ 8-token band) and `-ub 512` (no capture) are deterministic.  `-ub 8` and
  `-ub 1` diverge at `n_seq = 1` (r69's "multi-sequence" reading was incidental).
- **Batching/lazy are amplifiers**, not the cause: `BATCH=1` gives 84.9-90.4 % at `-ub 1` vs 96.5 %
  without.  (That is why they ship default-on and are documented as an amplifier.)
- **Bounded, re-converging divergence.**  `AIN_GRAPHSUM` (per-graph 64-byte checksum of the last OUTPUT
  tensor, D2H on the compute stream, no host sync) over 299 graphs: run A vs B differ at indices
  **97..128** and **289..298** only.  A diverged KV write cannot re-converge, so the differing value looks
  like something computed *within* a graph.  A repeat pair with the same instrumentation was identical —
  **the instrumentation perturbs the race**, so treat any "fixed" arm as suspicious until it is also
  slow-host-independent.

## WRONG ANSWERS — do not ship these

- **Any host delay.**  `AIN_SLEEP_US` / `AIN_SLEEP_BEFORE_US` (before the launch) *and*
  `AIN_SLEEP_LAUNCH_US` (after the launch) all produce the noise floor.  A delay that changes no ordering
  cannot be a fix; it only proves the effect is global host speed.
- **Any blanket host barrier** (`AIN_SYNC_SET/COPY/COMPUTE/DEVICE`, `AIN_SYNC_INPUTS`) — same reasoning,
  and they cost the whole async win.
- **Reverting PR #182 / re-blocking the host per input copy** — the maintainer's framing: the host must
  wait on a *real device signal* where one is missing, not be slowed down.

## Current best hypothesis (start here)

> **Superseded by session 2** — see [`FINDINGS-2026-10-19b.md`](FINDINGS-2026-10-19b.md) "Revised
> hypothesis".  The block below is kept as the record of what session 1 believed.

The graph is `nodes = 2025, splits = 2, input objects = 4, input tensors = 12, graphs reused = 61` on a
**single GPU**: the scheduler splits it across the CUDA backend and a second (CPU-side) backend, and the
hand-off is the one place the host consumes device output **without** the async ring's event discipline.

- **H-A (top):** the second split reads the first split's output before the CUDA graph has produced it —
  e.g. the scheduler's backend event record/wait does not cover a *captured graph* replay, or the
  consumer is a **host read of a pinned `result_output`** with no `ggml_backend_sched_synchronize()`.
  This is invisible to every CUDA-side instrument above, is fixed by any host slowdown, and cannot show
  in the delivery default (blocked host).
- **H-B:** the graph contains the delivery's own forked streams (`try_launch_concurrent_event`).  Not
  observed in the 4B (`-lv 4` shows no "Launching N streams") but must be checked on the MoE/MTP configs.
- **H-C (maintainer's guess):** a host/shared-state overwrite of the r84 class (two graph-shaped things,
  same node/leaf count) driven by the second scheduler / MTP.  Our reproducer has no MTP, so this is a
  *second instance of the same class*, not the cause here — keep it for the MTP configs.

## Next experiments (in order)

> **Superseded by session 2** — steps 1-3 below were run and are negative except that step 1 located the
> split structure.  The live ordered list is in [`FINDINGS-2026-10-19b.md`](FINDINGS-2026-10-19b.md)
> "Next experiments".

1. **Identify split 2.**  Print `sched->splits[i].backend->get_name()` (and the split's node range) once,
   from `ggml/src/ggml-backend.cpp` or via `sched_reserve`'s existing log, in the instrumented build.
   `GGML_SCHED_DEBUG=1` + `-lv 4` produced nothing in the last session — find the actual print path rather
   than assuming the env var works.
2. **Snapshot the split hand-off.**  Reuse the `AIN_VERIFY` trick at the boundary: enqueue a D2H of the
   split-1 output tensor on the stream the *consumer* uses, immediately after the scheduler's event
   wait, and compare against the producing graph's last-written bytes.  If it is stale, H-A is confirmed.
   Cheap alternative first: with `splits = 2`, does forcing the *CPU* split's input to be read through
   `ggml_backend_sched_synchronize` **at the split boundary** (not at step start) fix it?  That is a real
   signal, not a delay, and it is the shape of the likely fix.
3. **Check the output path**: is `result_output` in a `CUDA_Host` (pinned) buffer that the host/KLD reads
   directly?  If so, the read needs a synchronize, and the fix is in the output readback, not the input.
4. **Then the MTP/dual-scheduler configs** (`35B-A3B` 1 GPU / 2 GPU tensor, 27B MTP n3) with the same
   instrument, since those are where H-C lives and where a fix must also hold.

Design constraint for the eventual fix: it must be a **device→host or stream→stream dependency** that
survives a *fast* host (test: the fix + `AIN_SLEEP_LAUNCH_US=300` must give the same result; and the fix
must be deterministic with the sleep switches *off*).

## Tooling (already built, reuse it)

`tools/DEBUGBUILD2.patch` (session 2, applies on top of `DEBUGBUILD.patch`) adds the split/copy-path
prints, the zero-perturbation per-graph timeline (`AIN_TL`), per-node device hashes (`AIN_HK`), the
whole-run upload verifier (`AIN_VK`), `AIN_MAIN_COPY`, `AIN_CAP_WAIT`, `AIN_NO_FREE` and
`AIN_Q81_NOCACHE`; the table is in [`FINDINGS-2026-10-19b.md`](FINDINGS-2026-10-19b.md) "Tooling".  On
`~/llama.cpp`, branch `ain-debug` is `dcf22dfce`.

- `tools/DEBUGBUILD.patch` — the full `AIN_*` diagnostic patch against r89 (`git apply` on a scratch
  branch).  Switches: `AIN_TRACE` (counters, per-tensor histogram, sync-path names), `AIN_VERIFY`,
  `AIN_GRAPHSUM`, `AIN_SYNC_SET/COPY/COMPUTE/DEVICE/INPUTS`, `AIN_SLEEP_US`,
  `AIN_SLEEP_BEFORE_US`, `AIN_SLEEP_LAUNCH_US`, `AIN_EXPL_STREAM`, `AIN_DUMP_EVERY`.
- `~/llama.cpp` branch **`ain-debug`** = r89 + that patch (the patch is committed there as `dceb26d38`);
  `~/llama.cpp/build-ain` is the built tree.  `~/llama.cpp` itself is checked out on `rdna-boosts`
  @ `399610a83`; `build-rocm` holds the r89 build.  Rebuild the diagnostic tree with
  `cd ~/llama.cpp && BUILD_DIR=build-ain ~/bin/build-llama-rocm` (~1 min incremental, ccache on).
- `tools/ain_hip_test{,2}.cpp` — standalone HIP reproducers; build line in each file header.
- Runner pattern used for every arm (each arm = 2 passes, base then `--kl-divergence`):

```bash
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
C=/llm/models/wikitext-2-raw/wiki.test.raw
PPL=~/llama.cpp/build-ain/bin/llama-perplexity
run() { local tag=$1; local envs=$2; local f=/tmp/$tag.kld; rm -f $f
  for pass in base cmp; do local extra=""; [ $pass = cmp ] && extra="--kl-divergence"
    env $envs HIP_VISIBLE_DEVICES=0 timeout 1800 $PPL -m $M -f $C -c 512 --chunks 12 -ub 8 -b 8 \
      -ngl 99 -fa on -t 8 -ctk q8_0 -ctv q8_0 --kl-divergence-base $f $extra > /tmp/$tag-$pass.log 2>&1 || return 1
  done; rm -f $f
  printf '%-20s %s\n' "$tag" "$(grep -oE 'Same top p: [0-9.]+ ± [0-9.]+ %' /tmp/$tag-cmp.log)"
}
A="GGML_CUDA_ASYNC_INPUT=1 GGML_CUDA_ASYNC_INPUT_BATCH=0"
run base "$A"                       # expect ~95-98 %  (divergent)
run floor "$A AIN_SLEEP_US=20"      # expect ~99.97 %  (floor; proves the harness)
```

## Pitfalls

- **The KLD base file is ~3 GB per arm** — delete it (`rm -f`) after each arm; `/home` was at 93 % full.
- Instrumentation perturbs the race; always include a known-divergent arm and a known-floor arm in the
  same session before concluding anything.
- **35B-A3B Q4_K_M greedy text is run-to-run nondeterministic on stock r88** (1 GPU and 2 GPU tensor,
  400 tokens, `ASYNC_INPUT=0`), so a text hash cannot gate that model.  Use the 4B/27B + self-KLD.  That
  separate default-path nondeterminism is *not* tracked yet (a candidate TODO item).
- Never run benches in parallel; size `-t` away from the GPU IRQ cores; `llama-cli` always
  `--single-turn`.
- Do **not** run the container image build on the primary box.

## r90 checklist

Comment-only baseline (do this even if the fix does not land):

1. In `ggml/src/ggml-cuda/ggml-cuda.cu`, the comment above `g_async_input_on` still says the `done_ev`
   wait "does not order the HIP graph replay's read".  Replace it with the corrected statement (the ring
   is the trigger; the defect is elsewhere; point at `wip/async-input-graph-ordering/`).
2. If a fix lands: block ownership — `ggml/src/ggml-cuda/ggml-cuda.cu` = **block 10**,
   `ggml/src/ggml-backend.cpp` / `src/llama-context.cpp` = **block 09**.  A scheduler fix therefore
   amends block 09 (+ block 10 for the default flip) and needs the full gate set, not just this
   instrument.
3. Default flip only if deterministic: `g_async_input_on` becomes "on unless
   `GGML_CUDA_ASYNC_INPUT=0`" (default-on, env only disables — see the default-on policy in `AGENTS.md`).

Release mechanics (same as r89; the record is `WORKLOG.md` 261019.004/005):

```bash
# canonical chain: amend the owning block commit on ~/llama.cpp's rdna-boosts, then replay the tail
cd ~/llama.cpp
git checkout <block-NN commit>            # e.g. 3b3b7599b was r88's block 10
git apply <the change>                    # or edit, then: git add -A
git commit --amend --no-edit              # new block-NN commit
NEW=$(git rev-parse HEAD)
git rebase --onto $NEW <old block-NN commit> rdna-boosts
BUILD_DIR=build-rocm ~/bin/build-llama-rocm     # then run the gates (see below)

cd ~/llama-cpp-rdna-boosts
./scripts/make-patches.sh ~/llama.cpp 097cf33d9 $(git -C ~/llama.cpp rev-parse --short rdna-boosts)
./scripts/make-release.sh --tip <tip> --tree <tree> --release v12-097cf33d9-r90
./scripts/validate-set.sh                 # must be green (strict 12/12, tree matches)
# docs: patches/README.md release block, MANIFESTS.md, BASELINE.md, README.md, TODO.md state, WORKLOG
# entry (YYMMDD.VVV, newest first), ENVIRONMENT.md if a variable changes, archive the wip campaign if
# it is promoted (archive/work/<name>/ + RESOLUTION.md, delete the wip entry, update CAMPAIGNS.md)
git add -A && git commit -m "r90: ..."
git tag -a v12-097cf33d9-r90 -m "v12-097cf33d9-r90: ..." <release commit>
git push origin main && git push origin v12-097cf33d9-r90
cd ~/llama.cpp && git push --force-with-lease origin rdna-boosts     # PERSONAL FORK ONLY
# then record the publication in WORKLOG (new entry) and commit/push; the radiance worker builds images
```

Gates to run before tagging (r89 baseline values in brackets):

- 4B Q8_0 `-n 64` seed 42 same-seed greedy with async off / on [all `7386359e5dac`].
- `test-logits-width-probe` W=1..8: identical hashes, `width_purity=PASS`.
- 27B 1-GPU and 27B MTP 2-GPU 400-token greedy byte-identical across arms.
- `scripts/gate-prefill-logits.sh` [0.000707 / 98.755 %] — and, if the async path becomes default-on,
  also run it **with `-ub 8` explicitly**, because that is where this defect lives.
- `test-backend-ops` `ADD`/`GET_ROWS`/`CPY`/`MUL_MAT`; `scripts/check-gpu-faults.sh`.
- If async goes default-on: the **MTP four-axis gate** (`benchmarks/mtp-adaptive-methodology.md`) on
  35B-A3B 1 GPU / 2 GPU tensor and 27B MTP, plus the PR #182 A/B table (35B-A3B 2 GPU tensor 109.5 →
  115.8 t/s, 27B MTP 83.2 → 86.2).

## Reporting rules

Report findings to the maintainer **before** any delivery-affecting commit and **before** any tag, and
wait for a decision (see AGENTS.md).  Surface open questions and nits rather than burying them.
