# NEXT-SESSION-PROMPT — PR #150 + issue #139 (copy-paste into the fresh session)

Cold-start handover for the delivery repo `~/llama-cpp-rdna-boosts`. **Read `AGENTS.md` first** (policies,
invariants, gates), then `WORKLOG.md` 261014.001/261014.002 and this prompt. Current release is **r57**:
`release.json` = base `097cf33d9`, canonical tip `a300075f7`, net tree
`8a56fb31267ecc523c423845662d76571f934a51`, **12 blocks**. `~/llama.cpp` is on `rdna-boosts` (same tree)
and built with `~/bin/build-llama-rocm-714`. Everything (docs, tooling) is pushed on `main`; the container
build now self-prunes GHCR (`CONTAINERS.md` §Registry cleanup) and needs the `delete:packages` scope.

**Order of work: do PR #150 first and drive it all the way to completion before starting issue #139.**
Do not open the AR work until #150 is closed (folded + released + published + responded to + archived, or
refuted/parked with the decision recorded).

Each item is a full pass through the session flow we used for r56/r57: assess -> adapt + rebuild -> gates
-> fold into the owning block -> regenerate `patches/` + `release.json` -> `scripts/validate-set.sh` ->
docs -> commit/push -> fork `rdna-boosts` refresh -> tag -> containers -> GitHub response -> archive the
record with a `RESOLUTION.md`. Report the plan for the item to me **before any delivery-affecting
commit**.

- **#150** (below): if it lands, cut **`v12-097cf33d9-r58`**; if it is refuted or parked, record that
  and move on (no release).
- **#139** (below), started only after #150 is done: same flow; if it lands, cut **`v12-097cf33d9-r59`**
  (use r58 instead only if #150 produced no release).

## 1. PR #150 — `wip/dflash-layer-inp-preset` (@overdoingism)

<https://github.com/stew675/llama-cpp-rdna-boosts/pull/150>. It adds `wip/dflash-layer-inp-preset/` with
one `git am` patch cut against **r56** (now r57), touching `common/common.cpp`, `src/llama-context.cpp`,
`src/llama-ext.h`, `src/llama-model.cpp`, `src/llama-model.h` (nothing under `ggml/`).

**What it does.** A DFlash draft calls `llama_set_embeddings_layer_inp()` for its target layers when it
attaches — *after* the target context was reserved — so the first request re-reserves and frees +
reallocates every compute buffer (954.6 -> 1064.6 MiB device; the growth landed in shared GPU memory on a
near-full Windows card, prefill ~200 t/s). The patch reads the draft GGUF's `<arch>.target_layers` before
`common_init_from_params()` creates the target context and presets them via a new
`llama_model_set_embeddings_layer_inp_preset()`, cleared right after the context is created so a later MTP
context from the same model does not inherit it; plus an early return in `set_embeddings_layer_inp()` when
the flag does not change. Both halves are required (the preset alone still re-reserved at the same size).

**Drive it.**
- Review the preset lifetime / the early return / the "DFlash or DSpark draft only" gate against r57;
  `git am` it and rebuild.
- Gates: outputs **and** draft acceptance byte-identical with vs without; the reserve count 5 -> 4 and no
  free+allocate at the first request. The PR's config is Qwen3.8-27B Q5_K_M + DFlash2 (Q4_K_M, draft 3),
  `-c 220000 -ub 1280`, q8_0 KV; a DFlash2 drafter is present under `/llm/models/Qwen3.8/27B/` (e.g.
  `Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` with the Q4_K_XL target) — use the closest available pair and
  record what you used. Re-confirm the normal coherence + `gate-prefill-logits.sh` gates.
- Fold into the owning block: **DFlash is block 11**; the `common/common.cpp` draft-setup hunk might belong
  to **block 02** (adaptive MTP) — decide and say why. Then respond on the PR, merge/close, archive.

## 2. Issue #139 — AR spin timeout leaves a call unreduced (@briansp2020)

<https://github.com/stew675/llama-cpp-rdna-boosts/issues/139>. Two independent root causes, both with data
in the issue; the fix belongs in **block 01** (`ggml/src/ggml-cuda/allreduce-hip.cu`).

- **`clock64()` step-back (the clear bug).** On gfx1201 `clock64()` reads `HW_REG_SHADER_CYCLES_LO/HI` and
  can go **backwards** (two instrumented waits: 1089107279 -> 1075226717 over 1.6 ms of wall time), so
  `(uint64_t)(clock64() - wait_t0)` wraps to ~1.8e19 and passes any budget — a slightly-late peer is
  treated as lost. briansp2020's local fix: time the spin with `wall_clock64()` (`s_sendmsg_rtn
  GET_REALTIME`, constant 100 MHz) and size the budget as `GGML_CUDA_AR_SPIN_TIMEOUT_MS *
  hipDeviceAttributeWallClockRate` (kHz). The `GGML_CUDA_AR_PROFILE` `prof_start`/`prof_spin` fields still
  use `clock64()` and have the same step-back.
- **Queue stops (the pressure bug).** amdgpu `evict_process_queues_cpsch` stops all of a process's GPU
  queues while the kernel reclaims GPU-mapped host pages (p50 5 ms / p99 178 ms / max 517 ms), so a stop
  over the budget also trips a *legitimate* timeout. The r49 loader reclaim helped; the remaining ~54 GiB
  pinned `ROCm_Host` master is the lever (`wip/host-expert-master-drop/`, TODO #55).
- **The correctness gap.** A timed-out call returns with the local shard; the poison flag is only seen at
  the *next* call (~allreduce-hip.cu:1393 / early return ~398), so the timed-out step is consumed
  unreduced (0/1191 acceptance, wrong greedy sha). The r57 comment at the timeout branch is deliberate.

**Land a real solution, not just a bigger budget.** Minimum: the wall-clock budget (kills the false
timeout). Preferred: **also re-run the timed-out call through the butterfly before returning**, so a
timeout costs time instead of correctness — or an equivalent that guarantees no unreduced result is ever
observed by the graph. Keep the default-on policy in mind: if it is beneficial and gated, it ships on with
an opt-out (`FEATURE=0`), not an opt-in.

**Validate.** `GGML_CUDA_AR_PROFILE` spin times (`clock64()` bug is visible there), spin-timeout counts
(should be 0 in a clean run), same-seed greedy shas vs a known-good reference, 35B-A3B draft-MTP
acceptance, and — if you can get bpftrace on the box — `evict_process_queues_cpsch` stop counts (that is
the actual goal: fewer stops / no unreduced output). Coordinate the memory-pressure half with
`wip/host-expert-master-drop/`: it is a separate deliverable, so do not fold it in unless it is validated
as a combination.

## Constraints (from AGENTS.md — do not re-derive)

- **Never push to upstream.** Only the delivery repo (`github.com:stew675/llama-cpp-rdna-boosts`) and the
  personal fork (`git@github.com:stew675/llama.cpp.git`, branch `rdna-boosts`, `--force-with-lease`).
- The set must apply **whitespace-clean**; use `scripts/apply-all.sh` (never a concatenated `git apply`).
- `llama-cli` **always** `--single-turn` (+ `--no-display-prompt`); wrap anything blocking in `timeout`.
- Threads: leave the GPU IRQ cores (13,14,15 with 3 GPUs) free; prefer `HIP_VISIBLE_DEVICES=0,1`; run GPU
  work **sequentially** (never parallel/background benches).
- A prefill-kernel / default-on approximate change also needs `scripts/gate-prefill-logits.sh` (mean KLD
  <= 0.005, same-top-p >= 98 %). Record results in the new `WORKLOG.md` entry (newest first,
  `YYMMDD.NNN`).
- **Report findings, open questions and nits to me before any delivery-affecting commit / release / tag**,
  and stop to ask on anything that is not a straightforward change. If I am asleep, use my blanket
  go-ahead for the delivery but leave a recorded finding for anything judgement-call-shaped.
- New WIP goes under `wip/` and is **not** part of the delivery; a promoted campaign folds into its owning
  block and is archived to `archive/work/<name>/` with a `RESOLUTION.md` (no redirect stubs).

Start by reading `AGENTS.md`, then the PR and the issue in full, then tell me the plan and the priority
order before touching `patches/`.
