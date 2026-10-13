# HANDOVER — WIP cleanup: resolve four campaigns in one session

**Status:** OPEN / cold-start handover (seeded 2026-10-13, immediately after the r53 release).  This
document lives in `wip/dflash-meta-split/` but covers **four** campaigns the maintainer wants driven to a
recorded resolution in a single session.

**Goal:** for each campaign below, reach a decision and record it: **promote** (fold into the owning
block, regenerate, validate, archive, WORKLOG) or **close/refute/park** (archive with a `RESOLUTION.md`,
update `wip/CAMPAIGNS.md`, record in `WORKLOG.md`).  "Resolved" means the maintainer's decision is
recorded, not that a fix necessarily landed.  Report findings to the maintainer **before** any
delivery-affecting commit and stop on anything that is not a straightforward change.

> **Cold start.**  Read `AGENTS.md` first (report-before-ship, the pushing policy, the WIP/promotion rules,
> the default-on policy, "Critical facts", "Common tasks"), then this file, then each campaign's own
> `README.md`.  Do not start editing before that.

## 0. Current state and where to work

- **Delivery:** `v12-097cf33d9-r53`; `release.json`: base `097cf33d9`, tip `4b27ac644`, tree
  `4083fd0ebed7c4659bd3254b2ab5a9f9906a209e`, 12 blocks.  `WORKLOG.md` 261013.003 is the release record.
- **Fork checkout:** `~/llama.cpp` is on branch `rdna-boosts` (re-cut for r53, tip `f74905dea`, tree ==
  `release.json.tree`).  Its `origin` is the personal fork (`git@github.com:stew675/llama.cpp.git`).
- **Work in a disposable worktree / scratch branch, never on the fork's `rdna-boosts` branch.**  The WIP
  rule: nothing under `wip/` may be applied to the fork or any delivery checkout unless the maintainer
  explicitly asks for that item.  Suggested:
  ```bash
  cd ~/llama.cpp
  git worktree add ~/llama-wip-r54 -b wip-r54 <r53 tip or 097cf33d9>
  # apply a campaign's patch/hack only inside ~/llama-wip-r54, never in ~/llama.cpp
  ```
  (A worktree off the r53 tip gives you the delivery behaviour as the baseline; off `097cf33d9` gives the
  unadulterated base.  Pick per campaign; state which you used.)
- **Build** (ccache is on; `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is required on CMake >= 4.3):
  ```bash
  cd ~/llama-wip-r54
  BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
  # fast loop: cmake --build build-rocm-hybrid --target llama-cli llama-bench -j 16
  ```
- **GPU:** 3x R9700 (gfx1201).  **3 GPUs have known issues in some paths, so prefer 2 GPUs
  (`HIP_VISIBLE_DEVICES=0,1`) for A/B and MTP work** unless the campaign is specifically about 3-GPU
  behaviour.  Pin `-t 8` (or a `--cpu-mask` with `-t` inside it) so the GPU IRQs on the top cores stay
  free.  Models under `/llm/models/`; prompts in `prompts/` (hash-stable, never edited in place).
- **Gates/commands:** `scripts/validate-set.sh`, `scripts/gate-prefill-logits.sh`,
  `scripts/gate-qwen4exp-quant-coherence.sh`, `scripts/gate-per-block-build.sh`, `scripts/check-gpu-faults.sh`,
  `scripts/extract-generated.py`.  Benchmarks/method: `benchmarks/mtp-adaptive-methodology.md` (MTP gate)
  and `benchmarks/prefill-logit-methodology.md` (prefill-logit gate).
- **Cohort of this handover:** `dflash-meta-split` (TODO #58), `host-expert-master-drop` (#55),
  `meta-segmented-expert-upload` (parked TODO), `ssm-alpha-beta-fusion` (TODO item 5(b)).

## 1. `wip/dflash-meta-split/` — `-sm tensor` + DFlash abort (TODO #58)

**Type:** correctness/limitation (a hard abort).  **Newest** of the four (found while validating r53).

**Symptom:** a DFlash target plus drafter (`-md`, `--spec-type draft-dflash`) under `-sm tensor` aborts at
load:
```
/home/.../ggml/src/ggml-backend-meta.cpp:707: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```
after the expected `dflash requires ctx_other to be set (this warning is normal during memory fitting)`.
`-sm layer` works.  **r52 and r53 both abort** (pre-existing, not a PR #144 regression).

**Repro (2 GPUs, r53 tree):**
```bash
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
D=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1 ./build-rocm-hybrid/bin/llama-cli -m "$M" -md "$D" \
  --spec-type draft-dflash --spec-draft-n-max 3 -ngl 99 -sm tensor -c 16384 \
  -ctk q8_0 -ctv q8_0 -f prompts/prose-rdna-boosts.txt -n 512 --seed 42 --temp 0 \
  --reasoning off --no-display-prompt --single-turn
```
Related, possibly the same root cause: a standalone `-md` MTP head under `-sm tensor` hits
`GGML_ASSERT(!suffix_fallback.empty())` at `src/llama-model.cpp:470` (documented in
`benchmarks/mtp-adaptive-methodology.md`, "a draft head cannot be loaded standalone").

**r53 code sites:** assert `ggml/src/ggml-backend-meta.cpp:707`; the split-state cache around it; the
`--fit` probe path that emits the `ctx_other` warning; `src/llama-model.cpp:470`.

**What "resolve" means:** either (a) a real Meta split-state / graph fix so `-sm tensor` loads and runs,
or (b) an explicit, clean rejection with a `-sm layer` hint (replace the raw assert) documented as an
accepted limitation, or (c) an upstream report.  Report the finding and let the maintainer pick.

**First steps:** add a temporary print at the assert (tensor name/op/shape/split state) to identify which
op creates a `SPLIT_AXIS_0` state on the draft graph; compare with the working `-sm layer` state; check
whether the draft's split-state cache entry collides with a target entry.

**Gates:** the repro loads and generates coherently under `-sm tensor`; single-sequence F1 on == off;
`--parallel 2` host vs F1 per-slot identical (protocol in `archive/work/dflash-multiseq/`);
`GGML_META_SS_VERIFY=1`; `test-backend-ops` unaffected.

## 2. `wip/host-expert-master-drop/` — drop the full pinned master under `--host-experts pool` (#55)

**Type:** memory/perf feature (Phase 3 of the r35 host pool).  **Largest** of the four.

**Current reality (r49 numbers; re-confirm on r53):** the pool is only a bounce tier; the full pinned
`ROCm_Host` master is still kept (`ggml/src/ggml-cuda/moe-expert-cache.cu:474/477` logs
"Phase 2b; full pinned master kept").  `RssShmem` is identical with and without the pool: 18.7 GiB on
35B-A3B Q4_K_M `-ncmoe 40`, 54.1 GiB on Flash-Next IQ4_NL `-ncmoe 40`.

**Why:** (i) the ~54 GiB of GPU-mapped host memory is the remaining footprint; (ii) **issue #139** (@briansp2020):
the amdgpu driver stops all of a process's GPU queues when the kernel migrates a GPU-mapped host page
(bpftrace: ~2.4 `evict_process_queues_cpsch`/s, p99 178 ms); fewer pinned pages = fewer stops.

**Work (from the README):** enumerate every reader of a host expert's pinned `cur->data`
(`llama-model-loader.cpp`'s `moe_cache_set_host_source`, the per-ubatch op-offload H2D upload, the cache
fill) and make each pool-aware under `--host-experts pool`, leaving the default `pinned` path
byte-identical; quantify the pool-only cold-serve cost at `MOE_HOST_POOL_FRAC` 25/explicit/100 on 35B-A3B
and Flash-Next IQ4_NL; prove numerics (bit-identical); re-run Brian's #139 field config (2x R9700,
Flash-Next IQ3_S, `-sm tensor -ncmoe 48`, MTP n-max 3) and count queue stops.

**r53 code sites / env:** `ggml/src/ggml-cuda/moe-expert-cache.cu` (pool `g_pool_*`, the "full pinned
master kept" log, `moe_host_pool_on`), `ENVIRONMENT.md` §1.2.1 (`MOE_HOST_POOL_MIB`,
`MOE_HOST_POOL_FRAC`, `MOE_HOST_POOL_PREWARM`, `MOE_HOST_POOL_BGFETCH`, `MOE_HOST_POOL_DIO`,
`MOE_HOST_POOL_PREFETCH`), `common/arg.cpp:2711` (`--host-experts`), `src/llama-model-loader.cpp`
(`moe_cache_set_host_source`).  r35 promotion record: `archive/work/host-expert-dio-cache/`; r49 finding:
`archive/work/host-memory-reclaim/`.

**Gates:** `validate-set.sh` if promoted; coherence `1c5d32ac537d`; `gate-prefill-logits.sh`; `MUL_MAT_ID`
931/931; `test-backend-ops -o MUL_MAT -b ROCm0`; 35B-A3B `-ncmoe 99` + auto cache `none == n1 == n3 == n7`
(short) and `none == n3` (long); qwen4exp pure; `llama-batched-bench -npl 1,4,8`.  gfx1100 (`fingon`) is
the pressure target; gfx1201 is the portability check.

**Realistic note:** this is the least likely to fully land in the same session as the other three.  A
good session outcome is the reader enumeration + a scoped, measured fix (or a documented negative) and a
recorded decision.

## 3. `wip/meta-segmented-expert-upload/` — segmented async upload (parked TODO)

**Type:** feature lifting an explicit rejection; blocked on a non-deterministic hang.

**Item:** gemma4 + host-resident experts (`-ncmoe`/`-cmoe`) + `-sm tensor` is hard-rejected at
`src/llama-model.cpp:412` ("the fused expert tensor's segmented per-ubatch upload has no tensor-split
path").  All-resident or `-ngl`/`--fit`-offloaded gemma4 splits fine.  The rejection exists because
gemma4's fused `ffn_gate_up_exps` has a segmented split layout (`n_segments`/`nr`) and
`ggml_backend_meta_set_tensor_async` (`ggml/src/ggml-backend-meta.cpp:2295`) only handles a contiguous
single-segment slice; the MMQ tail pad `copy_experts` appends makes the range non-row-aligned.  The
synchronous buffer setter is `ggml_backend_meta_buffer_set_tensor` (`:1764`).

**Blocker:** a partial fix **hung non-deterministically** (one GPU spinning).  The parked TODO and the
campaign README are explicit: **do not retry the fix before a deterministic reproducer exists.**  Note
`GGML_META_PINHOST` (cited in the old README) no longer exists in the r53 tree; `GGML_CUDA_SPLICE_GATHER`
does (`ggml/src/ggml-cuda/ggml-cuda.cu:3637`).  Re-read the r48/r49 meta changes (issue #137 added
`host_buffer`/`events` proxying and the async pipeline) before assuming the old line numbers/surroundings.

**What "resolve" means:** produce a deterministic repro (`compute-sanitizer`, the splice force-mode, or a
new harness), then either fix the segmented async upload (and lift the `llama-model.cpp:412` rejection)
or keep it parked with the repro recorded so the next attempt starts from a real failure.

**Record:** `archive/work/moe-expert-cache/item3-findings-session20.md` (the partial fix and measurement),
`archive/work/issue-99/README.md` (the `llm_arch_supports_sm_tensor` relaxation).

**Gates (if fixed):** gemma4 + `-ncmoe`/`-cmoe` + `-sm tensor` starts, loads, coherent; the cases that
work today (all-resident `-sm tensor`, `-sm layer`, `--fit`-offloaded whole layers) unchanged;
`validate-set.sh` if promoted.

## 4. `wip/ssm-alpha-beta-fusion/` — single-walk `ssm_alpha` + `ssm_beta` (TODO item 5(b))

**Type:** small perf idea (`~0.3-0.6 % prefill`), numerics-moving (fp32 reorder).  **Most likely to be
closed rather than promoted.**

**Item:** qwen4exp/hybrid-SSM layers build two F32 GEMMs on the **same** `src1` (`cur`):
`src/models/qwen4exp.cpp:2511` `beta = build_lora_mm(ssm_beta, cur, …)` and `:2518`
`alpha = build_lora_mm(ssm_alpha, cur, …)`, with beta's `reshape/sigmoid` and alpha's `reshape` between
them.  Verified on r53: they are **not adjacent** in the scheduled graph, so the general `nodes i, i+1`
pair-matcher cannot fire.

**Prize:** the stacked `[2560x96]` rocblas `M=96` path is 0.410 ms vs 0.588 ms for the two `M=48` GEMMs
(1.43x), and is **not bit-identical** (~3e-7 max diff).  Routes from the discovery:
**(a)** load-time stacked weight + one graph `MUL_MAT` + per-half views (~0.27-0.4 %, loader plumbing);
**(b)** graph restructure + a custom single-walk kernel (~0.6 %, the real prize).  Both need a quality
gate + re-baseline.

**r53 code sites:** `src/models/qwen4exp.cpp:2511/2518`, the SSM builders in `src/llama-graph.cpp`.
Re-verify the "same `src1` tensor object" property and the current `cgraph` adjacency on r53 before
implementing (the archive cites older line numbers).

**What "resolve" means:** re-measure the prize on r53 and decide.  At 0.3-0.6 % with a numerics change it
may not clear the bar; a clean "closed as below the bar / parked with the measured number" is a valid
resolution.  If route (a) is chosen, the loader plumbing is the work.

**Gates:** `scripts/gate-prefill-logits.sh` (prefill numerics); `test-logits-width-probe` `W=1..8` one
hash; adaptive-MTP `-n 3000`; same-seed coherence; `validate-set.sh` if promoted; a new env knob in
`ENVIRONMENT.md` if it becomes a default-on feature.

## 5. Recommended session order

1. **Triage all four first** (30-60 min): confirm the dflash repro, confirm the r53 pool `RssShmem`
   numbers, re-check the gemma4 rejection still throws, and confirm the ssm-alpha/beta adjacency.  Then
   agree the order with the maintainer.
2. `dflash-meta-split` — smallest scope, clear repro.
3. `ssm-alpha-beta-fusion` — a measurement/decision, likely close.
4. `meta-segmented-expert-upload` — repro hunt; may legitimately stay parked.
5. `host-expert-master-drop` — largest; leave enough time or explicitly defer.

Do not run parallel/background benches.  Run GPU work sequentially.

## 6. Policy reminders (read `AGENTS.md` in full)

- **Report findings before shipping** (MANDATORY): surface findings, questions and nits to the maintainer
  **before** any delivery-affecting commit/tag/push, and wait for the decision.  Do not bury them in a
  commit message.
- **WIP/promotion:** nothing under `wip/` or `archive/work/` is applied to the fork or folded into
  `patches/` unless the maintainer explicitly asks for that item.  Promotion = give the win an env
  kill-switch (default on if it passed the gates), re-validate the **combination**, fold into the owning
  block, regenerate + validate, archive to `archive/work/<name>/` with a `RESOLUTION.md`, record in
  `WORKLOG.md`, **delete the `wip/` entry (no stub)** and update `wip/CAMPAIGNS.md`.
- **Default-on policy:** a beneficial, gated feature is enabled by default; its env var only disables it.
- **Never push to upstream; only the personal fork.**  Confirm the remote before any push.
- **Always fold into the owning block, never a trailing block.**
- **`llama-cli` needs `--single-turn`** (and `--no-display-prompt` for scripts); wrap blocking commands in
  `timeout`.  Never `git apply` the concatenated series; use `scripts/apply-all.sh`.
- If this handover's work touches `wip/dflash-meta-split/`'s own record, remember the handover lives here:
  if dflash is resolved first, move this file with its campaign (to `archive/work/dflash-meta-split/`) or
  relocate the cleanup handover, so it is not orphaned.

## 7. Pointers

- `AGENTS.md`; `ENVIRONMENT.md` §1.2.1; `TODO.md` #58, #55, the parked meta-upload section and item 5(b).
- The four campaign READMEs (this directory, `../host-expert-master-drop/`, `../meta-segmented-expert-upload/`,
  `../ssm-alpha-beta-fusion/`).
- `archive/work/moe-expert-cache/item3-findings-session20.md`, `archive/work/issue-99/README.md`,
  `archive/work/host-expert-dio-cache/README.md`, `archive/work/host-memory-reclaim/README.md`,
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`.
- Release record: `WORKLOG.md` 261013.003 (r53).
