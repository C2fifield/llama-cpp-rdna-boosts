# HANDOVER — r53: DFlash `--parallel` (PR #144) + FA staging prealloc (#138) + upstream `MUL_MAT_ID` dup

**Status:** OPEN / fresh-session handover (seeded 2026-10-13, after r52 shipped).
**Goal:** cut the next delivery release, **`v12-097cf33d9-r53`**, folding two contributor items into
their owning blocks and finishing the upstream `MUL_MAT_ID` duplicate-ids fix as an `upstream/`
candidate.  The workflow is the same one used for r52 (see `../WORKLOG.md` 261013.002 and the r52
kickoff in `../archive/work/`).

This document covers three items the maintainer asked to work in one session:

| item | source | disposition |
|---|---|---|
| **A** | PR [#144](https://github.com/stew675/llama-cpp-rdna-boosts/pull/144) `wip/dflash-multiseq` (overdoingism) | **promote into the delivery** (blocks 00 + 10 + 11) |
| **B** | issue [#138](https://github.com/stew675/llama-cpp-rdna-boosts/issues/138) comment `6148819285` (DanoPTT): `fa-stage-prealloc` still not in a release | **promote the beta patch** (blocks 09 + 10) |
| **C** | `wip/upstream-mul-mat-id-dup/` (this directory) | **upstream candidate only, NOT in `patches/`** |

Items A and B are the r53 delivery changes; item C is an upstream artifact produced in the same
session (the delivery ships pure on this bug).

---

## 0. Current state (r52) and the r52 playbook

- Delivery `main` at the r52 release commit, tag `v12-097cf33d9-r52`.
- `release.json`: base `097cf33d9`, canonical tip `2f3349306`, net tree
  `8b177a7b0d33a86e11aeaec6bae71fd0bda74b57`, 12 blocks.
- Personal fork `rdna-boosts` branch: rebuilt tip `14bb91dc9` (same tree).
- `wip/` at r52 has no r53 work yet; the r52 campaigns are archived under `archive/work/`.

The r52 fold was a `git rebase -i` over the block chain (edit the owning block, `git am` the PR hunk,
`git commit --amend`, `git rebase --continue`), then:

```
# canonical chain source: ~/llama.cpp (or a fresh worktree at release.json.tip)
git worktree add ~/llama-integrate -b v2-integrate <canonical-tip>
#   rebase -i 097cf33d9, edit the owning blocks, apply the PR hunks, amend, continue
scripts/make-patches.sh ~/llama-integrate 097cf33d9 <new-tip>
scripts/make-release.sh --tip <new-tip> --tree <new-tree> --release v12-097cf33d9-r53
scripts/validate-set.sh
scripts/gate-per-block-build.sh --changed-from <previous-tip>     # only the changed blocks
```

Then the GPU gates (below), the docs, `archive/`, commit + tag, delivery push, fork refresh
(`apply-all.sh` + `git push --force-with-lease origin rdna-boosts`), and PR replies.  **r53 is a
no-Docker release** (the maintainer deferred container builds to a later tag; confirm again at the
start of the session).

**Always fold into the owning block, never into a trailing block.**  Determine ownership with:

```
git log --oneline -S '<symbol>' 097cf33d9..HEAD -- <file>
grep -n '^diff --git' patches/00NN-*.patch
```

---

## A. PR #144 — DFlash with `--parallel >= 2` (F1 for several sequences + readback wait)

**PR:** from `overdoingism:dflash-multiseq`, commit `7337054`, one `git am` file with two commits
against r50: `wip/dflash-multiseq/dflash-multiseq.patch` (252 lines, 4 files, `git apply --check`
clean on r50; **re-verify against r52**).  Nothing under `ggml/` changes.

### A1 — F1 for several sequences

`common/speculative.cpp` and the F1 target API.  A multi-sequence batch is split by `split_equal`, so
the ubatch order can differ from the batch order; `output_reorder()` fixes only the host rows.  The
target now keeps the inverse of `embd_batch_idxs` and the draft maps each batch row to its device row.

Files/hunks and their **r52 owning blocks** (confirmed with `git log -S`):

| hunk | symbol | owner block | notes |
|---|---|---|---|
| `common/speculative.cpp` (DFlash F1 `lf_dev`, `GGML_LF_DFLASH_DEV`) | `GGML_LF_DFLASH_DEV` -> block 11 | **11** (`campaign remainder`, DFlash) | extends the single-sequence F1 to `n_seq > 1`; adds the host-experts guard |
| `src/llama-context.cpp` (`get_lf_dev_rows`, `lf_dev_rows`, `output_reorder` guard) | `get_lf_layer_inp_dev` -> block 10 | **10** (qwen4exp, F1 target API) | `lf_layer_dev` is a block-10 symbol |
| `src/llama-context.h` (`get_lf_dev_rows` decl + `lf_dev_rows` member) | `lf_layer_dev` -> block 10 | **10** | |
| `src/llama-ext.h` (`llama_lf_get_dev_rows`) | `llama_lf_set_layer_inp_dev` -> block 10/11 | **10** (declaration) | the API is declared here and defined in `src/llama-context.cpp` |

**Guard to keep:** when the target has host-resident MoE experts (`-ncmoe`/`-cmoe`,
`llama_model_moe_host_expert_dev_count(...) > 0`), several sequences stay on the **host path**.  With
the movable-boundary slab armed for them the draft's device reads were observed to break (Qwen3.6-35B-A3B,
3 slots, `--n-cpu-moe 20`: acceptance `0/1860` vs `121/1279` host path; one sequence or
`GGML_CUDA_SLAB=0` is fine).  One sequence is unchanged.  This is a documented guard, not a fix; the
PR author points at the draft/slab overlap of issue #127 in its MoE form.

### A2 — `GGML_READBACK_SYNC` wait for a ubatch without outputs

`src/llama-context.cpp`, the `sync_readback` / `GGML_READBACK_SYNC` block -> **block 00** (the r43
readback fix; `sync_readback` is introduced by block 00).  The wait must run when
`n_outputs > 0 || extract_all_idxs`, and the per-token DFlash layer inputs (`cparams.embeddings_layer_inp`)
must be synchronized too.  Seen on Windows (r49/r50); not reproducible on Linux; the cost measured
neutral on Linux (10 request pairs: 60.43/60.58 s vs 60.62/60.44 s).

### A merge risk to check

`src/llama-context.cpp` is also touched by blocks 02, 03, 05, 09, 10; the `decode()` function is a hub.
Apply A1 at block 10 and A2 at block 00 **at the intermediate block states** (the r52 method), not only
at the final tree.  `git apply --check` at each target block commit first.

### A validation (do on r52 + the folds)

- 1 slot, standard benchmark: target output identical to r52; prefill/decode/acceptance unchanged.
- `--parallel >= 2` DFlash, host path vs F1: same tokens and acceptance, F1 prefill faster (the PR
  measured +15-24 %; our 3x R9700 number is the useful one).
- The host-experts guard fires: `-ncmoe` + several sequences logs the warning and keeps the host path.
- `test-backend-ops -o MUL_MAT_ID` unchanged; `ggml/src/ggml-cuda` library builds byte-identical
  (the PR changed no `ggml/` source).

---

## B. `beta/fa-stage-prealloc` promotion (issue #138)

**Artifact:** `beta/fa-stage-prealloc/fa-stage-prealloc.patch` (sha256
`f1a0f0b1c87dc77e6bce796af94b63262c109dce5a8a93d09685d262137d7f2a`), cut against r48, **applies clean
on r52** (verified this session with `git apply --check`).  `BETA-TESTING.md` in that directory has the
reporter's checklist.

### What it does

`ggml_backend_cuda_context::fattn_stage_try_get()` grows the per-stream F16 prefill staging arena in
1.25x steps, `cudaMalloc` new + `cudaFree` old.  On Windows/WDDM the freed steps stay committed (the
ladder commits ~5x the final arena); the reporter measured +2.65 GiB at 119k and +3.6 GiB at 164k, and
a spill to shared memory near the VRAM limit (pp 793 -> 255 t/s).  `--fit` already reserves
`min(2*GGML_CUDA_FA_STAGE_MAX_MB, f16_kv_bytes)`, so the ladder is what exceeds the budget.

The fix passes a size hint (F16 size of the K/V view roots, capped per operand by
`GGML_CUDA_FA_STAGE_MAX_MB`) into `fattn_stage_try_get`, which tries the hint first **only while
`optional_alloc_within_free_cap(...)` allows it**, then falls back to the 1.25x step.  New kill switch
`GGML_CUDA_FA_STAGE_PREALLOC=0` (default on) restores the ladder.

### Ownership on r52 (confirmed with `git log -S`)

| hunk | symbol | owner block |
|---|---|---|
| `ggml/src/ggml-cuda/common.cuh` (`fattn_stage_try_get` definition) | introduced block 09 | **09** (host-resident MoE iface) |
| `ggml/src/ggml-cuda/fattn-common.cuh` (`launch_fattn` stage call site / `stage_buf`) | introduced block 09, changed block 10 | **10** (qwen4exp) |

So the promotion is a **two-block fold (09 + 10)**, not a single "block 15" fold.  Apply the
`common.cuh` hunk at block 09's edit stop and the `fattn-common.cuh` hunk at block 10's.

> The old `beta/fa-stage-prealloc/README.md` says "block 15" (pre-V2 numbering).  That is historical;
> use the V2 ownership above.

### B validation

- Linux gfx1201 already validated pre-fold (r48): coherence unchanged, arena A/B via prefill-logit KLD
  mean **0.000000** / same-top-p **99.997 %**, standard prefill-logit gate PASS, `pp16384` unchanged
  within noise.  Re-run the standard gate on r53.
- After the fold, re-confirm the reporter's Windows numbers were already collected on the beta
  (issue #138 comment 2: dedicated growth -77 %/-76 %, pp/tg unchanged, greedy hash identical).
- `GGML_CUDA_FA_STAGE_MAX_MB=0` (unbounded) must pass no hint (keeps the ladder); `=1` keeps growth
  near zero at ~11 % pp cost.
- On promote: add a `GGML_CUDA_FA_STAGE_PREALLOC` row to `ENVIRONMENT.md`, fold the note into
  `patches/README.md`, move `beta/fa-stage-prealloc/` to `archive/work/fa-stage-prealloc/` with a
  `RESOLUTION.md`, and update issue #138 (the reporter is holding their Windows stack for this).

---

## C. Upstream `MUL_MAT_ID` duplicate-ids (this directory)

**Delivery disposition: upstream candidate only.  Do not add it to `patches/`.**  The `wip/README.md`
in this directory has the full symptom, the upstream confirmation (`097cf33d9` reproduces the same F16
failures + `quantize_mmq_q8_1` fault) and the candidate homes.

### Task

1. Reproduce on unadulterated master with `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` (F16 fails,
   q8_0/q4_0 pass) and confirm the `quantize_mmq_q8_1` memory fault.
2. Fix the F16 dup path (repeated expert ids within a row).  The MMQ path already dedups via
   `ggml_cuda_launch_mm_ids_helper`; the F16 path (`mmvf.cu` `mul_mat_vec_id` / the non-MMQ
   `MUL_MAT_ID` dispatch in `ggml-cuda.cu`) does not.  Treat the `quantize_mmq_q8_1` fault as a second
   symptom and re-check the MMQ `src1_q8_1` sizing separately.
3. Validate on the fix branch: `test-backend-ops -o MUL_MAT_ID` green including `dup=1`, `-o MUL_MAT`
   green, no `Memory access fault`, `scripts/check-gpu-faults.sh` clean.
4. Produce `upstream/UPSTREAM-PR-mul-mat-id-dup.md` + `.patch` (self-contained against unadulterated
   `ggml-org/llama.cpp`, re-created from current master, the `test_mul_mat_id_dup` cases already
   upstream) following the `upstream/README.md` conventions.  Do not file without the maintainer's
   go-ahead.
5. Only if the maintainer decides the delivery must carry it: that is a separate change with its own
   env kill-switch and a purity/KLD gate; by default the delivery stays pure.

Once the upstream candidate exists, move this `wip/upstream-mul-mat-id-dup/` directory (README +
candidate) to `archive/work/` per the WIP rule, and record it in `WORKLOG.md`.

---

## D. r53 release gate checklist (the r52 set)

- [ ] `scripts/make-patches.sh` + `scripts/make-release.sh` -> `v12-097cf33d9-r53`, then
      `scripts/validate-set.sh` (strict 12/12 `git am`, tree match).
- [ ] `scripts/gate-per-block-build.sh --changed-from <r52-tip>` (only the edited blocks + tip).
- [ ] `scripts/gate-prefill-logits.sh` (mean KLD <= 0.005, same-top-p >= 98 %).
- [ ] `scripts/gate-qwen4exp-quant-coherence.sh`.
- [ ] Coherence: 4B same-seed identical; DFlash `--parallel >= 2` host-path vs F1 output identical.
- [ ] 35B-A3B draft-MTP `-sm tensor -c 16384 -n 3000` bf16 KV (acceptance unchanged, MTP >= plain).
- [ ] `test-logits-width-probe` on 4B and 35B-A3B.
- [ ] `test-backend-ops -o TOPK_MOE` and `-o MUL_MAT_ID -p 'n=1,'`.
- [ ] stock-relative `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (dense 27B Q4_K_XL, q8_0 KV).
- [ ] `ENVIRONMENT.md`: `GGML_CUDA_FA_STAGE_PREALLOC` (default on, kill-switch).  No new env var from
      PR #144 (it reuses `GGML_LF_DFLASH_DEV` and `GGML_READBACK_SYNC`).
- [ ] Docs: `WORKLOG.md` (new dated r53 entry), `README.md`, `MANIFESTS.md`, `BASELINE.md`,
      `patches/README.md` current-state headers; `archive/work/` RESOLUTIONs; `wip/CAMPAIGNS.md`.
- [ ] Commit + tag `v12-097cf33d9-r53`; push delivery `main` + tag; refresh the personal fork
      `rdna-boosts` (`apply-all.sh`, tree check, `--force-with-lease`); **no Docker**.
- [ ] Reply to PR #144 and close it (no em-dashes); post the issue #138 outcome.
- [ ] Report all findings and any design questions to the maintainer **before** pushing.

## E. Open questions for the maintainer

1. **PR #144 host-experts guard.**  Keep it as the PR ships (several sequences -> host path when
   `-ncmoe`/`-cmoe`), or dig into the slab/draft interaction (issue #127) and fix it instead?  Default:
   keep the guard and ship; file the interaction as a follow-up.
2. **`llama_lf_get_dev_rows` API.**  PR #144 extends the F1 API in `src/llama-ext.h`.  Confirm the
   signature/name is acceptable before folding into block 10.
3. **fa-stage-prealloc default.**  Default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` (matches the PR and
   the default-on policy).  The `optional_alloc_within_free_cap` guard keeps a slab run from losing
   its free VRAM; confirm.
4. **Docker.**  Assume no container build for r53 (confirm at session start).
5. **Upstream dup fix.**  Confirm it stays an `upstream/` candidate and is not folded into `patches/`.

## F. Pointers

- r52 release record: `../WORKLOG.md` 261013.002; fold/gates example: this repo's `main` history.
- r52 canonical chain: `release.json.tip` (`2f3349306`); fork branch `rdna-boosts` `14bb91dc9`.
- per-block ownership: `../archive/work/patch-block-reorg/FILE-OWNERSHIP.md`.
- PR #144 patch: `wip/dflash-multiseq/dflash-multiseq.patch` (branch `overdoingism:dflash-multiseq`,
  fetched as local branch `pr144` in the delivery repo).
- issue #138: comment `6148819285` is DanoPTT's "not in r51" note; the beta dir is
  `../beta/fa-stage-prealloc/`.
- upstream candidate conventions: `../upstream/README.md`.
