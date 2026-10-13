# NEXT-SESSION-PROMPT — r53 (copy-paste this into the fresh session)

We are cutting the next delivery release, **`v12-097cf33d9-r53`**, from the r52 delivery. Read
`AGENTS.md` first, then read the full handover at **`wip/upstream-mul-mat-id-dup/HANDOVER-r53.md`**.
Work in the delivery repo (`~/llama-cpp-rdna-boosts`); the fork is `~/llama.cpp` (branch
`rdna-boosts`, currently the r52 tree).

There are three items:

1. **Promote PR #144** (`wip/dflash-multiseq`, overdoingism) into the delivery. It has two commits:
   F1 (device-resident DFlash target features) for several sequences, and the `GGML_READBACK_SYNC`
   wait for a ubatch without outputs. Fold into the **owning blocks** as the handover maps:
   the `common/speculative.cpp` DFlash hunk goes to **block 11**, the F1 target API
   (`src/llama-context.{cpp,h}`, `src/llama-ext.h`) to **block 10**, and the readback wait to
   **block 00**. Keep the host-resident-MoE-experts guard exactly as the PR ships (several sequences
   stay on the host path when `-ncmoe`/`-cmoe`).

2. **Promote `beta/fa-stage-prealloc/fa-stage-prealloc.patch`** (issue #138). It applies clean on r52.
   It is a **two-block fold**: the `ggml/src/ggml-cuda/common.cuh` `fattn_stage_try_get` hint goes to
   **block 09**, the `ggml/src/ggml-cuda/fattn-common.cuh` `launch_fattn` hint to **block 10**.
   Default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` as the kill switch (the old "block 15" note in the
   beta README is pre-V2 numbering; use the V2 ownership).

3. **Finish the upstream `MUL_MAT_ID` duplicate-ids fix** in `wip/upstream-mul-mat-id-dup/`. Fix the
   F16 `dup=1` path and the `quantize_mmq_q8_1` fault, validate with `test-backend-ops`, and produce
   `upstream/UPSTREAM-PR-mul-mat-id-dup.md` + `.patch` against unadulterated master. **Do not add it
   to `patches/`**; the delivery ships pure. Then archive the `wip/upstream-mul-mat-id-dup/` directory
   to `archive/work/` and record it in `WORKLOG.md`.

The fold method is the r52 one: build a canonical chain from `release.json.tip` (a worktree on a new
branch), `git rebase -i 097cf33d9`, `edit` the owning block, `git am` the PR/beta hunk at that
intermediate state, `git commit --amend --no-edit`, `git rebase --continue`. Verify each hunk with
`git apply --check` at its target block before the rebase. Then `scripts/make-patches.sh` and
`scripts/make-release.sh --release v12-097cf33d9-r53`, `scripts/validate-set.sh`, and
`scripts/gate-per-block-build.sh --changed-from <r52-tip>` (this only builds the changed blocks; do
not run the full 12-block sweep).

Run the full r52 release gate set on the new build (prefill-logit, qwen4exp quant-coherence, coherence,
the 35B-A3B MTP gate, width purity, `TOPK_MOE`, `MUL_MAT_ID` mmvq, dense batched B=1/4/8) plus the
PR/beta-specific checks in the handover. Update all the current-state docs (`WORKLOG.md`,
`ENVIRONMENT.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`, `patches/README.md`), add `RESOLUTION.md`
for each promoted campaign, and update `wip/CAMPAIGNS.md`.

**No Docker build for r53** (deferred to a later tag; confirm this). Commit, tag
`v12-097cf33d9-r53`, push the delivery `main` + tag, refresh the personal fork `rdna-boosts` branch and
`git push --force-with-lease origin rdna-boosts`, then reply to and close PR #144 and post the issue
#138 outcome (no em-dashes).

Report all findings and any design questions to me before any delivery-affecting commit, tag or push,
and stop to ask about anything that is not a straightforward fold. The r52 record in `WORKLOG.md`
(261013.002) is the model for the release shape.
