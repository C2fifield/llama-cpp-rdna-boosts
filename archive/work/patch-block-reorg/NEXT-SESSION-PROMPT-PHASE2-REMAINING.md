# Next-session prompt — drive the remaining Phase-2 work to completion

Copy the block below into the next session.

---

We are in `/home/stew675/llama-cpp-rdna-boosts` working the `wip/patch-block-reorg/` campaign.
Phase 1 is complete; Phase 2 (the self-contained refactor, V2.0 = Phase 1 + Phase 2) is in progress.
Your job is to **drive the remaining Phase-2 work to completion with minimal interactivity** — do not
stop to report each step, only stop for a genuine structural finding or for the delivery-affecting
release cut.

**Read first, in this order:**

1. `wip/patch-block-reorg/HANDOFF-PHASE2-REMAINING.md` — the cold-start state, exact refs, what is done,
   the remaining items + method + pitfalls, and the V2.0 definition of done. **This is the authority.**
2. `wip/patch-block-reorg/PHASE2-STATUS.md` — the dated Phase-2 log (per-cluster gate tables, cluster A–G).
3. `wip/patch-block-reorg/PHASE2-REFACTOR.md`, `REBASE-RESILIENCE.md`, `FILE-OWNERSHIP.md`, `README.md`.
4. `AGENTS.md` — "Report findings before shipping", "WIP and promotion rules", "Critical facts".
5. `TODO.md` #56/#57 — known **delivery** defects, **not** reorg artifacts; do not "fix" them here.

**Current state (verify it before starting):** branch `~/llama-v2` `v2-phase2` = `9d6730dc8`; blocks 08/09/10
= `fcce5864e`/`96a7d0d16`/`5d472a3eb`. `v2-phase1` = `a40841e1e` is the safe rollback. Checkpoint tags
`pin/v2-phase2-ckpt1..8` and their bundles under `wip/patch-block-reorg/checkpoints/`. The delivery
`patches/` and `release.json` are **untouched**; nothing has been pushed to the fork.

**Do, in order (each step: change on `v2-phase2` → incremental build → gate → re-home into the owning
block → full per-block gate → checkpoint + docs commit/push):**

1. **Item 3 step 3 (optional):** lift the shared vec-dot/VDR selectors into `mmvq-vecdot.cuh`, then move
   `mul_mat_vec_q_moe` + `mul_mat_vec_q_moe_launch` to `mmvq-moe.cu`. If it resists after a careful
   attempt, leave it inline and say so — steps 1–2 already captured the substantive value.
2. **Item 4:** the FA band/native/derived chooser → `fattn-rdna.cuh` (bodies only; generated instances
   stay with their axis). Highest risk — clear the full FA gate set + prefill-logit.
3. **Item 5:** the `llama-context.cpp` / `llama-model-loader.cpp` glue (FA tensor-parallel hint, logits
   readback ordering, host-expert policy, mmap release) → named helpers with one call point.
4. **Item 6:** the maintainer's `ggml-cuda.cu` split (`ggml-cuda-try-fuse.cu`, `ggml-cuda-slab.cu`,
   `ggml-cuda-moe-cache-hooks.cu`, `ggml-cuda-arena.cu`) — pure function-group moves; keep the slab↔MoE
   cache invariants together.
5. Re-measure the per-block modified-vs-added metric; update `FILE-OWNERSHIP.md`'s region map; update
   `PHASE2-STATUS.md` for every step.

**Rules:**
- Re-home with `git apply -3` + `git commit --amend` + `git rebase --onto`, then **verify the rebased tip
  is byte-identical** to the validated working tree for every touched file (record new files separately —
  `git stash create` drops untracked files). After every re-home run the full per-block `all` build gate
  and the gate set in `HANDOFF-PHASE2-REMAINING.md` §6.
- One concern per commit inside its owning block; behaviour-preserving moves first; any semantic change
  gets its own A/B.
- Commit + push the campaign docs to the delivery repo `main` (`~/llama-cpp-rdna-boosts`) as you go. Tag
  a checkpoint (`pin/v2-phase2-ckptN`) + bundle after each landed step.
- **Stop and ask** only for (a) a genuine structural finding that changes the plan, or (b) the V2.0
  release cut: regenerating `patches/`/`release.json`, the fork `rdna-boosts` push, and the container
  images are delivery-affecting and need the maintainer's explicit go-ahead per `AGENTS.md`.
- Never push to upstream `ggml-org/llama.cpp`; never push anything to the fork without explicit
  instruction.

**Definition of done:** see `HANDOFF-PHASE2-REMAINING.md` §8 (all items landed and gated, full per-block
build N/N, `patches/`/`release.json` regenerated, `validate-set.sh` green, full release gates, fork
refresh, containers, docs re-issued).

When the refactor items are complete and gated, stop and present a summary with the new refs and the
gate results, and ask whether to proceed with the V2.0 release cut.
