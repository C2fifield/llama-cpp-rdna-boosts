# Next-session prompt (copy-paste) — Phase 2

Paste the block below as the first message of a fresh session. It is self-contained; the session should
read `HANDOFF-PHASE2.md` first and execute Phase 2.

---

We are in `/home/stew675/llama-cpp-rdna-boosts` working the `wip/patch-block-reorg/` campaign.  **Phase 1
is complete**; this session is **Phase 2** — the self-contained refactor that makes the block set
rebase-resilient (V2.0 = Phase 1 + Phase 2).

**Read first, in this order:**
1. `wip/patch-block-reorg/HANDOFF-PHASE2.md`  ← the cold-start state, refs, block map, extraction list, method/pitfalls
2. `wip/patch-block-reorg/PHASE2-REFACTOR.md` ← the Phase-2 work plan (per-concern extraction targets + gates)
3. `wip/patch-block-reorg/PHASE1-STATUS.md` §18–§23 ← this session's findings/decisions (block 08/09 fixes, the QSA⊂qwen4exp merge, the dropped-content sweep, the completed chain)
4. `wip/patch-block-reorg/REBASE-RESILIENCE.md`, `FILE-OWNERSHIP.md`, `README.md`
5. `AGENTS.md` ("Report findings before shipping", "WIP and promotion rules", "Critical facts")
6. `TODO.md` #56/#57 (known *delivery* defects, NOT reorg artifacts — do not "fix" them here).

**State (exact).**  `~/llama-v2` (worktree of `~/llama.cpp`), branch **`v2-phase1` = `a40841e1e`**:
**12 blocks (00–11)**, per-block fresh `all` build **12/12**, and the tree is **byte-identical to the
published delivery** (`git diff a40841e1e af0595360` empty; `release.json.tree`).  Prefill-logit
**0.000707 / 98.755 %**, 4B same-seed and qwen4exp dense+sparse coherence == r50 (`~/llama-rebase/build-r50`).
The delivery `patches/` and `release.json` are **untouched**.  Block 10 = `qwen4exp` (merged with QSA —
QSA is intrinsic to the arch); block 11 is a catch-all `campaign-remainder`.

**Your task: do Phase 2.**  Following `PHASE2-REFACTOR.md` §0 (one concern per function/file; only the
call point stays in the shared file; preserve arm order; no forward deps; one extraction per commit),
start with the highest-value/lowest-risk item:

1. **`ggml_cuda_try_fuse` arms** (`ggml/src/ggml-cuda/ggml-cuda.cu`) — move each feature's matcher body
   into the concern's file (`gdn-conv.cu`, `norm.cu`, `mmvq.cu`, `mmb.cu`, `hc-mix.cu`, `indexer-score.cu`,
   `moe-expert-cache.cu`, `cpy-batch.cu`, …) and reduce `try_fuse` to an ordered list of guarded one-line
   calls.  Then the maintainer's `ggml-cuda.cu` split (`ggml-cuda-try-fuse.cu`, `-slab.cu`,
   `-moe-cache-hooks.cu`, `-arena.cu`).
2. Then `mmvq.cu` dispatch / MMQ `J` selection / FA chooser / context+loader glue, per the plan.

**Mandatory:**
- Every block must build on its own with a **fresh `all`** build:
  `~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama-v2 --base 097cf33d9 --tip "$(git -C ~/llama-v2 rev-parse v2-phase1)" --build-dir build-per-block`.
- Run the extracted feature's gate (`PHASE2-REFACTOR.md` §1 table) + a same-seed / byte-identity check;
  for any prefill-kernel change also run `LLAMA_PPL=~/llama-v2/build-per-block/bin/llama-perplexity
  ./scripts/gate-prefill-logits.sh`.
- Disable fusion for the fusion A/B: `GGML_CUDA_DISABLE_FUSION=1`.
- **One extraction per commit inside the owning block** (amend + rebase the later blocks; expect
  `ggml-cuda.cu` conflicts).  After any amend, verify the committed tree (`git diff <commit> af0595360`
  must be empty at the tip — a silent non-commit bit once this session).
- The delivery `patches/`/`release.json` are **not** modified until V2.0 is done and gated.
- Commit + push the campaign docs in `~/llama-cpp-rdna-boosts` (`main`) as you go; do **not** push
  anything to the fork without an explicit instruction.
- Surface any new structural finding to the maintainer and wait before changing the plan.

**Pitfalls:** run builds in the FOREGROUND with a generous `timeout` and grep the log (ccache makes warm
builds fast); never keyword-triage the squashed old-06/13/15 diffs — read every hunk; `git checkout
--theirs -- <file>` takes the whole file; `llama-cli` needs `--single-turn` (+ `--no-display-prompt`);
compare against **both** r50 (`~/llama-rebase/build-r50`, the delivery) and the base
(`~/llama-base/build-base`, upstream) to tell a delivery defect from a reorg one.

Report progress after each extraction (what moved + the gate result) and ask before any
delivery-affecting commit.
