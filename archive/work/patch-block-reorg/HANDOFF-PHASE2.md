# HANDOFF — patch-block-reorg, Phase 2 (cold start)

**Written 2026-10-12, end of the Phase-1 session.** This is the cold-start entry point for **Phase 2**.
Read this first, then `PHASE2-REFACTOR.md` (the Phase-2 work plan), `PHASE1-STATUS.md` (§18–§23 are this
session's findings/decisions), `REBASE-RESILIENCE.md`, `FILE-OWNERSHIP.md`, and `AGENTS.md` ("Report
findings before shipping", "WIP and promotion rules", "Critical facts").

## TL;DR

**Phase 1 is COMPLETE.**  `v2-phase1` = **`a40841e1e`**: **12 blocks (00–11)**, per-block fresh `all`
build **12/12**, and the tree is **byte-identical to the published delivery**
(`git diff a40841e1e af0595360` is empty; `release.json.tree` = `f102245dfb116f02f6b3f572df7fb413a5c65cf8`).
Prefill-logit **0.000707 / 98.755 %**, 4B same-seed and qwen4exp dense+sparse coherence == r50.  The
delivery `patches/` and `release.json` are **untouched**.

**Phase 2 = the self-contained refactor (G8):** extract each concern into its own function/file, leaving
only the call point in the shared/churning file.  V2.0 = Phase 1 + Phase 2 (`PHASE2-REFACTOR.md` is the
working list).  Start with `ggml_cuda_try_fuse`'s arms (highest value, lowest risk) and the maintainer's
`ggml-cuda.cu` split.

## 1. Ref map (exact)

| ref | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts` (branch `main`, **pushed**) |
| Phase-1 worktree | `~/llama-v2` (worktree of `~/llama.cpp`), branch **`v2-phase1`** = **`a40841e1e`** (blocks 00–11, complete, clean) |
| clean checkpoint tag | `pin/v2-phase1-00-04` = `2a1502311` (blocks 00–04) |
| scratch | branch `v2-block10` = `8ddece0a6` (block-10 development scratch; ignore) |
| reference release | `release.json` base `097cf33d9` (tree `30c400cc…`), tip `af0595360`/`84058162b` (tree `f102245d…`) |
| reference build (r50) | `~/llama-rebase/build-r50` (the r50 delivery build, `-DGGML_HIP_RCCL=1` etc.) |
| upstream-base build | `~/llama-base/build-base` (base `097cf33d9`, used for "is this a delivery regression?" checks) |
| per-block gate build | `~/llama-v2/build-per-block` |
| ad-hoc build dir | `~/llama-v2/build-b10` |
| canonical chains | `~/llama.cpp` (rdna-boosts), `~/llama-rebase` |

Build: `cd ~/llama-v2 && BUILD_DIR=<dir> EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
(note: the script `rm -rf`s the build dir; **ccache makes warm rebuilds fast**).

## 2. Phase-1 block map (all committed, all gated)

| # | alias | commit | notes |
|---|---|---|---|
| 00 | foundation | `4143c62af` | band-uniformity, Meta-wrapper, allocator W4, readback |
| 01 | allreduce | `320fecf0d` | hybrid HIP all-reduce |
| 02 | mtp-adaptive | `8401ee71f` | adaptive MTP draft depth |
| 03 | gdn | `8f244219d` | chunked gated-delta-net |
| 04 | bf16-kv | `2a1502311` | BF16 KV cache + native-BF16 FA |
| 05 | fa | `05e88da52` | FA band + native K/V + derived kq-mask (no arena) |
| 06 | mmb | `298060c73` | MMB quantized prefill GEMM |
| 07 | kquant | `95d15bad9` | mmvq VDR, Q6_K MMQ, Q1_0, grid pad |
| 08 | fused-kernels | `5ae8cb848` | old-08 fusions + old-13 fused MoE + old-15 PR57/63 |
| 09 | slab-moe | `0a5d26280` | host-resident MoE (iface/scheduler/Meta/slab/cache) |
| 10 | qwen4exp | `27f159a7a` | qwen4exp + QSA (merged: QSA is intrinsic to the arch) + HC16 + GDN beta-sigmoid + lazy reader |
| 11 | campaign-remainder | `a40841e1e` | CPU wins (sampling/quants/repack/arch-fallback), ggml-alloc plumbing, backend iface NULL fills, DFlash F1, FA build grouping |

**Per-block gate:** `~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama-v2 --base
097cf33d9 --tip "$(git -C ~/llama-v2 rev-parse v2-phase1)" --build-dir build-per-block` → **12/12**.

### Gate results on the Phase-1 tip (this session)

| gate | result |
|---|---|
| per-block fresh `all` build | 12/12 |
| tree == `release.json.tree` | exact (`git diff a40841e1e af0595360` empty) |
| `scripts/gate-prefill-logits.sh` (with `LLAMA_PPL=~/llama-v2/build-per-block/bin/llama-perplexity`) | mean KLD **0.000707**, same-top-p **98.755 %** PASS |
| 4B Q8_0 same-seed greedy vs r50 | identical |
| qwen4exp (`Qwen3.8-Flash-Next-Q4_K_M -ngl 99 -sm tensor`) dense **and** sparse vs r50 | identical (see §5 for the commands) |
| `test-backend-ops -o FLASH_ATTN_QSA` | 22/22 |
| `test-lazy-reader` | all passed |
| `test-llama-archs -a qwen4exp` GPU rows | all OK |

## 3. Phase-2 goal and rules (G8)

Extract each concern into a **self-contained function (ideally its own file)**; leave only the **call
point** in the shared/churning file.  Full rules in `PHASE2-REFACTOR.md` §0.  The headline targets, in
order:

1. **`ggml_cuda_try_fuse` arms** (`ggml-cuda.cu`) — the biggest hub interleave.  Each arm's matcher body
   moves to the concern's file; `try_fuse` becomes an ordered list of guarded one-line calls.  **Arm order
   is semantic** (which fusion wins) — the extracted sequence must reproduce it exactly.
2. **`mmvq.cu` dispatch** (`calc_nwarps` / `calc_rows_per_block*` / item-split / VDR) → a feature-local
   helper (`mmvq-rdna.cuh`); band-uniformity (`GREEDY-PURITY.md` §§19/24/25) is the gate.
3. **MMQ `J` selection / config rows** (`mmq.cu`) → a feature-local config function/table.
4. **FA `type_KV`/band/native/derived chooser** (`fattn.cu`/`launch_fattn`) → predicates in a
   `fattn-rdna.cuh`; the type-axis instances stay in the generated TUs (G6).
5. **Context/loader glue** (`llama-context.cpp`, `llama-model-loader.cpp`) → named helpers with one call.
6. **Env reads** — each cached read stays in its owning block's file (never a central env file).

The maintainer's specific observation (record in `PHASE1-STATUS.md` §14): split `ggml-cuda.cu` into
`ggml-cuda-try-fuse.cu`, `ggml-cuda-slab.cu`, `ggml-cuda-moe-cache-hooks.cu`, `ggml-cuda-arena.cu`.

## 4. Phase-1 deviations worth revisiting in Phase 2

- **Block 11 is a catch-all** (G1 deviation): CPU campaign wins + backend iface NULL fills + DFlash F1 +
  the FA build grouping have no single concern.  Consider splitting or re-homing.
- **The planned separate `mmap-reclaim` block vanished**: taking r50's `llama-model-loader.cpp` into
  block 10 carried `LLAMA_LOAD_RELEASE_MMAP` with it.  Peel it out into its own block if it matters.
- **Block 10 = qwen4exp + QSA** (merged on the maintainer's instruction; QSA is intrinsic to the arch).
  The planned `11 qsa` no longer exists.
- `ggml-cuda.cu` is still the biggest hub; Phase 2's `try_fuse`/slab/cache-hooks/arena split is the fix.

## 5. Known DELIVERY defects (not reorg artifacts — do not "fix" them in Phase 2 without the maintainer)

- **`TODO.md` #56** — `test-llama-archs -a deepseek2` GPU memory fault with host experts + `--moe-cache-mib`;
  present in the published tip (r50), absent at the base.  Repro: `test-llama-archs -a deepseek2 -s 42`.
- **`TODO.md` #57** — `test-llama-archs` **CPU mixed-batch FAIL**: r50 `FAIL (3.52e-01)` vs base `OK
  (3.32e-14)`; the v2 chain reproduces r50 bit-for-bit.

Both are pre-existing in the delivery; the reorg faithfully reproduces them.  They are evidenced by
comparing against `~/llama-base/build-base` (passes) and `~/llama-rebase/build-r50` (fails).

## 6. Method / pitfalls (learned hard this session)

- **Run builds in the FOREGROUND** with a generous `timeout`, then grep the log.  ccache makes a warm
  build finish in well under a minute; never background + sleep.  Builds are non-optional after every
  edit.
- **Amending a block cascades hashes.**  Amending block NN requires `git rebase --onto <new-NN>
  <old-NN> v2-phase1` for the later blocks; expect conflicts in `ggml-cuda.cu` (the destructor and the
  `try_fuse` arms are common).  Resolve by keeping **both** sides where the tip keeps both.
- **After any amend, verify the committed tree**: `git diff <block> <next-block>` / `git show <block>:<file>`.
  This session a `git checkout <tip> -- <paths>` set of file replacements was silently not committed once;
  `git diff --stat <commit> af0595360` (should be empty at the tip) is the check.
- **`git checkout --theirs -- <file>`** in a `-3` conflict takes the WHOLE stage-3 file, not the conflict
  region.  Resolve by hand.
- **Never keyword-triage the squashed old-06/13/15 diffs** without reading every hunk — a `band` keyword
  matched the GLU→Q8_1 fusion; a QSA keyword misses HC.  `git log -S '<symbol>' <tip> -- <file>` finds the
  introducing block; `tools/select_hunks.py '<regex>'` extracts hunks but you must **read** them.
- **`llama-cli` MUST use `--single-turn`** (and `--no-display-prompt` for scripted output), wrapped in
  `timeout`.  The gdb crash handler prints a backtrace on abort; rely on it only for the assert message.
- **Compare against r50 AND the base**: r50 = the delivery (what the v2 chain must reproduce);
  `~/llama-base/build-base` = upstream (tells you whether a failure is a delivery defect or a reorg one).
- **Same-seed coherence alone is not enough**: it is blind to a uniform prefill shift — run
  `scripts/gate-prefill-logits.sh` for any prefill-kernel change.
- The qwen4exp coherence command (both arms; `LLAMA_QSA_OFF=1` forces the dense arm):
  ```bash
  HIP_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:$B/bin \
    timeout 500 $B/bin/llama-cli -m /llm/models/Qwen3.8/Flash-Next/Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00005.gguf \
    -ngl 99 -sm tensor -c 4096 -p "The capital of France is" -n 24 --seed 42 --temp 0 \
    --no-display-prompt --single-turn | sed 's/\x1b\[[0-9;]*m//g' | sed -n '/The capital of France is/,$p' | tail -n +2
  ```
  A prefill-crossover prefill run uses `-n 1`.

## 7. Process for each Phase-2 extraction

1. Pick one concern; put its logic in a function (ideally its own file); leave one call point.
2. Keep the shared file clean: `#include` + the guarded call only.
3. Rebuild the **owning block** fresh `all`; then the later blocks (`gate-per-block-build.sh` from the
   owning block's parent to the tip).
4. Run the feature's gate (`PHASE2-REFACTOR.md` §1 table) + a same-seed/byte-identity check.
5. **One extraction per commit**, inside the owning block (amend + rebase later blocks), or a sub-commit
   the release cut squashes.
6. Commit campaign docs to `~/llama-cpp-rdna-boosts` (`main`, push) as you go; surface any structural
   finding and wait (AGENTS.md).

## 8. Definition of done (V2.0)

- `ggml_cuda_try_fuse` is an ordered list of call points; each feature's matcher lives in its file.
- Max modified-upstream-file count per block drops (re-measure per `REBASE-RESILIENCE.md` §7).
- No forward deps (per-block `all` build N/N); `validate-set.sh` green.
- `patches/` regenerated (`make-patches.sh` + `make-release.sh`); full release gates green;
  fork `rdna-boosts` refreshed + container images per the Pushing policy (release only).
- Current-state docs re-issued; dated records stay historical.
