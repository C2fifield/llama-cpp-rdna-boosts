# `rs-rollback-boundary` — handover for the gfx1201 box

Everything needed to extract, apply, build and gate this change on the primary box.  It was developed and
measured on **gfx1151** (Strix Halo); the gfx1201 box does the cross-arch gates and the release integration.
Background, evidence and the reasoning are in [`README.md`](README.md) §7-§13 — read §13 first.

## 0. What the two patches are

| patch | files | what |
|---|---|---|
| `0001-rs-rollback-per-sequence-tracker.patch` | `src/llama-memory-recurrent.cpp` | The per-sequence rollback tracker folded **every** ubatch of the prepare list in, instead of `ubatches.back()` only.  Fixes the issue-#131 "rollback crossed a batch boundary" **false positive** (a sequence that ran in a non-final ubatch kept an older, prefill-sized count).  Also adds the `LLAMA_RS_ROLLBACK_DEBUG=1` `RSPREP`/`RSROLL` instrumentation used to prove it.  (README §7) |
| `0002-gdn-chunked-prefill-snapshot-tail.patch` | `ggml/src/ggml-cuda/gated_delta_net.cu` | The **real fix**: a chunked GDN prefill now runs the sequential kernel on a trailing `D = max(K>16?K:16, n_rs_batch)` tokens inside the same launch, so snapshot slots `0..D-1` exist for every batch and a rollback can never read a state the kernel did not write.  No extra memory, no extra graph, ubatch unchanged.  (README §13) |

They touch disjoint files and are independent; apply in numeric order.  `0002` supersedes the splitter
variant that was tried first (README §12 — correct but −13.7 % pp512; reverted, not in the set).

Both are cut against the **`release.json` r89 tree** (`base` `097cf33d9`, applied tree
`ca9598a22946e52b74a8d6869a1ff917b181455b`) and were verified to `git am` cleanly onto a pristine
`rdna-boosts` checkout, reproducing the developing tree byte-for-byte.

```bash
# on a clean ~/llama.cpp at the release tip (or a fresh clone + scripts/apply-all.sh)
cd ~/llama.cpp
git checkout -b wip/rs-rollback-boundary rdna-boosts
git am /path/to/wip/rs-rollback-boundary/000*.patch
~/bin/build-llama-rocm                      # rebuild llama-cli/-server/-bench/-perplexity/test-*
```

## 1. Environment knobs (both default to the safe/new behaviour)

| variable | default | meaning |
|---|---|---|
| `GGML_CUDA_GDN_TAIL` | on | `0` restores whole-batch chunked prefill (the old, unsafe behaviour).  Use as the A/B base arm and as the bisect knob. |
| `LLAMA_RS_ROLLBACK_DEBUG` | unset | `1` logs `RSPREP` per prepared ubatch and `RSROLL` per partial rollback (verbose; diagnosing only). |

Nothing else changes; no new allocation, so `ENVIRONMENT.md` needs these two rows plus
`MOE_EXPERT_CACHE_UNIFIED` (the other campaign's patch, see §5) at promotion time.

## 2. Gates, with the gfx1151 numbers to compare against

Run in this order; the last two are the release blockers.

1. **Backend oracle** — `HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/test-backend-ops -o GATED_DELTA_NET`
   → expect `2/2 backends passed` (gfx1151: OK).
2. **The bug itself** — `tools/prefill-rollback.cpp` (build line in README §8), 35B-A3B, `RS=3 K=1`:
   `prefill-rollback <model> <prompt> 2048 512` and `2050 512`.
   * `GGML_CUDA_GDN_TAIL` unset → **PASS at 2050 (0.0)**, ~1e-3 NMSE at 2048 (width parity, expected);
   * `GGML_CUDA_GDN_TAIL=0` → **0.65 NMSE at 2048** (the corrupt state this fixes).
3. **Prefill logits, against the exact reference** (the important one — do **not** use the default
   `--ctx 512`: each chunk resets, so the tail is invisible and the gate reports exactly 0):
   ```bash
   GGML_CUDA_GDN_CHUNKED=0 ./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED=0" \
       --ctx 2048 --chunks 16 --model /llm/models/Qwen3.6/35B-A3B/Q3_K_M/...gguf --ppl ~/llama.cpp/build-rocm/bin/llama-perplexity
   ```
   gfx1151: **KLD −0.000000, same-top-p 99.994 %** (the new path is at the exact reference).
   For contrast, the old arm (`GGML_CUDA_GDN_TAIL=0`, same base) is **0.002777 / 98.173 %** — so the
   change *improves* prefill fidelity and must not be judged against the old arm.
4. **Purity** — `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3`, seed 42, greedy,
   ≥64 tokens on the 35B: byte-identical text (gfx1151 `13497757a7bc…`).  The tail is a fixed `D`, so
   plain and MTP chunk a prefill identically.
5. **Recurrent suite** — `test-recurrent-state-depth` (**153 failures = the gfx1151 box baseline**, not the
   `AGENTS.md` 130 — establish your own baseline first) and `test-recurrent-state-rollback` (**0 FAIL**).
6. **The issue-#131 reproducer on 2 GPU** (the arm that originally recorded the warning):
   ```bash
   archive/work/moe-verify-fusions/tools/multiseq.sh <build> <tag>      # GPUS=0,1 M=.../Flash-Next/... D=.../mtp-... QK
   ```
   Expect **`rollback-crossed-warning count: 0`** with the per-sequence values now reported per ubatch.
   This is the one result this box cannot produce.
7. **Adaptive-MTP four-axis gate** (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`) — 35B-A3B
   `-ncmoe 99`, 2 GPU tensor, `none == n1 == n3 == n7` short and `none == n3` long.  gfx1151 could not run
   this (single device, arena off).
8. **Perf** — `llama-bench -p 512,2048 -n 128 -r 3`, `GGML_CUDA_GDN_TAIL` off vs on.  gfx1151:
   pp512 2096.7 → 2091.5 / 2073.3 → 2076.8 (noise), pp2048 2089.8 → 2080.4 (**−0.45 %**), tg128 flat.
   On gfx1201 the tail is a 16-token fp32 sequential GDN per prefill batch — expect the same shape; if it
   shows more than ~1 %, that is a finding, not a variant.

## 3. What to watch for

* **The `--ctx` trap in gate 3** — at the default `--ctx 512` the tail is invisible (KLD exactly 0) because
  each chunk restarts from an empty context.  That is not a pass.
* **`test-recurrent-state-depth` baseline** — the failure count is box/model specific; diff against your own
  before/after, not against 130 or 153.
* **gfx1201 uses the same bf16 chunked GDN kernel**, so the mechanism is identical; the k-quant/MoE shapes
  differ, so the *number* will not be identical.
* `GGML_CUDA_GDN_TAIL=0` must still reproduce the old behaviour exactly (it is the fallback and the A/B).

## 4. Promotion shape (for the release integration)

* `0002` belongs to **block 03** (chunked GDN prefill); `0001` to **block 02** (the per-sequence guard folded
  in r42).  Two block amendments, no new block.
* Suggested commit split: one commit per patch, both with the campaign link, so `make-patches.sh` keeps the
  block boundaries.
* `ENVIRONMENT.md`: add `GGML_CUDA_GDN_TAIL` (kill-switch, default on) and `LLAMA_RS_ROLLBACK_DEBUG`
  (diagnostic, default off).  `patches/README.md` block 02/03 sections get one paragraph each.
* The campaign's `README.md` §7-§13 carries the evidence; on promotion move the whole directory to
  `archive/work/rs-rollback-boundary/` with a `RESOLUTION.md` (no `wip/` stub) and record it in `WORKLOG.md`.

## 5. The independent second patch set (same session, different bug)

`wip/umem-arena-guard/umem-arena-guard.patch` (sha256 `b05afe53…`) is **not** part of this set: it refuses
the MoE expert arena on a shared host/device memory pool (an APU), and must not be judged on gfx1201 beyond
"no-op".  Its check there is: one 35B-A3B `-ncmoe 40` load with `MOE_EXPERT_CACHE_MIB` unset must still log
*"auto cache preflight … cache stays on"* and arm the slab (i.e. a discrete GPU is untouched).  Its own gates
are in that campaign's README §3.
