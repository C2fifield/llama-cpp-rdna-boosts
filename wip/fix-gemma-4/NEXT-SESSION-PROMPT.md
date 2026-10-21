# Prompt for the next session — drive `fix-gemma-4` to promotion + delivery readiness

Paste everything between the `---` markers into the next session.

---

We are continuing the **`wip/fix-gemma-4/`** umbrella campaign in `~/llama-cpp-rdna-boosts` (delivery repo)
against the fork checkout `~/llama.cpp` (branch `rdna-boosts`, built with `~/bin/build-llama-rocm`, ROCm
10.1.0, gfx1201 box = 3x R9700, 16 cores, models under `/llm/models/`).  Goal for this session: **clear the
three blocking items, then cut the campaign for promotion and take it to delivery readiness.**

**Cold start, in this order (do not skip):**

1. `AGENTS.md` — reporting rules (surface findings to me *before* any delivery-affecting commit), the
   WIP/promotion rules, the default-on policy, and the "Critical facts" invariants (slab/arena yield, the
   wholesale-fallback rule, no arena freed under an in-flight kernel, Meta split-state rules).
2. `wip/fix-gemma-4/README.md` — the campaign: what #76/#77 are, what was fixed, and the non-regression
   contract in §2.4.
3. `wip/fix-gemma-4/HANDOVER-2026-10-21.md` — **§0 my decisions, §2 the MTP determinism item, §3 item 2 the
   `--fit` item, §1 TODO #68, §5 the order, §6 the promotion checklist.**
4. `wip/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21.md` — every number from the last session.
5. `wip/fix-gemma-4/patch-state/README.md` — the 5-patch set (applies to `972f3b757`, reproduces the last
   session's working tree byte-for-byte).

**Current state.** `~/llama.cpp` has the patch set applied and built (so `git diff` there *is* the patch
set).  `#76` (iSWA KV-write fusion) is complete and gated.  `#77` (gemma4 expert cache) is fixed
(A sizing-account filter + capped slack, B async segmented upload with `GGML_META_SEG_ASYNC=0`, C
multi-slice cache tables): gemma4 26B-A4B Q8_0 `-ncmoe 40` went **7.9 -> 79.0 t/s** (`-sm layer`) and
**5.0 -> 74.7 t/s** (`-sm tensor`), both byte-identical to the prefix build, with all other models flat
(dense 27B, 4B, gemma4 12B, 35B-A3B text-identical; `test-llama-archs` 375/375; both prefill-logit gates
PASS).  Nothing is folded into `patches/` yet.

**Work items, in this order (my ordering, not negotiable without asking me):**

1. **Re-establish the baseline** (~20 min): re-apply `patch-state/` to a clean `972f3b757`, rebuild, re-run
   the coherence + prefill-logit gates from `README.md` §2.4, and confirm the gemma4 numbers still
   reproduce.  Report anything that does not.
2. **MTP must be run-to-run deterministic** (handover §2).  Today, with host-resident experts (`-ncmoe`) +
   an active expert cache + MTP + captured HIP graphs, two identical runs produce different greedy text
   (`8241a94d2f86` vs `afa48a6ea28a`).  It is deterministic with `GGML_CUDA_DISABLE_GRAPHS=1` (and then MTP
   == plain), with `GGML_CUDA_ASYNC_INPUT=0`, with `-ncmoe 0`, with `MOE_EXPERT_CACHE_MIB=0`, and with
   `--spec-draft-n-max 2` instead of 3.  This is the r69/TODO #75 captured-graph class, not something the
   cache introduced — but the cache is what makes the path fast enough to use, so it must be fixed, not
   worked around.  Bar: 6+ identical fresh-process runs with graphs on, the `-n 3000` P-axis text identical
   across 3 runs, `none == draft-mtp` at `n_max <= 7` on gemma4, and the 35B-A3B/27B 4-axis gates unchanged.
   Start from the two candidate mechanisms in handover §2 (the cache's async `t.remap_host` -> `t.remap_dev`
   staging vs the r90 "snapshot unconditionally" rule; and `g_alias_to_id` being keyed by a device-tensor
   pointer that two MTP graphs can reuse).
3. **The `--fit` arena shortfall under MTP** (handover §3 item 2) — investigate with a view to **fixing**.
   At the default context the draft + target contexts make `--fit` reserve 19.16 GiB/device, the slab is
   created at 9.31/7.50 GiB, the arena at 13.8 GiB instead of 23.1, and MTP falls to 50.2 t/s against plain
   79.3.  The slab reports 18.25 GiB of VA spare for `ggml_cuda_slab_extend`, so the capacity exists but the
   arena sizing latched before it was mapped.  Candidate directions (cheapest first) are listed in the
   handover; pick one, state the risk against the slab invariants, and measure.
4. **TODO #68 1b** — `gemma4` dense + `-sm tensor` aborts in a per-row op
   (`ggml-backend-meta.cpp:838 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` inside
   `handle_per_row`, i.e. the `RMS_NORM` family, at the warmup decode, **with the default f16 cache**).  This
   is a per-head norm over a head-split axis — a *local* reduction mis-typed as cross-device; the fix is in
   the Meta split-state reshape predicates, and it unblocks measuring any `-sm tensor` gemma4 config.
   `-sm layer` works, so it does not block items 2-3.
5. **TODO #68 1a** — the rotated-cache abort (`-ctk q8_0 -ctv q8_0`):
   `unsupported mul_mat split states … src0=MIRRORED#attn_inp_k_rot axis=10 src1=axis 0`.  Four directions
   are costed in the handover (teach `handle_mul_mat` the MIRRORED/axis-0 pair and check every `PARTIAL`
   consumer; rotate after the split gather; make `Kcur` MIRRORED for the rotation; or reject cleanly at
   params level).  Then re-run the `#77 -sm tensor` numbers — they should improve again.
6. **Promotion cut + delivery readiness** — handover §6 is the checklist.  In brief: fold #76 into **block
   10** and #77 into **block 09** (plus the `(file*).moe-heatmap` deletion **in block 09, at that cut** —
   my decision), regenerate with `scripts/make-patches.sh` + `scripts/make-release.sh`, `validate-set.sh`
   green from a fresh tarball, `gate-per-block-build.sh` for the amended blocks, the full non-regression
   contract, **the four-axis adaptive-MTP gate at `-n 3000` on 35B-A3B and dense 27B** (not yet run for this
   campaign), docs (`patches/README.md`, `ENVIRONMENT.md` for `GGML_META_SEG_ASYNC`, a `WORKLOG.md` entry,
   `TODO.md` loses #76/#77), archive the campaign to `archive/work/fix-gemma-4/` with a `RESOLUTION.md`,
   delete `wip/fix-gemma-4/`, update `wip/CAMPAIGNS.md` with no stub.

**Ground rules for this work**

- **Report findings before shipping.**  Anything material (a new defect, a measurement that contradicts a
  claim, a scope change, a purity question) comes to me *before* it goes into `patches/`, a commit message
  or a release.  I decide fix-now / defer / drop.
- **Non-regression is a hard gate**: no change may cost another model text identity or throughput.  The
  contract is `wip/fix-gemma-4/README.md` §2.4; dense 27B (1 GPU and 3-GPU tensor), 4B, gemma4 12B and
  35B-A3B `-ncmoe 40` are the reference cells, and the 35B-A3B cache configuration (240 tables / 61440
  slots / 18662 MiB) is a useful canary — an earlier draft of the sizing fix changed it to 239 tables and
  cost -3.6 % tg128.
- **Beneficial behaviour is default-on**, env vars only disable (`AGENTS.md` "Default-on policy").  `B`
  ships default-on with `GGML_META_SEG_ASYNC=0` as the bisect arm.
- **Measurement hygiene**: one job at a time, never a parallel/background bench; size `-t` to leave the GPU
  IRQ cores (13-15) free (`-t 8`); `llama-cli --single-turn --no-display-prompt`; hash text with
  `scripts/extract-generated.py` (never a raw terminal slice); MTP gates at `-n 3000`; run
  `scripts/gate-prefill-logits.sh` for anything touching prefill or a default-on approximate path; check
  logs with `scripts/check-gpu-faults.sh`.
- **Never push** to upstream `ggml-org/llama.cpp`.  The only push targets are this delivery repo and my
  personal fork, and the release ritual (tag, fork branch refresh, worker-built images) is described in
  `AGENTS.md` "Pushing policy" — ask me before you push anything.

**Definition of done for this session:** items 1-5 either fixed and gated or explicitly handed back to me
with evidence; item 6 either cut (docs + patches + validation green, campaign archived) or reduced to a
precise, reviewed list of what remains.  If you hit a wall on 2 or 3, say so early with the measurements
rather than pivoting silently.

---
