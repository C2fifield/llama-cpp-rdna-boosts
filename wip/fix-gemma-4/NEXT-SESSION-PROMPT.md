# Prompt for the next session — drive the MTP-determinism item to a conclusion

Paste everything between the `---` markers into the next session.

---

We are continuing the **`wip/fix-gemma-4/`** umbrella campaign in `~/llama-cpp-rdna-boosts` (delivery repo)
against the fork checkout `~/llama.cpp` (branch `rdna-boosts`, built with `~/bin/build-llama-rocm`, ROCm
10.1.0, gfx1201 box = 3x R9700, 16 cores, models under `/llm/models/`).  Goal for this session: **take the
MTP run-to-run determinism item from "narrowed" to "closed"**, i.e. a reproducible N/N-identical
configuration, then make it fast, then hand the whole campaign back for the promotion cut.

**Cold start, in this order (do not skip):**

1. `AGENTS.md` — reporting rules (surface findings to me *before* any delivery-affecting commit), the
   WIP/promotion rules, the default-on policy, and the "Critical facts" invariants.
2. `wip/fix-gemma-4/HANDOVER-2026-10-21b.md` — **the current record**: the instrument, the reproduction
   commands, what is established, the two retractions, the 7-patch set, the next bisect, and the deferred
   items (§4 and §7 are the work).
3. `wip/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21c.md` — the numbers and the retractions.  Read §0 first:
   **a greedy text hash is not a determinism instrument on this box**; the earlier text-based conclusions are
   void.
4. `wip/fix-gemma-4/patch-state/README.md` — the 7-patch set (applies to `972f3b757`, reproduces the current
   tree byte-for-byte), with the sha256s and the re-verification commands.
5. `wip/fix-gemma-4/HANDOVER-2026-10-21.md` and `README.md` — for the *other* campaign items (the `--fit`
   arena item, TODO #68 1a/1b) and the non-regression contract.

**Current state (2026-10-21 session b).**  Item 1 (baseline) is green: the tree is `972f3b757` +
`patch-state/0001..0006`, the binary is the gated one, every non-regression cell is text-identical to
`build-r99`, the 35B-A3B cache canary is unchanged (240 tables / 61440 slots / 18662.0 MiB), and all three
prefill-logit gates reproduce their recorded numbers exactly.  The **MTP determinism** item is *not* solved
but is properly instrumented and narrowed:

- the divergence is in the **draft model's own logits** (dense, no expert cache), at the **first draft call**,
  as a *continuous* probability change (0.439 / 0.453 / 0.458 — not garbage);
- it needs **two llama_contexts on two devices with captured graphs**: a single context (with the whole
  gemma4 host-experts + cache path) is **bit-identical 6/6** (`decodestart.sh`), and MTP on **1 or 3 devices**
  and with **`GGML_CUDA_DISABLE_GRAPHS=1`** is stable;
- one real cross-context bug is fixed (`0005`: the async-input ring's ordering state was per *device*; a
  stream is a llama_context, so the draft and target consumed each other's queued uploads).  That took the
  2-device arm from *unstable in every sample* to **8/10**; it is still not a closed item.

**Work items, in this order (my ordering):**

1. **Close the determinism item.**  Start from handover §4.  First separate "two stable values" from "a
   lottery": if the two fingerprint families (`6f6923ef36` vs `10e1f21f4c`) are each stable, the *draft's
   captured graph differs* — a deterministic capture bug, far easier than a race.  Then bisect the remaining
   per-device shared allocations both contexts touch (op-offload H2D staging ring, FA staging arena), and
   evaluate a **draft-context-only graph switch** as both a diagnostic and a plausible narrow fix (the draft
   graph is a 1-token decode; the target's is the one that matters for throughput).  Bar: **10/10 identical
   `tools/fp.sh` fingerprints** on the 2-device MTP config, plus the same at `-n 3000`, plus the target-only
   `decodestart.sh` still 6/6, plus the non-regression contract unchanged.  **Do not** re-try the arms listed
   in handover §4 "do not re-try" (all measured, no effect).
2. **Then make it fast** — the fix must not cost the campaign's throughput (the reference numbers:
   gemma4 26B-A4B `-ncmoe 40` `-sm layer` ~75 t/s, `-sm tensor` ~73 t/s; MTP ~89 t/s on that config; and the
   35B-A3B `-ncmoe 40` 2-GPU tensor cell must stay flat).
3. **Then the deferred items in handover §7, in that order**: the `--fit` arena shortfall under MTP, TODO #68
   1b (the f16 `handle_per_row` assert, which unblocks any `-sm tensor` gemma4 measurement), TODO #68 1a (the
   rotation matrix), and finally the promotion cut (fold #76 → block 10, #77 A/B/C + `0005`/`0006` → block 09,
   drop the `(file*).moe-heatmap` artifact in block 09 at that cut, regenerate with
   `scripts/make-patches.sh` + `scripts/make-release.sh`, `validate-set.sh` from a fresh tarball,
   `gate-per-block-build.sh` for the amended blocks, the full non-regression contract, the four-axis
   adaptive-MTP gate at `-n 3000` on 35B-A3B and dense 27B, docs, `WORKLOG.md`, `TODO.md` loses #76/#77,
   archive to `archive/work/fix-gemma-4/` with a `RESOLUTION.md`, delete `wip/fix-gemma-4/`, update
   `wip/CAMPAIGNS.md` with no stub).

**Ground rules for this work**

- **Report findings before shipping.**  Anything material (a new defect, a measurement that contradicts a
  claim, a scope change, a purity question) comes to me *before* it goes into `patches/`, a commit message or
  a release.  I decide fix-now / defer / drop.
- **Instrument discipline (this is the lesson of the last session).**  Judge run-to-run determinism on the
  **logit-level** observable only: `wip/fix-gemma-4/tools/fp.sh` (draft candidate logits) or
  `archive/work/kv-cell-offset-nondeterminism/tools/decodestart.sh` (single-context start-to-start).
  **Never** on a greedy text hash, and never with `scripts/extract-generated.py` on a `-v` log.  Use **N ≥ 6**
  fresh processes per arm (3-run samples produced two false conclusions last session); state N in the record.
- **Non-regression is a hard gate**: `wip/fix-gemma-4/README.md` §2.4; dense 27B (1 GPU and 3-GPU tensor),
  4B, gemma4 12B and 35B-A3B `-ncmoe 40` are the reference cells, and the 35B-A3B cache configuration
  (240 tables / 61440 slots / 18662 MiB) is a canary.
- **Beneficial behaviour is default-on**, env vars only disable (`AGENTS.md` "Default-on policy").  A fix that
  only moves the failure is not a fix — say so and hand it back rather than shipping it.
- **Measurement hygiene**: one job at a time, never a parallel/background bench; `-t 8` (GPU IRQ cores 13-15
  free); `llama-cli --single-turn --no-display-prompt`; MTP gates at `-n 3000`; run
  `scripts/gate-prefill-logits.sh` for anything touching prefill or a default-on approximate path; check logs
  with `scripts/check-gpu-faults.sh`.
- **Never push** to upstream `ggml-org/llama.cpp`.  The only push targets are this delivery repo and my
  personal fork; ask me before you push anything (the release ritual is in `AGENTS.md` "Pushing policy").

**Definition of done for this session:** the determinism item is either **closed** with the 10/10 bar plus the
non-regression contract green, or handed back with the remaining ~20 % reduced to a precise, evidenced list
(which buffer, which graph, which arm) — not "still nondeterministic".  Items 3-5 only start after item 1 is
settled, and only as far as the session allows.

---
