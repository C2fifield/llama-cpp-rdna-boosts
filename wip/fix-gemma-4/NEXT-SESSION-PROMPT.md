# Prompt for the next session — TODO #68 1a, the config nits, then promotion + release

Paste everything between the `---` markers into the next session.

---

We are continuing the **`wip/fix-gemma-4/`** umbrella campaign in `~/llama-cpp-rdna-boosts` (delivery repo)
against the fork checkout `~/llama.cpp` (branch `rdna-boosts`, built with `~/bin/build-llama-rocm`, ROCm
10.1.0, gfx1201 box = 3x R9700, 16 cores, models under `/llm/models/`).  Goal for this session: **finish the
campaign** — close TODO #68 1a, settle three config nits, then make the promotion cut and take it through the
release ritual.

**Cold start, in this order (do not skip):**

1. `AGENTS.md` — reporting rules (surface findings to me *before* any delivery-affecting commit), the
   WIP/promotion rules, the default-on policy, and "Critical facts".  Note the **new hipBLASLt bullet**:
   a fresh-process comparison is only valid with `ROCBLAS_USE_HIPBLASLT=0`, and never judge determinism on a
   text hash.
2. `wip/fix-gemma-4/HANDOVER-2026-10-21c.md` — **the current record**: what is closed (determinism → external),
   what is fixed (`0007` `--fit` draft-KV, `0008` gemma4 `-sm tensor`), what is left (§2), and the promoted-cut
   checklist (§2.3).
3. `wip/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21e.md` — the two fixes and their gates, including the
   `ne_per_dev=[0,512]` diagnosis behind the `-sm tensor` fix.  §3 lists the three config nits.
4. `wip/fix-gemma-4/patch-state/README.md` — the **9-patch set** (`0001..0008`) that applies to `972f3b757`
   and reproduces the current tree, with sha256s and the re-verification commands.
5. `wip/fix-gemma-4/README.md` §2.4 (the non-regression contract) and `HANDOVER-2026-10-21.md` §6 (the full
   promotion checklist), plus `evidence/MEASUREMENTS-2026-10-21d.md` if determinism comes up.

**Current state (2026-10-21 session c).**  Committed and pushed on delivery `main` (`a5e551f`, `f1d67d9`):
the determinism close (`AGENTS.md` + `GREEDY-PURITY.md` §44 + `README.md` issue #67 + `WORKLOG.md`) and the
campaign's `0007`/`0008`.  The tree is `972f3b757` + `patch-state/0001..0008`, built, `test-llama-archs`
**644/644**.  Headline numbers (all with `ROCBLAS_USE_HIPBLASLT=0`):

- gemma4 26B-A4B `-ncmoe 40 -sm layer`, 2 GPU, `-n 300`: plain **74.7** t/s, MTP n3 **93.0** t/s at the
  **default** context (it was 41.0 before `0007`; arena 23141 MiB / 7680 slots / `h=0.0000`).
- gemma4 26B-A4B `-ncmoe 40 -sm tensor`: **works** (gen 72.2 t/s) — unmeasurable before `0008`.
- gemma4 12B `-sm tensor`: runs, `c740be0e131a` (143 chars) byte-identical to `-sm layer` at 2 and 3 GPU.
- MTP determinism is **closed as external**: ROCm's hipBLASLt per-start solution pick (issue #67), not our
  code.  `0005` (per-origin async-input ring) is still a real fix and ships.

**Work items, in this order (my ordering):**

1. **TODO #68 1a — rotated q8_0 KV under `-sm tensor`** (still aborts):
   ```
   ggml/src/ggml-backend-meta.cpp: unsupported mul_mat split states:
     src0=Meta(ROCm0,ROCm1)#attn_inp_k_rot#0 axis=10 (MIRRORED)  src1=Kcur_pos-5 axis=0
   ```
   Start from the pattern `0008` just established: `Kcur`'s axis-0 split is legitimate because its rows are
   whole per device (`ne_per_dev` has no partial row), and a rotation is a per-row op — so teaching
   `handle_mul_mat` the (MIRRORED, axis-0-whole-rows) pair and returning axis 0 is the cheap path.  If the
   diagnosis says the rotation genuinely needs a cross-device reduction for some shape, the honest interim is
   the clear rejection message (the DFlash-style #78 precedent).  The four costed directions are in
   `HANDOVER-2026-10-21.md` §1a.  Bar: 12B `-sm tensor -ctk q8_0 -ctv q8_0` runs and matches `-sm layer` with
   the same KV types, plus the rotated-cache prefill-logit gate (recorded base `gemma4 q8_0 rotated
   -0.000000 / 99.990 %`).
2. **The three config nits** (`evidence/MEASUREMENTS-2026-10-21e.md` §3) — decide fix-now or document, and
   tell me which: (a) `-md <gemma4-assistant> --spec-type none` aborts on `ubatch->embd_state` (a trap for a
   "plain baseline with the drafter loaded" measurement); (b) the `E`-level `requires ctx_other` fit message
   should be a warning when the extra model declares shared KV; (c) `(file*).moe-heatmap` is written into the
   CWD by `test-llama-archs` — the shipped copy is deleted at the cut.
3. **Re-run the campaign gates on this tree** (before touching `patches/`), all with `ROCBLAS_USE_HIPBLASLT=0`
   on both sides of any A/B: `README.md` §2.4 cells (dense 27B 1-GPU + 3-GPU tensor with q8_0 KV, 4B, gemma4
   12B f16 **and** rotated q8_0, 35B-A3B `-ncmoe 40` 2-GPU tensor — text + tg128 flat), the 35B-A3B cache
   canary (**240 tables / 61440 slots / 18662.0 MiB**), `test-llama-archs` 644/644, the prefill-logit gates
   (27B `0.000707 / 98.755 %`, gemma4 f16 `-0.000000 / 100.000 %`, gemma4 q8_0 rotated `-0.000000 /
   99.990 %`), and the four-axis adaptive-MTP gate at `-n 3000` on 35B-A3B and dense 27B
   (`benchmarks/mtp-adaptive-methodology.md` rules 0-5, incl. `prompts/code-reasoning-mixed.txt`).
4. **Promotion cut** (`HANDOVER-2026-10-21c.md` §2.3): fold into **block 09** `0002`, `0003`, `0004a`,
   `0004b`, `0005`, `0006`, **`0007`**, **`0008`** + the `(file*).moe-heatmap` deletion (one amendment);
   fold into **block 10** `0001` (#76).  No new block.  `scripts/make-patches.sh` + `scripts/make-release.sh`
   from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh` (never hand-edit hashes),
   then `scripts/validate-set.sh` from a fresh tarball, `scripts/gate-per-block-build.sh` for the amended
   blocks, and re-run item 3's gate set on the **folded** tree.  Docs: `patches/README.md` (blocks 09/10),
   `ENVIRONMENT.md` (`GGML_META_SEG_ASYNC`, the iSWA `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` note), a
   `WORKLOG.md` entry (newest first, `YYMMDD.VVV`), `TODO.md` loses #76/#77 (and #68 once 1a lands or is
   parked), archive the campaign to `archive/work/fix-gemma-4/` with a `RESOLUTION.md`, delete
   `wip/fix-gemma-4/`, update `wip/CAMPAIGNS.md` **with no stub**.
5. **Release ritual** (`AGENTS.md` "Pushing policy"): tag, then refresh the fork's `rdna-boosts` branch (clean
   apply at `release.json.base`, build, gates, `git push --force-with-lease origin rdna-boosts` — **personal
   fork only**), and the container images are built off-host by the worker watcher on `radiance` — do not run
   the full image build on the primary box.

**Ground rules for this work**

- **Report findings before shipping.**  Anything material (a new defect, a measurement that contradicts a
  claim, a scope change, a purity question) comes to me *before* it goes into `patches/`, a commit message or
  a release.  I decide fix-now / defer / drop.  Ask before any push.
- **Every measurement from here on starts with `ROCBLAS_USE_HIPBLASLT=0`.**  Otherwise per-start hipBLASLt
  solution selection moves logits *and* throughput between runs (3 classes in 18 starts of one config), which
  is exactly what cost three sessions.  It stays **enabled** in the delivery (disabling it costs up to -15 %
  decode); it is the measurement/bisection switch.
- **Instrument discipline.**  Run-to-run determinism: logit-level only (`tools/fp.sh`,
  `archive/work/kv-cell-offset-nondeterminism/tools/decodestart.sh`), N ≥ 6 fresh processes, state N.
  Cross-build **identity** checks may use text (`llama-cli` without `-v` + `scripts/extract-generated.py`) —
  never determinism, and never on a `-v` log.
- **Non-regression is a hard gate**: `wip/fix-gemma-4/README.md` §2.4; the 35B-A3B cache configuration is a
  canary, and a per-box/per-model baseline is never compared to a number in a doc.
- **Beneficial behaviour is default-on**, env vars only disable.  A fix that only moves the failure is not a
  fix — say so and hand it back rather than shipping it.
- **Measurement hygiene**: one job at a time, never a parallel/background bench; `-t 8` (GPU IRQ cores 13-15
  free); `llama-cli --single-turn --no-display-prompt`; MTP gates at `-n 3000`; check logs with
  `scripts/check-gpu-faults.sh`; a change to a prefill kernel or a default-on approximate path needs
  `scripts/gate-prefill-logits.sh`.
- **Never push** to upstream `ggml-org/llama.cpp`.  Only this delivery repo and my personal fork.

**Definition of done:** TODO #68 1a is closed (or parked with the reason recorded), the three nits are decided,
and the campaign is **either released** (`v12-*` tag + fork refresh + worker image build, campaign archived,
`wip/` entry deleted, `CAMPAIGNS.md` updated) **or** promotion-cut-complete and validated, waiting only on my
push decision — with the full gate set green on the folded tree and the results reported to me before the tag.

---
