# HANDOVER 2026-10-21 (session c) — campaign state: determinism closed, two defects fixed, 1a + promotion left

**Read first:** `AGENTS.md` (reporting rule, WIP/promotion rules, default-on policy, "Critical facts" —
note the new hipBLASLt bullet), then this file, then `evidence/MEASUREMENTS-2026-10-21d.md` (the determinism
root cause) and `evidence/MEASUREMENTS-2026-10-21e.md` (today's two fixes).  The older records
(`HANDOVER-2026-10-21b.md`, `...-21.md`) still hold for anything this file does not restate.

**Box:** gfx1201, 3x R9700, **ROCm 10.1.0**, 16 cores, Linux.  Tree `~/llama.cpp` @ `rdna-boosts`
`972f3b757` **+ `patch-state/0001..0008`** (nine patches; `git diff` there *is* the set), built with
`~/bin/build-llama-rocm`.

**Measurements from here on: prefix `ROCBLAS_USE_HIPBLASLT=0`.**  Per-start hipBLASLt solution selection
(issue #67) otherwise moves logits *and* throughput between runs — 3 logit classes in 18 starts of one
config.  It is not a candidate: if an A/B moves under it, it is measuring the library.

---

## 0. One-paragraph state

**Item 1 of the session prompt (MTP determinism) is closed as external** — it is ROCm's per-start hipBLASLt
solution pick, not our code (documented in `GREEDY-PURITY.md` §44 + `AGENTS.md`; committed and pushed as
`a5e551f`).  Driving on through §7: the **`--fit` arena shortfall under MTP is fixed** (it was charging the
gemma4 drafter 18 GiB of KV it does not own: default-context MTP **41.0 → 93.2 t/s**, arena
13752 → 23141 MiB), and **TODO #68 1b is fixed** (gemma4 dense + `-sm tensor` runs; 12B byte-identical to
`-sm layer` at 2 and 3 GPU; 26B-A4B `-sm tensor` now measurable at 72.2 t/s; `test-llama-archs` 644/644).
**What is left: TODO #68 1a (rotated q8_0 KV under `-sm tensor`), three config nits, and the promotion cut
+ release.**

---

## 1. What is done (do not redo)

### 1.1 MTP determinism — CLOSED, external

ROcm hipBLASLt picks a solution **per process start** for an untuned FP32 GEMM size.  Here the only
BLASLt-dispatched GEMM in the run is gemma4's F32 **MoE router** (`m=128 n=375 k=2816 T,N`, 30 calls = one
per layer), `solution_index 140231` vs `140232` per process, and on that shape the two are **not**
bit-identical.  `ROCBLAS_USE_HIPBLASLT=0` → 10/10 identical at `-n 300`, 3/3 at `-n 3000` (904 draft calls).
**Retraction, important:** with the library path off, graphs on and graphs off land on the *same* class, so
every graph/capture arm of the two earlier sessions was the lottery, not a cause.  `0005` (per-origin
async-input ring) remains a real fix and ships.  Full numbers: `evidence/MEASUREMENTS-2026-10-21d.md`.

**Do not** "fix" the §44 finding in code, and **do not** set `ROCBLAS_USE_HIPBLASLT=0` in the delivery's
default environment (it costs up to -15 % decode; the upstream fix `rocm-libraries c50ea60130f` removes the
cause).  It is the tester's switch.

### 1.2 `--fit` arena shortfall under MTP — FIXED (`0007`)

`common_draft_reserve_estimate()` charged the drafter a full `n_ctx` KV — 18022 of the 19152 MiB reserved
per device — for a drafter that allocates **none** (gemma4-assistant shares the target's KV:
`attention.shared_kv_layers = 4`, and `llama_kv_cache`'s `layer_share_cb` reuses the target's tensors).
It now charges only the layers the draft owns (0 for `gemma4-assistant`).  **At the default 262144 context,
MTP is now as fast as the pinned-context reference:**

| gemma4 26B-A4B Q8_0 `-ncmoe 40 -sm layer`, MTP n3 | before | after |
|---|---:|---:|
| draft reserve /device | 19152 MiB | **1129 MiB** |
| arena / slots / `h` | 13752 / 4564 / 0.897 | **23141 / 7680 / 0.0000** |
| gen t/s (plain = 74.6) | 41.0 | **93.2** |

Second case: `Gemma4 31B-QAT Q4_K_XL` default context, plain 25.0 vs **MTP 51.2 t/s**.
Non-shared drafts are arithmetically unchanged (qwen4exp sidecar: `kv 563`, `(1 x 1 layers)`).

### 1.3 TODO #68 1b — FIXED (`0008`)

gemma4's **full-attention** layers have one KV head, so their 512-wide K/V row lands **whole on one device**
(`ne_per_dev=[0,512]`); the per-row ops over it (K norm, V `rms_norm`, rope) are then *local*, but
`handle_per_row` rejected any axis-0 split.  It now accepts one whose every per-device extent is empty or
the whole row (a straddling split still asserts), and `handle_bin_bcast` accepts an elementwise op against a
**MIRRORED same-shape** operand (the K-norm weight is MIRRORED; only `ne[axis]==1` was covered before).

| gate | result |
|---|---|
| 12B Q8_0 `-sm tensor`, 2 and 3 GPU, `-n 64` | **`c740be0e131a`** (143 chars) — identical to `-sm layer` |
| gemma4 26B-A4B `-ncmoe 40 -sm tensor`, plain, `-n 300` | prompt 182.8 / gen **72.2 t/s** (previously unmeasurable) |
| gemma4 26B-A4B `-ncmoe 40 -sm layer`, plain / MTP n3 | 74.7 / **93.0 t/s** |
| `test-llama-archs -s 42` | **644/644 passed** |

### 1.4 Committed / pushed

`a5e551f` on delivery `main`: `AGENTS.md` (Critical facts: pinned fresh-process comparisons, never a text
hash), `GREEDY-PURITY.md` (**§44** + invariant 9 + a §41 correction), `README.md` issue #67 (Linux 10.x *does*
reproduce), `WORKLOG.md` 261021.001, and the two evidence files with the campaign tools.  The two fixes are
**campaign patches only** (`patch-state/0007`, `0008`) — nothing is folded into `patches/` yet.

---

## 2. What is left, in order

### 2.1 TODO #68 1a — rotated q8_0 KV (`-ctk q8_0 -ctv q8_0`) under `-sm tensor`

Still aborts (re-verified this session), unchanged:

```
ggml/src/ggml-backend-meta.cpp:934: unsupported mul_mat split states:
  node=node_286 src0=Meta(ROCm0,ROCm1)#attn_inp_k_rot#0 axis=10 src1=Kcur_pos-5 (reshaped) axis=0
```

`attn_inp_k_rot` is the K rotation matrix (MIRRORED) against `Kcur` split on **axis 0** — the pair
`handle_mul_mat` does not implement.  Note 1.3 fixed the *other* half of the same shape problem: `Kcur`'s
axis-0 split is now legitimate because its rows are whole per device.  So the cheapest next step is to teach
`handle_mul_mat` the (MIRRORED, axis-0-whole-rows) pair and return axis 0 — the same reasoning as
`handle_per_row`: a rotation is per-row, so if each device holds whole rows the rotation is local.
The four costed directions (including the "reject with a clear message" interim) are in
`HANDOVER-2026-10-21.md` §1a; start with the *diagnosis* there and the 1.3 pattern.

Bar: 12B `-sm tensor -ctk q8_0 -ctv q8_0` runs, and its greedy text/logits match `-sm layer` with the same
KV types; then re-run the prefill-logit gate with a rotated cache (the recorded base is
`gemma4 q8_0 rotated -0.000000 / 99.990 %`).

### 2.2 Config nits (small; decide fix-now vs document)

1. **`-md <gemma4-assistant> --spec-type none` aborts** at `llama-graph.cpp:228`
   (`GGML_ASSERT(ubatch->embd_state)`) — with and without `--fit`.  A loaded-but-unused drafter still gets a
   context whose graph needs the MTP state embedding.  This is a trap for a "plain baseline with the drafter
   loaded" measurement, so it is worth a clean error (or not creating the draft context when no speculator
   is enabled).
2. `--fit` logs `E llama_init_from_model: failed to initialize the context: Gemma4Assistant requires
   ctx_other` before its (now correct) fallback.  Demote to a warning when the extra model declares
   shared KV.
3. `test-llama-archs` (and any gated run) writes `(file*).moe-heatmap` into the CWD.

### 2.3 Promotion cut (the checklist is `HANDOVER-2026-10-21.md` §6 — updated here)

Fold into **block 09** (no new block): #77 A/B/C (`0002`, `0003`, `0004a`, `0004b`), `0005`, `0006`,
**`0007`**, **`0008`**, and **delete the `(file*).moe-heatmap` artifact** in that same amendment.  Fold into
**block 10**: #76 (`0001`).  Then:

1. `scripts/make-patches.sh` + `scripts/make-release.sh` from a canonical fork rebuilt at
   `release.json.base` via `scripts/apply-all.sh` (never hand-edit hashes).
2. `scripts/validate-set.sh` (strict 12/12 from a fresh tarball) + `scripts/gate-per-block-build.sh` for the
   amended blocks.
3. The non-regression contract on the **folded** tree, with `ROCBLAS_USE_HIPBLASLT=0` on both sides of any
   A/B: dense 27B (1 GPU + 3-GPU tensor, q8_0 KV), 4B, gemma4 12B (f16 **and** rotated q8_0, now `-sm tensor`
   too), 35B-A3B `-ncmoe 40` 2-GPU tensor (text + tg128 flat) and the 35B-A3B cache canary
   (**240 tables / 61440 slots / 18662.0 MiB**); `test-llama-archs` 644/644; the prefill-logit gates
   (27B 0.000707 / 98.755 %, gemma4 f16 -0.000000 / 100.000 %, gemma4 q8_0 rotated -0.000000 / 99.990 %).
4. The four-axis adaptive-MTP gate at `-n 3000` on 35B-A3B and dense 27B (`benchmarks/mtp-adaptive-methodology.md`
   rules 0-5, incl. `prompts/code-reasoning-mixed.txt`).
5. Docs: `patches/README.md` (blocks 09/10), `ENVIRONMENT.md` (`GGML_META_SEG_ASYNC`; the iSWA note for
   `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS`), `GREEDY-PURITY.md` if a claim moves, a `WORKLOG.md` entry,
   `TODO.md` loses #76/#77 (+ #68 1b closed once 1a lands or is parked), archive this campaign to
   `archive/work/fix-gemma-4/` with a `RESOLUTION.md`, delete `wip/fix-gemma-4/`, update
   `wip/CAMPAIGNS.md` (no stub).
6. Release ritual (`AGENTS.md` "Pushing policy"): tag, **fork `rdna-boosts` refresh to the personal fork
   only**, worker image build (`scripts/container-watch.sh` on `radiance`) — nothing on the primary box.

**Report findings to the maintainer before step 1 touches `patches/`** (mandatory reporting rule), and
before any push.

---

## 3. Records in this directory

| file | what |
|---|---|
| `HANDOVER-2026-10-21c.md` | **this file** — campaign state after the determinism close and the two fixes |
| `HANDOVER-2026-10-21b.md`, `HANDOVER-2026-10-21.md` | the earlier records (instrument, #68 scope, `--fit` candidates, promotion checklist) |
| `evidence/MEASUREMENTS-2026-10-21d.md` | the hipBLASLt root cause + the graph-story retraction |
| `evidence/MEASUREMENTS-2026-10-21e.md` | the `--fit` fix (0007) and the `-sm tensor` fix (0008) + gates |
| `patch-state/` | `0001..0008`, sha256s, apply/verify commands |
| `tools/fp.sh`, `tools/clihash.sh`, `tools/diag-graph-gates.patch` | the determinism instruments (now env-or-CLI arg capable) and the throwaway graph gates |
