# HANDOVER 2026-10-21 (session b) — MTP determinism: instrument, findings, and where to go next

**Read first, in this order:** `AGENTS.md` (reporting rule, WIP/promotion rules, default-on policy, the
"Critical facts" invariants), then this file, then `evidence/MEASUREMENTS-2026-10-21c.md` (the numbers and
the two **retractions**), then `patch-state/README.md` (the 7-patch set that reproduces the current tree).

**Box:** gfx1201, 3x R9700, ROCm 10.1.0, 16 cores, models under `/llm/models/`.  Tree `~/llama.cpp` @
`rdna-boosts` `972f3b757` (r100) **+ this campaign's patch set** (`patch-state/0001..0006`), built with
`~/bin/build-llama-rocm` (canonical `build-rocm`, ROCm 10.1.0).  The patch set is applied and built in the
working tree right now; `git diff` there *is* the set (verified byte-identical to a clean `972f3b757` +
`0001..0006`, see `patch-state/README.md`).

---

## 0. One-paragraph state

Item 1 of the previous prompt (baseline re-establishment) is **green**.  Item 2 (MTP run-to-run
determinism) is **not solved but is now properly understood and instrumented**: the defect is in the
**draft model's own logits** with **two llama_contexts on two devices and captured graphs**; a single
context (including the whole gemma4 expert-cache path) is **bit-identical across 6 fresh processes**.  One
genuine cross-context bug was found and fixed on the way (the async-input ring's ordering state was per
*device*, so the draft and target contexts consumed each other's queued uploads).  Two earlier conclusions
in `evidence/MEASUREMENTS-2026-10-21b.md` are **wrong** and are retracted in `…-21c.md` §0.  Items 3-6 of
the previous prompt were deferred by the maintainer (see §7).

---

## 1. The instrument — use this, not a text hash

**A greedy text hash is not a determinism instrument on this box**, and `scripts/extract-generated.py`
**must not** be used on a `-v` log (the CLI interleaves debug lines into the generated region).  Worse, the
server's SSE chunk grouping varies run to run, so the *rendered* text can differ while the model output is
identical (measured: one run emitted `documentation`, another `documen` + `/tation` **and dropped a ` a`
token**).  Two level-appropriate observables exist:

### 1.1 The draft-logit fingerprint (primary; on the real product path)

The spec loop prints, per draft step, the draft model's top candidates with 3-decimal probabilities:

```
0.25.621.321 D spec        draft:  - seq_id 0, draft candidate   0, pos   0: 236786 (   0.439) '/'
```

That is a logit-level series of the **draft** context (~100 calls per run), available with `n_probs = 0`
(i.e. no perturbation of the path).  `tools/fp.sh <tag> [K=V ...]` runs one fresh CLI process and prints:

```
tag: calls=99 fp=6f6923ef36 call0_hist=5507 call0_pos0=[(0, 2129, '0.180'), …]
```

`fp` is the sha of the (per-call, sorted) candidate series; `call0_pos0` is the first draft call's candidate
list, which is where every divergence we have seen starts.  Two runs with the same `fp` are the same
trajectory; **N identical `fp` over N fresh processes is the pass criterion**.

Also in `tools/`: `clihash.sh <tag>` (hashes the server's SSE delta stream — a *semantic* text cross-check,
one line per run) — useful as a second opinion, never as the primary gate.

### 1.2 Single-context start-to-start logits (the delivery's own instrument)

For "is one context's decode graph deterministic at logits level" (it is, and this is what clears the
expert cache), use the pre-existing tool.  Note it needs the model's own `NCMOE`/`SPLIT` env, and `PTOK`
must be ≥ 256 (the probe decodes a W=1..8 band):

```bash
cd ~/llama-cpp-rdna-boosts
GPUS=0,1 NCMOE=40 SPLIT=layer PTOK=512 \
  M=/llm/models/Gemma4/26B-A4B/Q8_0/gemma-4-26B-A4B-it-Q8_0.gguf \
  archive/work/kv-cell-offset-nondeterminism/tools/decodestart.sh ~/llama.cpp/build-rocm tag 6
# PASS = exactly one distinct W=1 row-0 hash over the 6 fresh processes
```

### 1.3 Reproduction command for the defect (2 devices, MTP)

`tools/fp.sh` wraps exactly this; run it 6-10 times and compare `fp`:

```bash
M=/llm/models/Gemma4/26B-A4B/Q8_0
HIP_VISIBLE_DEVICES=0,1 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m $M/gemma-4-26B-A4B-it-Q8_0.gguf -md $M/mtp-gemma-4-26B-A4B-it.gguf \
  -ngl 99 -ncmoe 40 -sm layer --fit false -c 16384 \
  --spec-type draft-mtp --spec-draft-n-max 3 -n 300 -t 8 --seed 42 --temp 0 --ignore-eos \
  -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt \
  --no-display-prompt --single-turn -v
```

Parse `spec draft:` lines as in `tools/fp.sh`.  Every run is ~31 s (prefill of 5507 tokens dominates).

---

## 2. What is established (and how it was established)

| config (24–31 s per run) | draft-logit fingerprint | verdict |
|---|---|---|
| **target only**, no MTP (`decodestart.sh`, `NCMOE=40 SPLIT=layer`, 6 procs) | `c97d5e9d8bf6e0d4` ×6 | **bit-stable** |
| MTP, graphs **off** (`GGML_CUDA_DISABLE_GRAPHS=1`) | `8f795fbaf0` ×4 | stable |
| MTP, graphs on, **1 GPU** | `8f795fbaf0` ×3 | stable |
| MTP, graphs on, **3 GPU** | `8f795fbaf0` ×2 | stable |
| MTP, graphs on, **2 GPU** (campaign tree) | `0b95…`×2, `d596…`×4, `ddf4…`, `9523…` | **unstable** |
| … `ASYNC_INPUT=0` / `ASYNC_INPUT_BATCH=0` / `Q8_1_ARENA_FREE_OLD=1` / `SLAB=0` / `META_SEG_ASYNC=0` / `GRAPH_MEM_GEN=0` / `SCHED_EVENTS=0` / `META_SS_VERIFY=1` / `CACHE_VALIDATE=1` | ≥2 variants each, no validator warning | **not the cause** |
| … + **per-origin async-input ring** (`0005`) | `6f6923ef36` ×8, `10e1f21f4c` ×2 | much better, **not clean** |

Conclusions:

1. **The expert cache, host-expert path, slab, Q8_1 arena, split-state cache and eager-remap path are
   cleared** — a single context with all of them is bit-identical over 6 fresh processes.  The defect needs
   **two contexts** (MTP) **on two devices** with **captured graphs**.
2. The divergence is always at the **first draft call**, with the same 5507-token history, and is a
   *continuous* probability change (top-1 0.439 / 0.453 / 0.458), not a garbage read: the **dense drafter's**
   logits differ.  This is the r69 / `TODO #75` family ("a run is not start-to-start deterministic when the
   consumer is a captured HIP graph"), still alive in the MTP draft path.
3. **One real bug found and fixed:** the async-input ring's ordering state (`pending`/`ev`/`q`/`flush_ev`/
   `done_ev`) was **per device** while a stream *is* a llama_context, so the draft and the target consumed
   each other's queued uploads and a context could replay a graph without waiting for its own copy.  Keyed per
   origin (the calling thread's stream) it goes from *unstable in every sample* to 8/10.  Patch `0005`.
4. **`-ngld 0` is not a valid arm in this fork** — it silently runs *without* a speculator (no draft lines at
   all), so any "draft on CPU" result recorded earlier is void.

### 2.1 Retractions (do not inherit these)

`evidence/MEASUREMENTS-2026-10-21b.md` §2.3/§2.4 claim that restoring the scheduler's per-op two-device
synchronize makes MTP deterministic (3/3) and that `GGML_CUDA_DISABLE_GRAPHS=1` was deterministic by text
hash.  **Both are retracted** in `…-21c.md` §0: the first was a 3-run coincidence (12 runs → 50/50), the
second was measured with the invalid text instrument.  The bounded-barrier patch that came out of the first
claim has been **removed from the tree** — it did not fix anything and must not ship default-on.

---

## 3. Tree state and the patch set

`patch-state/0001..0006` applies in order to a clean `972f3b757` and reproduces the working tree
**byte-for-byte** (verified by applying into a detached worktree and `diff -r`; see `patch-state/README.md`
for the exact commands and the sha256s).

| patch | owner block | what |
|---|---|---|
| `0001` | 10 | TODO #76: iSWA KV builder expands `q, v, k` (rope+FWHT write fusion fires on iSWA) |
| `0002` | 09 | TODO #77/A: host-expert account counts routed banks only (`ffn_down_exps.scale` excluded) |
| `0003` | 09 | TODO #77/A+C: completeness slack + multi-slice cache tables (fused `gate_up` cacheable under `-sm tensor`) |
| `0004a` | 09 | TODO #77/B: row-aligned async segmented upload (`GGML_META_SEG_ASYNC=0` bisect arm) |
| `0004b` | 09 | scheduler call site for the extended `moe_cache_update` iface |
| **`0005`** | **09** | **async-input ring ordering state per origin (stream), not per device** — the cross-context bug above |
| **`0006`** | — (`common/`) | **spec verify batch vs `-b`**: raise `n_batch` to `n_max+1` with a WRN instead of aborting in `get_sub_batch` |

`0006` in detail (newly added this session, independent of the determinism work): `--spec-type draft-mtp`
needs a verify batch of `n_max + 1` tokens in one `common_batch::get()`, which is capped by
`llama_n_batch()`; `-b 1` therefore aborted in a server worker (`get_sub_batch: failed to add token 0 at
index 1 (error -1, n = 2)`) after the model had loaded.  `common_init_from_params` now raises `n_batch` to
`n_max + 1` with a WRN (visible at `-lv 2`; llama-cli's default threshold hides WRN).  Verified: the exact
aborting command runs; default runs are untouched (2048 ≫ 4).

---

## 4. Next bisect for the determinism item (in this order)

The remaining ~20 % divergence is still the **first draft decode on 2 devices with graphs**.  The two
"attractor" families seen (`6f6923ef36` vs `10e1f21f4c`) differ by ~4 % in probability at call 0 and by 99
vs 103 total draft calls.  Two possibilities to separate, in order of cheapness:

1. **Capture-path difference vs race.**  If `6f69` and `10e1` are each *stable values* (not a lottery), the
   draft's **captured graph differs** between them — a deterministic capture bug (much easier to fix than a
   race).  Instrument: `GGML_SCHED_DEBUG=2` + `-lv 4` for the *draft* context's graph dump, plus
   `GGML_CUDA_FUSE_TRACE=1`, compared across a `6f69` run and a `10e1` run.  (Note: the scheduler dump did
   not appear under `-lv 4` on the target; the draft context may need its own run with only the draft model
   loaded, or `sched->debug` prints only on graph *allocation*.)
2. **The remaining per-device shared allocations both contexts touch**: the op-offload H2D staging ring
   (`ggml_cuda_context::h2d_stage_buffer` / `h2d_pin_buffer` / `h2d_scratch`) and the FA prefill staging
   arena (`fattn_stage_*`).  Both are per *backend context*, but the **slab**-backed ring hole and the
   `GGML_CUDA_*` rings are worth an explicit read.  A cheap arm: force the cache's staging path off
   (`GGML_MOE_CACHE_STAGE=0`, `GGML_SCHED_STAGE_MODE` unset) and check the fingerprint.
3. **A draft-only graph switch** (`GGML_CUDA_DISABLE_GRAPHS` applied to the draft context only) is both a
   diagnostic and a plausible narrow fix: the draft graph is a 1-token decode, so its capture win is small,
   whereas the target's graph is the one that matters for throughput.

Do **not** re-try these (measured, no effect): async-input on/off/batch, Q8_1 arena retire mode, slab
off, graph-mem-gen, scheduler events, meta segmented-upload async, split-state verify, cache validate,
`-ncmoe 0`, 1/3 devices, `-fa off`, draft on CPU (`-ngld 0` is a no-op speculator).

---

## 5. How to run the gates once the determinism item is closed

Unchanged from the campaign contract (`README.md` §2.4) with the instrument rule from §1 here:
text identity **across models/builds** is still fine when comparing *two builds* on the *same* run
(`llama-cli` without `-v`, `extract-generated.py`), but **run-to-run determinism must be judged on `fp`**
(§1.1) or `decodestart.sh` (§1.2), never on a text hash.

Re-run after any change here: the non-regression cells (4B, 12B, 27B 1-GPU + 3-GPU tensor, 35B-A3B
`-ncmoe 40` — text vs `build-r99`, throughput flat), the 35B-A3B cache canary (**240 tables / 61440 slots /
18662.0 MiB**), both prefill-logit gates, and `test-llama-archs` 375/375.

---

## 6. Item 1 result (closed this session)

Baseline re-established and green: patch sha256s match; the tree is byte-identical to a clean
`972f3b757 + patch-state`; the binary is the gated one (the #76 record hash reproduces exactly:
`c740be0e131a`, gemma4 12B n64 capital, q8_0 KV); every §2.4 cell is text-identical to `build-r99` with flat
throughput; the 35B-A3B cache canary is unchanged; the prefill-logit gates reproduce the recorded numbers
exactly (27B **0.000707 / 98.755 %**, gemma4 f16 **-0.000000 / 100.000 %**, gemma4 q8_0 rotated
**-0.000000 / 99.990 %**).  Two discrepancies recorded, neither a regression: the 2026-10-21 evidence file's
gemma4 text hashes are not reproducible (the exact commands were never recorded), and gen measured 74.9 /
72.8 t/s vs the recorded 79.0 / 74.7 (same text, same cache config).

---

## 7. The rest of the campaign — preserved, deferred by the maintainer

Unchanged from `HANDOVER-2026-10-21.md` §3 and the previous prompt, in maintainer order, **after** the
determinism item:

1. **The `--fit` arena shortfall under MTP** (previous handover §3 item 2; maintainer decision: investigate
   with a view to fixing).  At the default context the draft + target contexts make `--fit` reserve
   19.16 GiB/device, the slab is created at 9.31/7.50 GiB, the arena at 13.8 GiB instead of 23.1, and MTP
   falls to 50.2 t/s against plain 79.3.  Candidate directions are listed there ((a) extend before sizing,
   (b) a one-shot grow after the post-prefill drop, (c) reserve the cacheable expert bytes, (d) check the
   draft context's KV accounting).  Must keep the slab invariants in `AGENTS.md`.
2. **TODO #68 1b** — gemma4 dense + `-sm tensor` aborts in `handle_per_row`
   (`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`) with the default f16 cache.  A per-head norm
   over a head-split axis is a local reduction; the fix is in the Meta split-state reshape predicates.
3. **TODO #68 1a** — rotated cache (`-ctk q8_0 -ctv q8_0`) → `unsupported mul_mat split states …
   src0=MIRRORED#attn_inp_k_rot axis=10 src1=axis 0`.  Four costed directions in the previous handover.
4. **Promotion cut** — previous handover §6: fold #76 → block 10, #77 A/B/C (+ `0005`, `0006`) → block 09,
   drop the stray `(file*).moe-heatmap` artifact in block 09 at that cut, regenerate with
   `scripts/make-patches.sh` + `scripts/make-release.sh`, `validate-set.sh` from a fresh tarball,
   `gate-per-block-build.sh` for the amended blocks, the full non-regression contract, the four-axis
   adaptive-MTP gate at `-n 3000` on 35B-A3B and dense 27B, docs (`patches/README.md`, `ENVIRONMENT.md` for
   `GGML_META_SEG_ASYNC`), a `WORKLOG.md` entry, `TODO.md` loses #76/#77, archive this campaign to
   `archive/work/fix-gemma-4/` with a `RESOLUTION.md`, delete `wip/fix-gemma-4/`, update `wip/CAMPAIGNS.md`.
   **Do not fold `0005`/`0006` until the determinism item is closed** — `0005` is a real fix but its effect
   is measured (8/10) and it belongs with whatever closes the item.

---

## 8. Records in this directory

| file | what |
|---|---|
| `HANDOVER-2026-10-21b.md` | **this file** — the cold-start record for the determinism work |
| `HANDOVER-2026-10-21.md` | the previous session's handover (#68 scope, the original determinism finding, the `--fit` item) |
| `evidence/MEASUREMENTS-2026-10-21c.md` | the numbers behind §2, the instrument, and the **retractions** |
| `evidence/MEASUREMENTS-2026-10-21b.md` | baseline re-establishment (green) + the **retracted** determinism conclusions (§0 of `-21c` supersedes) |
| `evidence/MEASUREMENTS-2026-10-21.md` | the first session's numbers (#76 launches, #77 A/B/C, MTP) |
| `tools/fp.sh`, `tools/clihash.sh` | the determinism instruments (§1) |
| `patch-state/` | the 7-patch set + sha256s + the apply/verify commands |
| `NEXT-SESSION-PROMPT.md` | the prompt for the next session |
| `ISWA-WRITE-FUSION.md` | the full #76 record |
| `README.md` | the umbrella campaign (items, plan, non-regression contract) |
