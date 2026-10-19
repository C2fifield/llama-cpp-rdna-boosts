# issue #184 — the r80 -> r82 `ggml-backend-meta.cpp` change is not a 4-stream regression here

**Status: investigated 2026-10-18 (r86-candidate tree).  No delivery change; no release.**  `@briansp2020`
reports that on his 2x R9700 box the r82 Meta change (the `[TAG_META_UNALLOC_ALIAS]` /
`[TAG_META_ALLOC_FAIL]` hunks of issue #170) costs **-3.48 % [-5.60, -1.37] at 4 concurrent greedy
decode streams**, with a host profile pointing at `ggml_backend_meta_get_split_state` on the graph
allocation path.  On this box the same pair of builds is **equal within noise** (<= 0.4 %, sign not
stable), and the compiled-code difference is **224 bytes** of `.text` in an 83 KB object with no
inlining change.  Reported back with the diagnostics that would settle it on his side
(`COMMENT-2026-10-18.md`).

## 1. What the two arms actually differ by

Both arms are the **same r86-candidate tree**; only `libggml-base.so.0.26.0` is swapped between runs,
which is exactly the reporter's setup ("the other libraries are byte-identical; only `libggml-base`
differs").

| arm | meta file | note |
|---|---|---|
| **A** | delivery (r86-candidate, block 00 with #170) | the reported-slow arm |
| **M** | A with the two `[TAG_META_UNALLOC_ALIAS]` hunks reverted in both functions (the r80 shape) | the reporter's "partial revert" |
| **C** | A with the #170 alias resolution moved out of the hot dispatcher into a `noinline, cold` helper (hot path = the exact pre-#170 two-test dispatcher) | candidate fix, see §5 |
| **P** | A with the r84 graph-layout fingerprint check disabled (`GGML_GALLOCR_NO_FP=1`, a session-local toggle) | tests the "graph identity for parallel sessions" angle |
| **D** | the pristine delivery build (no local toggles) | validates the A arm |
| **S / X** | A with a `volatile` add-loop (100 / 1000 iterations) at the top of the hot dispatcher | sensitivity calibration, §7 |

**Functional equivalence.**  A and M differ only for a tensor in the split-state walk that has **no
buffer of its own** — a nested alias the gallocr never initialised (the issue #170 `ggml_view_4d` ->
`ggml_permute` pair).  On r80 that case asserted in `ggml_backend_meta_buffer_n_bufs`; every graph that
ran on r80 therefore never reached it, so for the reporter's config the two arms are the *same
program* apart from code layout.

**Codegen** (exact `ggml-base` flags, `-O3 -DNDEBUG`, clang++ from the ROCm tree, both vintages):

| | ROCm 10.1.0 clang | ROCm 10.0.0 clang (the reporter's) |
|---|---|---|
| `.text` of `ggml-backend-meta.cpp.o` | 82 905 B (A) / 82 681 B (M) | 82 857 B (A) / 82 633 B (M) |
| dispatcher `get_split_state(const ggml_tensor *, bool)` | 454 B / 385 B, **out of line in both** | 454 B / 385 B, out of line in both |
| inner `get_split_state(stc &, const ggml_tensor *, bool)` | 4 265 B / 4 137 B | 4 249 B / 4 121 B |

So the delta is **+224 B of code (+0.27 %)** and the call structure is identical.  The only hot-path
difference is one `buffer == nullptr` test plus the pre-existing host/`is_meta` classification being
evaluated from a hoisted local — i.e. the reported effect cannot come from instruction volume.

## 2. How it was measured (and why the naive harness does not work)

Metric: 4 concurrent greedy `/completion` requests on `--parallel 4`, aggregate generated tokens /
wall.  A fresh server per arm, arms alternating, `.so` swapped between runs.

* Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (2 shards) + `mtp-Qwen3.8-Flash-Next-shared-Q8_0`.
* `-ngl 99 -sm tensor -ncmoe 47 -fa auto -ctk q8_0 -ctv q8_0 -c 131072 -b 2048 -ub 2048 -t 8 -np 4
  --fit off --load-mode none --lazy-mode off --spec-type draft-mtp-adaptive --spec-draft-n-max 3`,
  `HIP_VISIBLE_DEVICES=0,1`, `prompts/prose-rdna-boosts.txt` (first 2000 chars), `n_predict 192`.
* `--load-mode none --lazy-mode off` is deliberate: the PLE weights must be RAM-resident so the
  measurement is not a page-cache race.
* **Every slot's prompt is cached before measuring** (one serial `n_predict: 1, cache_prompt: true` per
  slot, then the four measured requests with `cache_prompt: true`).  This is the important part: when
  the four prefills are left in the window, the four decode streams do not overlap for the whole
  window (the server starts decoding a slot as soon as its prefill finishes), and the per-step cost
  then depends on how many streams happen to be decoding.  Measured that way the *same* build gives
  **63.8 / 63.8 / 63.8 / 139.2 t/s** — a 2.2x bimodality that swamps any 3.5 % effect.  Pre-warmed,
  the same build reproduces to **0.1-0.3 %** within an arm (all four slots report identical ms/token,
  e.g. `[36.85, 36.85, 36.85, 36.85]`).
* Harness: `ab2.py` (`python3 ab2.py <pairs> '{}' <tag> A,M`).

## 3. Result — A vs M (the reported comparison)

| session | pairs | A (r82 meta) | M (revert) | paired A vs M |
|---|---|---|---|---|
| `warm` | 2 | 99.87 t/s | 99.63 t/s | **+0.24 %** |
| `ab4` | 4 | 99.39 t/s | 99.77 t/s | **-0.39 %** |
| `am2` | 4 | 100.22 t/s | 100.24 t/s | **-0.02 %** |

Within-arm spread 0.1-0.3 %; the *same* `.so` drifts 0.8 % across sessions (A: 99.39 -> 99.71 ->
100.22 -> 100.06), so only same-session paired differences are meaningful.  Greedy text is identical in
every arm and every session (the four streams share one content hash, identical across all runs of a
session).  **Verdict: equal
within ±0.4 %, with no stable sign — at most one ninth of the reported 3.5 %.**

Control arms (same session discipline):

| comparison | pairs | arm 1 | arm 2 | paired |
|---|---|---|---|---|
| A vs C (cold-path restructure) | 4 | 100.07 | 100.13 | -0.06 % |
| A vs P (r84 layout fingerprint off) | 3 | 99.71 | 99.78 | -0.07 % |
| A vs D (pristine delivery build) | 2 | 100.06 | 100.11 | -0.05 % |

`A vs D` confirms the session-local `GGML_GALLOCR_NO_FP` toggle in arms A/M/C was inert as designed.

## 4. Two things the reporter's profile predicts, checked directly

* **"The verify graphs are re-allocated each step."**  The delivery already prints a per-request
  `graphs reused` / `graph reallocs = N / M (same size) / (changed size)` pair
  (`llama_context::sched_graph_realloc`, block 09/11), which is the instrument for this.  Pre-warmed,
  a 4-stream request reports **1 same-size + 1 changed-size reallocation** (not one per step); with
  the raw 512-token-ubatch harness (varying prefill batches) it is 10-12 same-size per request.  The
  `probe4.py` probe + `-v` also shows the `ggml_gallocr_needs_realloc` reason lines
  (`graph topology changed (fingerprint mismatch)` etc., `NDEBUG`-guarded, so debug logs only).
* **"Does the r84 graph-identity fingerprint make that worse?"**  No, here: disabling the fingerprint
  check entirely is worth -0.07 % (noise), i.e. the fingerprint is not adding per-step re-reserves in
  this configuration.

## 5. The candidate fix that was built and measured (not shipped)

`C` restores the pre-#170 hot dispatcher exactly (`if (ggml_backend_buffer_is_meta(tensor->buffer))`
first) and moves the whole #170 resolution — the view-chain lookup, the host-buffer test and the
host-view assert — into a `static __attribute__((noinline, cold))` helper.  It is semantically
identical to A (same three outcomes in the same order, same assert) and it is the shape to adopt **if**
the reporter's effect ever reproduces: measured **-0.06 % vs A** (i.e. nothing) on this box, so it was
**not** shipped — a hot path the #170 fix depends on should not be restructured for a measured zero,
and the two hunks are also staged upstream as
`upstream/UPSTREAM-PR-meta-split-unalloc-alias-and-alloc-fail.patch` (an amendment means re-cutting
that).

## 6. Separate finding raised by this session (not #184)

`gemma4` **dense** (12B Q8_0) under `-sm tensor` on **2 devices** with `-ctk q8_0` aborts at the warmup
decode:

```
ggml/src/ggml-backend-meta.cpp:900: unsupported mul_mat split states:
  node=node_286 src0=Meta(ROCm0,ROCm1)#attn_inp_k_rot#0 axis=10 src1=Kcur_pos-5 (reshaped) axis=0
```

`attn_inp_k_rot` is the **K rotation matrix** (`llama_kv_cache::build_input_k_rot`, `n_rot_k > 0` only
for a rotated-cache type), so the op is the rotation `ggml_mul_mat(rot, Kcur)` and its `src1` is split
on axis 0, which `handle_mul_mat` does not support.  `-ctk f16` / `-ctk bf16` load and answer
(`n_rot_k == 0`); it is independent of `-np`, `--kv-unified`, `-fa on` and
`GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` / `GGML_CUDA_DISABLE_FUSION`.  The `gemma4-assistant` MTP head is
*correctly* rejected for `-sm tensor` (`llm_arch_supports_sm_tensor`), so that half is intended; the
dense + rotated-KV abort is not, and is tracked as TODO #68.

## 7. Sensitivity calibration — is the null result informative?

A null is only worth something if the instrument could have seen the reported effect, so the hot
dispatcher was pessimised deliberately (a `volatile` add-loop at the top of every call, no other change)
and A/B'd the same way:

| arm (same source otherwise) | cost of the added loop | paired A vs arm |
|---|---|---|
| +100 volatile iterations per `get_split_state` call | ~120 ns/call | +0.23 % |
| +1000 volatile iterations per call | ~1.2 us/call | **+1.17 %** |

Linear, ~0.0012 % per iteration, so the harness resolves host-path cost to ~0.2 % and would show a 3.5 %
effect as ~3000 iterations per call (~350 ns on every one of the ~10k lookups a decode step performs).
The r82 delta adds **one null test** per call (~1-2 ns) — two orders of magnitude below the resolution
limit, and ~50x below what the 3.5 % claim would require.  Our per-step time is also 1.5x the reporter's
(102 ms vs ~70 ms at mean draft length 2.8), which makes host cost weigh *more* here, not less.  So the
null result is informative: on this compiler the r82 change cannot account for 3.5 % by instruction
count; if it is real on his box it is a layout/inlining effect (the `nm` ask in the reply settles that).

## 8. Repro

```bash
# both arms: build the tree twice with the .so swapped (only one TU changes)
cd ~/llama.cpp && cp build-rocm/bin/libggml-base.so.0.26.0 /tmp/base-A.so
#   arm M: revert the two [TAG_META_UNALLOC_ALIAS] hunks in ggml/src/ggml-backend-meta.cpp
#          (both functions), rebuild `cmake --build build-rocm --target llama-server`, save /tmp/base-M.so
python3 archive/work/issue-184-meta-r80r82/ab2.py 4 '{}' ab A,M
python3 archive/work/issue-184-meta-r80r82/probe4.py flashnext '{}' probe   # realloc counters, 4 slots
```

Session-local artefacts (arm `.so` files, server logs) were under `/tmp/i184/`; only the harnesses are
kept here.
