# `iswa-kv-write-fusion` — the q8_0 K/V cache-write fusion never fires on an iSWA model (Gemma4)

**Status: FIX VALIDATED on gfx1151 (2026-10-21); awaiting the main-box multi-device gate + the
maintainer's go-ahead to promote.**  TODO [#76](../../TODO.md).  Not part of the delivery; nothing here
may be applied to the fork without the maintainer's go-ahead (see the WIP and promotion rules in
`AGENTS.md`).

**Result in one line:** reordering the iSWA KV builder's expands to `q, v, k` (the non-iSWA builder's
order) makes the r81 K/V cache-write fusion fire on Gemma4: `k_set_rows_q8_0_group` 1152 -> 192 and
`rms_norm_mul_rope_f32<…,kv=true>` 0 -> 480 on a 12B run, with byte-identical greedy text, identical
`W=1..8` probe hashes, KLD 0.000000 / same-top-p 99.98 % on the rotated-cache path, and neutral
t/s on this APU (the win is 80 launches/decode step; it needs a gfx1201 number).  Fix:
[`fixA-iswa-expand-order.patch`](fixA-iswa-expand-order.patch), 5 lines, no new env variable.

**Box: gfx1151 (Strix Halo APU), 1 device, 128 GiB unified, ROCm 10.1.0.**
Tree: `~/llama.cpp` @ `rdna-boosts` `6adf0b802` (`v12-097cf33d9-r99`), `build-rocm`.

## 0. The symptom

With a **rotated** KV cache (`-ctk q8_0`, i.e. `n_rot_k > 0`) the r81 rope + FWHT fusion folds the two
q8_0 K/V cache `SET_ROWS` launches into the fused rope kernel (r81, block 10).  It fires on
qwen35/qwen35moe because those build attention through the **non-iSWA** `build_attn`, which expands K
*last*.  On an **iSWA** model (`Gemma4`) the builder interposes the V block between the K rotation and
the K cache write, so the matcher declines and the two launches remain.

No correctness problem: the unfused path is exactly r77's.  The cost is the two launches per layer per
step that r81 already removed everywhere else.

## 1. Code facts (established 2026-10-21, r99 tree)

* Matcher: `ggml_cuda_rms_norm_mul_rope_fwht_set_rows_count()` —
  `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu:945`.  It requires, from the anchor `i`:
  `RMS_NORM, MUL, ROPE` (i..i+2) → `RESHAPE` (i+3) → `MUL_MAT` Hadamard (i+4) → `RESHAPE` (i+5) →
  `VIEW` (i+6) → `SET_ROWS` q8_0 (i+7), optionally a V `VIEW`/`SET_ROWS` at i+8/i+9 (10 nodes), else
  8 (K only).  Kill switch `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS`.
* **No arch gate**: `ggml_cuda_should_fuse_rms_norm_mul_rope_fwht()` (`try-fuse.cu:881`) is
  shape/type-only, and `ggml_cuda_op_mul_mat_use_fwht()` (`fwht.cu:228`) is a `GGML_HINT_SRC0_IS_HADAMARD`
  + type/contiguity test.  So the pattern is reachable and measurable on gfx1151.
* Builders competing for the same node order:
  | builder | file | expand order | K write right after K chain? |
  |---|---|---|---|
  | non-iSWA KV | `src/llama-graph.cpp:3205` | `q, v, k` (comment: "expand k later to enable rope fusion which directly writes into k-v cache") | **yes** |
  | non-iSWA DSA | `src/llama-graph.cpp:3372` | `q, v, k` | yes |
  | **iSWA KV** | `src/llama-graph.cpp:3448` | **`q, k, v`** | **no — v chain sits between the K rotation and the K write** |
  | iSWA K-only (MLA) | `src/llama-graph.cpp:3543` | `q, k` | yes (no V write to fold) |

  So only the iSWA **KV** builder has the wrong order, and the non-iSWA builder carries the explicit
  comment saying the order exists *for this fusion*.  The iSWA builder looks like it simply never got
  that treatment (it takes a different rotation path: `self_k_rot_swa`/`self_v_rot_swa` selected per
  layer before the expands).
* Node dump of the r78 Gemma4 decode graph (recorded in
  `archive/work/kv-cache-q8-write/RESOLUTION.md`) shows the shape:
  `15 RMS_NORM 16 MUL 17 ROPE 18 RESHAPE 19 MUL_MAT(K rot) 20 RESHAPE | 21 MUL_MAT[Vcur] … 26 RESHAPE |
  27 VIEW 28 SET_ROWS[cache_k] 29 VIEW 30 SET_ROWS[cache_v]` — K write 7 nodes after the K `RESHAPE`,
  not 2.

## 2. Fix candidates

**A (preferred) — reorder the iSWA builder's expands to `q, v, k`**, mirroring the non-iSWA builder
that carries the fusion comment.  Two lines.  Then the K chain is emitted last, `cpy_k` immediately
follows it, and `cpy_v` produces the i+8/i+9 pair → the 10-node arm fires (both writes folded).
Risk: expand order also drives graph splits (the comment above the expands says the nodes are added
together to reduce splits) and changes the graph for **every** iSWA model at a rotated cache type, not
just Gemma4.  Must be measured for splits + coherence, and the prefill-logit gate applies (a
prefill-band graph order change).

**B — a bounded-window matcher** that tolerates the interposed V block (scan up to N nodes, require
single-use/non-output throughout).  CUDA-side only, no graph change, but it has to handle the
V-write-before-K-write order the current node dump shows, which the 10-node arm does not express
(it expects K write then V write).  Higher matcher complexity, lower blast radius.

Start with A; keep B as the fallback if A regresses graph splits or is rejected for scope.

**Result: A was taken and is clean on this box (§5.2-§5.7); B was not needed.**  A also removes the
interposed-V-block case instead of special-casing it in the matcher, so the iSWA builder now looks like
the builder that carries the fusion comment.

## 3. Repro / gates on this box

Model: `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` (12.7 GB, fits the APU outright).

* Rotated cache: `-ctk q8_0 -ctv q8_0` (mixed types are hard-rejected, so both).
* **Acceptance evidence is the launch count, not only t/s**: `rocprofv3 --kernel-trace` (or an
  instrumented counter) showing `k_set_rows_q8_0_group` per step drops, plus the fused
  `rms_norm_mul_rope_f32<…, kv=true>` instantiation appearing.
* A/B reference: `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` must be **identical** to default today (both
  inert) and must show a real delta after the fix.
* Correctness: same-seed greedy text vs the disabled arm (the fusion is claimed bit-identical),
  `test-logits-width-probe` at the verify widths, and `scripts/gate-prefill-logits.sh` (KLD <= 0.005,
  same-top-p >= 98 %) because the graph order changes for prefill too.
* Perf: decode t/s (`llama-bench -p 0 -n 128`, and a longer `llama-cli` run), one job at a time; thread
  count sized to leave the GPU IRQ cores free (`-t 6`-`8` on this 16-core APU).

## 4. Plan (status 2026-10-21: 1-4 done, 5 is the current state)

1. Reproduce: rocprofv3 launch counts on r99, default vs `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1`
   (expect identical — proving the fusion is inert) and confirm the 10-node pattern is absent.
2. Implement A (iSWA expand reorder) and re-run step 1: launch count must drop, disabled-arm delta must
   appear.
3. Gates: greedy text identity vs the disabled arm, width probe, prefill-logit KLD, `llama-batched-bench`,
   graph-split count (before/after) so the reorder cannot hide a split regression.
4. If A is clean, record the diff here as a promotion candidate (block 10 owns the fusion; the builder
   change is `src/llama-graph.cpp`, which block 10/11 also touch) and hand the delivery cut to the
   maintainer.  B is only if A fails.
5. Leave everything under this directory until promoted; then archive per the WIP rule.

## 5. Evidence

### 5.1 r99 baseline on gfx1151 — the fusion is inert (2026-10-21)

Model `Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf`, `-ngl 99 -ctk q8_0 -ctv q8_0 -t 8`, prompt
`The capital of France is`, `-n 8`, greedy, `rocprofv3 --kernel-trace`.  Same 14550 kernel rows in both
arms:

| kernel | default | `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` |
|---|---:|---:|
| `rms_norm_mul_rope_f32<256,false,float,256,**true**>` (the K/V-write fold) | **0** | 0 |
| `rms_norm_mul_rope_f32<256,false,float,256,false>` (rotation only) | 960 | 960 |
| `k_set_rows_q8_0_group` (the writes that should be folded) | **1152** | 1152 |

So on this box: the rotation fusion fires (960), the K/V-write fold never does (0), and the kill
switch changes nothing at all — the item reproduces verbatim on gfx1151 (no RDNA4 needed).  Decode on
the short run 15.1-15.4 t/s (not a benchmark).  Trace CSVs were kept at `/tmp/kvtrace/` for the session
(not in the repo — they are 6 MB each; rerun the command in §3 to regenerate).

Command used:

```bash
/opt/rocm-10.1.0-gfx1151/bin/rocprofv3 --kernel-trace -f csv -o /tmp/kvtrace/gemma_base -- \
  ~/llama.cpp/build-rocm/bin/llama-cli -m /llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf \
  -ngl 99 -p "The capital of France is" -n 8 --seed 42 --temp 0 --no-display-prompt --single-turn \
  -ctk q8_0 -ctv q8_0 -t 8
```

### 5.2 Fix A — the diff (validated, uncommitted in `~/llama.cpp`)

`fixA-iswa-expand-order.patch` in this directory: emit the V chain before the K chain in the iSWA KV
builder, exactly like the non-iSWA builder (which carries the same comment).

```diff
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -3477,16 +3477,17 @@
     // these nodes are added to the graph together so that they are not reordered
     // by doing so, the number of splits in the graph is reduced
+    // expand k later to enable rope fusion which directly writes into k-v cache
     ggml_build_forward_expand(gf, q_cur);
 
-    if (k_cur) {
-        ggml_build_forward_expand(gf, k_cur);
-    }
-
     if (v_cur) {
         ggml_build_forward_expand(gf, v_cur);
     }
 
+    if (k_cur) {
+        ggml_build_forward_expand(gf, k_cur);
+    }
+
```

Build: `~/bin/build-llama-rocm` (incremental, 1 TU + link).  No kernel change, no new env variable.

### 5.3 Launch counts (rocprofv3, same command as 5.1)

**Gemma4 12B** (48 layers = 40 SWA head_dim 256 + 8 full head_dim 512), `-n 8`:

| kernel | pre-fix | fix A | fix A + `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` |
|---|---:|---:|---:|
| `rms_norm_mul_rope_f32<256,false,float,256,**true**>` (K + V writes folded) | 0 | **480** | 0 |
| `rms_norm_mul_rope_f32<256,false,float,256,false>` (rotation only, Q) | 960 | 480 | 960 |
| `k_set_rows_q8_0_group` | **1152** | **192** | 1152 |
| total kernel launches | 14550 | **13590** | 14550 |

Per decode step (`-n 1` -> `-n 2` delta, fix A): `kv=true` **+40**, rotation-only **+40**,
`k_set_rows_q8_0_group` **+16**, `rms_norm_mul_rope_f32<256,true,float,0,false>` +16,
`fwht_cuda<512,float,false>` +16.  So every one of the 40 rotated layers folds **both** writes; the
residual 16 writes are the 8 full-attention (512-wide) layers — see the adjacent finding in 5.6.

**Gemma4 26B-A4B** (iSWA MoE), `-n 2`: `k_set_rows_q8_0_group` **360 -> 60**, `kv=true` 0 -> 150,
total launches 8100 -> 7800.  Same shape, so the fix is not 12B-specific.

### 5.4 Correctness — bit-identical

Fixed build = `fix A`; `prefix` = the r99 tree rebuilt *without* the reorder (stashed, built, restored);
`fix A-disabled` = the same fixed binary with the fusion kill switch.  All greedy, `--seed 42 --temp 0
--single-turn --no-display-prompt`, hash via `scripts/extract-generated.py`:

| model / run | prefix | fix A | fix A-disabled |
|---|---|---|---|
| 12B, `-n 64` "The capital of France is" | `ec73cb8199b1` | `ec73cb8199b1` | `ec73cb8199b1` |
| 12B, `-n 200 --ignore-eos`, two-paragraph+poem prompt | `1541e6a1c731` | `1541e6a1c731` | `1541e6a1c731` |
| 26B-A4B, `-n 120 --ignore-eos`, tides prompt | `7c56873f7382` | `7c56873f7382` | `7c56873f7382` |

`test-logits-width-probe` (`prompts/prose-rdna-boosts.txt`, P=256, ub=512): `W=1..8` hashes identical
prefix vs fix A in **both** cache types, `width_purity=PASS (worst maxdiff 0)`:

* `KV=q8_0`: `0ecedc3bae4f3f00 c91207b3bfd9c267 f5f6c718a0ecca99 c3f6202d2df14880 a11d8c1aacc2581e
  72a851c0f2c6eeb9 39076ec29c10b1f5 646bf5aa0130a6d4`
* `KV=f16`: `e89dca2e24dd7e2a a9a32b9334f6a1e3 5dc2a144f1f099d0 1770cb989cbcee3c 095b9484767829aa
  3584662c24a41519 551743d7c2453aa1 8783df4ee37232a5`

Prefill-logit gate (`scripts/gate-prefill-logits.sh`, Gemma4 12B, 40 chunks x 512, gfx1151):

| arm | mean KLD | same top p | verdict |
|---|---:|---:|---|
| standard gate (f16 KV; fusion not reachable) | **0.000000** | **100.000 %** | PASS |
| rotated-path gate (manual, `-ctk/-ctv q8_0`, base = kill-switch arm) | **0.000000** | **99.980 %** | PASS |

The rotated-path base and candidate PPL agree exactly (`770.0013` both); the gate script only runs f16
KV, so the q8_0 comparison was run by hand (commands below) — that is the arm this change actually
touches, and it should be added to whatever promotes this.

```bash
B=~/llama.cpp/build-rocm/bin/llama-perplexity; M=/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf
C=/llm/models/wikitext-2-raw/wiki.test.raw
GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1 $B -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
  -ctk q8_0 -ctv q8_0 --save-all-logits /tmp/iswa-gates/gemma4-q8.kld
$B -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
  -ctk q8_0 -ctv q8_0 --kl-divergence-base /tmp/iswa-gates/gemma4-q8.kld --kl-divergence
```

`test-backend-ops -o SET_ROWS -b ROCm0`: **159/159**, ROPE clean.  Scheduler metrics are unchanged by
the reorder (only timings differ): `nodes = 2549 / 2549, splits = 1 / 1`, ROCm0 compute buffer
117.50 MiB, ROCm_Host 16.82 MiB, same `graph reallocation` sequence.

**Non-iSWA inertness (direct proof):** Qwen3.5-4B Q8_0 (`-ctk/-ctv q8_0`, `-n 2`, rocprofv3) gives an
*identical* per-kernel counter in prefix vs fix A — 4055 launches both, `kv=true` 40 both (the shipped
M-RoPE fold, unchanged), `k_set_rows_q8_0_group` 0 both.  The changed builder is only called by iSWA
archs, so the delivery's qwen path is untouched.

### 5.5 Throughput — neutral on gfx1151 (the win is structural)

`llama-bench -p 512 -n 256 -r 3`, `-ctk/-ctv q8_0`, interleaved runs (3 rounds each, this APU's
run-to-run spread is ~0.3-0.6 %):

| arm | pp512 (mean of 3 rounds) | tg256 |
|---|---:|---:|
| prefix | 1236.1 t/s | 15.60 t/s |
| fix A | 1232.7 t/s (-0.28 %, inside the noise band) | 15.60 t/s |
| fix A-disabled | 1233.6 t/s | 15.59 t/s |

`llama-batched-bench -npl 1,4,8` (B=1/4/8): decode 15.41/57.53/107.35 -> 15.38/57.51/106.96, prefill
within the same ~1 % box noise.  So on this box the saving is 80 launches per decode step (960 per
8-step run, 6.6 % of all launches) with **no measurable t/s change** — a 66 ms token hides ~0.25 ms of
launch time.  The comparable figure from r81 was **+0.4 %** on 35B-A3B at 2 GPUs; the gfx1201 number
is the one that prices this, and it is a main-box measurement.

### 5.6 Adjacent finding — the full-attention (512-wide) layers are inert for a *different* reason

Not part of #76 and **not** changed here.  Gemma4 alternates sliding-window layers (head_dim 256) with
full-attention layers (head_dim 512): 40+8 of 48 at 12B, 25+5 of 30 at 26B-A4B (both verified from the
GGUF: `key_length=512`, `key_length_swa=256`, `sliding_window_pattern` = 5 T + 1 F).  The full layers
never reach the rope + FWHT fusion at all: `ggml_cuda_should_fuse_rms_norm_mul_rope_fwht` accepts only
`n in {64, 128, 256}`, so each runs `rms_norm_mul_rope_f32<256,true,float,0,false>` + a separate
`fwht_cuda<512,float,false>` for **Q and K**, plus 2 `k_set_rows_q8_0_group` — **6 launches per layer per
step, 4 of them foldable** (see the 5.3 per-step table: 16 rope + 16 fwht + 16 set_rows for 8 layers;
the 2 rope per layer are Q and K, not K and V).

**What it would take** (checked against the code, 2026-10-21):

1. `ggml-cuda-try-fuse.cu:881` — add 512 to the width gate (one predicate feeds both the rotation and
the set_rows matcher).
2. `rope.cu:974-985` — a `case 512:` in both `switch (fwht_n)` blocks = **4 new instantiations**
   (`<256,{true,false},float,512>` and `<256,{true,false},float,512,true>`; Gemma4's full layers are the
   `has_ff=true` pair).
3. `rope.cu`, the `kv` V-row write: `if (kv_args.v_src != nullptr && tid < ncols)` assumes
   `ncols <= block_size` (256) — at 512 it would quantize **half** the V row.  Must become a strided
   loop (`col = tid; col < ncols; col += block_size`; safe, `col % 32` still equals the in-group lane).
   That edits the shipping 256 fold, so the fix-A gate set has to be re-run on 256 as well.
4. Nothing else: `ggml_cuda_fwht_warp` already loops `h = warp_size .. N` (el_w = 16 at 512),
   `s_row[512]` is 2 KiB of shared memory and `reg[16]` per thread of the first warp.

**Numerical identity is by construction, not luck.**  `ggml_cuda_op_fwht_impl` handles n=512 with
`fwht_cuda<512>` (warp per row), which calls the *same* `ggml_cuda_fwht_warp<512, warp_size>` the fused
kernel calls, on the same F32 roped row.  (Only n >= 1024 uses `fwht_cuda_block`, whose butterfly
pairing differs and would need element-wise verification.)  So the fold should be bit-identical like the
256 one; the risk is in item 3, not in the arithmetic.

**Effect if implemented (12B):** 6 -> 2 launches per full layer, so **-4/layer/step = -32 launches per
decode step** (128 -> 96 after fix A; ~13590 -> ~13270 launches for the `-n 8` run).  26B-A4B: -20/step
(5 full layers).  Also removes the F32 rotated-row round-trip: 8 layers x 2 rows x 2 KiB = 32 KiB per
decode step, ~16 MiB per 512-token prefill ubatch, and the intermediate can fall out of the compute
buffer.  **t/s: below measurement noise here** (fix A's 80 launches/step moved nothing); pricing r81's
~0.4 % per ~90 launches on gfx1201 gives ~0.1 %, and the full layers are only 1/6 of the model.

**Reach is Gemma4-only.**  I checked: only `gemma4`, `gemma4-assistant` and `gemma-embedding2` read a
distinct `ATTENTION_KEY_LENGTH_SWA`; every other arch gets `n_embd_head_k_swa = n_embd_head_k_full`
(`src/llama-model.cpp:1560`), so gemma2/gemma3/gemma3n/cohere2/exaone4/laguna/dflash are *not* helped by
a 512 width — their rotations are already in the 64/128/256 set.  (This corrects an earlier note in this
file that claimed otherwise.)

**Recommendation:** a separate, optional block-10 item at most — 4 new instantiations of the delivery's
most templated kernel (build time + object size, cf. `archive/work/build-time-regression/`) and an edit
to the shipping 256 V-write for a ~0.1 % gfx1201 win on one arch's 1/6 of layers.  Otherwise document it
as a known gap: Gemma4's full-attention layers stay unfused because their rotation width is outside the
fused kernel's instantiated set.

**Decision (maintainer, 2026-10-21): documented as a known gap, not implemented.**  It is listed with
the accepted limitations in `TODO.md` ("Documented, deliberately NOT fixed") and is not to be
re-reported.  Nothing in this directory implements it, and this campaign's fix deliberately leaves the
512 path alone.

### 5.7 Not measurable here — hand to the main box before promotion

* **Multi-device splits.**  This box has one device (`splits = 1`), so the expand-order risk that the
  non-iSWA comment warns about ("the number of splits in the graph is reduced") cannot be observed.
  The promotion gate is a graph-split/node-count comparison on gfx1201 at `-sm layer` and `-sm tensor`
  for a rotated-cache iSWA model (Gemma4 12B is enough).
* **Other iSWA archs.**  `build_attn(llm_graph_input_attn_kv_iswa*)` is also used by gemma2, gemma3,
  gemma3n, cohere2, cohere2moe, exaone4, laguna and dflash.  Only Gemma4 GGUFs exist locally, so those
  are unmeasured (they are structurally identical builders; the change is per-builder, not per-arch).
* **gfx1201 throughput.**  Where the launch overhead is a larger fraction of a token.


## 6. Promotion state

| item | state |
|---|---|
| fix | `fixA-iswa-expand-order.patch` (5 lines, `src/llama-graph.cpp`, uncommitted in `~/llama.cpp`) |
| owner block | the rope + FWHT fusion is **block 10**; the builder lives in `src/llama-graph.cpp` (block 10/11 also touch it), so the maintainer picks the fold |
| env | none added; `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` stops being a no-op on iSWA models, so `ENVIRONMENT.md` needs a sentence, not a new row |
| gfx1151 gates | launch counts, 3 greedy arms x 3 model/prompt cells, `W=1..8` probe (q8_0 + f16), prefill-logit f16 + **q8_0**, SET_ROWS/ROPE, scheduler nodes/splits/buffers, llama-bench, batched-bench: **all green / neutral** |
| still open | gfx1201 multi-device graph-split comparison (`-sm layer`, `-sm tensor`); gfx1201 perf; `scripts/validate-set.sh` after the fold |
| decision | maintainer: fold as-is, or also take the 512-wide full-attention extension (5.6) as a second item |

If the fix is promoted, this record moves to `archive/work/iswa-kv-write-fusion/` with a
`RESOLUTION.md` and `TODO.md` #76 is replaced by one `WORKLOG.md` entry (WIP rule: no stub left behind).
