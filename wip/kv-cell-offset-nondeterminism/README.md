# `kv-cell-offset-nondeterminism` — the unified-KV cell-offset / attention-tile alignment non-determinism (TODO #52 part C)

**Status: PARKED (2026-10-11, after `v16-a55e952b8-r43`; mechanism refined 2026-10-11 — see
[`FINDINGS-offset-mechanism.md`](FINDINGS-offset-mechanism.md)).** Not part of the delivery; nothing here
may be applied to the fork without the maintainer's go-ahead (see the WIP and promotion rules in
`AGENTS.md`). This is the working home for the residual that survived the r42 (part A) and r43 (part B)
fixes to issue #134 / TODO #52.

> **2026-10-11 disposition — PARKED, not archived.** The pragmatic call is to stop here: a proper fix
> means per-sequence contiguous, tile-aligned KV regions (or per-region anchored reductions) in the
> unified attention cache, i.e. a fundamental change to upstream's design, and there are more pressing
> items. The finding is documented as a gotcha instead: **`-np > 1` + `--kv-unified` is not run-to-run
> deterministic — use `--no-kv-unified` when strict reproducibility is required.** It reproduces on
> unmodified upstream `a55e952b8`, so it is upstream-inherent rather than a delivery defect. Issue #134
> is closed as upstream-inherent (comment 2026-10-11); reopen this campaign only if @DanoPTT (or the
> maintainer) decides it is worth the upstream redesign.

> **2026-10-11 refinement:** the dependence is on the sequence's whole physical **cell set**, not just its
> start offset. A fixed allocation alignment fixes the one-shot prefill probe but **not** the decode/
> server case (the head-reset reuse fragments a growing sequence). The required quantum is the per-launch
> FA reduction/split period, which is model- and shape-dependent (4B GQA4 takes the generic paths; GSQ
> qwen4exp GQA12 now uses the RDNA4 band as of issue #133). Non-unified is green because each sequence owns a contiguous
> region from cell 0 — the proposed fix is per-sequence contiguous, tile-aligned regions in the unified
> attention cache. Evidence and probe: `FINDINGS-offset-mechanism.md`, `tools/`.
>
> **2026-10-11 (issue #133):** GSQ qwen4exp GQA 12 was **excluded** from the RDNA4 decode/verify band
> when this campaign was written (`4 < gqa_ratio <= 8`).  The #133 fix raises the band cap to 12 and the
> kernel folds the 12 heads into two `ncols2 = 8` groups (the second partial), so the GSQ model's
> decode/verify band now takes `flash_attn_ext_f16`/`flash_attn_stream_k_fixup_uniform` instead of
> `flash_attn_tile`.  Any re-run of this campaign's offset probe must be redone on that build: the
> reduction grouping changed, so the recorded GSQ quantum/table below is pre-#133.  The 4B (GQA 4) is
> unaffected.

> **Cold start.** Read this file in full, then `AGENTS.md` ("Report findings before shipping",
> "Default-on policy", "WIP and promotion rules"), then the cold-start prompt
> [`PROMPT-kv-cell-offset.md`](PROMPT-kv-cell-offset.md). The direct predecessor is
> `archive/work/moe-verify-fusions/` (the campaign that was closed when this one was seeded), in
> particular `FINDINGS-B-readback-ordering.md` and `HANDOVER-multiseq-bc.md`.

---

## 0. What this campaign is for

Root-cause and fix the **single remaining non-determinism** in concurrent multi-sequence serving:
under a **unified KV cache**, two sequences decoding at the same time produce a sequence's result that
depends on the *other* sequence's KV-cache allocation history. Identical greedy repetitions of the same
two requests are not byte-reproducible, and a concurrent run differs from the same request run solo.

This is **not** a model, quant, MoE, expert-cache, split-mode or fork issue — it reproduces on stock
upstream `a55e952b8`, on two different model files, with all experts on GPU, with layer and tensor split,
and with the expert cache off. It is a property of the unified-KV attention path.

The purpose of this campaign is to turn the mechanism below into a **performance-neutral fix** (the
maintainer's preferred end state is *shape-uniform kernels with cross-sequence batching and unified KV
both intact*, not a feature stand-down).

## 1. The finding (the state of knowledge)

**A sequence's attention result depends on the physical offset of its KV cells in the unified stream,
modulo the flash-attention reduction tile.** Move a sequence's cells by a tile-aligned amount and its
logits are byte-identical; move them by a non-aligned amount and the logits change at the rounding level
(and a greedy near-tie can flip).

Direct evidence — one unified context (`n_seq_max=2`), same seq0 tokens, only the companion fill on seq1
changes length, which shifts seq0's cells (`wip/.../test-multiseq-same.cpp`, `MSQS_ONLY0=1`):

| seq1 fill | seq0 last-token hash | max | verdict |
|---|---|---|---|
| 0    | `375b3dce0622dce5` | 24.326786 | reference |
| 64   | `375b3dce0622dce5` | 24.326786 | unchanged |
| **100**  | **`3c074b69bddb612b`** | **24.330185** | **changed** |
| 128  | `375b3dce0622dce5` | 24.326786 | unchanged |
| 192  | `375b3dce0622dce5` | 24.326786 | unchanged |
| 256  | `375b3dce0622dce5` | 24.326786 | unchanged |
| **300**  | **`8509e85d4a2343b2`** | **24.225134** | **changed** |
| 384 / 512 | `375b3dce0622dce5` | 24.326786 | unchanged |

The companion is fully masked out of seq0's attention, so its *contents* cannot matter — only the
**physical offset** it forces on seq0's cells. The alignment quantum observed here is **64** (the FA
K-tile / KV chunk). The change is purely rounding-level (~0.011 in the max logit), which is why it never
shows as corruption and only flips greedy near-ties.

**Mechanism (working model).** In unified KV the sequences share one K/V buffer. The FA kernel iterates
the dense key range `[0, n_kv)` in fixed tiles and masks out other sequences. Each sequence's keys are
therefore grouped into tile partial sums according to their **absolute** cell indices; if the co-resident
sequence's growth shifts them off the tile grid, the partial-sum grouping (and hence the fp32 rounding)
changes. In non-unified KV each sequence owns its own stream and always starts at a fixed offset, so the
grid alignment is stable and the result is reproducible.

**Server confirmation (the field symptom).** `llama-server -np 2 --kv-unified`, greedy
(`temperature 0, top_k 1, seed 1, cache_prompt false`), request A ~30k chars / 256 tokens and B ~16k
chars / 96 tokens, 3 concurrent repetitions (GSQ model, fork build):

```
A_solo  479dbd37464c
A_conc1 295c61b071ca   # != solo
A_conc2 70f263ee2307   # != conc1, != solo
A_conc3 d9aacf6b95a2   # != conc1/2, != solo
```

Same shape on IQ3_XXS (`--spec-type none`) and on unmodified upstream `a55e952b8`. `--no-kv-unified`
with the same requests is **fully green** (solo == conc1 == conc2 == conc3).

## 2. What is NOT the cause (tested and excluded — do not re-litigate)

| arm | result |
|---|---|
| `--no-kv-unified` | **green** (the only green axis) |
| `-sm layer` vs `-sm tensor` | both red |
| `-ncmoe 0` / no `-ncmoe` (all experts on GPU, 3 GPUs) | red, byte-identical to each other |
| `MOE_EXPERT_CACHE_MIB=0` | red |
| `GGML_CPU_MOE_OFFLOAD_THREADS=1` | red |
| `LLAMA_KV_N_PAD_MIN=65536` | red, **byte-identical** to default |
| `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1` | red, byte-identical |
| `LLAMA_KQ_MASK_DERIVED=0` | red, byte-identical |
| `NPB=256` (B does not finish early; no `seq_rm` during A) | red |
| model file: GSQ vs unsloth IQ3_XXS | both red |
| upstream `a55e952b8` (no fork) | red |
| single sequence | bit-reproducible (the trigger needs ≥2 live sequences) |

Note the readback-ordering fix delivered in **r43** (`GGML_READBACK_SYNC`, block 06) does **not** change
any of these arms — that fix addressed the multi-ubatch logits copy (job B), not this.

## 3. Repro

**Fork build** (the canonical r42/r43 trees both reproduce; no fork knob is needed):
```bash
git clone https://github.com/stew675/llama.cpp && cd llama.cpp
git fetch origin rdna-boosts && git checkout rdna-boosts
BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
```

**Server gate** (`archive/work/moe-verify-fusions/tools/multiseq.sh` and its `_plain`/`_np2` variants):
```bash
# unified, 2 slots, plain (no MTP), GSQ target; green iff A_solo == A_conc1..3
PROMPTS=<repo>/wip/kv-cell-offset-nondeterminism/prompts
M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  PROMPT_A=$PROMPTS/mixed30k-v2.txt PROMPT_B=$PROMPTS/msqB.txt \
  GPUS=0,1 PORT=8997 <repo>/wip/kv-cell-offset-nondeterminism/tools/msq_plain.sh  \
  <build-dir> tag
# compare: same with --no-kv-unified  -> green
# (the prompts are historical field prompts.  mixed30k-v2.txt is the same doc dump as mixed30k.txt
#  (README.md + ENVIRONMENT.md + CONTAINERS.md + archive/docs/baseline-history.md) with the GPU-fault
#  strings reworded, so a log grep for a fault no longer matches the echoed prompt:
#  99159 B, 33729 tokens, sha256 3e22e0fd6c60e507caac394d314725a15f4eaddf7a34ccc89dc47aefb42b2071.
#  The original mixed30k.txt (99174 B, 33733 tokens, sha256 69624f4d207f40bd...) is KEPT unchanged,
#  because the archived campaign results and the field tools refer to its hash.  msqB.txt is the short
#  companion (sha256 8079985bb6bc7c48ecbd2d95e60ebfd396e4a78e9c56b63051e840ddda565057).
#  Check faults with `<repo>/scripts/check-gpu-faults.sh --log <server.log>` rather than grepping the
#  log: it reads the kernel journal and filters the echoed prompt lines.)
```
(Create the `_np2` variant by adding `-np 2` to the server line: `multiseq.sh` never passes `-np`, so
the server otherwise auto-sets `-np 4` and forces `kv_unified=true`.)

**In-process offset probe** (`test-multiseq-same.cpp`, copied into `tests/`, add
`llama_build(test-multiseq-same.cpp)` to `tests/CMakeLists.txt`):
```bash
# sweep the companion fill; seq0 changes at non-multiples of 64
for bf in 0 64 100 128 192 256 300 384 512; do
  HIP_VISIBLE_DEVICES=0 MSQS_SEQ=2 MSQS_TOK=512 MSQS_UB=512 MSQS_UNI=1 MSQS_ONLY0=1 MSQS_BFILL=$bf \
    ./build/bin/test-multiseq-same -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -fa on -t 8
done
```

## 4. Hypotheses / fix directions (ordered; none implemented yet)

1. **Anchor the FA reduction grid to the sequence, not to cell 0.** If the K-tile schedule started at
   each sequence's first cell (or at `seq_pos_min(s)`'s cell) instead of the buffer base, a sequence's
   partial-sum grouping would be independent of the co-resident cells. Needs a per-sequence tile origin
   through the FA dispatch; the mask already knows the sequence's cells.
2. **Per-sequence contiguous cell ranges in unified mode.** Keep each sequence's cells contiguous and
   tile-aligned inside the shared buffer, so its offset is stable. The recurrent cache already does a
   `find_slot` gather/reorder for state cells; the attention cache does not. Costs fragmentation and a
   per-step compaction; measure.
3. **Make the FA reduction offset-invariant** by construction (e.g. fixed-size canonical tiles over the
   *unmasked* keys, or a reduction order keyed on position rather than cell index). The general
   "shape-uniform kernel" direction; the most work, the best end state.
4. **Scoped fallback** (last resort, per the maintainer's D1): if the condition cannot be repaired,
   stand unified down only for the exact case (multi-live-sequence decode) with the blast radius
   proven — but the goal is to keep unified intact, so this is not the target.

The distinguishing experiment for (1) vs (2): keep the *number* of live sequences and their
*contents* fixed, and only change the **allocation order** (e.g. prefill A-then-B vs B-then-A). If seq0's
logits change, the offset is the variable; if they do not, keep looking.

## 5. Gates to keep green

`scripts/validate-set.sh`; coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42);
`scripts/gate-prefill-logits.sh` (KLD <= 0.005 / same-top-p >= 98 %); `test-recurrent-state-depth`
(baseline: 130 failures) and `test-recurrent-state-rollback`; `MUL_MAT_ID` 931/931;
`FLASH_ATTN_QSA` >= 26/26; `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (rule-5, decode must not
regress); `archive/work/moe-verify-fusions/tools/multiseq.sh` (unified red vs non-unified green is the
reproducer, not a gate yet).

## 6. Pointers

- `archive/work/moe-verify-fusions/` — the closed predecessor campaign (A, B, the fused-verify work).
  `FINDINGS-B-readback-ordering.md` holds the offset sweep and the condition matrix;
  `HANDOVER-multiseq-bc.md` is the original B/C handover.
- `WORKLOG.md` 2026-10-10 (r42) and 2026-10-11 (r43); `TODO.md` #52.
- Issue #134 (`https://github.com/stew675/llama-cpp-rdna-boosts/issues/134`) and the upstream report it
  came from; #124 for the original field symptom.
- `benchmarks/mtp-adaptive-methodology.md` (rule 5), `benchmarks/prefill-logit-methodology.md`.
- `GREEDY-PURITY.md` — the purity rulebook (a width-uniform offset effect is exactly the class the
  `W=1..8` matrix is blind to).
