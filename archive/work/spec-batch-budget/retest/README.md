# Retest kit -- PR #169 `spec-batch-budget` (`LLAMA_SPEC_BATCH_BUDGET`)

Everything needed to re-run, or to re-test a revised patch from the contributor, in one place.
The analysis and the verdict live in `../RESOLUTION.md` ("Re-assessment on r79"); this directory is
only the apparatus.

**Status of the patch: not in the delivery.** The contributor (`@briansp2020`) is refining it at his
end; the maintainer's reply (`PR #169` comment 6235431120) asks for a width-based cap, the 2-slot and
3-slot rows, the adaptive-controller feed and per-step reporting. This kit exists so the next
revision costs one command instead of a session.

## Contents

| file | what |
|---|---|
| `../0001-server-optional-load-aware-MTP-draft-cap-LLAMA_SPEC_.patch` | the patch under test (applies cleanly to the r79 tree) |
| `../multislot-mtp-probe.py` | the probe: one `llama-server` per arm, N concurrent completions, aggregate + per-step metrics + per-slot text sha256 |
| `run-arms.sh` | the exact command line for every arm, in sets |
| `results/SUMMARY.md` | every recorded run as one table |
| `results/*.jsonl` | one row per arm run (the raw numbers the tables are built from) |
| `results/*.text` | the generated text per slot, with its sha256 (the coherence check) |

## What to do when a new revision arrives

1. **Build one binary** with the new patch applied (the arms are selected by env var, so both arms
   must share one binary):
   ```bash
   cd ~/llama.cpp && git worktree add --detach ~/llama-169 rdna-boosts   # or reuse an existing worktree
   cd ~/llama-169 && git apply <new>.patch
   BUILD_DIR=build-r79-169 ~/bin/build-llama-rocm          # ROCm 10.1.0 is the canonical tree
   ```
   Do **not** pass `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="`; the build script's include-order guard
   aborts the configure if the HIP flag is overridden (see `AGENTS.md`, Common tasks -> "Build the
   fork").
2. **Run the core set first** (the shape the PR claims, about 4 minutes):
   ```bash
   BIN=~/llama-169/build-r79-169/bin/llama-server ./run-arms.sh gsq-hybrid
   ```
   Pass = `gsq2-budget8-r*` clearly above `gsq2-cap3-r*` (the recorded baseline is 145-155 vs 76-87
   t/s), with the same or better coherence (step 3).
3. **Check coherence, not just t/s.** `results/<label>.text` holds the text per slot. On the same
   prompt set, cap 1 and cap 3 are **byte-identical at 2 slots** in the r79 record
   (`sha a59941368946` / `a2c9e2955a1c`); at 4 slots they differ between arms *and between two runs of
   the same arm*, which is the pre-existing concurrent-batching nondeterminism, not the patch. Scan
   for `////`, `<unused32>`, or a repetition loop, and compare lengths (`chars` in the jsonl).
4. **Then the rest**: `gsq-hybrid-sweep` (is a higher budget still neutral?), `width-sweep` (does the
   effect still track the verify width rather than the slot count?), `all-vram` (no regression on the
   shapes that rejected v1), `gsq-controls` (plain decode reference and the width-cliff controls).
5. **Also re-run the spec's own gate** unchanged: `scripts/gate-prefill-logits.sh` is not relevant here
   (no prefill change), but the server patch must still pass the normal coherence gate on a
   non-speculative run, and the adaptive arm (`--spec-type draft-mtp-adaptive` with a 4-slot burst
   followed by a single request) is still **untested** and is re-entry condition 3.

## Reproducibility

`run-arms.sh width-sweep` was run a second time after the kit was written, and reproduced the record
within noise (2 slots: 123.4 -> 125.9 unset, 113.0 -> 113.2 with `=4`; 4 slots with `=6`: 143.0 ->
157.6 t/s), and again produced **byte-identical 2-slot text** in both arms
(`sha a59941368946` / `a2c9e2955a1c`). The width sweep is the cheapest end-to-end check of the kit.

## Environment of the recorded numbers

3x R9700 (gfx1201, 34.2 GB each), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), delivery tree
`v12-097cf33d9-r79` + the patch, one binary for all arms. Server args (from the probe): `-ngl 99
-sm tensor -mg 0 -c 24576 -np 4 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp
--spec-draft-n-max 3 --spec-draft-p-min 0`, greedy (`temperature 0`, `top_k 1`), `n_predict 300`,
`ignore_eos`, `cache_prompt false`, a distinct prompt per slot, one untimed priming request per server.

Two caveats on the numbers: the server was left on its default `-t` (the arms all share it, so the
relative deltas stand, but a thread-tuned setup will differ in absolute t/s), and the model set is the
local one: the contributor's Flash-Next GSQ **IQ3_S** is not on this box, so **GSQ-RCO IQ3_XXS** stands
in for it. That is the most likely reason his paired figure is +3.5 % where this box measures +80-90 %
on the same shape; treat the shape as confirmed and the magnitude as open.

## What the cliff is (why the arms are chosen this way)

Per-step cost on the contributor's shape (Flash-Next GSQ, 2 GPU, `-ncmoe 47`), 4 slots, from
`results/SUMMARY.md`:

| verify width | cap | ms per accepted token |
|---|---|---|
| 4 (plain decode) | - | 5.97 |
| 8 | 1 | 6.4-6.9 |
| 12 | 2 | 11.5-12.2 |
| 16 | 3 | 11.5-13.0 |

The knee is between 8 and 12 rows, i.e. `MMVQ_MAX_BATCH_SIZE == 8` (MMVQ/MMVF -> MMQ, FA -> WMMA).
It is flat all-in-VRAM on 3 GPUs (5.29 vs 5.17) and flat on 35B-A3B with host experts (2.85 vs 2.71),
and `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` does not remove it, so it is specific to
**qwen4exp + host-resident experts + a 2-GPU tensor split**. The cap is a workaround for that path,
not a tuning win: on this config the *default* runs multi-slot MTP slower than plain decode
(76-87 vs 167.5 t/s). If the wide path is ever fixed, the cap's benefit should be re-measured rather
than assumed.

## Pitfalls hit while recording this

- Passing `-md <mtp head>` together with `--spec-type none` aborts (a draft head cannot be loaded
  standalone). The plain-decode control must omit `-md` entirely.
- One arm = one fresh server. Reusing a server across arms carries acceptance and cache state.
- Interleave or repeat arms: the 4-slot runs swing several t/s between fresh servers on this box.
- `LLAMA_SPEC_BATCH_BUDGET` is read once (function-local `static`), so changing it needs a new server.
- The MoE heat-map sidecar is written next to the first host-expert shard; delete stale ones if a
  model is re-quantised between runs.
