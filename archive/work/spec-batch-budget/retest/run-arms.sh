#!/bin/bash
# Retest kit for PR #169 (`spec-batch-budget`, LLAMA_SPEC_BATCH_BUDGET) -- see README.md.
#
# Every arm below is the exact command line used for the numbers in
# results/SUMMARY.md and in RESOLUTION.md ("Re-assessment on r79").  One fresh
# llama-server per arm; the probe primes it, runs N concurrent completions and
# writes results/<label>.jsonl + .text.
#
# Usage:
#   BIN=~/llama-169/build-r79-169/bin/llama-server ./run-arms.sh <arm-set>
#
#   arm-set: gsq-hybrid | gsq-hybrid-sweep | gsq-controls | all-vram | width-sweep | all
#
# BIN must be a build of the CURRENT delivery tree PLUS the PR #169 patch, i.e.
# the same binary for both the unset and the budget arms (that is the point: the
# env var alone selects the arm, so nothing else can drift).
set -u
if [ $# -eq 0 ]; then sed -n '2,20p' "$0"; exit 2; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBE="$HERE/../multislot-mtp-probe.py"
BIN="${BIN:?set BIN to the llama-server built from the tree + the PR patch}"
OUT="${OUT:-$HERE/results}"
NPREDICT="${NPREDICT:-300}"
REPS="${REPS:-2}"

# Models (resolve against /llm/models per AGENTS.md; names are host specific).
MOE35=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
DENSE27=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
GSQ=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
GSQ_MTP=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

arm() { # label model gpus cmoe slots budget mtp extra...
    local label=$1 model=$2 gpus=$3 cmoe=$4 npl=$5 budget=$6 mtp=$7; shift 7
    local b=()
    [ "$budget" != "-" ] && b=(--budget "$budget")
    local m=()
    [ "$mtp" != "-" ] && m=(--mtp "$mtp")
    echo "== $label (slots=$npl budget=$budget)"
    timeout 900 python3 "$PROBE" --bin "$BIN" --model "$model" "${m[@]}" \
        --gpus "$gpus" --cmoe "$cmoe" --npl "$npl" --nmax 3 \
        --npredict "$NPREDICT" --label "$label" --outdir "$OUT" "${b[@]}" "$@"
}

set="$1"
case "$set" in
gsq-hybrid)   # the contributor's shape: 2 GPU tensor + host-resident experts (the winning shape)
    for r in $(seq 1 "$REPS"); do
        arm "gsq2-cap3-r$r"   "$GSQ" 0,1 47 4 -  "$GSQ_MTP"
        arm "gsq2-budget8-r$r" "$GSQ" 0,1 47 4 8  "$GSQ_MTP"
    done
    ;;
gsq-hybrid-sweep)  # budget sweep at 4 slots: 8 = cap 1, 12 = cap 2, 16 = cap 3 = no cap
    for r in $(seq 1 "$REPS"); do
        arm "gsq2-budget12-r$r" "$GSQ" 0,1 47 4 12 "$GSQ_MTP"
        arm "gsq2-budget16-r$r" "$GSQ" 0,1 47 4 16 "$GSQ_MTP"
    done
    ;;
gsq-controls)  # controls that isolate the width cliff (see README)
    arm "coher-gsq-cap3"   "$GSQ" 0,1   47 4 -  "$GSQ_MTP"
    arm "coher-gsq-cap1"   "$GSQ" 0,1   47 4 8  "$GSQ_MTP"
    arm "ctrl-gsq-nospec"  "$GSQ" 0,1   47 4 -  -   --spec none
    arm "gsq3vram-cap3"    "$GSQ" 0,1,2  0 4 -  "$GSQ_MTP"
    arm "gsq3vram-cap1"    "$GSQ" 0,1,2  0 4 8  "$GSQ_MTP"
    arm "diag-cap3-noshexp" "$GSQ" 0,1  47 4 -  "$GSQ_MTP" --env GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1
    arm "diag-cap1-noshexp" "$GSQ" 0,1  47 4 8  "$GSQ_MTP" --env GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1
    ;;
width-sweep)  # same model/topology, vary slot count so the verify width moves
    arm "width-s2-cap3"  "$GSQ" 0,1 47 2 -  "$GSQ_MTP"   # width 8
    arm "width-s2-cap1"  "$GSQ" 0,1 47 2 4  "$GSQ_MTP"   # width 4 (budget/2-1 = 1)
    arm "width-s4-cap1b" "$GSQ" 0,1 47 4 6  "$GSQ_MTP"   # width 8
    ;;
all-vram)     # the shapes the original rejection used: no host experts
    for r in $(seq 1 "$REPS"); do
        arm "moe3-cap3-r$r"     "$MOE35"   0,1,2 0  4 -  -
        arm "moe3-budget8-r$r"  "$MOE35"   0,1,2 0  4 8  -
        arm "moe2-cap3-r$r"     "$MOE35"   0,1   0  4 -  -
        arm "moe2-budget8-r$r"  "$MOE35"   0,1   0  4 8  -
        arm "dense3-cap3-r$r"   "$DENSE27" 0,1,2 0  4 -  -
        arm "dense3-budget8-r$r" "$DENSE27" 0,1,2 0 4 8  -
    done
    arm "moe2hybrid-cap3" "$MOE35" 0,1 40 4 - -
    arm "moe2hybrid-cap1" "$MOE35" 0,1 40 4 8 -
    ;;
all)
    "$0" gsq-hybrid; "$0" gsq-hybrid-sweep; "$0" gsq-controls
    "$0" width-sweep; "$0" all-vram
    ;;
*)
    sed -n '2,20p' "$0"; exit 2
    ;;
esac
