#!/usr/bin/env bash
# WIP moe-layer-stagger bench harness.  NOT part of the delivery.
#
# Usage:
#   run-stagger-bench.sh <tag> <ncmoe> <layers-spec|- for contiguous> [llama-bench extra args...]
#
#   layers-spec: explicit comma list, e.g. "1,3,5,...,47"  ("-" = contiguous first N)
#
# Env passthrough: UB, P, R, HEATMAP, STRIDE, OFFSET, TT, extra MOE_* vars.
set -euo pipefail

TAG=${1:?tag}
NCMOE=${2:?ncmoe}
LAYERS=${3:?-}
shift 3

# If an explicit layer list is given, the -ncmoe count must match its length: the count is still
# consumed by the MoE-cache preflight / fit accounting even though the loader uses the override list.
if [[ "$LAYERS" != "-" ]]; then
    NCMOE=$(awk -F, '{print NF}' <<<"$LAYERS")
fi

MODEL=${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
BIN=${BIN:-/home/stew675/llama.cpp/build-rocm-hybrid/bin/llama-bench}
LOGDIR=${LOGDIR:-/home/stew675/llama-cpp-rdna-boosts/wip/moe-layer-stagger/logs}
UB=${UB:-2048}
P=${P:-2048,4096,8192}
R=${R:-3}
TT=${TT:-8}
HEATMAP=${HEATMAP:-0}

mkdir -p "$LOGDIR"
out="$LOGDIR/${TAG}.log"

env_args=(MOE_EXPERT_HEATMAP="$HEATMAP")
if [[ "$LAYERS" != "-" ]]; then
    env_args+=(LLAMA_NCMOE_LAYERS="$LAYERS")
fi
[[ -n "${STRIDE:-}" ]] && env_args+=(LLAMA_NCMOE_STRIDE="$STRIDE")
[[ -n "${OFFSET:-}" ]] && env_args+=(LLAMA_NCMOE_OFFSET="$OFFSET")
[[ -n "${EXTRA_ENV:-}" ]] && env_args+=($EXTRA_ENV)

echo "### $TAG  ncmoe=$NCMOE layers=$LAYERS ub=$UB p=$P heatmap=$HEATMAP $(date -Is)" | tee "$out"
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1} LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib \
    env "${env_args[@]}" timeout "${TIMEOUT:-3600}" \
    "$BIN" -m "$MODEL" -ngl 99 -sm tensor -ncmoe "$NCMOE" \
    -p "$P" -n 0 -ub "$UB" -b 4096 -r "$R" -t "$TT" -fa 1 -o md "$@" \
    2>&1 | tee -a "$out"
