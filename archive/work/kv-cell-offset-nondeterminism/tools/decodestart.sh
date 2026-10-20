#!/usr/bin/env bash
# Fresh-start decode-logits determinism check (the issue #151 / TODO #67 axis).
#
# `llama-perplexity` only exercises the forward/prefill axis (a single pass over the
# corpus), so it cannot see a *decode/graph* start-to-start difference.  This tool
# runs `test-logits-width-probe` in N separate processes (a fresh context each,
# a real W=1..8 decode batch) and prints the W=1..8 row hashes; any hash that differs
# between starts is start-to-start decode nondeterminism.  Run it once with graphs on
# and once with `GGML_CUDA_DISABLE_GRAPHS=1` to test the graphs axis.
#
#   tools/decodestart.sh <build_dir> <tag> <N> [K=V ...]
#
# Environment: M (model), P (prompt), PTOK (prefill prefix length), GPUS, D, TIMEOUT.
#   P=<repo>/prompts/prose-rdna-boosts.txt  PTOK=256  tools/decodestart.sh build-r69 graphs-on  4
#   GGML_CUDA_DISABLE_GRAPHS=1 P=<repo>/prompts/prose-rdna-boosts.txt PTOK=256 \
#       tools/decodestart.sh build-r69 graphs-off 4
#   KV=q8_0 M=<27B Q5_K_M> P=<repo>/prompts/recall.txt PTOK=512 \
#       tools/decodestart.sh build-r69 27b-q8-P512 4
set -u
BUILD=$1; TAG=$2; N=$3; shift 3 || { echo "usage: decodestart.sh <build_dir> <tag> <N> [K=V ...]"; exit 2; }
ENVSTR="${*:-}"
M=${M:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}
P=${P:-/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}
PTOK=${PTOK:-256}; TIMEOUT=${TIMEOUT:-900}
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0}
D=${D:-/tmp/kvoffset}; mkdir -p "$D"
LOG=$D/decode-$TAG.log; : > "$LOG"

for i in $(seq 1 "$N"); do
  # shellcheck disable=SC2086  # $ENVSTR is an intended word-split K=V list
  out=$(timeout "$TIMEOUT" env ${ENVSTR} "$BUILD/bin/test-logits-width-probe" "$M" "$P" "$PTOK" 512 2>/dev/null \
        | grep -E 'row0_row1|width_purity')
  printf 'run %d: %s\n' "$i" "$out"
  printf '%s\n' "$out" >> "$LOG"
done

echo "=== $TAG  (build=$BUILD model=$(basename "$M") P=$PTOK env: ${ENVSTR:-<default>}) ==="
echo "--- distinct W=1 row-0 hashes (want exactly one) ---"
grep -o 'row0_row1_hashes: 1:[0-9a-f]*' "$LOG" | sed 's/.*1://' | sort | uniq -c
