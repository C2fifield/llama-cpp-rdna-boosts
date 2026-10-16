#!/usr/bin/env bash
# Start-to-start logits determinism self-comparison for the kv-cell-offset campaign.
#
# The same build is run twice in separate processes (fresh start each time) and
# compared with `llama-perplexity --kl-divergence-base`.  If the two starts are
# bit-identical the mean KLD is ~0.001 %; any rounding-level start-to-start
# difference shows as a larger mean KLD / a same-top-p below 100 %.
#
# This is the instrument the r69 `GGML_CUDA_ASYNC_INPUT` check used (`-ub 8`), and
# it is how the reporter's "graphs-on start-to-start nondeterminism" claim (issue
# #151 follow-up / TODO #67) should be tested without depending on a greedy tie.
#
#   tools/selfkld.sh <build_dir> <tag> [K=V ...]
#
# Environment: M, CORPUS, CTX, CHUNKS, UB, GPUS, D, TIMEOUT.
#   e.g.  GGML_CUDA_DISABLE_GRAPHS=1 tools/selfkld.sh build-r69 graphs-off
#         GGML_CUDA_ASYNC_INPUT=0   tools/selfkld.sh build-r69 async-off
set -u
BUILD=$1; TAG=$2; shift 2 || { echo "usage: selfkld.sh <build_dir> <tag> [K=V ...]"; exit 2; }
ENVSTR="${*:-}"
M=${M:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}
CORPUS=${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}
CTX=${CTX:-512}; CHUNKS=${CHUNKS:-24}; UB=${UB:-8}; TIMEOUT=${TIMEOUT:-1800}
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0}
D=${D:-/tmp/kvoffset}; mkdir -p "$D"
BASE=$D/$TAG.kld

run() { # <label> <extra args...>
  local label=$1; shift
  local log=$D/$TAG-$label.log
  # shellcheck disable=SC2086  # $ENVSTR is an intended word-split K=V list
  timeout "$TIMEOUT" env ${ENVSTR} "$BUILD/bin/llama-perplexity" -m "$M" -f "$CORPUS" \
    -c "$CTX" --chunks "$CHUNKS" -ub "$UB" -b 512 -ngl 99 -fa on -t 8 \
    -ctk q8_0 -ctv q8_0 "$@" >"$log" 2>&1
}

if ! run base --kl-divergence-base "$BASE"; then
  echo "$TAG BASE-RUN-FAILED"; tail -5 "$D/$TAG-base.log"; exit 1
fi
if ! run cmp --kl-divergence-base "$BASE" --kl-divergence; then
  echo "$TAG CMP-RUN-FAILED"; tail -5 "$D/$TAG-cmp.log"; exit 1
fi

echo "=== $TAG  (build=$BUILD env: ${ENVSTR:-<default>}) ==="
grep -iE "mean KLD|same top p|rms|final estimate" "$D/$TAG-cmp.log"
