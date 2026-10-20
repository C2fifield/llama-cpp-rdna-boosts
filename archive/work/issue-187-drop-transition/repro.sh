#!/usr/bin/env bash
# Issue #187 reproduction: 2x R9700 (gfx1201), Flash-Next GSQ-IQ3_XXS + shared Q8_0 MTP head.
# 4 concurrent greedy 300-token requests against `-sm tensor -ncmoe 10`, MTP n-max 3.
# The r87 `n_tokens_prev > 8` forced-drop hunk is expected to fault the GPU after a slot leaves.
#
# Usage: ./repro.sh [--sm tensor|layer] [--np N] [--rounds N] [--port PORT] [--tag NAME]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BIN="${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
MTP="${MTP:-/models/Qwen3.8/Flash-Next/IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
SM=tensor
NP=4
ROUNDS=3
PORT=18087
TAG="repro"
NCMOE=10
THREADS=8

while [ $# -gt 0 ]; do
    case "$1" in
        --sm) SM="$2"; shift 2 ;;
        --np) NP="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --ncmoe) NCMOE="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

OUT="${OUT:-$HERE/runs}"
mkdir -p "$OUT"
LOG="$OUT/${TAG}-$(date +%Y%m%d-%H%M%S).log"
echo "log: $LOG"

# leave the GPU-IRQ cores free (top NUM_GPUS cores); one CCD (8) is within 1-2 % of best
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export LLAMA_KQ_MASK_DERIVED=0
export GGML_SCHED_SYNC_GRAPH_INPUTS=1
export LLAMA_MTP_SPARSE=0
export LLAMA_MTP_DRAFT_OP_OFFLOAD=0
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib:${LD_LIBRARY_PATH:-}

"$BIN" -m "$MODEL" -c 262144 -ngl 99 -ctk q8_0 -ctv q8_0 \
  -cram 2048 -ub 2048 -b 2048 --reasoning-effort low --temp 0.7 --top-k 20 --top-p 0.95 --min-p 0 \
  --spec-type draft-mtp -md "$MTP" -ngld 99 --spec-draft-n-max 3 \
  --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0 \
  -sm "$SM" --n-cpu-moe "$NCMOE" -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off \
  -t "$THREADS" --host 127.0.0.1 --port "$PORT" -np "$NP" >"$LOG" 2>&1 &
SRV=$!
echo "server pid $SRV"

cleanup() { kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then echo "server died during load"; tail -40 "$LOG"; exit 1; fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "server did not become healthy"; tail -40 "$LOG"; exit 1; fi
echo "server healthy; running workload ($NP streams x $ROUNDS rounds)"

if [ "${DRIVE:-rounds}" = "turns" ]; then
    python3 "$HERE/drive_turns.py" --port "$PORT" --turns "${TURNS:-6}" --cycles "${CYCLES:-2}" --max-tokens "${MAXTOK:-200}"
else
    python3 "$HERE/drive.py" --port "$PORT" --streams "$NP" --rounds "$ROUNDS" --max-tokens "${MAXTOK:-300}"
fi

sleep 2
echo "--- fault check ---"
"$REPO/scripts/check-gpu-faults.sh" --since "10 min ago" --log "$LOG" || true
echo "--- log tail ---"
tail -40 "$LOG"
