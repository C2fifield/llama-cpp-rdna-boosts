#!/usr/bin/env bash
# Block 12 gate: mixed K/V cache types with the per-type native-FA policy.
#
# WHY THIS EXISTS:
#   Pre-block-12, type_k != type_v was HARD-REJECTED at context creation.
#   Block 12 replaces the reject with the llama_kv_types_allowed policy:
#   same-type is always allowed; mixed is allowed when both types have
#   native-FA reads and the arch has no fused single-KV-type op (QSA/MLA
#   stay same-type). The CLI cache-type list (common/arg.cpp) is IDENTICAL
#   to the native-FA list, so every CLI-expressible mixed pair is legal and
#   a non-native type cannot be expressed at all (the helper's reject branch
#   is C-API-only defense-in-depth). This gate covers the CLI side.
#
# WHAT IT CHECKS (llama-server, NOT llama-cli: the single-turn TUI swallows
# startup logs, so the context-creation WARN is only observable here):
#   1. q8_0/q4_0  : the server starts (no abort) and logs exactly ONE
#                   llama_kv_types_allowed mixed-type WARN.
#   2. q8_0/q8_0  : the server starts and logs NO llama_kv_types_allowed WARN.
#   3. iq4_xs/q8_0: the arg parser rejects the type before context creation
#                   ("Unsupported cache type") -- the CLI negative test.
#
# The policy check is backend-agnostic and the server starts with -ngl 0 by
# default: the gate must run on a box whose GPU is busy.
#
# Usage: ./gate-mixed-kv.sh [llama-server]
# Environment:
#   LLAMA_SERVER   path to llama-server  (default: ~/llama.cpp/build/bin/llama-server)
#   MODEL          small GGUF to build a context (default: local Qwen2.5-Coder-1.5B Q8_0)
#   PORT           server port           (default: 18091)
#   NGPU           -ngl value            (default: 0)
#   CTX            context size          (default: 512)
#   START_TIMEOUT  per-case start budget (default: 180 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${1:-${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}}"
MODEL="${MODEL:-/home/chris/.lmstudio/models/ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF/qwen2.5-coder-1.5b-q8_0.gguf}"
PORT="${PORT:-18091}"
NGPU="${NGPU:-0}"
CTX="${CTX:-512}"
START_TIMEOUT="${START_TIMEOUT:-180}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

[ -x "$LLAMA_SERVER" ] || { echo "FAIL: llama-server not executable: $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

echo "==> block-12 mixed K/V gate"
echo "    server=$LLAMA_SERVER"
echo "    model=$MODEL  ngl=$NGPU ctx=$CTX port=$PORT"
echo "    logs: $KEEPOUT"

fails=0

# run_case NAME TYPE_K TYPE_V MODE   (MODE: accept | reject)
run_case() {
  local name="$1" k="$2" v="$3" mode="$4"
  local log="$KEEPOUT/$name.log" status="timeout"
  "$LLAMA_SERVER" -m "$MODEL" -ngl "$NGPU" -c "$CTX" -ctk "$k" -ctv "$v" \
    --port "$PORT" >"$log" 2>&1 &
  SRV_PID=$!
  local t
  for t in $(seq 1 "$START_TIMEOUT"); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      if grep -q "Unsupported cache type" "$log" 2>/dev/null; then status="argfail"; fi
      [ "$status" = "timeout" ] && status="crash"
      break
    fi
    if grep -q "llama_server: listening" "$log" 2>/dev/null; then status="ready"; break; fi
    if grep -q "Unsupported cache type" "$log" 2>/dev/null; then status="argfail"; break; fi
    sleep 1
  done
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
  SRV_PID=""

  local warns warn_lines
  warns=$(grep -c "llama_kv_types_allowed: mixed KV cache types $k/$v" "$log" 2>/dev/null || true)
  case "$mode" in
    accept)
      if [ "$status" = "ready" ]; then
        if [ "$k" != "$v" ]; then
          [ "$warns" = "1" ] || { echo "  FAIL $name: expected exactly 1 mixed-type WARN, got $warns (status=$status)"; fails=$((fails+1)); return; }
          echo "  PASS $name: $k/$v accepted, mixed-type WARN logged once"
        else
          [ "$warns" = "0" ] || { echo "  FAIL $name: expected no mixed-type WARN, got $warns"; fails=$((fails+1)); return; }
          echo "  PASS $name: $k/$v accepted, no mixed-type WARN (same-type silent)"
        fi
      else
        echo "  FAIL $name: server did not reach 'listening' (status=$status, see $log)"; fails=$((fails+1))
      fi ;;
    reject)
      warn_lines=$(grep -n "llama_kv_types_allowed" "$log" 2>/dev/null | head -3)
      if [ "$status" = "argfail" ] && [ -z "$warn_lines" ]; then
        echo "  PASS $name: $k rejected at arg parse (never reaches the policy gate)"
      else
        echo "  FAIL $name: expected an arg-parser rejection, got status=$status"; fails=$((fails+1))
      fi ;;
  esac
}

run_case q80-q40  q8_0   q4_0   accept
run_case q80-q80  q8_0   q8_0   accept
run_case iq4xs    iq4_xs q8_0   reject

echo "==> mixed-KV gate: $([ "$fails" = "0" ] && echo "ALL PASS" || echo "$fails FAIL(S)"); logs in $KEEPOUT"
[ "$fails" = "0" ]
