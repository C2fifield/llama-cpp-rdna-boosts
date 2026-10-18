#!/bin/bash
# issue #170 reproduction: emulate a 24 GiB card on a 32 GiB card, then start llama-server
# and report what it does (LISTENING / ABORTED at <assert> / EXITED with <error>).
#
#   repro.sh <logfile> <fill_mib_per_device> [extra llama-server args...]
#
# fill_mib 8192  -> 24210 MiB free of 32624 MiB  (7900 XTX class; the fit then reduces n_gpu_layers)
# fill_mib 0     -> no filler (a 2 x 32 GiB card fully offloads; this succeeds without the fix)
#
# Extra args for the two failure classes:
#   -ctk f32 -ctv f32 --fit off -ngl 99        KV cache OOM at load        (defect 2)
#   --fit off -ngl 99                          weight buffer OOM at load   (defect 2)
#   GGML_COMPUTE_BUFFER_MARGIN_PCT=3000 repro.sh …   compute buffer OOM   (defect 2)
set -u
# the reporter is on 2 cards; keep the filler and the server on the same set
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
LOG=${1:?logfile}; FILL_MIB=${2:-8192}; shift 2 || true
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HOME/llama.cpp/build-rocm/bin}
FILL_BIN=${FILL_BIN:-/tmp/vram_fill}

# build the filler once (source ships next to this script)
if [ ! -x "$FILL_BIN" ]; then
    hipcc -O2 -o "$FILL_BIN" "$HERE/vram_fill.cpp" --offload-arch=${OFFLOAD_ARCH:-gfx1201} || exit 1
fi

pkill -9 -x llama-server 2>/dev/null
pkill -9 -x vram_fill  2>/dev/null
sleep 3

if [ "$FILL_MIB" != "0" ]; then
    "$FILL_BIN" "$FILL_MIB" > /tmp/vram_fill.log 2>&1 &
    FILL=$!
else
    FILL=
fi
SRV=
cleanup(){ [ -n "${SRV:-}" ] && kill -9 "$SRV" 2>/dev/null; [ -n "${FILL:-}" ] && kill -9 "$FILL" 2>/dev/null; wait 2>/dev/null; sleep 3; }
trap cleanup EXIT
[ -n "${FILL:-}" ] && { sleep 8; cat /tmp/vram_fill.log; }

"$BUILD/llama-server" --model "$MODEL" --ctx-size 196608 --split-mode tensor \
    --no-mmproj --port 18099 "$@" > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do
    sleep 2
    if grep -q "listening on" "$LOG" 2>/dev/null; then echo "OUTCOME: LISTENING"; exit 0; fi
    if grep -q "GGML_ASSERT\|dumped core" "$LOG" 2>/dev/null; then
        echo "OUTCOME: ABORTED"; grep -a "GGML_ASSERT" "$LOG" | head -1; exit 2; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        wait "$SRV" 2>/dev/null; echo "OUTCOME: EXITED rc=$?"
        grep -aE "failed to allocate|failed to (reserve|initialize|create context|load model)" "$LOG" | tail -4
        exit 3; fi
done
echo "OUTCOME: TIMEOUT"; exit 4
