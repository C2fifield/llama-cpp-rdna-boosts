#!/bin/bash
# heat-growth-test.sh -- is the persisted MoE heat map accumulating, or doubling per flush?
#
# Part of wip/moe-arena-defrag FINDINGS-2026-10-17 (§4.3).  The delivery's save path re-added the whole
# running `t.heat` into `g_heat_loaded` at every debounced flush, so a server session doubled the map per
# flush; after ~6 flushes every entry is 0xffffffff and every heat-driven ranking silently degenerates to
# expert-id order.  This test makes that visible in three turns.
#
# Usage:  ./heat-growth-test.sh <label>     (label names the log: /tmp/heat-test-<label>.log)
# Expect (fixed):     25140 -> 33156 -> 41172   (linear; the file equals the session total)
#         (delivered): 25140 -> 58296 -> 99468   (2*prev + new; 2.4x in three turns)
# The model/knobs below are a cheap MoE (gpt-oss-20B MXFP4, -ncmoe 99, 4 GiB arena) so a full A/B costs
# a rebuild plus ~20 s.  Delete the sidecar the run leaves behind when you are done.
set -u
SIDE=/llm/models/gpt-oss-20b/mxfp4/gpt-oss-20B-MXFP4_MOE.moe-heatmap
LOG=/tmp/heat-test-$1.log
rm -f "$SIDE" "$SIDE".bak
cd ~/llama.cpp
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib \
MOE_EXPERT_CACHE_MIB=4096 MOE_EXPERT_HEATMAP=1 MOE_EXPERT_HEATMAP_DEBOUNCE_S=0 MOE_EXPERT_HEATMAP_MAX_FAILS=1 \
./build-rocm/bin/llama-server -m /llm/models/gpt-oss-20b/mxfp4/gpt-oss-20B-MXFP4_MOE.gguf \
  -ngl 99 -ncmoe 99 -c 4096 -t 8 -np 1 --port 8123 -lv 4 > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:8123/health 2>/dev/null | grep -q '"status":"ok"' && break
  sleep 1
done
echo "server up after ${i}s"
for turn in 1 2 3; do
  curl -s -o /dev/null -X POST http://127.0.0.1:8123/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Turn $turn: write two sentences about the sea. Then a third about mountains.\"}],\"max_tokens\":24,\"temperature\":0}"
  T=$(python3 - "$SIDE" <<'PY'
import struct,sys
try: b=open(sys.argv[1],'rb').read()
except FileNotFoundError: print("no-file"); raise SystemExit
magic,ver,nt,r0,mb,ntn,r1=struct.unpack_from("<4sIIIQII",b,0)
off=32; tot=0; sat=0; n=0
for i in range(nt):
    layer,role,ne,hb=struct.unpack_from("<i64sIQ",b,off); off+=80
    h=struct.unpack_from("<%dI"%ne,b,off); off+=4*ne
    tot+=sum(h); n+=ne; sat+=sum(1 for v in h if v==0xffffffff)
print("total=%d sat=%.2f%%"%(tot,100*sat/n))
PY
)
  echo "after turn $turn: $T"
done
kill $SRV; wait $SRV 2>/dev/null
grep -E "heat \(R2\)|saved .* heat entries|debounce|ignoring the heat map" "$LOG" | tail -8
