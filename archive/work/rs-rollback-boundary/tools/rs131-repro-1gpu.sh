#!/bin/bash
# rs-rollback-boundary reproducer, single-device (gfx1151) variant of
# archive/work/moe-verify-fusions/tools/multiseq.sh.
#
# Two concurrent greedy /completion requests (A long, B short) against a GDN/SSM hybrid with MTP, then
# grep the server log for the issue-#131 "rollback crossed a batch boundary" warning and for the
# `LLAMA_RS_ROLLBACK_DEBUG=1` RSPREP/RSROLL lines that show the prepare order.
#
#   rs131-repro-1gpu.sh <build_dir> <tag>
#
# env: M, PROMPT_A, PROMPT_B, NPA=256, NPB=96, PORT=8965, GPUS=0, SM=layer, NCMOE=0, CTX=65536,
#      KVUNI=1, UBD=512, BD=4096, DBG=1
set -u
BUILD=$1; TAG=$2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CAMP=$(cd "$(dirname "$0")/.." && pwd)
M=${M:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf}
D=${D:-}
PROMPT_A=${PROMPT_A:-$CAMP/prompts/issue131-A.txt}
PROMPT_B=${PROMPT_B:-$CAMP/prompts/issue131-B.txt}
NPA=${NPA:-256}; NPB=${NPB:-96}
PORT=${PORT:-8965}
NCMOE=${NCMOE:-0}
CTX=${CTX:-65536}
KVUNI=${KVUNI:-1}
SM=${SM:-layer}
UBD=${UBD:-512}; BD=${BD:-4096}
DBG=${DBG:-1}
LOGD=${LOGD:-/tmp/rs131}; mkdir -p $LOGD
export HIP_VISIBLE_DEVICES=${GPUS:-0}
[ "$DBG" = "1" ] && export LLAMA_RS_ROLLBACK_DEBUG=1

KVD="--kv-unified"; [ "$KVUNI" = "0" ] && KVD="--no-kv-unified"
SMD=""; [ -n "$D" ] && SMD="--spec-draft-model $D"

"$BUILD/bin/llama-server" -v -m "$M" \
  -sm "$SM" -ncmoe "$NCMOE" -ub "$UBD" -b "$BD" -c "$CTX" $KVD \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit off \
  --spec-type draft-mtp $SMD --spec-draft-n-max 3 \
  --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.out 2> $LOGD/${TAG}.err &
pid=$!
ok=0
for i in $(seq 1 600); do
  kill -0 $pid 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -20 $LOGD/${TAG}.err; exit 1; }
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ $ok -eq 1 ] || { echo "$TAG HEALTH-TIMEOUT"; kill $pid 2>/dev/null; exit 1; }

curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null

python3 - $PORT "$PROMPT_A" "$PROMPT_B" "$NPA" "$NPB" > $LOGD/${TAG}.seq 2> $LOGD/${TAG}.seqerr <<'PY'
import json, sys, threading, hashlib, urllib.request
port, pA, pB, npa, npb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
A = open(pA).read(); B = open(pB).read()

def gen(prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 1,
                       "cache_prompt": False, "ignore_eos": True, "return_tokens": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    return r.get("tokens", [])

def h(t):
    return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12]

for rep in (1, 2):
    res = {}
    def run(key, prompt, n):
        res[key] = gen(prompt, n)
    t1 = threading.Thread(target=run, args=("A", A, npa))
    t2 = threading.Thread(target=run, args=("B", B, npb))
    t1.start(); t2.start(); t1.join(); t2.join()
    print("rep%d A %d %s   B %d %s" % (rep, len(res["A"]), h(res["A"]), len(res["B"]), h(res["B"])))
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "=== $TAG (rs131, 1 device) ==="
cat $LOGD/${TAG}.seq
echo "warning count: $(grep -ac 'rollback crossed a batch boundary' $LOGD/${TAG}.err)"
echo "RSPREP lines:  $(grep -ac RSPREP $LOGD/${TAG}.err)   RSROLL lines: $(grep -ac RSROLL $LOGD/${TAG}.err)"
grep -a RSROLL $LOGD/${TAG}.err | head -5
