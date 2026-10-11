#!/bin/bash
# Issue #131 repro: `seq_rm: rollback crossed a batch boundary` with -np 2 + draft-mtp.
# A (short, slot 0) finishes prefill and decodes with MTP while B (long, slot 1) is still
# prefilling in 512-token ubatches.  Reports A's solo vs concurrent text hashes and the
# warning lines.
#   issue131_repro.sh <build_dir> <tag>
# Env: KVUNI=1|0, CHUNKED=0 to force sequential GDN, GPUS=0, PROMPTS.
set -u
BUILD=$1; TAG=$2
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M=${M:-/llm/models/Qwen3.8/27B/Q5_K/Qwen3.8-27B-UD-Q5_K_M.gguf}
PROMPT_A=${PROMPT_A:-$DIR/../prompts/issue131-A.txt}
PROMPT_B=${PROMPT_B:-$DIR/../prompts/issue131-B.txt}
NPA=${NPA:-128}; NPB=${NPB:-96}
PORT=${PORT:-8990}
CTX=${CTX:-32768}
UB=${UB:-512}
NB=${NB:-4096}
KVUNI=${KVUNI:-1}
LOGD=/tmp/srr131; mkdir -p $LOGD
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${GPUS:-0}
KVD=""; [ "$KVUNI" = "1" ] && KVD="--kv-unified"; [ "$KVUNI" = "0" ] && KVD="--no-kv-unified"
CHK=""; [ "${CHUNKED:-1}" = "0" ] && CHK="GGML_CUDA_GDN_CHUNKED=0"

env $CHK "$BUILD/bin/llama-server" -v -m "$M" \
  -ngl 99 -c "$CTX" -b "$NB" -ub "$UB" $KVD -np 2 \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port $PORT --no-webui --cache-ram 0 \
  > $LOGD/${TAG}.out 2> $LOGD/${TAG}.err &
pid=$!
ok=0
for i in $(seq 1 600); do
  kill -0 $pid 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -8 $LOGD/${TAG}.err; exit 1; }
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
    return json.load(urllib.request.urlopen(req, timeout=7200)).get("tokens", [])
def h(t): return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12]
a_solo = gen(A, npa)
print("A_solo %4d %s" % (len(a_solo), h(a_solo)))
res = {}
def run(key, prompt, n): res[key] = gen(prompt, n)
for rep in (1, 2, 3):
    t1 = threading.Thread(target=run, args=("A%d" % rep, A, npa))
    t2 = threading.Thread(target=run, args=("B%d" % rep, B, npb))
    t1.start(); t2.start(); t1.join(); t2.join()
    print("A_conc%d %4d %s" % (rep, len(res["A%d" % rep]), h(res["A%d" % rep])))
def fd(x, y):
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]: return i
    return None
for rep in (1, 2, 3):
    print("A_conc%d vs A_solo first-diff: %s" % (rep, fd(res["A%d" % rep], a_solo)))
for rep in (2, 3):
    print("A_conc%d vs A_conc1 first-diff: %s" % (rep, fd(res["A%d" % rep], res["A1"])))
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "=== $TAG (issue131) KVUNI=$KVUNI CHUNKED=${CHUNKED:-1} ==="
cat $LOGD/${TAG}.seq
echo "  rollback-crossed warnings: $(grep -ac 'rollback crossed a batch boundary' $LOGD/${TAG}.err)"
grep -a "rollback crossed a batch boundary" $LOGD/${TAG}.err | sed 's/^[0-9. ]*//' | head -4 | sed 's/^/    /'
grep -aoE 'draft acceptance = [0-9.]+' $LOGD/${TAG}.err | tail -1 | sed 's/^/  /'
