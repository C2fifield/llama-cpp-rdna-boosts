#!/bin/bash
# dod.sh <tag> <bin> <model> <ncmoe> <ctx> <hostmode> <prompt> [extra server args...]
tag="$1"; bin="$2"; model="$3"; ncmoe="$4"; ctx="$5"; hostmode="$6"; prompt="$7"; shift 7
log="/tmp/dod-$tag.log"
env HIP_VISIBLE_DEVICES=0,1 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_ALLREDUCE=ce \
  GGML_CUDA_WORK_ALLOC_DEBUG=1 "$bin" -m "$model" --alias m --fit off --port 8036 --host 0.0.0.0 \
  --n-gpu-layers all --split-mode tensor -ncmoe "$ncmoe" -c "$ctx" -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 --no-kv-unified --load-mode ${DIAG_LOADMODE:-none} --lazy-mode off \
  --cache-ram 0 --threads 7 --cpu-range 1-7 --cpu-strict 1 --threads-http 4 --verbosity 3 \
  --host-experts "$hostmode" "$@" > "$log" 2>&1 &
pid=$!
res=LOADFAIL
for i in $(seq 1 600); do
  grep -q "listening on" "$log" && { res=OK; break; }
  grep -q "Memory access fault\|GGML_ASSERT\|Aborted\|cannot be allocated" "$log" && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
if [ "$res" != OK ]; then echo "$tag LOADFAIL"; grep -m2 -iE "fault|abort|error" "$log"; kill $pid 2>/dev/null; exit 1; fi
python3 - "$prompt" > /tmp/dod-req.json <<'PY'
import json,sys
p=open(sys.argv[1]).read()
print(json.dumps({"model":"m","messages":[{"role":"user","content":p+"\n\nSummarise the above in one sentence."}],"max_tokens":128,"temperature":1.0}))
PY
# warm pass (fills the expert cache), then the measured pass
curl -s -m 1200 http://127.0.0.1:8036/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/dod-req.json -o /dev/null
curl -s -m 1200 http://127.0.0.1:8036/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/dod-req.json -o /dev/null
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "===== $tag ($hostmode) ====="
echo -n "ring values seen: "; grep -o "WORK_ALLOC.*ring=[0-9.]*" "$log" | grep -o "ring=[0-9.]*" | sort -u | tr '\n' ' '; echo
grep -oE "WORK_ALLOC dev=[0-9]+ need=[0-9.]+ MiB narrow=[01] .*ring=[0-9.]+" "$log" | head -4
grep -E "prompt eval time|eval time =" "$log" | tail -4
grep -c "Memory access fault" "$log" | sed 's/^/faults=/'
