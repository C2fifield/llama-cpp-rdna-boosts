#!/bin/bash
bin="$1"; tag="$2"; prompt="$3"
log="/tmp/iq4nl-$tag.log"
env HIP_VISIBLE_DEVICES=0,1 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_ALLREDUCE=ce ${DIAG_EXTRA_ENV:-} \
  "$bin" --model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  --alias m --fit off --top-k 20 --port 8035 --threads 7 --parallel 1 --top-p 0.95 --min-p 0.001 \
  --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --predict 98304 --cpu-range 1-7 --threads-http 4 \
  --load-mode ${DIAG_LOADMODE:-none} --cache-ram 16384 --ctx-size 131072 --flash-attn auto --temperature 1.0 \
  --batch-size 2048 --ubatch-size 2048 --n-gpu-layers all --no-kv-unified --cache-type-k q8_0 \
  --cache-type-v q8_0 --ctx-checkpoints 64 --image-min-tokens 1024 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 --lazy-mode off \
  --n-cpu-moe 48 --split-mode tensor --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
  --spec-draft-p-min 0.001 \
  --spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' > "$log" 2>&1 &
pid=$!
echo "[$tag] pid=$pid, waiting for load..."
for i in $(seq 1 600); do grep -q "listening on" "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
if ! grep -q "listening on" "$log"; then echo "[$tag] FAILED TO START"; tail -20 "$log"; kill $pid 2>/dev/null; exit 1; fi
echo "[$tag] loaded; sending 70K prompt"
python3 - "$prompt" > /tmp/iq4nl-req.json <<'PY'
import json,sys
p=open(sys.argv[1]).read()
print(json.dumps({"model":"m","messages":[{"role":"user","content":p+"\n\nSummarise the above in one sentence."}],"max_tokens":32,"temperature":1.0}))
PY
curl -s -m 900 http://127.0.0.1:8035/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/iq4nl-req.json -o /dev/null
grep -E "prompt eval time|eval time =|SLAB_INIT|slab .*GiB|H2D staging|shortfall|ring|WORK_ALLOC|narrow" "$log" | tail -40
kill $pid 2>/dev/null
