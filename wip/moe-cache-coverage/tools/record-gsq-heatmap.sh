#!/bin/bash
# record-gsq-heatmap.sh -- one clean MoE heat-map re-record on the maintainer's field GSQ config.
#
# Part of wip/moe-cache-coverage (2026-10-17): the previous sidecar was poisoned by the doubling bug
# (99.05% of its entries saturated at 0xffffffff) and was deleted, so this run starts cold with the r80
# build.  It is the maintainer's own llama-server command, verbatim, with only the port moved.
#
# What it does: two requests (a 16 KB prose prompt = prefill tally, then a long generation = decode
# tally), then it stops the server, which flushes the map at exit.  Afterwards:
#   wip/moe-cache-coverage/tools/heatmap-audit.py --all --arena-mib <arena> <sidecar>
set -u
LOG=${1:-/tmp/gsq-heatmap.log}
PORT=8033
SIDE=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.moe-heatmap
rm -f "$SIDE"

HIP_VISIBLE_DEVICES=2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 \
LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib \
/home/stew675/llama.cpp/build-rocm/bin/llama-server \
  --model /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --alias Qwen3.8-Flash-NextGSQ-RCO-IQ3_XXS --fit off --top-k 20 --port $PORT --threads 7 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --verbosity 3 --host 127.0.0.1 --cpu-strict 1 --predict 98304 \
  --cpu-range 1-7 --threads-http 4 --load-mode none --cache-ram 16384 --ctx-size 131072 \
  --flash-attn auto --temperature 1.0 --batch-size 4096 --ubatch-size 4096 --n-gpu-layers all \
  --no-kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --ctx-checkpoints 64 \
  --image-min-tokens 1024 --cache-idle-slots --reasoning-budget 65536 --reasoning-preserve \
  --checkpoint-min-step 4096 --lazy-mode off --n-cpu-moe 48 --split-mode tensor \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 7 --spec-draft-p-min 0.001 \
  --spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 600); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break
  sleep 1
done
echo "server ready after ${i}s"

python3 - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
prose = open("/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt").read()[:16000]
def ask(prompt, n, tag):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0}
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    txt = d["choices"][0]["message"].get("content") or ""
    print("%s: %s chars, finish=%s" % (tag, len(txt), d["choices"][0].get("finish_reason")))
ask("Read the following and then summarise its argument in five paragraphs.\n\n" + prose, 400, "prefill+decode")
PY

curl -s -m 5 "http://127.0.0.1:$PORT/metrics" -o /tmp/gsq-metrics.json 2>/dev/null
kill $SRV; wait $SRV 2>/dev/null
echo "--- cache / heat lines ---"
grep -E "arena [0-9]|expert cache|hit rate|heat \(R2\)|heat top|saved [0-9]+ heat|MoE arena" "$LOG" | tail -14
