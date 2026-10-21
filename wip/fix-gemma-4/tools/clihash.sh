#!/usr/bin/env bash
# Determinism witness on the real CLI path: per-run hash of the server's SSE delta stream
# (reasoning_content + content), taken from `llama-cli -v`.  This is the model's own text, not
# the terminal rendering, and it does not perturb the run (no n_probs).
# usage: clihash.sh <tag> [K=V ...]
set -u
tag="$1"; shift || true
M=/llm/models/Gemma4/26B-A4B/Q8_0
LOG=/tmp/g4base/cli-$tag.log
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
env "$@" timeout 1200 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m $M/gemma-4-26B-A4B-it-Q8_0.gguf -md $M/mtp-gemma-4-26B-A4B-it.gguf \
  -ngl 99 -ncmoe 40 -sm layer --fit false -c 16384 \
  --spec-type draft-mtp --spec-draft-n-max 3 -n 300 -t 8 --seed 42 --temp 0 --ignore-eos \
  -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt \
  --no-display-prompt --single-turn -v > "$LOG" 2>&1
python3 - "$tag" "$LOG" <<'PY'
import hashlib, re, sys
tag, log = sys.argv[1], sys.argv[2]
s = open(log, errors="replace").read()
deltas = re.findall(r'"delta":\{"(?:reasoning_content|content)":"((?:[^"\\]|\\.)*)"\}', s)
txt = "".join(deltas)
acc = re.search(r'"draft_n":(\d+),"draft_n_accepted":(\d+)', s)
pred = re.search(r'"predicted_n":(\d+)', s)
print(f"{tag}: chunks={len(deltas)} chars={len(txt)} hash={hashlib.sha256(txt.encode()).hexdigest()[:12]} "
      f"acc={acc.group(2)+'/'+acc.group(1) if acc else '?'} predicted_n={pred.group(1) if pred else '?'}")
PY
