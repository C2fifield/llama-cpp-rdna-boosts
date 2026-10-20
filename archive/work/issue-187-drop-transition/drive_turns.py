#!/usr/bin/env python3
"""Multi-turn chat against llama-server with prompt caching on (the #187 -sm layer case).

Each turn appends a user message and re-sends the whole conversation with cache_prompt=true,
so the server takes a context checkpoint and reuses the cached prompt prefix.
"""
import argparse, json, time, urllib.request

TURNS = [
    "Name three primary colours.",
    "Which of those is closest to the colour of a ripe banana?",
    "Now explain, in two sentences, why leaves look green.",
    "Summarise our conversation so far in one short paragraph.",
    "What is 17 minus 9? Answer with the number only.",
    "Give one example of a noble gas and one sentence about it.",
]

def send(port, messages, max_tokens):
    body = {"messages": messages, "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens,
            "cache_prompt": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    txt = data["choices"][0]["message"].get("content") or ""
    return txt, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--cycles", type=int, default=2, help="repeat the whole conversation this many times")
    a = ap.parse_args()

    for cyc in range(1, a.cycles + 1):
        messages = []
        for i, q in enumerate(TURNS[:a.turns]):
            messages.append({"role": "user", "content": q})
            txt, dt = send(a.port, messages, a.max_tokens)
            print(f"cycle {cyc} turn {i+1}: {dt:.2f}s {len(txt)} chars", flush=True)
            messages.append({"role": "assistant", "content": txt})

if __name__ == "__main__":
    main()
