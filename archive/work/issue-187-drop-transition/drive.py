#!/usr/bin/env python3
"""Drive N concurrent greedy chat completions against llama-server, in rounds.

Each round sends one request per stream simultaneously and waits for all of them.
Distinct short prompts, temperature 0, top_k 1, max_tokens 300, cache_prompt false.
"""
import argparse, json, sys, threading, time, urllib.request

PROMPTS = [
    "Name three primary colours.",
    "What is the capital of France?",
    "Explain why the sky is blue in one sentence.",
    "List two prime numbers greater than ten.",
    "What is 17 minus 9?",
    "Name a mammal that lives in the ocean.",
    "What does HTTP stand for?",
    "Give one example of a noble gas.",
]

def one(port, idx, round_no, max_tokens, out, err):
    body = {
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)] + f" (round {round_no})"}],
        "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens, "cache_prompt": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.load(r)
        txt = data["choices"][0]["message"].get("content") or ""
        out[idx] = (len(txt), time.time() - t0)
    except Exception as e:  # noqa: BLE001
        err[idx] = repr(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()

    # warm-up
    out, err = {}, {}
    one(a.port, 0, 0, 16, out, err)
    print("warmup:", out.get(0), err.get(0))

    rc = 0
    for rnd in range(1, a.rounds + 1):
        out, err = {}, {}
        ths = [threading.Thread(target=one, args=(a.port, i, rnd, a.max_tokens, out, err))
               for i in range(a.streams)]
        t0 = time.time()
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.time() - t0
        toks = sum(v[0] for v in out.values())
        print(f"round {rnd}: wall {wall:.1f}s toks {toks} t/s {toks/wall:.1f} errs {err}", flush=True)
        if err:
            rc = 1
            print("round failed; stopping", flush=True)
            break
    sys.exit(rc)

if __name__ == "__main__":
    main()
