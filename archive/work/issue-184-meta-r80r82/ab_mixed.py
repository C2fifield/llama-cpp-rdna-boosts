#!/usr/bin/env python3
"""Issue #184 mixed-workload A/B: 4 concurrent chats whose prefills sit inside the measured
window (the shape where the meta split-state walk runs: every request start/finish re-lays the
graph).  Only libggml-base is swapped between arms."""
import json, os, shutil, statistics, subprocess, sys, threading, time, urllib.request

LLAMA = "/home/stew675/llama.cpp"
SERVER = os.path.join(LLAMA, "build-rocm/bin/llama-server")
SO = os.path.join(LLAMA, "build-rocm/bin/libggml-base.so.0.26.0")
PORT = 8081
NP = int(os.environ.get("NP", "4"))
NPRED = int(os.environ.get("NPRED", "128"))
PLEN = int(os.environ.get("PLEN", "2000"))
PROMPT = open("/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt").read()[:PLEN]
ARMS = {"S": "/tmp/base-stock.so", "P": "/tmp/base-pmr.so"}


def env(extra):
    e = os.environ.copy()
    e.update({"LD_LIBRARY_PATH": "/opt/rocm-10.1.0-gfx120X/lib", "HIP_VISIBLE_DEVICES": "0,1"})
    e.update(extra)
    return e


def args():
    return [SERVER,
        "-m", "/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "-md", "/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        "-ngl", "99", "-ngld", "99", "-sm", "tensor", "-ncmoe", "47",
        "-fa", "auto", "-ctk", "q8_0", "-ctv", "q8_0",
        "-c", os.environ.get("CTX", "131072"), "-b", "2048", "-ub", "2048", "-t", "8",
        "-np", str(NP), "--fit", "off", "--load-mode", "none", "--lazy-mode", "off",
        "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "3",
        "--host", "127.0.0.1", "--port", str(PORT)]


def health(proc, t=600):
    t0 = time.time()
    while time.time() - t0 < t:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def post(payload, timeout=1800):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def one(arm, tag):
    shutil.copyfile(ARMS[arm], SO)
    log = f"/tmp/i184/mixed-{tag}-{arm}.log"
    f = open(log, "w")
    proc = subprocess.Popen(args(), env=env({}), stdout=f, stderr=subprocess.STDOUT)
    try:
        if not health(proc):
            print(f"{tag} {arm}: SERVER FAILED"); return None
        time.sleep(1.0)
        out = [None] * NP
        def go(s, i):
            out[i] = post({"prompt": PROMPT, "n_predict": NPRED, "temperature": 0,
                           "cache_prompt": False, "id_slot": s})
        th = [threading.Thread(target=go, args=(s, i)) for i, s in enumerate(range(NP))]
        t0 = time.time()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - t0
    finally:
        proc.terminate()
        try: proc.wait(30)
        except Exception: proc.kill()
        f.close()
    time.sleep(1.5)
    txt = open(log, errors="replace").read()
    pred = sum(o["timings"]["predicted_n"] for o in out)
    acc = sum(o["timings"].get("draft_n_accepted", 0) for o in out) / max(1, sum(o["timings"].get("draft_n", 0) for o in out))
    re = [l.split("|")[-1].strip() for l in txt.splitlines() if "graph reallocs =" in l]
    res = dict(arm=arm, agg=pred / wall, pred=pred, wall=wall, acc=acc, re=re,
               text=hash(tuple(o["content"] for o in out)))
    print(f"  {tag} {arm}: {res['agg']:.2f} t/s agg ({pred} tok / {wall:.2f}s) acc {acc:.3f} reallocs {re}", flush=True)
    return res


if __name__ == "__main__":
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tagp = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    rows = []
    for p in range(pairs):
        for arm in (["S", "P"] if p % 2 == 0 else ["P", "S"]):
            r = one(arm, f"{tagp}p{p}")
            if r:
                rows.append(r)
    xs = [r["agg"] for r in rows if r["arm"] == "S"]
    ys = [r["agg"] for r in rows if r["arm"] == "P"]
    print(f"\nS(stock): {statistics.mean(xs):.2f} t/s  {[round(x, 2) for x in xs]}")
    print(f"P(pmr):   {statistics.mean(ys):.2f} t/s  {[round(x, 2) for x in ys]}")
    d = [(y - x) / x * 100 for x, y in zip(xs, ys)]
    print(f"paired P vs S: {[round(x, 2) for x in d]} %   mean {statistics.mean(d):+.2f} %")
    print(f"text identical across arms: {len({r['text'] for r in rows}) == 1}")
