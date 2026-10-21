#!/usr/bin/env python3
"""Issue #184 A/B, pre-warmed: every slot's prompt is cached before measuring, so the four
streams decode fully concurrently from t0 and there is no prefill interleaving in the window."""
import json, os, shutil, statistics, subprocess, sys, threading, time, urllib.request

LLAMA = "/home/stew675/llama.cpp"
SERVER = os.path.join(LLAMA, "build-rocm/bin/llama-server")
SO = os.path.join(LLAMA, "build-rocm/bin/libggml-base.so.0.26.0")
PORT = 8081
NP = int(os.environ.get("NP", "4"))
NPRED = int(os.environ.get("NPRED", "192"))
PLEN = int(os.environ.get("PLEN", "2000"))
PROMPT = open("/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt").read()[:PLEN]
ARMS = {"S": "/tmp/base-stock.so", "P": "/tmp/base-pmr.so"}
ARM_ENV = {"S": {}, "P": {}}
LOG = "/tmp/i184"

def env(extra):
    e = os.environ.copy()
    e.update({"LD_LIBRARY_PATH": "/opt/rocm-10.1.0-gfx120X/lib", "HIP_VISIBLE_DEVICES": "0,1"})
    e.update(extra); return e

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
        if proc.poll() is not None: return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200: return True
        except Exception: time.sleep(0.5)
    return False

def post(payload, timeout=1800):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)

def one(arm, extra, tag):
    shutil.copyfile(ARMS[arm], SO)
    e2 = dict(ARM_ENV.get(arm, {})); e2.update(extra); extra = e2
    log = f"{LOG}/{tag}-{arm}.log"
    f = open(log, "w")
    proc = subprocess.Popen(args(), env=env(extra), stdout=f, stderr=subprocess.STDOUT)
    try:
        if not health(proc):
            print(f"{tag} {arm}: SERVER FAILED"); return None
        # warm every slot's prompt cache (serial, 1 token) so the measured window is pure decode
        for s in range(NP):
            post({"prompt": PROMPT, "n_predict": 1, "temperature": 0,
                  "cache_prompt": True, "id_slot": s})
        time.sleep(1.0)
        out = [None]*NP
        def go(s, i):
            out[i] = post({"prompt": PROMPT, "n_predict": NPRED, "temperature": 0,
                           "cache_prompt": True, "id_slot": s})
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
    pm = [o["timings"]["prompt_n"] for o in out]
    acc = sum(o["timings"].get("draft_n_accepted", 0) for o in out)
    drf = sum(o["timings"].get("draft_n", 0) for o in out)
    res = dict(arm=arm, agg=pred/wall, pred=pred, wall=wall, acc=acc/max(1,drf), prompt_n=pm,
               mstok=[round(o["timings"]["predicted_ms"]/o["timings"]["predicted_n"],2) for o in out],
               re=[l.split("|")[-1].strip() for l in txt.splitlines() if "graph reallocs =" in l],
               text=hash(tuple(o["content"] for o in out)))
    print(f"  {tag} {arm}: {res['agg']:.2f} t/s agg ({pred} tok / {wall:.2f}s)  "
          f"ms/tok {res['mstok']}  acc {res['acc']:.3f}  prefill_tokens {pm}", flush=True)
    return res

if __name__ == "__main__":
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    tagp = sys.argv[3] if len(sys.argv) > 3 else "ab"
    arms = sys.argv[4].split(",") if len(sys.argv) > 4 else ["A", "M"]
    rows = []
    for p in range(pairs):
        for arm in (arms if p % 2 == 0 else arms[::-1]):
            r = one(arm, extra, f"{tagp}p{p}")
            if r: rows.append(r)
    xs = [r["agg"] for r in rows if r["arm"] == arms[0]]
    ys = [r["agg"] for r in rows if r["arm"] == arms[1]]
    print(f"\n{arms[0]}: {statistics.mean(xs):.2f} t/s  {[round(x,2) for x in xs]}")
    print(f"{arms[1]}: {statistics.mean(ys):.2f} t/s  {[round(x,2) for x in ys]}")
    d = [(x-y)/y*100 for x, y in zip(xs, ys)]
    print(f"paired {arms[0]} vs {arms[1]}: {[round(x,2) for x in d]} %   mean {statistics.mean(d):+.2f} %")
    print(f"text identical across arms: {len({r['text'] for r in rows}) == 1}")
