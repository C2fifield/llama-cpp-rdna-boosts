#!/usr/bin/env python3
"""4-slot concurrent greedy probe: counts graph re-layouts (fingerprint mismatch) per config."""
import json, os, subprocess, sys, threading, time, urllib.request

LLAMA = "/home/stew675/llama.cpp"
SERVER = os.path.join(LLAMA, "build-rocm/bin/llama-server")
PORT = int(os.environ.get("PORT", "8081"))
PLEN = int(os.environ.get("PLEN", "6000"))
PROMPT = open("/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt").read()[:PLEN]
BASE = {"LD_LIBRARY_PATH": "/opt/rocm-10.1.0-gfx120X/lib", "HIP_VISIBLE_DEVICES": "0,1"}

def start(log, args, extra):
    env = os.environ.copy(); env.update(BASE); env.update(extra)
    f = open(log, "w")
    return subprocess.Popen([SERVER] + args, env=env, stdout=f, stderr=subprocess.STDOUT), f

def wait_health(proc, t=900):
    t0 = time.time()
    while time.time() - t0 < t:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False

def req(slot, out, idx, npred):
    body = json.dumps({"prompt": PROMPT, "n_predict": npred, "temperature": 0,
                       "cache_prompt": False, "id_slot": slot}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                              headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as resp:
        d = json.load(resp)
    out[idx] = (slot, time.time() - t0, d.get("timings", {}))

def run(tag, args, extra=None, npred=128):
    log = f"/tmp/i184/{tag}.log"
    proc, f = start(log, args, extra or {})
    res = {"tag": tag}
    try:
        if not wait_health(proc):
            print(f"{tag}: SERVER FAILED"); print(open(log, errors="replace").read()[-3000:]); return res
        t0 = time.time(); out = [None]*4
        th = [threading.Thread(target=req, args=(s, out, i, npred)) for i, s in enumerate([0,1,2,3])]
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - t0
        tot = sum(o[2].get("predicted_n", 0) for o in out if o)
        acc = sum(o[2].get("draft_n_accepted", 0) for o in out if o)
        drf = sum(o[2].get("draft_n", 0) for o in out if o)
        res.update(wall=wall, tot=tot, agg=tot/wall)
        print(f"{tag}: wall={wall:.2f}s tokens={tot} agg={tot/wall:.2f} t/s draft_acc={acc}/{drf}")
        for o in out:
            print(f"   slot={o[0]} {o[1]:.2f}s pred={o[2].get('predicted_n')} "
                  f"ms/tok={o[2].get('predicted_ms',0)/max(1,o[2].get('predicted_n',1)):.2f}")
    finally:
        proc.terminate()
        try: proc.wait(30)
        except Exception: proc.kill()
        f.close()
    time.sleep(2)
    txt = open(log, errors="replace").read()
    n = txt.count("graph topology changed")
    res["relayouts"] = n
    print(f"{tag}: graph-topology-change (re-layout) events = {n}")
    re_ = [l.split("|")[-1].strip() for l in txt.splitlines() if "graph reallocs =" in l]
    print(f"{tag}: per-request graph reallocs (same/changed): {re_}")
    ru = [l.split("=")[-1].strip() for l in txt.splitlines() if "graphs reused" in l]
    print(f"{tag}: per-request graphs reused: {ru}")
    return res

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gemma-mtp":
        args = ["-m", "/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf",
                "-md", "/llm/models/Gemma4/12B/Q8_0/gemma-4-12B-it-MTP-Q8_0.gguf",
                "-ngl", "99", "-ngld", "99", "-sm", "tensor",
                "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "-c", "32768", "-b", "2048", "-ub", "2048", "-t", "8",
                "-np", "4", "--fit", "off",
                "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "3",
                "--host", "127.0.0.1", "--port", str(PORT), "-v"]
    elif mode == "dflash":
        args = ["-m", "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                "-md", "/llm/models/Qwen3.6/35B-A3B/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf",
                "-ngl", "99", "-ngld", "99", "-sm", "layer",
                "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "-c", "65536", "-b", "2048", "-ub", "2048", "-t", "8",
                "-np", "4", "--kv-unified", "--fit", "off",
                "--spec-type", "draft-dflash", "--spec-draft-n-max", "5",
                "--host", "127.0.0.1", "--port", str(PORT), "-v"]
    elif mode == "flashnext":
        ctx = os.environ.get("CTX", "131072")
        args = ["-m", "/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                "-md", "/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
                "-ngl", "99", "-ngld", "99", "-sm", "tensor", "-ncmoe", "47",
                "-fa", "auto", "-ctk", "q8_0", "-ctv", "q8_0",
                "-c", ctx, "-b", "2048", "-ub", "2048", "-t", "8",
                "-np", "4", "--fit", "off", "--load-mode", "none", "--lazy-mode", "off",
                "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "3",
                "--host", "127.0.0.1", "--port", str(PORT)]
    else:
        sys.exit("unknown mode")
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    run(sys.argv[3] if len(sys.argv) > 3 else mode, args, extra)
