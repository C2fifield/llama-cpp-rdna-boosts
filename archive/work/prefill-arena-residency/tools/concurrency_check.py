#!/usr/bin/env python3
"""Two-slot concurrency check for the prefill-arena-residency (R2) prototype.

Scenario A: slot 0 runs a long (16k) prefill while slot 1 runs a stream of short
            prefill+decode requests.
Scenario B: slot 0 runs a long decode (a ~6k-token "novel") while slot 1 runs a
            stream of short ~4k prefills (and a control run with no concurrency).

Both use the field loader (`--load-mode none --lazy-mode off`) and
`GGML_MOE_CACHE_STAGE=1`.  Records per-request timings and greps the server log
for the cache report (H2D staged split, PRS rebuilds).

Usage:
  python3 tools/concurrency_check.py --quick     # harness smoke (short lengths)
  python3 tools/concurrency_check.py             # full run (~12-15 min)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(ROOT))
LLAMA = os.environ.get("LLAMA_DIR", "/home/stew675/llama.cpp")
SERVER = os.path.join(LLAMA, "build-rocm/bin/llama-server")
MODEL_DIR = "/llm/models/Qwen3.8/Flash-Next/IQ4_NL"
MODEL = os.path.join(MODEL_DIR, "Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf")
PORT = int(os.environ.get("PORT", "8033"))

P16K = "/tmp/prefill16k.txt"
PROSE = os.path.join(REPO, "prompts/prose-rdna-boosts.txt")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def server_env(stage):
    env = os.environ.copy()
    env.update({
        "LD_LIBRARY_PATH": "/opt/rocm-10.1.0-gfx120X/lib",
        "HIP_VISIBLE_DEVICES": "0,1",
        "LLAMA_HC_BLK16": "1",
        "LLAMA_HC_RES16": "1",
        "GGML_CUDA_MMB_DOWN16": "1",
        "GGML_MOE_CACHE_STAGE": stage,
        "GGML_CUDA_SLAB_STATS": "1",
    })
    return env


def start_server(log_path, stage="1"):
    args = [
        SERVER, "-m", MODEL, "-ngl", "99", "-sm", "tensor", "-ncmoe", "48",
        "-fa", "1", "-ctk", "f16", "-ctv", "f16",
        "-c", "131072", "-b", "4096", "-ub", "4096", "-t", "8",
        "--parallel", "2", "--load-mode", "none", "--lazy-mode", "off",
        "--fit", "off", "--spec-type", "none",
        "--host", "127.0.0.1", "--port", str(PORT), "-v",
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(args, env=server_env(stage), stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def post(payload, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    body["_wall_s"] = time.time() - t0
    return body


def timings(name, body):
    t = body.get("timings", {})
    return {
        "name": name,
        "prompt_n": t.get("prompt_n"),
        "prompt_tps": t.get("prompt_per_second"),
        "pred_n": t.get("predicted_n"),
        "pred_tps": t.get("predicted_per_second"),
        "wall_s": round(body.get("_wall_s", 0.0), 2),
    }


def warmup():
    post({"prompt": "Hello", "n_predict": 2, "temperature": 0, "seed": 1, "cache_prompt": False})


def scenario_a(quick):
    print("== Scenario A: long prefill + concurrent short decodes ==", flush=True)
    long_text = read(P16K)
    if quick:
        long_text = long_text[:4000]
    short_text = read(PROSE)
    n_short = 4 if quick else 14
    out = []

    def long_call():
        out.append(timings("long_prefill", post({
            "prompt": long_text, "n_predict": 8, "temperature": 0.0,
            "seed": 42, "cache_prompt": False,
        })))

    th = threading.Thread(target=long_call)
    th.start()
    time.sleep(0.3)
    for i in range(n_short):
        if not th.is_alive():
            break
        try:
            out.append(timings(f"short_{i}", post({
                "prompt": short_text, "n_predict": 32, "temperature": 0.2,
                "seed": 100 + i, "cache_prompt": False,
            })))
        except Exception as e:
            out.append({"name": f"short_{i}", "error": str(e)})
    th.join()
    return out


def stream_decode(prompt, n_predict, window_s, out):
    """Run a long decode with SSE streaming and count tokens in a fixed window."""
    payload = {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.8,
        "top_k": 20, "top_p": 0.95, "seed": 42, "ignore_eos": True,
        "cache_prompt": False, "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=window_s + 120) as r:
            while time.time() - t0 < window_s:
                line = r.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("content"):
                    n += 1
                if obj.get("stop"):
                    break
    except Exception as e:
        out.append({"name": "long_decode_window", "error": str(e)})
    dt = time.time() - t0
    out.append({"name": "long_decode_window", "tokens": n,
                "window_s": round(dt, 2), "tps": round(n / dt, 2) if dt else 0})


def scenario_b(quick):
    print("== Scenario B: long decode + concurrent short prefills ==", flush=True)
    long_prompt = "Write a short novel about a robot that learns to paint. Begin now.\n\n"
    short_text = read(PROSE)
    window = 20 if quick else 60
    n_dec = 10000
    if quick:
        short_text = short_text[:6000]
    else:
        short_text = short_text[:12000]
    out = []

    # B1 control: long decode alone, same fixed window
    print(f"  B1 control: long decode alone ({window}s window)", flush=True)
    stream_decode(long_prompt, n_dec, window, out)

    # B2: long decode with a concurrent stream of short prefills
    print(f"  B2: long decode + short prefills ({window}s window)", flush=True)
    th = threading.Thread(target=stream_decode, args=(long_prompt, n_dec, window, out))
    th.start()
    time.sleep(3.0)
    t_end = time.time() + window
    i = 0
    while time.time() < t_end:
        try:
            out.append(timings(f"short_prefill_{i}", post({
                "prompt": short_text + f"\n\n<!-- {i} -->\n", "n_predict": 4,
                "temperature": 0.0, "seed": 200 + i, "cache_prompt": False,
            }, timeout=600)))
        except Exception as e:
            out.append({"name": f"short_prefill_{i}", "error": str(e)})
            break
        i += 1
    th.join()
    return out


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()


def cache_report(log_path):
    kept = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if any(k in line for k in (
                "prefill cache staging", "prefill freeze", "activity prefill",
                "MoE expert cache:", "per-device sizing")):
                kept.append(line.rstrip())
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenario", choices=["a", "b", "both"], default="both")
    ap.add_argument("--stage", default="1")
    ap.add_argument("--out", default="/tmp/concurrency_check.json")
    args = ap.parse_args()

    all_results = {}
    for sc in (["a", "b"] if args.scenario == "both" else [args.scenario]):
        log_path = f"/tmp/conc_{sc}_stage{args.stage}.log"
        print(f"starting server for scenario {sc.upper()} (log {log_path})", flush=True)
        proc, log = start_server(log_path, args.stage)
        try:
            if not wait_health():
                print("server did not become healthy", file=sys.stderr)
                return 1
            warmup()
            results = scenario_a(args.quick) if sc == "a" else scenario_b(args.quick)
        finally:
            stop_server(proc, log)
        all_results[sc] = {"requests": results, "cache": cache_report(log_path)}
        longs = [r for r in results if r["name"].startswith("long")]
        shorts = [r for r in results if r["name"].startswith("short")]
        pts = [r["prompt_tps"] for r in shorts if r.get("prompt_tps")]
        pds = [r["pred_tps"] for r in shorts if r.get("pred_tps")]
        import statistics
        summary = {
            "long": longs,
            "short_n": len(shorts),
            "short_prompt_tps": {"median": round(statistics.median(pts), 1),
                                 "min": round(min(pts), 1), "max": round(max(pts), 1)} if pts else None,
            "short_pred_tps": {"median": round(statistics.median(pds), 1),
                               "min": round(min(pds), 1), "max": round(max(pds), 1)} if pds else None,
        }
        print(json.dumps(summary, indent=2), flush=True)
        print("cache tail:", json.dumps(all_results[sc]["cache"][-4:], indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
