#!/usr/bin/env python3
"""Multi-slot MTP throughput probe (PR #169 re-assessment).

Starts one llama-server, waits for health, posts N concurrent /completion requests
(same n_predict, distinct prompts, cache_prompt off) and reports:
  - aggregate decode t/s (sum of tokens / longest per-slot decode time)
  - per-slot t/s
  - MTP draft acceptance parsed from the server log ("draft acceptance = ..."),
    plus a derived per-step count and mean verify width.

Usage: ms169.py --bin <llama-server> --model <m> --gpus 0,1,2 --split tensor
                 [--extra-args ...] [--budget N] --label <name> [--reps 2]
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request

PROMPT = ("You are writing a detailed technical manual for a fictional software project. "
          "Explain, step by step and with concrete examples, how a job queue processes work, "
          "how retries and backoff are chosen, and how backpressure is signalled to producers. ") * 6


def post(port, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, t0, time.time()


def wait_health(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("server did not become healthy")


def run_arm(args, rep):
    port = args.port
    log = f"{args.outdir}/{args.label}.r{rep}.log"
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = args.gpus
    if args.budget is not None:
        env["LLAMA_SPEC_BATCH_BUDGET"] = str(args.budget)
    else:
        env.pop("LLAMA_SPEC_BATCH_BUDGET", None)
    for kv in (args.env or []):
        k, _, val = kv.partition("=")
        env[k] = val
    cmd = [args.bin, "-m", args.model, "-ngl", "99", "-sm", args.split, "-mg", "0",
           "-c", str(args.ctx), "-np", str(args.npl), "-b", "2048", "-ub", "2048",
           "-ctk", "q8_0", "-ctv", "q8_0",
           "--spec-type", args.spec, "--spec-draft-n-max", str(args.nmax),
           "--spec-draft-p-min", "0", "--metrics", "-v", "--port", str(port)]
    if args.mtp:
        cmd += ["-md", args.mtp, "-ngld", "99"]
    if args.cmoe:
        cmd += ["--n-cpu-moe", str(args.cmoe)]
    cmd += args.extra_args

    lf = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        load_s = wait_health(port, proc)
        # warmup: one short request, then let the server settle
        post(port, {"prompt": "warmup " + PROMPT, "n_predict": 16, "temperature": 0.0,
                    "cache_prompt": False})
        time.sleep(2.0)
        results, t0s = [], []
        lock = threading.Lock()

        def one(i):
            body = {"prompt": f"[slot {i}] " + PROMPT, "n_predict": args.npredict,
                    "temperature": 0.0, "top_k": 1, "ignore_eos": True,
                    "cache_prompt": False, "n_probs": 0}
            out, t0, t1 = post(port, body)
            with lock:
                results.append(out)
                t0s.append((t0, t1))

        th = [threading.Thread(target=one, args=(i,)) for i in range(args.npl)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = max(t1 for _, t1 in t0s) - min(t0 for t0, _ in t0s)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
        lf.close()

    text = open(log, errors="replace").read()
    acc = re.findall(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
                     r"mean len =\s*([\d.]+)", text)
    acc = acc[-args.npl:] if len(acc) >= args.npl else acc   # drop the warmup line
    import hashlib
    texts = [r.get("content", "") for r in results]
    shas = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts]
    with open(f"{args.outdir}/{args.label}.text", "w") as tf:
        for i, t in enumerate(texts):
            tf.write(f"===== slot {i} sha={shas[i]} chars={len(t)} tokens={results[i]['timings']['predicted_n']}\n")
            tf.write(t + "\n")
    pred_n = [r["timings"]["predicted_n"] for r in results]
    pred_ms = [r["timings"]["predicted_ms"] for r in results]
    total_n = sum(pred_n)
    decode_wall = max(pred_ms) / 1000.0
    agg = total_n / decode_wall
    per_slot = [r["timings"]["predicted_per_second"] for r in results]
    gen = sum(int(g) for _, _, g, _ in acc)
    accd = sum(int(a) for _, a, _, _ in acc)
    mean_len = (sum(float(m) for _, _, _, m in acc) / len(acc)) if acc else 0.0
    steps = (accd / (mean_len - 1.0)) if (mean_len > 1.0) else 0.0
    steps_per_slot = steps / len(pred_n) if pred_n else 0.0
    cap = args.nmax if args.budget is None else max(1, args.budget // args.npl - 1) if args.npl > 1 else args.nmax
    row = dict(label=args.label, rep=rep, budget=args.budget, gpus=args.gpus, split=args.split,
               npl=args.npl, nmax=args.nmax, cap=cap, load_s=round(load_s, 1),
               agg_tps=round(agg, 1), per_slot=[round(x, 1) for x in per_slot],
               tokens=total_n, decode_wall_s=round(decode_wall, 2),
               texts_sha=shas, chars=[len(t) for t in texts],
               acc_lines=len(acc), acc_ratio=round(accd / gen, 4) if gen else None,
               mean_len=round(mean_len, 2),
               steps_per_slot=round(steps_per_slot, 1),
               step_ms=round(decode_wall / steps_per_slot * 1000.0, 1) if steps_per_slot else None,
               accepted_per_step=round(total_n / steps_per_slot, 1) if steps_per_slot else None)
    print(json.dumps(row), flush=True)
    with open(f"{args.outdir}/{args.label}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--split", default="tensor")
    ap.add_argument("--cmoe", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=24576)
    ap.add_argument("--npl", type=int, default=4)
    ap.add_argument("--npredict", type=int, default=300)
    ap.add_argument("--nmax", type=int, default=3)
    ap.add_argument("--spec", default="draft-mtp")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", default="/tmp/ms169")
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--extra-args", nargs="*", default=[])
    ap.add_argument("--env", action="append", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for r in range(1, args.reps + 1):
        run_arm(args, r)


if __name__ == "__main__":
    main()
