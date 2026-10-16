# RDNA2 (`gfx1030`) beta test checklist

You need a working `gfx1030` ROCm host (RX 6800 / 6800 XT / 6900 XT, Radeon Pro V620 / W6800).  This
is a **coexistence / correctness** test, not a benchmark: the maintainer has no RDNA2 device and makes
no performance claim for this platform.

## 1. Get a build

The guards are in the delivery as of **r70**, so just pull the release image that carries `gfx1030`, or
build the release tree from source:

```bash
# fresh llama.cpp at release.json.base
bash <delivery>/scripts/apply-all.sh .
# then configure with gfx1030 in the target list using your ROCm toolchain and build llama-cli.
```

## 2. Smoke — does it start

```bash
HIP_VISIBLE_DEVICES=<rdna2> ./build/bin/llama-cli -m <model>.gguf -ngl 99 \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

Without the guards, an RDNA2 build aborts at the first fused op with `invalid kernel file`; with
them it should produce text and exit cleanly.

## 3. Correctness (informational)

* Start with `-fa off`.  Then try `-fa on` on a short prompt.  It is expected to work, but
  long-context Qwen3.8 / GLM may abort on the known upstream RDNA2 tile-kernel defect — that is a
  **finding to report**, not a failure of this candidate.
* Run one quantized dense model and one MoE model you have.  Record the exact model, quant, `-ngl`
  and the full command line.

## 4. What to report on issue #158

* `gfx1030` part, ROCm version, image tag or the exact build command;
* the startup log (backend / device enumeration lines);
* the smoke output and whether it is coherent;
* `-fa on` vs `-fa off` behaviour;
* any abort with the full backtrace;
* optional: rough prefill / decode t/s from `llama-bench`, clearly marked **informational**.
