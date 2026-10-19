# BETA-TESTING — `0013-slab-narrow-floor`

A slab **placement** fix: the always-resident narrow band is learned from the real decode graph instead
of a `split_only` estimate, so a decode view is never pushed above the H2D staging ring.  Behaviour of
the kernels is unchanged; what changes is where the compute buffers live and how often the arena is
culled.

## 1. Apply + build

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama-0013 && cd ~/llama-0013
git checkout 097cf33d9
bash <delivery>/scripts/apply-all.sh .                 # 12 blocks (r83)
git am -3 <delivery>/beta/0013-slab-narrow-floor/0013-slab-narrow-floor.patch
~/bin/build-llama-rocm
```

or, in one step, from a checkout that has the set:

```bash
bash <delivery>/scripts/apply-all.sh --include-beta ~/llama-0013
```

## 2. The primary signal (the loop must be gone)

Any host-expert MoE + MTP config on ≥ 2 GPUs works; this is the one that reproduces it here
(2× R9700 gfx1201, ROCm 10.1.0, `Qwen3.8-Flash-Next IQ4_NL`, stock env):

```bash
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=ce \
  <build>/bin/llama-server -m <Flash-Next IQ4_NL> --fit off --parallel 1 \
  --ctx-size 131072 -b 6144 -ub 6144 -ngl 99 -sm tensor -ncmoe 46 \
  -fa auto -ctk q8_0 -ctv q8_0 --load-mode none --lazy-mode off \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 2 --spec-draft-n-max 6 \
  --spec-draft-p-min 0.001 -md <mtp-*.gguf> --verbosity 3
# feed a ~74k-token prompt, then 90+ generated tokens with ignore_eos
```

Pass criteria:

* **no** `moe_cache_rearm: re-armed N stood-down expert-cache tables` repeating during decode — exactly
  **one** (the normal prefill → decode drop) is expected, and zero further `moe_cache_evict_slab_range`
  lines after it;
* decode rate back to the config's normal rate (45 t/s here for `-sm tensor`; 36 t/s `-sm layer`);
* prefill rate unchanged.

Before the fix the same run printed 18 cycles ~165 ms apart and ran 17–25 t/s.

Diagnostics if you see a cycle: add `GGML_CUDA_WORK_ALLOC_DEBUG=1` and check the `WORK_ALLOC` line —
`narrow=1 ret_off=0.0` on a decode view is what a correct fix looks like; `narrow=0 ret_off=<floor+ring>`
means the band is still being pushed above the ring (send that line, with `narrow_floor=`/`ring=`, in the
report).

## 3. Gates that must stay green

| gate | expected |
|---|---|
| dense coherence (`Qwen3.5-4B Q8_0`, seed 42, temp 0) | text identical to the same build without the patch |
| Flash-Next host-expert 16k prefill, staging off vs on | greedy hash identical (`a60a39e11133` on the reference config); prefill t/s within ~2 % of the unpatched build (the band is 512 MiB bigger, so ~−2 % is expected, not a regression) |
| `scripts/gate-prefill-logits.sh` | mean KLD ≤ 0.005, same-top-p ≥ 98 % (0 / 100 % here) |
| adaptive-MTP 4 axes + phase probe (`-n 3000`, reasoning pinned) | acceptance and mean len identical to the unpatched build |
| `-sm layer` and `-sm tensor` | both loop-free |
| two-slot concurrency (`--parallel 2`) | no change vs the unpatched build |
| `scripts/check-gpu-faults.sh` | clean |

## 4. Report

Please report: GPU/arch, ROCm line, the exact command line, `re-armed` cycle count and decode t/s
before/after, prefill t/s before/after, and any `narrow floor raise ... ignored` warnings from
`set_narrow_floor` (these are expected once at the prefill→decode drop and are informational).
