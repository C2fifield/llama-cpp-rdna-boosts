# BETA TESTING — `0012-slab-transient-absorption`

Thanks for testing this.  The goal is to find the configurations where the *outside-the-slab* VRAM cushion
(and the derived MTP draft reserve) is too small for your box — this is a **memory-lifetime** change, so the
failure mode is a hard VRAM abort, not a slow number.  Everything below is designed so a failure tells us
exactly which knob was short.

The patch's own `README.md` has the why and the measured numbers; this file is the checklist.

---

## 0. What you need

* llama.cpp checkout at the **baseline recorded in `release.json`** (`097cf33d9` for `v12-097cf33d9-r80`).
* The delivery repo at the same commit as this file.
* A model with **host-resident MoE experts** (`-ncmoe N` / `-cmoe`) — that is the only configuration where the
  slab exists at all.  A fully resident model has no slab and is unaffected: it is a valid "nothing changed"
  control run.
* Optionally an MTP draft model (`--spec-draft-model ... --spec-type draft-mtp-adaptive`): that is what
  exercises the derived-draft-reserve half.

## 1. Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9                       # release.json.base for v12-097cf33d9-r80
bash <delivery-repo>/scripts/apply-all.sh --include-beta .
```

Expected: **13 commits** (blocks 00-11, then `beta 12`), and a note that the checkout is *not* the release tree
(`release.json` tree `873d0335…`).  On this box the applied tree is
`18bb050f3ff2d47c54352c8ac04a986d4a4c4349` — if yours differs, tell me: the only expected cause is a different
`git am -3` merge, which should not happen at the recorded base.

Without the flag, `apply-all.sh` behaves exactly as before (12 blocks, byte-identical to `release.json`), so you
can build both arms from the same repo.

## 2. Build

Use your normal ROCm build.  The reference build is ROCm 10.1.0 / gfx120X; 7.14 is also interesting because it
was **not** re-run during development:

```bash
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 3. The three gates that matter (run them in this order)

**3a. Coherence (must be identical).**  Same seed, temp 0, same model, vs the *unpatched* build:

```bash
./build/bin/llama-cli -m <MoE model> -ngl 99 -sm tensor -ncmoe 40 -p "The capital of France is" \
  -n 64 --seed 42 --temp 0 --no-display-prompt --single-turn
```

Diff the output text against the same command on the 12-block build.  **Identical** (or, for a coarse quant,
identical argmax per token) is what we want — a difference here means allocation changed numerics.

**3b. The prefill/watch log — this is where the failures show.**  Run your real workload with

```bash
GGML_CUDA_SLAB_STATS=1 ./build/bin/llama-server ... 2>&1 | tee run.log
```

and look for:

| watch for | meaning |
|---|---|
| `SLAB_STATS wide-drop ... free=NNN MiB` | the outside cushion at the prefill → decode transition.  **< ~200 MiB means you are close**; report the value and your config |
| `POOL_WIDE_LEAK` | a pool allocation outlived the wide band — a real bug, report it immediately with the log |
| `freeing a wide-band block ... from a dead generation` | same class; report |
| `SLAB_WIDE_AUX ... refused` | the wide region could not grow (usually transient); only interesting if it repeats every request |
| `ggml_cuda_device_malloc: ... allocation failed ... the arena cannot yield physical back` | **out of headroom** — see §5 |

**3c. Your own workload, twice.**  Because the failure mode is a VRAM abort and this change moves the cushion,
please run your normal configuration for a while (a long prompt, then some decode, then a second long prompt —
the recompute path is where the drop and the reclaim happen).  Note the `free=` figure at each `wide-drop`.

## 4. If you have more than one GPU

Run your configuration in **both** split modes (`-sm layer` and `-sm tensor`), and if you have a `-ts a,b`
split please also run that (it takes the conservative path today — we would like to know it still works).
Report per-device `free=` values from `SLAB_STATS` (the line is printed once per device).

## 5. If you run out of VRAM

This is the useful signal, not a nuisance — it means the cushion is too small for your config.  Please:

1. Re-run with `GGML_CUDA_SLAB_HEADROOM_MIB=1024` (or `2048`).  If it passes, we know the cushion is the
   problem and your config gives us a new data point for the default.
2. If `1024` does not fix it, try `MOE_EXPERT_CACHE_AUX_RESERVE_MIB=4096` too (the old flat draft reserve),
   and separately `GGML_CUDA_POOL_WIDE=0` — that tells us which term was short.
3. `GGML_CUDA_SLAB=0` is the full escape hatch back to the pre-slab allocation path.
4. Send: the exact command line, `SLAB_INIT`/`SLAB_STATS` lines around the failure, the failing allocation
   size and the `free=` value just before it, GPU model(s), VRAM, ROCm version, and the model + `-ncmoe`.

## 6. What to report (even if everything works)

* Your GPU(s) + VRAM + ROCm version + `gfxNNNN`.
* Apply result: commit count, the applied tree, any `git am -3` fallback.
* `3a` result (identical / different / different-but-argmax-identical).
* The `SLAB_STATS wide-drop ... free=` figures for your workload, per device.
* Whether you ran with an MTP draft, and `-c` / `-ub` / `-ncmoe`.
* Anything from the watch list in `3b`.

That is enough to decide the default cushion for configurations other than the development box.
