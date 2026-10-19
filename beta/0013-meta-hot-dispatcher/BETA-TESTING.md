# TESTING — `0013-meta-hot-dispatcher`

This candidate is a **pure code motion** in block 00 (`ggml-backend-meta.cpp`): no behaviour change, no
new env knob, nothing to tune.  It exists because the maintainer decided (2026-10-18) to fold the
`ggml_backend_meta_get_split_state` hot-path shape into block 00 at the next revision rather than cut a
standalone release for it.  Use this file if you want to exercise it before then, or as the checklist
when it is promoted.

The why, the measured numbers and the promotion steps are in `README.md`; this file is the checklist.

---

## 0. What you need

* llama.cpp checkout at `097cf33d9` (the `release.json` base of the `v12-097cf33d9-r86` candidate).
* The delivery repo at the same commit as this file.
* Any model — the change is unconditional, but the path it touches only exists for a **multi-device
  `-sm tensor`** run (a single device still creates a one-device Meta backend, so it is worth running
  there too, and a plain `-sm layer` run is the "nothing changed" control).

## 1. Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9
bash <delivery-repo>/scripts/apply-all.sh --include-beta .
```

Expected: **14 commits** (blocks 00-11, `beta 12`, `beta 13`), all with strict `git am`, and the
`NOT the release tree` note.  Applied tree for this patch alone (no beta 12): `82360b48…`; with beta 12
on top: `bb825861…`.

## 2. Build

```bash
~/bin/build-llama-rocm          # or: cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 \
                                #       -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
```

## 3. Gates (in this order)

1. **Coherence, byte-identical** to the same tree *without* this patch — same seed, same prompt, both
   `-sm tensor` and `-sm layer`:
   ```bash
   HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -p "$(cat prompts/reasoning.txt)" \
     -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn
   ```
   The generated text must be identical; only the graph-allocation shape changed.
2. **Issue #170 regression** (this is the hunk's reason to exist): the 24 GiB-parity reproduction in
   `archive/work/issue-170-meta-alloc/` — hold VRAM so the card reports ~24 GiB free on 2 devices, then
   the reporter's `-sm tensor` command (plus the `--fit off`/f32-KV/20000-MiB variants).  Expected:
   loads and answers, no `GGML_ASSERT(ggml_backend_buffer_is_meta(...))`, and the genuine KV / compute /
   weight OOMs still fail cleanly (exit 1, no VRAM leak).
3. **`scripts/gate-prefill-logits.sh`** — expected mean KLD 0 / same-top-p 100 % (no kernel or
   approximate path is touched; the gate is run because the graph-allocator path is in line).
4. **Adaptive-MTP 4-axis + phase-switching** (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`,
   reasoning pinned) — expected exactly equal to the pre-change build.
5. **`llama-batched-bench -npl 1,4,8`** stock-relative (the multi-sequence decode path).
6. `scripts/check-gpu-faults.sh` on the log of a long run (cheap; no kernel changed, so this is a
   sanity check only).

## 4. If you also want the issue #184 measurement

The 4-concurrent-stream harness used for the original report is archived:
`archive/work/issue-184-meta-r80r82/ab2.py` (pre-warms every slot's prompt, then fires four greedy
`n_predict 192` requests; arms alternate, one fresh server per arm, only `libggml-base.so` swapped).
Reference result on this box: **-0.06 %** vs the delivery tree, greedy text identical.
