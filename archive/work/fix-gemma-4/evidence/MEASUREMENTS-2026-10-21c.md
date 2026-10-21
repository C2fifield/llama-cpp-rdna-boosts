# Measurements 2026-10-21 (c) — the determinism instrument, and what it actually shows

Box: gfx1201, 3x R9700, ROCm 10.1.0, 16 cores.  Tree: `~/llama.cpp` @ `972f3b757` + this campaign's patch set
(`patch-state/0001..0004b`), `build-rocm`.  Config for every MTP run below: gemma4 26B-A4B Q8_0 +
`mtp-gemma-4-26B-A4B-it.gguf`, `-ngl 99 -ncmoe 40 -sm layer --fit false -c 16384 --spec-type draft-mtp
--spec-draft-n-max 3 -n 300 -t 8 --seed 42 --temp 0 --ignore-eos -f prompts/prose-rdna-boosts.txt`.

## 0. Retractions — read this before trusting `MEASUREMENTS-2026-10-21b.md`

Two of that file's conclusions are **wrong**, and both came from using the wrong instrument:

1. **"The full two-device synchronize made it deterministic (3/3)"** — a 3-run coincidence. With 12 runs the
   same arm was 50/50 again. Nothing about the scheduler's synchronize is the cause.
2. **"`GGML_CUDA_DISABLE_GRAPHS=1` is deterministic (4/4)"** — true at the *text* level only by luck; on the
   instrument below it is the one reliably stable arm, but the text hash that "proved" it was measuring a
   different thing (see next point).

**A greedy text hash is not a determinism instrument here, and `scripts/extract-generated.py` cannot be used
on a `-v` log.**  With `-v` the CLI's log interleaves debug lines into the generated region, so the extracted
text contains timing/graph lines.  Worse, the SSE chunk grouping differs run to run: one run emitted
`documentation`, another `documen` + `/tation` and dropped a ` a` token from the rendered stream — i.e. the
*displayed* text can differ with the same model output.  Every earlier "N/N identical" text claim in this
campaign is therefore unproven, including the handover's.

## 1. The instrument (this is the durable result)

Two level-appropriate observables, both saved in `../tools/`:

* **`tools/fp.sh`** — runs `llama-cli -v` and fingerprints the **draft model's candidate logits**, which the
  spec loop prints per step: `spec draft: - seq_id 0, draft candidate N, pos P: <tok> (<prob>) '<text>'`.
  Grouped per draft call and sorted, this is a logit-level series of the *draft* context (~100 calls per
  run), available on the real product path and with `n_probs = 0` (no perturbation).
* **`tools/clihash.sh`** — hashes the server's SSE delta stream (semantic text, not terminal rendering) as a
  coarse cross-check.
* `archive/work/kv-cell-offset-nondeterminism/tools/decodestart.sh` + `test-logits-width-probe` — the
  delivery's existing start-to-start logits instrument, used below for a *single* context.

## 2. What the instrument shows

| config (2 GPU unless stated) | draft-logit fingerprint (per run) | verdict |
|---|---|---|
| **target only** (no MTP): `decodestart.sh`, `NCMOE=40 SPLIT=layer`, 6 fresh processes | `c97d5e9d8bf6e0d4` ×6 | **bit-stable** — the expert cache + host experts + graphs are NOT the cause |
| MTP, graphs **off** | `8f795fbaf0` ×4 | stable (= the 1-GPU and 3-GPU graphs-on value) |
| MTP, graphs on, **1 GPU** | `8f795fbaf0` ×3 | stable |
| MTP, graphs on, **3 GPU** | `8f795fbaf0` ×2 | stable |
| MTP, graphs on, **2 GPU** (campaign tree) | `0b95…`×2, `d596…`×4; `ddf4…`, `9523…` | **unstable** |
| … + `ASYNC_INPUT=0` / `ASYNC_INPUT_BATCH=0` / `Q8_1_ARENA_FREE_OLD=1` / `SLAB=0` / `META_SEG_ASYNC=0` | still ≥2 variants | not the cause |
| … + **per-origin async-input ring** (`patch-state/0005`) | `6f6923ef36` ×8, `10e1f21f4c` ×2 | **much better, not clean** |

The divergence is always at the **first draft call** (same 5507-token history) and is a *continuous* change in
probability (e.g. top-1 0.439 vs 0.458 vs 0.453), not a garbage read: the drafter's own logits differ.  The
drafter is dense (no expert cache), so this is not a cache-band defect.

## 3. Conclusions

1. **The cache, the host-expert path, the slab, the Q8_1 arena and the graphs-recapture machinery are
   cleared**: a single context with all of them is bit-identical across 6 fresh processes.
2. **The trigger is two contexts on two devices with captured graphs** (1 and 3 devices are stable; graphs
   off is stable).  This is the r69 / TODO #75 family ("not deterministic when the consumer is a captured
   HIP graph"), still alive in the MTP draft path.
3. **A real bug was found and fixed on the way**: the async-input ring's ordering state was per device, but a
   stream is a llama_context, so the draft and target contexts consumed each other's `pending`/`q`/`done_ev`
   and could replay a graph without waiting for its own upload.  Per-origin (per calling-thread stream) state
   takes the arm from *unstable in every sample* to 8/10 stable.  Fix: `patch-state/0005-…`.
4. **`-ngld 0` is not a valid arm in this fork** — it silently runs *without* a speculator (no draft lines at
   all), so the earlier "draft on CPU → deterministic" note in `MEASUREMENTS-2026-10-21b.md` §2.1 is void.

## 4. Where to go next

The remaining ~20 % divergence is still the first draft decode on 2 devices with graphs.  Next bisect, in
order: (a) the H2D op-offload staging ring and the FA staging arena (the other per-device shared
allocations both contexts touch); (b) `GGML_SCHED_DEBUG=2` graph dump of the *draft* context to see whether
the captured draft graph differs between a `6f69` and a `10e1` run (a capture-path difference, not a race,
would explain "stable value A vs stable value B"); (c) a `GGML_CUDA_DISABLE_GRAPHS`-equivalent for the draft
context only, which is both a diagnostic and a plausible narrow fix.
