# `dflash-meta-split-impl` — Meta distributed `TOP_K` (or split→mirrored gather) for the DFlash2 selector (TODO #58 follow-up)

**Status: OPEN (seeded 2026-10-13), scoping + design.**  Not part of the delivery.  This is the *real
fix* behind the clean rejection folded into r54
([`archive/work/dflash-meta-split/RESOLUTION.md`](../../archive/work/dflash-meta-split/RESOLUTION.md)).

> **Cold start.**  Read `AGENTS.md` ("Report findings before shipping", the WIP/promotion rules, the
> Meta invariants under "Critical facts"), then the r54 resolution record, then the code sites below.

## 0. The item

Under `-sm tensor`, a DFlash2 drafter with a selector builds `ggml_top_k(draft_logits)`
(`src/models/dflash.cpp:538`).  `-sm tensor` vocab-splits the draft logits (axis 0), and the Meta
backend's `handle_per_row` (`ggml/src/ggml-backend-meta.cpp:707`) rejects a per-row op whose src0 has
`GGML_BACKEND_SPLIT_AXIS_0`.  r54 replaces the raw assert with a clean `use -sm layer` rejection at
model load (block 11).  This campaign lands the real fix so `-sm tensor` DFlash2 runs.

## 1. Two routes

- **(a) Distributed top-k.**  Give the Meta backend a `GGML_OP_TOP_K` handling for a split-axis-0
  input: each device computes its local top-k over its vocab slice (with local indices), then a merge
  step produces the global top-k indices over all devices.  The merge needs a gather of per-device
  candidates (small: `top_k x n_tokens` per device) to a host or a leader device, then a final
  top-k, then a broadcast of the indices.  This is a new operation on a reduction axis, comparable in
  spirit to the existing `PARTIAL` reduce path.
- **(b) Split→mirrored gather.**  Materialise a mirrored (full-vocab) copy of the draft logits on every
  device before the selector runs.  Simpler conceptually, but it must be a real Meta op (each device
  reads all other devices' slices); the draft logits are `n_vocab x n_tokens` and the selector only
  runs on a small DFlash block, so the copy is small.  Look for an existing mirror/gather utility
  (`GGML_META_SPLIT_COPY`, the op-offload H2D staging, `moe_cache_gather`) before adding one.

The choice is a design decision for the maintainer; (b) is likely the smaller and safer change if a
mirror facility can be reused, (a) is the clean long-term shape.

## 2. Constraints and invariants to respect

- The Meta split state must stay consistent for every tensor the new path produces; a mixed/unknown
  state is a crash on whichever backend reaches it (`GREEDY-PURITY.md` §20).
- Do not regress the working configs: all-resident `-sm tensor`, `-sm layer`, DFlash2 `-sm layer`,
  and every non-DFlash arch.
- The width-purity rule: any per-row kernel's reduction order must be uniform across the decode width
  band (`W=1..8`) if it becomes part of a decode path.
- Keep the clean rejection as the fallback for any config the fix does not cover.

## 3. Repro

```bash
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
D=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1 ./build-rocm-hybrid/bin/llama-cli -m "$M" -md "$D" \
  --spec-type draft-dflash --spec-draft-n-max 3 -ngl 99 -sm tensor -c 16384 \
  -ctk q8_0 -ctv q8_0 -f prompts/prose-rdna-boosts.txt -n 512 --seed 42 --temp 0 \
  --reasoning off --no-display-prompt --single-turn
```

Currently: clean rejection (r54).  Target: loads and generates coherently.

## 4. Gates when a fix exists

- The repro loads and generates coherently with `-sm tensor`.
- Single-sequence DFlash `-sm tensor` F1 on == off, and `--parallel 2` host vs F1 identical (the
  `archive/work/dflash-multiseq/` protocol).
- `test-backend-ops` unaffected; Meta split-state gates (`GGML_META_SS_VERIFY=1`) green.
- If the change touches a prefill path: `scripts/gate-prefill-logits.sh`.
- `scripts/validate-set.sh` if promoted; a new env knob documented in `ENVIRONMENT.md` if any.

## 5. Pointers

- `archive/work/dflash-meta-split/RESOLUTION.md` (this session's root cause + decision).
- `ggml/src/ggml-backend-meta.cpp` (`handle_per_row` `:707`; the `GGML_OP_TOP_K` case ~`:1210`;
  `GGML_BACKEND_SPLIT_AXIS_PARTIAL` handling and the reduce path).
- `src/models/dflash.cpp:517/538/888`; `src/llama-hparams.h:265-266`; TODO #58.
- `patches/README.md` block 11 (DFlash) and block 09 (Meta).
