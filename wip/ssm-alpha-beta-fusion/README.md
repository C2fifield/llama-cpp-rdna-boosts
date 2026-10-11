# `ssm-alpha-beta-fusion` — single-walk `ssm_alpha` + `ssm_beta` (TODO item 5(b))

**Status: PARKED (seeded 2026-10-11).** Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the parked TODO
item *"`ssm_alpha` + `ssm_beta` single-walk fusion (item 5(b))"*.

> **Cold start.** Read this file, then
> [`archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`](../../archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md)
> (the investigation and the three implementation constraints). Report findings to the maintainer before
> any delivery change and wait.

## 0. The item

On the qwen4exp / hybrid-SSM recurrent layers (~36 of 48) the graph builds two F32 GEMMs on the **same**
`src1` (`blk.N.ssm_beta = mm(ssm_beta_w, cur)`, `blk.N.ssm_alpha = mm(ssm_alpha_w, cur)`). A single walk
would read `cur` once and compute both. This is comparable to the already-promoted dense `mmb`/`mmq`
single-walk wins.

The measured prize is **modest: ~0.3–0.6 % prefill**:

- `pairbench` stacked rocblas `M=96` on `[2560×96]` weights = **0.410 ms** vs the two `M=48` GEMMs
  **0.588 ms** (1.43× — rocblas' M96 config only reaches ~51 GB/s, not 2×).
- The stacked path is **not bit-identical** (~3e-7 max diff; different rocblas config / K order) — a
  documented fp32 near-lossless deviation. The maintainer previously accepted "essentially correct" fp32
  reordering, but it still needs the quality gate.

## 1. Why it is parked: the graph expansion order

The two `MUL_MAT`s are **not adjacent** in the scheduled graph — `alpha-MM@52`, `beta-MM@58`, with alpha's
`ADD/softplus/mul` chain (54–56) between them. The `cgraph` order is a topological DFS of the expanded
outputs, **not** the creation order (`build_layer_attn_linear` creates beta first yet alpha executes first),
so the general scheduler pair-matcher (`nodes i, i+1`) cannot fire. Making them adjacent needs a
`src/models/qwen4exp.cpp` restructure **and** control over the expansion DFS, which is unreliable.

The robust routes the discovery records:

- **(a)** load-time stacked `[2560×96]` weight + **one** graph `MUL_MAT` + per-half views (~0.27–0.4 % wall;
  loader plumbing), or
- **(b)** graph restructure + a custom single-walk kernel (~0.6 %, the real prize; reads the 21 MB once at
  ~150–200 GB/s).

Both are numerics-moving (near-lossless fp32) and need a quality gate + re-baseline before promotion.

## 2. Current reality check (r48)

- The pair is structurally safe: both operands are F32 and **share the same `src1` tensor object** (unlike
  the attn/ffn inject pair, which only *happens* to be byte-equal at runtime and was correctly rejected as
  a correctness trap).
- The recurrence/GDN and the qwen4exp attention paths have changed since 2026-09-06 (r29/r30 GDN BF16,
  r46 GQA cap, r48 block 06). Re-verify the "same `src1`" property and the current `cgraph` adjacency on
  the r48 tree before implementing — the DFS order may have moved.
- `src/models/qwen4exp.cpp` builds the pair; `src/llama-graph.cpp` has the SSM builders. Confirm the
  tensor names/op sites on r48 (the archive cites line numbers from an older tree).

## 3. Gates

- Prefill-logit KLD gate (`scripts/gate-prefill-logits.sh`) — this is a prefill-numerics change.
- `test-logits-width-probe` `W=1..8` one hash (the SSM GEMMs are per-token in decode; a single-walk kernel
  must not change the decode reduction order).
- Adaptive-MTP gate (`-n 3000`) and same-seed greedy coherence.
- `scripts/validate-set.sh` if promoted; a new env knob documented in `ENVIRONMENT.md`.
- Report the measured win before proposing a block — at 0.3–0.6 % this may not clear the bar.

## 4. Pointers

- `TODO.md` → "Parked (not planned now)" → item 5(b).
- [`archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`](../../archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md)
  (the microbench, the rejected inject fusion, the adjacency analysis, and routes (a)/(b)).
- `src/models/qwen4exp.cpp`, `src/llama-graph.cpp` (SSM alpha/beta builders).
- `benchmarks/prefill-logit-methodology.md`; `GREEDY-PURITY.md` §27 (GDN chunked, the adjacent SSM
  prefill kernel).
