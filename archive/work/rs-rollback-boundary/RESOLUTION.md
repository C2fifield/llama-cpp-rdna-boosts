# `rs-rollback-boundary` — RESOLUTION (closed in `v12-097cf33d9-r91`, 2026-10-19)

**Status: CLOSED. Both parts are in the delivery.**  No part of this campaign is open, and no `wip/`
copy remains (this directory is the record).  Promoted in **block 03** (the per-sequence tracker) and
**block 10** (the sequential snapshot tail); the reasoning below is the part that is *not* obvious from
the patches and is the reason this file exists.

## What was open

Issue #131's guard in `llama_memory_recurrent::seq_rm` warns when a partial rollback lands on a batch
wider than `n_rs_batch`, because the whole-batch chunked GDN prefill writes no per-token snapshots.  Two
questions were open:

1. **Why is the condition reached at all** — is the "large prefill as the last prepared batch" ordering
   real, or a bookkeeping artefact?
2. **Is the restored recurrent state actually stale** in that case?

## Answers

### 1. A stale per-sequence tracker entry (fixed, block 03)

`prepare()` folded the per-sequence last-batch shape in from `ubatches.back()` only.  One decode call
routinely carries several ubatches, so a sequence that ran in a **non-final** ubatch kept whatever it was
the last time it happened to *be* the back ubatch — typically **its own prefill chunk**.  The guard then
compared a rejection rollback against a completed prefill (the recorded `rollback=2 vs 4571 tokens` case).

Every ubatch is now folded in, oldest first.  Measured: gfx1201 2-GPU GSQ+MTP `multiseq.sh`
**1 warning -> 0**, text bit-identical to r90 on all four streams; gfx1151 95 rollbacks -> **0/95**,
identical text.  `LLAMA_RS_ROLLBACK_DEBUG=1` (`RSPREP`/`RSROLL`) is the instrument, kept as a diagnostic.

### 2. Yes, a rollback into a chunked prefill batch restores a wrong state (closed, block 10)

`tools/prefill-rollback.cpp` (MODE=ref, build line in README §8) compares the state a rollback *restores*
against the state the same kernel produces when the prefill simply *ends* at that position:

| arm (35B-A3B Q3_K_M, RS=3) | P=2048 (nmse, k=1/2/3) | P=2050 (k=1,2) |
|---|---|---|
| `GGML_CUDA_GDN_TAIL=0` (r90) | 0.18 / 0.58 / 0.60 (maxabs up to 12.5) | exact 0.0 |
| tail on (r91) | 7.6e-4 / 1.1e-3 / 1.9e-3 | **exact 0.0** |

The tail makes the batch's last `D = max(K>16?K:16, n_rs_batch)` tokens run the **sequential** kernel in
the same launch, so slots `0..D-1` exist for every single-sequence batch.  `GGML_CUDA_GDN_TAIL=0` is the
A/B and the bisect knob.

### 3. The bound is enforced by `n_rs_batch`, not observed — the campaign's ngram claim was wrong

The README §8.1 said the exposure is "a future change that makes a verify batch longer than
`max(K,16,n_rs_batch)` — e.g. an ngram-mod draft (n_max up to 64)".  That is **not** true: `common`
derives `cparams.n_rs_batch = common_speculative_n_max() + 1` over **every** enabled speculator type
(draft simple/mtp/adaptive/eagle3/dflash/dspark **and** ngram simple/map-k/map-k4v/mod/cache), and the GDN
op's threshold is floored at it:

| config (measured from the `llama_context` log) | `n_rs_seq` | `n_rs_batch` | threshold |
|---|---:|---:|---:|
| `draft-mtp --spec-draft-n-max 3` | 3 | 4 | 16 |
| `draft-mtp --spec-draft-n-max 20` (clamped 15) | 15 | 16 | 16 |
| `ngram-mod --spec-ngram-mod-n-max 64` | 0 | **65** | 65 |
| ngram-mod 64 + draft-mtp 3 | 3 | **65** | 65 |

So a speculative verify batch is always at or below the threshold, always falls through to the sequential
kernel, and always restores snapshots it wrote.  With ngram-mod alone `n_rs_seq == 0`, i.e. there is no
recurrent snapshot path at all.  `r91` replaced the kernel's "Verified: ..." wording with this structural
statement (and `AGENTS.md` "Critical facts" now carries it so it is not re-derived).

What the tail therefore closes is the **non-speculator** partial rollback — a server cancel /
`--cache-reuse` rewind that lands inside a prefill batch.  Never observed, but not structurally excluded.

## gfx1201 cross-arch evidence (this box, r91 tree)

| gate | result |
|---|---|
| `prefill-rollback`, tail on / off | exact 0.0 at P=2050; 1e-3 vs 0.18-0.60 at P=2048 |
| `test-backend-ops -o GATED_DELTA_NET` | 2/2 backends, 48/48 |
| plain == MTP text | 35B-A3B 64 tok **and** `-n 3000` on all four axes, identical (also identical to r90) |
| `test-recurrent-state-depth` / `-rollback` | 130 == 130 (per-arm set identical) / 0 FAIL |
| 2-GPU `multiseq.sh` | 0 warnings (r90: 1) |
| `llama-bench -p 512,2048 -n 128 -r 3` | pp512 −0.09 %, pp2048 −0.58 %, tg128 +0.02 % |
| prefill-logit, 27B Q8_0 (instrument clean) | tail vs the arm it replaces **0.002278 / 98.870 %** (PASS) |
| prefill-logit, Flash-Next IQ3_XXS 2 GPU | tail 0.021655 / 95.119 % vs **r90 0.022895 / 95.039 %** — the tail is the closer arm; the gap itself is pre-existing |

Two pre-existing findings were recorded (not fixed) in `WORKLOG.md` 261019.010: the prefill-logit
instrument is noise-dominated on 35B-A3B at `--ctx 2048` (also on unpatched r90), and Flash-Next IQ3_XXS
has a large chunked-GDN-vs-sequential gap in r90 that needs its own campaign.

## Designs considered and NOT promoted

* **Trailing-ubatch split** at `split_equal` (`LLAMA_RS_TAIL_TOKENS`): correct, but **−13.7 % pp512 /
  −3.5 % pp2048** — one extra graph per ubatch plus every prefill ubatch shrinking from 512 to `512-D`.
* **Fixed rollback window** `D = 64`: `D+1` state planes = **3.8 GiB** on 35B-A3B (one plane is 60 MiB:
  40 GDN layers × 128×128 × 24 heads × f32).
* **Checkpoint + replay**: replays up to `D` tokens *every decode step* unless the window covers the
  verify width, so it needs the window anyway — more plumbing for no memory win.

The in-op fixed-`D` tail dominates: zero extra memory, no extra graph, ubatch unchanged, prefill-only.

## Re-running it

```bash
# probe (build line: README §8)
NGL=99 RS=3 K=1,2,3 QUIET=1 ./prefill-rollback <35B-A3B gguf> <a >2k-token text file> 2048 512
NGL=99 RS=3 K=1,2,3 QUIET=1 ./prefill-rollback ... 2050 512     # exact 0.0 is the PASS
# the 2-GPU reproducer
archive/work/moe-verify-fusions/tools/multiseq.sh <build> <tag>  # point its LD_LIBRARY_PATH at the build's ROCm
```

`TESTING.md` is the original handover; note that its gate-3 command has a shell trap (the leading
`GGML_CUDA_GDN_CHUNKED=0` also forces the *candidate* arm) — see
`benchmarks/prefill-logit-methodology.md` §4 and §5.
