# Greedy purity — the rulebook

**What this is.**  The delivery's purity contract, one entry per finding: the claim, the rule a change
must satisfy, the numbers that back it, and where the record lives.  Read the entry for the `§` you are
about to touch; read the seven invariants before shipping anything that changes a decode or prefill
kernel.

**Where the rest is.**  The investigation narratives (how each finding was made, which instruments failed,
which false trails were ruled out) are in
[`archive/docs/GREEDY-PURITY-FINDINGS.md`](archive/docs/GREEDY-PURITY-FINDINGS.md) under the same `§`
numbers.  The delivery-side record of each amendment is in `patches/README.md` and `WORKLOG.md`.  Section
numbers are **stable** — dozens of docs cite them — so they are never renumbered; new findings append.

> **Numbering.**  This document uses the **V2.0 12-block** numbering for new text.  Entries written
> before 2026-10-13 use the **pre-V2.0 16-block** numbering ("block 10" = the k-quant/VDR umbrella, now
> **block 07**; "block 13" = fused MoE/`mmvq`, now **block 08**; "block 14" = qwen4exp, now **block 10**;
> "block 15" = the campaign remainder, now **block 11**).  §39/§40 appear twice in the original numbering;
> the later pair is labelled **§39b/§40b** here.

## The invariants (everything below is evidence for these)

1. **Inside the decode/verify band, one arm — chosen from the `n_tokens` band and `n_kv`, never from the
   exact width, `n_kv` alone, or `n_tokens == 1`.**  A W=1 decode and a W=(`--spec-draft-n-max` + 1)
   verify of the same state must take the same kernels (§§11, 13-17, 25, 26, 32).
2. **A batch you can roll back into must be written by the kernel whose snapshots you read** — the
   rollback bound is a purity bound, not a tuning knob, and it is derived from the widest enabled
   speculator, not from the snapshot depth (§27).
3. **Same real-number result, different rounding order.**  The k-quant/fused-MoE numerics changes are
   exactly that; stock is the reference *standard*, not the mathematically special order (§§1-9).
4. **A newly enabled KV type / op argument / fusion is a new kernel family** and needs the full band
   sweep, the text gate, the oracle and the perplexity comparison (§§20-22, 29).
5. **Purity outranks raw non-MTP throughput**: land bit-identical decode/verify even at a few percent of
   the widest verify batch, record the delta, then repay it structurally (§§19, 24).
6. **Use the instrument that can see the defect**: the raw-logit width probe for boundaries, a CPU
   reference or the dense-masked path for a fused tensor op, random text for a causal leak, and an
   *executed-graph* dump when a chain looks dead (§§11, 21, 23).
7. **A prefill-logit shift is a purity failure even when decode is clean.**  Same-seed text and MTP
   sample the decode distribution *from the post-prefill state* and cannot see a uniform prefill shift;
   run the prefill-logit KLD gate (`benchmarks/prefill-logit-methodology.md`) before shipping a prefill
   kernel.  Issue #113: the default-on BF16 chunked GDN passed coherence at mean KLD 0.03.
8. **A shape-scoped launch parameter must be forwarded to every kernel that reproduces that launch.**
   `nwarps` / `rows_per_block` / VDR are width-uniform only if the fused single-token kernels that
   replace a dense weight launch call the same selector (§19 addendum, §39).

## Index

*status*: **doctrine** = read before shipping · **current** = live state · **fix** = closed finding.

| § | claim | status |
|---|---|---|
| **§1** | the k-quant VDR work computes the same real-number result through a different fp32 addition order | doctrine |
| **§2** | the two mechanisms: VDR constants + the RDNA3_5 parameter table | doctrine |
| **§3** | the dot products are exact; only the fp32 accumulation can round, and the cross-thread tree is identical | doctrine |
| **§4** | VDR changes *which* partials feed which leaves (and does not add rounding events) | doctrine |
| **§5** | "bit-identical to stock" is a reproducibility requirement, not a correctness one | doctrine |
| **§6** | speculative verify runs the same kernels on a wider batch, so the variance composes orthogonally | doctrine |
| **§7** | measured magnitude: max logit diff 0.184, PPL identical | current |
| **§8** | practical guidance: how to reproduce stock bits, and how to A/B | doctrine |
| **§9** | the fused-MoE `mmvq` rewrite is a **second** decode-numerics source | doctrine |
| **§10** | block 00 makes the verify width irrelevant on the small-batch FA path | doctrine |
| **§11** | the guarantee is `n_max <= 7`; the causes are one fixed AR dispatch and one deliberate kernel switch, and since r95 the RDNA3_0 band takes the same fusions | current |
| **§12** | the guarantee depends on the KV-cache type | fix + trade |
| **§13** | qwen4exp's hyper-connection band is `1 <= nt <= 8` | fix |
| **§14** | F1: the FA chooser spanned VEC and TILE inside the band — one family now | fix |
| **§15** | F2: upstream's per-type mmvq caps were numeric boundaries inside the band | fix |
| **§16** | cause 3: the QSA dense decode arm was `n_tokens == 1` | fix |
| **§17** | the MoE shared-expert epilogue is token-generic with `nwarps` pinned | fix |
| **§18** | the QSA sparse gfx1151 width-dependences did not reproduce; the residuals were two other findings | fix |
| **§19** | purity first: a correctness fix may cost a few percent — land it, record it, repay it | doctrine |
| **§20** | a newly enabled KV type is a new kernel family | doctrine |
| **§21** | a shared staging tile makes a block head-homogeneous (and width purity cannot see it) | doctrine |
| **§22** | `iq4_nl`: the type is not the only thing an enablement touches | fix |
| **§23** | a shadowed variable left the mask chain dead and unfilled | fix |
| **§24** | the cost of a band amendment can be structural, not arithmetic | doctrine |
| **§25** | the RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical | current |
| **§26** | a regime policy is only pure if the *band* takes one arm — including the prefill half | doctrine |
| **§27** | a rollback bound is a purity bound (and since r91 it is *enforced*, not observed) | doctrine |
| **§28** | an export-only tail must not redefine the logits path | doctrine |
| **§29** | a flattened batch is not a token count | fix |
| **§30** | a support predicate decides the *layout*; the layout decides an address-gated fusion | doctrine |
| **§31** | an address-selected fusion must be bit-identical to the path it replaces | doctrine |
| **§32** | a decode/verify band policy must scale with the configured draft depth | fix |
| **§33** | a gate must run the content the axis intends, long enough to reach steady state | doctrine |
| **§34** | a K/V staging policy is a decode-depth policy (and can be a load margin) | doctrine |
| **§35** | a prefill kernel config must be arch- and split-keyed | doctrine |
| **§36** | §14's `W=1..8` purity is a *measured* claim; relax it for the coarse quants | doctrine |
| **§37** | a fully-masked KV group is an exact no-op; the skip must stay prefill-only | doctrine |
| **§38** | the guarantee is scoped to a fixed batch; `--kv-unified` serving is not deterministic | doctrine |
| **§39** | a decode-only fusion must match the standalone launch it replaces | doctrine |
| **§39b** | the MoE expert-cache band follows the routed MMVQ band | current |
| **§40** | a thread count is a scheduling parameter, not a numerical one | doctrine |
| **§40b** | a reserve/measure graph must match its memory context's stream count | doctrine |
| **§41** | a runtime, address-selected fusion is a cross-process purity risk; a compile-time one is not | fix |
| **§42** | a width gate is not a purity rule: extending a fused chain *down* to one token | fix |

---

## 1. The one-sentence version

The k-quant VDR work (pre-V2.0 "block 10", now **block 07**) does not make decode more or less *correct*
than stock — it computes the **same real-number result** through a **different order of fp32 additions**.
Stock is not the mathematically special order; it is the *reference standard* that shipped first and that
everything else reproduces.

## 2. What block 10 changes

Two mechanisms, both in the `mmvq` decode path, and nothing else in the block changes production numerics
(the MMQ scale-load hoist, the MoE `mmid` whitelist and the perf-harness cases are arithmetic-preserving or
test-only):

| mechanism | change | effect |
|---|---|---|
| **VDR** | `VDR_Q4_K/Q5_K/Q8_0 2→4`, `VDR_Q6_K 1→2` | each thread accumulates 4 (or 2) chunk partials serially instead of 2 (or 1) |
| **RDNA3_5 parameter table** | split from the RDNA2 fallback, `nwarps=2` for Q8_0 on gfx1151 | fewer warps → different cross-warp combine tree |

## 3. The reduction structure

The dot products are **exact** (`ggml_cuda_dp4a` is an integer SIMD dot product; the scale multiplies are
single multiplies); only the **fp32 accumulation** can round.  The per-thread serial chain
(`tmp[j][i] += vec_dot(...)` once per `blocks_per_iter`) and the cross-thread combine (shared-memory across
warps, then `warp_reduce_sum`'s hard-coded `16,8,4,2,1` shuffle butterfly) are **identical** in stock and
in the boosted path.  What changes is *which* partial sums feed which leaves of that tree.

## 4. Why VDR changes the rounding path

`blocks_per_iter = vdr * nwarps*warp_size / qi` and `kqs = vdr * (tid % (qi/vdr))`, so VDR sets the
per-thread serial chain length and the thread-to-chunk assignment.  The *set* of partial products is
unchanged; the **association order** is not.

*A note on "fewer rounding accumulations".*  VDR=4 does **fewer** serial rounding steps per thread, not
more (the outer `tmp +=` chain runs half as often, ~17 % fewer serial fp32 adds).  "The boosted path rounds
more" is the wrong intuition, and neither path is provably closer to the exact result.

## 5. What "correctness" means here

* **Same real-number result** — no approximation is introduced.
* **Stock is not more correct** — its ordering was chosen for performance on the hardware of the day.
* **Bit-identity is a reproducibility requirement, not a correctness one** — golden vectors, cross-build
  harnesses, `verify == decode` batches, audit trails.  Reproducing stock's bits means reproducing its
  exact association order, which is why the block is structured as one excludable patch.

## 6. Speculative decoding composes orthogonally

The verify batch runs the *same* `mmvq` kernels at `ncols_dst = n_draft+1`.  Two invariants keep it
bounded: **VDR is per-type, not per-column-count**, and the `ncols_dst <= MMVQ_MAX_BATCH_SIZE` rule forces
the verify batch onto decode's `nwarps`.  So the acceptance criterion still reproduces the target
distribution exactly.

| | temp = 0 (greedy) | temp > 0 (stochastic) |
|---|---|---|
| effect | a ≤0.184 logit drift can flip near-tie acceptances → a different greedy stream | the drift perturbs probabilities, not decisions; acceptance *rate* may shift slightly |
| correctness | none — both are valid deterministic paths | none — the distribution is preserved; PPL is bit-unchanged |

## 7. Measured magnitude of the variance

| metric (gfx1201, ROCm 7.14) | value |
|---|---|
| max abs logit diff, block 10 in vs out | **0.184** |
| same, flash-attn on vs off (yardstick) | 0.203 |
| per-token probability change at the worst logit (t=1.0) | ~20 % |
| top-2 logit gap typically | 0.5-5 nats (0.184 is ~9 % of a 2-nat gap) |
| PPL (wikitext-2, Q8_0) | identical to pre-block-10 |
| greedy streams | deterministic within a build; 1 of 3 prompts diverged at one character |

**Honest framing:** 0.184 is *not* vanishingly small (~20 % relative probability on the single worst
token), but it is a **max** over all tokens, PPL is bit-identical, and greedy flips need a near-tie.  If you
need stock's bits, exclude the block; if you need a correct, deterministic, high-quality model, the block
is fine.

## 8. Practical guidance

* Within a build, greedy output is fully deterministic — the variance appears only across *builds*.
* To reproduce stock bits, exclude the k-quant block (**V2.0 `0007`**).  In V2.0 that is **no longer a
  one-line drop**: `0007` is mid-series and later blocks edit the same `mmvq.cu` region, so the set must
  be re-cut without it (pre-V2.0 it was the last patch).  Do not apply it partially or hand-tune VDR in
  production.
* A/B on **distributions** (PPL, benchmark scores, sampling quality), not individual greedy tokens.

## 9. The fused-MoE `mmvq` rewrite is a second decode-numerics source

Pre-V2.0 block 13 (now **block 08**) replaces the §3 K-split loop with an item-split over
(token, row, kblock) plus an RDNA `rows_per_block` override, feeding different partial sums into the same
tree.  The 2026-09-02 fix restored `ncols 2..8` and `ncols==1` with `K >= 4096` to the K-split kernel, so
dense long-K decode is back to block-10-only variance; the remaining surface is `ncols==1` with `K < 4096`
(MoE down/router and other short-K projections).  Multi-token MoE decode takes the unfused path (the
`rms_norm`→Q8_1 fold is gated off `MUL_MAT_ID`), and single-token MoE carries fusion-ordering drift.

**Consequence:** reproducing stock bits with this block installed is not a one-line drop; the `ksplit`
dispatch condition in `mul_mat_vec_q_switch_ncols_dst` is the gate.

## 10. Block 00 makes the verify width irrelevant on the small-batch FA path

`launch_fattn`'s `parallel_blocks` keyed off `ntiles_dst` (a function of `Q->ne[1]`), so single-token
decode and a verify batch grouped the online-softmax/PV partials differently and produced different logits
(issue #25).  Block 00 evaluates the heuristic as if `n_q == 1` for every `n_q <= 8`, so decode and every
verify width reduce identically.  `n_q = 1` can still differ from a verify batch because the FA *kernel* is
selected from `Q->ne[1]` — a decode-vs-verify difference, not a draft-length one.

## 11. The real purity range: `n_max <= 7`

The boundary is on the **verify batch width** (`K = n_max + 1`), not on `n_max`.  Two causes:

* **Cause A (fork-specific, FIXED):** the size-based all-reduce dispatch switched to NCCL above a
  per-device-count element count, and the two paths are not bit-identical (the internal pipeline
  BF16-round-trips).  Under `-sm tensor` the reduced tensor scales with the batch width, so a 7-token
  verify crossed the old 2-device limit.  The crossover is now 131072, and MTP throughput *rose*
  (+12 % at `n_max 6` and 12) because the internal pipeline is faster at these sizes.
* **Cause B (deliberate):** the FA tile→WMMA switch at `Q->ne[1] > 8`, and (with FA off) the
  `mmvq`→`MMQ` switch at `ncols == 8`.  Beyond the band the fast kernels have a different reduction
  order; keeping the guarantee would mean giving up WMMA for 9..N-token batches.  There is **no
  corruption** in 1..15 (deterministic recurrent-state sweep).

**Rules.**  Use the **raw-logit probe, not the text gate**, to establish a boundary (text equality is
evidence *for* purity, never *against* divergence — near-ties are rare).  Above the range, gate on
acceptance + MTP-vs-plain throughput; adaptive MTP's `n_max = 12` is deliberately outside.  Upstream is
worse: on `9cf3bf256` (CPU) it diverges at the first width step.

**RDNA3_0 joins the band (r95).**  The boundary above is about the *kernels* a width selects, not about
an architecture, so RDNA3_0 (gfx1100-1102) takes the same verify-band fusions RDNA4 has had since r78:
the 2..8-token Q8_1 pre-quantize fusions (through the one `ggml_cuda_should_fuse_mul_mat_vec_q(...,
verify_band)` predicate all three share), the direct GDN conv for 2..255-token batches, and the batched
f32 `GGML_OP_CPY` run.  It is a **purity** claim, not only a performance one, and it is measured at the
widths:

| gfx1100, `KV=q8_0`, default vs all three fusions off | row 0, `W=1..8` | verdict |
|---|---|---|
| 27B UD-Q4_K_M, P=256 | `c37241bd00c1d561` (both arms) | PASS, worst maxdiff 0 |
| 27B UD-Q4_K_M, P=3584 | `86541410b7c618b0` (both arms) | PASS, worst maxdiff 0 |
| 30B-A3B Q4_K_M, P=256 | `899608bf91600e1b` (both arms) | PASS, worst maxdiff 0 |
| 30B-A3B Q4_K_M, P=3584 | `63ee46bb518130a7` (both arms) | PASS, worst maxdiff 0 |
| 35B-A3B Q4_K_M, P=256 | `423a5f353f0a0e45` (both arms) | PASS, worst maxdiff 0 |
| 35B-A3B Q4_K_M, P=3584 | `6cdc0c5a6601264d` (both arms) | PASS, worst maxdiff 0 |

Same-binary kill-switch identity: default, `GGML_CUDA_FUSE_Q8_1_VERIFY=0`,
`GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, `GGML_CUDA_FUSE_CPY_BATCH=0` and all three off give the same per-`W`
hashes in every cell above; greedy text at 400 tokens is identical for `none` and `draft-mtp n3` (27B
prose `0f57c09e9ce2`, 35B-A3B prose `4111be939ff4`, 27B code `cfcc349aaa5a`); prefill-logit is mean KLD
`-0.000000` / same-top-p `100.000 %`; and acceptance does not move (R axis `0.52065`, acc per pos
(0.722, 0.500, 0.340)).  RDNA3_5 stays excluded (`0x1150` is outside `GGML_CUDA_CC_IS_RDNA3_0`) and was
verified inert on gfx1151 (same `W=1..8` hashes as r94 and as all-off, no `cpy_batch` run, no
`GDN_CONV_DIRECT` for the verify band).  **gfx1101-gfx1102 are inside the predicate but are not
explicitly validated**, and RDNA3_0 multi-device is an accepted open gap (`TODO.md` #79,
`archive/work/rdna3-verify-fusions/RESOLUTION.md`).

Record: `archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3; narrative
`archive/docs/GREEDY-PURITY-FINDINGS.md` §11.

## 12. The guarantee depends on the KV-cache type

| K/V (same type) | status |
|---|---|
| f16, bf16 | pure — the reference configuration |
| q8_0, q4_0 | **were impure** (`W=1,2` != `W=3..8`) — fixed by §14 |
| q4_1, q5_0, q5_1, iq4_nl | pure, but through the generic F16-staging fallback (~3.4x slower at the time) |

**Rule:** the impure set was exactly the two types with a *fast native* both-quantized FA path, so this is
a **trade, not an oversight** — any future native path must be width-invariant by construction.

## 13. qwen4exp's hyper-connection band is `1 <= nt <= 8`

`HC_MIX` / `HC_COMBINE` were gated to a single-token batch, so decode and verify took different arithmetic
(98 dispatches at `W=1`, **0** at `W>=2`).  The whole decode/verify band (`HC_FUSED_MAX_TOKENS`) now runs
the fused ops with the token index on `blockIdx.y`.  **Caveat:** a `<= 8`-token *prefill* chunk takes the
fused path too — indistinguishable from a verify batch, which is the point.

## 14. F1: the FA chooser spanned two kernel families inside the band

§12's `q8_0`/`q4_0` impurity was `ggml_cuda_get_best_fattn_kernel()` returning **VEC** for `n_q <= 2` with a
quantized K/V and **TILE** from `n_q = 3`.  Both VEC conditions sit inside the band, so the branch is
deleted and the band is TILE throughout.  Measured (3x gfx1201): 4B q8_0 `W=1..8` all `31a0c1bace68` (1 GPU),
`abebfb93` (2-GPU tensor), `7fe106f5` (3-GPU); 27B q8_0 KV `plain == n_max 3 == n_max 7` = `3537bc2b36be`.
Every new value equals that config's **previous verify** value, so only `W = 1,2` moved: MTP is
bit-unchanged and the cost is decode-only (-0.9 % 4B / -0.5 % 27B tg128).

**F3 reframing:** §12's "no native FA kernel" explanation is wrong in detail — `q4_1`/`q5_0`/`q5_1`/
`iq4_nl` are rejected by `ggml_cuda_fattn_kv_type_supported()` unless `GGML_CUDA_FA_ALL_QUANTS`, so
attention takes the generic fallback: width-invariant by construction, hence pure.

**Harness lesson:** apply backspaces, strip the banner and the `[ Prompt: … ]` footer before hashing
`llama-cli` output, and always run a control that must agree.

## 15. F2 cause 2: upstream's per-type mmvq caps were numeric boundaries inside the band

`get_mmvq_mmid_max_batch_*` sizes `mul_mat_vec_q_moe`'s launch bound *and* chooses mmvq vs MMQ, and the two
reduce in different orders — so every cap boundary inside the band is a numeric boundary.  (It is a
graph-identical bug: the same node counts at `W=4` and `W=5`; the mixed-quant UD file is what exposes it.)
**Fix:** floor the cap at `MMVQ_MAX_BATCH_SIZE` for every AMD lookup and size the MoE kernel's launch bound
at the band.  Every call site is `MUL_MAT_ID`-only, so dense models cannot be affected.  Result: `W=1..8`
one hash per split (each equal to that split's pre-fix `W=1` value), `n_max 7` **+16-18 %**, and
`llama-batched-bench` `pl=5..8` **+14-26 %** — upstream's caps were costing exactly the speculative-verify
widths on RDNA4.

## 16. Cause 3: the QSA dense decode arm was `n_tokens == 1`

The indexer has three arms (dense+store below the selection width, the dense arch policy, the sparse top-k
selection) and arm 2 was gated on `n_tokens == 1`.  From the first decode graph (`n_kv = 2304 > 2051`)
`--spec-type none` took the dense arm and `draft-mtp` took the sparse one — the same state, two attention
regimes, purely from the batch width.  **Fix:** `QSA_DECODE_BAND = 8` (arm 2 takes `n_tokens <= band`);
prefill keeps the sparse arm.  `plain == n_max 3 == n_max 7` = `804de0576868`.  The band became
depth-derived in §32.

## 17. The MoE shared-expert epilogue is band-uniform too

The fused down epilogue was gated decode-only, with an in-code note that the fused gate reduction does not
reproduce the standalone `mmvq` order.  **Fix:** the kernels are token-generic (one warp per token; one
block per `(output row, token)`) and the whole band takes the fused path with **`nwarps` pinned to the
single-token value** — `calc_nwarps` returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets
`blocks_per_iter`, i.e. the reduction order.  `W = 1,2,3,4,8` all `ac8825358d9adfda`;
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` gives the uniform unfused reference.  The band's `pl=8` cost was
repaid structurally in §24.

## 18. The QSA sparse gfx1151 width-dependences did not reproduce

Re-measured with the current delivery, neither recorded width-dependence holds: the fused indexer score is
byte-identical to the per-op chain, and the residual split was **§25's** RDNA3_5 single-token mmvq fusion
(reproduced exactly with `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`).  Default gfx1151 configs are
pure (dense decode on every KV type; deep sparse decode at ~74K on f16 and q8_0).  The remaining forced-arm
q8_0 shallow residual was root-caused to (a) the `embeddings_nextn` export deferral (§28) and (b) the
flattened indexer score crossing `MMVF_MAX_BATCH_SIZE` (§29) — both fixed, gfx1151 re-validated
2026-09-12 (14).

## 19. Purity first: the measured trade

**A purity fix may cost a few percent of raw non-MTP throughput.  That is not a gate failure.**  The
asymmetry is structural: the draft's proposals *are* decode steps, so any arithmetic difference between
decode and verify shows up as draft-vs-verify disagreement, and lost acceptance costs a multiple of
whatever the "faster" path saved.

* the shared-expert band fix (§17) cost ~2.4 % at `pl=8` and **raised MoE acceptance 0.51 → 0.817**
  (167.3 t/s vs plain 96.9, **+73 %**);
* the QSA decode-arm fix (§16) cost ~1.5-2 % at `pl=5/6` and gave **+28 %** (63.9 vs 50.1 t/s).

**Gate for a purity fix:**

1. **correctness** — `plain == draft-mtp` across the whole band (`n_max <= 7`) on every supported KV type,
   or a documented reason why the regime is unreachable by default;
2. **MTP** — pos-1 acceptance >= ~0.45 and `draft-mtp` >= `plain` at the default depth (`n_max 3`); a fixed
   high depth may lose (that is over-drafting, not a regression);
3. **perf** — no hard requirement at the wide verify widths: record the delta, file the optimisation
   follow-up, land the purity fix first, and prefer the side whose arithmetic is the **decode** side.

**Corollary:** a "win" measured before a purity fix may have been measured with a width-dependent path.

**Addendum (2026-10-20, r95) — admitting an architecture to a band is a purity claim, so it is gated at
the widths.**  Extending the verify-band fusions to RDNA3_0 (PR #192 + PR #193) is not a perf-tuning
change: it decides *which arithmetic the RDNA3_0 verify batch runs*, so the deliverable evidence is the
`W = 1..8` probe, not a throughput number.  The design that made it safe is that every fusion was
already **bit-identical to the path it replaces** — the Q8_1 pre-quantize kernels write the values
`quantize_q8_1` writes, the GDN conv keeps its 4-tap order and `silu` with the snapshot tail still
materialised, and the batched CPY copies in `ggml_cpy`'s flat order — so no kernel had to change, only
the arch predicate.  Gate any future arch admission the same way: one predicate, the probe on every
model class the band reaches, `none == draft-mtp`, and the same-binary kill-switch identity.  Numbers
and hashes: §11; record `archive/work/rdna3-verify-fusions/RESOLUTION.md`.

**Addendum (2026-10-20, r93) — a shape-scoped launch parameter must be forwarded to every kernel that
reproduces that launch.**  The fused single-token kernels (`ssm_gate_beta`, `ssm_conv_l2_gatebeta`,
`shexp_down_gate`, the `rms_norm`→Q8_1 fold, the gate/bias folds) each *replace* a dense mmvq weight
launch, so they must call the same selector (`calc_nwarps_weight(type, ncols, table, long_k, <row scope>)`).
A scope applied to the plain launch but not the fused one runs `W=1` at one warp count and `W=2..8` at
another (measured: 35B-A3B 0.150, gemma-26B **3.35** on the width probe).  The scope was the `!has_fusion`
exclusion in the `archive/work/nwarps/` candidate — not a missing reduction-mapping derivation.  The host
must compute the value once and pass it to both the launch dims and the kernel template (§20).

## 20. A newly enabled KV-cache type is a new kernel family

Each newly reachable (K,V) diagonal gets its own `W = 1..8` sweep on every split, its own
`plain == draft-mtp` text gate, its own MTP acceptance reading, and (per §21) the oracle and perplexity
comparison.

* **On gfx1201 the enabled types are band-uniform by construction** — a quantized cache always takes TILE
  with the launcher's F16 staging (that is F1, §14).  The new diagonal instances exist for the other
  backends and for predicate consistency.
* **A type the model's *other* attention ops cannot read is a correctness trap.**  qwen4exp + a quantized
  cache + `-sm tensor` aborted in the meta splitter because the graph kept building the fused sparse QSA
  op for a type its kernel cannot read, so that op was never split.  Block 14 now takes the dense masked
  path when the cache type is not QSA-native.  When a type is enabled for one op, check **every** op that
  consumes the cache, in every split mode.

## 21. A shared staging tile makes a block head-homogeneous

The QSA kernel gives one block up to `QSA_MAX_HEADS = 16` query heads (one warp each) and stages **one**
K/V tile for all of them, with each thread adding *its own head's* K/V offset.  That is only valid while
every q-head in the block maps to the same K/V head.  qwen4exp is 24 q-heads / 2 kv-heads = **gqa 12 < 16**,
so heads 12..15 wrote the *second* K/V head's rows into the same smem slots.  The chunking is now
`min(QSA_MAX_HEADS, gqa_ratio)`.

**The general rule:** whenever several heads/sequences/rows share one staged buffer, the block's work
partition must be homogeneous in every index that the *staging* reads but the *compute* does not re-apply.

**Why nothing caught it:**

1. **Width purity cannot see it** — the bug is perfectly width-uniform (every width takes the same wrong
   rows).  An internal consistency gate can prove widths agree with *each other*, never that they agree
   with the definition of the op.
2. **The probe was blind to the op** — `GGML_OP_FLASH_ATTN_QSA` is only built above the indexer selection
   width (`indexer_top_k + r - 1` = 2051) and the probe's `n_ctx` is 2048.  Run the QSA gate with
   `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`.
3. **MTP acceptance points the wrong way** — draft and target run the same wrong attention, so the
   corrupted pair is self-consistent and accepts *more* (0.65 vs 0.49).  Acceptance is a quality signal
   only when both sides are known-good.
4. **What does catch it:** the **CPU reference** (`test_flash_attn_qsa`, 18 cases across KV types, cache
   widths, head sizes and `n_tps` — old kernel NMSE ~1.0, fixed < 5e-4) and the **dense masked path as an
   oracle** (PPL 7.3269 old vs 6.5306 dense; fixed 6.5267).  Keep the oracle in the test suite.

## 22. `iq4_nl`: the type is not the only thing an enablement touches

`iq4_nl` is a first-class FA cache type; the full checklist (§20) ran: pure everywhere, text
`acd18ad2d55c` (tensor) / `a38a6e2d8efa` (layer), MTP pos-1 0.757, `FLASH_ATTN_QSA` 22/22.

Three lessons that generalise:

* **A type rejected by the FA probe hides *unreachable code paths*.**  The non-contiguous staging
  converter never listed `iq4_nl`, and `launch_fattn` calls it unconditionally when K/V is a *view* — so
  the enablement turned a dormant `nullptr` into a null-pointer call on the first token (found by
  `test-backend-ops -o FLASH_ATTN_EXT`, whose 336 `iq4_nl` cases had only ever been skipped).  When a
  predicate starts accepting a type, assume its coverage in every consumer is stale.
* **"The type is 3.4x slower" is a statement about the pre-state, not the type** — that number was the
  no-FA path; with FA it is within 2 % of f16 on dense models.  Record the pre-state *with the regime*.
* **A type-specific cost can live outside the type's kernels** — the qwen4exp `iq4_nl` prefill delta is
  host/launch-side (`rocprofv3 --kernel-trace` + a full-graph dump; identical kernel sums and an identical
  1010-node graph).  Prove it with a profile before optimising.

## 23. A silently dead chain: the shadowed variable

An *outer* `ggml_tensor * kq_mask_top_k = nullptr;` was shadowed by a new local inside the mask chain, so
the dense masked arm attended with **no mask at all** (random-text PPL `1.0205` broken vs `19.0589`
healthy; qwen4exp `1.0558` vs `6.49-6.55`).

1. **A graph tensor with no consumer is silently dropped** — the chain was unreachable from the output, so
   it was never emitted, the packed mask was never allocated, and its `if (self_kq_mask->buffer)` fill guard
   skipped filling it.  "Created in the graph" proves nothing about being *filled*.
2. **When an executed-graph dump is *missing* nodes, suspect the graph builder, not the allocator.**
3. **`-Wshadow` would have caught this class outright** (see `archive/work/shadow-warnings/`).
4. **The leak instrument to reach for first is random text** — a model that can see the target predicts
   even noise; repetitive text scores ≈1 in a healthy build too.

## 24. The cost of a band amendment can be structural, not arithmetic

The §17 amendment's `pl=8` −2.4 % was the *launch shape* (`grid = (nrows, ncols)`, one block per
(output row, token): the down-weight row re-read per token and both barriers, the cross-warp reduction and
the epilogue duplicated per token), not the invariant.  A `ncols_dst`-templated kernel with the token loop
inside the k-block loop made it **+3.1 %** with zero numerical change.

1. **Band-uniformity constrains the arithmetic, not the launch shape.**  When a band fix costs a few
   percent, ask whether the cost is the *invariant* or the *shape* it was implemented with.
2. **The strongest control for "this changes nothing" is two binaries, not one hash** — keep both
   `libggml-hip.so` builds of the tip and swap them (`tools/sobench.sh`), so the two builds agree *with
   each other* at every gate.
3. **A no-op at the gate config can still be a large wall-clock win** — measure both.
4. **Check the premise of your own control** — the assumed "plain" reference model did take the RDNA4
   routed-compact MMQ dispatch (`rocprofv3`: 480 launches per `pp512`/`ub512`), so the usable control was
   the dispatch's *reach*, not the model choice.

## 25. The RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical

Two fusions route `W=1` through kernels that do not reproduce the standalone arithmetic: the **dense
gate+up+GLU** fusion (`mmvq.cu` restricts fusion to `ncols_dst == 1`) and the **MoE weighted-down tail**
(RDNA3_5-only, single-token by shape fingerprint).  Measured (qwen4exp UD-IQ4_XS, f16, gfx1151):

| config | `W=1` | `W=8` |
|---|---|---|
| default | `8abc6206d1e80709` | `453eaa618738273d` |
| `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1` | `8d036a7b8c8a5ce5` | `453eaa618738273d` |
| dense-GLU fusion disabled | `969940599c5426e9` | `453eaa618738273d` |
| **both disabled** | **`453eaa618738273d`** | **`453eaa618738273d`** |

**Fix:** skip both on RDNA3_5 unless `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (A/B).  Post-fix
`W = 1,2,4,8` is one hash per config (qwen4exp f16 `453eaa61`, q8_0 `113696b9`, MoE `18999a78`); the
gfx1201 path is untouched.  **Cost** ≈ −0.9 % `tg128` on qwen4exp, prefill flat (§19's trade).  The
follow-up that would restore it: make the fused `ncols_dst == 1` kernel reproduce the standalone reduction
instead of skipping the fusion.

**Scope:** this is *within* the 8-wide band — it broke draft-vs-verify purity at `n_max <= 7`.

## 26. A regime policy is only pure if the band takes one arm — including the prefill half

**An arm may only be selected from state that is identical for every graph computing the same sequence
position: the `n_tokens` *band* (`<= QSA_DECODE_BAND` vs above) and `n_kv`.**  The prefill arm is defined
as `n_tokens > band && n_kv < threshold`, disjoint from the decode arm by construction.  An arm gated on
`n_tokens == 1` is what broke it in §§16/32.

**The prefill arm ships disabled, and that is its purity guarantee:** `LLAMA_QSA_DENSE_PREFILL_UNTIL`
defaults to `0` = QSA prefill always (the documented arch policy), so the delivered default selects one
arm at every depth and no reference hash moves.  Verified byte-identical to the pre-amendment build
(f16 `0fc4910d5824`, q8_0 `e8f8bba3942b` on gfx1151).

Corollaries if it is ever enabled: a regime knob is a quality knob and the default is the arbiter; a
self-limiting arm cannot lose at depth (long prefill keeps the faster sparse chunks).

## 27. A rollback bound is a purity bound

Only the **sequential** GDN kernel writes the `K = n_rs_seq + 1` snapshot planes; the whole-batch chunked
kernel writes slot 0 only, so it is safe for a batch only if that batch is never rolled back into.  The
original constant ("a batch longer than `max(K, 16)` cannot be a verify batch") was false for ngram-style
long drafts, so the bound now comes from the widest speculator draft:
`n_rs_batch = common_speculative_n_max() + 1`, consumed as
`GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch)`.  A whole-batch rollback also needs a
**pre-batch plane** (`0 < n_tokens < K`), which the graph copies into slot `n_tokens`.

**Current state (r91, 2026-10-19).**  The bound is **enforced, not observed**: `common` sets
`cparams.n_rs_batch = common_speculative_n_max() + 1` over **every** speculator type (draft
simple/mtp/adaptive/eagle3/dflash/dspark *and* ngram simple/map/mod/cache, so ngram-mod's default n_max 64
gives 65), `llama_context` floors it at `n_rs_seq + 1`, and the GDN op's threshold is floored at it — so a
speculative verify batch can never take the chunked path.  The campaign's "ngram is an exposure" claim was
wrong.  The remaining source of a partial rollback is a **non-speculator** rewind into a prefill batch
(server cancel / `--cache-reuse`), which block 10's fixed-`D` sequential tail covers: the last
`D = max(K>16?K:16, n_rs_batch)` tokens of a single-sequence prefill batch run the sequential kernel in the
same launch (`GGML_CUDA_GDN_TAIL=0` is the A/B; probe NMSE 0.65 → 1e-3).  The **issue-#131 warning** was a
stale per-sequence tracker entry (block 03 now folds every ubatch in).  Record:
`archive/work/rs-rollback-boundary/RESOLUTION.md`.

**Rules:** every batch that can be rolled back into must run the kernel that writes the snapshots it will
read; derive the bound from the speculator, not from the snapshot depth; keep the guard as a real
invariant check.

## 28. An export-only tail must not redefine the logits path

An extra graph output (the MTP draft's `t_h_nextn`) may add work but must not move the arithmetic of the
*logits* path.  The trap: an unmasked export needs a hidden row for **every** token, so the last layer's
output-row gather was deferred until after the export was taken — and the last-layer tail then ran on the
full prefill ubatch instead of the gathered rows, shifting the prefill's last-position logits by a ULP
(`ad3acaa7…` vs `b624a79f…`).

**Fix (block 10):** always gather the output rows before the tail (exactly what `--spec-type none` does)
and build a **second, full-row tail** whose only consumer is `t_h_nextn`, only when the export is on and
the chunk drops rows.  It must be expanded explicitly (`ggml_build_forward_expand`) — `ggml_set_output()`
alone does not add a tensor to the graph.

**Generalise:** when a graph gains an output read by a different consumer, check the batch width of every
node the logits traverse, and prefer "compute the logits path exactly as the plain arm does, then compute
the export separately".

## 29. A flattened batch is not a token count

The "keep a verify batch on the MMVF family" guard keys on the matmul's `ne11`, which is the token count
only while N is the token count.  The QSA indexer score carries the **indexer heads** in N, so its `ne11`
is `4 * n_tps`: from `n_tps = 3` decode stayed on MMVF (`ne11 = 4`) while the verify batch fell to MMF
(`ne11 = 12`), and the families accumulate differently.

**Signature:** bit-identical to decode for many steps, then one divergence at a fixed position (a
selection flip, not drift); the greedy text may stay equal, so only a per-step logits replay (`mstep`)
sees it.

**Fix (block 10):** `MMVF_MAX_BATCH_SIZE_FLAT` (`= 32`) with `mul_mat_vec_f` instantiated for
`ncols_dst` 9..32.  **Rule:** the band guard's "batch size" must be the **maximum flattened N** the op can
produce in the band; audit any matmul whose N is a per-token × per-head/expert/group product.

## 30. A support predicate decides the *layout*, and the layout decides an address-gated fusion

A CUDA `GET_ROWS` predicate required `ne[0] % QK_K == 0`, so a 128-wide `iq4_nl` indexer-key gather was
rejected and that node ran on the CPU (13 host round trips per prefill graph, GPU busy/span 0.62 vs 0.96).
Widening it re-addressed the whole compute buffer, and the MoE-router `topk_moe` fusion is selected by
**buffer-address overlap** — so the model output moved (`c0d44c479ee1` → `14a1a3f257f4`) even though the
moved op itself is bit-exact.

**Rules:** a support predicate is a correctness surface for *layout*, not only for reachability — when you
widen one, re-run the full gate set, the observable can move in a *different* op.  And an
address-overlap-driven fusion whose arithmetic is not bit-identical makes the output allocation-dependent
(pre-existing upstream behaviour; the delivery's own invariants are unaffected because coverage is
width-uniform within a config).

**Instrument:** `GGML_SCHED_DEBUG=1` prints the per-graph split count, `=2` the per-node backend — cheaper
than a profiler for "which op left the GPU".  Both need **`-lv 5`** (`-lv 4` prints nothing).

## 31. An address-selected fusion must be bit-identical to the path it replaces

The fused MoE router (`ggml_cuda_op_topk_moe`) is selected by buffer-address overlap but was **not**
bit-identical to the generic `soft_max → argsort → get_rows → norm` chain, so the output was a function of
the allocation plan.  Three fixes make it transparent:

* **softmax reduction order** — the generic path's per-"virtual warp" butterfly then cross-warp phase, not
  one flat butterfly over `l + i*32` (36 % of random 512-value rows differed by up to 2.4e-7 relative);
* **normalization** — accumulate in `reduce_rows_f32` order and **divide**, not multiply by `1/sum`;
* **argsort tie-break** — the generic CUDA bitonic sort is non-stable; the fused argmax broke ties by
  smaller index and CUB's sort is stable.  The bitonic comparator now breaks ties by index (ties are real:
  4 occurred in one qwen4exp run).

**Rules:** a fusion selected by memory addresses must be numerically transparent; a non-stable sort is a
latent fusion hazard; verify with the kill-switch (`fused == GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` for every
native KV type on both split modes), not just `plain == draft-mtp` (both paths are internally width-uniform).

## 32. A decode/verify band policy must scale with the configured draft depth

`QSA_DECODE_BAND = 8` was a fixed constant while the verify width is `n_max + 1`.  Once the clamp moved
from 7 to 15, a depth-8..15 verify became a 9..16-token batch: above the indexer selection width
(`indexer_top_k + r - 1` = 2051) the W=1 decode took the dense arm and the verify fell to sparse — the
"diverges after ~2051 tokens" qwen4exp depth-15 bug.

| build | `W = 1..8` | `W = 9..16` |
|---|---|---|
| pre-fix | `643a8166d8dad677` | `1354757f9daf03db` (= the forced-sparse hash) |
| post-fix | `643a8166d8dad677` | `05be2f7f30dbc426` (the dense arm) |

**Fix:** `qsa_decode_band = max(QSA_DECODE_BAND, cparams.n_rs_batch)`; default configs leave the band at 8
and are byte-identical to the pre-amendment build.

**Rules:** derive a band constant from the quantity it is matching; a bug that only fires past a regime
threshold needs the threshold crossed in the test (the 1..8 probe was green because below 2051 the shortcut
arm covers every width); a depth-policy change re-runs the width matrix for every model family it touches.

## 33. A gate must run the content the axis intends, long enough to reach steady state

The four-axis adaptive-MTP gate was first run with the model's default reasoning mode at `-n 256`, which
invalidated it in both directions: the "code" axis never reached code (it measured the thinking trace,
acceptance 0.56 instead of 0.89-0.92), and `-n 256` measures warm-up, not the mode (code axis −5 % at 256
vs **+28 %** at 3000 — the ranking inverted).  Purity above `n_max 7` is a probabilistic near-tie property,
so it needs enough tokens to hit one.

**Rules:** run the gate at a length matching the workload (`-n 3000`, `-n 2000` floor); pin the content
mode (`--reasoning off` for P/C/K, `on` for R); a spot test that passes is not evidence a long run passes.

## 34. A K/V staging policy is a decode-depth policy

The V4/V5 native-staging arms were documented as a *memory* feature with decode "unaffected".  For a
quantized K/V the F16 staging pass is a **whole-cache conversion that runs every decode step and scales
with `n_kv`** — so the staging source is a decode-depth policy and can be the load margin: 27B q8_0
`tg64` d65536 18.92 → **23.29 t/s** for ~1.2 % prefill, and the same ~744 MiB scratch was the **260 MiB**
the draft context was short at `-c 196608`.

**Rules:** a native staging arm must reproduce the launcher's conversion bit-for-bit (same-seed text, then
the width matrix, then acceptance); the policy is per type class (native is the default for sub-F16 quants,
bf16's MMA-scratch arm stays opt-in; `GGML_CUDA_FA_KV_NATIVE` is three-state: unset = auto, 1 = force on,
0 = force staging); a memory win can be a load-time win — measure the *total* budget, not the failing
allocation.

## 35. A prefill kernel config must be arch- and split-keyed

Two globals were wrong: the `ncols=64` row was a gfx1151 "halo row" applied on gfx1201 too (delivery
prefill slope ~51 % steeper than stock, crossing at ~64K), and `ncols2` is a compute-vs-bandwidth trade —
a whole card is compute-bound and prefers a divisor of the GQA ratio, a tensor split is per-GPU
bandwidth-bound and prefers the wider value.

**Rules:** key the config on the arch with host/device agreement; treat the split regime as a first-class
axis and pass it explicitly (`ggml_set_fa_tensor_parallel`, set from `split_mode() == TENSOR &&
n_cuda_dev > 1` — `llama_model::n_devices()` is **1** under tensor split, so count CUDA sub-devices);
`op_params` is not a reliable channel (the graph copy loses it); `-sm tensor` hides single-card
regressions, so screen with a `t = a + b*n` slope fit **and** measure one card.

## 36. §14's `W=1..8` purity is a measured claim — relax it for the coarse quants

The kernel-family part holds (on AMD the whole band is TILE and the KV split is width-invariant).  What
remains is a **logits-level band edge** between `n_q = 1` and `n_q >= 2` that shows up only as a
rounding-boundary lottery (measured edges: bf16 at P=200, q4_0 at P=224/256, q4_1 at P=200 on the 4B).
At the measured edge the argmax is unchanged and the delta is 0.014-0.064 against a top-2 margin of 2.2-2.7,
i.e. one to two orders below the error the coarse cache itself imposes.  Pre-existing and independent of
the native arms; the launcher dump shows the reduction structure is identical at every width, so the
cause is in the FA path's *values* (leading hypothesis: the mask-derived KV support range), not proven.

**Decision (maintainer, 2026-09-15): won't fix.**  The reward is unmeasurable (greedy token never moved,
acceptance bit-identical), and the fix adds work to the latency-critical single-token width.  The trigger
to revisit is an **argmax change** (a quality event); a hash change with a 2.2+ margin is not one.

**Doctrine, tiered by coherence:**

* **f16 / bf16 / q8_0** — the guaranteed contract is a **text/acceptance** one, and it is *tested*: the
  greedy text is bit-identical across the band (`plain == draft-mtp` for `n_max <= 7`, acceptance
  unchanged).  Logits-level purity held at every (prompt, P) tested **except bf16 at P=200 on the 4B**.
* **q4_0 / q4_1 / q5_0 / q5_1 / iq4_nl** — a good-faith effort: values bit-identical to the reference
  conversion and a uniform kernel family, but **no logits-level promise**; a near-tie may flip, argmax
  stays.  The cache is already the dominant long-context loss at these quantizations, and every retrofit
  measured cost 0.5-9 % on the axis it touched.  Use f16/bf16/q8_0 when you need long-context coherence.

## 37. A fully-masked KV group is an exact no-op

A KV group whose kq-mask entries are all `-INF` contributes exact zeros and leaves `KQ_max`/`KQ_rowsum`/
`VKQ` unchanged, so skipping it is **bit-identical**.  `GGML_CUDA_FA_MASK_SKIP` (default on) is gated on
`Q->ne[1] > 8` for the packed mask (and the derived `cell_pos` path is already prefill-only), so the
decode/verify band's round-robin KV split and the width-purity invariants are untouched.

**Why it is needed:** a `--kv-unified` cache puts the other slots' fully-masked interior cells inside the
attention range, and `flash_attn_mask_to_KV_max` only trims the tail (and is not even launched at `-ub
512`).  **The gate:** the classification must be per (Q stream, query tile, 256-cell group) on the packed
path (continuous batching mixes sequences in a ubatch) and batch-wide on the derived path.  `FLASH_ATTN_EXT`
**6354/6354**, `-kvu` B=4 prefill +22 % (MMA) / +72 % (tile), acceptance and text byte-identical.

## 38. The guarantee is scoped to a fixed batch; `--kv-unified` serving is not deterministic

The bit-identity invariants hold for a **fixed** batch.  They do not promise that a *concurrent server* is
run-to-run deterministic, and it is not — the batch composition changes the kernel and reduction order.
It reproduces on **one GPU** with no all-reduce, so it is not the hybrid AR or `-sm tensor`; sequential
requests on the same server are byte-identical.

**The exact rule (closed 2026-10-19):** it is **unified KV** specifically.  A sequence's result is a function
of its KV cells' tile indices **modulo the per-launch KV-split period** (`pb × nbatch_fa`; `pb` is
`parallel_blocks` for the tile kernel, derived from the *shared* view length, and the fixed band `P`
otherwise).  A shift by exactly one period is bit-identical; any smaller shift can flip a near-tie.
Under `--kv-unified` every new **decode** token takes the shared head, so the residue follows the
co-resident session's allocation history.  Measured on r91 (gfx1151): period 960 cells on the 4B, 1920 on
35B-A3B, both verified bit-identical under a one-period shift.  It reproduces on unmodified upstream, so
it is upstream-inherent.  The cheap fix (chunk sessions by band width / align prompt starts) is
**refuted** — the quantum is the period, not the tile.  A sufficient fix (period-granular per-sequence cell
ownership, ~30-38 MiB per live session on the tested models) is an upstream allocator redesign with no
throughput win and is deliberately not implemented.

**Consequence:** **use `--no-kv-unified` when strict reproducibility is required**; never use a
concurrent-server text hash as a purity gate or a bisection oracle.  The reporter's separate HIP-graphs
start-to-start flip (surviving `--no-kv-unified`) is **not** this root, was not reproduced on Linux, and is
tracked as TODO #67.  Record:
`archive/work/kv-cell-offset-nondeterminism/RESOLUTION.md`.

## 39. A decode-only fusion must match the standalone launch it replaces

A fusion that fires only in one width of the band (here `ne[1] == 1`) must reproduce the arithmetic of the
standalone kernel at that width, or the band is not width-uniform.  Block 08's `ssm_gate_beta` fused the
qwen35moe alpha/beta Q8_0 projections at `ne[1] == 1` but kept plain `calc_nwarps()` (1 warp) while the
standalone dense weight launch uses `calc_nwarps_weight()` (8 warps for Q8_0 with `K < 4096`) — so `W=1` and
`W>=2` reduced K with different warp counts.  Symptom: 35B-A3B `-ncmoe 99` `none = 6744006631df` vs
`n3 = 431bbf3a1605`, diverging ~200 tokens in.

**Fix:** select `calc_nwarps_weight(GGML_TYPE_Q8_0, 1, table, long_k = ne[0] >= 4096)` — exactly the
standalone selector; a no-op for `K >= 4096`.  `none == n1 == n3 == n7 == 431bbf3a1605`, acceptance
unchanged.

**Rule:** any new decode/verify fusion whose matcher has a width condition (`ne[1] == 1`, `ne[2] == 1`,
`n_tokens == 1`) must either be bit-identical to its unfused chain at that width (preferred) or serve the
whole band (`shexp_down_gate`).  A `ne[1] == 1` fusion that does neither is a latent width-impurity
everywhere it fires.

## 39b. The MoE expert-cache band follows the routed MMVQ band

(Originally mis-numbered `§39`.)  The cache took over a routed `MUL_MAT_ID` only for `<= 8` tokens, but the
cache serves such a batch only through the routed-expert **MMVQ** kernel, and since the extended routed
band landed that kernel serves **16** columns on RDNA4 — so 12-16-token verify batches were stranded on
the host path with the arena unused.

**Fix:** the band is `moe_cache_max_tok_dev(device)`, the narrowest `get_mmvq_mmid_max_batch` over the
device's quantized types (never below 8): **16 on RDNA4**, 8 on RDNA3 (where the 16-wide band is incorrect
on gfx1100) and NVIDIA.  It has a single owner through a `moe_cache_band` backend iface hook, so the
scheduler's band decisions and the Meta `moe_cache_update` cannot disagree.  `GGML_MOE_CACHE_MAX_TOK=8`
restores the old band.

**Purity:** nothing changes inside `n_max <= 7` (`none == draft-mtp n7` byte-identical to pre-r31), and at
the newly served widths (9-16) the arithmetic is the same routed MMVQ kernel all-VRAM already used, so the
band-16 cache path is bit-identical to all-VRAM at `n_max 8`.

## 40. A thread count is a scheduling parameter, not a numerical one

The CPU backend's op decomposition never lets the thread count change a result — every output element is
reduced by exactly one thread — so narrowing or widening the pool changes *when* work happens, not *how* it
adds up.  The r19 offloaded-MoE decode cap is byte-identical at `-t 1/8/12/16`, with
`GGML_CPU_MOE_OFFLOAD_THREADS=0/12`, with `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` and under
`--cpu-mask`.

**Rule:** a change that only selects a thread count (graph, backend or op) is a perf change and needs no
purity re-run.  A change that moves a reduction boundary, a `vec_dot`, or a K-split *inside* an op does.
State which one a patch is before calling it purity-neutral.  (§39's `calc_nwarps_weight` is the contrast.)

## 40b. A reserve/measure graph must match its memory context's stream count

(Originally mis-numbered `§40`.)  `build_attn_mha` splits the query by the cache's stream dim
(`n_stream = k->ne[3]`) and `ggml_flash_attn_ext` asserts `q->ne[3] == k->ne[3]`; a graph built against a
memory context whose stream count differs cannot be made valid (the query view truncates, the attention
output has zero tokens).

**Fix:** the TODO #42 post-prefill re-reserve passed a hard-coded `n_seqs = 1` while holding the current
ubatch's context, so a multi-sequence decode reserved a 1-sequence graph and `llama-batched-bench -npl 4`
aborted.  It now uses `max(1, ubatch.n_seqs)`.  Single-sequence (MTP) reserves are unaffected.

## 41. A runtime, address-selected fusion is a cross-process purity risk; a compile-time one is not

Reproduced at the logits level (27B Q6_K, P=256, W=1 row-0): the only lever on the reporter's per-start
flip was the **address-overlap-gated fusion subset**, specifically the upstream `ROPE → VIEW → SET_ROWS`
fusion (disabling just it reproduced the all-fusions-off hash; disabling its sibling did not).  The
arithmetic was then separated from the layout side-effect: a canonicalised per-graph allocation dump is
byte-identical between the configs, both instantiations compute the same `x0`/`x1`/`cos`/`sin` and the
same float result — but `<float,__half>` stored -0.3562 while `<float,float>` + `k_set_rows` stored
-0.3564, because clang contracts the two instantiations' multiply-adds differently and one element of the
prefill write crosses an f16 rounding boundary.  **Fixed in r25** with `#pragma clang fp contract(off)` in
`ggml/src/ggml-cuda/rope.cu` (W=1 hash `60e77916673db071` for both configs).

**The residual per-start flip is hipBLASLt solution selection in ROCm** (the F32 `ssm_alpha`/`ssm_beta`
`mul_mat` at 512 prefill columns picks a per-process solution, reference pair `140231`/`140232` vs
deviating `140216`/`140217`).  It reproduces on stock upstream; this is
[ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126), workaround
`ROCBLAS_USE_HIPBLASLT=0`.  Issue #67 is closed as external.

**Rule:** a per-type *compile-time* constant may move the rounding path uniformly across widths and starts;
a *runtime* selection that reads allocator addresses may not.  If a fusion's
`ggml_cuda_check_fusion_memory_ranges()` outcome can change with the allocator, its fused kernel must be
bit-identical to the chain it replaces.

## 42. A width gate is not a purity rule: extending a fused chain *down* to one token

Two r78 block-10 fusions extend an existing chain or move a launch, and both keep §39's rule:

* **ADD + `rms_norm_q8_1` at one token (PR #174).**  The matcher excluded `ne[1] <= 1` because the
  *attention* residual ADD is folded into the output projection's mmvq epilogue at one token — but the
  *FFN* residual ADD is not (its producer is not a matmul).  The fused kernel's values are the unfused
  pair's, and it is measured: same-seed text byte-identical to r77 (dense 27B `7ef45a05dcfe`, MoE 35B-A3B
  `e726a23ca0cc`, `-n 3000`, `none == draft-mtp n3`), width probe unchanged, prefill-logit unchanged.
  The lesson is the converse of §19: a band gate can be a *superset* of the shapes a fusion is safe for,
  and widening it needs its own measurement.
* **KV-rotation FWHT in the rope kernel, and the projection Q8_1 in the attention-output FWHT (PR #172).**
  Bit-identical by construction (same helper, same contraction mode; the Q8_1 write is `quantize_q8_1`'s
  own reduction under the mmvq launcher's own cache key) and measured: width probe maxdiff 0 at
  P=512/3584, Gemma4 prefill-logit **0.000000 / 100.000 %**, greedy text identical with and without
  `GGML_CUDA_GRAPH_OPT=1`.

**Reachability is part of the claim.**  #172 fires only for NORMAL/NEOX rope with the norm+mul+rope chain
(Gemma4, and gpt-oss-20b for the Q8_1 half); on `rope_multi` models it does nothing, so a "purity + speed"
claim from one model does not transfer.  PR #173 was rejected for r78 for exactly this reason — its matcher
cannot fire on any model here, so no correctness result about it could be produced at all.
