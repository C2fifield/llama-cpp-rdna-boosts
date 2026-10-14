# ngram-mod-host-experts — `ngram-mod` + `draft-mtp-adaptive` roughly halves decode when the MoE experts are host-resident

**Status: OPEN (seeded 2026-10-14), scoping + measurement.**  Not part of the delivery.  Nothing here may
be applied to the fork without the maintainer's go-ahead.

> **Cold start.**  Read `AGENTS.md` ("WIP and promotion rules", the `-ncmoe`/expert-cache invariants under
> "Critical facts"), then the README "Recommended configuration — adaptive MTP + `ngram-mod`" section and
> the adaptive-MTP baseline gate (`benchmarks/mtp-adaptive-methodology.md`).

## 0. The symptom

The delivery's recommended speculative-decoding config is `--spec-type draft-mtp-adaptive,ngram-mod`
(`--spec-ngram-mod-n-match 45`, `--spec-draft-n-max 9 --spec-draft-n-start 9`).  It is measured on
**fully device-resident** models.  On a config where the MoE experts are **partially or fully
host-resident** (`-ncmoe`/`-cmoe`, i.e. the pinned `ROCm_Host` master, the MoE expert cache, or the
op-offload H2D staging path active), adding `ngram-mod` to `draft-mtp-adaptive` **roughly halves decode
throughput**.  Dropping `ngram-mod` and using plain `--spec-type draft-mtp-adaptive` restores the decode
rate expected from the adaptive controller.

The adaptive controller alone handles replayed/verbatim content well: it drafts deeper while the content
is predictable, so it replays prior text quickly.  It is not as fast as `ngram-mod` at verbatim recall,
but it is far ahead of a halved decode, so on a host-expert config plain `draft-mtp-adaptive` is the
recommended pick.  The README documents this; this campaign finds out why it happens.

## 1. What is known

- The regression is **decode-side** (the report is about `tg`); prefill is not implicated.
- It is correlated with the **host-resident expert path** (MoE caching active), not with one model: the
  same combo on a fully device-resident model is the measured best configuration.
- It is an interaction: plain `draft-mtp-adaptive` and (presumably) `ngram-mod` alone are each fine.
- The trigger appears to be the combination, so the prime suspect is something the combo adds that the
  host-expert path pays for per step (verify width, draft-length variability, a graph shape change).

## 2. Hypotheses to rule in or out

1. **Verify width vs the host-expert serve cost.**  `ngram-mod` with `--spec-draft-n-max 9` makes the
   verify batch up to 10 rows, wider than the MTP-only band.  With host-resident experts, each missed
   expert is served by the per-op H2D upload / op-offload / expert-cache fill (or a pinned-master read),
   so the host-expert work scales with the verify width.  Plain `draft-mtp-adaptive` may hold the width
   low.  Measure the per-step H2D bytes and the verify-width distribution with and without `ngram-mod`.
2. **The MoE cache band.**  The expert-cache decode/verify band follows the routed-expert MMVQ band
   (16 tokens on RDNA4).  Check whether the combo crosses it or changes the hit/fill/eviction/takeover
   behaviour (`moe_cache_report`).
3. **Graph capture / recapture.**  `ngram-mod` makes the accepted draft length vary call to call, so the
   verify graph shape varies.  On the host-expert path that may force recaptures or drop out of a
   captured graph, paying the per-op upload every step.  Check `llama-cli -v` "graphs reused" /
   "graph reallocs" in both arms.
4. **Staging-ring width gate.**  The op-offload H2D staging ring is gated at >= 64 tokens (below it the
   upload is synchronous).  Confirm whether the combo's verify shapes sit below the gate.
5. **Scheduler / Meta split interaction.**  `-ncmoe` + the combo may route or split differently
   (`-sm tensor` in particular).  Compare the scheduler plan / split state if 1-4 are inconclusive.

## 3. Investigation plan

1. Reproduce on a host-expert MoE model both arms share, e.g. 35B-A3B Q8_0 `-ncmoe N` (or Flash-Next
   IQ4_NL `-ncmoe`), `-sm tensor`, `llama-bench -p 0 -n 128`, arms
   `--spec-type draft-mtp-adaptive` and `--spec-type draft-mtp-adaptive,ngram-mod`.
2. Record for both arms: decode t/s; acceptance and verify-width distribution; `moe_cache_report`;
   graph reuse/realloc counters; and, with the op-offload trace, the per-step H2D bytes.
3. Bisect the combo: `ngram-mod` alone, MTP-only, and the combo; vary `--spec-draft-n-max` (e.g. 3, 6,
   9) to separate the cap effect from `ngram-mod` itself.
4. Decision: a width/band fix, an automatic gating rule (disable `ngram-mod` when host experts are
   active), or a documented limitation.

## 4. Gates (when a fix exists)

- The host-expert config with the combo recovers to at least the plain `draft-mtp-adaptive` decode rate.
- The fully device-resident combo does not regress from the current README numbers.
- Same-seed coherence (and the prefill-logit gate if a prefill kernel or the MoE path changes).

## 5. Evidence

- (to be filled in.)
