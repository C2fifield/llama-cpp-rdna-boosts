# `rs-rollback-boundary` — the recurrent rollback-boundary guard and the "large prefill as the last prepared batch" ordering (issue #131)

**Status: OPEN, active (being resumed).** Not part of the delivery; nothing here may be applied to the
fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). The
per-sequence guard fix itself has been folded into the delivery (block 02); this campaign is the
**remaining** investigation, which is not a delivery correctness fix. The r44 `WORKLOG.md`/`TODO.md`
wording that called this ordering "parked" is **historical**: the campaign was seeded, never closed, and
`wip/CAMPAIGNS.md` lists it as OPEN. **Resume from §3 (open questions).**

> **2026-10-19 (gfx1151, single device) — open question 1 is answered: the ordering is a bookkeeping
> artefact.**  `prepare()` folded the per-sequence last-batch tracker in from `ubatches.back()` **only**,
> so a sequence that ran in a non-final ubatch of the same call kept an older count.  One server call
> routinely carries several ubatches (`[seq 1 verify 4 tokens] [seq 2 prefill 6136 tokens]`), so the
> recorded "4571-token prefill of that same sequence" was a `prep=1/2`/`prep=2/2` split, not a real
> scheduling order.  Every ubatch is now folded in, oldest first; the single-device reproducer goes from
> a stale-count risk to **0/95 warnings** with identical text.  Detail, evidence and the patch
> ([`rs-rollback-tracker.patch`](rs-rollback-tracker.patch)) are in §7.  Question 2 (is the restored
> snapshot stale?) is untouched and no longer reachable via this ordering.

> **Cold start.** Read this file in full, then `AGENTS.md` ("Report findings before shipping",
> "WIP and promotion rules", "Default-on policy"), then issue #131 and the predecessor records
> [`archive/work/moe-verify-fusions/FINDINGS-single-seq-rollback.md`](../moe-verify-fusions/FINDINGS-single-seq-rollback.md)
> and [`HANDOVER-single-seq-rollback.md`](../moe-verify-fusions/HANDOVER-single-seq-rollback.md).
> Report findings to the maintainer before any delivery change and wait.

---

## 0. What this campaign is for

Issue #131 reports that `seq_rm: rollback crossed a batch boundary` (in
`src/llama-memory-recurrent.cpp`) fires in normal `-np 2` plus `--spec-type draft-mtp` use, and argues
it is a false positive caused by tracking the "last batch" globally instead of per sequence. The
**report** is closed as not reproducible on our current tree with the reporter's model class, and the
**per-sequence guard** is folded into block 02. This campaign exists for the two things that are still
open:

1. **Why is the guard's condition still reached at all?** The per-sequence tracker shows that in the
   one reproducer that fires on our tree, the rolling sequence's **own** last prepared batch is a large
   prefill, not another sequence's chunk. If that is real, the guard is warning about a genuine
   "rollback after a chunked prefill" (the very case it exists to catch); if it is a bookkeeping
   artefact, the tracker is not measuring what we think it is. Either answer changes the fix.
2. **Is the restored recurrent state actually stale in that case?** The guard assumes the whole-batch
   chunked GDN prefill writes only the newest snapshot, so restoring snapshot `rollback` reads an
   unwritten plane. Nobody has yet demonstrated a bit-level consequence for that specific ordering.

The maintainer's current position (2026-10-11): the warning is a benign timing/scheduling artefact and
not a delivery correctness fix. This campaign is where a future session proves or refutes that.

## 1. The code

`llama_memory_recurrent::prepare()` records the shape of the batch the memory is preparing:

```cpp
if (!ubatches.empty()) {
    const llama_ubatch & ub = ubatches.back();
    last_ubatch_nseq_tokens = ub.n_seq_tokens;      // GLOBAL (the original)
    last_ubatch_pos_last    = ub.n_tokens > 0 ? ub.pos[ub.n_tokens - 1] : -1;
}
```

`llama_memory_recurrent::seq_rm()` checks it on a partial rollback:

```cpp
if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
    if (!warned_rollback_boundary && last_ubatch_nseq_tokens > n_rs_batch) {
        LLAMA_LOG_WARN("... rollback crossed a batch boundary ...");
    }
    set_rs_idx(seq_id, (uint32_t) rollback);
    cell.pos = p0 - 1;
    return true;
}
```

The rollback state itself is per sequence: `rs_idx[seq_id]`, and the GDN snapshot plane is selected by
`rs_idx_cur * size + cell` (per cell slot), so the restore does not read another sequence's plane.
The guard is a **warn-once diagnostic**; it does not change the restore.

**The delivery fix (block 02):** track the last-batch token count **per sequence** (only the sequences
present in the last ubatch are updated; the rest keep the count from the last batch they ran in), and
guard on the rolling-back sequence's own value. The diagnostic message prints the per-sequence count.

## 2. What was measured (2026-10-11, r43, `/home/stew675/kv-offset-r43`)

**Dense Qwen3.8-27B Q5_K** (the reporter's class), `-np 2`, `--kv-unified`, `-c 180224 -b 6144 -ub 512`,
q8_0 KV, `--spec-type draft-mtp --spec-draft-n-max 3`, A about 24k chars / B about 40k chars, solo plus
3 concurrent repetitions:

| arm | partial rollbacks | warning | notes |
|---|---|---|---|
| `--kv-unified` | 106 | **0** | every rollback preceded by a 2 to 4 token verify last batch |
| `--no-kv-unified` | 55 | **0** | the recurrent cache is independent of `--kv-unified` |

With the per-sequence tracker instrumented, the firing reproducer is the **MoE** `qwen4exp` GSQ
(`archive/work/moe-verify-fusions/tools/multiseq.sh`, `-sm tensor -ncmoe 48`, shared MTP head). There
the warning fires once, and the per-sequence value for the rolling-back sequence is a **4571 token
prefill of that same sequence** (the global value is another sequence's 6136 token prefill). The
`RSPREP` log shows the sequence's prepare order as `4, 4, 4, 1, 4571, 6136, 6136, 6136`, then the
rollback for that sequence.

So: the global tracker is wrong (confirmed), but per-sequence tracking does **not** by itself remove the
warning in the one firing case. The open question is what the 4571 prepare is and whether a rollback can
legitimately follow it.

## 3. Open questions (the actual work)

1. **Whose prepare is the 4571 token one, and when does the MTP verify prepare for that sequence run?**
   Add the sequence ids to the `RSPREP` log (the instrumentation already does this in the WIP tree) and
   compare with the `slot ... task ...` and `accepted N/M draft tokens` lines. In particular, check
   whether the draft/verify batch goes through `llama_memory_recurrent::prepare()` at all for that
   context (a `llama_memory_hybrid_idx` MTP context skips the recurrent prepare in some paths; see
   `hybrid_idx_no_recr`), which would leave the tracker showing the last *prefill*.
2. **Is the snapshot stale?** With the per-sequence guard in place, run the firing reproducer and force
   the sequential GDN (`GGML_CUDA_GDN_CHUNKED=0`) versus the chunked default and diff the generated
   text and the MTP acceptance. If the chunked arm is silently reading an unwritten plane, the two
   should differ at the rounding level, and the difference should vanish when the rollback is prevented
   (e.g. pin the sequence into its own snapshot, or never let the large prefill be the last prepare).
3. **Where does the ordering come from?** The warning needs a large prefill to be the last prepared
   batch immediately before a rollback. Decide whether that is (a) a real scheduling order produced by
   the server, (b) a stale tracker entry that per-sequence tracking should reset, or (c) an artefact of
   the MTP draft context sharing the target memory.
4. **If it is real:** the fix is not the guard. It is the §27 invariant ("every batch that can be rolled
   back into must run the kernel that writes the snapshots it will read"): either never present a
   rollback whose sequence's last decoded batch exceeded `n_rs_batch`, or write the pre-batch plane for
   the chunked prefill when a rollback can follow it, or pin the rollback into the batch's own snapshot.
   Confirm with a 2 sequence by prefill-in-flight by `GGML_CUDA_GDN_CHUNKED` matrix.

## 4. Reproducers and tools

- `tools/issue131_repro.sh <build_dir> <tag>`: the dense 27B repro. Env `KVUNI`, `CHUNKED`, `CTX`, `NB`,
  `UB`, `GPUS`, `PROMPT_A`/`PROMPT_B`, `NPA`/`NPB`. Does A solo plus 3 concurrent A+B repetitions,
  prints the text hashes and the warning count.
- Prompts: `prompts/issue131-A.txt` (24000 B, sha256 `0282e1856ed4…`) and
  `prompts/issue131-B.txt` (40000 B, sha256 `403c8c20b66c…`) live under **this campaign's** `prompts/`
  (not the repo-level `prompts/`, whose files are the adaptive-MTP axes).  Both were cut from
  `../kv-cell-offset-nondeterminism/prompts/mixed30k.txt` (sha256 `69624f4d…`).
  New runs should cut from `mixed30k-v2.txt` (the fault-grep-clean revision, sha256 `3e22e0fd…`), so a
  fault grep over the server log does not match the echoed prompt; the old cut is kept so the recorded
  result stays reconstructible.
- `tools/rs131-repro-1gpu.sh <build_dir> <tag>` **(added 2026-10-19)**: the single-device variant of
  `multiseq.sh` for a box without a second GPU (`-sm layer`, `NCMOE=0`, `-np 2 --kv-unified`, prompts A/B,
  two concurrent greedy requests, dumps the warning count and the `RSPREP`/`RSROLL` lines).  Env: `M`,
  `PROMPT_A/B`, `NPA`/`NPB`, `SM`, `NCMOE`, `CTX`, `UBD`/`BD`, `KVUNI`, `GPUS`, `DBG`.
- The MoE firing reproducer is `archive/work/moe-verify-fusions/tools/multiseq.sh` (GSQ + shared MTP
  head, 2 GPUs, `-ncmoe 48`).
- WIP instrumentation **on branch `wip/umem-arena-guard`** (`LLAMA_RS_ROLLBACK_DEBUG=1`): `RSPREP` per
  prepared ubatch (`n_tokens`, `n_seq_tokens`, `n_seqs`, `n_seqs_unq`, `n_pos`, first/last pos,
  `seq_ids`) and `RSROLL` per partial rollback (global and per-sequence last-batch count/pos).  The
  per-sequence tracker lives in `src/llama-memory-recurrent.{h,cpp}`; the same patch carries the §7 fix.
  Not in the delivery.
- Build: `BUILD_DIR=build-rocm-hybrid ~/bin/build-llama-rocm`
  in the worktree, then `cmake --build build-rocm-hybrid --target llama-server -j 16`.

## 5. Gates to keep green

`scripts/validate-set.sh`; coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42);
`scripts/gate-prefill-logits.sh` (KLD <= 0.005 / same-top-p >= 98 %);
`test-recurrent-state-depth` (baseline 130 failures) and `test-recurrent-state-rollback`;
`MUL_MAT_ID` 931/931; `FLASH_ATTN_QSA` >= 26/26;
`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (rule 5, decode must not regress). The dense 27B
repro above is a reproducer, not a gate.

## 6. Pointers

- Issue #131 (closed 2026-10-11 as not reproducible on the current tree; reopen if the reporter shows a
  real case). Issue #134 / TODO #52 for the separate unified-KV cell-offset residual.
- `archive/work/moe-verify-fusions/FINDINGS-single-seq-rollback.md` (§7 already separates this from
  #50/#51/#52) and `HANDOVER-single-seq-rollback.md` §6.
- `archive/work/gdn-rs-rollback/` (the snapshot-bound work).
- `TODO.md` #52 (the `seq_rm` boundary warning is described there as a false positive).
- **Not related:** issue #133 (long-context Flash-Next decode falls back to the tile FA kernel because
  GQA 12 is outside the band-WMMA gate) is attention-kernel selection, a different subsystem from the
  recurrent memory rollback guard. The shared use of Flash-Next + MTP is incidental; there is no shared
  root cause. See `wip/CAMPAIGNS.md` for the current open set.
- The delivery fix: `patches/0002-rdna-boosts-block-02-fused-chunked-gated-delta-net-pre...patch`
  (per-sequence guard); record in `WORKLOG.md` for the release that carries it.

## 7. 2026-10-19 — the "large prefill before a rollback" ordering is a stale tracker entry

Run on **gfx1151 (Strix Halo, one device, 16 cores, 123 GiB unified)**, `build-rocm` = the `patches/`
r89 tree plus the campaign's own instrumentation (no delivery change in this section's evidence run).
Model Qwen3.6-35B-A3B UD-Q4_K_M (GDN hybrid MoE, embedded MTP head, fully resident, `-ncmoe 0`),
`llama-server -np 2 --kv-unified -c 32768 -ctk/-ctv q8_0 -fa on -t 8 --fit off --spec-type draft-mtp
--spec-draft-n-max 3`, prompts A (24000 B) and B (40000 B) concurrently, two repetitions.

| arm | rollbacks | warning | A / B text (both reps) |
|---|---:|---:|---|
| `-ub 512 -b 4096`, 256/96 tokens | 61 | **0** | identical |
| `-ub 6144 -b 6144`, 48/24 tokens | 16 | **0** | identical |
| `-ub 512 -b 4096`, 512/192 tokens | 95 | **0** | `ff07fb9502e2` / `2b1ad896a5b6` |

So the warning is not reachable on a single device in this configuration — which is the *result*, not a
dead end: the `RSPREP` lines show why it ever fired at all.

### 7.1 The mechanism

One `llama_decode` call carries a **list** of ubatches, and a two-slot server under `--kv-unified` builds
one ubatch per distinct graph shape.  Two real examples from the run:

```
RSPREP prep=1/2 n_tokens=512 n_seq_tokens=256 n_seqs=2 n_seqs_unq=2 pos=[12662..7682] seq_ids=[1,2]
RSPREP prep=2/2 n_tokens=512 n_seq_tokens=256 n_seqs=2 n_seqs_unq=2 pos=[12918..7938] seq_ids=[1,2]

RSPREP prep=1/2 n_tokens=2 n_seq_tokens=2 n_seqs=1 n_seqs_unq=1 pos=[13271..13272] seq_ids=[1]
RSPREP prep=2/2 n_tokens=4 n_seq_tokens=4 n_seqs=1 n_seqs_unq=1 pos=[8028..8031] seq_ids=[2]
```

The second pair is a seq-1 decode and a seq-2 verify in the same call.  `prepare()` recorded the
per-sequence last-batch shape from **`ubatches.back()` only**, so seq 1's entry was not updated by that
call at all — it kept whatever it was the last time seq 1 happened to BE the back ubatch, typically **its
own prefill chunk**.  That is exactly the recorded firing case: the log's prepare order
`4, 4, 4, 1, 4571, 6136, 6136, 6136` followed by a rollback, where 6136 was the *other* sequence's
prefill (the back ubatch) and **4571 was the rolling sequence's own earlier prefill**, still latched
because its 4-token verify ran in the non-back ubatch.  The guard then compared a rejection rollback
against a prefill that had already completed.

So open question 1 is answered, and open question 3 is **(b) — a stale tracker entry that per-sequence
tracking should reset**; the ordering is not produced by the server.  `/tmp` logs:
`/tmp/rs131/{t1,t2}.err`, `/tmp/rs131c/t4.err` (pre-fix), `/tmp/rs131d/t5.err` (post-fix).

### 7.2 The fix

Fold **every** ubatch of the prepare list into the per-sequence tracker, oldest first (reset then count,
per ubatch), leaving the global `last_ubatch_*` as the back ubatch's shape for the message.  Patch:
[`rs-rollback-tracker.patch`](rs-rollback-tracker.patch) (sha256 `43c769c6…`, one file, +55/−14; the same
patch adds the `LLAMA_RS_ROLLBACK_DEBUG` instrumentation).  The tracker is `patches/` block 02's code, so
promotion is a block-02 amendment.

Verified pre/post fix on the longest arm above: **95 rollbacks, 0 warnings, and byte-identical
generated text for both sequences and both repetitions** (`ff07fb9502e2` / `2b1ad896a5b6`) — the change
touches only the guard's bookkeeping, never the restore.

### 7.3 What is left

* **Open question 2 (is the restored snapshot stale?) is unchanged and now unreachable by this route.**
  The only ordering that could still present a rollback after a genuinely large last batch is a real one
  (e.g. a prefill and a rejection in the same ubatch); with the tracker fixed, the guard would only warn
  about a batch that really was the sequence's last.  The deterministic instrument for the staleness
  question is `archive/work/moe-verify-fusions/tools/rrewind.cpp` extended to "prefill a large batch,
  then roll it back" with `GGML_CUDA_GDN_CHUNKED` on/off — not done here.
* The single-device arms do not reproduce the *multi-device* `multiseq.sh` scenario (`-sm tensor`,
  `-ncmoe 48`, `-ub 6144`, two devices).  The fix is topology-independent, but the 2-GPU arm should be
  re-run on the primary box to confirm the warning is gone there too (it is the arm that recorded it).
* `TODO.md` #52's wording still calls this "OPEN and being resumed"; after the 2-GPU confirmation it can
  be closed as a false positive with a bookkeeping root cause.

### 7.4 Gates run with the fix (2026-10-19, gfx1151, baseline = `rdna-boosts` tip `15552e758`)

| gate | baseline | with the fix | verdict |
|---|---|---|---|
| `tools/rs131-repro-1gpu.sh` (35B-A3B GDN MoE + MTP, `-np 2 --kv-unified`, `-ub 512`, 512/192 tokens) | 95 rollbacks / 0 warnings | 95 / **0**, text `ff07fb9502e2` / `2b1ad896a5b6` **identical** | PASS |
| `test-recurrent-state-depth` (35B-A3B) | **153** failures | **153** | unchanged (the box baseline is 153, not the `AGENTS.md` 130) |
| `test-recurrent-state-rollback` | 0 failures | **0** (`recurrent rollback checkpoint restored`, `multi-seq split replay matched`, max diff 0) | PASS |
| dense coherence 4B Q8_0 seed 42, 32 tokens | `b0e99b1137b4…` | **identical** | PASS |
| `scripts/gate-prefill-logits.sh` (9B Q6_K, 40 x 512) | PPL 7.2099, base `06d1b8e5…` | mean KLD **0.000000**, same-top-p **100.000 %** | PASS |

`validate-set.sh` is not applicable until the change is folded into block 02.

## 8. 2026-10-19 — open question 2 answered: a rollback into a chunked-GDN prefill batch IS wrong

Tool: [`tools/prefill-rollback.cpp`](tools/prefill-rollback.cpp) (`MODE=ref`, the default).  It builds two
contexts and compares **semantics**, not the state blob:

```
ctx A: prefill P tokens (one `ub` batch -> the chunked kernel), seq_rm(0, P-k, -1), then replay the last
       k tokens one at a time and record the logits at each step
ctx B: prefill P-k tokens (same kernel family), then replay the SAME k tokens one at a time
```

Both contexts then have the same kernel history for the suffix, so the only thing that can differ is the
pre-state the rollback restored.  Build:
```bash
g++ -O2 -std=c++17 tools/prefill-rollback.cpp -I~/llama.cpp/include -I~/llama.cpp/ggml/include \
    -L~/llama.cpp/build-rocm/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin -o /tmp/prefill-rollback
```

Run (35B-A3B UD-Q3_K_M / GDN MoE, gfx1151, one device, `RS=3`):

| arm | k | reference | result |
|---|---:|---|---|
| chunked (default) | 1 / 2 / 3 | P=2048, P-k | **FAIL**, logit maxabs 6-12, nmse **0.17-0.65** |
| chunked, aligned (`P=1986`, `P-k=1984=31*64`) | 2 | one full chunk shorter | **FAIL**, maxabs 5.1-5.8, nmse **0.18-0.19** |
| chunked, fp32 (`GGML_CUDA_GDN_CHUNKED_BF16=0`) | 1 | P=2048 | **FAIL**, maxabs 14.9, nmse **0.72** (not a bf16 issue) |
| **sequential** (`GGML_CUDA_GDN_CHUNKED=0`) | 1 / 2 / 3 | P=2048, P-k | k=2 **exact (0.0)**; k=1/3 rounding-level (nmse 1.6e-3 / 2.7e-4) |
| **sequential**, aligned (`P=1986`, `P-k=1984`) | 2 | one full chunk shorter | **exact (0.0, both positions)** |

The aligned arm is the controlled one: with the reference's final GDN chunk boundary identical, the
**sequential kernel restores bit-exactly** and the **chunked kernel is off by ~0.18 nmse** — four orders of
magnitude above the chunked kernel's documented rounding error (op NMSE ~1e-5).  The unaligned sequential
k=1/k=3 residue is a *reference-structure* artefact (a shorter final chunk in ctx B), which is why the
aligned arm exists.

So: **the state the chunked kernel leaves behind after a large prefill is not a valid rollback target.**
That is not a surprise in the code — `gated_delta_net.cu` states the precondition explicitly:

> "A batch with more tokens than max(K, 16, n_rs_batch) cannot be a speculative verify batch ... and is
> therefore never rolled back into, so the K rollback snapshots are useless to it and the sequential kernel
> can be skipped entirely: zero cost, unbounded n_max, no KTAIL."
> "(Verified: over llama-cli draft-mtp and llama-server with `--cache-reuse`, every partial rollback was
> preceded by a batch of exactly K or fewer tokens.)"

### 8.1 What this changes

* **Open question 2 is answered: yes, the restored recurrent state is stale/wrong** when a rollback lands
  in a chunked prefill batch, by a wide margin, and `GGML_CUDA_GDN_CHUNKED=0` is bit-exact.  The issue-#131
  guard is therefore **protective**, not merely diagnostic.
* **But the ordering is never presented** by the scheduler, in either of the two ways it was suspected of
  being presented: the §7 tracker artefact (a false alarm) and the code's own verification above.  So no
  user-visible corruption has a route today; the exposure is a *future* change that makes a verify batch
  longer than `max(K, 16, n_rs_batch)` — e.g. an `ngram-mod` draft (`--spec-draft-n-max` up to 64) or a
  raised `--spec-draft-n-max` cap.
* Recommended follow-ups (maintainer's call, none done here):
  1. **Assert the precondition in `seq_rm`** (debug-only) instead of only warning: if a partial rollback
     lands on a batch the chunked kernel served, that is a correctness bug, and the existing WARN is the
     right shape but should say "unsupported ordering" rather than "set GGML_CUDA_GDN_CHUNKED=0".
  2. **Make the bound a real invariant**, not a code comment: derive the "longest batch that can be rolled
     back into" from `n_rs_batch` (it already is) *and* from the enabled speculator, and refuse (or fall
     back to sequential) when a speculator can exceed it.
  3. Keep the guard's per-sequence fix (§7): it removes the only false-positive route.

### 8.2 What is the longest *supported* ordering? **16 tokens** (within the shipped draft cap)

Same probe, sweeping the prefill batch size `P` (the rollback target's batch) with `n_ubatch=512` and
`k=2` (35B-A3B UD-Q3_K_M, gfx1151).  PASS = the rollback restores the true prefix state.

| `n_rs_seq` (RS) | K = RS+1 | `n_rs_batch` | `GDN_CHUNKED_MIN_TOKENS` = max(K>16?K:16, n_rs_batch) | P=14 | 15 | 16 | 17 | 18 | 20 | 21 | 22 |
|---:|---:|---:|---:|---|---|---|---|---|---|---|---|
| 3 (MTP n3) | 4 | 4 | **16** | OK | ~3e-4 | **OK** | **0.09** | 0.07 | 0.19 | 0.33 | 0.23 |
| 15 (the `--spec-draft-n-max` cap) | 16 | 16 | **16** | OK | ~3e-4 | **OK** | **0.09** | 0.07 | 0.19 | 0.33 | 0.23 |
| 20 (past the cap) | 21 | 21 | **21** | OK | ~3e-4 | OK | ~3e-4 | **OK** | **OK** | ~5e-4 | **0.08** |

Numbers are logit NMSE for the first replayed position; **bold** marks the transition.  So:

* **The longest batch that can be rolled back into is `max(16, n_rs_seq+1, n_rs_batch)` tokens**, exactly
  the kernel's own threshold — i.e. **16** for every draft depth the delivery allows
  (`--spec-draft-n-max <= 15` -> `K <= 16` -> a verify batch of at most 16 tokens).  The maintainer's
  guess of 16 is right, and it is not a coincidence: **the clamp is the bound.**  A 17-token batch needs
  `n_max >= 16`, which the clamp forbids; past the cap the bound moves with `K` (21 in the RS=20 arm, and
  the break moves to 22ff).
* Below the threshold the residue is **rounding-level and batch-width dependent** (`~2e-4..9e-4` nmse at odd
  widths, exact `0.0` at even widths) — the sequential kernel's tail/vectorisation, not a correctness break.
  So "rewind is bit-exact" holds for even widths (`rrewind`'s W=4 is exact); an odd-width claim should say
  "rounding-level".
* One robustness nit found on the way: `llama_batch_allocr::split_equal` has
  `GGML_ASSERT(n_ubatch > n_keep_tail)` with `n_keep_tail = n_rs_seq + 1`, so `-ub <= n_max+1` **aborts the
  process** at decode (a server with `--spec-draft-n-max 15 -ub 16` dies).  Servers run `-ub` in the
  hundreds so it does not bite in practice, but the constraint is undocumented and the failure is an abort.

## 9. Scoping note (2026-10-19): make the rollback window a fixed depth instead of `n_max+1`

**Maintainer's proposal:** reserve a fixed rollback history (their example: 64 entries) so that *any* batch
can always be rolled back into, instead of relying on "the last batch was at most `max(16, K)` tokens".

### 9.1 Verdict: possible, and the machinery is mostly there

1. **The slot semantics already match the idea.**  `gated_delta_net.cu:167` says *"snapshot slot mapping:
   slot 0 = most recent state, slot s = s tokens back"*, and `delta-net-base.cpp` copies the pre-batch state
   to slot `n_seq_tokens`.  A rollback of `r` reads slot `r`.  So the "copy the last 64 entries to 0-63"
   step is **not needed**: a per-token tail writes slots 0..D-1 directly, at their natural distance index.
2. **The chunked kernels already take a prefix limit** (`n_tokens_limit`, "process only the first
   n_tokens_limit tokens (MTP prefix)").  The missing piece is a **sequential tail**: chunk the first
   `n - D` tokens, then run the existing sequential kernel on the last `D`.  That is exactly the "KTAIL"
   the code deliberately removed — removed not because it is unsound but because a **K-dependent** boundary
   made plain decode (`K=1`) and MTP (`K=n_max+1`) chunk differently, which broke `none == draft-mtp`.
   A **fixed** `D` restores the property they wanted: both paths chunk the same prefix and run the same
   tail, so purity holds.
3. **Why the prior batch is "dropped" today:** the chunked kernel never *computes* per-token states (that
   is its whole speedup — a parallel chunk scan yields only chunk-boundary states), so there is nothing to
   keep.  The fix is not to retain them, it is to recompute a bounded tail sequentially — cheap, one D-token
   run per batch, not per token.

### 9.2 The cost that decides `D`: the state plane is not small

`llama-memory-recurrent` allocates `mem_size * (1 + n_rs_seq)` rows per layer, and on the 35B-A3B
(40 GDN layers, 1 cell, f32) one plane is **60.00 MiB** (`llama_memory_recurrent: … S (f32): 60.00 MiB`):

| depth `D` | planes | 1 slot | 2 slots |
|---:|---:|---:|---:|
| 3 (MTP n3, today) | 4 | 240 MiB | 480 MiB |
| 15 (today's clamp) | 16 | 960 MiB | 1.9 GiB |
| **16** | 17 | ~1.0 GiB | 2.0 GiB |
| **32** | 33 | ~1.9 GiB | 3.9 GiB |
| **64** (the proposal) | 65 | **~3.8 GiB** | **~7.6 GiB** |

(For reference the whole KV cache in that log is 5.0 GiB.)  So `D = 64` is affordable on a 24/32 GiB card
but it is a real budget line, and it scales with `n_seq_max` on a server.  A per-plane dtype knob
(f16/bf16 state storage) would halve it — the state is already fp32 while the KV cache is q8_0.

### 9.3 The change, in pieces

1. **Decouple the rollback depth from the draft depth**: a new constant/knob `D` (default 16, max 64) with
   `n_rs_batch = max(n_max + 1, D + 1)` and the state allocation `(1 + D)` planes — i.e. today's
   `n_rs_seq`-derived sizing becomes `D`-derived.  `K` (op param 0) and the chunked threshold follow `D`.
2. **Sequential tail in the CUDA GDN dispatch** when `n_tokens > D`: `launch_chunked(state, n_tokens - D)`
   (already supported) then the sequential kernel restricted to the last `D` tokens (needs a token-range
   argument; the snapshot slot mapping then uses `D` as `n_tokens`, which is the correct distance index).
3. **Unconditional tail**: it must run on *every* path that can produce a post-prefill state others compare
   against, including plain decode (`n_rs_seq == 0`), or the plain-vs-MTP boundary asymmetry comes back.
4. **Purity/no-regression gates**: this changes the post-prefill recurrent state on every GDN prefill
   (chunked prefix + sequential tail instead of chunked-whole-batch), i.e. exactly the class of change that
   produced issue #113.  It must pass `scripts/gate-prefill-logits.sh`, the `none == draft-mtp` battery and
   the four-axis adaptive-MTP gate before it is default-on.  It is *not* a rounding-free change.
5. **Then** the invariant becomes: a rollback of up to `D` is always safe, whatever the batch was; the
   §8 "unsupported ordering" case disappears for good, and the §7 guard can stay as a cheap assertion.

### 9.4 Recommendation

Worth doing **once**, as a block-03 change, with `D` defaulting to **16** (which keeps today's memory
footprint at the clamp, halves the risk surface, and covers every speculator in the tree) and an opt-in
`D = 32/64` for ngram-style drafts.  Measure: prefill cost of the tail (expect <1 % at `-ub 2048`),
memory delta, KLD gate, and the `prefill-rollback` probe showing PASS at `P > D`.

## 10. Why a plane costs 60 MiB, and why "recompute instead" is half-right (2026-10-19)

### 10.1 The cost is not per token, it is per *whole-model state*

A snapshot is not a token's worth of anything — it is the recurrent state of **every GDN layer at once**:

```
plane bytes = n_gdn_layers x S_v x S_v x H_v x sizeof(state)
            = 40 x 128 x 128 x 24 x 4 B      (35B-A3B: ssm.state_size 128, 40 GDN layers)
            ~ 60 MiB                          (matches `llama_memory_recurrent: … S (f32): 60.00 MiB`)
```

`llama_memory-recurrent` allocates `mem_size x (1 + n_rs_seq)` rows, so `D` rollback distances cost
`D+1` planes, and it scales again with `n_seq_max` on a server.  That is why "4160 entries" is not the
right unit: a chunked prefill has *no per-token state to store*, so the choice is `D x 60 MiB`, not
`tokens x something`.

### 10.2 Recompute: the fold means you can only rewind from a checkpoint you kept

The GDN recurrence is a **fold over the prefix**, so a state can only be reconstructed by re-running the
inputs, and the inputs (q, k, v, g, beta) are activations that are gone once the step ends.  The memory
layer therefore cannot recompute anything; only a re-decode can.  And today a **large batch keeps no
checkpoint at all**: `delta-net-base.cpp` copies the pre-batch state into slot `n_seq_tokens` only
`if (0 < n_seq_tokens && n_seq_tokens < K)` — for a chunked prefill neither the per-token slots nor the
pre-batch slot exist.

So the two levers are genuinely different:

| design | planes (35B-A3B, f32) | common path (verify rollback every step) | pathological path (rollback into a big prefill batch) |
|---|---:|---|---|
| today: `D = n_max+1` (n3) | 4 = **240 MiB** | free | **silently wrong** (§8) |
| fixed window `D = 16` | 17 = **1.0 GiB** | free | safe, wrong beyond 16 |
| fixed window `D = 64` | 65 = **3.8 GiB** | free | safe everywhere |
| **checkpoint + replay** | 2 = **120 MiB** | **replays up to 16 tokens EVERY step** — doubles decode; not viable alone | correct, one replay of the batch |
| **recommended hybrid** | 17 + 1 = **1.1 GiB** (0.55 GiB in f16) | free (the window covers the verify width) | correct via one checkpoint replay |

The key asymmetry: a verify rollback happens **every decode step**, so the per-token window must exist
(D must be >= the widest verify batch = `n_max+1`, i.e. 16 at the clamp) — that part you cannot trade for
replay without halving decode.  But the *pathological* distance is rare, so that part should be recomputed
rather than stored: `D = 16` covers every ordering the scheduler produces; anything beyond it is handled by
a checkpoint replay instead of by 48 extra planes.

### 10.3 The hybrid, concretely

1. **Write the pre-batch checkpoint unconditionally** (drop the `n_tokens < K` condition): one plane copy
   per batch, 60 MiB of buffer, ~1 MiB of traffic — negligible.
2. **Fixed window** `D = max(16, n_max+1)` planes (today's sizing at the clamp) + the fixed sequential tail
   of §9.  Every ordering the scheduler produces is then free, and the tail keeps `none == draft-mtp`
   intact because the boundary is constant.
3. **Replay fallback** in the decode loop: if `seq_rm` is asked for a distance the window cannot serve,
   rewind the KV to the checkpoint position and re-decode that batch.  `seq_rm` would return a distinct
   "not representable" instead of restoring a wrong plane.  Cost: O(batch) once; never in the common path.
4. Then the §8 class of bugs is **gone**, not merely detected, at ~1.1 GiB instead of 3.8 GiB — and the
   §7 guard becomes a debug assertion that the fallback never fires unexpectedly.
5. Optional and independent: store the state planes as **f16** (the KV cache is q8_0 while these are f32) —
   halves every row of the table above.  Separate gate, since it is a rounding change to the rollback state.

Open question for the maintainer: is the replay path worth the decode-loop plumbing, or is `D = 16` plus the
existing WARN/assert enough?  The window alone leaves a silent-wrong class for any speculator that exceeds
16 (which the clamp currently forbids); the replay removes the class entirely.

## 11. The boundary ban is already the policy — it is aimed at the wrong object (2026-10-19)

**Maintainer's proposal:** never predict past a boundary (a verify batch must close the boundary, so the
draft depth is capped by the remaining room).

### 11.1 It is already implemented — twice

* **At the speculator:** `--spec-draft-n-max` is clamped at 15, i.e. `K = n_max+1 <= 16`, which is exactly
  the kernel's snapshot threshold (§8.2 measured it).
* **At the splitter:** `llama-memory-hybrid.cpp` calls
  `split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0)`, with the comment *"the trailing
  (1 + n_rs_seq) tokens of each seq must stay in the same ubatch so that the rollback snapshots remain
  valid"*, and `split_equal`'s `n_keep_tail` logic refuses to split a sequence whose remaining tokens are
  fewer than `n_keep_tail`, emitting it as its own ubatch instead.

So "don't predict past a boundary" is not missing; it is the design.

### 11.2 The ban is on the wrong object

Both mechanisms protect the *rollback window*: they guarantee the trailing `n_rs_seq+1` tokens of a sequence
sit together in **one** ubatch.  Neither bounds the **size** of that ubatch.  A 4096-token prefill chunk is
one such ubatch, and §8 shows a rollback into it restores a wrong state because that ubatch took the chunked
path (no snapshots).  The kernel's answer was an *assumption* — "a batch with more tokens than
`max(K, 16, n_rs_batch)` cannot be a speculative verify batch and is never rolled back into" — which is true
of the speculator but is not enforced anywhere.

### 11.3 The fix the proposal points at: make the trailing window its OWN ubatch

Instead of banning prediction, ban the *window's ubatch from being large*.  When `n_rs_seq > 0`, split any
ubatch so that the sequence's last `D = max(16, n_rs_seq+1)` tokens are emitted as a separate trailing
ubatch.  Then:

* the GDN op for that trailing ubatch sees `n_tokens = D <= threshold`, so **the existing sequential kernel
  runs and writes the snapshots** — no kernel change at all (the chunked/sequential decision is per ubatch);
* the prefix ubatch still takes the fast chunked path, so the cost is one `D`-token sequential GDN run per
  prefill ubatch — `16/512` of the GDN op, and GDN is a minor fraction of prefill (**~0.1 % of prefill**,
  to be measured);
* **no extra memory** (the snapshots already live in the existing `K` planes), **no replay**, no new kernel,
  no API change — a splitter rule plus the existing `D` constant;
* a rollback of `r <= D` is then *structurally* safe for every ordering, including the §8 case, because the
  ubatch it lands in is snapshot-writing by construction.

### 11.4 The one non-negotiable

The tail split must be **unconditional** (a fixed `D`, applied even when `n_rs_seq == 0`), for the same
reason as §9/§10: a `K`-dependent boundary makes plain decode chunk `n` while MTP chunks `n-D + tail`, and the
two post-prefill states disagree — the `none == draft-mtp` purity break the current code avoids by chunking
everything.  With a fixed `D` both paths are identical, so the same rule serves purity and rollback.

### 11.5 Ranking against the alternatives

| approach | extra memory | hot path | rare path | code surface | numerics |
|---|---:|---|---|---|---|
| clamp only (today) | 0 | free | **silently wrong** | — | unchanged |
| fixed window `D=64` (§9) | 3.8 GiB | free | safe | kernel tail launch | prefill state changes |
| checkpoint + replay (§10) | 60 MiB | free | O(batch) re-decode | decode loop | rounding on the rare path |
| **tail ubatch (§11)** | **0** | **free** | **safe** | **splitter rule** | **prefill state changes** |

The tail ubatch dominates: same correctness as `D = 64`, no memory, and the smallest code surface — and it
is the maintainer's own proposal with "boundary" read as *ubatch* rather than *prediction length*.

### 11.6 Work item (this host)

1. A `D` constant (`max(16, n_rs_seq+1)`, env-overridable for A/B) and the trailing-ubatch split in
   `split_equal`/`llama-memory-hybrid` (all hybrid memories: `-iswa`, `-idx`, `dsv4` too).
2. Unconditional (also `n_rs_seq == 0`), so `none == draft-mtp` holds.
3. Gates: `prefill-rollback` at `P > D` must PASS; `gate-prefill-logits.sh` (the prefill state changes —
   mandatory, issue #113 class); `none == draft-mtp` four-axis MTP gate; `test-recurrent-state-*`; prefill
   cost measurement at `-ub 512/2048`.
4. Keep the §7 per-sequence tracker fix and turn the §8 WARN into a debug assertion that the tail rule is in
   force (it should now be unreachable).

## 12. The tail rule implemented at the splitter: correct, but costs 14 % of prefill — **REVERTED, superseded by §13**

**Implemented** (branch `wip/umem-arena-guard`, `LLAMA_RS_TAIL_TOKENS`, default **0 = off** until the
next step below):

* `llama_rs_tail_tokens(n_ubatch)` (`src/llama-batch.{h,cpp}`) — the fixed tail length, independent of the
  draft depth (purity), 0 when `n_ubatch` is not larger.
* `split_equal()`: when a single-sequence ubatch **finishes** its sequence and is larger than the tail,
  pop `D` tokens so the sequence's last ubatch is exactly `D` → it takes the **existing** sequential GDN
  kernel, which writes the rollback snapshots.  No kernel change, no extra memory.
* All four hybrid memories (`-hybrid`, `-idx`, `-iswa`, `dsv4`) pass `llama_rs_tail_tokens(n_ubatch)`.
* **Truncate after the fill, never break inside it**: a mid-loop break leaves `used[]` half-marked and the
  next call decays into a halving cascade (`512 -> 248 -> 124 -> 62 -> …`, measured).  With the post-fill
  rule the split is `496 + 16` per 512-token batch, clean.

### 12.1 Correctness: the bug is gone (probe)

`tools/prefill-rollback.cpp`, `k=1`, `-ub 512`, RS=3, 35B-A3B:

| arm | P=64 | 1986 | 2048 | 2050 | 4096 |
|---|---|---|---|---|---|
| tail off (today) | — | — | **0.65** | — | — |
| **tail on** | 3.1e-3 | 2.7e-3 | **7.7e-4** | **0.0** | 1.3e-3 |
| tail on + sequential GDN (control) | — | — | 4.1e-4 | — | 1.8e-3 |

The systematic staleness (o.65 NMSE) is **gone**; what remains is the width-parity rounding already
documented in §8.2 (the control reproduces the same magnitude, so it is the reference's shorter final
batch, not the restore).

### 12.2 Gates

| gate | result |
|---|---|
| `gate-prefill-logits.sh --ab LLAMA_RS_TAIL_TOKENS=0 --ctx 2048` (state must carry across ubatches — at `--ctx 512` the tail is invisible, KLD exactly 0) | **PASS**: mean KLD **0.002317**, same-top-p **98.412 %** (limits 0.005 / 98.0) |
| dense coherence 4B seed 42 | `b0e99b1137b4…` **identical** |
| `test-recurrent-state-depth` | **153 == 153** (box baseline) |
| `test-recurrent-state-rollback` | **0** failures |
| `tools/rs131-repro-1gpu.sh` | 0 warnings, ubatch sizes `496/488/492/490 + 16` |

### 12.3 Performance: **-13.7 % pp512, -3.5 % pp2048, decode flat** — wrong layer

`llama-bench` (35B-A3B Q3_K_M, q8_0 KV, `-r 3`):

| arm | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| tail off | 2092.31 | 2074.95 | 63.31 |
| **tail on (16)** | **1804.35 (-13.8 %)** | **2001.53 (-3.5 %)** | 63.24 (flat) |

The cost does **not** scale with the tail length — `D=4` 1874, `D=8` 1824, `D=16` 1806 — so it is not the
16 sequential tokens: it is (a) one extra graph per prefill ubatch and (b) every prefill ubatch shrinking
from 512 to `512-D`, which costs GEMM/MMB tile efficiency.  **The splitter is the wrong layer.**

### 12.4 Next step: the same rule inside the GDN op (one graph, full-size ubatch)

Move it into `ggml_cuda_op_gated_delta_net`: when `n_seqs == 1 && n_tokens > D`, launch the chunked
kernel on the first `n_tokens - D` (its existing `n_tokens_limit` argument) and the **sequential kernel on
the last `D` tokens** in the same launch (a token-range offset on the q/k/v/g/beta pointers, with the slot
mapping using `D`).  Then the ubatch stays 512 (no tile loss), no extra graph, and the only cost is one
D-token sequential GDN per batch — the ~0.1 % originally estimated.  Needs: the sequential launcher to take
a token offset, the input state for the tail taken from the chunked kernel's slot 0, and
`test-backend-ops -o GATED_DELTA_NET` as the oracle.

Until that lands, the splitter rule is **default off** (`LLAMA_RS_TAIL_TOKENS=0`) — correct but not
shippable at -14 %.

## 13. Final form: the tail lives INSIDE the GDN op — correct and free (2026-10-19)

**Implemented** (fork branch `wip/umem-arena-guard`; `ggml/src/ggml-cuda/gated_delta_net.cu`, one hunk):

When `n_seqs == 1 && n_tokens > D` (with `D = GDN_CHUNKED_MIN_TOKENS = max(K>16?K:16, n_rs_batch)`):

1. the **chunked** kernel runs on the prefix `n_tokens - D` (its existing `n_tokens_limit`), writing its
   final state to the snapshot slot **explicitly** — the chunked op derives the default tail offset from
   the *limited* token count, so the caller now passes `state_d_ext` = the full-batch tail base (or the
   fused cache slot);
2. the **sequential** kernel then runs on the last `D` tokens in the same launch, reading that state and
   writing snapshot slots `0..D-1` (q/k/g/beta advance by `sq2` per token, v by `sv2`, the attention
   scores by `S_v*H`; the state is read into registers before any slot is written, so aliasing the input
   state with slot 0 is safe, and slot 0 is written last).

So the ubatch stays full size, there is no extra graph, no extra memory, and no splitter change.
`GGML_CUDA_GDN_TAIL=0` restores whole-batch chunking (the A/B and the bisect knob).

The splitter variant (§12) was **reverted**: it produced the same correctness at −13.7 % pp512.

### 13.1 Gates (all run on gfx1151, 35B-A3B Q3_K_M unless stated)

| gate | result |
|---|---|
| `prefill-rollback`, tail on | P=64 **8.9e-4**, 512 **2.3e-3**, 1986 **5.9e-4**, 2048 **1.5e-3**, 2050 **0.0**, 4096 **2.5e-3** |
| `prefill-rollback`, `GGML_CUDA_GDN_TAIL=0` | P=2048 **0.65** (the wrong state, as before) |
| `test-backend-ops -o GATED_DELTA_NET` | **2/2 backends passed** |
| `gate-prefill-logits.sh --ab "GGML_CUDA_GDN_TAIL=0" --ctx 2048` | **PASS**: mean KLD **0.002389**, same-top-p **98.069 %** (limits 0.005 / 98.0) — **tight, see 13.3** |
| `--spec-type none` vs `draft-mtp n3`, 35B, seed 42, 64 tokens | byte-identical (`13497757a7bc…`) — purity held with the fixed `D` |
| dense coherence 4B seed 42 | `b0e99b1137b4…` identical |
| `test-recurrent-state-depth` | **153 == 153** (box baseline) |
| `test-recurrent-state-rollback` | **0 FAIL** |

Performance (`llama-bench`, q8_0 KV, `-r 3`):

| arm | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| `GDN_TAIL=0` | 2096.68 / 2073.30 | 2089.80 | 63.24 / 63.29 |
| **tail on** | 2091.46 / 2076.75 | 2080.42 | 63.24 / 63.22 |

**Within noise (−0.25 % … +0.17 % pp512, −0.45 % pp2048, decode flat)** — the ~0.1 % the design predicted.

### 13.2 What this closes

A rollback of up to `D` tokens is now correct **for every ordering**, including the §8 case (a rollback into
a large prefill batch), because that batch's last `D` tokens are always snapshot-written by the sequential
kernel.  The §7 per-sequence tracker fix removes the only false-positive route.  So the "rollback window"
bugbear is closed structurally, not by a bigger window: the window stays `1 + n_max` planes (240 MiB at
MTP n3), nothing grows for ngram-style drafts, and no replay/checkpoint machinery is needed.

### 13.3 The margin question, resolved: **the tail is more accurate than the arm it replaces**

The first KLD A/B (`--ab GDN_TAIL=0`) reports 0.002389 / 98.069 % — but that compares the new path against
the **unsafe** arm, i.e. it *is* the size of the chunked kernel's prefill error, not an error the tail adds.
Measured against the **sequential-everywhere (exact) reference** (`GGML_CUDA_GDN_CHUNKED=0` as the base arm,
`--ctx 2048`):

| candidate arm | mean KLD vs the fp32-sequential reference | same-top-p |
|---|---:|---:|
| **chunked prefix + fp32 tail (new)** | **−0.000000** | **99.994 %** |
| chunked whole batch (`GDN_TAIL=0`, today) | 0.002777 | 98.173 % |

So the tail removes the chunked kernel's state error for the last `D` tokens, and — because the state handed
to the *next* batch is now the fp32 sequential result — the bf16 chunked kernel's error **no longer
accumulates across batches**. Prefill fidelity is *better* than before, not worse. A `--ab
GGML_CUDA_GDN_CHUNKED_BF16=0` control (both arms fp32) still shows 0.001934 between tail and no-tail, so the
remaining delta is the two kernels' **algorithm** (per-chunk gram/KKT solve vs per-token recurrence), not
the dtype.

**Therefore `gfx1151` supporting BF16 does not help here, and a bf16 sequential kernel is the wrong lever:**
it could not be bit-comparable to the bf16 *chunked* kernel (different algorithm), it would perturb the
*decode* kernel at every width 1..8 — the purity anchor, documented as bit-identical to
`GGML_CUDA_GDN_CHUNKED=0` — and the measurement above shows there is nothing left to recover.

## 14. Packaging for the gfx1201 box (2026-10-19)

The handover is [`TESTING.md`](TESTING.md) plus two `git am`-able patches, cut against the `release.json`
r89 tree and verified to apply cleanly onto a pristine `rdna-boosts` checkout, reproducing the developing
tree byte-for-byte (`git am` on `15552e758`; both files then hash-identical to the branch):

| file | sha256 | block |
|---|---|---|
| `0001-rs-rollback-per-sequence-tracker.patch` | `5bb41d48a0969366…` | 02 |
| `0002-gdn-chunked-prefill-snapshot-tail.patch` | `fd9441e4238cd334…` | 03 |

`TESTING.md` has the apply line, the two knobs (`GGML_CUDA_GDN_TAIL`, `LLAMA_RS_ROLLBACK_DEBUG`), the eight
gates in order with the gfx1151 numbers to compare against, the `--ctx 512` trap in the KLD gate, the
promotion shape, and the independent `umem-arena-guard` patch set that shares the session but not the bug.

The old `rs-rollback-tracker.patch` was replaced by `0001` (same content, format-patch form).
