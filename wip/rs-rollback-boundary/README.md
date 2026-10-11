# `rs-rollback-boundary` — the recurrent rollback-boundary guard and the "large prefill as the last prepared batch" ordering (issue #131)

**Status: OPEN (seeded 2026-10-13).** Not part of the delivery; nothing here may be applied to the
fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). The
per-sequence guard fix itself has been folded into the delivery (block 02); this campaign is the
**remaining** investigation, which is not a delivery correctness fix.

> **Cold start.** Read this file in full, then `AGENTS.md` ("Report findings before shipping",
> "WIP and promotion rules", "Default-on policy"), then issue #131 and the predecessor records
> [`archive/work/moe-verify-fusions/FINDINGS-single-seq-rollback.md`](../../archive/work/moe-verify-fusions/FINDINGS-single-seq-rollback.md)
> and [`HANDOVER-single-seq-rollback.md`](../../archive/work/moe-verify-fusions/HANDOVER-single-seq-rollback.md).
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

The maintainer's current position (2026-10-13): the warning is a benign timing/scheduling artefact and
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

## 2. What was measured (2026-10-13, r43, `/home/stew675/kv-offset-r43`)

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
- Prompts: `/tmp/issue131-A.txt` (24000 chars) and `/tmp/issue131-B.txt` (40000 chars), both cut from
  `../kv-cell-offset-nondeterminism/prompts/mixed30k.txt` (sha256 in that directory's README).
- The MoE firing reproducer is `archive/work/moe-verify-fusions/tools/multiseq.sh` (GSQ + shared MTP
  head, 2 GPUs, `-ncmoe 48`).
- WIP instrumentation (in the `/home/stew675/kv-offset-r43` worktree, branch `wip/kv-cell-offset`,
  **not** committed to the delivery): `LLAMA_RS_ROLLBACK_DEBUG=1` prints `RSPREP` (per prepare, with
  `n_seq_tokens`, `n_seqs`, `n_seqs_unq`, `seq_ids`) and `RSROLL` (per partial rollback, with the global
  and per-sequence last-batch counts). The per-sequence tracker is in `src/llama-memory-recurrent.{h,cpp}`.
- Build: `BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  in the worktree, then `cmake --build build-rocm-hybrid --target llama-server -j 16`.

## 5. Gates to keep green

`scripts/validate-set.sh`; coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42);
`scripts/gate-prefill-logits.sh` (KLD <= 0.005 / same-top-p >= 98 %);
`test-recurrent-state-depth` (baseline 130 failures) and `test-recurrent-state-rollback`;
`MUL_MAT_ID` 931/931; `FLASH_ATTN_QSA` >= 26/26;
`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (rule 5, decode must not regress). The dense 27B
repro above is a reproducer, not a gate.

## 6. Pointers

- Issue #131 (closed 2026-10-13 as not reproducible on the current tree; reopen if the reporter shows a
  real case). Issue #134 / TODO #52 for the separate unified-KV cell-offset residual.
- `archive/work/moe-verify-fusions/FINDINGS-single-seq-rollback.md` (§7 already separates this from
  #50/#51/#52) and `HANDOVER-single-seq-rollback.md` §6.
- `archive/work/gdn-rs-rollback/` (the snapshot-bound work).
- `TODO.md` #52 (the `seq_rm` boundary warning is described there as a false positive).
- The delivery fix: `patches/0002-rdna-boosts-block-02-fused-chunked-gated-delta-net-pre...patch`
  (per-sequence guard); record in `WORKLOG.md` for the release that carries it.
