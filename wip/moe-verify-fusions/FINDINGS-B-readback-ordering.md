# FINDINGS — job B: the held batching change is correct; the perplexity break is a multi-ubatch logits-readback ordering hazard

**Date:** 2026-10-12 (follow-up to `HANDOVER-multiseq-bc.md`; WIP, NOT for delivery).
**Status:** B root-caused. B's split logic is NOT the defect. The defect is a pre-existing
(pinned) async D2H logits readback that is not ordered against later ubatches' graph compute.
A synchronization after the per-ubatch readback makes `llama-perplexity` correct again
(PPL 8.51, exactly matching B off). The same hazard is very likely C's root cause.

## TL;DR

`llama_context::decode` extracts each ubatch's logits with
`ggml_backend_tensor_get_async(..., logits_out = logits.data + n_outputs_prev*n_vocab, ...)`
(`src/llama-context.cpp`, the "extract logits" block). The output buffer is **pinned**
(`ggml_backend_dev_host_buffer_type(output_dev)`, `output_reserve`), and the *device* source
tensor (`res->get_logits()`) is **reused across ubatches** (same `t->data` pointer, verified
with `GGML_STREAMDBG`). On ROCm the async pinned D2H copy is not reliably ordered with the
following ubatch's kernels, so the copy for ubatch *i* can read the device tensor after ubatch
*i+1* (or *i+2*) overwrote it. The host rows for earlier ubatches then hold later ubatches'
logits.

- With B on, `llama-perplexity -b 2048` produces 4 single-sequence ubatches, **all output-bearing**:
  the host rows are `[seq0, seq2, seq3, seq3]` instead of `[seq0, seq1, seq2, seq3]`.
- Adding `ggml_backend_synchronize(backend_res)` after the per-ubatch copies (the WIP
  `LLAMA_DBG_ROW=1` probe) makes every row correct and **PPL = 8.5100** (B off = 8.5100).
- Forcing the output buffer **pageable** (`GGML_NO_PINNED_OUTPUT=1`) also makes PPL = 8.5100:
  HIP's pageable D2H copy is synchronous, so the extra sync is unnecessary in that arm. This
  isolates the pinned async copy as the trigger.

**The B split logic is therefore sound** — the probe's "bit-identical solo vs co-batch" and the
per-sequence forward (device tensors, via a synced dump) are both correct. What breaks is the
host readback.

## Evidence

### 1. Perplexity (Qwen3.5-4B-Q8_0, wikitext, `-c 512 -b 2048 --chunks 4`, canonical r42 tree)

```
B on,  default (pinned, no sync)          [1]5.9840,[2]2093.6111,[3]13352.6339,[4]2097.7516  PPL 2097.75
B on,  LLAMA_DBG_ROW=1 (sync after copy)  [1]5.9840,[2]8.4928,[3]8.6391,[4]8.5100            PPL 8.5100
B on,  GGML_NO_PINNED_OUTPUT=1            [1]5.9840,[2]8.4928,[3]8.6391,[4]8.5100            PPL 8.5100
B off, default                            [1]5.9840,[2]8.4928,[3]8.6391,[4]8.5100            PPL 8.5100
```

### 2. Controlled probe `tests/test-multiseq-same.cpp` (WIP; four sequences, distinct tokens,
one `llama_process`, all tokens output; reference = a fresh context per sequence)

With B on (`-ub 512`, 4 ubatches of 64 tokens, every ubatch output-bearing):

```
                   device dump* (synced)     host readback (no sync)     expected (solo)
seq0  pos63        24c8bfdd3aef3620          24c8bfdd3aef3620            24c8bfdd3aef3620
seq1  pos63        31ff116ee801ac97          7d8ec964e049925f  (= seq2)  31ff116ee801ac97
seq2  pos63        7d8ec964e049925f          c75bd9f74265edd5  (= seq3)  7d8ec964e049925f
seq3  pos63        c75bd9f74265edd5          c75bd9f74265edd5             c75bd9f74265edd5
```
\* `LLAMA_DBG_LOGITS=1` syncs the compute stream before reading the device tensor.

The `OUTIDS` mapping is correct (`output_ids[k]=k`), so this is **not** the `out_ids`/output-swap
path — the host rows themselves hold the wrong ubatches' data.

### 3. The hazard is general, not B-specific

Single sequence, `MSQS_SEQ=1 MSQS_TOK=512 MSQS_UB=128 MSQS_AT=300` (4 ubatches, all output):

```
no sync       argmax=279   hash=092225b1ed278661   (wrong)
ROW sync      argmax=33075 hash=812352323fabba8d   (correct)
```

So any decode with several output-bearing ubatches is exposed; B just makes `-b 2048` perplexity
produce that shape. `GGML_MULTI_SEQ_UBATCH=0` with the default `-ub 512` happens to produce only
two output-bearing ubatches (perplexity requests logits only for `pos >= n_ctx/2`) and dodges it.

Note: the pure single-sequence, 2-output-ubatch `llama-perplexity -b 512 -ub 128` is correct in
both the pinned and pageable arms — the exact timing/size boundary of the copy-engine reorder is
an open question; the fix below does not depend on it.

## Fix options (in preference order)

1. **Order the readback (minimal).** After the per-ubatch logits/embd async copies, make the
   device tensor safe to overwrite: `ggml_backend_synchronize(backend_res)` (or an equivalent
   event/stream wait). Because the next ubatch overwrites the *same* device tensor, this
   serialization is semantically required anyway on a correct stream; it is not new work, it is
   recovering ordering that ROCm's pinned async copy is not providing. Guard it so a
   single-ubatch decode (the common case) is untouched. Must be validated for perf on the
   rule-5 `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` gate and on `pp`-heavy paths.
2. **Double-buffer the device logits** per ubatch (a distinct graph output tensor / staging
   buffer), so the copy can overlap the next compute without a sync. Larger change; touches the
   graph result plumbing and the scheduler's compute-buffer reuse.
3. **Pageable output** (`GGML_NO_PINNED_OUTPUT` semantics). Correct and simple, but gives up the
   pinned fast path for every logits readback — measure before considering.

Option 1 is the recommended first cut: it is one code site, matches the intent of the existing
loop, and can be A/B'd exactly as above.

## Relationship to job C (hypothesis)

C's green configuration is "no cross-sequence ubatching + `--no-kv-unified`" — i.e. **one
output-bearing ubatch per decode**. Every non-green C arm lets a decode carry more than one
output-bearing ubatch. That is exactly the condition this hazard needs, and the hazard is
independent of MTP, the snapshot path, the expert cache and the all-reduce — matching C's
independence matrix. The determinism difference (C varies between repetitions, this probe is
deterministic per arm) is explainable: the reorder outcome depends on the copy-engine/stream
timing, which varies with co-resident work and across repetitions. **Validate C's
`tools/multiseq.sh` after fixing the readback** before treating C as a separate unified-KV defect.

## Repro files

- `wip/moe-verify-fusions/tools/test-multiseq-same.cpp` — the same-input, per-sequence probe
  (`MSQS_SEQ`, `MSQS_TOK`, `MSQS_UB`, `MSQS_AT`, `MSQS_REF_UB`, `MSQS_UNI`). Build by copying to
  `tests/` and adding `llama_build(test-multiseq-same.cpp)` to `tests/CMakeLists.txt`.
- Instrumentation used: `LLAMA_DBG_ROW=1` (the sync after each host copy), `LLAMA_DBG_LOGITS=1`
  (synced per-ubatch device dump), `LLAMA_DBG_UB=1` (ubatch structure), `LLAMA_DBG_MAP=1` /
  `LLAMA_DBG_OUTIDS=1` (out_ids/output_ids). All are WIP-only probes; the durable form is the
  fix, not the probes.

## Not done / next

- The exact ROCm copy-engine condition (copy size vs next-compute duration) is not pinned.
- Perf of fix option 1 is not measured; the prefill-logit KLD gate was not re-run with the fix.
- C's `tools/multiseq.sh` was not re-run with the fix (needs the GSQ model + 30k prompt).
- No delivery change made. Awaiting the maintainer's decision on the fix location (fork delivery
  vs `upstream/` PR candidate).

---

## UPDATE (same session) — Option 1 implemented and validated; C is NOT this bug

**Fix:** `readback-sync-fix.patch` (in this directory). `llama_context::decode`'s ubatch loop now
calls `ggml_backend_synchronize` on the backends of the tensors it just read asynchronously, but
**only when another ubatch follows** (`if (!mctx->next()) break;` before the sync). A single-ubatch
decode pays nothing. `GGML_READBACK_SYNC=0` is the A/B kill-switch.

**Validation (canonical r42 + held-B + fix unless noted):**
- B on perplexity `PPL = 8.5100` (was 2097.75); B off unchanged at 8.5100.
- `test-multiseq-same` bit-identical solo vs co-batch, n_seq=4 tok=512.
- `scripts/gate-prefill-logits.sh` on 27B Q8_0: **mean KLD 0.000707, same-top-p 98.755 % — PASS**,
  byte-for-byte the r42 baseline (B no longer fails the gate).
- `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (4B Q8_0): TG unchanged; ~2 % PP cost only in
  B's own multi-sequence prefill path (every ubatch output-bearing); the clean r42+fix perplexity is
  unchanged.
- Clean patch compiles standalone on `a3c13e9ca` (`/tmp/fixclean`); r42 perplexity 8.5100.

**C is a different defect.** Server A/B, GSQ + MTP, `GGML_FORCE_SEQ_UBATCH=1`, `-np 2`:
- `--no-kv-unified` → **fully green** (A_solo == conc1 == conc2 == conc3, first-diff None).
- `--kv-unified` → divergent (A_conc1/2/3 and A_solo all differ; first-diff at token 9/9/65).
- The divergence is unchanged by the readback fix (`GGML_READBACK_SYNC=0` == `=1`), by
  `LLAMA_KV_N_PAD_MIN=65536`, and by `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1` (byte-identical
  hashes) — so it is not the readback, not `n_kv` padding, not the mmvq band. The `n_kv`
  pin test also means it is not simply the co-resident sequence extending `n_kv`.
- In-process probe (`test-multiseq-same`, unified, up to 512 tokens) is bit-exact; the server needs
  the real scale/timing. Root cause not yet pinned; C remains open and is unified-KV-specific.
