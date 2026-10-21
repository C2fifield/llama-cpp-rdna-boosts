# FINDINGS 2026-10-20 — issue #151 follow-up (TODO #67) CONFIRMED external: the graphs-on flip is hipBLASLt

**Status: CLOSED 2026-10-20.**  TODO #67 is removed from `TODO.md`; the closure is recorded in
`WORKLOG.md` 261020.012.  The reporter ran the bisection the item asked for and it confirms the working
hypothesis on file: the start-to-start decode flip is the **Windows/ROCm hipBLASLt solution-selection
issue** (`ROCm/rocm-libraries#12126`), **not** HIP graphs and **not** this delivery.

Source: `@DanoPTT`, issue #151 comment
[6216586361](https://github.com/stew675/llama-cpp-rdna-boosts/issues/151#issuecomment-6216586361)
(2026-10-16), quoted here because it is the evidence that closes the item.

## The reporter's matrix

Box: R9700, **Windows 11**, Ryzen 5 PRO 4650G, r68 delivery + the reporter's 9 carry patches.  Model: Swift
1.5 Q5_K_M, q8_0 KV, `-c 204800 -np 2 --no-kv-unified`, `draft-mtp` n3, `ASYNC_INPUT=0`.  Each start is a
fresh server with one request on slot 0: the same ~100 607-token prompt, greedy, 128 tokens, `n_probs 5`,
hashing the **top-5 log-probs at every position** (decode logits, not text).

| arm | starts | logit classes |
|---|---|---|
| graphs on, `ROCBLAS_USE_HIPBLASLT=0` | 4 | **1** (`6BB6E39E3CFB`) |
| graphs on, `ROCBLAS_USE_HIPBLASLT=0`, `DISABLE_ROPE_SET_ROWS=1`, `DISABLE_RMS_NORM_MUL_ROPE=1` | 3 | 1 (same `6BB6…`) |
| graphs on, hipBLASLt **on** | 7 | 3 (`70EB…` x4, `E46C…` x2, `89D8…` x1) |
| graphs **off**, hipBLASLt **on** | 6 | 2 (`89D8…` x5, `70EB…` x1) |

Reading it:

* With `ROCBLAS_USE_HIPBLASLT=0` all 7 starts are bit-identical, and the two rope switches change nothing
  (so r25's rope bit-transparency fix holds on that platform).
* With hipBLASLt on, the same classes appear **with graphs on and off**, so graphs play no part — the
  earlier graphs-off control had simply landed on one class by luck.
* In every class the top-5 already differs at position 0 (the first token after the prefill), which is the
  signature of a prefill-side GEMM solution choice, matching rocm-libraries#12126.
* The 128-token **text** was identical on all 20 starts in that run; the earlier Swift 1.0 text flip is most
  likely the same mechanism landing on a closer near-tie.

The reporter also re-ran the async-input KL self-compare in the same comment (`llama-perplexity -c 512
-b 2048 -ub 8`, n_seq 4, 32 chunks, 27B UD-Q5_K_XL, q8_0 KV, graphs off, hipBLASLt off): no divergence
between `ASYNC_INPUT=1` and the base run in mean/max KLD, RMS Δp, same-top-p or PPL ratio — i.e. the async
ring is not implicated either.

## What this means for the delivery

Nothing to fix.  `ROCBLAS_USE_HIPBLASLT=0` is the workaround, and it is the *known-good* arm here: the
delivery has never set `ROCBLAS_USE_HIPBLASLT`, and its Linux gates are run with rocBLAS.  The r25
`rope -> view -> set_rows` `#pragma clang fp contract(off)` fix stands on its own (it was a real
bit-transparency bug); this is the separate, external residual.

Cross-reference for the next reader: `README.md` -> "Cross-start determinism on ROCm (issue #67)" carries the
consumer-facing version of this, and `ENVIRONMENT.md` notes the same layout-sensitive hipBLASLt divergence
in the KV-cache layout context.  Both are now marked closed/explained rather than tracked.
