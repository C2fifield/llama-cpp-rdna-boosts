# mtp-prefill-tail: less MTP draft work during prompt processing (output unchanged)

Two format-patches on top of `v16-097cf33d9-r50`, applied in order with `git am`.  They touch `common/speculative.cpp`,
`common/speculative.h` and `tools/server/server-context.cpp`.  0001 is rebased over upstream's vision-token handling in
`draft_mtp::process()`, and 0002 uses the prompt end that 0001 records, so apply both.

## 0001: `LLAMA_MTP_INGEST_TAIL=<tokens>`: the MTP draft ingests only the tail of each prompt (opt-in)

The MTP draft has its own KV cache, so during prefill every prompt row also goes through the draft layer.  On a 32k
prompt that cost us about 12 % of prefill throughput on our local build.  With the env var set:

- The server tells the speculative state where a new prompt ends (`tokens.pos_next()`).  A new
  `common_speculative_set_prompt_end()` is called when a slot moves to `SLOT_STATE_PROCESSING_PROMPT`; the base `common_speculative_impl` gets a virtual no-op,
  and only `draft_mtp` overrides it.
- `draft_mtp::process()` skips prompt rows before `n_prompt - tail`.  The first kept row still pairs with the target
  hidden state of the row before it.  For a batch where every row is skipped, `pending_h` is still updated.
- The prompt phase ends at the first generation row (pos >= n_prompt), so a later context shift can't make the draft
  skip generation rows.

The draft only loses look-back over the skipped prefix.  The target verifies every drafted token, so the output is
unchanged; the only possible cost is draft acceptance on long prompts.  Unset or 0: unchanged.

## 0002: `draft_mtp` copies only the last target row for prompt batches

`process()` copied every row's next-token hidden state into `verify_h`.  For qwen4exp, `n_embd` is 10240 floats, so a
2048-token prompt ubatch copied about 84 MB.  `accept()` only ever follows a verify batch, whose own `process()`
refills `verify_h`, so a prompt batch needs only its last row (the next `pending_h`).  A sequence's rows are trimmed only
while its last row is still inside the prompt reported by `common_speculative_set_prompt_end()` (from 0001), so verify
batches of any length (for example ngram-mod's 64-token drafts) keep every row, and callers that never report a prompt
end keep the full copy.  The prompt phase also ends here, so the shared-memory draft path (which skips the ingest loop)
ends it too; we have not tested that path.  This one is always on; it does not need the env var.

## Measurements (2 x R9700 32 GB, gfx1201)

Flash-Next GSQ-RCO IQ3_XXS + MTP (n-max 3), `-sm tensor`, all experts in VRAM, a 31,537-token prompt after one
warm-up prompt, 2 runs per config, alternated:

| | prefill t/s |
|---|---:|
| r49 build with these patches (*), tail unset | 2644 / 2637, 2623 / 2634 |
| same, `LLAMA_MTP_INGEST_TAIL=4096` | 2886 / 2890, 2888 / 2892 (+9.6 %) |
| r50 + these patches + topk + allreduce-events-p2p (0001 default, 0002 off) | 2602 / 2602, 2560 / 2561 |
| same + `LLAMA_MTP_INGEST_TAIL=4096` + `GGML_CUDA_AR_P2P_LARGE=1` | 2893 / 2878, 2882 / 2869 (+11.6 %) |

(*) The r49 build also had the topk and all-reduce patches and two local scheduler-event patches of ours, the same
in both arms; only the env var differs.  The r49 rows used an earlier 64-row rule for 0002; the r50 rows are this
version of both patches.

The greedy sha is unchanged in every run (`c39d78416cd1` on r49, `a7a81f01b3bc` on r50).  Greedy decode with the tail
set: 138.6-139.1 vs 138.7-139.3 t/s without it (r49).

Draft acceptance on two natural-text prompts with a 4096-token tail, on our local build: 0.763 -> 0.739 at 9.4k
tokens, and 0.626 -> 0.626 at 32.6k.  (The greedy runs above use a short prompt, so the tail does not apply to them.)
Tails of 4096 and 8192 gave the same prefill speed.

## Notes

- The tail value is a trade between prefill time and how far back the draft can attend.  4096 was the point where
  prefill stopped improving for us; we have not tuned it beyond that.
- Only the server reports prompt ends, so in other tools the env var has no effect and 0002 keeps the full copy.
