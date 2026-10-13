# RESOLUTION — mtp-prefill-tail (PR #143)

**Status:** FOLDED into the delivery as **block 02** (`common/speculative.cpp`,
`common/speculative.h`, `tools/server/server-context.cpp`, r52, 2026-10-13); `0001` **opt-in**, `0002`
**always on**.  **Owner:** `@briansp2020` (PR #143).

## What was adopted

- `0001-speculative-LLAMA_MTP_INGEST_TAIL-tokens-feeds-the-M.patch` — opt-in
  `LLAMA_MTP_INGEST_TAIL=<tokens>`: the MTP draft ingests only the last N positions of each prompt.
  A new `common_speculative_set_prompt_end()` (a virtual no-op for the other draft types; the server
  calls it via `slot.task->tokens.pos_next()`) records each prompt's end, `draft_mtp::process()` skips
  the prompt rows before it, and the prompt phase ends at the first generation row.  The target
  verifies every drafted token, so the output is unchanged; only draft look-back is lost.
- `0002-speculative-draft_mtp-copies-only-the-last-target-ro.patch` (always on) — for prompt batches
  `draft_mtp` copies only the last target hidden-state row into `verify_h` instead of every row
  (~84 MB per 2048-token ubatch on qwen4exp).  `accept()` only follows verify batches, which refill
  `verify_h` themselves; callers that never report a prompt end keep the full copy.

The tail is a **tuning** value, not a kill-switch: it trades prefill speed for draft acceptance, so it
stays opt-in (unset/0 = every row).

## Validation on the delivery tree (gfx1201, 3x R9700)

- 35B-A3B draft-MTP (`-sm tensor -c 16384 -n 3000`, bf16 KV, prose): acceptance **0.733** identical to
  r51, **160.4** vs **159.2 t/s** — the always-on `0002` is neutral-to-positive, no regression.
- 35B-A3B long prompt (prose x6, ~31k tokens): default acceptance **0.82081** (identical to r51),
  prefill 2933 vs 2915 t/s; `LLAMA_MTP_INGEST_TAIL=4096` measured prefill **+4.8 %** (3074 vs 2933 t/s)
  but acceptance **0.821 -> 0.715** and decode **-4.4 %** (145.6 vs 152.3 t/s).  The trade-off is real,
  so the variable stays opt-in.
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS).
