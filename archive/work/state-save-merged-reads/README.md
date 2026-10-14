# state-save-merged-reads: read a state save's tensor ranges as merged spans (faster slot saves, output unchanged)

One format-patch on top of `v12-097cf33d9-r52`, for `git am`.  It touches `src/llama-context.cpp` only.

## What we saw

When a new request takes a slot, the server first saves that slot's sequence state to the `--cache-ram` prompt
cache (`llama_state_seq_get_data_ext`).  `llama_io_write_host` collects every `write_tensor` and flushes each one in
its destructor as its own synchronous `ggml_backend_tensor_get`.

Under concurrent decoding, a sequence's KV cells are interleaved with the other slots' cells a few tokens at a time.
So the cell ranges of one sequence are many and short.  We counted, with a debug build: a slot save after 6
concurrent chats on 4 slots was 8,000-10,000 reads over 109 tensors, taking 340-400 ms (~40 us per read on ROCm).

## The patch

In the `llama_io_write_host` destructor, consecutive reads of the same tensor whose gaps are at most
`LLAMA_STATE_GET_MERGE_GAP` bytes (default 256 KiB) become one `ggml_backend_tensor_get` of the span, and the pieces
are copied out on the host.  If the pieces are contiguous in both the tensor and the output, the span is read straight
into the output.  `LLAMA_STATE_GET_MERGE_GAP=0` keeps the old per-range reads.

The bytes written to the state buffer are the same.  Every merged span starts and ends on a piece boundary, so
wherever a piece read passes the meta buffer's row checks under `-sm tensor`, the span does too.  If the host buffer
for a span can't be allocated, that run falls back to per-range reads (the destructor must not throw).

With the same debug build: 109 gets per save (one per tensor), 25-65 ms.

## Measurements (2 x R9700 32 GB, gfx1201)

Flash-Next IQ3_XXS + MTP (n-max 3), all experts in VRAM, `-sm tensor`, 4 slots, `-cram 2048`, one binary with
`LLAMA_STATE_GET_MERGE_GAP=0` vs the default.  The e2e test runs 6 concurrent 400-token chats, then short new requests
(the first one has to save an evicted slot first).

r52 + this patch, no tuning env, one run each:

| | merge off | merge on |
|---|---:|---:|
| first short request after the chats, wall ms (server prompt ms) | 3120 (392) | 975 (360) |
| later short requests, wall ms | 285, 251, 249, 249 | 273, 243, 244, 245 |
| 4 concurrent 300-token chats, t/s | 150.0 | 144.0 |
| 6 concurrent 300-token chats, t/s | 116.5 | 134.9 (+15.8 %) |

On our local build (two rounds each): first short request 2168 / 2137 -> 661 / 668 ms, and 6 concurrent chats
140.3 / 137.9 -> 162.1 / 155.9 t/s.  6 chats on 4 slots means some requests have to save a slot first, so the faster
save shows up in throughput; 4 chats were within noise.

Greedy output (`a7a81f01b3bc`) and the e2e request output (`4ee69a9c85e7`) are the same in every run.

## Notes

- The gap bound limits how much extra is read per span: at most 256 KiB between two pieces, and never more than the
  tensor itself.  The scatter uses one host buffer of the largest span's size, reused within a save and freed at its
  end; when one sequence's cells spread across the whole cache, that can reach the size of a per-layer K or V tensor.
- The restore side (`llama_io_read_host`) still sets each range on its own.  A restored sequence gets fresh cells,
  which are usually one range per tensor, so we left it.
- Separate from this patch, in case it's useful: with qwen4exp (Flash-Next), a prompt-cache entry also carries copies of the
  slot's context checkpoints (~113 MiB each; 2 for a short chat, 12 for a 21.6k-token prompt), so entries reach
  ~1.8 GB, and `prompt_cache.alloc` spends 50-300 ms copying them.  At `-cram 2048` that leaves room for one entry.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd prefer it done differently.
