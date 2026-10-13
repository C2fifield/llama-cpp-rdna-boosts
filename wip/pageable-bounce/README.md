# pageable-bounce: copy pageable host tensors through a pinned chunk (faster prompt-cache save, output unchanged)

One format-patch on top of `v12-097cf33d9-r52`, for `git am`.  It touches `ggml/src/ggml-cuda/ggml-cuda.cu` and adds
five pointer-attribute aliases to `ggml/src/ggml-cuda/vendors/hip.h`.

## What we saw

On ROCm, a copy between the device and *pageable* host memory is slow at every size.  A large one runs at about
2 GB/s through the runtime's own staging.  A small one still costs about 25 us (1-D) or 57 us (2-D) on gfx1201.

The server makes a lot of these copies:

- After every prompt it saves a context checkpoint: ~118 MB of qwen4exp recurrent state, in 512 KiB tensors.
- When a new request takes a slot, it saves that slot's state to the server's `--cache-ram` prompt cache.  That reads the KV and
  recurrent state as thousands of small row ranges.  Under `-sm tensor` there are twice as many, as one 2-D copy per
  device.

So the first short request after a few concurrent chats waited about 3 s before its prompt started.

## The patch

- One reused 64 MiB pinned chunk per device.
- `buffer_get_tensor`, `buffer_set_tensor` (not for weight buffers, so model loading is unchanged) and
  `buffer_get_tensor_2d` copy through that chunk when the host pointer is pageable.
- A pinned host pointer (`cudaPointerGetAttributes` says host, device or managed) keeps the direct copy.
- If the chunk can't be allocated, the copy stays direct, and the allocation isn't retried on every call.
- `GGML_CUDA_BOUNCE_MIN=<bytes>` sets the threshold.  The default is 1 (every size); `0` turns it off.

## Measurements (2 x R9700 32 GB, gfx1201)

Flash-Next IQ3_XXS + MTP (n-max 3), all experts in VRAM, `-sm tensor`, 4 slots, `-cram 2048`.  After 6 concurrent
400-token chats, five sequential short new requests (each one first saves an evicted slot's state).  No tuning env,
one run each:

| | r52 | r52 + this patch |
|---|---:|---:|
| first ~3.9k-token request (incl. the checkpoint save), wall ms | 6422 | 2521 |
| first short request after the chats, wall ms (server prompt ms) | 2407 (476) | 1715 (337) |
| next four short requests, wall ms | 442, 313, 313, 315 | 238, 214, 213, 213 |
| the ~3.9k request repeated at the end, wall ms | 2929 | 2511 |

The greedy sha (`a7a81f01b3bc`) and the repeated request's output (`4ee69a9c85e7`) are the same in both builds.

On our earlier local build the same test gave 3083 ms -> 1470-1922 ms, and 6 concurrent streams went
122.8 -> 136.2 t/s, because the slot saves no longer stall the other streams.  Single-stream decode and prefill were
unchanged.  A prompt restored from the RAM cache gave the same output as a fresh run.

## Notes

- The bounce is synchronous, like the copies it replaces.  The extra host `memcpy` is much cheaper than the
  pageable-copy overhead at every size we measured, which is why the default is every size.
- `-sm layer` gains less (its saves are 1-D), but it uses the same path.
- The chunk lives for the process (64 MiB pinned per device that uses it).
- The pointer-attribute lookup runs once per copy; it is included in the times above.
- MUSA: `vendors/musa.h` doesn't get the five pointer-attribute aliases, so a MUSA build would need them.  We can't
  build MUSA here, so we left it out rather than add untested defines.

No urgency on our side.  Happy to re-run anything on your config, or rework it if you'd prefer it done differently.
