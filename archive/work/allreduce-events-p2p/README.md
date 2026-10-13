# allreduce-events-p2p: no per-AR `ev.ker` on the kernel path, and an opt-in peer-to-peer copy-engine path (bit-identical)

Two format-patches on top of `v16-097cf33d9-r50`, applied in order with `git am` (both also apply cleanly on r49).
Both change only `ggml/src/ggml-cuda/allreduce-hip.cu`.

## 0001: the chunked kernel path no longer records `ev.ker` per all-reduce

`acquire_slot` synchronizes the host on `ev_pool[*][slot].ker` from the all-reduce that last used the slot.  The
chunked kernel path recorded that event after every all-reduce on every device.  On ROCm each `hipEventRecord` is a
marker packet that the GPU processes after the AR kernel, before the next graph launch, so it adds a gap after every
AR.  In our trace, gaps of 5 us or more per MTP verify step went from 2.07 to 1.47 ms once the record was removed
(roughly 7 us per AR).

On the kernel path, slot reuse is already ordered on the GPUs.  A device's kernel for AR N completes only after it has
seen the peer's arrival token for N (except on a spin timeout, covered below).  The peer writes that token in its own
kernel N, which starts only after its kernel N-1 has finished reading this device's slot data.  The tokens are monotonic, so a stale arrival cannot satisfy
a later wait.

So the kernel path now records `ev.ker` only when `GGML_CUDA_AR_KER_EVENTS=1` is set or `GGML_CUDA_AR_PACE` is on.
`acquire_slot` syncs only when the slot's last user recorded the event (`ker_recorded[slot]`).  The copy-engine path
still records it, unchanged.

One behavioural difference: the host is no longer held to at most 2 ARs ahead of the GPU on the kernel path.  After a
spin timeout the result is already wrong either way, and the existing fallback still resyncs.  The only change there is
that the timeout may be noticed a little later.  `GGML_CUDA_AR_KER_EVENTS=1` restores the old behaviour exactly.

## 0002: `GGML_CUDA_AR_P2P_LARGE=1`: the copy-engine path pulls the peer's half directly (opt-in)

Prefill all-reduces (~10 MB at `-ub 2048` for Flash-Next) take the copy-engine path.  That path bounces both halves
through pinned host memory (D2H, then H2D on the peer), so every byte crosses PCIe twice.  With the env var set:

1. Each device's AR stream waits for its compute (zeroing the buffer if the shard is inactive) and records `cpy[0]`.
2. Each device waits for the peer's `cpy[0]` and for the previous add's `dev_tmp_kernel_done`, then pulls the peer's
   contribution into `dev_tmp` with `hipMemcpyPeerAsync`.
3. Each compute stream waits for its own pull and the peer's pull (the peer's pull reads this device's buffer, so it
   has to finish before this device's add can overwrite it).  Then the same add kernel runs.

Same add kernel on the same bytes, so the output is bit-identical.  Unset or 0: unchanged.  This is independent of the
existing `GGML_CUDA_AR_P2P` (#129), which switches the kernel path, not the copy-engine path.

A 10.5 MB exchange between our two cards takes 808 us via host against 437 us peer-to-peer (peer access on or off
made no difference).

## Measurements (2 x R9700 32 GB, gfx1201, PCIe 5.0 x8 each)

Stock r50 + these two patches (plus the topk patch), Flash-Next GSQ-RCO IQ3_XXS + MTP (n-max 3), `-sm tensor`, all
experts in VRAM.  Same binary with env toggles, alternated, 2 runs each, after one warm-up request.  "Before" sets
`GGML_CUDA_AR_KER_EVENTS=1 GGML_CUDA_TOPK_SORTED=0`:

| | before | 0001 + topk (defaults) |
|---|---:|---:|
| greedy 400 tokens, t/s | 139.5 / 141.1, 139.2 / 141.0 | 143.2 / 144.9, 143.5 / 144.8 |
| greedy sha | `a7a81f01b3bc` | `a7a81f01b3bc` |

That is about +2.8 % for 0001 and the topk patch together.  On r49, 0001 alone (topk unchanged) was 136.2 -> 138.9 t/s
(+2.0 %), same sha `c39d78416cd1`.

For prefill, on an r49 build with these patches (plus two local scheduler-event patches of ours, the same in both
arms), a 31.5k-token prompt went from 2634 to 2668 t/s with `GGML_CUDA_AR_P2P_LARGE=1` (+1.3 %, 2 runs each,
alternated).  The r50 run measured it only together with the MTP prompt-tail change we're sending separately.

No faults and no AR timeouts in any run.
