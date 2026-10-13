# RESOLUTION — allreduce-events-p2p (PR #142)

**Status:** FOLDED into the delivery as **block 01** (`ggml/src/ggml-cuda/allreduce-hip.cu`, r52,
2026-10-13); `0001` **default-on**, `0002` **opt-in**.  **Owner:** `@briansp2020` (PR #142).

## What was adopted

- `0001-allreduce-hip-chunked-kernel-path-no-longer-records-.patch` — the chunked kernel path no longer
  records `ev_pool[*][slot].ker` after every all-reduce (on ROCm each record is a marker packet the GPU
  processes after every AR kernel).  Slot reuse on that path is already ordered on the GPUs by the
  arrival tokens, so `acquire_slot` now syncs only when the slot's last user recorded the event.
  `GGML_CUDA_AR_KER_EVENTS=1` (or `GGML_CUDA_AR_PACE=1`) restores the per-AR recording.  The author
  flags the one behavioural difference: without the per-AR sync the host can run more than 2 ARs ahead,
  so a spin timeout surfaces a little later (the output is wrong either way, and the existing fallback
  still resyncs).
- `0002-allreduce-hip-copy-engine-path-can-pull-the-peer-s-h.patch` — opt-in
  `GGML_CUDA_AR_P2P_LARGE=1`: the copy-engine path pulls the peer's half with `hipMemcpyPeerAsync`
  instead of bouncing both halves through pinned host memory (one PCIe crossing instead of two).  Same
  add kernel on the same bytes, so bit-identical.  Kept **opt-in** (the 2-device copy-engine path; the
  reporter measured +1.3 % on a 31.5k prefill).

## Validation on the delivery tree (gfx1201, 3x R9700)

- 35B-A3B draft-MTP (`-sm tensor -c 16384 -n 3000`, bf16 KV, prose): acceptance **0.733** identical to
  r51, **160.4** vs **159.2 t/s**; plain **106.5 t/s** (MTP +50 %).  No AR timeouts, no faults.
- 35B-A3B long prompt (prose x6, ~31k tokens) default: acceptance **0.82081** identical to r51, prefill
  2933 vs 2915 t/s.
- `gate-prefill-logits.sh`: mean **0.000707** / same-top-p **98.755 %** (PASS).
