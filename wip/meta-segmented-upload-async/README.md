# `meta-segmented-upload-async` — make the segmented per-op expert upload asynchronous (perf follow-up to r54)

**Status: OPEN (seeded 2026-10-13), perf follow-up.**  Not part of the delivery.  This is the
performance debt of the correctness fix shipped in `v12-097cf33d9-r54`
([`archive/work/meta-segmented-expert-upload/RESOLUTION.md`](../../archive/work/meta-segmented-expert-upload/RESOLUTION.md)).

> **Cold start.**  Read `AGENTS.md` ("Report findings before shipping", the WIP/promotion rules, the
> Meta invariants under "Critical facts"), then that resolution record, then the code sites below.

## 0. The problem

r54 made gemma4 + host-resident experts (`-ncmoe`/`-cmoe`) + `-sm tensor` **correct**: the scheduler's
per-op upload of a pruned `ffn_gate_up_exps` group (row-aligned, with the `copy_experts`
`min(expert_size,512)` MMQ tail pad) is routed to the **synchronous** `set_tensor`, which places each
device's segmented slice and handles the tail.  That is correct but blocking:

| config | prompt t/s | gen t/s |
|---|---:|---:|
| gemma4 26B-A4B `-ncmoe 40 -sm tensor` (cache-on) | ≈9 | ≈9 |
| the same with the cache off | ≈31–45 | ≈31 |
| gemma4 `-sm layer -ncmoe 40` | ≈? | ≈? |

Blast radius: only gemma4 MoE + `-ncmoe`/`-cmoe` + `-sm tensor`.  Every other arch's expert tables are
single-segment and use the existing contiguous splice; all-resident and `-sm layer` are unaffected.
The config was previously rejected, so r54 is still a strict improvement — but the sync path is far
below the delivery's usual bar.

## 1. Goal

Make the row-aligned segmented per-op upload async again — per-row/1-D `ggml_backend_tensor_set_async`
on the simple devices (or a device-side segmented gather), with the meta backend's wait/event so the
H2D overlaps the compute that follows.

## 2. Why the earlier attempt hung (do not repeat)

An earlier partial fix (archived in `archive/work/moe-expert-cache/item3-findings-session20.md`) did
exactly this (segmented port + tail walk + per-row 1-D async sets) and **hung non-deterministically**
(one GPU spinning).  The r54 correctness fix gives that work a **deterministic failure/reference**: the
async arm can now be A/B'd against the working sync arm on the same config, and `cache-on == cache-off`
byte-identity is the correctness oracle.

Debugging rules learned (from the tensor-split campaigns):
- Never run with `HIP_LAUNCH_BLOCKING=1` or `AMD_SERIALIZE_KERNEL=3` — both deadlock this path.
- A device fault is asynchronous: bisect with a targeted `ggml_backend_synchronize`.
- The meta compute container is double-buffered and cleared per `graph_compute`; a split intermediate
  that crosses a `graph_compute` boundary must stay alive for the consumer.

## 3. Plan

1. Instrument the sync segmented path to record the per-upload bytes and host-block time under the
   target config (this is the budget the async path must recover).
2. Implement the async segmented write (per-row 1-D `set_async` per device + the tail flat), with the
   correct ordering/event so the consumer waits.  Reuse the meta `host_buffer`/`events` proxying
   (r48, issue #137) and the existing `wait_before_overwrite` machinery.
3. Keep the sync path as the fallback/kill-switch; A/B the async arm against it.
4. Gate: cache-on == cache-off byte-identical; the target config coherent; prompt/gen back to the
   cache-off order; `test-backend-ops -o MUL_MAT_ID` no regression; `scripts/validate-set.sh` if
   promoted.

## 4. Pointers

- `archive/work/meta-segmented-expert-upload/RESOLUTION.md` (r54 fix + the exact code changed).
- `archive/work/moe-expert-cache/item3-findings-session20.md` (the hang + the original design).
- `ggml/src/ggml-backend-meta.cpp` (`set_tensor`, `set_tensor_async`, the segmented branches;
  `stage_input`; the `host_buffer`/`events` proxying).
- `ggml/src/ggml-backend.cpp` (`copy_experts`, `write_partial_row`, `wait_before_overwrite`).
- `TODO.md` "parked: segmented expert upload performance" (re-seeded from this campaign).
