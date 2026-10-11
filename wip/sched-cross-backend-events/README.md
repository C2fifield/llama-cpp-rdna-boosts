# `sched-cross-backend-events` — PR #104 follow-ups: cross-backend event wait + per-change kill-switches (TODO #33)

**Status: OPEN (seeded 2026-10-11), non-blocking.** Not part of the delivery; nothing here may be applied
to the fork without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is
the same item as **TODO #33**. The fix itself shipped in `v16-a55e952b8-r12` (blocks 06 + 13); the
campaign is the **residual hardening** and the missing **reporter-side FAIL→PASS**, recorded in the now
archived [`archive/work/2gpu-sched-fixes/VERIFICATION.md`](../../archive/work/2gpu-sched-fixes/VERIFICATION.md)
("Open follow-ups"). This directory is the live campaign.

> **Cold start.** Read this file, then `AGENTS.md` ("Report findings before shipping", "WIP and promotion
> rules"), issue #103 / PR #104, and
> [`archive/work/2gpu-sched-fixes/VERIFICATION.md`](../../archive/work/2gpu-sched-fixes/VERIFICATION.md).
> Report findings to the maintainer before any delivery change and wait.

## 0. What is still open

Three separate follow-ups, all non-blocking and all about **robustness / bisectability**, not a live bug:

1. **Foreign-event safety of the cross-backend wait.** Block 06's fix calls
   `ggml_backend_event_wait(input_backend, event)` where the event was created on `split_backend`. This is
   the scheduler's **first cross-backend event wait**, and several backends' `event_wait` implementations
   assume their own event object type. Today the delivery is CUDA/meta/CPU only, so it is never hit; on a
   mixed CUDA + SYCL/Vulkan/Metal scheduler it is at best a host block and at worst an abort.
2. **Per-change env kill-switches.** The promotion rule wants an env switch for every behaviour change.
   Block 06's cross-backend wait has none (only the global `GGML_SCHED_EVENTS=0`, which disables *all*
   per-split events at a measured prefill cost), and block 13's alias guard (`moe-expert-cache.cu:227`)
   has none either.
3. **Patch 0001 needs a reporter-side FAIL→PASS.** Block 13's device/layer alias guard was retained as a
   clearly-correct check but was **not** reproduced failing on this box (no page fault and no stale-alias
   warning even with the prefill-rebalance harness). It needs a reporter rerun against a tree that also
   has block 06's fix.

## 1. The code (r48 tree)

Cross-backend wait — `ggml/src/ggml-backend.cpp:2840` (block 06):

```cpp
if (input_backend != split_backend && input_backend->iface.event_wait != NULL &&
    sched->events[split_backend_id][sched->cur_copy] != NULL &&
    !(input->buffer != NULL && ggml_backend_buffer_is_host(input->buffer))) {
    ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
    ggml_backend_event_wait(input_backend, sched->events[split_backend_id][sched->cur_copy]);
}
```

Generic entry point: `ggml/src/ggml-backend.cpp:835` `ggml_backend_event_wait()`.

Backends that implement `event_wait` and the ones whose implementation assumes its own event type
(verify before touching this):

| backend | impl | note |
|---|---|---|
| CUDA | `ggml_backend_cuda_event_wait` (`ggml-cuda.cu:9075`) | casts to its own event struct |
| meta | `ggml_backend_meta_event_wait` (`ggml-backend-meta.cpp:3452`) | **new in r48** (issue #137): fans out to the simple devices' events |
| SYCL | `ggml_backend_sycl_event_wait` (`ggml-sycl.cpp:6330`) | `static_cast<sycl::event*>` |
| Vulkan | `ggml_backend_vk_event_wait` (`ggml-vulkan.cpp:15101`) | casts to `vk_event*` |
| Metal | `ggml_backend_metal_event_wait` (`ggml-metal.cpp:563`) | casts to `ggml_metal_event_t` |
| CANN / Hexagon / RPC / WebGPU | own impls | not in the delivery |
| CPU / BLAS / OpenCL / ... | `NULL` | skipped by the `event_wait != NULL` guard |

`GGML_SCHED_EVENTS=0` (`ggml-backend.cpp:2996`) is the only existing switch; it disables the whole
per-split event mechanism (measured prefill cost) and is not a per-change kill-switch.

Alias guard — `ggml/src/ggml-cuda/moe-expert-cache.cu:227` (block 13): a stale alias whose table device /
layer disagrees with the op's device / layer is refused with a warn-once. No env switch.

## 2. Current reality check (r48)

- r48 (issue #137) **added the meta event proxy / `host_buffer` capability** so the loader's chunked async
  upload engages under `-sm tensor`. That makes the meta backend a *stronger* event participant than when
  #33 was written, and is the most likely backend to encounter a foreign event in a meta-graph scheduler.
  Re-read `ggml-backend-meta.cpp` lines ~156–200 and `ggml_backend_meta_event_wait` before proposing a
  guard.
- The delivery still instantiates only CUDA / meta / CPU (plus the CPU host buffer types), so #33 remains
  non-blocking. The concern is future-proofing a public scheduler path, not a current failure.
- `TODO.md` #33 says "both landed in r12, non-blocking"; the item is legitimately still open because the
  hardening and the reporter rerun were never done.

## 3. Fix directions (pick one with the maintainer)

For (1), two shapes:

- **Capability-based:** add a backend/dev capability (e.g. `caps.foreign_events` or
  `supports_event_wait_from(other)`) and only issue the cross-backend wait when the source backend can wait
  on a foreign event; otherwise fall back to `ggml_backend_event_synchronize(sched->events[...])` (a host
  block — correctness-preserving, just slower). This keeps one code path and fails safe.
- **Same-family guard:** only take the async wait when `input_backend` and `split_backend` are the same
  backend *family* (CUDA↔CUDA is the only case the fix exists for); otherwise take the existing
  synchronize fallback. Simpler, but a backend that legitimately supports foreign events gains nothing.

For (2), add narrowly-scoped switches so a bisect can isolate each change without disabling the whole
event or cache machinery, e.g. `GGML_SCHED_CROSS_EVENT=0` (block 06) and an alias-guard opt-out for block
13. Keep them default-on (the changes are beneficial) per the default-on policy — the switch only *disables*.

For (3), hand the reporter a tree that is r12-or-later (so patch 0001 and block 06's 0002 are both
present) and the harness recipe in `archive/work/2gpu-sched-fixes/VERIFICATION.md`, and record the
FAIL→PASS.

## 4. Gates

- Any change here touches the scheduler hot path: re-run the coherence gate on 2- and 3-GPU `-sm tensor`,
  `-sm layer`, and the `-ncmoe` host-expert config, plus `scripts/validate-set.sh` if promoted.
- A guard that only *removes* the async wait must be bit-identical (it changes ordering, not arithmetic);
  a capability that *adds* a wait must be shown to be a no-op on the CUDA-only path.
- A new env switch must be documented in `ENVIRONMENT.md` (default, and that it is a kill-switch).

## 5. Pointers

- `TODO.md` #33; issue #103; PR #104.
- [`archive/work/2gpu-sched-fixes/VERIFICATION.md`](../../archive/work/2gpu-sched-fixes/VERIFICATION.md) —
  the change, the harness that reproduces it, and the three open follow-ups.
- `patches/README.md` block 06 (and block 13's alias guard); `WORKLOG.md` 2026-10-05 (r12).
- `ggml/src/ggml-backend.cpp:835` / `:2840`, `ggml/src/ggml-backend-meta.cpp` (~156–200, `:3452`).
