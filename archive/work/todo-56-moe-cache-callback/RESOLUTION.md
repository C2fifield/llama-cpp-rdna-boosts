# RESOLUTION — `todo-56-moe-cache-callback` (TODO #56) — FIXED in r99

**Status: PROMOTED (2026-10-20) into `v12-097cf33d9-r99` as a block-09 amendment.**  TODO #56 is closed.
The campaign's own `README.md` and patch are kept beside this file.  No `wip/` stub remains.

## Root cause (confirmed on gfx1201)

The delivery's rewritten `ggml_backend_sched_copy_input` kept no `sched->callback_copy` call, and that call
is the only thing in the tree that drives `llama_moe_cache::copy()`.  With `--moe-cache-mib` set, the slot
map stayed `0xff` (-1) while the graph had already replaced the routed op's `src0` with the cache bank and
its ids with `GET_ROWS(slot_map, ids)`, so `MUL_MAT_ID` indexed the bank with `-1`: a silent wrong value
where the page below the bank is mapped (gfx1201 default arm, NMSE 4.02e-03) and an HSA page fault where it
is not (gfx1151; gfx1201 with `GGML_CUDA_ASYNC_INPUT=0`).

## What landed (block 09)

- `ggml_backend_sched_copy_input`: base's `callback_copy` + `is_host_weight` call restored, at base's
  position (after the wait-before-overwrite) and in the branch with no `MUL_MAT_ID` consumer, which is where
  the slot map lands (its consumer is a `GET_ROWS`).  The delivery's expert branch (device gather, cache
  take-over, pruning) is byte-for-byte unchanged.
- `llama_context::sched_copy_experts` answers only for the upstream cache; its per-op staging/pruning body
  and the `copy_experts_info` state are deleted (dead since the block-09 scheduler took that over, and they
  would double-stage every host expert table now that the callback is wired).

## Gates (gfx1201, ROCm 10.1.0, on the canonical r99 build)

| check | r98 | r99 |
|---|---|---|
| `test-llama-archs -s 42` | 373/375 | **375/375** |
| same, `GGML_CUDA_ASYNC_INPUT=0` | 373/375 | **375/375** |
| `deepseek2` `MoE cache` row | NMSE 4.02e-03 | **2.10e-10** (the upstream base's own value) |
| host-expert greedy text (35B-A3B `-ncmoe 40`) | `4ae79b6e45fc` | `4ae79b6e45fc` (identical) |
| prefill-logit KLD | 0.000000 / 100.000 % | 0.000000 / 100.000 % |
| `test-backend-ops -o FLASH_ATTN_EXT` | 6391/6391 | 6391/6391 |

The campaign's gfx1151 evidence (373/375 -> 375/375 there, and a real 35B-A3B `-ncmoe 40 --moe-cache-mib`
decode that faulted now running coherent at a 62.08 % hit rate) is in the kept `README.md`.

## Accepted limitations

- The upstream cache's large-batch `copy_experts()` arm stays unreached (this block's scheduler owns that
  pruning).  Values are identical either way (the bank holds the same expert bytes), so it is a missing
  upstream optimisation on a path the delivery serves with its own cache, not a correctness gap.
- `--moe-cache-mib` is now correct but slower than `MOE_EXPERT_CACHE_MIB` on this box (35B-A3B `-ncmoe 40`
  decode 50.1 -> 37.8 t/s), because the upstream engine synchronises the split backend once per cached layer
  per pass.  Documented as "works, not recommended"; the delivery's own engine stays the path to use.
- The multi-device `MoE cache, layer split` row is still unmeasured: this box has no second device class for
  it and the upstream engine rejects `LLAMA_SPLIT_MODE_TENSOR` outright.  Nothing in the change is
  device-count dependent.
