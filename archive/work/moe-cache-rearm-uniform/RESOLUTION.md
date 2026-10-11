# RESOLUTION — moe-cache-rearm-uniform (PR #126)

**Status:** PARTIALLY FOLDED into the delivery as **block 16** on 2026-10-14 (release
`v16-a55e952b8-r45`).  **Owner:** `@briansp2020` (PR #126).

## What was adopted

`0001-moe-cache-re-arm-a-stood-down-layer-s-other-tables-w.patch` — `moe_cache_rearm` now adds a
layer's other **allocated** tables to the set it re-arms, so a layer whose storage was only
partially taken by slab chunks comes back slot-uniform instead of, e.g., `ffn_up_exps` at 193
slots next to `gate`/`down` at 502.

## What was not adopted (superseded)

`0002` and `0003` do not apply to the current tree and target machinery that no longer exists:

- `0002` fixes the pipelined used-list readback (`used_dev`/`used_host`/`used_host2`,
  `used_pending`/`used_toggle`) in `alloc_table_locked`.  **r40 removed the device-remap
  (`DEVMAP`) machinery entirely**; those fields no longer exist and there is no readback pipeline.
- `0003` uploads the device `expert -> slot` map (`slot_dev`/`slot_pin`) at the end of
  `apply_prefill_seed_rank_locked`.  r40 removed the device slot map; the current prefill seed
  already sets `t.slot_dirty = true` and the decode band reads the arena directly, with no
  per-token host routing readback.

The reported tight-headroom wrong-greedy-output on r37 (`GGML_CUDA_SLAB_HEADROOM_MIB=3072`) has not
been reproduced on the current tree.  If it reappears, start a fresh campaign against the current
r45 tree with a reproducer.
