# RESOLUTION — `moe-layer-stagger` (TODO item 63)

**Status: CLOSED — REFUTED / SUPERSEDED, no delivery change (2026-10-15).**  The campaign's premise —
that striding the host/device expert-layer split spreads the H2D upload and beats the contiguous split
at the same host-byte count — **does not hold**.  With the page-cache confound removed, *which* MoE
layers are host-resident has **no effect** on prefill at a fixed host-byte count, and the speedup the
campaign was created to unlock is already covered by the delivery's own tail-set host residency (r63).
**Nothing was promoted; no release was cut.**

## What the campaign was

`-ncmoe N` keeps the routed experts of a **contiguous first N** trunk layers on the host
(`llm_add_n_cpu_ffn_overrides`, `common/common.h:1274`); there was no stride/ratio knob.  The
hypothesis: spreading the host layers evenly (e.g. even device / odd host at a 50:50 split) would give
every layer some host bytes to hide behind a neighbouring device-resident layer's compute, instead of
the contiguous stretch being one exposed, bandwidth-bound run.  Design options A (stride),
B (pattern) and C (ratio/Bresenham) are in `README.md` §3.

## What was measured

All numbers: 2x R9700 gfx1201, ROCm 7.14.1, delivery tip `7f15fb91d` (r61), `-ngl 99 -sm tensor
-ncmoe 24 -fa 1 -b 4096 -t 8`, `MOE_EXPERT_HEATMAP=0`.  Raw logs: `logs/`; harness:
`tools/run-stagger-bench.sh`; scratch knob: `tools/ncmoe-layer-select.patch`.

* **Placement is a no-op (the finding).**  Run from a **tmpfs model copy** (the disk-backed first pass
  drifted 1168 -> 1500 t/s over ~30 min, a +28 % page-cache warm-up), four **disjoint** host sets are
  identical to 0.1 % at `-ncmoe 24`, `-ub 2048`:

  | host set | pp4096 | pp8192 |
  |---|---:|---:|
  | contiguous `0..23` | 1578.8 | 1640.3 |
  | even `0,2,..,46`   | 1576.8 | 1640.5 |
  | odd `1,3,..,47`    | 1578.8 | 1640.9 |
  | contiguous `24..47`| 1578.6 | 1640.6 |

* **The real cost is the H2D of the whole host table, per ubatch.**  Marginal cost ~52-67 ms per host
  layer per ubatch (`-ncmoe 24` re-uploads 24 x 912.5 MiB = 21.4 GiB per ubatch, ~15-18 GB/s against a
  PCIe **x4** upstream link).  The config lever is ubatch width, not placement: `-ub 2048 -> 4096` is
  **+37 %** at the same `-ncmoe 24`.
* **No prefetch to exploit.**  The ring overlaps split N-1's compute with split N's upload (one split
  deep) but does not lead it; a deeper ring pipeline cannot help while the link is the binding
  constraint — the same bytes cross the same link regardless of when they are issued.

Detail and the correctness history (the `-sm tensor` + `-ncmoe` corruption found here, fixed in r62
and folded into block 10) are in `FINDINGS-2026-10-14.md`.

## Why it is superseded

The speedup the campaign wanted is a delivery feature now: **r63** makes `-ncmoe N` host the **last N**
trunk layers (rear-fill) instead of the first N.  The tail layers carry the colder expert routing and
have preceding resident compute to overlap the upload with.  Measured 35B-A3B Q8_0, 2x R9700,
`-ncmoe 20`: `pp4096` **4264 vs 4120**, `tg500` **87.5 vs 82.6**, `pp512` **871 vs 635**
(`WORKLOG.md` 261014.008, `archive/work/moe-upload-overlap/RESOLUTION.md`).  A per-layer stride
generalises nothing on top of that, and the placement measurement above shows it would not help anyway.

## Maintainer's call

On the maintainer's decision (2026-10-15) the campaign is closed as refuted, not resumed.  `TODO.md`
#63 is closed; `wip/moe-layer-stagger/` moved here with this resolution and the `wip/` entry deleted
(no stub); `wip/CAMPAIGNS.md` updated.

## Artifacts

`FINDINGS-2026-10-14.md` (the full measurement record), `logs/` (raw bench logs), `tools/`
(`run-stagger-bench.sh`, the env-gated `ncmoe-layer-select.patch` scratch knob — inert unless
`LLAMA_NCMOE_LAYERS`/`LLAMA_NCMOE_STRIDE`/`LLAMA_NCMOE_OFFSET` are set).  Nothing here is applied by
`apply-all.sh`; the delivered form of the tail placement is in `patches/` (block 09).
