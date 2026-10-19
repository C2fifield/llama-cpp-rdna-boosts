# RESOLUTION — `slab-transient-absorption`: PROMOTED into `v12-097cf33d9-r86`

**Status: folded into the delivery set (2026-10-19).**  The campaign began as `wip/slab-transient-absorption/`
and was staged for testers as `beta/0012-slab-transient-absorption` (BETA 2, 2026-10-18).  It ships in
`v12-097cf33d9-r86` (tree `bb8258610763da19d03d375865838aac7e79ed30`), folded per owning block.

## What was promoted

| concern | block |
|---|---|
| the draft reserve derived from the draft GGUF, the draft model loaded before the target context, the slab serving the common-layer and all-reduce transients | **09** |
| the wide-phase auxiliary region (FA staging arena, QSA3 workspace, chunked-GDN scratch), the workspace-pool redirect, capture safety while the wide band is armed, the per-device `SLAB_STATS` diagnostics, default headroom `640 + 128 x n_gpu` MiB | **10** |
| the loaded-draft ownership handoff (`common/speculative.cpp`) | **11** |

The candidate is a whole-tree memory-lifetime change, so it is not one commit: each hunk applies at the
block that last owns the file, keeping the intermediate commits self-consistent.

## Validation (promotion gates, tree `bb825861`)

- `scripts/validate-set.sh`: strict 12/12 `git am` on a fresh tarball, tree matches `release.json`.
- dense `Qwen3.5-4B Q8_0` and host-expert `35B-A3B -ncmoe 40` greedy text byte-identical to the
  r86-candidate build (tree `f6f55ffc`).
- prefill-logit: mean KLD **0.000707**, same-top-p **98.755 %** — PASS.
- adaptive-MTP 4 axes + phase-switching: exactly equal to the r86-candidate on every axis
  (R 0.66461, P 0.51005, C 0.76070, K 0.96262, phase 0.62752); prose `7759fa4a7c8d`, recall
  `5ebd156073f6`.
- `llama-batched-bench -npl 1,4,8`: TG equal; issue #170 24 GiB-parity reproduction LISTENING and a
  genuine weight OOM exits cleanly; `check-gpu-faults.sh` clean.

Full record: `WORKLOG.md` 261019.001; the BETA 2 tester checklist is `BETA-TESTING.md` and the BETA 2
README (with the measurements) is `BETA-README.md`.

## Open follow-up (not part of r86)

The draft's per-device KV is charged whole to every device (safe, bounded by `(n-1)/n` of the KV).  If
the draft's layer placement is later available from the loaded model, the KV can be measured the same way
as the weights; a move to streaming KV weights may make the term moot.  Deliberately deferred.
