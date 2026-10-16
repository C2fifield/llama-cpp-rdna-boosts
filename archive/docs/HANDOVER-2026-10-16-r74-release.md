# HANDOVER — r74 release: gate testing + publishing

**Date:** 2026-10-16.  **Release:** `v12-097cf33d9-r74` (base `097cf33d9`, 12 blocks, canonical tip
`9450cbb89b7d04885040370f29731aa2289ae7ce`, net tree `0fe32003b4014c2adf520eccc2e2ba1647ae3511`).
**Status: promoted + tagged locally, NOT pushed.**  This document drives the remaining gate testing and the
publish.  It is a handover, not part of the delivery.

## 0. What r74 contains

The session produced two delivery changes, both now in `patches/`:

1. **r73 (naming, block 09):** the MoE heat-map sidecar is one unified `<base>.moe-heatmap` per model
   (`-NNNNN-of-NNNNN` + `.gguf` stripped); the retired shard-named layout is never read/written and a
   stale one logs a warning.  (r73 is an untagged intermediate included in r74.)
2. **r74 (arena, blocks 09/10/11):**
   - the arena is **sized at the warmup** (the priming pass now calls `alloc_all_locked`), not on the
     first user decode;
   - `slab_extend_all` is **skipped** at the priming sizing (it over-mapped the reserve against the empty
     context and a later graph capture OOM'd);
   - the arena is sized to the **full** region (15.62 GiB on the field GSQ config) and the **seed stops at
     `narrow_floor + ring + wide`** — the band a wide prefill immediately reclaims.  That band's tables
     are the coldest (heat-ordered last), stay unseeded, are reclaimed cheaply by the first prefill, and
     fill lazily during decode.  The seed boundary is the shared `ggml_cuda_slab_ring_layout` helper's
     `wide_floor` output (also used by the real wide placement).
   - **kill switch:** `MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` restores the old deferred sizing.

Verified once on the field GSQ command (1 GPU): arena 15666.6 MiB at load, 97/144 tables seeded, first
prefill evicts 6150 MiB (the band), re-arm restores the arena, decode 75.0 t/s.  Log:
`archive/work/first-prefill-latency/seed-floor-full-arena-1gpu.log`.

## 1. Current state

- **Delivery repo** `~/llama-cpp-rdna-boosts`: `main` at `6170003` (r74 corrected), tag
  `v12-097cf33d9-r74` on it, `release.json` = r74, working tree clean.
- **Fork** `~/llama.cpp`: `rdna-boosts` at `9450cbb89`, clean; `build-rocm/` and `build-rocm-hybrid/` were
  built from this content (version string may still read `d6f59a05d`; rebuild if you want it exact).
- `scripts/validate-set.sh` is **green** (checksums + strict 12/12 `git am` + tree match).
- Nothing has been pushed; the fork `rdna-boosts` branch and the container images are untouched.

## 2. Gate testing (run BEFORE any push)

Run in this order and record results.  Do **not** run benches in parallel/background.

### 2.1 Delivery apply
```bash
cd ~/llama-cpp-rdna-boosts && ./scripts/validate-set.sh
```
Expected: green (already verified).  Re-run after any further edit.

### 2.2 Coherence (mandatory)
Same-seed `llama-cli`, identical output vs a known-good (r72/r73) build.  Run at least 2- and 3-GPU
`-sm tensor`, and the arena A/B (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0`):
```bash
HIP_VISIBLE_DEVICES=0,1   ./build/bin/llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 --seed 42 --temp 0 \
  --no-display-prompt --single-turn
# repeat for 0,1,2; and with MOE_EXPERT_CACHE_SIZE_AT_PRIME=0 as the A/B arm
```
**Expected risk:** the arena is now resident at a different time, and the fused / non-fused MoE paths are
not bit-identical (`TODO.md` #50 / code `OPEN 2`), so a flipped near-tie vs r73 is a plausible, benign
result — if it happens, record it (token, prompt, both texts, `MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` same?)
rather than treating it as a regression.

### 2.3 Prefill-logit KLD (mandatory for a prefill-adjacent change)
```bash
./scripts/gate-prefill-logits.sh            # requires mean KLD <= 0.005 and same-top-p >= 98 %
```
Protocol: `benchmarks/prefill-logit-methodology.md` (issue #113).  The arena change does not alter expert
bytes, so this is expected to pass; it is a release gate.

### 2.4 Adaptive-MTP gate
Per `benchmarks/mtp-adaptive-methodology.md`: `-n 3000` (2000 floor), `--reasoning on` for R and
`--reasoning off` for P/C/K, and the phase-switching `prompts/code-reasoning-mixed.txt`.  Judge on the
four axes, not just t/s.  Run the stock-relative `llama-batched-bench -npl 1,4,8` width gate if any
decode/mmvq path changed (it did not this session, but confirm).

### 2.5 Arena-specific checks (this session's change)
1. **Full arena at load:** on the maintainer's GSQ command, `alloc_all_locked` at load reports
   `arena 15666.6 MiB` (the full region), NOT the rejected draft's 9671.
2. **Seed capped:** with `-lv 4`, the `heat-seeded`/`prefill-seeded` count is ~97/144 (the band tables
   skipped).  No `heat-seeded` for a table whose arena is below `narrow_floor + ring + wide`.
3. **First prefill:** eviction ≈ the band (~6.1 GiB), then `moe_cache_rearm` restores the arena to the
   full size.  No OOM.
4. **A/B:** `MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` reproduces the old deferred sizing (arena sized on the
   first decode); the default should be no worse than it.
5. **2/3-GPU re-measure:** the earlier 2/3-GPU table (arena 23340/40925, 84.4/56.9 t/s) was the
   **rejected persistent-reservation draft** and is invalid.  Re-measure `-sm tensor` on `0,1` and
   `0,1,2` for the corrected design (full arena, seed floor per device).  See
   `archive/work/first-prefill-latency/README.md` §9.
6. **Different models / prompts:** at least the 27B dense (no cache — the change must be inert), a
   host-expert model at a *small* `-ncmoe` (partial residency) and a fully-resident `-sm tensor` arm.

### 2.6 Regression surface
- `--no-warmup`: the priming sizing does not run; sizing falls back to the first decode exactly as before.
  Confirm no crash and the arena is sized.
- `MOE_EXPERT_CACHE_MIB=0` (cache off): the whole path must be inert.
- Dense model: the slab is not armed; the seed floor is 0; no behavior change.

## 3. Publishing (only after the gates pass and the maintainer gives the go-ahead)

Per `AGENTS.md` → "Pushing policy".  **Never push to upstream `ggml-org/llama.cpp`.**

1. **Push the delivery** (branch + tag) to `github.com:stew675/llama-cpp-rdna-boosts`:
   ```bash
   cd ~/llama-cpp-rdna-boosts
   git push origin main
   git push origin v12-097cf33d9-r74
   ```
   This triggers `release.yml` (GitHub Release assets: `rdna-boosts-all.patch`, `patches.tar.gz`,
   `release.json`, `SHA256SUMS`).

2. **Refresh the fork's `rdna-boosts` branch** (consumers clone this branch):
   - clean `~/llama.cpp`, check out upstream `master` at the **baseline point** (`.base` =
     `097cf33d9`, never the tip of `master`);
   - delete and re-create `rdna-boosts` with `scripts/apply-all.sh` (strict `git am`; the applied tree
     must equal `release.json.tree` `0fe32003b4014c2adf520eccc2e2ba1647ae3511`);
   - build it, run the coherence + MTP gates and `scripts/gate-prefill-logits.sh`, `validate-set.sh`
     green;
   - `git push --force-with-lease origin rdna-boosts` (the **personal fork**
     `git@github.com:stew675/llama.cpp.git` only; never a bare `--force`).

3. **Publish the ROCm container images — off-host by default.**  Do **not** run the full image build on
   the primary box during a release (it pins all cores for tens of minutes).  The worker watcher
   (`scripts/container-watch.sh` on `radiance`, cron */10) picks up the new tag and runs
   `scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`.  If the worker is unavailable:
   ```bash
   ./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune
   ```
   See `CONTAINERS.md`.  A manual-only fallback exists at `.github/workflows/docker-ghcr.yml`.

## 4. Known open items / risks

- **MoE near-tie drift** (the arena residency timing; `TODO.md` #50, `GREEDY-PURITY.md`): if the
  coherence gate differs from r73, it is likely this.  Record, do not silently accept.
- **3-GPU decode deficit** — a separate, revived campaign: `wip/three-gpu-performance/` (TODO #51).
  r74 is not expected to fix it; the 3-GPU arm is a re-measure target.
- **2/3-GPU arms of the corrected arena design** are unmeasured (see §2.5.5).
- **`keep-survivors`** re-arm prototype is inert in this config and stays un-promoted
  (`archive/work/first-prefill-latency/keep-survivors.patch`).
- **r73 is untagged**; r74 includes it.  If a separate r73 tag is wanted, say so.

## 5. Pointers

- `release.json` — tip/tree/hashes; `patches/README.md` — per-release notes; `WORKLOG.md` 261016.008
  (r73) and 261016.009 (r74).
- `ENVIRONMENT.md` — `MOE_EXPERT_CACHE_SIZE_AT_PRIME` row.
- `archive/work/first-prefill-latency/` — measurement, diagnosis, rejected draft, seed-floor log.
- `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md` — gate protocols.
- `AGENTS.md` — "Pushing policy", "Report findings before shipping", "Default-on policy".
