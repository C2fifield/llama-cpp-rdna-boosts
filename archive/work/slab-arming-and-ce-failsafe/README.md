# `slab-arming-and-ce-failsafe` — the `--fit off` slab blackout, the CE hard-abort, the fail-soft yield, and the issue #135 close

**Status: PROMOTED / RESOLVED (r47, 2026-10-16).**  All items were folded into
`v16-a55e952b8-r47` (canonical block-15 tip `6daf81569`, net tree `0099dd85`); `validate-set.sh` green.
See [`RESOLUTION.md`](RESOLUTION.md) for the per-item outcome and the validation record.  The text below
is the campaign as seeded (2026-10-11) for a cold session, kept as the historical plan and evidence; the
original "nothing here may be applied" caveat was satisfied by the maintainer's go-ahead, recorded in that
resolution and in `WORKLOG.md`.

> **Cold start.**  Read this file in full, then `AGENTS.md` ("Report findings before shipping",
> "Pushing policy", "Default-on policy", "WIP and promotion rules"), then issues
> [#136](https://github.com/stew675/llama-cpp-rdna-boosts/issues/136) (the `--fit off` slab blackout) and
> [#135](https://github.com/stew675/llama-cpp-rdna-boosts/issues/135) (the gfx11 all-reduce build
> failure), then [`REPRO-AND-EVIDENCE.md`](REPRO-AND-EVIDENCE.md) in this directory (the exact repro
> commands, the 4-way flag matrix, the gdb evidence and the log excerpts).  All line anchors below are
> against the r46 tree (`release.json`, tip `2e19a9d398f5cbc16b6f549c244133d276c19e13`, tree
> `a6ca8905f3ce5eb9ac7bf67ff8863a0b43419293`); re-resolve them if the tree has moved.

---

## 0. The invariant this campaign restores

> **If the MoE expert cache is enabled, the movable-boundary slab MUST be armed, unless the slab is
> explicitly disabled with `GGML_CUDA_SLAB=0`.**

The cache and the slab are not independent.  `alloc_table_locked` selects the arena backing with
`use_slab = ggml_cuda_slab_enabled()` (`moe-expert-cache.cu:1665`) and, when true, allocates through
`ggml_cuda_slab_arena_alloc` with **no raw fallback**.  `ggml_cuda_slab_enabled()` is
`env && g_slab_armed` (`ggml-cuda.cu:863`), so the invariant is really "cache on implies `g_slab_armed`".
Today that is violated on the `--fit off` path (item A), and the violation is invisible (item B).

### Block ownership (fixed; every fix folds into an existing block, the set stays 0000-0015)

| fix | owner block | why (verified against `patches/*.patch`) |
|---|---|---|
| A: parse ordering | **06** | `moe_cache_preflight`, `parse_env`, `moe_cache_init` are all introduced by block 06 |
| B: not-armed diagnostic | **06** | the invariant lives in the cache enable path (`alloc_all_locked`) and the device preflight, both block 06 |
| C: CE fail-soft | **12** | `ggml_backend_cuda_comm_allreduce_ce` is block 12 |
| D: fail-soft yield | **06** | `ggml_cuda_device_malloc` and `moe_cache_shrink_step` are block 06 |
| E: #135 verify/close | **12** | `allreduce-hip.cu` is block 12 (no code change unless the RDNA3 fast path is wanted) |

Block 15 amends the slab and `ggml_cuda_device_malloc` (r37+), so folding these into block 06 can force a
re-cut of block 15.  Apply the fix to the owner, then rebase/re-cut block 15 on top.  Never add a block.

## 1. Item A (BLOCKER) — `--fit off` never arms the slab, so the cache runs on a raw arena

### Symptom

A host-expert MoE run (`-ncmoe`/`-cmoe` > 0) with `--fit off`:

* logs **no** `SLAB_INIT dev=0` / `SLAB_INIT dev=1` / `slab … GiB (chunk …)` lines (these are `W` level,
  so they are visible at the default server verbosity), yet
* logs `alloc_all_locked: MoE expert cache (auto): arena …` with a large arena, and
* the arena is a raw `cudaMalloc`: `ggml_cuda_slab_arena_alloc` was never reached (its call site is
  guarded by `use_slab`), so the experts are held outside the slab.

Consequence: the pre-slab failure mode.  The arena grows to nearly all free VRAM, the
`ggml_cuda_device_malloc` fail-soft yield is the only protection, and the first allocation that cannot
be satisfied aborts the process.  Observed abort: the copy-engine all-reduce scratch
(`ggml_backend_cuda_comm_allreduce_ce`, `ggml-cuda.cu:2831-2833`), after ten futile
`moe_cache_shrink_step` stand-downs.

### Root cause (proven with gdb, see `REPRO-AND-EVIDENCE.md` §3)

`moe_cache_preflight()` (`moe-expert-cache.cu:2197`) opens with:

```c
bool moe_cache_preflight(int device, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    if (!g_enabled) {
        return false;          // <-- bails, so the caller never sets g_slab_armed
    }
```

`g_enabled` / `g_auto` are set **only** by `parse_env()` (`moe-expert-cache.cu:411`), which is called
**only** from `moe_cache_init()` (`moe-expert-cache.cu:3702`, at line 3707), which is called **only**
from `ggml_backend_cuda_init()` (`ggml-cuda.cu:11094`).

`common_init_result` calls `llama_model_moe_cache_preflight()` (`common/common.cpp:1352`) **before**
`llama_init_from_model()`.  That preflight is what reaches
`ggml_backend_cuda_device_moe_cache_preflight()` (`ggml-cuda.cu:10701`) and sets
`g_slab_armed = true` (`ggml-cuda.cu:10715`).

* `--fit on` (or no `--fit`): `common_fit_params()` runs a probe load first, which initialises the CUDA
  backend and therefore runs `parse_env()` **before** the preflight.  At the preflight, gdb shows
  `g_enabled = 1`, `g_auto = 1`; the slab arms.  ✅
* `--fit off`: the fit block is skipped, so no backend is initialised before the preflight.  At the
  preflight, gdb shows `g_enabled = 0`, `g_auto = 0`; `moe_cache_preflight` returns at the `!g_enabled`
  guard, `g_slab_armed` stays false.  The backend is initialised later, inside
  `llama_init_from_model()`: `moe_cache_init()` runs `parse_env()`, the cache auto-enables, and
  `use_slab` is false for the whole run.  ❌

Measured values at the `ggml_backend_cuda_device_moe_cache_preflight` entry (gdb, this box, 2× R9700):

| `--fit` | `g_slab_armed` | `g_enabled` | `g_auto` |
|---|---|---|---|
| off | 0 | **0** | **0** |
| on  | 0 | **1** | **1** |

### Fix A (proposed)

Make the env parse independent of backend-init ordering.  Factor `parse_env()` behind an idempotent
helper (reuse the `g_init_done` guard used by `moe_cache_init`) and call it at the very top of
`moe_cache_preflight()`, before the `!g_enabled` check:

```c
// moe-expert-cache.cu
static void ensure_env();   // idempotent: g_init_done-guarded parse_env()

bool moe_cache_preflight(int device, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    ensure_env();           // <-- do not depend on ggml_backend_cuda_init() ordering
    if (!g_enabled) { ... }
```

and have `moe_cache_init()` call `ensure_env()` instead of `parse_env()` directly.

Notes / traps:

* `parse_env()` is a plain global initialiser; it does not touch a device, so it is safe to run before
  any backend exists.  Verify it is genuinely idempotent (it must not reset `g_preflight_host_bytes` or
  any counter that `moe_cache_init` relies on being set once).
* Do **not** move the whole `moe_cache_init()` body: it also sets `g_device`, registers `atexit` and logs;
  only the parse is ordering-sensitive.
* Consider the alternative of forcing an early backend init on the `--fit off` path, but that is a
  larger change and re-introduces an ordering dependency; prefer the parse-early fix.
* After the fix, BOTH `--fit off` and `--fit on` must produce `SLAB_INIT` on every host-expert device,
  and the r22 field config's reservation (`headroom + aux`, i.e. `8.00 GiB` when an MTP draft is
  present) must appear in the `SLAB_INIT` line.

## 2. Item B — a configured-on slab that is never armed must be loud

There is no diagnostic for the state "the slab is enabled by configuration, the model has host experts,
but `g_slab_armed` is false".  That is how item A hid.  Add a one-time `WARN` at the invariant boundary.
The best single place is the cache's own enable path, because "cache on + slab not enabled" is exactly
the violation:

* Suggested: in `alloc_all_locked()` (`moe-expert-cache.cu`), when `g_enabled` and `g_tables` is
  non-empty, warn once if `!ggml_cuda_slab_enabled()` and `GGML_CUDA_SLAB` is not explicitly `0`.
* Complementary: in `ggml_backend_cuda_device_moe_cache_preflight()`, warn once when it returns false
  while the slab is wanted by configuration (this catches the `!g_enabled` bail and the auto-floor
  disable, both of which leave the cache off, so keep the two messages distinct).

Keep it one-time and non-fatal; the point is that the state can never again be silent.

## 3. Item C — the CE all-reduce scratch must be fail-soft, not a hard abort

`ggml_backend_cuda_comm_allreduce_ce` (`ggml-cuda.cu:2794`) grows its scratch with bare `CUDA_CHECK`s:

```c
CUDA_CHECK(cudaMalloc(&comm_ctx->ce_buf[i],  need));   // :2831
CUDA_CHECK(cudaMalloc(&comm_ctx->ce_tmp[i],  need));   // :2832
CUDA_CHECK(cudaMalloc(&comm_ctx->ce_tmp2[i], need));   // :2833
```

This is the only device allocation in the tree that does not go through `ggml_cuda_device_malloc`
(`ggml-cuda.cu:163`), the single choke point that carries the fail-soft policy.  It aborts on OOM.

### Fix C (proposed)

1. Route all three allocations through `ggml_cuda_device_malloc(ptr, need, (int) comm_ctx->dev_ids[i])`.
2. On failure, do **not** abort and do **not** `cudaFree` mid-graph (the grow path's own comment explains
   why frees are unsafe under capture).  Retire any buffers already allocated in this attempt onto
   `comm_ctx->ce_old` (freed at teardown, exactly like the growth path at `:2828-2830`), leave
   `comm_ctx->ce_bytes` unmodified, and `return false`.
3. `return false` is a supported degradation: `ggml_backend_cuda_comm_allreduce_tensor`
   (`ggml-cuda.cu`, the dispatch after `try_allreduce`) falls back to
   `ggml_backend_cuda_comm_try_allreduce_internal` when the selected arm returns false and an internal
   pipeline exists.  Confirm this path is taken for a large tensor and that the result stays coherent.
4. Optional but recommended: bound the scratch.  If `need` exceeds a budget (an env knob, or a fraction
   of free VRAM in the spirit of `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT`), return false before allocating
   so an oversized prefill degrades to `internal` instead of eating the headroom.  The `ce` win is
   +2..4 % prefill, so a graceful fallback is strictly better than an abort.

Do not change the default algorithm; `ce` stays opt-in.

## 4. Item D (latent, fix alongside) — the fail-soft yield is wrong when the arena is slab-backed

Two defects in the same code path, both worth fixing while here:

1. **The arena cannot yield physical when slab-backed.**  `free_arena_backing`
   (`moe-expert-cache.cu:1589`) calls `ggml_cuda_slab_arena_free` whenever `reserved != 0`, which returns
   the range to the slab's free list, never to the driver.  So `while (moe_cache_shrink_step())` can
   never make a `cudaMalloc` succeed.  `ggml_cuda_device_malloc` already knows this and skips the loop
   when `ggml_cuda_slab_active(device)` (`ggml-cuda.cu:198-204`), but the predicate is the *failing
   device's* slab, while `use_slab` is `ggml_cuda_slab_enabled()` (global).  When the slab is armed but a
   *different* device failed, the guard falls through and the loop churns.  Change the guard to match the
   predicate that chose the arena backing (`ggml_cuda_slab_enabled()`), or to "any table has
   `arena_reserved != 0`".
2. **`moe_cache_shrink_step()` is device-blind.**  It picks the largest table across all devices
   (`moe-expert-cache.cu:2361`) and stands it down regardless of which device needs the VRAM.  Pass the
   failing device and only consider that device's tables, so a fault on device 1 cannot evict device 0's
   residency.

With item A fixed, the slab is armed whenever the cache is on, so the loop should be unreachable in the
common case; D is the belt-and-braces so a per-device slab decline (VMM unavailable, floor) cannot
resurrect the churn.

## 5. Item E — issue #135 (gfx11 `s_wait_storecnt`) is already compile-fixed; verify and close

Issue #135 reports that `allreduce-hip.cu` fails to build for gfx1100 because `s_wait_storecnt` is
gfx12-only.  **This is already fixed in the delivered r46**: `patches/0012-…` carries the guard, and the
tree has it at `allreduce-hip.cu:196-199`:

```c
#if defined(RDNA4)
    const bool p2p_lite = p2p && lite;
#else
    const bool p2p_lite = false;
    GGML_UNUSED(lite);
#endif
```

and the asm itself is guarded at `:266-273`.  `RDNA4` is derived from the per-target `__GFX12__`
(`vendors/hip.h:215-217`), so it is correct for a multi-target (fat) build as well, which the reporter's
suggested `#if defined(__gfx12__)` would also have been.

### Remaining work

1. **Verify** the whole arch list compiles from the current tree: gfx1100, gfx1150, gfx1151, gfx1200,
   gfx1201 (the `docker-ghcr.yml` matrix does this; also do a local `clang -x hip -c allreduce-hip.cu`
   per target if a full matrix build is too slow).  Record the command and result in this campaign.
2. **Close #135** with a note pointing at the release that carries the guard.
3. **Optional (maintainer's call):** the reporter suggested a native RDNA3 fast path
   (`s_waitcnt_vscnt null, 0x0`) to replace the `__threadfence_system()` fallback for P2P-lite on RDNA3
   pairs.  That needs a 2× RDNA3 (gfx11) pair to validate, which the reporter does not have.  Do **not**
   ship it unvalidated; it stays a scoped option, and it must be env-gated if implemented.  Without
   hardware, close #135 as compile-fixed and leave the fast path as a note in the issue.

## 6. Validation plan

Run all of it on the fixed build before any release.  Use the full feature set (no opt-ins) and the
thread sizing from `AGENTS.md`.

1. **Repro flip (item A).**  On a host-expert MoE model, `--fit off` must now log `SLAB_INIT dev=0` and
   `SLAB_INIT dev=1` (and the `slab … GiB` follow-up) at startup, exactly as `--fit on` does.  Capture
   both.
2. **Reservation check.**  With an MTP draft, the `SLAB_INIT` line must show `aux=4.00 GiB` and
   `reserve=8.00 GiB`; without a draft, `aux=0.00 GiB` and `reserve=4.00 GiB`.  (See
   `REPRO-AND-EVIDENCE.md` §4 for the healthy reference lines.)
3. **Cache/arena sanity.**  `alloc_all_locked` must report a residency that fits the slab arena, and no
   `arena alloc failed for layer=…` lines.
4. **Coherence gate** (mandatory after any change): same-seed `llama-cli` diff, `-sm tensor`.
5. **MTP gates** (`benchmarks/mtp-adaptive-methodology.md`, rule 0): `-n 3000`, `--reasoning on` for R,
   `--reasoning off` for P/C/K; compare acceptance against the known-good base.
6. **Prefill-logit gate** (`scripts/gate-prefill-logits.sh`): required because the slab/allocator work
   touches prefill memory paths.  Mean KLD <= 0.005, same-top-p >= 98 %.
7. **CE fail-soft (item C)**:
   * `GGML_CUDA_ALLREDUCE=ce`, `-sm tensor`, host experts: output byte-identical to `hybrid`/`internal`
     for the same seed.
   * Force the fallback (e.g. an artificially small headroom, or a scratch budget knob) and verify the
     run degrades to `internal` with coherent output instead of aborting.
8. **Regression sweep on the field config**: `-sm tensor -ncmoe 48`, `--fit off`, `-ub 6144 -c 204800`,
   q8_0 KV, MTP n-max 3, the "generate ~8k then prefill ~70k" repro from issue #136.  It must not
   abort.
9. **Other backends unburned** (`AGENTS.md` scope policy): the fixes are AMD/ROCm-scoped, but keep the
   shared code compiling for CUDA/SYCL/Vulkan/CPU (the slab and the cache are compiled out there; the
   CE guard must not break the non-HIP build).

## 7. Release checklist (do not skip; see `AGENTS.md`)

Only after the maintainer has approved the findings and the fixes:

1. Rebuild a **canonical fork** at `release.json.base` with `scripts/apply-all.sh` (strict `git am`; the
   applied tree must equal `release.json.tree`).  Fold every fix into the **existing** block that owns the
   code; there is no new block (the set stays 0000-0015, 16 blocks).  Ownership is fixed by the patch set
   (verified with `grep` over `patches/*.patch`):
   * **block 06** (the general system-operations bucket) owns `moe-expert-cache.cu` (`parse_env`,
     `moe_cache_init`, `moe_cache_preflight`, `alloc_all_locked`, `moe_cache_shrink_step`),
     `ggml_cuda_device_malloc` (the fail-soft choke point), and the movable-boundary slab
     (`ggml_cuda_slab_*`, `g_slab_armed`, the device preflight).  **Items A, B and D fold here.**
   * **block 12** (hybrid HIP all-reduce) owns `ggml_backend_cuda_comm_allreduce_ce` and
     `allreduce-hip.cu`.  **Items C and E fold here.**
   * **block 15** (campaign memory wins) amends the slab and `ggml_cuda_device_malloc` (r37+).  A fix that
     lands in block 06 can therefore force a re-cut of block 15's hunks: apply the fix to the owner (06),
     then rebase/re-cut block 15 on top and re-run `scripts/apply-all.sh` and `scripts/validate-set.sh`.
     Do **not** create a new block to dodge that rebase.
2. `scripts/make-patches.sh`, then run the clean-apply simulation (fresh worktree, apply, build,
   coherence, and `scripts/gate-prefill-logits.sh`).
3. `scripts/validate-set.sh` must be green (strict 16/16, reconstructed tree == `release.json`).
4. Update `release.json` with `scripts/make-release.sh` (never hand-edit hashes), the header sections of
   `patches/README.md` / `README.md` / `ENVIRONMENT.md` (new env knobs, the slab invariant), and add a
   new dated `WORKLOG.md` entry (newest first).  Never edit dated validation records in place.
5. **Refresh the fork's `rdna-boosts` branch** (`AGENTS.md` "Pushing policy"): clean checkout of upstream
   `master` at the baseline, re-cut the branch with `scripts/apply-all.sh`, build, run the coherence +
   MTP + prefill-logit gates, `validate-set.sh` green, then
   `git push --force-with-lease origin rdna-boosts` to the **personal fork only**
   (`git@github.com:stew675/llama.cpp.git`).  Never upstream, never bare `--force`.
6. Push this delivery repo, tag the release, and confirm `docker-ghcr.yml` builds the arch matrix
   (which is also item E's verification).
7. Close issues #136 and #135 with a pointer to the release.

## 8. Open questions for the maintainer (ask before implementing)

1. **Item E scope**: verify-and-close only, or also attempt the RDNA3-native `s_waitcnt_vscnt` P2P-lite
   path?  The latter needs a 2× gfx11 pair to validate, which is not on this box.
2. **Item C budget**: should the CE scratch get an explicit size budget/env knob (degrade to `internal`
   above it), or just route through `ggml_cuda_device_malloc` and rely on the arena yield?
3. **Item B wording**: is a single invariant WARN at the cache-enable path enough, or is a dedicated
   startup diagnostic (e.g. "SLAB disabled: GGML_CUDA_SLAB=0" vs "SLAB NOT ARMED: <reason>") preferred?
4. Whether `--fit off` on the field server should also be revisited: the slab is only half the story if
   the field config deliberately force-fits.

## 9. Files and anchors (r46 tree)

| what | where |
|---|---|
| `moe_cache_preflight` / the `!g_enabled` guard | `ggml/src/ggml-cuda/moe-expert-cache.cu:2197` |
| `parse_env` (the lazy initialiser) | `ggml/src/ggml-cuda/moe-expert-cache.cu:411`, called at `:3707` |
| `moe_cache_init` | `ggml/src/ggml-cuda/moe-expert-cache.cu:3702` |
| `alloc_table_locked` / `use_slab` | `ggml/src/ggml-cuda/moe-expert-cache.cu:1665` |
| `free_arena_backing` (slab free = VA only) | `ggml/src/ggml-cuda/moe-expert-cache.cu:1589` |
| `moe_cache_shrink_step` (device-blind) | `ggml/src/ggml-cuda/moe-expert-cache.cu:2361` |
| `ggml_cuda_slab_enabled` / `_active` | `ggml/src/ggml-cuda/ggml-cuda.cu:863` / `:885` |
| device preflight + `g_slab_armed = true` | `ggml/src/ggml-cuda/ggml-cuda.cu:10701` / `:10715` |
| `ggml_cuda_device_malloc` (the fail-soft choke point) | `ggml/src/ggml-cuda/ggml-cuda.cu:163` |
| `ggml_cuda_slab_arena_alloc` | `ggml/src/ggml-cuda/ggml-cuda.cu:1424` |
| `ggml_backend_cuda_init` -> `moe_cache_init` | `ggml/src/ggml-cuda/ggml-cuda.cu:11094` |
| CE arm + raw `cudaMalloc`s | `ggml/src/ggml-cuda/ggml-cuda.cu:2794` / `:2831-2833` |
| CE dispatch + fallback to `internal` | `ggml_backend_cuda_comm_allreduce_tensor` in the same file |
| the driver preflight call | `common/common.cpp:1352` |
| the #135 guard | `ggml/src/ggml-cuda/allreduce-hip.cu:196-199`, `:266-273`; `vendors/hip.h:215-217` |

## 10. Do not

* Do not "fix" item A by special-casing `--fit off` in `common/`.  The invariant is in the backend; the
  ordering bug is that the backend reads its own config too late.
* Do not make the cache refuse to enable when the slab is unarmed as the *primary* fix.  That would turn
  a working (if slower) host-streaming run into a cache-off run.  Arm the slab; warn if you cannot.
* Do not ship the item-E RDNA3 fast path without gfx11 hardware validation.
* Do not apply anything in this directory to the fork without the maintainer's go-ahead.
