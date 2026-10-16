# ENVIRONMENT.md — environment variables that affect this repo

Scope: everything **this repo adds or repurposes**, plus the upstream variables and the non-ggml
runtime variables (HIP/ROCm) that materially affect how this delivery behaves.  Defaults are taken from
the `rdna-boosts` tree at release `v12-097cf33d9-r52` (V2.0; the pre-r52 values were unchanged from the
            16-block set, which was a pure re-organization); the table was generated from the code, not from
memory (see [Maintaining this file](#maintaining-this-file)).

## How to read this

| class | meaning |
|---|---|
| **Kill-switch** | the feature is **ON by default** because it passed its gate; the variable exists to disable it for A/B, bisection or a workaround. Setting it to `0` restores the old behaviour. |
| **Opt-in** | OFF by default; the feature is not gated or not for production. |
| **Tuning** | takes a number/string; the default is the validated value. |
| **Diagnostic** | no effect on the default path — it only logs, dumps, or gates a debug helper. Never set in production. |

Two idioms are used in the code and they are **not** interchangeable:

* *presence-only*: `getenv("X") != nullptr` — **any** value (including `0`) enables it.
* *value-gated*: `getenv("X") != nullptr && atoi(...)` — `1` enables, `0` leaves it off.

The tables below say which one each variable uses, because guessing wrong is a classic time sink.

Every behaviour-affecting variable is also resolved **once** (into a function-local static or value) on
first use, so setting one after the process has started has no effect.  This is deliberate: `getenv()`
takes a lock and rescans the environment block on Windows, and several hot paths (per matmul, per
device allocation, per graph input, per batch) would otherwise read it thousands of times.  It also
avoids caching a `const char *` returned by `getenv()` across calls, which is unsafe on Windows because
the MSVC CRT reuses its buffer for the next `getenv()`.

## The short answer: what you actually need

**Nothing.** Every feature the delivery ships is default-on, and this is deliberate policy (see
`AGENTS.md`, "Default-on policy"). You do **not** need to set anything for the recommended server or
`llama-cli` invocation. The variables below exist to *disable* things, to bisect a regression, or to
tune a value that has a validated default.

The only ones worth knowing by heart:

| variable | default | why you would touch it |
|---|---|---|
| `GGML_COMPUTE_BUFFER_MARGIN_PCT` | `10` (HIP only) | `0` disables the compute-buffer slack — the fastest way to reproduce the old TODO #42 abort |
| `MOE_EXPERT_CACHE_MIB` | unset = auto | `0` disables the MoE expert cache entirely (CPU expert path) |
| `GGML_CUDA_SLAB` | **on** | `0` disables the movable-boundary slab allocator (back to per-allocation `cudaMalloc`; reintroduces the TODO #42 server abort).  Since **r37** the slab is additionally **armed only for models with host-resident experts** (`-ncmoe`/`-cmoe` > 0); a dense model never creates one.  Since **r47** the arm is independent of `--fit` (issue #136): if the cache is enabled the slab is armed |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `4096` | VRAM left outside the slab after it reclaims the unused reserve. **Raising it buys steady-state margin at the cost of arena**; `2048` aborts the 16k path inside hipBLASLt |
| `GGML_CUDA_ALLREDUCE` | `hybrid` | force `internal` \| `nccl` \| `ce` for multi-GPU A/B |
| `LLAMA_DROP_COMPUTE_BUFFERS` | **on for every tool** | `0` keeps the wide-prefill layout (smaller arena). Under the slab a later wide prefill reclaims it via a boundary move, so a server may drop too |
| `GGML_CUDA_OP_TIMING` | off | per-op GPU timing/profiling |

---

## 1. Compute buffers and the MoE arena

### 1.1 `GGML_COMPUTE_BUFFER_MARGIN_PCT` — **tuning**, default `10`, HIP/ROCm only

Pads every **compute** buffer allocation by that percentage (`ggml-alloc.c`, via the buffer type's
`get_compute_margin_pct`).  The graph allocator sizes the compute buffer from a *measure* graph, but a
runtime graph can carry a different live-tensor set (host-expert staging, MTP taps) and need a little
more — measured +3.3 %.  Growing it is a free-then-allocate-larger, so it needs a contiguous block
**bigger than the one just released**, which fails once the free VRAM belongs to the MoE arena.  Taking
the slack up front moves the cost to before the arena is sized.

* `0` disables it — and **reproduces the original abort**, so it is the natural A/B control.
* Cost, measured on 2× R9700 (`-sm tensor -ncmoe 48`): 1275 MiB of arena residency (63.8 % → 61.5 %),
  ≈127 MiB per percentage point.
* Only the RDNA/ROCm path opts in (`#if defined(GGML_USE_HIP)`); every other backend's allocation sizes
  are unchanged.  Read once per process.
* **Unchanged in r37:** the margin still applies to dense models (about 10 % of a now-small layout).  Only
  the **chunk** is slab-gated (`GGML_COMPUTE_BUFFER_CHUNK_MIB`); together with the r37 host-expert slab gate
  this is what restores a dense model's pre-slab compute-buffer sizes (issue #120).
* Compute-only: model weights, the KV cache and the arena are untouched.

### 1.2 MoE expert-cache arena

| variable | default | class | notes |
|---|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | unset = **auto** | tuning | sizes each device from its free VRAM. `0` disables the cache **completely** (CPU expert path) — prefer this over a tiny arena, which stands the cache-aware MoE fusions down and is ~8× slower than the cache-less path. |
| `MOE_EXPERT_CACHE_RESERVE_MIB` | `1024` | tuning | VRAM held back from the arena for the compute side / KV cache.  The old server workaround raised this to `8192`; **that is no longer needed** (the r20 slack + the fail-soft yield cover it). |
| `MOE_EXPERT_CACHE_MIN_MIB` | `0` | tuning | auto-sizing floor; below it the cache declines itself instead of building a useless arena.  Decided **early** (before the context exists, i.e. before any graph) since r24 -- a late `g_enabled = false` would plan the MTP draft and the target with different kernels.  If a config still trips it late, it logs an ERROR and keeps the small arena (correct, slower). |
| `MOE_EXPERT_CACHE_MIN_RES_PCT` | `18` when auto | tuning | minimum residency worth keeping.  Live since r24 (it was silently inert while the early floor never ran). |
| `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` | unset | tuning | extra reserve for auxiliary buffers. |
| `MOE_EXPERT_CACHE_SLOTS` | `0` | tuning | explicit slot count hint (0 = derive from the budget). |
| `MOE_EXPERT_CACHE_PERIOD` | `32` | tuning | promotion/rebalance period, in tokens. |
| `MOE_EXPERT_CACHE_FILL` | `1` | kill-switch | fill-on-miss. |
| `MOE_EXPERT_CACHE_TOUCH` | `2` | tuning | `admit=touch`: uses since last resident before a fill is allowed. |
| `MOE_EXPERT_CACHE_PROVISIONAL` | `1` | kill-switch | provisional eviction during a pass. |
| `MOE_EXPERT_CACHE_PREFILL_SEED` | `1` | kill-switch | seed the cache during prefill. |
| `MOE_EXPERT_CACHE_PREFILL_SEED_N` | `0` | tuning | how many prefill seeds (0 = default). |
| `MOE_EXPERT_HEATMAP` | `1` | kill-switch | **persistent cross-run expert heat map** (r60, block 09).  On: the cache tallies a long-horizon per-expert use count (device-side on the prefill band, host-side on the eager decode band) and writes it to a sidecar next to the model; the next run seeds the arena with the hottest experts at sizing and allocates the hottest **layers** first (so the work-pool boundary reclaims the coldest).  Measured on `MOE_EXPERT_CACHE_MIB=4096`: **+34 % decode over a 48-token turn** (38.15 -> 51.30 t/s, hit 0.767 -> 0.940, fills 11748 -> 2976) and **+10.3 % over 256 tokens** (48.53 -> 53.55 t/s, fills 19452 -> 6912); prefill unchanged.  It is a placement prior only (the expert bytes are unchanged), so the output is bit-identical.  `0` restores the exact pre-change behaviour.  `load` (aliases `ro`, `read-only`) loads and seeds from the map but never writes it.  A **read-only heat-map file** is also loaded and frozen: the file's mode is the freeze signal, and it does not fall back to a cache directory. |
| `MOE_EXPERT_HEATMAP_DIR` | unset | tuning | directory for the sidecar when the model directory is not writable.  Otherwise the fallback order is `$XDG_CACHE_HOME/llama-moe-heatmap`, `$HOME/.cache/llama-moe-heatmap`, `/tmp`. |
| `MOE_EXPERT_HEATMAP_DEBOUNCE_S` | `60` | tuning | minimum seconds between heat-map writes (the server flushes at turn end; process exit always flushes). |
| `MOE_EXPERT_HEATMAP_MAX_FAILS` | `3` | tuning | consecutive write failures before persistence is disabled for the session (non-fatal throughout). |
| `GGML_MOE_CACHE_MAX_TOK` | device MMVQ band (16 on RDNA4; 8 on RDNA3, NVIDIA) | kill-switch | caps the expert-cache **decode/verify band** (routed `MUL_MAT_ID` tokens the cache serves from the arena).  It defaults to the routed-expert MMVQ kernel's own band on the device since r31; `8` restores the pre-r31 band (band widening in `archive/work/moe-cache-band16/`).  A non-numeric value silently yields 1. |
| `GGML_MOE_GATHER_ONCE` | off | diagnostic | gather each expert once per pass. |

> **Heat map (r60).**  The per-expert heat map is **on by default** whenever the cache is active on
> host-expert tables (`MOE_EXPERT_CACHE_MIB` auto/positive and a `-ncmoe`/`-cmoe` model).  Its sidecar
> is `<gguf>.moe-heatmap` (geometry-keyed, atomic `tmp`+`rename`, debounced 60 s, non-fatal on failure);
> if the model directory is read-only it falls back to `MOE_EXPERT_HEATMAP_DIR` / the XDG cache.  It is
> a **placement prior only**: the expert bytes are unchanged, so a run's output is bit-identical and
> `MOE_EXPERT_HEATMAP=0` reproduces the pre-change behaviour exactly.  The in-run
> `MOE_EXPERT_CACHE_PREFILL_SEED` remains as a second (weaker) seed source; the heat map is applied
> first.  **Freezing a map (r61):** set `MOE_EXPERT_HEATMAP=load`, or simply make the heat-map file
> read-only (`chmod 400 <gguf>.moe-heatmap`).  A read-only file is loaded and used to seed, is never
> overwritten, and is not relocated.  See `patches/README.md` (block 09, r60/r61) and
> `archive/work/moe-heatmap-persistence/`.

> **REMOVED — do not reintroduce (2026-10-10, released r40).**  The device-remap path and its whole family
> (`MOE_EXPERT_CACHE_DEVMAP`, `MOE_EXPERT_CACHE_DEVPOLICY`, `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT`,
> `MOE_EXPERT_CACHE_KSLOT`, `MOE_EXPERT_CACHE_DEV_EAGER`) plus the device-policy knob `MOE_HOST_POOL_POLICY`
> were **deleted from the code base**.  The eager
> host-routing path (materialized remap) is the only `-sm tensor` split policy; it is MTP-pure on qwen4exp
> (`none == n3`), within noise of the old `DEVMAP=0` performance, and resolves `TODO.md` #50.  The device
> path's divergence was a layout-sensitive **hipBLASLt** solution flip (issue #67), not a cache bug; the
> only workaround (`ROCBLAS_USE_HIPBLASLT=0`) costs ~12-17 % prefill on gfx1201/ROCm 7.14, and ROCm 10.1.0
> does not fix it.  See `wip/moe-verify-fusions/FINDINGS-devmap-removal.md`.  Do **not** add any
> `*DEVMAP*`/`*KSLOT*`/`*DEVPOLICY*` variable or device slot-map path back. |

> The arena is the **lowest-priority** VRAM consumer: without a slab, any device allocation that runs short
> frees the largest table (then the whole arena) and retries.  With the slab on (the default, see §1.3) that
> path is *skipped* — freeing a table cannot return bytes to the driver once the slab owns them — and the
> allocation instead fails with a message naming `GGML_CUDA_SLAB_RESERVE_MIB`.  There is no variable for the
> yield behaviour, and it should stay that way.  See `ARENA-UB-TENSION.md` §13/§14.

### 1.2.1 Optional host-expert pool (`--host-experts pool`, block 06)

The pool is an **option** for systems that cannot hold the weights resident; the default `--host-experts
pinned` (and `auto`) keeps the usual behaviour and allocates **no** pool.  When enabled, it is a bounded
pinned bounce buffer over the page cache, one per host tensor, process-wide budget, and it serves the MoE
expert-cache fills.  r35.

**It does not reduce host RSS yet.**  It keeps the full pinned `ROCm_Host` master (the source logs
*"Phase 2b; full pinned master kept"*), so measured `RssShmem` is identical with and without the pool
(18.7 GiB on 35B-A3B `-ncmoe 40`; 54.1 GiB on Flash-Next IQ4_NL `-ncmoe 40`).  Dropping the master is a
separate campaign: `wip/host-expert-master-drop/`.

| variable / flag | default | class | notes |
|---|---|---|---|
| `--host-experts pool` | `pinned` | option | select the bounded host pool.  `pinned`/`auto` = no pool; `mmap` was removed in r33. |
| `MOE_HOST_POOL_MIB` | `auto` | tuning | pool size when enabled: unset/`auto` = `MOE_HOST_POOL_FRAC` % of the MoE host expert bytes (`moe_host_expert_bytes`); `N` = explicit MiB; `0` = disable even with `--host-experts pool`. |
| `MOE_HOST_POOL_FRAC` | `25` | tuning | the auto percentage (1–100). |
| `MOE_HOST_POOL_POLICY` | — | **removed (r40)** | the pooled-table device-policy selector.  The device-policy fill (and the whole device-remap family) was deleted 2026-10-10; pooled tables always fill from the pinned pool with a master fallback.  The variable is ignored. |
| `MOE_HOST_POOL_PREFETCH` | `0` | opt-in | routing-driven speculative pool prefill; **off** because a long-prompt sweep showed it hurts until the pool nears full residency (`1` enables). |
| `MOE_HOST_POOL_PREWARM` | `1` | kill-switch | one-shot slot fill when a pool is built (ranked by the prefill tally / arena residents). |
| `MOE_HOST_POOL_BGFETCH` | `1` | kill-switch | low-priority background eviction prefetch; `0` is the synchronous path. |
| `MOE_HOST_POOL_DIO` | `0` | diagnostic | `1` = O_DIRECT fill (bypasses the page cache). |

### 1.3 The movable-boundary slab (block 06) — `GGML_CUDA_SLAB`, default **on**

**This is the delivery's central allocator.**  ONE slab per device is reserved and mapped **exactly once**
(`cuMemAddressReserve` + one `cuMemMap`), then split by a movable **boundary**: the work pool (the compute
buffer) below it, the MoE arena above it.  Growing the work region is a *boundary move inside the
already-mapped slab* — the lowest arena chunks change owner and the tables living there are evicted — so
**HIP is not called at runtime** and the work region's base VA never moves (a growing layout keeps every
tensor address).  It replaces the per-allocation VMM pool (`GGML_CUDA_COMPUTE_VMM`, **removed**) and the
"chunk the compute buffer so the arena can yield" plan: nothing is ever unmapped, so the ROCm sub-range
`hipMemUnmap` limitation stops mattering.

**r47 (issue #136):** the slab is **armed whenever the MoE expert cache is enabled, independent of
`--fit`**.  `moe_cache_preflight` runs an idempotent env parse before its cache gate, so `--fit off` (no
fit probe load, and therefore no backend init before the preflight) no longer leaves `g_slab_armed` false
with the cache on a raw `cudaMalloc` arena.  `GGML_CUDA_SLAB=0` is the only up-front opt-out; if the slab
is armed but cannot be **created** (no VMM, bad chunk geometry, or too little free VRAM for the reserve)
the run logs the reason and **streams** the experts from the host, and a one-time WARN in the
cache-enable path makes the (now bug-only) "cache on, slab not armed" state loud.  The fail-soft arena
yield now guards on the global `ggml_cuda_slab_enabled()` (not the failing device's slab) and the
stand-down is device-scoped.

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CUDA_SLAB` | **on** | kill-switch | `0` disables the slab.  The plain `cudaMalloc` path returns, together with the pre-slab failure mode (a later wide prefill aborting — the TODO #42 crash).  Since **r47** the slab is armed independent of `--fit`; a slab that is armed but cannot be created logs the reason and the cache streams from the host (issue #136). |
| `GGML_CUDA_SLAB_CHUNK_MIB` | `64` | tuning | the boundary-move granularity (must be a power of two).  Arena *allocations* use the finer VMM granularity (2 MiB), so this does not waste arena. |
| `GGML_CUDA_SLAB_RESERVE_MIB` | `max(8192, 25 % of the device)` | tuning | left OUTSIDE the slab for the weights / KV / draft / workspaces.  This is a **hard** constraint: ROCm will not unmap a sub-range of the slab's single mapping, so a value too small ENDS THE RUN (measured: 4096 -> a failed hipBLASLt workspace, exit 134). |
| `GGML_CUDA_SLAB_MIN_ARENA_MIB` | `2048` | tuning | hard cache floor: the slab is created only if it fits `estimated work buffer + this`.  Otherwise it declines, logs an error and the cache STREAMS from the host (`MOE_EXPERT_CACHE_MIB=0` semantics). |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `4096` (or `6144` under `-sm layer` with host experts, r36) | tuning | how much stays free outside the slab when `slab_extend` reclaims the reserve the model did not need.  A **MiB value, not a boolean** (the only knob for this; `0` disables the reclaim).  **Not slack**: it must cover every allocation made after that point, including ones we cannot redirect (see below).  Since r36 the MoE cache asks for at least **6144** when every cache table is unsplit and the tables span more than one device (`ggml_cuda_slab_headroom_at_least_mib`); the 255k `-sm layer -ts 59,41` run OOMed at 4096 and passes at 6144.  An explicit value still wins.  **3072 was trialled and rejected (2026-10-10):** the H2D ring moved inside the slab so it no longer spends ~1 GiB here, and 3072 measured **+8.5 % decode step rate** on the field config (arena residency 61.7 -> 64.9 %) — but on the §2.2 wide-prefill config (`--fit off`, IQ4_XS, `-ngl 99 -sm tensor -ncmoe 48 -ub 6144 -c 204800`) 4096 is clean (312 t/s, 0 corruption) while 3072 dies in `hipblaslt.cpp:164`.  The default stays 4096. |
| `GGML_CUDA_POOL_MIN_FREE_MIB` | `512` when the cache holds host-expert tables, else `0` | tuning / kill-switch | r36.  The workspace pool's **first** attempt refuses like a real OOM when it would leave less than this much VRAM free, so its existing flush + retry runs instead.  The retry has no floor, so it never aborts.  `0` disables the floor. |
| `GGML_CUDA_SLAB_RING_MIB` | `6144` | tuning | MiB cap on the H2D staging-ring region the slab carves as a **work sub-region** (r39).  The auto size is `slots x (largest host table + 512)`, and that estimate comes from `llama_model::max_host_weight_tensor_bytes()`, which can be a MERGED host tensor (every layer/expert in one -- measured 27 GiB on Qwen3.8-Flash-Next where the largest staged slice is 270 MiB), so this cap keeps the region proportional to a real table.  The ring lives between the narrow work floor and the wide work view, never in the arena, and is reclaimed on the prefill->decode drop.  `0` puts the ring back outside the slab (the raw `cudaMalloc` + G4 path).  The field tables (270 MiB) fit in `2560` and give nearly the same prefill as `6144`. |
| `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` | `50` | tuning / kill-switch | caps an **optional** raw allocation at this percentage of the device's *free* VRAM.  Since r39 the op-offload H2D staging ring lives **inside the slab** (see `GGML_CUDA_SLAB_RING_MIB` above), so this now effectively guards the **FA prefill staging arena** only (`GGML_CUDA_SLAB_RING_MIB=0` restores the ring to this path).  A refusal is graceful: the ring falls back to the serial path, the FA staging to the native K/V read (one-time warning) -- and on `fingon` that refusal is *faster* than allowing it (2502.2 vs 2315.1 t/s prefill), so this is an adaptive win, not a limitation.  `0` restores the pre-change unbounded behaviour.  **Inert unless the slab is active on the device**, so a non-host-expert / non-ROCm run is unchanged; it is a cap, never a reservation.  `GGML_SCHED_STAGE_MAX_MB` remains the hard ring override. |
| `GGML_COMPUTE_BUFFER_CHUNK_MIB` | `256` | tuning | the compute buffer is allocated in whole chunks + one spare (`(ceil(need/C)+1)*C`), so growth inside the chunk is free.  `0` falls back to `GGML_COMPUTE_BUFFER_MARGIN_PCT`.  Since **r37** the chunk applies **only when the slab is in use**: a dense model (no slab) falls back to the percentage margin, which removes the +739 MiB of issue #120. |
| `GGML_CUDA_POOL_CACHE_MB` | unset / `0` = unlimited (r46, block 06) | **opt-in tuning** | cap on the bytes of FREED buffers the **legacy** `ggml_cuda_pool_leg` keeps cached.  A long prefill's temporaries grow with the KV length and each smaller predecessor stays cached, so at ~256K the idle VRAM starves allocations outside the pool.  Before a new device allocation, once the cached bytes exceed this MiB value the pool frees them (never during graph capture).  Unset/0/negative/garbage = today's behaviour.  Only the legacy pool is touched (HIP uses it; CUDA's VMM pool is unaffected), and the release is a whole-cache flush, not largest-first eviction. |

**Why the cards sit near-full.**  `GGML_CUDA_SLAB_HEADROOM_MIB` is what is free *at the moment the slab is
extended*, and the steady state is that minus everything allocated afterwards — measured ~3.7 GiB of it: the
MTP draft buffer (762 MiB), the FA-QSA workspace (762 MiB), hipBLASLt's Tensile code objects and internal
workspace, and the workspace-pool high-water mark.  So ~200-350 MiB free is normal, and raising the headroom
raises it one-for-one (8192 -> ~4.3 GiB free) at the cost of an equal amount of arena.

> **Do not disable hipBLASLt to buy headroom.**  The ROCm BLAS stack allocates its own Tensile code objects
> and workspace behind the application's back, and that class of allocation CANNOT be routed into the slab.
> Measured with `ROCBLAS_USE_HIPBLASLT=0` at `GGML_CUDA_SLAB_HEADROOM_MIB=2048`: the output was CORRUPT
> (`////`), and at 4096 a 16k request returned one token then EOS.  It is not a supported configuration.

---

## 2. Scheduler and H2D staging (block 06)

The whole-shard H2D staging path for host-resident MoE expert tables.  Defaults below are the validated
configuration.

| variable | default | class | notes |
|---|---|---|---|
| `GGML_SCHED_STAGE` | on | kill-switch (value) | master switch for the staging path. |
| `GGML_SCHED_STAGE_MODE` | unset | tuning | staging mode; the delivery uses the `stage_d2d` form (redirect is corruption-prone). |
| `GGML_SCHED_STAGE_MIN_TOKENS` | `64` | tuning | width gate floor.  Below it the copy loop's gather branch serves the input instead. |
| `GGML_SCHED_STAGE_SLOTS` | unset | tuning | ring slot count. |
| `GGML_SCHED_STAGE_MAX_MB` | unset | tuning | hard ring budget cap; the slab-gated free-VRAM cap `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` (§1.3) also applies to a growth. |
| `GGML_SCHED_STAGE_TABLE_REF_MB` | `0` | tuning | `0` = width-only gate (r18 disabled the stale r7 table-size scaling; a positive value restores it for A/B). |
| `GGML_SCHED_EVENTS` | on | kill-switch | set `0` to drop the cross-device event ordering. This was the pre-r12 workaround for the `-sm layer` + host-expert garbage-prefill bug; it is **fixed** in block 06, so leave it on. |
| `GGML_SCHED_DEVGATHER` | on | kill-switch | device-gather path. |
| `GGML_SCHED_SYNC_GRAPH_INPUTS` | on | kill-switch | extra graph-input sync. |
| `GGML_SCHED_DEBUG` | unset | diagnostic | scheduler debug output. |
| `GGML_SCHED_DEBUG_REALLOC` | unset | diagnostic | log buffer reallocations. |
| `GGML_META_SPLIT_COPY` | unset | tuning | meta-backend split copy handling. |
| `GGML_META_SS_VERIFY` | off | diagnostic | `1` recomputes every outermost meta split-state cache hit and aborts on a stale entry (r25 added the 8-version + compacted cache; this verifies it). |
| `GGML_STAGE_META_REDIRECT` | off | diagnostic | meta redirection instead of the explicit d2d completion. |
| `GGML_STAGE_NO_RESTORE` | off | diagnostic | skip the redirect restore (debug only — unsafe). |
| `GGML_STAGE_GATHER_SCRATCH` | off | diagnostic | force the gather scratch path. |
| `GGML_MOE_CACHE_INPLACE` | on | kill-switch | r36.  With `-ncmoe` host experts, `0` keeps the cache-aware staging copy (arena + host, one kernel) instead of reading the routed table in place. |
| `GGML_MOE_CACHE_STAGE` | on | kill-switch | r36.  `0` restores the plain whole-table upload (pre-r36 behaviour, the full staging copy). |

---

## 3. MTP / speculative decoding

| variable | default | class | notes |
|---|---|---|---|
| `MTP_DRAFT_N_UBATCH` | `512` | tuning | caps the MTP draft's encoder-injection chunk.  The draft only drafts `n_max+1` tokens, so a 512-token chunk is plenty, and the ~1.6 GiB/device the wide chunk held at `-ub 8192` goes to the MoE arena.  `0` restores the target's `-ub` (no cap); a tiny cap (e.g. 4) costs prefill speed. |
| `LLAMA_SPEC_DRAFT_N_MAX_CLAMP` | `1` = clamp on | kill-switch | clamps `--spec-draft-n-max` to 15 (the recurrent snapshot bound). `0` escapes the clamp; **purity is only promised to 7**. |
| `LLAMA_MTP_DRAFT_OP_OFFLOAD` | unset | tuning | op-offload behaviour for the draft model. |
| `LLAMA_MTP_SPARSE` / `LLAMA_MTP_SPARSE_DECODE` | unset | tuning | sparse MTP variants. |
| `LLAMA_MTP_INGEST_TAIL` | unset/`0` = every row | **opt-in tuning** (r52, block 02) | feeds the MTP draft only the last N positions of each prompt (the server reports each prompt end via `common_speculative_set_prompt_end()`); generation rows are always fed.  The draft loses look-back over the skipped prefix only, so output is unchanged, but draft acceptance drops: on a ~31k-token prompt, `4096` measured prefill **+4.8 %** and decode **-4.4 %** (acceptance 0.821 -> 0.715).  Kept opt-in because it is a speed/look-back trade, not a pure win.  Tools other than the server never report a prompt end, so there the variable has no effect. |
| `LLAMA_DRAFT_SCHED2` | **on** (r65, block 11) | kill-switch | runs the MTP draft context's graphs **with outputs** on a second scheduler (`llama_context::sched_alt`), so the no-output ingest graph and the 1-token draft-step graph both stay allocated and are reused instead of re-split/re-allocated every cycle.  Needs a per-scheduler meta view slot and a `slab_compute_plain` device-iface entry (the second scheduler's small COMPUTE buffer bypasses the slab).  On Qwen3.8-Flash-Next + shared MTP (`-sm tensor`, 2 GPUs) the draft's graph reallocations drop 1068 -> 34 and generation is about +1.5 %; kill switch `0` restores the single scheduler. |
| `GGML_LF_DFLASH_DEV` | **on** | kill-switch | DFlash device-resident layer-input features (F1).  `0` uses the host path.  Default-on since r26; if the device buffers cannot be allocated the target warns once and stays on the host path for the rest of the context.  Since r53 F1 also covers several sequences (`--parallel >= 2`); with host-resident MoE experts (`-ncmoe`/`-cmoe`, `llama_model_moe_host_expert_dev_count(...) > 0`) several sequences stay on the host path and log a warning (the movable-boundary slab's device reads, issue #127 MoE form).  One sequence is unchanged. |

The tuned adaptive-MTP controller is selected with **`--spec-type draft-mtp-adaptive`** (a CLI flag, not
an environment variable).  Plain `--spec-type draft-mtp` does not use it.

---

## 4. Model loading and context

| variable | default | class | notes |
|---|---|---|---|
| `LLAMA_MMAP_HOST_EXPERTS` | — | **removed (r33)** | the legacy alias for `--host-experts`.  The pageable `--host-experts mmap` mode was dropped in r33 (issue #116): a no-XNACK GPU (gfx1201 reports `XNACK enabled: NO`) cannot read a pageable host master in a kernel, so `-ncmoe`/`-cmoe` experts are always pinned (`ROCm_Host`).  The variable is now ignored.  The bounded pinned DIO host tier that replaces the reclaimable-page-cache idea is the `archive/work/host-expert-dio-cache/` campaign. |
| `LLAMA_LOAD_RELEASE_MMAP` | **on** (block 06, r49) | kill-switch | unmaps each model-mapping range right after the tensor is copied to its device buffer, and skips the up-front whole-file prefetch, so a `-ncmoe` load does not hold the full file mapping and the pinned host expert master at the same time.  `madvise(MADV_DONTNEED)` is not enough here (the ROCm runtime re-faults those pages on the next pageable H2D copy), so the range is munmapped; the exact ranges a host-pointer tensor points into (a tied embedding is loaded host-side and device-side from the same range) are preserved.  `0` restores the pre-r49 prefetch-then-unmap-at-end behaviour.  Skipped under `--load-mode mlock`/`--mlock`, `--check-tensors` and lazy tensors. |
| `LLAMA_TENSOR_HOST_BUFT` | unset | tuning | host buffer type for overridden tensors. |
| `LLAMA_DEVICE_INPUT` | off | **opt-in** | device-side input handling. |
| `LLAMA_OUTPUT_PINNED` | **on** (r65, block 11) | kill-switch | when the output device is a meta (tensor-split) device with no host buffer type of its own, `output_reserve` uses the first simple device's pinned host type for the output buffer (logits/embeddings/MTP hidden state).  Without it those per-step D2H reads land in pageable memory and each blocks the host; the output buffer is never scheduled on, so its `.device` does not matter.  About +2.7 % MTP decode on Flash-Next (`-sm tensor`, 2 GPUs) and about +4.7 % on the dense 27B; `0` restores the CPU-buffer fallback.  The meta device's own `get_host_buffer_type` is unchanged. |
| `LLAMA_DROP_COMPUTE_BUFFERS` | **on** (every tool) | tuning / kill-switch | drop the wide-prefill compute layout at the prefill→decode transition so a wide `-ub` and a large arena coexist (`-ub 8192` cache-auto cli decode 45.6 → 78.7 t/s).  Follows `common_params::drop_compute_buffers`; set the env to `0`/`1` to override.  **A server may drop too** (this was cli-only until r22): the movable-boundary slab reclaims a later wide layout with a boundary move, which is verified on `wide1 → short → wide2` at `-ub 8192` (0 aborts, all coherent).  `0` is only needed where the compute layout CANNOT be reclaimed — i.e. with `GGML_CUDA_SLAB=0`. |
| `LLAMA_DROP_EXTRA_RESERVE_MIB` | `0` | tuning | when the drop fires, VRAM held out of the arena per device for a later compute growth.  Default `0` — the compute-buffer margin covers the small post-drop growth and the arena's layer-uniform re-size absorbs fragmentation, so a reserve only costs arena. |
| `LLAMA_LAZY_BUF_MB` / `LLAMA_LAZY_IO_THREADS` / `LLAMA_LAZY_READER_STATS` | unset | tuning / diagnostic | lazy-mode buffer size, reader thread count, reader statistics. |
| `LLAMA_KV_N_PAD_MIN` | `256` | tuning (**opt-in**) | raises the n_kv padding floor (r25).  n_kv is padded to this many cells, so a larger value keeps the per-device tensor-split graphs constant for longer (the PR measured `1024` as the useful value, `4096` gave nothing more).  Default unchanged. |
| `GGML_READBACK_SYNC` | **on** | kill-switch (block 06, r43) | completes the per-ubatch device->host logits/embd readbacks before the next ubatch's graph reuses the same device output tensor.  Required because ROCm's pinned async D2H copy is not ordered against the following compute, which otherwise lets an earlier ubatch's host rows receive a later ubatch's data (multi-ubatch per-token outputs: perplexity / KLD).  `0` restores the unordered readback for A/B.  A single-ubatch decode is unaffected either way. |
| `LLAMA_STATE_GET_MERGE_GAP` | `262144` (256 KiB) | tuning (r56, block 11) | in `llama_io_write_host` (the state / prompt-cache save), consecutive deferred `write_tensor` ranges of one tensor whose gaps are at most this many bytes are read as one `ggml_backend_tensor_get` span and scattered on the host; contiguous ranges are read straight into the output.  Under concurrent decoding a slot save was ~8,000-10,000 short synchronous reads on ROCm (340-400 ms); merged it is one get per tensor (25-65 ms).  The saved bytes are identical.  `0` restores the per-range reads. |

---

## 5. CPU MoE offload threading

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CPU_MOE_OFFLOAD_THREADS` | derived (block-06 cap) | tuning | overrides the thread cap for `-ncmoe` offloaded-MoE decode.  The cap only covers that case. |
| `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD` | off | diagnostic | disables the single-thread path for tiny graphs. |

Thread sizing is **not** an environment variable and is not optional: `/usr/local/bin/pin_gpu_irqs.sh`
puts the GPU IRQs on the top `NUM_GPUS` cores, and the default `-t` puts the worker pool there and
starves the GPU.  Size `-t` (or `--cpu-mask` with `-t` inside the mask) to leave those cores free.
See `AGENTS.md`.

---

## 6. Multi-GPU comms

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `hybrid` on Linux | tuning | `hybrid` (NCCL for large, internal for small) \| `internal` \| `nccl` \| `ce` (SDMA copy-engine, 2 GPU).  Unrecognised values warn and fall through to the platform default.  `nccl` is **not** bit-identical under `-sm tensor` (the internal AR BF16-round-trips), so use it as a smoke comparison only. |
| `GGML_CUDA_AR_SPIN_TIMEOUT_MS` | `2000` (r59, block 01) | **diagnostic / tuning** | the internal all-reduce's **soft** spin deadline in ms, measured with `wall_clock64()` at the device's wall-clock rate.  A peer wait that exceeds it is a late peer (a driver queue stop), not a lost one: the kernel reports it once and **keeps waiting**, so the reduction still completes.  `0` disables the soft check (and, with it, the hard cap below).  Before r59 the budget was a `clock64()` cycle count with a 20 ms default; on gfx1201 `clock64()` can step backwards while a queue is stopped, so the budget wrapped and a slightly-late peer was treated as lost. |
| `GGML_CUDA_AR_SPIN_HARD_MS` | `20000` (r59, block 01) | **kill-switch / safety** | the internal all-reduce's **hard** spin cap in ms.  If a peer wait exceeds it the kernel treats the peer as lost: it sets the host poison flag and the next call falls back to the meta-backend butterfly.  This is the last-resort bound that keeps an unbounded spin from making the compute queue non-removable (the MES `REMOVE_QUEUE` hazard).  `0` = never poison (unbounded).  A value below the soft budget is clamped up to it. |
| `GGML_CUDA_AR_CE_SCRATCH_MIB` | `0` = unlimited (r47, block 12) | **opt-in tuning** | budget for the copy-engine (`GGML_CUDA_ALLREDUCE=ce`) all-reduce scratch: each of the three per-rank staging buffers is `n_rank * ceil(ne/n_rank) * 2 B`.  Above it, CE returns `false` **before** any collective phase and the dispatcher falls back to the internal pipeline for that call; the fallback is sticky after the first failed grow, so an allocation failure also degrades cleanly instead of aborting.  `0` keeps the pre-r47 unbounded behaviour. |
| `GGML_CUDA_P2P` | off | tuning | enables peer access.  Note VMM allocations need explicit `cuMemSetAccess` for P2P (see `FOLLOWUP-compute-arena-chunking.md`). |
| `GGML_CUDA_AR_SMALL_BLOCKS` | `4` (r46, block 12) | kill-switch / tuning | block count for a **small** internal all-reduce (decode/verify, `<= GGML_CUDA_AR_SMALL_BYTES`).  Each block signals and polls its own arrival slot, so fewer blocks shorten the call.  `GGML_CUDA_AR_SMALL_BLOCKS=8` restores the pre-r45 launch; the fused stage and larger messages always keep 8. |
| `GGML_CUDA_AR_SMALL_BYTES` | `65536` (r46, block 12) | tuning | the small-message threshold in bytes for the knob above. |
| `GGML_CUDA_AR_P2P` | off (r46, block 12) | **opt-in** | 2-device, not-fused P2P push all-reduce: each device writes its wire data and arrival token into the peer's fine-grained VRAM and reads locally, replacing the host ring and the three all-thread `__threadfence_system()` calls with a store-count drain plus one system-scope release store.  Falls back to the host ring if peer access / fine-grained allocation fails or there are not exactly 2 devices.  Bit-identical; MTP-neutral on the reporter's box, +2.6 % one-token decode. |
| `GGML_CUDA_AR_P2P_LITE` | `1` (r46, block 12) | kill-switch | `0` keeps the full `__threadfence_system()` fences on the P2P path (A/B only). |
| `GGML_CUDA_AR_KER_EVENTS` | `0` = off | **opt-out / diagnostic** (r52, block 01) | the chunked kernel path no longer records `ev_pool[*][slot].ker` after every all-reduce (each record is a ROCm marker packet the GPU processes after every AR kernel); slot reuse on that path is already ordered on the GPUs by the arrival tokens, so `acquire_slot` syncs only when the slot's last user recorded the event.  `1` restores per-AR recording for A/B; pacing (`GGML_CUDA_AR_PACE=1`) also forces it. |
| `GGML_CUDA_AR_P2P_LARGE` | off (r52, block 01) | **opt-in** | 2-device copy-engine path: each device pulls the peer's contribution straight into `dev_tmp` with `hipMemcpyPeerAsync` (one PCIe crossing) instead of bouncing both halves through pinned host memory (two).  Same add kernel on the same bytes, so bit-identical; a 10.5 MB exchange measured 808 -> 437 us on the reporter's pair, +1.3 % on a 31.5k prefill.  Unset/`0` keeps the host-bounce path. |

---

## 7. Fusion A/B switches

Each of these disables exactly one optimization so it can be bisected.  **The feature is on by default;
the variable turns it off.**  Read the idiom column carefully — setting a presence-only switch to `0`
still disables the feature.

**The fusions are ON, and stay ON.**  The one known stand-down path — the cache-aware guard against
per-table slab eviction (`TODO.md` #50, evidence in `archive/work/slab-ring-region/FINDINGS-numerics.md` F3) — is a
pre-existing property of the arena, not a reason to turn them off.  Measured on the field config
(Qwen3.8-Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`, 32 358-token prompt, `--spec-type none`, warm):

| | prefill t/s | decode t/s |
|---|---:|---:|
| fusions on (default) | 945-972 | 61.3-65.2 |
| `GGML_CUDA_DISABLE_FUSION=1` | 845-890 | 57.5-65.3 |

so a full disable costs **~9-10 % prefill and ~6 % decode**, and the fusion-off prefill (845-890) is
*below* the unpatched stock reference (922) — it gives back the delivery's prefill win.

**To restore the full speed** after an A/B or a debug recipe, make sure none of the switches below is set:
unset `GGML_CUDA_DISABLE_FUSION` and the rest of §7.2/§7.3, and unset (or `=1`) every `GGML_CUDA_FUSE_*` in
§7.1.  A clean environment is already the full-speed configuration — there is nothing to add, only
switches to remove.  (`GGML_CUDA_DISABLE_FUSION=1` is a **sledgehammer**: it also skips the
`graph_optimize` alloc-dependency pass and its shared-expert node reorder, so its cost is not the cost of
any single fusion.)

### 7.1 `GGML_CUDA_FUSE_*` — presence of the variable with a value, `0` disables

`GGML_CUDA_FUSE_ADD_RMS_Q8`, `GGML_CUDA_FUSE_CPY_BATCH`, `GGML_CUDA_FUSE_GATE_BETA_VERIFY`,
`GGML_CUDA_FUSE_GDN_BETA_SIGMOID`, `GGML_CUDA_FUSE_GDN_CONV_VERIFY`, `GGML_CUDA_FUSE_GDN_GATE`,
`GGML_CUDA_FUSE_GLU_Q8_1`, `GGML_CUDA_FUSE_HC_COMBINE_MIX`, `GGML_CUDA_FUSE_Q8_1_VERIFY`,
`GGML_CUDA_FUSE_SIGMOID_MUL_ADD`, `GGML_CUDA_FUSE_SWIGLU_MMQ`.
These use `getenv(...) == nullptr || atoi(...) != 0`, so unset = **on** and `0` = off.
(Git history note: these were originally *opt-in* `=1` switches; they are now default-on kill-switches.)

The same idiom — and therefore also default-on despite the name — covers
`GGML_CUDA_DISABLE_IDX_RELU_SUM`, `GGML_CUDA_CONCAT_VERIFY` and (r46, block 15) `GGML_CUDA_FUSE_GDN_STATE_GATHER`.

### 7.2 `GGML_CUDA_DISABLE_*` — value-gated (`=1` disables, `0` keeps it on)

`GGML_CUDA_DISABLE_FUSED_ADDMUL`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_DISABLE_GDN_CPY`,
`GGML_CUDA_DISABLE_HC_FUSION`, `GGML_CUDA_DISABLE_L2_NORM_PAIR`, `GGML_CUDA_DISABLE_MOE_DOWN_FOLD`,
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`, `GGML_CUDA_DISABLE_MWR`, `GGML_CUDA_DISABLE_NORM_Q8_1`,
`GGML_CUDA_DISABLE_NORM_ROWS`, `GGML_CUDA_DISABLE_ROPE_SETROWS`, `GGML_CUDA_DISABLE_SCALE_UNARY`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`, `GGML_CUDA_DISABLE_SNAKE`, `GGML_CUDA_DISABLE_SSM_CONV_IN`,
`GGML_CUDA_DISABLE_SSM_PRESCAN`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN`.

`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` is the A/B switch for the byte-identical fused shared-expert band
(`1 <= nt <= 8`).

### 7.3 `GGML_CUDA_DISABLE_*` — **presence-only** (any value, including `0`, disables)

`GGML_CUDA_DISABLE_CONV_FUSION`, `GGML_CUDA_DISABLE_MMID_512`, `GGML_CUDA_DISABLE_MMQ_ROUTED`,
`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`, `GGML_CUDA_DISABLE_MMVQ_MOE_BAND`,
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE`, `GGML_CUDA_DISABLE_ROPE_SET_ROWS`,
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION`.

### 7.4 Other kernel tuning

| variable | default | notes |
|---|---|---|
| `GGML_CUDA_MMQ_J_MAX` | tuning | MMQ `J` tile width cap. |
| `GGML_Q6_COMPACT_J` | tuning | Q6 compact-`J` variant. |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS` | off (**opt-in**, RDNA3.5 only) | the qwen4exp weighted-down single-token chain. |
| `GGML_PAIR_OFF` / `GGML_PAIR_DENSE_OFF` / `GGML_PAIR_2X` | off | paired-kernel variants. |
| `GGML_CUDA_SPLICE_GATHER`, `GGML_CUDA_SCALE_UNARY`, `GGML_CUDA_HC_MIX_BAND`, `GGML_CUDA_HC_MIX_PREQ`, `GGML_CUDA_DISABLE_HC_COMB`, `GGML_CUDA_DISABLE_HC_MIX`, `LLAMA_FUSED_HC_MIX`, `LLAMA_FUSED_HC_COMBINE`, `LLAMA_FUSED_DSV4_HC_PRE/POST`, `LLAMA_HC_MIX_BF16`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100/GFX1201`, `GGML_LF_FAST_TOPK`, `LLAMA_INDEXER_NOBLOCK`, `LLAMA_INDEXER_NOGROUP` | off / default-on | model-specific fused chains. A/B only. |
| `GGML_CUDA_DISABLE_VERIFY_GRAPHS`, `GGML_HIP_GRAPH_FORCE_UPDATE` | off | HIP graph-capture controls. |
| `GGML_HC_UP_V2` | on | kill-switch | `0` restores the row-at-a-time BF16 `hc_mix` up/collapse kernel (r27); v2 is bit-identical and 4-21 % faster on the op. |
| `GGML_CUDA_GRAPH_MEM_GEN` | on | kill-switch | `0` disables the per-device graph memory generation (r25): a graph that captured pool temporary / FA-staging / H2D-ring memory freed since is otherwise recaptured. |
| `GGML_CUDA_Q8_1_ARENA_FREE_OLD` | off | kill-switch (**opt-in**) | `1` restores the immediate `cudaFree` of a grown-out Q8_1 input arena (the r24 behaviour, A/B only): captured decode/verify graphs keep pointers into it, so this reintroduces the `quantize_q8_1` page fault. |
| `GGML_CUDA_BF16_SRC1_CACHE` | on | kill-switch | r36.  `0` disables the per-graph reuse of the BF16 copy of a shared F32 src1 across the qwen4exp hc-mixer cuBLAS BF16 GEMMs (prefill only, `>= 64` columns); bit-identical either way. |
| `LLAMA_HC_PIN_BLOCK_OUT` | off | kill-switch (**opt-in**) | `1` restores pinning every qwen4exp layer's `block_out` as a prefill graph output (r24 behaviour; costs ~1.9 GiB of the `-ub 2048` compute buffer). |
| `LLAMA_HC_MIX_PLANAR` | on | kill-switch | `0` restores the interleaved HC_MIX output (r24) instead of the planar layout (r25; BF16 CUDA path only). |
| `LLAMA_CONV_TAIL_CONT` | off | kill-switch (**opt-in**) | `1` restores the `cont` + `cpy` conv-state tail copy instead of the direct strided 2D memcpy (r25). |
| `GGML_FORCE_NO_INTEGRATED` | off | ignore integrated GPUs when picking devices. |
| `GGML_CUDA_ASYNC_INPUT` | **on** (r68) | **kill-switch** | async graph-input uploads: a `set_tensor` of `<= 1 MiB` into a COMPUTE buffer stages through a per-device 16 MiB pinned ring with a done-event ordering, so the host does not block on every copy; any setup failure falls back to the synchronous path.  **Default-on since r68** (issue #151): the per-copy `cudaStreamSynchronize` is invisible on a fast host and measured ~1.4 % of decode on a slow one (and +0.21 % even here).  `=0` restores the synchronous path.  The historical single-run impurity that kept it opt-in (see `archive/work/decode-host-overhead/`) is closed by the done-event ordering; r68 re-validated byte-identical (prefill-logit, coherence, MTP n3000). |
| `GGML_CUDA_BOUNCE_MIN` | `1` = every size | **kill-switch / tuning** (r56, block 09) | `buffer_get_tensor`, `buffer_set_tensor` (not weight buffers) and `buffer_get_tensor_2d` copy pageable host data through a reused 64 MiB pinned chunk per device.  A pageable device<->host copy is slow on ROCm at any size (~25 us 1-D / ~57 us 2-D on gfx1201), and the server's context checkpoint + `--cache-ram` slot saves make thousands of them.  A pinned host pointer keeps the direct copy; a failed chunk allocation leaves the copy direct and is not retried.  `0` (or unparsable) disables the bounce; a byte count is the minimum copy size that bounces.  HIP/CUDA only — the pointer-attribute helper is not defined for MUSA, which stays direct. |
| `GGML_CUDA_MOE_RPB_SHORT` | unset = on (RDNA4) | **kill-switch / tuning** (r52, block 10) | rows per block for a short-K (`< 8` K-blocks per row) `MUL_MAT_ID` expert GEMM: RDNA4 uses **8** instead of `ceil(8 / K-blocks)` (the Flash-Next q2_0 down split under `-sm tensor` measured 43.0 -> 29.8 us per 4-token verify call).  `0` restores the old rule; `2`/`4`/`8` force a value on any architecture; other values are ignored. |
| `GGML_CUDA_TOPK_SORTED` | unset = on | **kill-switch** (r52, block 08) | `topk_moe`'s argmax uses DPP lane exchanges (`ggml_cuda_xor_lane`) and per-lane sorted 64-bit keys for the gating outputs (softmax/sigmoid/sqrt-softplus, no bias, no delayed softmax, power-of-2 candidates per lane).  Same ids, weights and order (ties to the lower index).  `0` restores the per-round scan; the argmax rounds drop 7.1 -> 3.8 us per 512-expert/10-used call. |
| `GGML_CUDA_GRID_STALL_PAD` | on, HIP only (r46, block 15) | kill-switch | `0` disables the gfx1201 grid-size dispatch workaround: a launch whose block count is a multiple of 256 (or work-item count a multiple of 65536) waits ~6-11 us on gfx1201 (ROCm/legacy-rocm-build#6689), so `mul_mat_vec_f`/`mul_mat_vec_q_ksplit`/`mul_mat_vec_q_moe` add one returning block.  Bit-identical; delete the helper and its call sites once AMD ships the fix. |

---

## 8. Flash attention / QSA (qwen4exp)

| variable | default | class | notes |
|---|---|---|---|
| `GGML_HIP_FA_BAND_WIDE` | on | kill-switch | `0` disables the wide band. |
| `GGML_HIP_FA_BAND_WMMA` / `GGML_HIP_FA_BAND_WMMA_SPLIT` / `GGML_HIP_FA_BAND_WMMA_GQA_MAX` | unset | tuning | band kernel selection.  `GQA_MAX` (default `12`) is the largest GQA ratio the band folds into its `ncols2 = 8` groups; `<= 8` keeps the old whole-group fold (issue #133). |
| `GGML_CUDA_FA_WMMA_256` / `GGML_CUDA_FA_WMMA_MAX_HEAD` | unset | tuning | WMMA tile / head caps. |
| `GGML_CUDA_FA_KV_NATIVE` / `GGML_CUDA_FA_MASK_SKIP` | unset | tuning | native-KV and mask-skip paths. |
| `GGML_CUDA_FA_STAGE_MAX_MB` | unset | tuning | FA staging budget. |
| `GGML_CUDA_FA_STAGE_PREALLOC` | **on** | kill-switch | `0` restores the 1.25x growth ladder (issue #138).  The prefill staging arena is allocated once at the F16 size of the K/V view roots, capped per operand by `GGML_CUDA_FA_STAGE_MAX_MB` and only while `optional_alloc_within_free_cap` allows it; an unbounded cap (`GGML_CUDA_FA_STAGE_MAX_MB=0`) passes no hint. |
| `LLAMA_QSA_OFF` | off | kill-switch | disables QSA. |
| `LLAMA_QSA_SPARSE_FA` | on | kill-switch | `0` = the dense masked path. **This is the QSA oracle**: the `W=1..8` matrix is blind to a width-uniform corruption, so measure the perplexity ratio against `LLAMA_QSA_SPARSE_FA=0` when touching the QSA op. |
| `LLAMA_QSA_KEYS_ONLY`, `LLAMA_QSA_PLE_HOSTGATHER`, `LLAMA_QSA_DENSE_SHORTCUT`, `LLAMA_QSA_SCORE_BOUNDS`, `LLAMA_QSA_SCORE_STRIP`, `LLAMA_QSA_SCORE_WMMA`, `LLAMA_QSA_SCORE_WMMA_MB`, `GGML_CUDA_QSA_IDENTITY`, `GGML_CUDA_QSA_SLICES`, `GGML_CUDA_QSA3`, `GGML_CUDA_QSA_INDEXER_CACHE`, `GGML_CUDA_QSA_INDEXER_SCORE`, `GGML_QSA_DERIVED_BIAS`, `GGML_QSA_DERIVED_VIS`, `GGML_QSA_SCORE_MEM`, `LLAMA_KQ_MASK_DERIVED` | off / unset | tuning | QSA variants and indexer tuning. |

The QSA op has a CPU oracle: `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum).

---

## 9. GDN / SSM (block 02)

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CUDA_GDN_CHUNKED` | on | kill-switch | compared as a **string** against `"0"` — literally `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel (correct, bit-identical, slow). The chunked prefill is K-independent with threshold `max(K>16?K:16, n_rs_batch)`. |
| `GGML_CUDA_GDN_CHUNKED_BF16` | on | kill-switch | BF16/WMMA chunked GDN variant, **default-on since r30** (RDNA3/RDNA4, `S_v == 128` prefill). Near-lossless, not bit-exact (bf16 operands, fp32 accumulation; ~1e-5 op NMSE; prefill mean KLD 0.0007 / 0.00005 / 0.00015 on gfx1201 / gfx1100 / gfx1151, same-top-p >= 98.7 %). `=0` runs the fp32 chunked kernel. r29 had defaulted it off as a workaround for the n_seqs > 1 `A_sc` stride aliasing bug (issue #113), fixed in r30. |
| `GGML_CUDA_FUSE_GDN_STATE_GATHER` | on (r46, block 15) | kill-switch | in the one-sequence `<= 16`-token decode/verify band the graph skips the per-GDN-layer recurrent-state `GET_ROWS` and the sequential kernel reads `s0` from the cache row directly (`GET_ROWS` is a plain copy, so bit-identical).  Uses the §7.1 presence idiom (unset = on, `0` = gather as before). |
| `GGML_CUDA_GDN_GATHER_INDEX` | unset = on (r68) | **kill-switch / diagnostic** | replaces the `O(gathers × nodes)` `ggml_cuda_gdn_state_gather_deferrable` forward walk with one index per graph evaluation, built **lazily on the first candidate** so a non-GDN graph pays only one pointer check per `GET_ROWS`.  `0` = the per-gather walk; `2` = both, `GGML_ABORT` on a different answer.  Conservative relative to the walk (a disagreement can only drop a deferral, never add one), so the kernel path is unchanged.  Issue #151: 0.63 ms/graph less host work on gfx1201 (13 %), ~+0.3 % decode on a slow host. |
| `GDN_DBG_SKIP_KKT`, `GDN_DBG_SKIP_SCAN` | off | diagnostic | skip stages (debug). |

---

## 10. MMB (block 08)

The `mmb` campaign is part of the delivery.  `GGML_CUDA_MMB` defaults to `1`; the rest are tuning knobs
for tile/shape selection and cache sizing.  Treat them as internal: the validated configuration is the
default, and they exist to reproduce measurements from `benchmarks/`.

`GGML_CUDA_MMB`, `GGML_CUDA_MMB_BF16W`, `GGML_CUDA_MMB_CACHE`, `GGML_CUDA_MMB_CFG`,
`GGML_CUDA_MMB_DENSE`, `GGML_CUDA_MMB_DENSE_TYPES`, `GGML_CUDA_MMB_DOWN16`, `GGML_CUDA_MMB_F32SPLIT`,
`GGML_CUDA_MMB_F32SPLIT_MIN_K`, `GGML_CUDA_MMB_F32SPLIT_MIN_M`, `GGML_CUDA_MMB_GLU`,
`GGML_CUDA_MMB_GLU_THRESH`, `GGML_CUDA_MMB_HC16`, `GGML_CUDA_MMB_IQ3XXS`, `GGML_CUDA_MMB_MIN_T`,
`GGML_CUDA_MMB_RDNA3`, `GGML_CUDA_MMB_ROUTED`, `GGML_CUDA_MMB_ROUTED_THRESH`, `GGML_CUDA_MMB_SHADOW`,
`GGML_CUDA_MMB_SHADOW_MB`, `GGML_CUDA_MMB_TALL`, `GGML_CUDA_MMB_TALL_MIN_M`, `GGML_CUDA_MMB_TILE`,
`GGML_CUDA_MMB_TINY_M`, `GGML_CUDA_MMB_TINY_TT`, `GGML_CUDA_MMB_TYPES`, `GGML_CUDA_MMB_LOG`,
`GGML_CUDA_MMB_MARK_LOG`, `LLAMA_MMB_CVT_LOG`, `LLAMA_MMB_F32SPLIT`, `LLAMA_HC_BLK16`,
`LLAMA_HC_GATEMIX`, `LLAMA_HC_RES16`.

---

## 11. Diagnostics and profiling

None of these change the computed result on the default path.

| variable | notes |
|---|---|
| `GGML_CUDA_OP_TIMING` | per-op GPU timing. |
| `GGML_CUDA_GCDBG` | CUDA-graph debug. |
| `GGML_STREAMDBG` | stream debug. |
| `GGML_ALLOCATOR_DEBUG` | build-flag companion for allocator tracing. |
| `GGML_CUDA_MWR_DEBUG`, `GGML_CUDA_HC_BLK16_DEBUG`, `GGML_CUDA_IDX_RELU_SUM_LOG`, `GGML_CUDA_FUSE_CPY_BATCH_DEBUG`, `GGML_CUDA_CONV_DEBUG`, `GGML_CUDA_CONV_FUSION_MULTI`, `LLAMA_HC_CN_DEBUG`, `LLAMA_HC_GATEMIX_DEBUG`, `LLAMA_QSA_SCORE_STRIP_DEBUG`, `GGML_CUDA_QSA_DEBUG` | feature-specific debug output. |
| `FA`, `KV`, `RS`, `GDN_DBG_A`, `GDN_DBG_P`, `GDN_DBG_ONES`, `GDN_DBG_DUMP`, `GDN_DBG_SAVE` | **internal probes.** Single/double-letter or `_DBG_` names used by debug dump blocks. Never set these in production — two of them take a value that selects a dump *mode*. |

---

## 12. Non-ggml environment (HIP/ROCm runtime)

These are not llama.cpp variables but they control how the delivery runs.

| variable | notes |
|---|---|
| `HIP_VISIBLE_DEVICES` | which GPUs are visible, in order. Part of every validated command (`0,1` or `0,1,2`). |
| `LD_LIBRARY_PATH` | must include the matching ROCm runtime libs, e.g. `/opt/rocm-7.14.1-gfx120X/lib`, or the build will not start. |
| `ROCR_VISIBLE_DEVICES` | lower-level alternative to `HIP_VISIBLE_DEVICES`; prefer `HIP_VISIBLE_DEVICES`. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | forces managed allocations; not used in this repo's validated configurations. |

There is **no** environment variable for thread pinning; use `-t` / `--cpu-mask` as described in §5.

---

## 13. Retired / superseded

| variable | status |
|---|---|
| `MOE_EXPERT_CACHE_RESERVE_MIB=8192` | the pre-r20 server workaround. **No longer needed** — the default `1024` plus the r20 compute-buffer slack and the fail-soft arena yield cover it. |
| `MOE_ARENA_HEADROOM_MIB` | tried and **removed**. A flat arena headroom changes nothing: the failure is a realloc needing a block bigger than the one just freed, so the slack has to be on the *allocation*, not the arena. |
| `MTP_DRAFT_N_UBATCH=4` | was a workaround for the same failure; a cap is now the default (`512`), so this is just an over-tight value. |
| `GGML_HIP_NO_VMM` (build option, not runtime) | HIP VMM is **off by default** (`ON` = do not use VMM).  Whether to enable it is an open investigation — see `archive/work/moe-cache-autosize/FOLLOWUP-compute-arena-chunking.md`. |

---

## 14. Build-time gates (not environment variables)

These are compile-time, because a device-side runtime gate does not work cleanly here.

| flag | default | what |
|---|---|---|
| `-DGGML_HIP_NO_DPP_XOR` | off (DPP on) | restores `__shfl_xor` for the wave32 warp reductions on RDNA3/RDNA4 (PR #110's DPP lane exchange, folded into block 15).  The DPP path is **bit-identical** and speeds up the HC_MIX / Flash-Next band (measured on gfx1201 / ROCm 7.14: BF16 HC_MIX 58.84 -> 45.26 us at 4 tokens, 100.00 -> 62.55 us at 8, i.e. +23 % / +37 %; bandwidth-bound matmuls unchanged).  **There is deliberately no runtime env var**: a device-side runtime gate was implemented and A/B-tested but reproducibly wedged the FA prefill at ~4096 tokens with MTP, so it was dropped. |

The `llama` C++ target compiles with `-Wshadow=local` under **GCC** and plain `-Wshadow` under **clang**
(block 00).  The container build's host compiler is GCC (the `rocm/dev-ubuntu` image default, unlike the
local `~/bin/build-llama-rocm-714`, which forces clang), and GCC's `-Wshadow` additionally reports
"declaration shadows a member" / "hides constructor" for the ubiquitous `Foo(int x) : x(x) {}` idiom.
`-Wshadow=local` keeps the class the gate exists for (a local shadowing a local) and drops that
member/global output; clang has no `=local` form.  The container's `RUN cmake` step zeroes ccache stats
(`ccache -z`) before the build, so its trailing `ccache -s` reports the rate of that build rather than
the lifetime average.

---

## Maintaining this file

The tables were derived from the code, so re-derive rather than trust them after a block change:

```bash
# every variable the delivery adds or repoints, with the idiom used
for f in ~/llama-cpp-rdna-boosts/patches/*.patch; do
  grep -hoE 'getenv *\( *"[A-Za-z0-9_]+"|GGML_ENV_STR *\( *"[A-Za-z0-9_]+"|env_int *\( *"[A-Za-z0-9_]+"' "$f"
done | grep -oE '"[A-Za-z0-9_]+"' | tr -d '"' | sort -u
```

Then, for each name, grep the tree for its usage.  **Do not** classify by name alone: `*_DISABLE_*` is
not consistently value-gated, and `*_FUSE_*` semantics changed from opt-in to kill-switch.  A variable
that is genuinely new should follow the repo's conventions: default-on if it helps (with the variable as
a *kill-switch* that restores the old behaviour), default-off only for a known-risky correctness
workaround, and a first line in the relevant block's `patches/README.md` note.
