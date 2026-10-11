# `slab-arming-and-ce-failsafe` — RESOLVED (r47, 2026-10-16)

**Status: PROMOTED.**  All items were folded into release **`v16-a55e952b8-r47`** (canonical block-15 tip
`6daf815693704a214246bea7b60523ddff00a171`, net tree
`0099dd85130db65cbfbcfd738abfc57566bffe6f`); `scripts/validate-set.sh` green (strict 16/16 `git am`).
The campaign was seeded for a cold session and lived at `wip/slab-arming-and-ce-failsafe/`; this record is
its archived resolution.  The plan and the gdb/field evidence stay in [`README.md`](README.md) and
[`REPRO-AND-EVIDENCE.md`](REPRO-AND-EVIDENCE.md).

## Outcome by item

| item | block | what shipped |
|---|---|---|
| **A** — `--fit off` never armed the slab | 06 | `ensure_env()` (function-local-static-guarded `parse_env()`) runs at the top of `moe_cache_preflight()` and from `moe_cache_init()`.  Env parsing no longer depends on `ggml_backend_cuda_init()` ordering, so the slab is armed whenever the cache is enabled, independent of `--fit`.  `g_init_done` keeps its device-bookkeeping meaning. |
| **B** — not-armed diagnostic | 06 | `ggml_cuda_slab_configured()` (env-only) backs a one-time WARN in `alloc_all_locked()` when the cache is enabled, tables exist and the slab is not armed. |
| **C** — CE scratch hard abort | 12 | The three per-rank CE staging buffers grow through `ggml_cuda_device_malloc` (the fail-soft choke point); a partial attempt retires to `ce_old` and the function returns `false`, so the dispatcher falls back to the internal pipeline.  Sticky after the first failed grow.  Optional budget `GGML_CUDA_AR_CE_SCRATCH_MIB` (default 0 = unlimited).  CE stays opt-in. |
| **D** — slab-backed / device-blind yield | 06 | `ggml_cuda_device_malloc()` guards the yield on `ggml_cuda_slab_enabled()` (the predicate that chose the arena backing), not the failing device's `ggml_cuda_slab_active()`; `moe_cache_shrink_step(int device)` is device-scoped and no longer disables the whole cache when the failing device has nothing to free. |
| **E** — #135 verify + close | 12 | No code change (the RDNA4 guard already landed in r46).  Re-verified by compiling `allreduce-hip.cu` for gfx1100/gfx1150/gfx1151/gfx1200/gfx1201 (all clean); the maintainer also built r46 on gfx1100 (`ssh fingon`) without error.  Closed with a reply.  The optional RDNA3 `s_waitcnt_vscnt` fast path was **not** implemented (no gfx11 hardware to validate it). |
| **F** (added during review) — armed-but-uncreated slab | 06 | A slab that is armed but cannot be created (no VMM, bad chunk geometry, `cudaMemGetInfo` failure, or too little free VRAM for the reserve) records a decline reason, logs one ERROR naming the cause and the workarounds, and disables the cache to stream from the host; if detected after `moe_cache_is_sized()` it `GGML_ABORT`s instead.  A raw arena is never built silently. |

## Validation (gfx1201, fixed build `6daf81569` vs the r46 base `a6ca8905`)

- **4-way flag matrix** (`none` / `--fit off` / `--load-mode none` / both): `SLAB_INIT dev=0` and
  `dev=1` in **all four rows**.  Pre-fix, `--fit off` logged no `SLAB_INIT` while the cache auto-enabled.
- **Reservation** with an MTP draft: `aux=4.00 GiB`, `reserve=8.00 GiB`.
- **Coherence**: same-seed `llama-cli` (`-sm tensor`) generated text identical to the r46 base.
- **MTP rule 0** (four-axis, `-sm tensor`, 27B Q8_0, `-n 3000`): acceptance and generated text
  byte-identical to the base on all four axes.  Host-expert MoE + slab row (`-sm tensor -ncmoe 40`,
  35B-A3B Q4_K_M): acceptance identical (0.64122), output identical.
- **Prefill-logit gate**: mean KLD **0.000707**, same-top-p **98.755 %** (PASS).
- **CE**: output byte-identical to `hybrid`/`internal`; a forced fallback
  (`GGML_CUDA_AR_CE_SCRATCH_MIB=1`) logs one WARN and degrades to `internal` coherently.
- **Issue #136 field repro**: arena resident after decode, then a **68 216-token** prefill
  (`-sm tensor -ncmoe 48 --fit off -ub 6144 -c 204800`, q8_0 KV, MTP n3, `GGML_CUDA_ALLREDUCE=ce`)
  completes with `SLAB_INIT` on both devices, 530.9 t/s prompt, and 0 OOM / 0 aborts / 0 stand-downs / 0
  `////`.
- **Forced slab decline** (`GGML_CUDA_SLAB_RESERVE_MIB=40000`): ERROR names the cause, cache streams,
  run completes.  **`GGML_CUDA_SLAB=0`**: plain path unchanged, no spurious warning.

## Decisions recorded for the maintainer

- Item B is a log-only backstop; it is not reachable without a bug post-fix (and does not fire on healthy
  runs, on `GGML_CUDA_SLAB=0`, or when item F disables the cache first).  A single WARN at the
  cache-enable path was chosen over the campaign's complementary preflight warning (which would be noise
  for an intentional `MOE_EXPERT_CACHE_MIB=0`).
- `--fit` no longer participates in the slab decision at all: the slab is armed and auto-sized, with
  `GGML_CUDA_SLAB=0` as the only up-front opt-out and a creation failure degrading to streaming.
- The libllama-without-`common/` preflight path was left as is (a custom embedder that never calls
  `llama_model_moe_cache_preflight` still gets no slab); item B would surface it.
