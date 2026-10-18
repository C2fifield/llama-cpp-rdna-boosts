# `prefill-arena-residency` — serve prefill from the expert-cache arena, upload only the misses

**Status: OPEN / ACTIVE (seeded 2026-10-18), scoping + revival decision needed.**  Not part of the
delivery; nothing here may be applied to the fork or to `patches/` without the maintainer's go-ahead (WIP
and promotion rules in `AGENTS.md`).

## 0. Headline: this was already built, measured, shipped — and then removed as collateral

The maintainer's idea is **not new to this repo**.  It was implemented by @briansp2020 as PR #121
(`wip/moe-cache-inplace`, r33 commit `670765b`, folded into the 16-block delivery at **r36**
`a89e1d0`), measured with large wins, and then **removed at r40** (`f7c427e`) when the device-remap
(DEVMAP) decode path was removed "for good" — the cache-aware staging and in-place prefill reads were
part of the removed machinery.

**r33 measured result** (GSQ-RCO IQ3_XXS + shared Q8_0 MTP, 2× R9700, `--n-cpu-moe 48`, 256K ctx, q8_0
KV, `-ub 2048`, MTP n-max 3, pinned host experts; full record below in §1):

| config | r33 (whole-table staging) | r33 + PR #121 |
|---|---:|---:|
| `-sm tensor`, 2.9k-token prefill | 794 t/s | **1 754 t/s** (+121 %) |
| `-sm tensor`, 61k-token prefill | 1 384 t/s | **2 422 t/s** (+75 %) |
| `-sm layer -ts 59,41`, 2.9k prefill | 745 t/s | **1 743 t/s** (+134 %) |
| `-sm layer -ts 59,41`, 61k prefill | 1 214 t/s | **1 931 t/s** (+59 %) |
| greedy decode | 110.4-110.9 t/s | 110.8-113.1 t/s |
| output | — | greedy shas **identical** to r33; `MUL_MAT_ID` 932/932; 255k needle PASS |

The r36 fold's own WORKLOG numbers (a different run) were the same shape: `-sm tensor` 2.9k
**431.8 → 1 214.1** t/s, 61k **1 005.1 → 2 471.6**; `-sm layer -ts 59,41` 2.9k **491.4 → 1 460.8**,
61k **682.3 → 1 998.3**.

So the answer to "can prefill load only the experts that are not already VRAM resident?" is **yes, and
it roughly halved-to-thirded prefill time on the previous campaign's config**.  The work now is a
**revival + re-validation on the r82 tree**, plus a decision on how to do it without reviving the
device-remap path the r40 note forbids (§4).

## 0.1 Current state (r82, measured 2026-10-18)

The r82 tree has only the **whole-table staging** path.  The r36 hooks survive as a **stub**:

```c
// ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu
// MoE expert cache: whole-table staging from the cache arena (removed with the device-remap path; this
// now always declines so the caller falls back to the plain staging upload).
static bool ggml_backend_cuda_stage_from_cache(...) { return false; }
```

`moe_cache_stage_table`, `moe_cache_inplace_*`, `moe_cache_stage_kernel`, the MMQ in-place prefill read
and the device slot map are all **gone** from the tree (`git log --all` has the r33/r36 code; the
pre-removal working tree is `archive/work/moe-verify-fusions/wip-since-block15.diff`).  `ENVIRONMENT.md`
**still documents** `GGML_MOE_CACHE_INPLACE` / `GGML_MOE_CACHE_STAGE` as live r36 kill-switches (§2,
"staging" rows) — that is stale and should be corrected whichever way this campaign goes.

Baseline on the maintainer's target model (the campaign's premise measurement):

```sh
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_SLAB_STATS=1 LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1 \
GGML_CUDA_MMB_DOWN16=1 [GGML_SCHED_DEVGATHER=1] \
./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  -ngl 99 -sm tensor -ncmoe 48 -fa 1 -ctk f16 -ctv f16 -c 32768 -b 2048 -ub 2048 -t 8 -st -v \
  -p "$(cat prompts/prose-rdna-boosts.txt)" -n 1 --seed 42 --temp 0 --no-display-prompt
```

| quantity | value |
|---|---|
| host-resident expert bytes (both devices) | **64 800 MiB** (48 layers × 3 tables × 512 experts × ~0.88 MiB) |
| arena at sizing | **47 207.8 MiB = 72.9 % residency** (373/512 slots/table, 288 tables, 106 050 slots) |
| arena after the prefill's slab evictions + re-arm | 46 604.0 MiB (~72 %) |
| H2D bandwidth calibration | **14.5 GB/s** per device |
| staging width gate | min_tokens = **1542** (`-ub 2048` stages; the 1202-token tail does not) |
| prefill, default whole-table staging | **329 t/s** (16.12 s / 5 298 tokens) |
| prefill, `GGML_SCHED_DEVGATHER=1` (no arena use) | **311 t/s** (17.02 s / 5 298 tokens) |

Per staged ubatch the whole host expert set crosses PCIe: 48 × 3 × ~225 MiB/device ≈ **32.4 GiB per
device per ubatch**.  Two staged ubatches + one non-staged tail ≈ ~96 GiB/device ≈ 7 s at 14.5 GB/s;
the measured prefill is 16 s, i.e. the upload is a large share and the staging overlap hides only part
of it.  The r33 PR #121 measurement (2.9k/61k) is the empirical proof that removing that upload is worth
roughly 2×.

## 1. What PR #121 actually did (the revival reference)

Source: `git show 670765b` (`wip/moe-cache-inplace/README.md` + an 816-line format-patch); folded at
`a89e1d0`; removed at `f7c427e`.  Three layers, all gated by `GGML_MOE_CACHE_STAGE` /
`GGML_MOE_CACHE_INPLACE`:

1. **Cache-aware staging** (`moe_cache_stage_table` + `moe_cache_stage_kernel`): when a whole-table
   staging copy is exactly a registered table's slice, fill it with one kernel — resident experts from
   the **arena**, the rest from the **pinned host alias** — instead of a plain full-table H2D.  Same
   bytes in the destination, so every consumer is unchanged.  The copy stream is ordered after compute
   only when `g_write_seq` moved (an arena/slot-map write), so during prefill it keeps overlapping.
2. **In-place reads** (the bigger win): the staging is *deferred*; `stage_d2d` hands the record to the
   tensor and copies nothing.  The routed-compact MMQ kernel reads the table in place — arena slot for
   a resident expert, pinned host alias for a cold one — and the *other* consumers (fused gate stream,
   non-compact MMQ) fill the tensor from the same two sources.  The geometry is read live at the
   consumer, so an eviction in between only turns a read cold.  A routed, non-resident **last** expert
   is copied into a small zero-padded device scratch for MMQ's one-K-tile over-read.
3. **Below the staging width gate** (17-63 tokens): a split still stages when every host weight in it
   is served in place (staging then copies nothing) via a new `moe_cache_inplace` iface, and
   `offload_op` sends a 17-31-token routed op to the GPU.

Gated on `cudaHostGetDevicePointer` succeeding (pinned master), on the slot map being current, and on
the geometry matching exactly; anything else falls back to the r33 path.  The cold half reads the full
pinned master, so `--host-experts pool` bypasses it (noted in the PR).

## 2. Why the arena alone cannot be the prefill read source

A compact slot remap needs **every routed expert of the ubatch to be simultaneously resident**.  At a
prefill width (even 64 tokens × top-8 over 512 experts) the distinct routed set is ≈ all 512 — above 373
slots — so the decode approach (remap ids into the arena) cannot serve one prefill op.  Options are the
PR #121 shape (per-expert source at the consumer, no simultaneous residency), micro-batching (kills MMQ
efficiency), or a full-table assembly from arena + host.  PR #121 chose the first and proved it.

## 3. Design options for the r82 revival

| option | what | device slot map? | size / risk | expected |
|---|---|---|---|---|
| **R1 — revive PR #121 as-is** | cache-aware staging + in-place MMQ reads, re-cut to r82 | **yes** (`slot_dev`) | 816-line patch: `moe-expert-cache.cu/.h`, `mmq.cuh`, `ggml-cuda.cu`, scheduler gate, hooks | the proven ~2×; largest blast radius |
| **R2 — cache-aware staging only** | `stage_from_cache` fills the ring slot from arena + host (one kernel); consumer unchanged | **yes** (read-only) | ~150 lines + the staging hook is already wired (stub) | H2D falls to `1 - residence`; keeps the staging overlap; no MMQ change |
| **R3 — arena-aware device gather** | extend `moe_cache_gather_kernel` to source resident routed experts from the arena (path A of the original scoping) | **yes** (read-only) | ~100 lines, reuses `moe_cache_gather_host`, `GGML_SCHED_DEVGATHER=1` | H2D falls to `1 - residence`; no overlap (compute stream) unless re-issued on the copy stream |
| **R4 — eager-host-remap in-place** | materialize the prefill remap on the host (resident → slot, cold → `n_res + e`) and teach MMQ the source selection | **no** (uses the kept eager path + `remap_dev`) | MMQ change + ids readback + a full sync per MoE op | honours the no-device-map note, but pays 144 readback/syncs per ubatch; win unproven |
| **R5 — host-driven copy list** | the host issues per-run D2D (arena) / H2D (host) copies into the ring slot from `expert_slot` | **no** | simplest code, per-run copy storm (~10²-10³ copies/table) | overlap preserved; launch overhead may eat the win |

The r40 note is explicit: **"Do not add any `*DEVMAP*`/`*KSLOT*`/`*DEVPOLICY*` variable or device
slot-map path back."**  R1-R3 all need a device-side map for the copy/read.  The r40 note's target was
the **decode** device-remap policy (a stale `slot_dev` generation feeding the decode remap), and the
r33 in-place reads were found *correct* — they were removed only as part of the machinery.  Still, this
is a maintainer call, not an implementation detail: a **prefill-only, read-only** map rebuilt from the
authoritative host `expert_slot` on every write generation (and never consulted by decode) is a
different object, but it needs an explicit "yes".

If the answer is "no device map at all", R4 (eager host remap) or R5 (host copy list) are the only
paths, and both are less proven.

## 4. Recommended next step

1. **Maintainer decision:** accept a prefill-only read-only device map (R1/R2/R3) or require R4/R5.
2. If R1/R2/R3: revive the r36 code from `git show 670765b` / `archive/work/moe-verify-fusions/wip-since-block15.diff`,
   re-cut to the r82 tree, default-off, and A/B on the target config.  R2 first (smallest, consumer
   unchanged, directly answers "only H2D the non-resident experts"); R1 next if R2 validates.
3. Measure on IQ4_NL 2-GPU `-sm tensor -ncmoe 48` at 32k and the field 131k config; sweep
   `MOE_EXPERT_CACHE_MIB` to get the prefill-vs-residence curve (this also answers whether prefill is
   upload-bound on the current tree).
4. Gates: same-seed greedy coherence (dense + host-expert MoE), `gate-qwen4exp-quant-coherence.sh`,
   prefill-logit KLD (0 / 100 % expected — the bytes are identical), adaptive-MTP 4 axes + phase probe.
5. Promote only with the same-change kill switch, re-cut block(s) and the full gate matrix; update
   `ENVIRONMENT.md` (the stale `GGML_MOE_CACHE_INPLACE`/`STAGE` rows must match reality either way).

## 5. Risks / invariants

* The arena is frozen during prefill (prefill never calls the admission hooks), but the movable-boundary
  slab can **stand a table down mid-prefill** (`moe_cache_evict_slab_range`; measured ~4.2 GiB in the
  baseline run).  Every use must re-check `arena`/`slots`/geometry and fall back to host.  A stale
  resident map reads the wrong expert — the exact failure mode the r40 note is guarding against, and the
  reason a revived map must be rebuilt on `g_write_seq` and never cached across an arena move.
* The gather/MMQ **head/tail guard** (MMQ reads one tile past an expert) must be preserved: the removed
  PR's tail scratch is the reference for the cold last expert.
* The pinned-master gate (`cudaHostGetDevicePointer`) is mandatory on a no-XNACK gfx1201; a pageable
  master must decline (issue #116).
* `--host-experts pool` is bypassed by the in-place cold half (the PR's own note); decide whether to
  decline for pooled tables or read the pool.
* Purity: the bytes must be identical to the host path, so prefill logits must be unchanged; the r33
  gate recorded KLD 0.000707 / 98.755 % (the delivery's base) and identical greedy shas.

## 6. Code sites (current tree)

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu` | the `stage_from_cache` stub; `stage_upload`/`stage_gather` call sites (already wired) |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `moe_cache_gather_host`/`_kernel` (R3); `table_t`; `g_write_seq`; `moe_cache_get_cold` (R4 reference); the absent `moe_cache_stage_table`/`moe_cache_stage_kernel` (R1/R2 reference in the archive) |
| `ggml/src/ggml-cuda/mmq.cuh`, `mmq.cu` | routed-compact MMQ consumer (R1/R4) |
| `ggml/src/ggml-backend.cpp` | `sched_input_gatherable`, `sched_stage_issue`, `ggml_backend_sched_copy_input` |
| `ggml/src/ggml-backend-meta.cpp` | `ggml_backend_meta_stage_input` (the `-sm tensor` staged path) and the per-device gather forwarder |
| `ggml/src/ggml-cuda/ggml-cuda-slab.cu` | boundary moves that stand tables down mid-prefill |
| `archive/work/moe-verify-fusions/wip-since-block15.diff` | the pre-removal r36 tree (the exact `moe_cache_stage_table`, `moe_cache_stage_kernel`, `moe_cache_inplace_*`, `stage_d2d` take) |
| `archive/work/moe-verify-fusions/remove-devmap.diff` | the r40 removal diff (what was deleted and from where) |
| `archive/work/moe-verify-fusions/FINDINGS-devmap-removal.md` | the removal decision and rationale |

## 7. Open questions for the maintainer

1. **The device-map gate:** is a prefill-only, read-only `expert -> slot` map (never consulted by
   decode, rebuilt on `g_write_seq`) acceptable, or does the r40 note forbid it outright?  Without it,
   only R4/R5 are available.
2. Revive the **proven** R1 (in-place MMQ reads; ~2× measured on r33) or start with the smaller R2
   (cache-aware staging; still leaves the consumer unchanged)?
3. Is `--host-experts pool` in scope for the cold half, or is declining for pooled tables acceptable?
4. The stale `ENVIRONMENT.md` rows for `GGML_MOE_CACHE_INPLACE`/`STAGE`: fix now (documentation) or
   with the revival?
5. Should this be a **new block** or an amendment to block 09 when promoted?  (WIP only until then.)

## 8. Record

* **2026-10-18** — campaign seeded.  Baseline measured on the target IQ4_NL 2-GPU config (§0.1).
  Prior art located and read: r33 `670765b` / PR #121, r36 fold `a89e1d0`, r40 removal `f7c427e`;
  the r36 code survives only in `archive/work/moe-verify-fusions/`.  Current tree's
  `stage_from_cache` is a stub; `ENVIRONMENT.md` still advertises the removed knobs.  Decision needed
  on the device-map gate before any code is revived.
