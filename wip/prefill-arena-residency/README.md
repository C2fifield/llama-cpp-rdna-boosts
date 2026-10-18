# `prefill-arena-residency` — serve prefill from the expert-cache arena, upload only the misses

**Status: OPEN / ACTIVE (seeded 2026-10-18).  Maintainer decision received 2026-10-18: the old
mutating device-remap implementation is NOT to return; a greatly reduced, focused, immutable-after-
snapshot prefill-only residency map IS approved.**  Not part of the delivery; nothing here may be applied
to the fork or to `patches/` without the maintainer's go-ahead (WIP and promotion rules in `AGENTS.md`).

> **Decision (2026-10-18).**
> * **Not revived:** `DEVMAP` / `DEVPOLICY` / `KSLOT` / `DEV_EAGER`, the mutating `slot_dev` device
>   policy and its deferred promotion — the machinery behind the multi-session wrong-expert outputs.
> * **Approved:** a **prefill-only** residency map, **immutable after a snapshot**, rebuilt from the
>   authoritative host `expert_slot`, never mutated by decode, and not the decode remap.  Call it the
>   **prefill residency snapshot (PRS)**, not a device map.
> * **Concurrency ruling:** if a decode session is active, snapshot/freeze the **top ~75 % (hottest)** of
>   the arena for prefill and leave the bottom ~25 % to decode admission; with no decode active, the
>   prefill may snapshot the whole arena.

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

> **Loader confound (2026-10-18, found while prototyping).**  The table above used the **default
> loader** (no `--load-mode none --lazy-mode off`).  Flash-Next has a 27.8 GiB host-gathered **PLE**
> (per-layer token embedding); on the default lazy/mmap path it faults during prefill and dominates the
> timing.  With the field loader flags the same 15.8k-token prompt runs at **580 t/s**, not 329 — and
> the R2 A/B is **+45 %**, not +11 % (§4.1).  The 329/311 numbers are still a valid "the cache-aware
> staging is inert on the default loader" control, but they are **not** the field config.  Any prefill
> number in this campaign must name `--load-mode none --lazy-mode off`.

## 0.2 Devmap vs the prefill reader — **not the same feature**

The r40 sweep removed two things that shared one component, and it is worth keeping them apart:

* **DEVMAP (decode)** — the device-built slot remap plus deferred post-graph promotion.  This *is* the
  one that produced the incoherent/wrong-expert outputs: the device `slot_dev` map was built from the
  **pre-promotion** state, a later fill refilled that slot, and the remap pointed at a slot that by then
  held a different expert — `FINDINGS-single-seq-rollback.md` §5f records the validator line
  `SLOT-DEV-MISMATCH layer=0 expert=2 dev=19 host=-1` and the mechanism ("a device-built remap from that
  map points at slot 19, which by then holds a *different* expert, so the routed read returns the wrong
  expert").  Setting `MOE_EXPERT_CACHE_DEVMAP=0` (the eager host-routing remap) made **every** divergent
  pair pure.  A generation guard (`res_gen == dev_gen`) then removed the map-mismatch class, a residual
  divergence remained, and the final attribution was a layout-sensitive **hipBLASLt** solution flip, not
  the cache (`FINDINGS-devmap-removal.md` §2).  That multi-session hunt (and the ~14 % decode it bought
  versus the eager path) is what drove the removal.
* **The prefill reader (PR #121)** — cache-aware staging and in-place MMQ reads.  It *consumed* the same
  device slot map as a **read-only** expert source (guarded by `dev_gen == res_gen`), but it did not build
  remaps, promote, or mutate residency.  Its own gates were clean: greedy output identical to r33,
  level-3 slot-content mismatches **0**, `MUL_MAT_ID` 932/932.  It was deleted because the r40 "remove
  DEVMAP for good" sweep also removed the shared `slot_dev` / generation plumbing, not because it was
  incoherent.

So: **devmap was the decode remap policy that caused the wrong-expert outputs; the prefill reader was a
separate, correct feature that happened to share its map.**  The reason this campaign still needs a
decision is that reviving the prefill reader as-is reintroduces that shared map.  The stale-map failure
mode needs a decode-side mutation between map-build and read; the prefill reader has no such mutation
(the arena is frozen during prefill, and the map would be rebuilt from the authoritative host
`expert_slot` on `g_write_seq`), but the r40 note's blanket wording still applies until the maintainer
scopes it.

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

## 2.5 Concurrency: one session prefilling, one decoding — what actually happens

Verified in this tree (`tools/server/server-context.cpp` `update_slots`): there is **one `llama_decode`
at a time per context and one MoE cache per context**.  A batch can contain **both** generating slots'
tokens (`iterate(generating, ... handle_last_sampled_token)`, including MTP draft/verify tokens) and
prompt tokens (`while ... batch.size() < n_batch`), so "two sessions" means **interleaved mixed
ubatches**, not two graphs running concurrently.

Consequences:

* **A mixed ubatch is above the decode band** whenever the prefill chunk is non-trivial (2048 prompt
  tokens + 1-15 decode tokens).  The whole `MUL_MAT_ID` op then goes through the **prefill path**,
  including the decode tokens: `moe_cache_update_host` returns before any admission (it only tallies),
  `moe_cache_take_over` is gated on `n_tokens <= band`, and the scheduler's take-over/gather gates are
  band-gated.  **The arena is not mutated for that op.**
* **Arena mutation comes from a decode-only (or small mixed, `n_tokens <= 16`) ubatch**, where the
  existing eager decode path runs.  That ubatch is not a prefill ubatch, so there is no simultaneous
  read/write inside a graph.  Within a graph the arena is read-only for the prefill path as well (the
  eager fills are enqueued before the graph; a slab stand-down happens on the host side).
* **So the two-session hazard is not a data race — it is erosion.**  A decode-only ubatch between two
  prefill ubatches can evict/refill slots the prefill snapshot was using, so the next prefill ubatch
  reads more cold and decode re-fills later: churn, not garbage.  The snapshot must be rebuilt whenever
  the cache writes (`g_write_seq`) or a table is stood down/re-armed, and the **freeze partition is the
  performance fix**, not a correctness fix.
* **The freeze is a slot mask, not a policy rewrite.**  While a prefill is in flight, the cache marks the
  top ~75 % of each table's slots (by the existing heat/count rank) as **frozen**: decode admission may
  only evict from the bottom ~25 %, and the PRS marks only frozen experts as arena-resident (a
  resident-but-mutable expert in the 25 % is treated as **cold** by prefill, so a decode fill can never
  change what the snapshot points at).  With no decode active, the PRS snapshots all slots.  A slab
  stand-down of a table drops its snapshot (cold fallback for that table); a re-arm rebuilds it.
* **Arming "decode active":** the safe cheap signal is context-level, not per-op.  A first cut can arm
  the freeze when a prefill-band op is seen after a decode-band op (the process has decoded since), and
  release it at the prefill -> decode transition the context already detects for the compute-buffer
  drop.  Refine to an explicit context hint ("a generating slot exists") if the heuristic mis-arms on
  the first request.

## 3. Design options for the r82 revival

All of R1-R3 use the **PRS**: a small device array built per table from the authoritative host
`expert_slot`, filtered to the frozen set, uploaded once per snapshot, and **never mutated by decode**.
None of the removed `slot_dev`/`res_gen`/`dev_gen`/promotion machinery is revived.

| option | what | PRS consumer | size / risk | expected |
|---|---|---|---|---|
| **R1 — revive PR #121's in-place read** | cache-aware staging + in-place MMQ reads, re-cut to r82 | in-place MMQ (`mmq.cuh`) | 816-line-era change: `moe-expert-cache.cu/.h`, `mmq.cuh`, `ggml-cuda.cu`, scheduler, hooks | the proven ~2×; largest blast radius |
| **R2 — cache-aware staging only** | `stage_from_cache` fills the ring slot from arena + host (one kernel); consumer unchanged | one staging copy kernel | ~150 lines; the staging hook is already wired (stub) | H2D falls to `1 - residence`; keeps the staging overlap; no MMQ change |
| **R3 — arena-aware device gather** | extend `moe_cache_gather_kernel` to source resident routed experts from the arena | `moe_cache_gather_kernel` | ~100 lines, reuses `moe_cache_gather_host`, needs `GGML_SCHED_DEVGATHER=1` | same H2D cut; no overlap (compute stream) unless re-issued on the copy stream |
| **R4 — eager-host-remap in-place** | materialize the prefill remap on the host (resident → slot, cold → `n_res + e`) and teach MMQ the source selection | no device array (kept eager path + `remap_dev`) | MMQ change + ids readback + a full sync per MoE op | no PRS at all, but pays ~144 readback/syncs per ubatch; win unproven |
| **R5 — host-driven copy list** | the host issues per-run D2D (arena) / H2D (host) copies into the ring slot from `expert_slot` | none | simplest code, per-run copy storm (~10²-10³ copies/table) | overlap preserved; launch overhead may eat the win |

**Recommendation: R2 first.**  It captures the PCIe win (only the non-resident experts cross H2D)
with the consumer untouched, so the PRS and the freeze stay isolated in one copy kernel.  The resident
75 % is a D2D copy into the ring slot (~24 GiB/ubatch/device at VRAM speed, tens of ms), which should
be close to R1's in-place read for this purpose.  R1 is the follow-on if R2 validates and the remaining
ring-copy cost matters.

The r40 naming ban still applies: no `*DEVMAP*`/`*KSLOT*`/`*DEVPOLICY*` variable, no mutating device
policy, no `res_gen`/`dev_gen` generation dance.  The PRS is a one-way snapshot re-read from the host,
not a device-side authority.

## 4. Recommended next step

1. **Maintainer decision: received 2026-10-18** — no mutating device map; a trimmed, immutable PRS and
   the 75/25 freeze while decode is active are approved.
2. Implement **R2** first (cache-aware staging + PRS + freeze), reusing the r36 `moe_cache_stage_table` /
   `moe_cache_stage_kernel` shape from `git show 670765b` /
   `archive/work/moe-verify-fusions/wip-since-block15.diff`, but with the PRS instead of `slot_dev` and
   without any of the removed promotion/generation machinery.  Default-off
   (`GGML_MOE_CACHE_STAGE` / a new focused kill-switch).  R1 (in-place MMQ) is the follow-on if R2
   validates and the remaining ring-copy cost matters.
   **→ R2 prototype DONE and measured (§4.1): +31-45 % prefill, byte-identical greedy.  Freeze is next.**
3. Measure on IQ4_NL 2-GPU `-sm tensor -ncmoe 48` at 32k and the field 131k config; sweep
   `MOE_EXPERT_CACHE_MIB` to get the prefill-vs-residence curve (this also answers whether prefill is
   upload-bound on the current tree).
4. Gates: same-seed greedy coherence (dense + host-expert MoE), `gate-qwen4exp-quant-coherence.sh`,
   prefill-logit KLD (0 / 100 % expected — the bytes are identical), adaptive-MTP 4 axes + phase probe.
5. Promote only with the same-change kill switch, re-cut block(s) and the full gate matrix; update
   `ENVIRONMENT.md` (the stale `GGML_MOE_CACHE_INPLACE`/`STAGE` rows must match reality either way).

## 4.1 R2 prototype — measured (2026-10-18)

**Status: implemented, env-gated OFF, measured, byte-identical output.  Not promoted.**  Diff:
[`r2-prototype.patch`](r2-prototype.patch) (224 insertions, 3 files; sha256 `8a209261…`).  The
`~/llama.cpp` working tree currently has it applied (uncommitted; **not** in `patches/`).

What it does: the `stage_from_cache` stub is live again, but with the **PRS** instead of the removed
`slot_dev`: a whole-table staging copy is filled by one kernel from the arena (resident experts,
D2D) + the pinned host alias (the rest, H2D), into the ordinary ring slot.  The consumer is unchanged.
No deferred in-place path, no device policy, no `res_gen`/`dev_gen`.  Env: `GGML_MOE_CACHE_STAGE=1`
(default 0 for the prototype).  `moe_cache_report` now prints the staged arena/host split and the
decline counts.

**Config:** `Qwen3.8-Flash-Next IQ4_NL`, 2× R9700, `-sm tensor -ncmoe 48 -fa 1 -ctk/-ctv f16
-t 8 --load-mode none --lazy-mode off`, MTP off (`--spec-type none` equivalent), greedy `-n 64`
seed 42 temp 0.  `stage=0` = whole-table staging; `stage=1` = R2.

| prompt | `-ub` | stage 0 | stage 1 (R2) | gain | greedy sha (both) |
|---|---:|---:|---:|---:|---|
| 15 800 tok (`/tmp/prefill16k.txt`, 48 268 B) | 2048 | 581.8 t/s | **838.8 t/s** | **+44.2 %** | `7171139467ce` |
| 15 800 tok | 4096 | 1 091.8 t/s | **1 433.1 t/s** | **+31.3 %** | `a60a39e11133` |
| 31 555 tok (`/tmp/prefill32k.txt`, 96 558 B) | 4096 | 1 183.6 t/s | **1 561.3 t/s** | **+31.9 %** | `98800fdba634` |
| 15 800 tok, `-v` control | 2048 | 579.9 t/s | 839.7 t/s | +44.8 % | — |

* **Coherence:** stage 0 and stage 1 generate **byte-identical** greedy text at every width/depth
  above (the `extract-generated.py` hashes match), and the text is coherent (the model reasons about
  the pasted README).  The prefill path is a byte-identical copy, as intended.
* **Staged split (15.8k/ub2048 run):** `ok=1666`, `geom_decline=0`, `unusable=308` (tables the slab
  stood down mid-prefill — they correctly fall back to plain staging), `nomatch=0`; **arena 272.6 GiB
  vs host 101.6 GiB = 27.1 % H2D**, i.e. the 72.9 % residence fraction is realised exactly.
* **Why the gain is +31-45 % and not 3.7×:** the baseline's H2D is partly overlapped with compute, so
  removing 73 % of it does not remove 73 % of the wall time.  `-ub 4096` amortises the per-ubatch
  overhead better (1 092 vs 582 t/s baseline) and therefore has a smaller *relative* gain, but a larger
  absolute one.  The remaining serial cost is the ring→`input_cpy` D2D (redirect mode did not change
  it materially) and the compute itself; the removed PR #121 **in-place read** (R1) is what removes
  the ring copy and gets closer to the compute floor.
* **Residency at depth:** 32k ctx → arena **47 207.8 MiB = 72.9 %**; 64k ctx → **46 195.3 MiB at sizing
  / 45 712.8 MiB after the prefill = ~70.5 %**, decode hit `h=0.7698`.  The residence fraction barely
  moves between 32k and 64k on this config; the field's 50-60 % is its 131k/MTP context.

**What is not in the prototype:** the 75/25 freeze for a concurrent decode session (§2.5) — it
snapshots the whole arena, which is correct for the single-session measurement.  The next step is the
freeze (top 75 % frozen, bottom 25 % decode-mutable, whole arena when no decode is active), then the
R1 in-place read if the ring copy is still material.

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

1. **Freeze details:** is the 75/25 split a fixed fraction, or should it be `min(75 %, slots - band)` so
   a very small arena always leaves at least one decode token's experts evictable?  (Proposal: the
   latter; the frozen count is `max(0, min(3*slots/4, slots - n_expert_used))`.)
2. **Freeze ordering:** rank frozen slots by heat/count (the existing, r80-fixed heat map), or freeze a
   contiguous slot range?  Heat rank matches the decode-hot set and is the point of the freeze; a
   contiguous range would make the copy kernel's resident run longer but throw away decode hotness.
3. **Arming:** accept the "decoded since the last prefill" heuristic, or add an explicit context hint
   ("a generating slot exists") from the server/context?  The latter is more correct on the first
   request but touches the context/server boundary.
4. Implement **R2** or go straight to **R1**?  R2 is recommended (consumer unchanged, ~150 lines); R1 is
   the proven ~2× but the larger change.
5. Is `--host-experts pool` in scope for the cold half, or is declining for pooled tables acceptable?
6. The stale `ENVIRONMENT.md` rows for `GGML_MOE_CACHE_INPLACE`/`STAGE`: fix now (documentation) or
   with the revival?
7. Should this be a **new block** or an amendment to block 09 when promoted?  (WIP only until then.)

## 8. Record

* **2026-10-18** — campaign seeded.  Baseline measured on the target IQ4_NL 2-GPU config (§0.1).
  Prior art located and read: r33 `670765b` / PR #121, r36 fold `a89e1d0`, r40 removal `f7c427e`;
  the r36 code survives only in `archive/work/moe-verify-fusions/`.  Current tree's
  `stage_from_cache` is a stub; `ENVIRONMENT.md` still advertises the removed knobs.
* **2026-10-18 (later)** — maintainer decision: no mutating device map; a **trimmed, immutable PRS** is
  approved, with the **top ~75 % frozen / bottom ~25 % decode-mutable** split when a decode session is
  active.  Concurrency verified in `update_slots`: prefill+decode are interleaved mixed ubatches, not
  concurrent graphs; a mixed ubatch is above the decode band and does not mutate the arena, so the
  freeze is a churn/performance control, not a data-race fix.  R2 (cache-aware staging + PRS) is the
  first implementation, R1 (in-place MMQ) the follow-on.
* **2026-10-18 (R2 prototype measured)** — `r2-prototype.patch` applied to the r82 tree, `GGML_MOE_CACHE_STAGE=1`.
  Field loader flags (`--load-mode none --lazy-mode off`) matter: the default-loader 329 t/s / +11 %
  result was a PLE-lazy-fault confound.  Correct A/B: 15.8k **581.8 → 838.8** (`-ub 2048`, +44.2 %),
  **1 091.8 → 1 433.1** (`-ub 4096`, +31.3 %), 31.6k **1 183.6 → 1 561.3** (`-ub 4096`, +31.9 %);
  H2D staged **27.1 %** (arena 272.6 GiB vs host 101.6 GiB), `geom_decline=0`; greedy sha identical in
  every pair.  Freeze and R1 are the remaining steps.
