# `moe-upload-overlap` — rear-fill the host-resident experts (was: make the H2D upload overlap compute)

**Status: DECODE BLOCKER FIXED in scratch (2026-10-14) — see [`FIX-2026-10-14.md`](FIX-2026-10-14.md).**
The blocker was **not** the slab: it is a premature arena-sizing latch (`alloc_all_locked` sized on the
6 tables a prefill's 1-token tail ubatch primed), plus a second, independent `llama-bench`-only gap
(`llama-bench` never calls the MoE-cache preflight, so it never arms the slab for *any* placement).
With the universal preflight (in `llama_model_load_from_file`) + the robust deferred sizing
(`tools/block09-cache-sizing-and-preflight.patch`), both placements reach full residency
(60 tables / 4080 MiB/device) and **rear >= front on prefill and decode**.  The fix is cut for **block
09** (maintainer's decision).  Nothing is in `patches/` and nothing has been applied to the fork branch.

* **Prefill wins**: `pp4096` `0–19` = 4136 vs `20–39` = **4309 (+4 %)** at `-ncmoe 20`.
* **Decode blocker FIXED** in scratch: `LLAMA_NCMOE_FROM_END=20` now reaches **60 tables /
  4080 MiB per device** and `tg500` is 87.8 vs the front's 83.9 (`pp512` 849 vs 654, universal
  preflight).  Root cause: the arena sizing latched on the one layer a prefill's non-staged 1-token tail
  primed (the bulk prefill is staged and does not register tables); `llama-bench`'s two prefill passes
  made it fire before the decode registered the rest.  Full write-up + validation in
  [`FIX-2026-10-14.md`](FIX-2026-10-14.md).
* **MTP re-verified**: 35B embedded head `none == draft-mtp n3` bit-identical (+83 %); GSQ-RCO
  Qwen3.8-Flash-Next + separate MTP head coherent and bit-identical front vs tail.
* **`-sm layer` tail prefill does NOT regress** (clean `pp4096`: tail 2799 vs front 2429).
* **NEW (not this change):** `-lm none` + `-sm tensor` + `-ncmoe` aborts during model load at
  `ggml-backend-meta.cpp:1783` on all three models; the delivery's `gate-qwen4exp-quant-coherence.sh`
  hardcodes `-lm none`.  Flagged for the maintainer (`FIX` §4h/§7).

Earlier corrected results are in [`FINDINGS-2026-10-14.md`](FINDINGS-2026-10-14.md): staging is +41 % over
`GGML_SCHED_STAGE=0`; the campaign's "placement is a no-op" was an inert-knob artifact (`llama-bench`
ignores the `common.h` knob); the first "BACK +5.7 %" reading was byte-confounded; and `blk.39` is a
full-attention trunk layer, **not** dense.
Not part of the delivery; nothing here may be applied to the fork without the maintainer's go-ahead.

## 0. Why this exists (and what it replaces)

`wip/moe-layer-stagger/` (now parked) asked whether *striding* the host/device layer split helps.  It
does not — but chasing it produced the real finding:

> **The host-expert upload is not overlapped with compute at all.**  Staggering (or any prefetch) can
> only help once the ring transfer and the compute are travelling concurrently; today the transfer is
> simply added to the compute time, so there is nothing to overlap and no schedule to improve.

The maintainer's framing, and this campaign's thesis: *the H2D staging ring cannot improve anything
unless the compute + ring transfer occur concurrently.*

## 1. Measured (gfx1201, 2× R9700, ROCm 7.14.1, r62 `e909c529a`, cache off, heatmap off)

Model `Qwen3.6-35B-A3B-Q8_0`: 41 layers (40 trunk + nextn), 256 experts, **816 MiB routed experts per
layer**.  `-sm tensor`, `-p 4096 -b 4096 -ub 4096 -r 3 -fa 1 -t 8` (one ubatch ⇒ one upload/host layer):

| config | pp4096 t/s | time |
|---|---:|---:|
| `-ncmoe 0` (full residency) | 8013 | 0.511 s |
| `-ncmoe 4`, host `{0,1,2,3}` | 6587 | 0.622 s |
| `-ncmoe 4`, host `{18,19,20,21}` | 6583 | 0.622 s |
| `-ncmoe 4`, host `{36,37,38,39}` | 6585 | 0.622 s |

`-sm layer` is the same (full-res 6716; `-ncmoe 4` `{0..3}` 5141 vs `{36..39}` 5146).

* **Placement is a no-op in both split modes.**  Moving 4 host layers from the front to the back
  changes nothing, so the upload is not being hidden behind the 36 device layers' compute.
* **The added cost is exactly the raw transfer.**  +0.111 s for 4 layers = 4 × 816 MiB = 3.4 GB,
  ~1.7 GB/card at ~15 GB/s ≈ 0.11 s.  Per layer that is a flat **23–27 ms**, i.e. `bytes / link-rate`,
  with no threshold or curvature (`-ncmoe` 0/4/8/12/16/20 → 8013/6581/5742/5082/4560/4127 t/s).
* **It is additive, not overlapped:** `T(partial) ≈ T(full) + bytes/rate`.  Per-layer compute is
  `c ≈ 0.511/40 ≈ 12.8 ms`; the upload is `u ≈ 23–27 ms`.  Perfect overlap would put `-ncmoe 4` at
  `max(compute, upload) ≈ 0.51 s` (i.e. ≈ full residency).  It measures 0.62 s.
* **The `-ub` lever is the same effect in reverse:** at `-ub 2048` the tables are re-uploaded per
  ubatch, so the bill doubles (`-ncmoe 20 -ub 2048` 2494 vs `-ub 4096` 4122 t/s).  That is buying the
  loss back by moving fewer bytes, not by overlapping.

## 2. Instrumentation evidence (r62 tree; prints added to `ggml-backend.cpp` / `ggml-backend-meta.cpp`)

**The seed's claimed evidence (refuted below — kept for the record):**

1. In `ggml_backend_sched_copy_input`, every host expert tensor prints
   `usage=1 ishost=1 ishw=1` — `blk.0.ffn_{gate,up,down}_exps.weight` … — so
   `sched_stage_is_host_weight()` **is** true for them.
2. **But** in `sched_stage_issue`, `n_host_inputs == 0` for **every** split: 242 `DBGSS` scans
   (121 splits), zero with a host-weight input, in **both** `-sm tensor` and `-sm layer`.  And the meta
   `stage_input` (the `-sm tensor` staging entry point) is **never called**.
3. `meta_graph_compute` is called once per split (first split `n_nodes=3532`, then 99/99/73/28…).

So the staged hand-off **does** engage, and the correlation is settled (full write-up in
`FINDINGS-2026-10-14.md`):

* `sched_stage_issue` sees all 12 host tables (`nhw=1`, `n_host_inputs=1`, `bt=4096 >= mn=1542`) and
  returns only at `reason=no_stage_buffer` — **the meta backend has no `stage_buffer` by design** and
  stages per device through `stage_input`.  `stage_split_ok` was set before that return, so the
  hand-off still runs.  All three seed candidates are refuted: the gate reads `bt=4096` correctly, the
  split identity is right, and slots 8 vs 16 changes nothing.
* `copy_input` then takes `branch=stage_input took=1` for every table; `META_STAGE_INPUT … ret=1
  chunks=2`; `META_GRAPH_COMPUTE … pending=1`.
* The earlier "`n_host_inputs == 0` / `stage_input` never called" reading came from the bookkeeping
  running *after* the disabled/width-gate returns (and/or a stale binary), not from the code path.

The transfer is nonetheless exposed: the uploads are async pinned 2-D H2D on the per-device copy stream
(host issue 1–16 µs), but they are issued only at their own split and the host is drained by the
graph-input host synchronizes the fragmented host-layer splits introduce.  `T(partial) = T(full) +
bytes/rate`, forcing the copies onto the main stream costs only ~5 %, and FRONT/MIDDLE/BACK is a no-op —
so **the missing piece is a lookahead, not the ring, the gate or the slots** (see `FINDINGS` §4–§5).

## 3. Fix directions (the seed's list is moot; the real lever is a lookahead)

The seed's directions all assume the hand-off is not engaging, which is false (see §2 and
`FINDINGS-2026-10-14.md`).  The ring, the width gate and the slot count are **not** the blocker; the
uploads already ride the event-ordered meta `stage_input` / per-device copy-stream path, and the r62
main-stream fix concerns the below-gate splice, not this path.  The real blocker is that an upload is
issued at the top of its own split and there is **no lookahead**, while the host is repeatedly drained by
graph-input host synchronizes (`inp_pos` 98 ms, `out_ids` 42 ms).  What would make the overlap real:

* **A real lookahead.**  Issue a host layer's tables several splits ahead (or all at the top of the
  pass), so the per-device copy stream has in-flight device compute to run behind.  New block-09 code.
* **Remove/avoid the graph-input host drains** so the host can run ahead to issue the lookahead.
* Placement alone cannot help (measured identical) and neither can ring depth (measured identical).

Whatever the fix, keep the r62 corruption fixed (below-gate `-sm tensor -ncmoe` must stay coherent) and
**report to the maintainer before changing delivery code**.  Owning blocks: 09 (host-resident MoE /
scheduler / Meta), 10 (`ggml-cuda.cu`).

**Update 2026-10-14 — prototyped.**  `tools/prefetch-prototype.patch` stages every host table at the top
of the pass; it is bit-identical but **~4 % slower** (BACK: 6654–6695 vs 6927–6958).  The prefetch does
remove the transfer from the compute (GPU kernel time 3101 vs 3339 ms) yet the pass is longer: ~88 ms of
scheduler idle appears.  Concurrent H2D is nearly free in isolation (`tools/contendprobe.cpp`), so this
is **not** a hardware limit — it is a design/idle problem to fix.  Separately, the corrected placement
A/B puts **BACK at +5.7 % over FRONT** (6944 vs 6570).  See `FINDINGS` §7 and `logs/ab-prefetch.txt`.

## 4. Repro / instrumentation recipe

```bash
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
B=~/llama.cpp/build-rocm-hybrid/bin/llama-bench
C="MOE_EXPERT_HEATMAP=0 MOE_EXPERT_CACHE_MIB=0 HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib"
env $C $B -m $M -ngl 99 -sm tensor -ncmoe 4 -p 4096 -n 0 -b 4096 -ub 4096 -r 3 -t 8 -fa 1 -o md
# add LLAMA_NCMOE_LAYERS=<list> (knob: wip/moe-layer-stagger/tools/ncmoe-layer-select.patch) to place the host layers.
```

Instrumentation (temporary, revert after): print `input->name` + `usage`/`ishost`/`sched_stage_is_host_weight`
at the top of `ggml_backend_sched_copy_input` (`ggml-backend.cpp:~2597`); print the split index,
`n_inputs`, `sched->stage_enabled`, and `n_host_inputs`/`stage_n_slots` at each `return` in
`sched_stage_issue` (`:2432`); a print at the top of `ggml_backend_meta_stage_input`
(`ggml-backend-meta.cpp:2521`); and one at `ggml_backend_meta_graph_compute` (`:2745`).

## 5. Ceiling and the hard floor

Overlap can at best make `T = max(total_compute, total_upload)`; it can never beat the link.  On this
box the exposed upload at `-ncmoe 20 -ub 4096` is ~0.46 s of which ~`n_host × min(u,c)` is recoverable:
`20 × 12.8 ms ≈ 0.26 s` → **up to ~+20 %** (`-ub 4096`) and relatively more at `-ub 2048` (upload
doubles).  So this is a real but bounded win — and it must not trade away the r62 correctness.

## 6. Gates (a promotion must clear all)

Same-seed coherence (the affected models from r62: 35B-A3B, gemma4 26B-A4B, Flash-Next GSQ);
`scripts/gate-prefill-logits.sh` (**KLD ≤ 0.005 / top-p ≥ 98 %** — the upload is data movement, the
numerics must not change); adaptive-MTP `none == n3` on `-sm tensor -ncmoe 99` + auto cache;
`llama-batched-bench -npl 1,4,8` decode not regressed; the r62 below-gate corruption repro stays clean;
`scripts/validate-set.sh` green if promoted.

## 7. Pointers

* Transfer path: `ggml/src/ggml-backend.cpp` `sched_stage_issue` (`:2432`), `ggml_backend_sched_copy_input`
  (`:2597`), `sched_stage_min_tokens_for`/`sched_stage_batch_tokens` (`:2251`/`:2289`);
  `ggml/src/ggml-backend-meta.cpp` `ggml_backend_meta_stage_input` (`:2521`),
  `ggml_backend_meta_graph_compute` (`:2745`);
  `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu` `stage_upload`/`stage_gather`/`stage_wait`/`stage_d2d`;
  `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_backend_cuda_set_tensor_2d_async` (`:3720`, the r62 fix).
* r62 record: `archive/work/meta-split-copy-splice-race/`, `WORKLOG.md` 261014.007 (block 10).
* Parked/refuted: `wip/moe-layer-stagger/FINDINGS-2026-10-14.md`.
* Delivery block owners: block 09 (host-resident MoE / scheduler / Meta), block 10 (`ggml-cuda.cu`).
