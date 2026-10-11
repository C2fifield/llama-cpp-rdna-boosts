# FINDINGS — the unified-KV cell-offset non-determinism is a *cell-set* (not start-offset) dependence; a fixed alignment is not the fix

**Date:** 2026-10-13 (follow-up to the campaign README; WIP, NOT for delivery).
**Status:** mechanism refined and model-dependence pinned. A fixed allocation alignment is **not**
sufficient. Root cause: the FA reduction grouping is a function of the sequence's whole physical
*cell set* relative to the kernel's reduction/split grid; the unified allocator lets that set depend on
scheduling/allocator history (gaps from padding, head-reset reuse), so the result is not reproducible.
The non-unified cache is green because each sequence owns a contiguous region starting at cell 0.

> **Report to the maintainer before any `patches/`/`release.json`/docs-header change. This is WIP.**

---

## 0. TL;DR

- Confirmed: a sequence's FA logits depend on the physical cells it occupies, not on its token
  contents. Non-unified is green; unified is red (issue #134 / TODO #52 part C).
- **The dependence is on the sequence's cell *set*, not merely its start offset.** A uniform
  start-alignment (`LLAMA_KV_ALIGN=64`) fixes the one-shot prefill probe but does **not** fix the
  decode/server case.
- The required quantum is the **kernel's reduction/split period**, which is model- and shape-dependent:
  it is not a constant. 4B (GQA 4) takes the RDNA4 band WMMA; GSQ/qwen4exp (GQA **12**) is *excluded*
  from the band (`4 < gqa_ratio <= 8`) and takes generic MMA with a different `nbatch_fa`/split.
- Alignment itself creates gaps, and the allocator's head-reset then reuses them, **fragmenting a
  growing sequence**; suppressing the reset reduces but does not remove the residual.
- **Proposed fix (direction 2):** give each sequence a contiguous, tile-aligned region in the unified
  attention cache so its cell set is a function of its own history only, and view its K/V from the
  region base (i.e. reproduce non-unified anchoring inside the shared pool). Kernel-side anchoring
  (direction 1) is a complement, but only works if a query tile never mixes sequences.

---

## 1. Instrumentation

- `tools/test-multiseq-offset.cpp` — the in-process offset probe. Modes:
  - `MSQS_MODE=only0` (default) / `solo` / `both`.
  - `MSQS_PRE=N MSQS_DEC=M` — prefill N then a single decode batch of M (exercises the band).
  - `MSQS_PRE=N MSQS_STEPS=S` — N prefill then S steps of a batch with **seq0 and seq1 both decoding**
    (server-like); `MSQS_ALONE=1` drops seq1 from the step batches.
  - `MSQS_BFILL_LIST=a,b,c` — sweep the companion fill in one process.
- `tools/align-diag.diff` — the diagnostic change to `src/llama-kv-cache.cpp`:
  `LLAMA_KV_ALIGN=N` rounds the allocation start up to a multiple of N;
  `LLAMA_KV_NO_RESET=1` disables the head-reset heuristic; `LLAMA_KV_OFFSET_DEBUG=1` prints the
  allocated cell range per `find_slot` call (`KVSLOT seq=… first=… last=…`).
- `tools/msq_plain.sh` — the server reproducer (`-np 2`, `--kv-unified`, `--spec-type none`).

## 2. The start-offset effect is real, and the quantum is 64 at the probe's scale

One unified context, seq0 tokens fixed, only the companion (seq1) prefill length `bfill` changes, so
seq0's first physical cell = `bfill`:

| model / KV / path | bfill=0 | 32 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|
| 4B f16, one-shot prefill | ref | **changed** | ref | — | ref | ref |
| GSQ q8_0, one-shot prefill | ref | **changed** | ref | **changed** | ref | ref |
| GSQ q8_0, prefill 256 + decode 4 | ref | **changed** | ref | **changed** | ref | ref |
| 4B f16, prefill 256 + 16 interleaved steps | ref | **changed** | ref | **changed** | ref | ref |

Every arm is a function of `bfill mod 64`. (`MSQS_ALONE=1` gives the same hashes as the interleaved
case, so seq1's *concurrent decode* is not the trigger — the companion's prefill length is.)

## 3. A fixed `LLAMA_KV_ALIGN=64` fixes prefill but not decode/server

- One-shot prefill probe: `LLAMA_KV_ALIGN=64` makes **every** bfill value return the reference hash
  (both 4B and GSQ). Good.
- Interleaved / decode probe (`MSQS_PRE=256 MSQS_STEPS=16`, 4B, align 64): **still** `bfill mod 64`
  dependent (`0/64/128/256` one value, `32/96` another).
- Server (GSQ, `msq_plain.sh`, `-np 2 --kv-unified`):

  | arm | A_solo vs A_conc1..3 |
  |---|---|
  | baseline | red (A_solo `479dbd37464c`, all concurrent differ) |
  | `LLAMA_KV_ALIGN=64` | red (A_solo changed to `e6cea154bd83`, concurrent still all differ) |

So a global start alignment is not the fix; it even perturbs the solo hash.

## 4. Why alignment fails: the sequence's cell *set* fragments

`LLAMA_KV_OFFSET_DEBUG=1`, 4B, unified, align 64, `MSQS_ALONE=1 MSQS_STEPS=3`:

```
bfill=32:  seq1 [0,31]   seq0 prefill [64,319]   then single tokens at 32, 320, 33
bfill=64:  seq1 [0,63]   seq0 prefill [64,319]   then single tokens at 320, 384, 321
```

The 64-alignment inserts an unused gap `[32,64)`. The `find_slot` head-reset heuristic
(`head_cur > used + 2*n_tokens`) then resets the head to 0 and reuses that gap, so seq0's cell set is
no longer the contiguous `[64, 320, …]` — it becomes `{64..319, 32, 320, 33, …}`, and the exact
scattering depends on `bfill` and the step count. With `LLAMA_KV_NO_RESET=1` the number of distinct
outcomes drops from 6 to 2, but a residual remains (the band's split has a period > 64 in absolute
cells), so even non-fragmenting aligned starts are not all equal.

## 5. It is unified-specific, and non-unified is exactly the fix shape

Probe, 4B, align 64, `MSQS_ALONE=1 MSQS_STEPS=16`:

| `MSQS_UNI` | result |
|---|---|
| 0 (`--no-kv-unified`) | **all bfill identical** (`26af2b891662b889`) — green |
| 1 (`--kv-unified`) | bfill 0/64/128/256 = `26af…`; 32/96 = `5a85…` — red |

In non-unified each sequence has its **own stream/region**, its K/V view starts at cell 0, and it is
contiguous — so the reduction grid is anchored to the sequence by construction. In unified, all
sequences share one region and the K/V view starts at 0 with a mask, so the grid is anchored to the
shared buffer and the sequence's cells are wherever the allocator put them.

## 6. The quantum is model/kernel-dependent (answers "is alignment layout-dependent?" — yes)

| model | arch | head_k/v | GQA | KV | FA path | `nbatch_fa` |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | qwen35 | 256 | **4** | f16 | RDNA4 **band** WMMA + generic MMA prefill | band 64; prefill 32 |
| GSQ Flash-Next | qwen4exp | 256 | **12** | q8_0 | band predicate needs `4 < gqa_ratio <= 8` → **excluded**; generic MMA `ncols2=4` | 64 (W≤4) / 32 (W 5-8, prefill) |

(`ggml_cuda_fattn_band_wmma_applies`: `gqa_ratio > 4 && gqa_ratio <= 8`.) So a single hardcoded
alignment constant cannot be correct across models, KV types, or verify widths; the correct grid is the
per-launch reduction/split period, which is exactly why the fix belongs in the kernel/allocator
anchoring, not in a fixed pad value.

## 7. Proposed fix (maintainer's D1 / campaign direction 2)

**Per-sequence contiguous, tile-aligned regions in the unified attention cache**, with the sequence's
K/V consumed from its region base. This reproduces the non-unified invariant (each sequence's cell set
is its own contiguous range, independent of co-resident allocation and reuse) while keeping the shared
pool and cross-sequence batching. The recurrent cache already has this shape (`find_slot`
gather/reorder); the attention cache does not.

Sub-points:
- A one-time pad/align of each sequence's region start (≤ one tile of waste per sequence) removes the
  offset *modulo the tile*; contiguity removes the fragmentation/reuse dependence.
- The head-reset reuse must not re-open a gap inside a live sequence's region (or regions must be
  defragmented on reuse), otherwise §4 recurs.
- Alternatively/complementarily, direction 1 (anchor each query tile's K-loop origin to the query
  tile's first visible cell) makes the reduction offset-invariant, but only if a query tile never
  mixes sequences; with the shared `ncols1` band tile that is not guaranteed today, so it needs the
  query tile to be sequence-homogeneous.

**Not the fix:** a global `LLAMA_KV_ALIGN` padding (validated insufficient, §3-§4); it is retained only
as a diagnostic.

## 8. Gates (unchanged, must stay green)

`scripts/gate-prefill-logits.sh` (KLD ≤ 0.005 / same-top-p ≥ 98 %), coherence `1c5d32ac537d`,
`scripts/validate-set.sh`, rule-5 `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, and
`archive/work/moe-verify-fusions/tools/multiseq.sh` / `tools/msq_plain.sh` (unified red vs
non-unified green is the reproducer).
