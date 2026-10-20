# `kv-cell-offset-nondeterminism` — RESOLUTION (CLOSED 2026-10-19, no delivery change)

**Status: CLOSED.**  The mechanism is fully characterised, the cheap fix (chunking / band-boundary
alignment) is **refuted by measurement**, and a *sufficient* fix is identified but deliberately **not
implemented** (it is an upstream unified-allocator redesign with no throughput win — see §4).  Nothing in
this directory is part of the delivery; no `patches/` or `release.json` change accompanies this file.

**The answer in one paragraph.**  A sequence's flash-attention result is a function of its KV cells'
**tile index modulo `pb`**, where `pb` is the per-launch **KV-split accumulator count** (`parallel_blocks`
for the tile kernel; the fixed band `P` otherwise) — plus the in-tile offsets/contents, which never
change in this problem.  A shift of the sequence's cells by exactly `pb × nbatch_fa` cells is
**bit-identical**; any smaller shift can change the logits.  In a unified cache every new **decode**
token takes the next cell from the *shared* head, so its residue (and therefore the result) is a function
of the co-resident session's allocation history — that is the non-determinism.  **Workaround (unchanged,
re-verified): `--no-kv-unified` when strict reproducibility is required.**  Issue #134 stays
upstream-inherent.

---

## 1. Why this file exists

The campaign was parked on 2026-10-11 with the working model "a sequence's attention result depends on the
physical **set** of KV cells it occupies, modulo the FA reduction tile (`nbatch_fa` = 32 or 64,
model/shape dependent), and a fixed start alignment is insufficient".  That model was directionally right
but not actionable: it named the wrong quantum (the tile) and did not explain why the *server* case stays
red when the prompt start *is* aligned.

This session re-derived the rule from the launcher code and measured it on r91 (gfx1151, ROCm 10.1),
including the maintainer's proposed fix ("chunk all unified sessions by the decode band width and start
every prompt at a band boundary").  Result: the quantum is the **split period**, not the tile, and the
**decode head's residue** is the decisive variable — so prompt-start alignment cannot work, but
period-granular *ownership* would.

## 2. Method

Build: canonical r91 tree in `~/llama.cpp` (HEAD tree `90507dfb528cbfe0c688261226d18934653b1b61` =
`release.json.tree`), `~/bin/build-llama-rocm` (ROCm 10.1), `halo` (gfx1151, 1 GPU, nsm 20).
No delivery code was changed; the tree was returned to clean after the measurements.

| tool | what it does |
|---|---|
| [`tools/test-multiseq-runs.cpp`](tools/test-multiseq-runs.cpp) | **new.**  Run-script probe: `MSQS_RUNS="1:288,0:256,1:3552"` processes those `seq:n_tokens` runs in order, then `MSQS_STEPS=N MSQS_STEP_SEQS="0,1"` decode steps.  The report sequence's token stream and *positions* are identical across arms — only its **physical cell placement** changes.  Hashes the full last-token logit vector (FNV-1a, bitwise). |
| [`tools/fa-config-diag.patch`](tools/fa-config-diag.patch) | **new.**  `GGML_CUDA_FA_CONFIG_DEBUG=1` prints the FA launch config (`q, ncols2, n_kv, nbatch_fa, tiles_kv, pb, grid, stream_k, band, nsm`) — the instrument that pinned `pb`.  WIP diagnostic, not for delivery. |
| `LLAMA_KV_CACHE_DEBUG=2` | cell-map dump; used to confirm the intended placement (`find_slot` appends linearly in these scripts, no head reset). |
| [`tools/test-multiseq-offset.cpp`](tools/test-multiseq-offset.cpp) | the 2026-10-11 companion-fill probe (kept; `MSQS_BFILL_LIST` sweep). |
| [`tools/msq_plain.sh`](tools/msq_plain.sh) | the server reproducer (unified red vs non-unified green). |

Build the probes into a scratch tree:

```bash
cp archive/work/kv-cell-offset-nondeterminism/tools/test-multiseq-runs.cpp ~/llama.cpp/tests/
printf 'llama_build(test-multiseq-runs.cpp)\n' >> ~/llama.cpp/tests/CMakeLists.txt   # (add next to the other llama_build lines)
git -C ~/llama.cpp apply archive/work/kv-cell-offset-nondeterminism/tools/fa-config-diag.patch   # optional: FACFG
cmake -S ~/llama.cpp -B ~/llama.cpp/build-rocm && cmake --build ~/llama.cpp/build-rocm --target test-multiseq-runs -j 16
```

## 3. The evidence

All runs: 2-sequence unified context, 4B = `Qwen3.5-4B-Q8_0` (f16 KV, `nbatch_fa=32`, decode `pb=30`),
`T` = total cells occupied before the final decode step (held constant unless stated), `MSQS_CTX=8192`.

### 3.1 Tile alignment is NOT the quantum (the parked model's error)

Prompt at offsets that are all multiples of `nbatch_fa` (32) — four different results:

| offset (cells) | 32 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|
| logits hash | `b8bc508b86b0614e` | `5ab10c344414cf93` | `41b96d97e4bac7f2` | `9ba3b5ea1888eb3b` | `ae54bcf50512d150` |

### 3.2 The quantum is `pb × nbatch_fa` (the split period)

Shift every arm by **960 cells = 30 × 32 = `pb × nbatch_fa`** — bit-identical:

| offset | base hash | +960 | hash |
|---|---|---|---|
| 32 | `b8bc508b86b0614e` | 992 | `b8bc508b86b0614e` |
| 64 | `5ab10c344414cf93` | 1024 | `5ab10c344414cf93` |
| 128 | `9ba3b5ea1888eb3b` | 1088 | `9ba3b5ea1888eb3b` |
| 256 | `ae54bcf50512d150` | 1216 | `ae54bcf50512d150` |
| 288 | `fbc7a801f51babc0` | 1248 | `fbc7a801f51babc0` |

Cross-model confirmation on **35B-A3B Q4_K_M** (`ncols2=8` -> decode `pb=60`, `nbatch_fa=32`, so period
1920): offset 32 = `a0c2043d16ce7ee1` = offset **1952**; offset 512 = `1197fe0ab0cede02` = offset 992;
offset 272 = `45ee2f525b98d804` (different).  The period is `pb × nbatch_fa` on both models, and `pb`
itself is model/config dependent (30 vs 60 here).

### 3.3 The decisive variable is the DECODE HEAD's residue, not the prompt start

Prompt cells **fixed** at offset 288 (tiles 9..16); only `T` (hence the final decode cell's absolute cell)
varies:

| final cell tile -> block (`tile mod 30`) | hash | note |
|---|---|---|
| 35 -> 5 | `fbc7a801f51babc0` | head before the prompt's blocks |
| **42 -> 12** | **`ba09fe281f969372`** | **head shares a partial with a prompt tile** |
| 50 -> 20 | `fbc7a801f51babc0` | head after |
| 80 -> 20 | `fbc7a801f51babc0` | head after |
| 110 -> 20 | `fbc7a801f51babc0` | head after |

So *moving one token's cell into a residue class already used by the prompt* changes the result, while
keeping it in an unused class does not.  Under `--kv-unified` that cell is whatever the shared head
happens to be; under `--no-kv-unified` it is deterministic.

### 3.4 Control: non-unified is green, and the neutral arm matches it

| `MSQS_UNI` | off 32 | off 288 | off 544 | off 1024 |
|---|---|---|---|---|
| 1 (unified) | `b8bc508b86b0614e` | `fbc7a801f51babc0` | `fbc7a801f51babc0` | `5ab10c344414cf93` |
| 0 (no-kv-unified) | `fbc7a801f51babc0` | `fbc7a801f51babc0` | `fbc7a801f51babc0` | `fbc7a801f51babc0` |

### 3.5 The prefill path is NOT exposed (only decode is)

* Prefill logits are invariant to the **tile-aligned offset** (`1:B,0:512`, B = 0,32,64,...,3072 -> one
  hash `db4f6a59b83fb73c`; B = 1/8/16/48 each differ).
* Prefill logits are invariant to the **padded view length** with the cells held fixed:
  `1:320,0:256` under `LLAMA_KV_N_PAD_MIN` = 256/512/1024/2048/4096 (`n_kv` 768 -> 4096, `nblocks_stream_k`
  changes with it) -> one hash `aa11739d495fcf46` for every arm.

So prompt processing is already placement-invariant at tile granularity; the campaign's `LLAMA_KV_ALIGN`
probe "fixed prefill but not decode" for exactly this reason.

### 3.6 Why this is the rule (code)

`launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`):

* tile kernel, `n_q <= 8` (decode/verify): `parallel_blocks` is the KV split; the kernel takes KV
  iterations **round-robin** — `k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 += gridDim.y*nbatch_fa` — so a
  tile's partial accumulator is `tile mod pb`.
* `pb` comes from an efficiency search over `ntiles_KV = ceil(n_kv/nbatch_fa)` — i.e. from the **shared
  view length**, which includes the other session's cells.  Measured: 4B `pb` = 8 (`n_kv` 256), 16 (512),
  24 (768), **30 (>=1280, capped)**; 35B-A3B `pb` = 24 (768), **60 (4352, capped)**.
* `flash_attn_combine_results` then sums the `pb` partials **linearly in block-index order**.  A
  fully-masked tile contributes exactly zero, so appending masked tail tiles is a no-op (the code comment
  says so) — but *re-assigning* a live tile to a different accumulator (i.e. shifting by a non-multiple of
  `pb`) changes the association and the fp32 rounding.
* RDNA4 band: `P` is deliberately fixed by device (`nsm`, or `(3*nsm)/4` for 2-byte K/V) and independent
  of `n_q`/`n_kv`, so the same rule holds with period `P × nbatch_fa`; not measurable on this box (the
  band is RDNA4-only and `halo` is RDNA3_5) — code-derived from `fattn-rdna.cuh` +
  `launch_fattn`'s band branch.

The 2026-10-11 "quantum 64" observation is explained: in the regime where `pb == ntiles_KV` every tile owns
its own accumulator, so *any* tile-aligned placement is invariant and the apparent quantum is the tile.
The rule collapses to "tile" only there.

## 4. The proposed fix, and what a sufficient fix would take

**Refuted: "chunk unified sessions by the decode band width and start every prompt at a band boundary".**

1. The quantum is `pb × nbatch_fa`, not the band/tile width (§3.1) — a tile-aligned start still changes the
   result.
2. Even period-aligned **prompt** starts leave each new **decode** token at the shared head, whose residue
   follows the co-resident session's allocation history (§3.3).  That is the part the field symptom is
   made of.

**Sufficient (identified, NOT implemented): period-granular per-sequence *ownership* of cells in the
unified pool.**  Each session's cells are whole, period-aligned `pb × nbatch_fa` blocks that no other
session may write into (the session's head lives inside its own block; a partially-filled block's unwritten
cells are simply unused and masked out).  Then every cell's residue class is a function of the session's own
history, and shifts by whole periods are bit-identical (§3.2) -> deterministic.

Cost and caveats:

| item | 4B Q8_0 (f16 KV) | 35B-A3B Q4_K_M (f16 KV) |
|---|---|---|
| KV per cell (measured, `llama_kv_cache: size = ...`) | 32 KiB | 20 KiB |
| period (`pb × nbatch_fa`) | 960 cells | 1920 cells |
| worst-case reserved-unwritten waste per live session | **30 MiB** | **37.5 MiB** |

* `pb` must be taken from the FA config (capped value), and it differs by model, device and K/V type; it
  is currently computed inside `launch_fattn`, so the allocator needs a policy input plumbed from the
  backend.
* `pb` steps at shallow context (`pb == ntiles_KV` until the cap) — a deterministic event, but keyed on the
  **shared** total, so a period-granular allocator should use the capped value for the whole session.
* The allocator surfaces that assume token-granular packing and would all need review: `find_slot` /
  `apply_ubatch` / the head-reset heuristic, `seq_rm` / `seq_cp` / `seq_keep` / shift, SWA cell reuse
  (which deliberately recycles stale cells topologically), and the mask-skip's 256-cell group
  classification.  This is direction 2 of the campaign README (per-sequence regions) at block
  granularity — an upstream redesign.
* **No throughput angle**: it buys nothing measurable, and the field symptom is a rounding-level greedy
  near-tie flip.  The maintainer's decision (2026-10-19) is therefore to close, not implement.

## 5. Re-entry conditions and the validation plan

Reopen only if (a) upstream is willing to carry a period-granular unified allocator, or (b) a user needs
run-to-run determinism *and* unified KV (memory sharing) at the same time.

Mechanism gate (one process per arm, one GPU): `test-multiseq-runs` with
`MSQS_RUNS="1:OFF,0:256,1:T-256-OFF"`, `MSQS_STEPS=1 MSQS_STEP_SEQS="0,1"`, sweeping OFF:

* `OFF` and `OFF + period` must be **bit-identical** (period from `FACFG` = `pb × nbatch_fa`);
* `OFF` and `OFF + period/2` must differ (sanity: the gate can see the effect);
* with `MSQS_UNI=0` every arm must hash the same (the reference).

End-to-end gate: `tools/msq_plain.sh` — `--kv-unified` A_solo == A_conc1..3 (today: red), and the same
with `--no-kv-unified` (today: concurrent repeats green).  Keep the standard gates unchanged:
`scripts/validate-set.sh`, coherence, `scripts/gate-prefill-logits.sh`, the rule-5 batched bench.

## 6. Records

* [`README.md`](README.md) — the campaign handover (finding, condition matrix, hypotheses).
* [`FINDINGS-offset-mechanism.md`](FINDINGS-offset-mechanism.md) — 2026-10-11: the cell-*set* refinement,
  `LLAMA_KV_ALIGN` insufficiency, the diagnostic diff.
* [`FINDINGS-2026-10-16-revalidation.md`](FINDINGS-2026-10-16-revalidation.md) — r69 revalidation; the
  issue #151 graphs-axis claim not reproduced on Linux/ROCm 7.14 (TODO #67).
* `WORKLOG.md` 261019.012 (this closure), 2026-10-11 (r43), 2026-10-10 (r42); `TODO.md` #52; issue #134.
* `GREEDY-PURITY.md` §38 (the rulebook entry), `README.md` (the consumer gotcha), predecessor
  `archive/work/moe-verify-fusions/`.
