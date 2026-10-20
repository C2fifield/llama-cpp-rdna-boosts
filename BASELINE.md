# BASELINE — provenance and drift policy

**What this file answers:** what the patches are generated *against*, how they are cut, and what to do when
upstream moves.  It is not a changelog — the current release, tip/tree and per-artifact hashes are in
[`release.json`](release.json) (the single source of truth) and the dated record of every re-base is in
[`WORKLOG.md`](WORKLOG.md).  Dated baseline records from the pre-V2.0 era (the 16-block set, the
`9113cc188`..`ebbb18522` re-bases, the old `baseline/*` checkpoints) are preserved in
[`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).

## Current baseline

| item | value |
|---|---|
| fork point (upstream `ggml-org/llama.cpp` master) | **`release.json .base`** |
| delivery | 12 blocks, `block 00` + `blocks 01-11` (`release.json .n_blocks`) |
| canonical chain | `release.json .tip` / `.tree` |
| release identity | `release.json .release` (`v12-<base>-r<N>`) |

The fork point has been held since the V2.0 set was cut: every `rN` is a **block amendment** at the same
base, never a re-base, so the block series and its recorded trees stay comparable.  A re-base is a
deliberate event with its own `WORKLOG.md` entry.

## What the patches are

Each `patches/000N-*.patch` is **one block commit** exported with `git format-patch`.  The blocks are the
delivery's unit of review, bisection and amendment; they are not a file split, and most blocks touch
several subsystems.  Block 00 is the foundation every later block assumes (band uniformity, the Meta
wrapper, allocator/graph-layout fixes).

| block | subject | concern |
|---|---|---|
| 00 | foundation: band uniformity, Meta wrapper, allocator | the decode/verify band rule, `ggml-backend-meta` split state, `ggml-alloc` view release, the multi-ubatch logits readback |
| 01 | hybrid HIP all-reduce | the internal AR for small tensors + per-size hybrid dispatch vs RCCL |
| 02 | adaptive MTP draft depth | `--spec-type draft-mtp-adaptive` and its controller |
| 03 | chunked gated-delta-net prefill | bf16/WMMA chunked GDN (two arch-segregated kernels) + the recurrent rollback tracker |
| 04 | BF16 KV cache + native-BF16 flash attention | the bf16 cache type and its native FA arm |
| 05 | flash-attention band / native quantized K-V / derived kq mask | the RDNA FA band kernels, native KV staging, the derived mask |
| 06 | MMB quantized prefill GEMM | bf16-WMMA dequant-weight GEMM and its tuning |
| 07 | k-quant boosts | `mmvq` VDR, Q6_K MMQ, Q1_0, grid pad — **the decode-numerics block** |
| 08 | fused kernels | conv1d/RMS/top-k MoE/verify fusions, fused MoE gate+up+GLU, per-shape `mmvq` `nwarps` |
| 09 | host-resident MoE | iface, scheduler, Meta, movable-boundary slab, expert cache, prefill cache staging |
| 10 | qwen4exp | model, hyper-connections, lazy reader, QSA/indexer, the async graph-input ring |
| 11 | campaign remainder | CPU wins, backend iface fills, DFlash, FA build grouping, mmap reclaim |

Per-block *content* and edit history is [`patches/README.md`](patches/README.md); per-block verification is
[`MANIFESTS.md`](MANIFESTS.md).  (`patches/README.md`'s dated notes below its header use the **pre-V2.0
16-block** numbering and are kept as provenance.)

## How the patches are cut

The **canonical chain is the `rdna-boosts` branch of the maintainer's personal fork**, rebuilt from
`release.json.base` with [`scripts/apply-all.sh`](scripts/apply-all.sh) and pushed on every release.  The
working `~/llama.cpp` checkout is *not* the canonical chain: it drifts past the base, so a raw
`git format-patch <base>..<tip>` there can export upstream commits as delivery blocks.

To regenerate: apply the set to a clean checkout at `release.json.base`, make the change as a commit on
the owning block (amend, do not append), then run `scripts/make-patches.sh` (defaults from
`release.json`) and `scripts/make-release.sh` (rewrites `release.json` and `rdna-boosts-all.patch`).  Never
hand-edit a committed patch as the fix — regenerate and re-verify.  See AGENTS.md → "Regenerate the
patches".

## Provenance of the two fixes that are not upstream's

Two changes carried since the fork lineage are worth knowing about when a hunk conflicts, because they are
not upstream code and have no upstream counterpart to merge:

1. **Test-harness seeding** (block `0003`) — the fork's chunked-GDN work had replaced
   `test-backend-ops`' `std::random_device` seeding with a fixed seed, which deterministically exposes a
   pre-existing numerical fragility in `rms_norm_back` / `cross_entropy_loss_back` on RDNA4.  The delivery
   keeps `random_device` seeding; the GDN kernels are unaffected either way.
2. **Meta compute-container headroom** (block `0000`/`0009`) — hybrid recurrent models create
   `~2*(n_rs_seq+1)` conv-state snapshot views per recurrent layer during graph allocation, which
   overflowed the original 16x container.  The headroom is now supplied by the compute-margin path
   (`GGML_COMPUTE_BUFFER_MARGIN_PCT`), see AGENTS.md → critical facts.

The full diagnostics for both are in [`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).

## Drift policy

The patches are static against the fork point.  When upstream master moves:

1. `scripts/apply-all.sh` is already the tested flow: strict `git am`, with a `git am -3` fallback on
   drift (3-way merge against the baseline blobs).
2. If the 3-way merge fails — the normal case in a fresh clone, where the pre-image blobs are absent —
   re-base the failing hunks against current master by hand, in the block that owns them.
3. **Do not hand-edit the committed patches as the permanent fix.**  When more than one block needs manual
   hunks, re-base the block commits on a clean checkout at the new base, regenerate the whole set
   (`scripts/make-patches.sh`), refresh `release.json` (`scripts/make-release.sh --base … --tip …
   --tree …`), and update the current-state headers (`README.md`, `MANIFESTS.md`, `patches/README.md`,
   `ENVIRONMENT.md` where a default moved).
4. Re-verify: `scripts/validate-set.sh` (strict apply on a fresh tarball, tree equality), a full build,
   the coherence gate, and the release gates in `MANIFESTS.md`.  Record the re-base in `WORKLOG.md` with
   the new base, tip and tree.

A patch that resolves *naturally* because upstream fixed the same thing locally is a re-base conflict, not
a bug: keep upstream's fix, drop the local hunk, and note it in the `WORKLOG.md` entry.
