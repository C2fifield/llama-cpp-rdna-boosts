# MANIFESTS — apply order and verification contract

**What this file answers:** in what order the blocks apply, what "verified" means for each one, and what to
do when an apply fails.  The current release identity (fork point, tip/tree, block count, per-artifact
sha256) is [`release.json`](release.json); the dated record of every release and re-validation is
[`WORKLOG.md`](WORKLOG.md).  Dated validation records from the pre-V2.0 era (the 16-block set, its
per-block table and its release-by-release narrative) are preserved in
[`archive/docs/validation-history.md`](archive/docs/validation-history.md) and do **not** describe the
current delivery.

## Current delivery

12 blocks (`block 00` + `blocks 01-11`) against `release.json.base`, canonical chain `release.json.tip` /
`.tree`.  `scripts/validate-set.sh` is green (checksums, then a strict 12/12 apply on a fresh tarball,
tree equality).  Each block is **one** `git am` commit and the series is **linear**: apply in order, never
reorder, never insert.

## Apply order

| # | patch | subject | notes |
|---|---|---|---|
| 00 | `0000-…-block-00-foundation-…` | foundation: band uniformity, Meta wrapper, allocator | assumed by every later block |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-…` | hybrid HIP all-reduce | RDNA4 gate, RCCL elsewhere |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth | `--spec-type draft-mtp-adaptive` |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill | two arch-segregated kernels + rollback tracker |
| 04 | `0004-…-block-04-BF16-KV-cache-…` | BF16 KV + native-BF16 FA | |
| 05 | `0005-…-block-05-flash-attention-band-…` | FA band / native quantized K-V / derived kq mask | |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM | |
| 07 | `0007-…-block-07-k-quant-boosts-…` | k-quant boosts (`mmvq` VDR, Q6_K MMQ, Q1_0, grid pad) | **the decode-numerics block** — see `GREEDY-PURITY.md` §2/§9 |
| 08 | `0008-…-block-08-fused-kernels-…` | fused kernels | fusions + fused MoE gate+up+GLU + per-shape `mmvq` `nwarps` |
| 09 | `0009-…-block-09-host-resident-MoE-…` | host-resident MoE | slab, expert cache, prefill cache staging |
| 10 | `0010-…-block-10-qwen4exp-…` | qwen4exp | model, hyper-connections, QSA/indexer, async graph-input ring |
| 11 | `0011-…-block-11-campaign-remainder-…` | campaign remainder | CPU wins, backend iface fills, DFlash, FA build grouping, mmap reclaim |

`rdna-boosts-all.patch` (repo root) is the entire net as ONE patch.  It is a convenience for a one-off
build at the fork point; it is **not** the reviewable increment and not a substitute for the per-block
flow.

## Apply (fresh checkout at the fork point)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(jq -r .base /path/to/rdna-boosts/release.json)"
bash /path/to/rdna-boosts/scripts/apply-all.sh .     # strict 12/12 git am, tree checked vs release.json
```

`scripts/apply-all.sh` is the only supported flow.  **Do not `git apply` the concatenated series** — it
silently drops hunks (measured: 30 files / 2483 lines instead of 35 / 6094).  The script:

1. refuses a dirty tree or an existing `rdna-boosts` branch;
2. applies the whole series with strict `git am` (one commit per block, original subjects);
3. on failure, aborts and retries the series **once** with `git am -3` (3-way against the recorded blob
   ids), warning that the merged tree may differ from the canonical one;
4. on the strict path, asserts the applied tree equals `release.json.tree` — a stale fork point or a
   partially regenerated set fails here rather than at the build;
5. `--include-beta` then applies `beta/*/*.patch` on top of the *verified* tree (one commit each).  The
   checkout is then **not** the release tree, which the script reports.

## Release gates (all mandatory, recorded in the release's `WORKLOG.md` entry)

`scripts/validate-set.sh` covers patch integrity only (checksums + a fresh-tarball strict apply) and runs
in CI.  The model-level gates are manual:

1. **Coherence** — same-seed `llama-cli` output identical to a known-good build (dense and, for a change
   that touches the MoE path, a host-expert config).  `GGML_CUDA_ALLREDUCE=nccl` is not a bit-identical
   reference under `-sm tensor`; treat it as a smoke comparison.
2. **MTP** — the adaptive-MTP four-axis gate, `-n 3000` (2000 floor), `--reasoning on` for R and `off` for
   P/C/K, plus the stock-relative `llama-batched-bench -npl 1,4,8` arm.  Protocol:
   [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md).
3. **Prefill logits** — `scripts/gate-prefill-logits.sh`: mean KLD <= 0.005 and same-top-p >= 98 % against
   a recorded known-good base.  **Required for any change to a prefill kernel** (GDN/SSM chunked, FA,
   MMQ/MMVF/MMB, MoE prefill) or to a default-on approximate path.  Protocol:
   [`benchmarks/prefill-logit-methodology.md`](benchmarks/prefill-logit-methodology.md).
4. **Per-block build** — `scripts/gate-per-block-build.sh` on a clean `all` build per commit (bisection and
   review walk the blocks individually, so every block must build on its own).  Use `--changed-from` to
   rebuild only the blocks a release changed.
5. **GPU faults** — `scripts/check-gpu-faults.sh` on the run's journal + log (never grep a log by hand).

Run the GPU gates **alone**: two concurrent GPU jobs change the arena/band layout and invent failures.

## Verification per block

The per-block command is a *targeted* check for a block's own surface; the release gates above still run
on the tip.

| block | verify | expected |
|---|---|---|
| 00 | `test-backend-ops -o FLASH_ATTN_EXT`; `test-logits-width-probe` W=1..8 | pass; one hash across the band |
| 01 | same-seed coherence on 2 and 3 GPUs (`-sm tensor`), MTP `none == draft-mtp` | identical |
| 02 | `tests/test-speculative-adaptive`, `tests/test-arg-parser`; `llama-server --spec-type draft-mtp-adaptive` smoke | pass |
| 03 | `test-backend-ops -o GATED_DELTA_NET`; `test-recurrent-state-depth`, `test-recurrent-state-rollback` | per-arm FAIL/PASS set unchanged from the recorded baseline (the counts differ per box — never gate on zero) |
| 04 | `test-backend-ops` bf16-KV FA cases + a bf16-cache model run | pass; coherence identical to f16 |
| 05 | `test-backend-ops -o FLASH_ATTN_EXT` (full matrix); width probe; prefill-logit gate | 6391/6391 on the r92 tree; probe PASS; KLD <= 0.005 |
| 06 | `test-backend-ops -o MUL_MAT` (MMQ cases) + prefill A/B | pass; prefill not slower at depth |
| 07 | `test-backend-ops -o MUL_MAT`/`MUL_MAT_ID` (K-quant cases); `llama-perplexity` before/after | pass; **PPL must be unchanged** (it is a numerics change, not a quality change) |
| 08 | `test-backend-ops -o MUL_MAT_ID`; MoE decode + `llama-batched-bench -npl 1,4,8`; width probe on a MoE model | pass; one hash across the band; verify widths not slower than stock-relative |
| 09 | host-expert coherence (`-ncmoe`/`-cmoe`), `test-llama-archs`, `MOE_EXPERT_CACHE_VALIDATE=1`; for the batched f32 CPY: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_CPY_BATCH=0` | pass; no `////`, no fault; cache-on == cache-off text; probe maxdiff 0 and identical hashes |
| 10 | `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum); `test-llama-archs -a qwen4exp`; sparse-vs-dense perplexity ratio (`LLAMA_QSA_SPARSE_FA=0`); `scripts/gate-qwen4exp-quant-coherence.sh`; for the verify-band fusions: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_Q8_1_VERIFY=0` + `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` | pass; ratio close to 1; probe maxdiff 0 and identical hashes (`GREEDY-PURITY.md` §11) |
| 11 | dense coherence + the block's own arm (DFlash/CPU/mmap): `-ncmoe` peak-RSS check with and without the reclaim | identical text; RSS as recorded in `patches/README.md` |

## Failure handling (agent instruction)

Apply in the order above, to a branch from `release.json.base`, with `scripts/apply-all.sh`.  If a block
fails to apply:

1. Let the script's `git am -3` retry run; if the merged tree differs from `release.json.tree`, **stop** —
   the checkout was not at the recorded base, or the set is stale.
2. For a true conflict, resolve the hunks **in the block that owns the code** (never in the last block),
   then regenerate the set (`scripts/make-patches.sh`) rather than leaving a hand-edited patch.
3. Do not skip a block.  The series is linear and each block assumes its predecessors; there is no
   supported "apply a subset" flow (including for the k-quant block — see `GREEDY-PURITY.md` §8).
4. Record what happened in `WORKLOG.md`, with the base/tip/tree you ended up with.

## Known notes (working)

- **`test-backend-ops.cpp` is shared by several blocks** (03/04/05/07/08/10).  Their hunks sit in different
  case regions, so a re-base is local to the failing file — but grep every patch that touches the file
  before concluding a case is missing.
- **Block 08 is one diff on purpose.**  Its matchers and the `mmvq` / `mmq` / `norm` / `quantize` kernels
  they dispatch to co-developed over one kernel region; do not try to split it.  (The `ggml_cuda_try_fuse`
  translation unit itself is block 10's.)
- **Blocks own their code.**  A fix goes into the block whose surface it changes, even if that means
  amending an early block; appending it to the last block is what made the per-block build gate necessary.
- **The set must apply whitespace-clean.**  `apply-all.sh` prints no git whitespace warnings; a warning is
  a regeneration bug, not a cosmetic issue.
- **Block-number collisions in old records:** documents written before 2026-10-13 use the
  pre-V2.0 16-block numbering ("block 12" = hybrid all-reduce there, but "block 10" = the k-quant
  umbrella).  `WORKLOG.md`, `archive/` and the dated notes in `archive/docs/patch-notes-history.md` are
  historical and keep their numbering; `patches/README.md` and this file use the V2.0 numbers, and the
  block subjects are the table above.

## History

Dated validation records, the retired block-structure churn and the pre-V2.0 per-block verification table:
[`archive/docs/validation-history.md`](archive/docs/validation-history.md).  Baseline/re-base records:
[`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).
