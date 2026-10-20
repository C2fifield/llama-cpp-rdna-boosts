# Historical: the pre-V2.0 16-block notes

Preserved verbatim from `../../README.md` (2026-10-20) when the README was cut down to the current set.
**Nothing here describes the current delivery**: the patch files are now `0000`-`0011` (block 00 + blocks
01-11, the V2.0 re-organisation), the block numbers below are the pre-V2.0 16-block numbering, and the
purity guidance in the last note of this section is superseded by `../../GREEDY-PURITY.md` (the k-quant
block is no longer the last patch, so "one line to drop from `apply-all.sh`" no longer applies).

The V2.0 block subjects and the old->new mapping are in `../../README.md` and `../../BASELINE.md`; the
per-block reference is `../../patches/README.md`; the dated amendment record is
`patch-notes-history.md`.

---

<details>
<summary>Historical: the pre-V2.0 16-block notes (provenance/edit history only)</summary>

### The 16 blocks

| patch | what |
|-------|------|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25) + Vulkan masked-V/freed-cell fixes (dead columns never read V). The base every later block applies on top of. |
| `0001` | adaptive MTP draft depth (`--spec-type draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn (+ the HIP masked-V/freed-cell fixes since 2026-09-10) |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf (WMMA path also runs on RDNA3.0/3.5, tuned head limits) |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | **general system-operations bucket** — the delivery's **catch-all** for changes that fit no other block: the FA instance build-time work, `--fit` under `-sm tensor`, the host-buffer input layer, the tiny-CPU-split single-thread fix and (r12) the op-offload **H2D staging ring** + tensor-split op-offload.  Named for what it is since r12; the original host-buffer revert content is long gone (upstream reverted #24233 in #28604).  Amended r13 (issue #52): the tiny-CPU-split heuristic now counts what a graph *reads* — not only its node outputs — so a CPU-offloaded FFN chunk keeps the full thread pool |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04; amended 2026-09-07 with the mul_mat+add through-view shape guard, PR #15). **Now folds the `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning, and the GDN/PLE conv1d + narrow-row RMS-norm prefill fusions** (absorbed from the former `archive/work/mmb-general` campaign). |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4 + RDNA3_5 + RDNA3_0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes). The masked-V/freed-cell fixes it once carried now live in blocks 00 (Vulkan) and 03 (HIP). |
| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 + **V5** native bf16 K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert in `archive/work/block-15-campaign-wins/ab/`).  ~3.4 GiB/GPU + ~1.2 GiB host saved on qwen4exp, ~800 MiB/GPU + ~800 MiB host on dense models, at ~1.3 % prefill / ~0.3 % decode. **Now also folds the qwen4exp/QSA/HC/indexer campaign** (`qsa3` packed-block WMMA attention, fused indexer top-k + prefill score fusions, HC16 native-BF16 producers, `hc_gate_mix`, sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the host-buffer/CPU/meta fixes) — the former `archive/work/mmb-general` work.  Also carries the r14-r21 **MoE expert-cache** work: auto sizing, host-pinned expert buffers, the op-offload H2D staging path, the compute-buffer slack, the fail-soft arena yield, the per-layer uniform slot re-size (`t.slots` fix), the `MTP_DRAFT_N_UBATCH` cap, and the cli-only wide-prefill drop.  **r22 adds the movable-boundary slab allocator** (ONE slab per device, reserved and mapped once, split by a movable boundary: compute buffer below, MoE arena above, HIP never called at runtime) which SUPERSEDES the per-allocation VMM pool (removed) and the fail-soft arena yield, sizes the arena on a decode-band pass, right-sizes the compute reserve to the observed workload, and makes the wide-prefill drop default-on for servers too — the TODO #42 abort is gone. **r27 folds PR #114's four qwen4exp decode fusions:** the latency-scheduled BF16 `hc_mix` up/collapse (`GGML_HC_UP_V2=0`), `HC_COMBINE` fused into that norm (`GGML_CUDA_FUSE_HC_COMBINE_MIX=0`), the shared-expert `sigmoid`-`mul`-`add` gate (`GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0`), and the GDN `beta` sigmoid folded into the sequential kernel (`GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0`); +2.4 % decode on qwen4exp, byte-identical text. |

> **The delivery is `0000`–`0015` (16 patches).**  Block `0016` was added in
> `v16-a55e952b8-r45` (the contributor decode/latency PRs) and **folded back into the 0000-0015 set in
> `v16-a55e952b8-r46`**: host-side/system work into `0006`, small-message all-reduce into `0012`, the
> decode/latency remainder (and issue #133) into `0015`.  (`v16-a55e952b8-r16` had briefly shipped a
> *different* `0016`/`0017` pair — op-offload staging redirect and per-device tensor-split guard — which
> r17 folded into `0006`/`0013`/`0015`; that old pair is unrelated.)

> **Block 15 (attention-memory wins) is part of the delivery since
> 2026-09-12** (`patches/0015`, promoted from
> `archive/work/block-15-campaign-wins/`; a fresh set is now **16 patches**,
> blocks 00-15).

> **Greedy-purity note (read before shipping):** on the K-split decode
> paths, block 10 (`0010`) is the only patch that changes decode numerics on
> ANY architecture — its VDR kernels reorder the fp32 reduction. Compute
> outputs are not bit-identical to a build without it (max logit diff 0.184
> vs 0.203 for flash-attn on/off; greedy streams are deterministic within a
> build but can flip across configs). This is a different rounding path, not
> a correctness change. If you require 100% greedy purity across builds, do
> not install `0010-…k-quant-boosts…patch` — it is one line to drop from
> `scripts/apply-all.sh`. Full discussion:
> [`GREEDY-PURITY.md`](../../GREEDY-PURITY.md). **Block-13 caveat (2026-09-02):**
> block 13 rewrites the small-batch mmvq decode kernel and is a second
> decode-numerics source on the rows that run it (short-K K<4096 ncols==1
> rows, MoE projections; ncols 2..8 and long-K rows were restored to the
> pre-block-13 K-split kernel by the 2026-09-02 fix). Excluding block 10 no
> longer reproduces stock bits exactly on those rows — see
> GREEDY-PURITY.md §9.

</details>
