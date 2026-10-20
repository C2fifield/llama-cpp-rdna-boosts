# RESOLUTION — `rdna3-verify-fusions` (PR #192 + PR #193, @briansp2020)

**Status: RESOLVED (2026-10-20), promoted into `v12-097cf33d9-r95` as blocks 09/10.**  Both grant
RDNA3_0 (gfx1100-1102) the verify-band fusions that were RDNA4-only.  Nothing was left behind as a
`wip/` stub: the directory moved here from `wip/rdna3-verify-fusions/` and its `wip/CAMPAIGNS.md` row
was removed.  The two submitted format-patches, the acceptance note (`README.md`), the fold recipe
(`PROMOTION.md`) and the handover prompt are preserved unchanged in this directory.

## What landed

- **block 09** (`cpy-batch.cu`, `cpy-batch.cuh`): the batched run of up to 16 consecutive same-layout f32
  `GGML_OP_CPY` nodes is admitted on RDNA3_0 as well.  The gate becomes
  `GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_0(cc)`; `GGML_CUDA_FUSE_CPY_BATCH=0` still turns
  it off.
- **block 10** (`mmvq.cu`): `ggml_cuda_should_fuse_mul_mat_vec_q(..., verify_band = true)` admits
  RDNA3_0 through a single `verify_band_arch` term.  That predicate is the one gate the three 2..8-token
  Q8_1 pre-quantize fusions share (residual ADD + `rms_norm` → Q8_1 via `norm.cu:1210`, plain
  `rms_norm` → Q8_1 via `norm.cu:1245`, gated unary → Q8_1 via `ggml_cuda_should_fuse_mul_q8_1`, and
  GLU → Q8_1 via `ggml_cuda_glu_q8_1_mark`), so one predicate change admits all of them.
  `GGML_CUDA_FUSE_Q8_1_VERIFY=0` still turns them off.
- **block 10** (`gdn-conv.cu`): the direct GDN conv takes the 2..255-token verify batches on RDNA3_0 too
  (`verify_ok` becomes `conv_verify && T >= 2 && (RDNA4 || RDNA3_0)`).  The pre-existing single-device
  restriction (`GGML_CUDA_CONV_FUSION_MULTI` / `ggml_backend_cuda_get_device_count() <= 1`) is kept.
  `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` still turns it off.
- **block 10** also carries the comment updates in `ggml-cuda.cu` (`ggml_cuda_should_fuse_mul_mat`) and
  `ggml-cuda-try-fuse.cu` (the GLU → Q8_1 marking note).

RDNA3_5 stays out by design: `GGML_CUDA_CC_IS_RDNA3_0(cc)` is `0x1100 <= cc < 0x1150`, so gfx1150/1151
and RDNA4/older evaluate exactly as before.  No device-code path changed — only host-side predicates —
so the "RDNA4 and RDNA3_5 unchanged" claim is structural, not merely measured.

## Fold

Confirmed with `git blame` on the pre-fold 12-block tip (no lines had moved): `cpy-batch.*` → **block
09**, the other four files → **block 10**.  Blocks 00-08 came out as identical commit objects; block 11's
delta is byte-identical.  Pre-fold tree (r94 + the two contributor commits) and folded tree are both
`5129694738a265725473fb173ad0cbfedc472c98`; `validate-set.sh` is green on a fresh tarball.

## Release gates on the folded tree

**gfx1100** (RX 7900 XTX 24 GiB, ROCm 10.1.0-gfx1100, single visible GPU) — the platform the change
targets.  All arms are same-binary A/B: default vs `GGML_CUDA_FUSE_Q8_1_VERIFY=0
GGML_CUDA_FUSE_GDN_CONV_VERIFY=0 GGML_CUDA_FUSE_CPY_BATCH=0` (the r94 path).

- `test-logits-width-probe` `W=1..8`, `KV=q8_0`, default == all-off, **PASS (worst maxdiff 0)** in all
  six cells: 27B P=256 `c37241bd00c1d561`, 27B P=3584 `86541410b7c618b0`, 30B-A3B P=256
  `899608bf91600e1b`, 30B-A3B P=3584 `63ee46bb518130a7`, 35B-A3B P=256 `423a5f353f0a0e45`,
  35B-A3B P=3584 `6cdc0c5a6601264d`.  The first four reproduce the candidate note exactly.
- Greedy text, 400 tokens, seed 42, temp 0, `none == draft-mtp n3`: 27B prose `0f57c09e9ce2`, 35B-A3B
  prose `4111be939ff4`, 27B code `cfcc349aaa5a` (identical within each pair).
- Four-axis adaptive-MTP gate, `-n 3000`, reasoning pinned, 27B Q4_K_M, 12k ctx, q8_0 KV, default vs
  all-off: **R 71.3 vs 68.8 (+3.6 %), P 92.6 vs 89.6 (+3.3 %), C 99.1 vs 95.5 (+3.8 %), K 108.3 vs 104.4
  (+3.7 %)**, phase-switching prompt **76.8 vs 73.9 (+3.9 %)**.  Acceptance on the R axis (with
  `--log-verbosity 4`): **0.52065** (1828 accepted / 3511 generated), mean len 2.56, acc per pos
  (0.722, 0.500, 0.340) — far above the ~0.45 pos-1 floor, so no draft-vs-verify divergence.
- `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, q8_0 KV, 3 interleaved rounds, 27B UD-IQ3_S (the Q4_K_M
  arm OOMs a single 24 GiB card at the rs cache): B=1 flat (49.45/49.49 default vs 49.60/49.50 off,
  round 1 is a cold 47.40), **B=4 139.5/139.8 vs 135.1/135.2 (+3.2/+3.4 %)**, **B=8 178.1/178.2 vs
  173.3/174.4 (+2.2/+2.8 %)**.
- `scripts/gate-prefill-logits.sh --ab "<all three off>"`: mean KLD **-0.000000**, same-top-p
  **100.000 %** (PASS).
- `test-backend-ops -b ROCm0 -o RMS_NORM,ADD,MUL_MAT,CPY,CONCAT,SSM_CONV`: **3130/3130**, 2/2 backends.
- `scripts/check-gpu-faults.sh` over every log plus the kernel journal: clean.

**gfx1201** (3x R9700, ROCm 10.1.0-gfx120X) — must be byte-identical to r94:

- Coherence (4B Q8_0, 3 GPU, `-sm tensor`, seed 42): `1c5d32ac537d`, identical for the r94 and folded
  binaries and equal to the r94 record.
- `test-logits-width-probe` 27B Q8_0, q8_0 KV: P=256 `0183b188feb7435c` and P=3584 `469018da1cb5f614`,
  identical r94 vs folded, PASS (worst maxdiff 0) on both.
- `gate-prefill-logits.sh`: **0.000707 / 98.755 %** for both binaries (identical), matching the r94 record.
- `test-backend-ops` subset 3130/3130; GDN model + MTP under `-sm tensor -ncmoe 40` (the config where
  `cpy_batch` fires 1350 times) text `6586f4fafda3` identical r94 vs folded; faults clean.

**gfx1151** (Strix Halo, gfx1151) — must be inactive.  Both trees built.  4B Q8_0 coherence
`1c5d32ac537d` for r94 and folded.  35B-A3B Q3_K_M width probe P=256: r94 == folded == folded-all-off
`5de85975d341f80f`, PASS maxdiff 0.  Prefill-logit (9B Q6_K, ctx 512 x 40 chunks): both trees
**0.000000 / 100.000 %**.  MTP n3 text `ddfdb23c0607` identical, the only `GDN_CONV_DIRECT` line is the
pre-existing `C=8192 T=512` prefill conv (identical in both), and `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1`
prints **no** `cpy_batch` run — all three fusions are inert as designed.

**Per-block build:** `gate-per-block-build.sh --changed-from r94-ref` — blocks 00-08 unchanged (identical
commits), 09/10/tip built: **3/3, 0 failed**.

## Open gap (accepted, not measured)

The batched f32 CPY half is **not** single-device restricted and is an actively used multi-device path:
on 3x gfx1201 with the GDN model + MTP, `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1` shows **450 `cpy_batch` runs
under `-sm layer` and 1350 under `-sm tensor`**, default-on.  No RDNA3_0 multi-GPU box exists in the
fleet (fingon is one gfx1100 plus a gfx1036 iGPU, halo is gfx1151), so RDNA3_0 `-sm tensor` / `-sm layer`
was **not** measured.  The maintainer's decision (2026-10-20) is to ship default-on: the code is
arch-identical and the RDNA4 multi-device path is heavily exercised, and community members running
multiple gfx1100 cards will report.  Likewise **gfx1101 and gfx1102 are inside the macro but were not
explicitly validated** (the author did not test them and neither did this release).

## Files

- `0001-mmvq-the-verify-band-Q8_1-fusions-on-RDNA3_0-too.patch` (PR #192, as submitted on r94)
- `0002-cuda-the-GDN-conv-verify-path-and-the-batched-f32-CP.patch` (PR #193, as submitted on r94)
- `README.md`: the gfx1100 acceptance note for the candidate tree
- `PROMOTION.md`: the fold recipe and release-gate checklist used for r95
- `HANDOVER-PROMPT.md`: the handover prompt for the main development box
