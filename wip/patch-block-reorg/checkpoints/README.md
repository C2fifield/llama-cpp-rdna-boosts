# patch-block-reorg Phase-2 checkpoints

Exact, restorable snapshots of the Phase-2 `v2-phase2` chain.  A checkpoint is a git bundle of the
range `<release.json.base>..v2-phase2`; it needs `097cf33d9` present in the target repo (it is the
published upstream base, so any llama.cpp clone has it).

## ckpt13 — `v2-phase2-ckpt13.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt13` = `94f73ef44` |
| branch tip | `v2-phase2` = `94f73ef44` |
| block 5 / 8 / 9 / 10 | `6097ad7d5` / `134f545fd` / `7a1bbfcbb` / `408d8fd1a` |

Item 6b: the movable-boundary slab moved to `ggml-cuda-slab.cu` (block 10), with state accessors added
to `ggml-cuda-vmm.h`.  `ggml-cuda.cu` 7722 → 6640 lines.  `-ncmoe 40` and the standard same-seed gates
identical to r50; prefill-logit PASS; per-block `all` 12/12.  See `../PHASE2-STATUS.md` §13.

## ckpt12 — `v2-phase2-ckpt12.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt12` = `f52f3288f` |
| branch tip | `v2-phase2` = `f52f3288f` |
| block 5 / 8 / 9 / 10 | `6097ad7d5` / `134f545fd` / `7a1bbfcbb` / `318fec62f` |

Item 6a: the fusion dispatcher moved to `ggml-cuda-try-fuse.cu` / `.cuh` (block 10).
`ggml-cuda.cu` 9819 → 7722 lines.  Same-seed 4B/qwen4exp/35B identical; per-block `all` 12/12.
See `../PHASE2-STATUS.md` §12.

## ckpt11 — `v2-phase2-ckpt11.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt11` = `ea547cb4d` |
| branch tip | `v2-phase2` = `ea547cb4d` |
| block 5 / 8 / 9 / 10 | `6097ad7d5` / `134f545fd` / `7a1bbfcbb` / `65c266b5e` |

Item 5: the FA tensor-parallel hint (`llama-context.cpp`, block 05) and the host-expert policy
(`llama-model-loader.cpp`, block 10) extracted to named file-scope helpers with one-line calls.  Log
lines preserved by passing the caller's `__func__`.  Per-block `all` 12/12; same-seed gates green.
See `../PHASE2-STATUS.md` §11.

## ckpt10 — `v2-phase2-ckpt10.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt10` = `3802688c1` |
| branch tip | `v2-phase2` = `3802688c1` |
| block 5 / 8 / 9 / 10 | `75cb9c0b1` / `fcce5864e` / `96a7d0d16` / `6d6f9692e` |

Item 4: the FA native-KV / band predicate bodies moved to `fattn-rdna.cuh`, re-homed in block 05.
`fattn-common.cuh` 2421 → 2007 lines.  `FLASH_ATTN_EXT` 6389/6389 in default / `GGML_CUDA_FA_KV_NATIVE=0`
/ `GGML_HIP_FA_BAND_WMMA=0`; width purity PASS; prefill-logit 0.000707 / 98.755 %; per-block `all` 12/12.
See `../PHASE2-STATUS.md` §10.

## ckpt9 — `v2-phase2-ckpt9.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt9` = `18a2fd0f8` |
| branch tip | `v2-phase2` = `18a2fd0f8` |
| block 8 / 9 / 10 | `fcce5864e` / `96a7d0d16` / `fdd20f2c6` |

Item 3 step 3: the shared MMVQ vec-dot/VDR selectors moved to `mmvq-vecdot.cuh` and the dedicated
MoE kernel `mul_mat_vec_q_moe` + its launch moved to `mmvq-moe.cu` / `mmvq-moe.cuh` (re-homed in
block 10; 23 explicit launch instantiations).  `mmvq.cu` 3745 → 3368 lines.  `width_purity` PASS,
prefill-logit 0.000707 / 98.755 %, per-block `all` 12/12.  See `../PHASE2-STATUS.md` §9.

## ckpt8 — `v2-phase2-ckpt8.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt8` = `9d6730dc8` |
| branch tip | `v2-phase2` = `9d6730dc8` |
| block 8 / 9 / 10 | `fcce5864e` / `96a7d0d16` / `5d472a3eb` |

Item 3 step 2: the remaining RDNA launch-parameter helpers moved to `mmvq-rdna-params.cuh`,
re-homed in block 08.  `width_purity` PASS.  See `../PHASE2-STATUS.md` §9.

## ckpt7 — `v2-phase2-ckpt7.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt7` = `24ef342b3` |
| branch tip | `v2-phase2` = `24ef342b3` |
| block 8 / 9 / 10 | `51bfc6c7f` / `eb2105b51` / `50f5fa1d1` |

Item 3 step 1: the band-uniform `calc_nwarps` policy moved to `mmvq-rdna.cuh` via an override hook
(upstream branches restored), re-homed in block 08.  `width_purity` PASS.  See
`../PHASE2-STATUS.md` §9.

## ckpt6 — `v2-phase2-ckpt6.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt6` = `cf9c2e474` |
| branch tip | `v2-phase2` = `cf9c2e474` |
| block 8 / 9 / 10 | `8296626db` / `9cf2c7181` / `bf1fbfa55` |

**`ggml_cuda_try_fuse` extraction complete** (`try_fuse` 2244 → 1064 lines): the last hoist arms
(`glu_q8_1`, `shared_expert`) are in block 09 with their helpers (`topk_moe` is upstream, skipped).
See `../PHASE2-STATUS.md` §7b (cluster G).

## ckpt5 — `v2-phase2-ckpt5.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt5` = `8326733e1` |
| branch tip | `v2-phase2` = `8326733e1` |
| block 8 / 9 / 10 | `8296626db` / `7ffdd65e9` / `11ddfb213` |

Adds over ckpt4: the five clean block-08/09 arms re-homed in their introducer blocks (`gdn_gate`,
`scale_unary`; `l2_norm_pair`, `gate_beta_verify`, `shexp_down_gate`).  See `../PHASE2-STATUS.md`
§7b (cluster F).

## ckpt4 — `v2-phase2-ckpt4.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt4` = `9c1ec61d6` |
| branch tip | `v2-phase2` = `9c1ec61d6` |
| block 10 (amended) | `3238b04e9` |

All block-10-owned `try_fuse` arms extracted (`try_fuse` 2244 → 1262 lines).  See
`../PHASE2-STATUS.md` §7b (cluster E).

## ckpt3 — `v2-phase2-ckpt3.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt3` = `463c1d598` |
| branch tip | `v2-phase2` = `463c1d598` |
| block 10 (amended) | `952aa59f7` |

Adds over ckpt2: routed weighted-down (`mmvq.cu`) and the MUL_MAT pair (`mmq.cu`).  See
`../PHASE2-STATUS.md` §7b.

## ckpt2 — `v2-phase2-ckpt2.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt2` = `d86bcf088` |
| branch tip | `v2-phase2` = `d86bcf088` |
| block 10 (amended) | `fbed3f4c0` |

Adds over ckpt1: cluster A (conv1d + narrow-row norm), cluster B (cpy-batch / GDN / SSM / norm-fold
arms + helper hoists), and MWR.  See `../PHASE2-STATUS.md` §7b.

## ckpt1 — `v2-phase2-ckpt1.bundle`

| | |
|---|---|
| tag (in `~/llama.cpp`) | `pin/v2-phase2-ckpt1` = `bf7824a5d` |
| branch tip | `v2-phase2` = `d8ad56737` |
| block 10 (amended) | `1fa36b3ea` |
| Phase-1 checkpoint | `v2-phase1` = `a40841e1e` (tree == `af0595360`) |

Contents: the 4 `ggml_cuda_try_fuse` extractions inside block 10 — HC_COMBINE→HC_MIX (`hc-mix.cu`),
indexer relu-sum (`indexer-score.cu`), HC_MIX (`hyperconn.cu`), sigmoid-mul-add (`unary.cu`).  See
`../PHASE2-STATUS.md` for the per-extraction gates.

### Restore

```bash
cd ~/llama.cpp            # or any llama.cpp clone containing 097cf33d9
git fetch /path/to/v2-phase2-ckpt1.bundle refs/heads/v2-phase2:v2-phase2-restored
git log --oneline -3 v2-phase2-restored
# per-block build gate:
~/llama-cpp-rdna-boosts/scripts/gate-per-block-build.sh --repo ~/llama.cpp \
    --base 097cf33d9 --tip v2-phase2-restored --build-dir build-per-block
```

Do **not** apply this bundle to the delivery `patches/`; it is campaign WIP (see `AGENTS.md`).
