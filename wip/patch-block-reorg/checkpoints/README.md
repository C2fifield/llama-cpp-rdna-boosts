# patch-block-reorg Phase-2 checkpoints

Exact, restorable snapshots of the Phase-2 `v2-phase2` chain.  A checkpoint is a git bundle of the
range `<release.json.base>..v2-phase2`; it needs `097cf33d9` present in the target repo (it is the
published upstream base, so any llama.cpp clone has it).

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
