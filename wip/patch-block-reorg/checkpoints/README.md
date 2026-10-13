# patch-block-reorg Phase-2 checkpoints

Exact, restorable snapshots of the Phase-2 `v2-phase2` chain.  A checkpoint is a git bundle of the
range `<release.json.base>..v2-phase2`; it needs `097cf33d9` present in the target repo (it is the
published upstream base, so any llama.cpp clone has it).

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
