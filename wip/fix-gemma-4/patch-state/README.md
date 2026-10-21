# Patch state — `fix-gemma-4`, session 2026-10-21

Five patches, cut against the delivery tree **`972f3b757`** (`v12-097cf33d9-r100`, net tree
`4775fc92b9007b279062ad43a63150a8a2ebe7db`).  They apply **in order** with `git apply` to a clean
checkout at that commit, and reproduce the session's working tree byte-for-byte (verified: applied to a
detached worktree at `972f3b757`, then `diff`-ed against `~/llama.cpp`).

**Not part of the delivery.**  Nothing here is folded into `patches/`; that is the maintainer's call
(WIP/promotion rules in `AGENTS.md`).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 097cf33d9
bash <delivery>/scripts/apply-all.sh .            # the 12-block release tree (tip 972f3b757)
for p in <this-dir>/00*.patch; do git apply "$p"; done
~/bin/build-llama-rocm                            # or: cmake --build build-rocm --target llama-cli llama-bench
```

| patch | owner block | what |
|---|---|---|
| `0001-iswa-kv-builder-expand-order.patch` | 10 (rope+FWHT fusion) | **TODO #76**: the iSWA KV builder expands `q, v, k` instead of `q, k, v`, so the r81 q8_0 K/V cache-write fusion fires on iSWA archs. 5 lines in `src/llama-graph.cpp`, no env, no kernel change. |
| `0002-host-expert-account-banks-only.patch` | 09 (host-resident MoE) | **TODO #77/A**: `moe_host_expert_account()` only counts routed expert banks (`ne[2] >= 2`), so per-expert side tensors such as gemma4's `ffn_down_exps.scale` (512 B/layer) no longer make the cache's completeness check unsatisfiable. |
| `0003-expert-cache-sizing-and-multi-slice-tables.patch` | 09 | **TODO #77/A + C**: the completeness slack (and why it must stay ≪ one table), plus multi-slice cache tables — a table's device slot may be assembled from up to 8 ordered host ranges, which is what makes gemma4's fused `ffn_gate_up_exps` cacheable under `-sm tensor`.  One shared `fill_slot_from_host()` now serves the eager fill and the identity fill, so the two cannot disagree.  Backend iface: `moe_cache_update` takes an optional `ggml_backend_moe_slice` list. |
| `0004a-meta-async-segmented-upload.patch` | 09 | **TODO #77/B**: a row-aligned segmented `set_tensor_async` splice (per-device `set_async`/`set_2d_async` + the MMQ tail pad), so the scheduler's per-op expert upload no longer blocks the host.  Mid-row ranges (the loader's chunked upload) keep the synchronous setter.  Kill switch `GGML_META_SEG_ASYNC=0`. |
| `0004b-scheduler-cache-iface-callsite.patch` | 09 | The scheduler call site for the extended `moe_cache_update` iface (passes `nullptr, 0` for the slice list). |

sha256:

```
58cc5c95d4d59769f90d08f540f4ac01c273296d849ee9cd94db599b3806f19c  0001-iswa-kv-builder-expand-order.patch
120050aefeea1b426d7a73f9c216f76d697fc8af261f9b622fe962cd32a6d52a  0002-host-expert-account-banks-only.patch
a82339e00f6bc070d5fe72f11d60b156e5d4f8b5fa304dec51b5c37f98c7540d  0003-expert-cache-sizing-and-multi-slice-tables.patch
7452103d22f804a6087461b9b434acd2e236f77e2bb482ee69282bed57e68c9a  0004a-meta-async-segmented-upload.patch
c41c18097e79337541403c24c2d4e393b61406939f21bc31848a45eb51a43f3f  0004b-scheduler-cache-iface-callsite.patch
```

Files touched: `src/llama-graph.cpp`, `src/llama-model-loader.cpp`, `ggml/src/ggml-backend.cpp`,
`ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend-meta.cpp`,
`ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.{cu,cuh}`, `ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`.

## Verification status (see `../evidence/MEASUREMENTS-2026-10-21.md`)

Gated and green against the `README.md` §2.4 contract: gemma4 text identity vs the prefix build in both
split modes, 35B-A3B + 27B dense + 4B non-regression (text and throughput), `test-llama-archs` 375/375,
`test-backend-ops` (MUL_MAT_ID/SET_ROWS/UPSET_ROWS/RMS_NORM/CPY), both prefill-logit gates, no GPU faults.

Not done (deliberately): the full 4-axis adaptive-MTP gate at `-n 3000` on the delivery's stock models
(35B-A3B / 27B) — the MTP numbers in the evidence file are gemma4-only, and gemma4 MTP is `-sm layer`-only
until TODO #68 is fixed.  Run the 4-axis gate before this patch set is promoted.
