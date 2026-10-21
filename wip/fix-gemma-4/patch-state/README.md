# Patch state — `fix-gemma-4`, current tree (2026-10-21, session b)

Seven patches, cut against the delivery tree **`972f3b757`** (`v12-097cf33d9-r100`, net tree
`4775fc92b9007b279062ad43a63150a8a2ebe7db`).  They apply **in order** with `git apply` to a clean checkout
at that commit and reproduce the current `~/llama.cpp` working tree **byte-for-byte** (verified: applied to
a detached worktree at `972f3b757`, then `diff -r` against `~/llama.cpp` — empty).

**Not part of the delivery.**  Nothing here is folded into `patches/`; that is the maintainer's call (WIP
and promotion rules in `AGENTS.md`).  See `../HANDOVER-2026-10-21b.md` §3 for the fold plan.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 097cf33d9
bash <delivery>/scripts/apply-all.sh .            # the 12-block release tree (tip 972f3b757)
for p in <this-dir>/000*.patch; do git apply "$p"; done
~/bin/build-llama-rocm                            # or: cmake --build build-rocm --target llama-cli llama-bench
```

| patch | owner block | what |
|---|---|---|
| `0001-iswa-kv-builder-expand-order.patch` | 10 (rope+FWHT fusion) | **TODO #76**: the iSWA KV builder expands `q, v, k` instead of `q, k, v`, so the r81 q8_0 K/V cache-write fusion fires on iSWA archs. 5 lines in `src/llama-graph.cpp`, no env, no kernel change. |
| `0002-host-expert-account-banks-only.patch` | 09 (host-resident MoE) | **TODO #77/A**: `moe_host_expert_account()` only counts routed expert banks (`ne[2] >= 2`), so per-expert side tensors such as gemma4's `ffn_down_exps.scale` (512 B/layer) no longer make the cache's completeness check unsatisfiable. |
| `0003-expert-cache-sizing-and-multi-slice-tables.patch` | 09 | **TODO #77/A + C**: the completeness slack (and why it must stay ≪ one table), plus multi-slice cache tables — a table's device slot may be assembled from up to 8 ordered host ranges, which is what makes gemma4's fused `ffn_gate_up_exps` cacheable under `-sm tensor`.  One shared `fill_slot_from_host()` now serves the eager fill and the identity fill.  Backend iface: `moe_cache_update` takes an optional `ggml_backend_moe_slice` list. |
| `0004a-meta-async-segmented-upload.patch` | 09 | **TODO #77/B**: a row-aligned segmented `set_tensor_async` splice (per-device `set_async`/`set_2d_async` + the MMQ tail pad).  Mid-row ranges keep the synchronous setter.  Kill switch `GGML_META_SEG_ASYNC=0`. |
| `0004b-scheduler-cache-iface-callsite.patch` | 09 | The scheduler call site for the extended `moe_cache_update` iface (passes `nullptr, 0` for the slice list). |
| **`0005-async-input-ring-per-origin.patch`** | 09 | **MTP determinism (in progress)**: the async graph-input ring's ordering state (`pending`/`ev`/`q`/`flush_ev`/`done_ev`) was per **device**, but a stream is a `llama_context` — MTP's draft and target consumed each other's queued uploads, so a context could replay a graph without waiting for its own copy.  Keyed per **origin** (the calling thread's stream).  Measured: the 2-device MTP draft-logit fingerprint went from *unstable in every sample* to 8/10 identical.  **Does not close the item** — see `../HANDOVER-2026-10-21b.md` §4. |
| **`0006-spec-verify-batch-vs-nbatch.patch`** | — (`common/`) | **New this session**: a speculative verify batch carries `n_max + 1` tokens in one `common_batch::get()`, capped by `llama_n_batch()`; `-b 1` therefore aborted in a server worker (`get_sub_batch: failed to add token 0 at index 1 (error -1, n = 2)`) after the model had loaded.  `common_init_from_params` raises `n_batch` to `n_max + 1` with a WRN (visible at `-lv 2`), same shape as the existing `--spec-draft-n-max` clamp. |

sha256:

```
58cc5c95d4d59769f90d08f540f4ac01c273296d849ee9cd94db599b3806f19c  0001-iswa-kv-builder-expand-order.patch
120050aefeea1b426d7a73f9c216f76d697fc8af261f9b622fe962cd32a6d52a  0002-host-expert-account-banks-only.patch
a82339e00f6bc070d5fe72f11d60b156e5d4f8b5fa304dec51b5c37f98c7540d  0003-expert-cache-sizing-and-multi-slice-tables.patch
7452103d22f804a6087461b9b434acd2e236f77e2bb482ee69282bed57e68c9a  0004a-meta-async-segmented-upload.patch
c41c18097e79337541403c24c2d4e393b61406939f21bc31848a45eb51a43f3f  0004b-scheduler-cache-iface-callsite.patch
3eb50c5d0040d82510b82a809c6bf4d8be5b239832313aff8124c059c6819556  0005-async-input-ring-per-origin.patch
37528713c745f81d204c287b3009712b6f939aa856221e3ef37c71fd833f0440  0006-spec-verify-batch-vs-nbatch.patch
```

Files touched by the whole set: `src/llama-graph.cpp`, `src/llama-model-loader.cpp`, `common/common.cpp`,
`ggml/src/ggml-backend.cpp`, `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend-meta.cpp`,
`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.{cu,cuh}`,
`ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`.

## Rebuild-from-scratch verification (do this before trusting the set)

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/v && cd /tmp/v && git checkout 097cf33d9
bash <delivery>/scripts/apply-all.sh .
for p in <delivery>/wip/fix-gemma-4/patch-state/000*.patch; do git apply "$p" || echo "FAILED $p"; done
diff -r --exclude=.git --exclude='build*' /tmp/v ~/llama.cpp | grep -v '^Only in /home/stew675/llama.cpp'
# expect: no output
```

## Verification status

Gated and green: gemma4 text identity vs the prefix build in both split modes, the 35B-A3B + 27B dense + 4B
non-regression cells (text and throughput), the 35B-A3B cache canary (240 tables / 61440 slots /
18662.0 MiB), `test-llama-archs` 375/375, both prefill-logit gates (27B 0.000707 / 98.755 %, gemma4 f16
-0.000000 / 100.000 %, gemma4 q8_0 rotated -0.000000 / 99.990 %), and the newly instrumented determinism
work in `../HANDOVER-2026-10-21b.md` §2.

Not done (deliberately): the full 4-axis adaptive-MTP gate at `-n 3000` on 35B-A3B / 27B — run it before
this patch set is promoted.
