# UPSTREAM-PR: ggml-backend-meta — resolve unallocated aliases in the split state, and report a failed sub-allocation instead of aborting

**Status:** prepared 2026-10-18; applies clean (`git apply --check`) and **builds** on the fork point
`097cf33d9` (ROCm 10.1.0, gfx1201). **Not filed.**

**Where it lives in the delivery:** **block 00** (upstream structural/correctness fixes). Both hunks
fix code that is upstream's, and neither depends on any fork-local feature. The campaign record,
reproduction and the full validation log are `wip/issue-170-meta-alloc/` (in flight; the *fork-local*
half of that campaign — gating the meta `offload_op` hook to host-resident MoE experts — is **not** in
this patch, because the hook itself is fork-local).

## The two bugs

Both are in `ggml/src/ggml-backend-meta.cpp` and both were found reproducing
[stew675/llama-cpp-rdna-boosts#170](https://github.com/stew675/llama-cpp-rdna-boosts/issues/170)
(`-sm tensor`, hybrid Qwen3.8 27B, a mixed CPU/GPU layer assignment caused by `--fit` reducing
`n_gpu_layers`).

### 1. `ggml_backend_meta_get_split_state()` walks an alias that the allocator never initialised

The split-state calculation recurses into `tensor->src[]`. For a *view*, one of those sources can be an
intermediate alias that is deliberately never visited by `ggml_gallocr_init_tensor()`: a view is
initialised from its `view_src` (`ggml_backend_view_init()`), never from its `src[0]`, and a view op with
`GGML_OP_NONE` is treated as a leaf by `ggml_visit_parents_graph()`. `build_attn_mha()` creates exactly
such a pair:

```c
q = ggml_view_4d(ctx0, q, ...);     /* "Qcur-3 (view)", a node            */
q = ggml_permute(ctx0, q, 0, 2, 1, 3); /* "Qcur-3 (view) (permuted)", alias, src[0] = the view, no buffer ever */
```

The permuted tensor can become a scheduler split-input / alloc-dep node
(`ggml_view_tensor(input)`), and initialising it asks for the split state of the permuted tensor, whose
`src[0]` (the intermediate view) has `buffer == nullptr`. `ggml_backend_meta_buffer_n_bufs(NULL)` then
hits `GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf))` and the process aborts during the warmup
decode:

```
ggml_backend_meta_buffer_n_bufs -> assert
ggml_backend_meta_get_split_state -> ggml_backend_meta_buffer_init_tensor_impl
ggml_gallocr_alloc_graph -> ggml_backend_sched_alloc_graph -> llama_context::process_ubatch
```

Reachable whenever an op with host-resident weights runs on the meta backend. In unpatched upstream
the meta device has no `offload_op` hook, so it is hard to reach there; the invariant violation is
latent, and a debug build of any graph that op-offloads onto Meta trips it.

### 2. A failed constituent allocation aborts instead of returning NULL

`ggml_backend_meta_buffer_type_alloc_buffer_usage()` and `..._alloc_buffer_n()` asserted on a failed
per-device allocation:

```c
GGML_ASSERT(bufs.back() != nullptr);                    /* usage   */
GGML_ASSERT(meta_buf_ctx->bufs[i]);                     /* _n      */
```

so a genuine OOM (KV cache, weights, compute buffer) was a core dump:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12288.00 MiB on device 0: cudaMalloc failed: out of memory
ggml_backend_buft_alloc_buffer_n_default: failed to allocate ROCm0 buffer of size 12884901888
ggml-backend-meta.cpp:2287: GGML_ASSERT(meta_buf_ctx->bufs[i]) failed
```

Every caller already handles `NULL`: `ggml_vbuffer_alloc()` reports "failed to allocate ... buffer" and
fails the reserve, `ggml_backend_alloc_ctx_tensors_from_buft()` returns it, `llama_kv_cache` throws
"failed to allocate buffer for kv cache", and the model loader reports "unable to allocate Meta()
buffer". This also matches the existing ggml-cuda choke point, which already returns NULL after
yielding its arena.

## The fix

| area | change |
|---|---|
| new helper | `ggml_backend_meta_view_chain_buffer()` follows `view_src` while `buffer == nullptr` and returns the buffer of the chain root (the buffer a view inherits). |
| split-state, both entry points | when `tensor->buffer == nullptr`, resolve through that helper: a chain root in a meta buffer is classified normally (the alias is computed from its own op and shape, exactly as when the allocator later assigns it); a host root, another device's buffer, or a chain with nothing allocated is "not split".  **Allocated tensors take the byte-identical old path**, so no graph that works today changes. |
| `..._alloc_buffer_usage()` | on a NULL sub-buffer: log, free the sub-buffers already allocated, return NULL. |
| `..._alloc_buffer_n()` | on a NULL sub-buffer: log, clear the tensors' `buffer`/`data` (they must not point at the meta buffer being freed), free the meta buffer (which releases the sub-buffers allocated so far), return NULL. |

## Validation (gfx1201, 2 x R9700, ROCm 10.1.0, delivery tree)

The reporter's configuration is reproducible on a 32 GiB card by holding 8192 MiB per device so the
card presents a 7900 XTX's free-memory figure (`../wip/issue-170-meta-alloc/vram_fill.cpp`).

* **Before:** abort at `ggml-backend-meta.cpp:711` during the warmup decode with the reporter's minimal
  command; abort at `:2287` for the KV/weight/compute OOM cases.
* **After:** the reporter's minimal command and his full command line (adaptive MTP + `ngram-mod`,
  `--kv-unified`, `--parallel 2`) load, listen and serve a **5246-token** prompt (HTTP 200, coherent,
  no assert); `-ctk/-ctv f32` likewise; `-sm layer` and 1 GPU unaffected.
* **OOM paths now fail cleanly** (exit 1, no core, no VRAM leak), including a *partial* failure
  (`1 of 2 sub-buffers allocated`).
* **Coherence:** same-seed greedy text identical to the pre-change build (`1033 chars
  sha=383655591392`, dense 27B UD-Q4_K_XL, 3 GPUs, `prompts/reasoning.txt`, `-n 128 --seed 42 --temp 0`).
* Hermetic check of this file: applied to a pristine `097cf33d9` worktree and built there (full tree,
  ROCm 10.1.0) with no warnings in the touched file.

## What was NOT validated

* No NVIDIA/MUSA/SYCL/Vulkan hardware: the change is meta-backend-only code, so the other backends are
  untouched by construction.
* No gfx1100 hardware (the reporter's card). The reproduction is a free-memory-parity reproduction on
  gfx1201 under the **same ROCm 10.1.0**, which the reporter has confirmed he runs.
* The fork-local companion (gating the meta `offload_op` hook to `MUL_MAT_ID`) is not part of this PR;
  upstream does not have the hook at all.
