# `--load-mode none` + `-sm tensor` startup GPU fault → meta async pinned upload

**Status: PROMOTED (2026-10-17, `v16-a55e952b8-r48`, folded into block 06, issue #137).**  The campaign
is closed; the change is in `patches/0006-rdna-boosts-block-06-general-system-operations-bucke.patch`.
The historical `TODO #38` (host-resident expert load page fault under `--load-mode none`) is closed by
this release.

## The bug

On the field config (2x R9700 gfx1201, Qwen3.8-Flash-Next IQ4_NL, `-sm tensor -ncmoe 48
--load-mode none --lazy-mode off --fit off`, MTP n3), the model load died ~1 run in 10 with
`Memory access fault ... Page not present or supervisor privilege`.

The kernel journal is authoritative (and names the client):

```
amdgpu 0000:09:00.0: [gfxhub] page fault (src_id:0 ring:157 ...)
  Faulty UTCL2 client ID: SDMA0 (0xd)        <- the copy engine, not a shader
  RW: 0x0  PERMISSION_FAULTS: 0x3  MAPPING_ERROR: 0x1  WALKER_ERROR: 0x5
  in page starting at address 0x0000558b447aa000 ...    (+0x1000 page walk)
```

An **SDMA DMA read of pageable anonymous host memory** faulted.  Under `-sm tensor` the device weights
live in the **Meta** buffer, and the loader's pinned, event-ordered async-upload pipeline
(`upload_backend`) was disabled because the Meta device reported placeholder capabilities:

```
D load_all_data: device Meta() does not support async, host buffers or events
```

`ggml_backend_meta_device_get_props` hard-coded `host_buffer = false` / `events = false` and ANDed them
with the simple devices, so the result was always false (`// TODO replace placeholders`).  The loader
fell back to a pageable `std::vector<no_init<uint8_t>> read_buf` and issued a raw async H2D from it.
`--load-mode auto`/`mmap` sources from the file-backed mapping (stable page cache) and `-sm layer` uses
per-device buffers with a real `upload_backend`, which is why both were stable.

## The fix (block 06)

Adapted from upstream **PR #22466** ("ggml-backend-meta : async pinned upload for `-sm tensor` model
load"; open and mergeable upstream, unmerged since 2026-09-04, and not among the 183 upstream commits
since base `a55e952b8`):

* `ggml-backend-meta.cpp` — the Meta device reports `host_buffer`/`events` (ANDed with the simple
  devices), proxies the device event iface (`event_new`/`free`/`synchronize`) and the backend
  `event_record`/`event_wait` to each simple backend, and `ggml_backend_meta_set_tensor_async` handles the
  loader's chunked ranges (partial head/tail rows) and multi-segment/partial layouts (an accumulator that
  delegates to the synchronous segmented handler).  That subsumes the block-15 `TODO #43` per-device tail
  guard.
* `src/llama-model-loader.cpp` — the delivery-specific wrinkle: because the delivery made `ROCm_Host`
  **per device**, `ggml_backend_dev_host_buffer_type(Meta)` is null, so `upload_backend` falls back to a
  real device's pinned host buft for its staging ring.  (Loosening `get_host_buffer_type` instead was
  tried and rejected: it leaks into `make_cpu_buft_list`, moves the 27.8 GiB PLE into pinned `ROCm_Host`,
  and restores the whole-model 92.6 GiB pin.)

## Validation (gfx1201)

* Loader logs `using async uploads for device Meta(), buffer type Meta(), backend Meta(ROCm0,ROCm1)`;
  buffer layout unchanged (`CPU 27806.97`, `Meta() 1789.79`, `ROCm_Host 64800.00` MiB).
* Prefill-logit gate **PASS** (mean KLD `0.000707`, same-top-p `98.755 %`, identical to the recorded base).
* `test-backend-ops -o MUL_MAT_ID` **931/931**.
* Width purity on the dense 27B Q8_0, 2-GPU `-sm tensor`, `--load-mode mlock`:
  `--spec-type none == draft-mtp --spec-draft-n-max 7` **byte-identical** (`c0f0edeac619`, `-n 1024`,
  `prompts/prose-rdna-boosts.txt`).
* Dense 4B Q8_0 coherence: `-sm tensor == -sm layer == 1 GPU`, byte-identical to the unpatched build
  (`b065f37593b3`).
* Field config: 10/10 loads clean, coherent, `////` = 0 over a 512-token code run.
* `scripts/validate-set.sh` green (strict 16/16 `git am`, reconstructed tree == `release.json`).

Caveat carried forward: the field config's greedy text is not run-to-run deterministic (pre-existing
`TODO.md` #50, the global fusion guard vs per-table slab eviction), so text hashing is not a gate there.
The fault is a ~1-in-10 race that also did not reproduce on the unpatched build across 7 runs, so this
close is evidence-based rather than statistical.  Reopen if it recurs.

## Files

* `meta-async-pinned-upload.patch` — the working-tree diff this campaign produced (delivery r47 tree).
* Upstream PR: https://github.com/ggml-org/llama.cpp/pull/22466
