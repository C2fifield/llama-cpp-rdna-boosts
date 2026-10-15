# `block09-cache-ring-arming` — host-expert decode + `-lm none` load + first-prefill ring (block 09)

**Status: OPEN (seeded 2026-10-15).** Nothing here is part of the delivery; nothing may be applied to the
fork without the maintainer's explicit go-ahead (see `AGENTS.md`).  The diff in this directory is the
current uncommitted working-tree state of `~/llama.cpp` (branch `rdna-boosts`, r66) and **already fixes
two of the three shipped bugs**; the remaining work is the wide-view / ring-arming fix described in §4.

> **Cold start.** Read `AGENTS.md` (report-before-shipping, WIP/promotion rules, the `-ncmoe` / expert-cache
> invariants under "Critical facts"), then `archive/work/moe-upload-overlap/` (the campaign that introduced
> the gate) and `archive/work/slab-ring-region/` (the ring design).  Then read this file top to bottom.

## 0. TL;DR

Three r63..r66 regressions on host-resident MoE (`-ncmoe`/`-cmoe`) + `-sm tensor` were root-caused on
gfx1201 (2× R9700), reproduced, and two are fixed in the working tree:

| # | symptom | root cause | state |
|---|---|---|---|
| 1 | host-expert decode collapses (GSQ `-ncmoe 48` **14.8 t/s** vs 62 expected) | r63's `moe_cache_tables_complete_locked()` compares each device's registered bytes against an **even** `bytes/n_dev` preflight guess; the real split is 256/384 → the gate never latches → the arena is never sized → the cache never serves | **FIXED** (§2.1) |
| 2 | IQ4_NL `-lm none -sm tensor` intermittently dies at load (`SDMA0` page fault on a pageable host heap address) | r63 makes the loader's `upload_backend` return `nullptr` for a Meta device → the loader reads each tensor into a pageable `std::vector` → `ggml_backend_cuda_buffer_set_tensor` skips its pageable bounce for `USAGE_WEIGHTS` → a direct `cudaMemcpyAsync` from pageable memory (issue #137 class) | **FIXED** (§2.2) |
| 3 | IQ4_NL 70K prefill "slab pointer adjustment failed / straight streaming" (`ring=0`) | the ring region is carved only once the **narrow (decode/verify) floor** is pinned, and that pin lived only inside the `drop_now` block (the prefill→decode transition) → the **first** prefill runs with `ring_region == 0` | **PARTIAL** — the region is now carved for the first prefill and the field config went **483 → 607 t/s (+25.6 %)**; the ring is still **not armed** for the 35B/GSQ `-ub 2048` configs (§4) |

DoD matrix, artifacts and the exact test harness are in §3/§5.  The open problem is §4.

## 1. Artifact

* **Diff (base r66 tree `1d8346485`):** [`block09-cache-ring-arming-r66.patch`](block09-cache-ring-arming-r66.patch)
  sha256 `44ea933d538c4f5572cf5a6829dd3b8305a2e8cfaddd031be8005512dcf0aba1`, 138 lines, 4 files:
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/moe-expert-cache.cu`,
  `src/llama-context.cpp`, `src/llama-context.h`.
* `~/llama.cpp` is on branch `rdna-boosts` at `1d8346485` (r66 tree) with the diff uncommitted; build
  dir `build-rocm` (fixed) and `build-rocm-hybrid` (r65 reference).
* Apply to a clean r66 tree with `git apply block09-cache-ring-arming-r66.patch`.
* Test harness + logs: `tools/`, `logs/` (this directory).

## 2. The three fixes

### 2.1 Cache completeness (block 09, `moe-expert-cache.cu`)

r63 added `moe_cache_tables_complete_locked()` so the deferred arena sizing waits for the full host table
set (a prefill's non-staged 1-token tail ubatch used to latch the arena on one layer — the "tail decode
collapse").  The gate compares `registered[d]` against `g_preflight_device_bytes[d]`, which
`moe_host_expert_account()` (loader) computes as an even `bytes / n_dev`.  The real meta split is **not**
even: measured on GSQ, `ffn_gate_exps` `expert_bytes` 189440/284160 of `host_bytes` 473600 (256/384 of the
640-wide expert FFN dim, i.e. 40 %/60 %; `GGML_CUDA_CACHE_GEOM_DEBUG=1`).  So `registered[device]` never
reaches the even guess for one device, `g_sized` stays false, `alloc_all_locked()` returns on every pass,
the arena is never sized and the cache-aware fusions stand down — every step is served from the pinned
host master over PCIe.

**Fix:** compare the **total** registered bytes to the **total** preflight expectation.  The per-device
split is decided by the meta backend at graph-build time and is not predictable by the loader, so the total
is the only honest invariant; a partial pass (one host layer = 6 tables) stays far below it.  The
`g_preflight_device_bytes[]` array stays (still used by the auto floor and the unregisterable subtraction).

Measured decode win (2× R9700, `-sm tensor`, q8_0 KV, `-ub 2048`, no MTP):

| config | r65 | fixed |
|---|---:|---:|
| GSQ `-ncmoe 48` | 14.7 / 14.8 t/s | **57.9 / 62.7 t/s** |
| GSQ `-ncmoe 20` | 24.0 / 24.6 | **57.9 / 62.0** |
| IQ4_NL `-ncmoe 48` (even split → r65 already cached) | 35.6 / 42.1 | 36.1 / 43.0 |

### 2.2 Pageable weight source on the Meta `-lm none` load (block 06/09, `ggml-cuda.cu`)

r63's `llama_model_loader::upload_backend` returns `nullptr` for a Meta device ("read the whole tensor and
write it once"), so `load_all_data` takes the `else` branch: `std::vector<no_init<uint8_t>> read_buf(n_size)`
(pageable heap) + `ggml_backend_tensor_set`.  For a Meta buffer that dispatches to the per-device CUDA
`set_tensor`, whose pageable bounce is skipped for `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
("model weights keep the direct copy"), so the H2D is a bare `cudaMemcpyAsync` from pageable anonymous
memory.  On ROCm the copy engine then reads non-resident pages:

```
amdgpu 0000:09:00.0: [gfxhub] page fault (src_id:0 ring:157 vmid:8 pasid:501)
amdgpu 0000:09:00.0:          Faulty UTCL2 client ID: SDMA0 (0xd)
amdgpu 0000:09:00.0:   in page starting at address 0x0000558c107db000 from client 10   <- process heap
```

Intermittent (cold file / busy host): `-sm layer` loads clean, `-sm tensor` faults; both r65 and r66
(r66's fix was the unrelated `MUL_MAT_ID dup=1` fault).  **Fix:** pageability alone decides — drop the
`usage != WEIGHTS` clause so *any* pageable source bounces; a pinned source (the loader's async ring, a
CUDA_Host input) still copies directly, so both the host-pinned and the bounce modes are correct.

Validated: IQ4_NL `-lm none -sm tensor` **5/5** clean loads (was faulting), and **0 faults** across the
entire §3 matrix.

### 2.3 Early narrow-floor pin (block 09, `llama-context.{h,cpp}`)

The ring region `[narrow_floor, narrow_floor + ring_region)` is carved only when the narrow floor is known,
and `ggml_backend_dev_slab_narrow_floor()` was called **only inside `if (drop_now)`** (the prefill→decode
transition).  So every prefill before the first decode ran with `ring_region == 0` and fell back to a raw
`cudaMalloc` bounded by the slab's 50 % optional-alloc cap.

**Fix:** measure the narrow layout once with the same `split_only` reserve the `drop_now` path uses and pin
the floor at the top of `process_ubatch`, **before the first wide view is placed** (a new
`slab_narrow_pinned` member guards it).

Proof (IQ4_NL **field config**: MTP n7 p-min 0.001, `--cache-ram 16384`, `-c 131072`, `-ub 2048`,
`-lm mmap`, 70K prompt, `GGML_CUDA_WORK_ALLOC_DEBUG=1`):

```
r65   : WORK_ALLOC dev=0 need=2048.0 narrow=0 ret_off=0.0    narrow_floor=0.0    ring=0.0    boundary=1792.0   -> pp 483.12
fixed : WORK_ALLOC dev=0 need=2048.0 narrow=0 ret_off=7936.0 narrow_floor=1792.0 ring=6144.0 boundary=1792.0   -> pp 607.03  (+25.6 %)
```

## 3. Definition-of-done matrix (2× R9700, gfx1201, `-sm tensor`, q8_0 KV, `-ub 2048`, `--cache-ram 0`, no MTP unless stated)

`ring` = the region seen in `WORK_ALLOC` during the first prefill.  `pp` is the 30K prefill (70K for the
IQ4_NL `-ncmoe 48` rows); `tg` is warm/settled.

| model | `-ncmoe` | host mode | build | load | ring | pp t/s | tg t/s |
|---|---|---|---|---|---|---|---|
| 35B-A3B Q8_0 | 20 | pinned | r65 | OK | 4160 | 2060 | 89.7 / 97.9 |
| 35B-A3B Q8_0 | 20 | pinned | fix | OK | 4160 | 1958 | 89.8 / 98.2 |
| 35B-A3B Q8_0 | 20 | pool | fix | OK | 4160 | 1950 | 90.1 / 98.1 |
| GSQ-IQ3_XXS | 20 | pinned | r65 | OK | 6144 | 1067 | 24.0 / 24.6 |
| GSQ-IQ3_XXS | 20 | pinned | fix | OK | 6144 | 1087 | 57.9 / 62.0 |
| GSQ-IQ3_XXS | 20 | pool | fix | OK | 6144 | 1087 | 58.0 / 62.1 |
| GSQ-IQ3_XXS | 48 | pinned | r65 | OK | 6144 | 651 | 14.7 / 14.8 |
| GSQ-IQ3_XXS | 48 | pinned | fix | OK | 6144 | 662 | 57.9 / 62.7 |
| GSQ-IQ3_XXS | 48 | pool | fix | OK | 6144 | 663 | 57.9 / 62.8 |
| IQ4_NL | 48 | pinned | fix | OK | 6144 | 451 (70K) | 38.5 / 45.7 |
| IQ4_NL | 48 | pool | fix | OK | 6144 | 445 (70K) | 13.7 / 31.1 |
| IQ4_NL | 48 | pinned | r65 (`-lm mmap`) | OK | 6144 | 434 (70K) | 35.6 / 42.1 |
| IQ4_NL | 36 | pinned | fix | OK | 6144 | 583 | 42.7 / 50.0 |

Historical references: 35B `-ncmoe 20` `tg500` 87.5 / `pp4096` 4264 / `-ub 2048` `pp` 2487.9
(`WORKLOG.md` 261014.008, 261014.007); GSQ field prefill 992 / decode 61.1
(`archive/work/slab-ring-region/HANDOVER.md` §3); field MTP decode 139.85 (r65 gate, with
`-ot per_layer_token_embd.*=CPU`).

**Host-pool works on all three** (neutral on 35B/GSQ, ~30 % slower decode on the IQ4_NL, as expected).

## 4. THE OPEN PROBLEM — the ring is carved but not armed (§2.3 is incomplete)

For the 35B/GSQ `-ub 2048` configs the ring region is 4160/6144 MiB but the **prefill is classified
`narrow`**, so no wide view is placed and `s.ring_armed` stays false:

```
35B -ncmoe 20 fixed : WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 narrow_floor=1280.0 ring=4160.0 boundary=1792.0
GSQ -ncmoe 20 fixed : WORK_ALLOC dev=0 need=1536.0 MiB narrow=1 ret_off=0.0 narrow_floor=1536.0 ring=6144.0 boundary=1536.0
```

`ring_armed = ring_region > 0 && wide_off > narrow_floor && boundary >= wide_off`
(`ggml-cuda-slab.cu:947`, `:981`); `wide_off = narrow_floor + ring_region` (`:890`).  With
`boundary == narrow_floor` this is false.  The ring's H2D slots are then refused
(`ggml_cuda_slab_ring_slot_capacity()` returns 0 when `!s.ring_armed`, `:1066`) and the prefill streams.

**Why the prefill is narrow.**  The sched's compute buffer for these models is essentially
**size-independent**: `GRAPH_RESERVE n_tokens=2048` → `COMPUTE_ALLOC 1280.0 MiB`, and the early-pin
`split_only` probe (`cparams.n_rs_batch` tokens) also measures 1280 MiB.  So `need <= narrow_floor` holds
for the prefill and it is placed at base 0 (narrow), never above the ring.  The IQ4_NL field config is the
exception: prefill 2048 > narrow 1792 → wide → armed.

**Scale check:** the prefill does grow with `-ub` (35B `-ncmoe 20`: `-ub 2048` 1958 t/s, `-ub 4096`
3016 t/s) — that is the `-ub` lever, not the ring.

**Proposed approaches (pick one, measure):**
1. **Classify by ubatch width, not by size.**  Treat a work request from a prefill ubatch (`n_tokens > 8`,
   or `>= 64` to match the staging gate) as wide even when `need <= narrow_floor`, so it is placed at
   `wide_off = narrow_floor + ring_region` and the ring arms.  The views are transient/serial, so a prefill
   view above the ring and a decode view at base 0 need not coexist (verify with `work_needs`).
2. **Decouple the ring from the wide view.**  Reserve the ring region when `ring_set(arm=true)` is called
   for a wide ubatch, moving the boundary to at least `narrow_floor + ring_region` (an ordinary boundary
   move, evicting the coldest arena tables), instead of requiring a wide *view* to do it.
3. **Make the narrow floor actually narrow.**  If the probe returns the prefill size, fix the probe
   (`cparams.n_rs_batch` may be too large; check `n_rs_seq`/`n_rs_batch` in the init log).  This helps only
   if the true decode compute buffer is smaller than the prefill one.

Whichever: keep the anti-latch property from §2.1 and re-run the §3 matrix + `gate-qwen4exp-quant-coherence.sh`
+ `gate-prefill-logits.sh`.

**Other open questions:**
* 35B `-ncmoe 20` prefill is 1958 t/s (fixed) vs 2060 (r65) and 2487.9 recorded at `-ub 2048`.  Confirm the
  recorded config/prompt before treating it as a regression — the `-ub 4096` number (3016) suggests the
  measurement is `-ub`-sensitive.
* The `gsq-ncmoe48-pinned-fix.log` shows `optional device allocation refused: … (cap 50 %)` — confirm it is
  the FA prealloc hint (benign, the ladder then succeeds) and not the ring.

## 5. Test harness

* `tools/dod.sh <tag> <bin> <model> <ncmoe> <ctx> <hostmode> <prompt> [extra args…]` — starts
  `llama-server` on port **8036**, waits for `listening on`, sends the prompt twice (warm + measured),
  kills, then prints the `WORK_ALLOC` ring values, `prompt eval time` and `eval time`.  Env:
  `DIAG_LOADMODE=mmap|none`, `DIAG_EXTRA_ENV="GGML_CUDA_WORK_ALLOC_DEBUG=1 …"`.
* `tools/iq4nl-field-run.sh <bin> <tag> <prompt>` — the IQ4_NL **field** config (MTP n7 p-min 0.001,
  `--cache-ram 16384`, `-c 131072`, port 8035).
* `tools/prose30k.txt` = `prompts/prose-rdna-boosts.txt` ×6 (~31.5K tokens); the 70K prompt is ×13
  (`/tmp/prompt70k.txt` in the session; regenerate with the same loop).
* Matrix (examples):
  ```
  B=~/llama.cpp/build-rocm/bin/llama-server; B65=~/llama.cpp/build-rocm-hybrid/bin/llama-server
  ./tools/dod.sh fix35b-pin  $B  /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf                  20 65536  pinned tools/prose30k.txt
  ./tools/dod.sh fixgsq-pin   $B  /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf 48 131072 pinned tools/prose30k.txt
  ./tools/dod.sh r65-gsq48    $B65 /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/…00001-of-00002.gguf            48 131072 pinned tools/prose30k.txt
  DIAG_LOADMODE=mmap ./tools/dod.sh fixiq4-70k $B /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf 48 131072 pinned /tmp/prompt70k.txt
  ```
* `--host-experts pool` = the pool mode; use **port 8036** (8037 is the user's `llama-slot-proxy`).

## 6. Gotchas / corrections

* **The PLE never goes to the GPU.**  `src/llama-model-loader.cpp:1302` forces `LLAMA_TENSOR_PER_LAYER_TOKEN_EMBD`
  onto `buft_list_cpu`; the `-ncmoe 36` layout log shows `CPU model buffer size = 27806.97 MiB`.  The earlier
  `failed to allocate ROCm0 buffer of size 10087080576` at `-ncmoe 36` was **stale VRAM from a previously
  killed server** (the 100 GB teardown lags); on a clean retry it loads and runs (ring 6144, pp 583, tg 50).
  Always `pkill -x llama-server; sleep 4-5` between the big-model runs.
* Port **8037** is the user's `llama-slot-proxy` — do not use it for a test server.
* `llama-cli` **must** be run with `--single-turn`; wrap blocking commands in `timeout`.
* Prefill numbers are `-ub`-sensitive; always record `-b`/`-ub`, `-c`, `--cache-ram` and the prompt.

## 7. Logs (this directory's `logs/`)

| file | what |
|---|---|
| `iq4nl-field-70k-r65-WADBG.log` / `-fix-WADBG.log` | the §2.3 proof (`ring=0` → `ring=6144`, `ret_off=0` → `ret_off=7936`) |
| `35b-ncmoe20-pinned-{r65,fix}.log` | the §4 "narrow" `WORK_ALLOC` + the `GRAPH_RESERVE n_tokens=2048 / COMPUTE_ALLOC 1280` |
| `gsq-ncmoe{20,48}-pinned-{r65,fix}.log` | the fix #1 decode win (24→58 / 14.8→62.7) |
| `iq4nl-ncmoe36-layout.log` | the PLE-on-CPU layout (corrects the `-ncmoe 36` story) |
| `iq4nl-ncmoe48-{pinned,pool}-fix.log`, `iq4nl-ncmoe48-70k-pinned-r65.log` | host-mode + 70K prefill A/B |
