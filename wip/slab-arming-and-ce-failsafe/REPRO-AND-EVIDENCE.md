# `slab-arming-and-ce-failsafe` — reproduction and evidence

All commands below were run on the maintainer's box: 2× AMD Radeon AI PRO R9700 (gfx1201, 34 GiB),
ROCm 7.14.1, build `build-rocm` (reports `b11392-b7837d027`, i.e. the r46 tree).  Model:
`/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (MoE, so `-ncmoe` produces host
experts).  Reproduce the minimal case with any host-expert MoE model.

## 1. The 4-way flag matrix (the trigger is `--fit off`)

Common flags: `-ngl 99 -sm tensor -ncmoe 40 -p "hi" -n 1 --temp 0 --no-display-prompt --single-turn -t 8
-c 8192 -v`.  Only the trailing flags differ.

| run | trailing flags | `auto cache preflight` | `SLAB_INIT dev=0/1` | arena |
|---|---|---|---|---|
| 1 | (none) | yes | **yes** | slab-backed |
| 2 | `--fit off` | no | **no** | raw `cudaMalloc` |
| 3 | `--load-mode none` | yes | **yes** | slab-backed |
| 4 | `--fit off --load-mode none` | no | **no** | raw `cudaMalloc` |

So `--fit off` is the sole trigger; `--load-mode` is irrelevant.

## 2. Exact commands

Run 2 (the broken case):

```bash
cd ~/llama.cpp
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
timeout 300 ./build-rocm/bin/llama-cli -v \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -sm tensor -ncmoe 40 -p "hi" -n 1 --temp 0 \
  --no-display-prompt --single-turn -t 8 -c 8192 \
  --fit off > /tmp/fit-off.out 2> /tmp/fit-off.err
grep -aE "SLAB_INIT|slab [0-9.]+ GiB|MoE expert cache \(auto\)" /tmp/fit-off.err
# -> only:  MoE expert cache (auto): arena ...
```

Run 1 (the healthy case): the same command without `--fit off`:

```bash
grep -aE "SLAB_INIT|slab [0-9.]+ GiB|MoE expert cache \(auto\)" /tmp/fit-on.err
```

## 3. gdb proof of the ordering bug

The preflight entry point is `ggml_backend_cuda_device_moe_cache_preflight`.  Break there and read the
globals.  The statics have symbols in the built library:

```
$ nm build-rocm/bin/libggml-hip.so.0 | grep -E "g_slab_armed|g_enabled|g_auto"
00000000121a94c8 b _ZL12g_slab_armed
00000000121ac138 b _ZN12_GLOBAL__N_16g_autoE
00000000121ac120 b _ZN12_GLOBAL__N_19g_enabledE
```

`--fit off`:

```bash
gdb -q -batch -ex 'set breakpoint pending on' \
  -ex 'break ggml_backend_cuda_device_moe_cache_preflight' -ex 'run' \
  -ex "print *(char*)&'_ZL12g_slab_armed'" \
  -ex "print *(char*)&'_ZN12_GLOBAL__N_19g_enabledE'" \
  -ex "print *(char*)&'_ZN12_GLOBAL__N_16g_autoE'" \
  --args ./build-rocm/bin/llama-cli -m <moe>.gguf -ngl 99 -sm tensor -ncmoe 40 \
        -p hi -n 1 --no-display-prompt --single-turn -t 8 -c 4096 --fit off
```

Output:

```
Thread 9 "llama-cli" hit Breakpoint 1 ... in ggml_backend_cuda_device_moe_cache_preflight ...
$1 = 0 '\000'      # g_slab_armed
$2 = 0 '\000'      # g_enabled   <-- parse_env() has NOT run yet
$3 = 0 '\000'      # g_auto
```

`--fit on`, same breakpoint:

```
$1 = 0 '\000'      # g_slab_armed
$2 = 1 '\001'      # g_enabled   <-- parse_env() HAS run
$3 = 1 '\001'      # g_auto
```

`moe_cache_preflight` opens with `if (!g_enabled) return false;`, so on the `--fit off` run it returns
false, `g_slab_armed` is never set, and `ggml_cuda_slab_enabled()` (= `env && g_slab_armed`) is false for
the entire run.

The gdb backtrace at the breakpoint is always:

```
#0  ggml_backend_cuda_device_moe_cache_preflight (...)
#1  llama_model_moe_cache_preflight ()
#2  common_init_result::common_init_result(common_params&, bool)
#3  common_init_from_params(common_params&, bool)
```

i.e. the driver preflight at `common/common.cpp:1352`, which runs before `llama_init_from_model()`.

## 4. Healthy reference (what the fixed build must produce)

`--fit on`, no draft (from `/tmp/diag2.err`; the `aux=0.00 GiB` because there is no MTP draft in
`llama-cli`):

```
moe_cache_init: MoE expert cache enabled on device 0: budget=auto MiB/device, period=32 steps, ...
moe_cache_preflight: auto cache preflight: device 0 projected arena 30188.0 MiB ...
W SLAB_INIT dev=0 free=30.32 GiB reserve=4.00 GiB headroom=4.00 GiB aux=0.00 GiB size(VA)=26.31 GiB mapped=26.31 GiB boundary=0.75 GiB va_spare=0.00 GiB chunk=64.0 MiB ring_req=4.03 GiB
W ggml_cuda_slab_init_locked: device 0: slab 26.31 GiB (chunk 64.0 MiB): work 0.75 GiB + arena 25.56 GiB (4.00 GiB reserve, 0.00 GiB VA spare ...)
W SLAB_INIT dev=1 free=30.32 GiB reserve=4.00 GiB headroom=4.00 GiB aux=0.00 GiB size(VA)=26.31 GiB ...
W ggml_cuda_slab_init_locked: device 1: ...
I cmn          init: llama threadpool init, n_threads = 8
W alloc_all_locked: MoE expert cache (auto): arena 18662.0 MiB of 18662.0 MiB host experts (100.0% residency)
```

On the field server config (MTP draft present) the same lines show `aux=4.00 GiB` and
`reserve=8.00 GiB` (see `/tmp/srr/fe-mtp.err`: `SLAB_INIT dev=0 free=27.52 GiB reserve=8.00 GiB
headroom=4.00 GiB aux=4.00 GiB ...`).

Key ordering: `moe_cache_init` -> `moe_cache_preflight` -> `SLAB_INIT` -> `threadpool init` ->
`alloc_all_locked`.  On the `--fit off` run, only the last is present.

## 5. The field crash this produced (issue #136)

Serveur config: `HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=ce`, `-sm tensor -ncmoe 48`,
`-ub 6144 -c 204800`, q8_0 KV, `--fit off`, `--spec-type draft-mtp --spec-draft-n-max 3`,
Qwen3.8-Flash-Next IQ4_XS.  Repro: generate ~8k tokens, then prefill ~70k.

```
2.31.974.556 W alloc_all_locked: MoE expert cache (auto): arena 44790.4 MiB of 56762.5 MiB host experts (78.9% residency)
...
4.52.908.218 W moe_cache_shrink_step: stood down the largest table (layer=2 role=blk.2.ffn_down_exps.weight, 399.4 MiB) for a compute allocation
4.52.908.221 W ggml_cuda_device_malloc: 60.00 MiB allocation failed on device 1; shrank the MoE arena and retrying
...   (ten futile stand-downs; the alias warning names a TABLE ON DEVICE 0 while the fault is on device 1)
4.55.187.876 W alias_find_checked: ignoring a stale MoE-cache alias (table device 0 layer 9, op device 0 layer 6); 1 refused so far
4.59.375.436 E ROCm error: out of memory
4.59.375.439 E   current device: 1, in function ggml_backend_cuda_comm_allreduce_ce at ggml-cuda.cu:2833
4.59.375.440 E   hipMalloc(&comm_ctx->ce_tmp2[i], need)
```

There is no `SLAB_INIT` anywhere in the log, yet the cache is on and the arena is large: that is item A.

The `shrank the MoE arena` message is emitted only by the `ggml_cuda_device_malloc` branch that runs
when `ggml_cuda_slab_active(device)` is false (`ggml-cuda.cu:216`).  With the slab armed, the same failure
instead prints `... the arena cannot yield physical back to the driver.  Raise GGML_CUDA_SLAB_RESERVE_MIB
...` and returns (`:198-204`).  The stand-downs were also device-blind: `moe_cache_shrink_step()` picks
the largest table on **any** device (item D), which is why the alias warning names device 0 while the
faulting allocation and the abort are on device 1.

## 6. Byte-level verification that the arena is raw under `--fit off`

`alloc_table_locked` selects the arena with `use_slab = ggml_cuda_slab_enabled()`
(`moe-expert-cache.cu:1665`).  In the build, `ggml_cuda_slab_enabled` is called immediately before
`ggml_cuda_slab_arena_alloc` in the arena path, and `ggml_cuda_slab_arena_alloc` returns `nullptr` when
`!s.inited` (`ggml-cuda.cu:1424`).  Since `use_slab` is global, a slab-less run takes the raw `cudaMalloc`
arm and, if the slab were merely declined, every table would instead fail with `arena alloc failed for
layer=... cache disabled for this table`.  The observed run has neither an `arena alloc failed` line nor
a `SLAB_INIT`, and does have a 44790 MiB arena: the raw arm.

## 7. Notes for the fixing session

* The `--fit off` path is the one the maintainer's field server uses.  Fix item A first and re-run the
  §5 repro end to end before touching B/C/D.
* The gdb break trick needs `set breakpoint pending on` because both libraries are loaded after start;
  and the anonymous-namespace statics must be printed by their mangled names
  (`_ZN12_GLOBAL__N_1…`) or by address.
* If `parse_env()` is made eager (fix A), re-run the 4-way matrix: all four rows must show `SLAB_INIT`.
