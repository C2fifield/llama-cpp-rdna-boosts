# Measurements — 2026-10-21 session (b): baseline re-establishment + MTP determinism

Box: gfx1201, 3x R9700 (32 GiB), ROCm 10.1.0, 16 cores.  Tree `~/llama.cpp` @ `rdna-boosts` `972f3b757`
(r100) **+ this campaign's patch set** (`patch-state/0001..0004b`), `build-rocm`, built 2026-10-21 02:04
(newer than every source).  Reference (prefix) build: `build-r99`.  One job at a time, `-t 8`, GPUs idle.
`bash` helper: `llama-cli <args> > log; scripts/extract-generated.py log` (never a raw terminal slice).

Verification of the tree before any measurement: applying `patch-state/00*.patch` in order to a detached
worktree at `972f3b757` and `diff -r` against `~/llama.cpp` is **empty** — the working tree is the patch
set, byte for byte.  Patch sha256s match `patch-state/README.md` exactly.

---

## 1. Baseline re-establishment (handover §5 item 1) — GREEN

### 1.1 The recorded #76 hash reproduces exactly

`gemma-4-12b-it-Q8_0`, `-ngl 99 -sm layer -ctk q8_0 -ctv q8_0 -n 64 -p "The capital of France is"
--seed 42 --temp 0`, 1 GPU: **`c740be0e131a`** — the value in `README.md` §1.  So the binary in
`build-rocm` is the one that was gated, not a stale or instrumented build.

### 1.2 Non-regression contract (§2.4), patched vs the r99 prefix build, identical commands

`-f prompts/prose-rdna-boosts.txt --ignore-eos --seed 42 --temp 0 -t 8`, one job at a time.
"patched" = `build-rocm` (patch set), "prefix" = `build-r99`.

| model / config | patched | prefix (r99) | verdict |
|---|---|---|---|
| Qwen3.5 4B Q8_0, 3 GPU `-sm tensor`, q8_0 KV, `-n 64` | `be904e86f248` | `be904e86f248` | identical |
| Gemma4 12B Q8_0, 1 GPU, q8_0 KV, `-n 64`, capital prompt | `953f7ca1f689` | `953f7ca1f689` | identical |
| Qwen3.8 27B Q8_0, 1 GPU, q8_0 KV, `-n 128` | `35d6cf76cfcb`, tg 19.6 | `35d6cf76cfcb`, tg 19.6 | identical, flat |
| Qwen3.8 27B Q8_0, 3 GPU `-sm tensor`, q8_0 KV, `-n 128` | `efad2aa9a83e`, tg 33.1 | `efad2aa9a83e`, tg 33.1 | identical, flat |
| Qwen3.6 35B-A3B UD-Q4_K_M, 2 GPU `-sm tensor -ncmoe 40`, `-n 200` | `bbd5235ebafa`, tg 100.2 | `bbd5235ebafa`, tg 100.3 | identical, flat |
| 35B-A3B cache canary (`-v`) | **240 tables / 61440 slots / 18662.0 MiB**, 120/120 per device, 100 % residency | (record) 240 / 61440 / 18662.0 | identical |

### 1.3 Gemma4 26B-A4B Q8_0 "the cache now sizes" cells

`-ncmoe 40`, 2 GPU, `-n 120 --ignore-eos -f prompts/prose-rdna-boosts.txt`.

| arm | recorded (`MEASUREMENTS-2026-10-21.md` §2.3) | this session | text vs r99 prefix |
|---|---|---:|---|
| `-sm layer` | 79.0 t/s, `34475f215443` | **74.9 t/s**, `892e7ec784f3` | **prefix `892e7ec784f3`, 7.8 t/s — identical** |
| `-sm tensor` | 74.7 t/s, `fd8f9380246c` | **72.8 t/s**, `d966d9ea7304` | **prefix `d966d9ea7304`, 5.6 t/s — identical** |
| `-sm layer -c 16384` (pinned ctx) | — | 74.9 t/s, same text | — |

Cache report for `-sm layer -c 16384` (`-v`): **60 tables, 7680 slots, arena 23141.2 MiB** (the full
expert set), `takeover=0 decline_all=0 get_ok=299 cold=uva cold_reaches=0`, `h=0.0000 (0/0 reaches)`.
So the decode band runs the **identity fast path / get_ok path**, not per-token lookups.

Two discrepancies against the record, both explained, neither a regression:

1. **The recorded text hashes do not reproduce** (`34475f215443` / `fd8f9380246c`) because the exact
   prompt + length of those runs is not recorded anywhere.  With my command the patched build is
   text-identical to the r99 prefix build, which is the actual non-regression test.
2. **74.9 vs 79.0 t/s (-5 %) and 72.8 vs 74.7 (-2.5 %)**.  Same binary, same text, same cache
   configuration; not reproducible as a code difference (pinning `-c 16384` does not move it either).
   Recorded as measurement-condition variance; a future evidence file should name the exact command.

### 1.4 Prefill-logit KLD gate (issue #113) — all three arms PASS

| arm | base | result |
|---|---|---|
| Qwen3.8 27B Q8_0, `scripts/gate-prefill-logits.sh` | recorded Oct 8 base | **0.000707 / 98.755 %** — identical to the r100 record |
| Gemma4 12B, f16 KV (base recorded from `build-r99`) | r99 | **-0.000000 / 100.000 %** — identical to the record |
| Gemma4 12B, `-ctk q8_0 -ctv q8_0` (base from `build-r99`) | r99 | **-0.000000 / 99.990 %** — identical to the record |

Bases written: `~/.cache/rdna-boosts/gates/prefill-logit-gfx1201-gemma4-12b-f16kv.kld` and
`…-gemma4-12b-q8kv.kld` (both recorded from `build-r99`, meta files present).

---

## 2. MTP determinism (handover §2, maintainer decision §0.2)

### 2.1 Reproduction (and the handover's controls do **not** hold)

Base config: Gemma4 26B-A4B Q8_0 + `mtp-gemma-4-26B-A4B-it.gguf`, `-ngl 99 -ncmoe 40 -sm layer
--fit false -c 16384 --spec-type draft-mtp --spec-draft-n-max 3 -n 300 -t 8 --seed 42 --temp 0
--ignore-eos -f prompts/prose-rdna-boosts.txt`, 2 GPUs.  MTP is active: 89.3 t/s vs plain 75.0 t/s,
acceptance 68/90, mean accepted length 3.27.

Six fresh processes, default env:

```
8db9697b1563   8db9697b1563   396aacc43d93   8db9697b1563   396aacc43d93   396aacc43d93
```

A third text, `48660b77bcac`, appears in some arms below.  The first differing **token** (from the
`llama-server` chunk deltas, `-v`) is token **82 of 300** — run A emits `" fused"`, run B emits `" Mo"`
after `…WMMA flash-attention,`; the texts share 771 characters, so it is one flipped greedy near-tie,
not two different generations.

| control | handover claim | this session (runs) | verdict |
|---|---|---|---|
| `GGML_CUDA_DISABLE_GRAPHS=1` | deterministic, == plain | `24b13476117d` ×4 (== plain) | **confirmed** |
| `-ncmoe 0` | deterministic | `24b13476117d` ×3 | **confirmed** |
| `GGML_CUDA_ASYNC_INPUT=0` | deterministic | `396aa` ×3, `8db9` ×1, `4866` ×2 | **REFUTED** (2-run coincidence in the handover) |
| `MOE_EXPERT_CACHE_MIB=0` | deterministic | `b9cf347fe33f` ×2 (CPU MoE, different kernel) | confirmed, but it changes the op path |
| `--spec-draft-n-max 2` | deterministic | `396aa`, `8db9`, `8db9` | **REFUTED** |
| `--spec-draft-n-max 1` | — | `8db9`, `396aa`, `396aa` | nondeterministic |

Arm matrix run to isolate the trigger (2 runs unless noted):

| arm | results | verdict |
|---|---|---|
| **1 GPU** (same config) | `809d2d4f7a64` ×3 | **deterministic** |
| 3 GPU | `24b13476117d` ×2 (== plain) | deterministic in 2; MTP gives no speedup there (73.9/74.0 vs plain 75.0), so it is not a clean control |
| 2 GPU, `-fa off` | `e4272a465ec8` ×2 | deterministic (different kernel set) |
| 2 GPU, draft on CPU (`-ngld 0`) | `396aa` ×2 | candidate-deterministic (2 runs only) |
| Gemma4 **12B dense** + its MTP head, 2 GPU `-sm layer` | `a55588d7a26e` ×3 | **deterministic** (no host experts, no cache) |
| `GGML_CUDA_ASYNC_INPUT_BATCH=0` | `396aa`, `4866`, `4866` | nondeterministic |
| `GGML_CUDA_ASYNC_INPUT_LAZY_DONE=0` | `396aa`, `396aa`, `8db9` | nondeterministic |
| `GGML_CUDA_GRAPH_MEM_GEN=0` | `8db9`, `396aa`, `396aa` | nondeterministic |
| `GGML_CUDA_SLAB=0` | `396aa`, `4866`, `4866` | nondeterministic → **not the slab** |
| `GGML_SCHED_EVENTS=0` | `809d`, `809d`, `688a` | nondeterministic |
| `GGML_META_SS_VERIFY=1` | `8db9`, `396aa`, no verify abort/warning | split-state cache **clean** |
| `MOE_EXPERT_CACHE_VALIDATE=1` | `396aa`, `8db9`, no validate warning | cache invariants **clean** |

Note: 35B-A3B was **not** a valid MTP control — there is no MTP-head GGUF for it under `/llm/models/`,
so `--spec-type draft-mtp` without `-md` simply ran plain (`f669a0cb153e` ×3).

Incidental (unrelated) abort found while probing the batch band: `--spec-draft-n-max 3 -ub 1 -b 1`
aborts in `common/common.cpp:2467 get_sub_batch: failed to add token 0 at index 1 (error -1, n = 2)`.
Pre-existing, not part of this item; worth its own TODO entry.

### 2.2 The defect is the **identity fast path** (`slots == n_experts`), not the cache in general

| arm | runs | text |
|---|---|---|
| `MOE_EXPERT_CACHE_SLOTS=128` (explicit, identity ON) | `396aa`, `809d`, `396aa` | nondeterministic |
| `MOE_EXPERT_CACHE_SLOTS=64` (slots < n_experts → **remap path**) | `24b13476117d` ×3 | **deterministic and exactly == plain** |
| plain decode, identity | `24b13476117d` | — |
| plain decode, `SLOTS=64` (remap) | `24b13476117d` ×2 | **identical** → the identity path is exact for the 1-token decode band |

So: single-token decode through the identity arena is correct; the **multi-token verify batch**
(`ne[2] = n_max+1`) through the identity arena is wrong *and* run-to-run unstable; the remap path
(which reads a cache-owned remapped-ids buffer filled from a **host round-trip of the ids**, behind a
`ggml_backend_synchronize`) is exact.

### 2.3 What the identity path removes (code facts, read this session)

`ggml/src/ggml-backend.cpp` ~2802: the takeover fast path returns **before** `wait_before_overwrite()`
and before the ids readback.  Under a meta split that wait is a **full host synchronize of both
devices per expert input** ("the meta backend has no event_record/event_wait" — the code's own
comment), and removing it is worth the documented +25 %.  The identity consumer
(`ggml-cuda.cu` ~3668 / `moe_cache_redirect_fused` ~4922) then reads the graph's **live routing ids
tensor** with the arena as `src0`; the remap consumer instead reads `t.remap_dev`, written on the
compute stream.

### 2.4 Decisive A/B: restoring that ordering makes it deterministic

Temporary diagnostic (since reverted; the tree is byte-identical to `patch-state/` again): an
env-gated `ggml_backend_synchronize(split_backend)` inserted in the takeover branch of
`ggml-backend.cpp`, i.e. exactly the wait the fast path skips.  One binary, both arms:

| arm (same binary) | runs | text |
|---|---|---|
| `GGML_MOE_TAKEOVER_SYNC=1` | `688ae096023e` ×3 | **deterministic** |
| default (sync skipped) | `396aa`, `8db9` | nondeterministic |

So the text is a function of *whether that ordering is in place*, which means a data dependency that
the serialization was covering is unprotected in the fast path.  Note the three outcomes are three
*different* texts (`24b1` graphs-off/plain, `688a` serialized, `396aa`/`8db9` fast path), so this is a
correctness defect in the identity path, not merely a run-to-run nuisance.

### 2.5 Consequence for the campaign

The identity fast path is what makes the campaign's win (`7.9 → 79.0 t/s`); the remap path is
deterministic *and* plain-exact but slower.  A fix therefore cannot be "turn identity off" (that is
also not the shallow-`llama-bench` answer the maintainer wants): it must restore the ordering for the
multi-token identity consumer at a cost far below the per-op two-device host synchronize.
