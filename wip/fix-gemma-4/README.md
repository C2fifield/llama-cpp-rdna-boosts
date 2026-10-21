# `fix-gemma-4` — making Gemma4 actually work and perform on RDNA

**Status: OPEN (seeded 2026-10-21).**  Umbrella campaign for every Gemma4 defect on this delivery.  Not
part of the delivery; nothing here may be applied to the fork without the maintainer's go-ahead (WIP and
promotion rules in `AGENTS.md`).

This campaign absorbs three previously-separate items, which are now worked together because they all sit
on the same arch and the same configs:

| item | subject | state |
|---|---|---|
| **TODO #76** | the r81 q8_0 K/V cache-write (rope+FWHT) fusion is inert on iSWA models | **fix validated, complete** (`#76-iswa-write-fusion`) |
| **TODO #77** | the expert cache never sizes on gemma4 → per-op expert upload with no residency (`-sm layer` 7.9 t/s, `-sm tensor` 5.5 t/s) | **A+B+C implemented and gated** — `-sm layer` **79.0** t/s, `-sm tensor` **74.7** t/s (`#77-expert-cache`) |
| **TODO #68** | `gemma4` dense + `-sm tensor` aborts (rotated-K-cache `mul_mat split states`; **and** an f16 `handle_per_row` abort found 2026-10-21) | **1b FIXED** (`0008`): f16 `-sm tensor` runs and is byte-identical to `-sm layer` at 2 and 3 GPU, 26B-A4B `-sm tensor` now measurable; **1a (rotated q8_0 KV) still open — next session** |
| **MTP determinism** | MTP + host experts + `-sm layer` on 2 devices gives different greedy text run-to-run | **CLOSED 2026-10-21 — external**: it is ROCm's hipBLASLt per-start solution pick (issue #67) on the gemma4 F32 MoE router; `ROCBLAS_USE_HIPBLASLT=0` gives 10/10 at `-n 300` and 3/3 at `-n 3000`.  `0005` (per-origin async-input ring) is a real fix that ships.  `evidence/MEASUREMENTS-2026-10-21d.md`, `GREEDY-PURITY.md` §44 |

The two former campaign directories (`wip/iswa-kv-write-fusion/`, `wip/meta-segmented-upload-async/`) are
folded into this one and deleted, with their content preserved here (`ISWA-WRITE-FUSION.md` for the #76
evidence, `#77-expert-cache` sections below for the perf campaign).  No stubs are left behind.

**Box: gfx1201, 3x R9700, ROCm 10.1.0, 16-core.**  Tree: `~/llama.cpp` @ `rdna-boosts` `972f3b757`
(`v12-097cf33d9-r100`), `build-rocm`.  gfx1151 (Strix Halo) numbers from the earlier #76 session are
quoted where they exist.

## 0. Why one campaign

Gemma4 is the only arch in the fleet that combines **iSWA attention with a distinct full-attention head
dim** (256 SWA / 512 full), a **fused gate+up expert table** (`ffn_gate_up_exps`, 2 segments), a
**per-layer `ffn_down_exps.scale` tensor**, and a **rotate-KV-cache** path.  Each of those hits a
different part of the delivery (graph builder, Meta split state, expert cache, cache accounting) and each
failure is hidden by another: the cache defect made every gemma4 perf number look like "the sync upload is
slow", and the tensor-split aborts kept the `-sm tensor` arm unmeasurable.  Working them in one place is
what lets a config be measured honestly end to end.

## 1. What #76 was, and its final gate set (`#76-iswa-write-fusion`)

The r81 rope + FWHT fusion folds the two q8_0 K/V cache `SET_ROWS` launches into the fused rope kernel.
It fires on qwen35/qwen35moe because those use the **non-iSWA** `build_attn`, which expands K last *for
this reason*.  `build_attn(llm_graph_input_attn_kv_iswa*)` expanded `q, k, v`, so the V block sat between
the K rotation and the K cache write and the matcher declined: **inert on every iSWA arch**.

**Fix (5 lines, `src/llama-graph.cpp`):** expand `q, v, k` in the iSWA KV builder, mirroring the
non-iSWA builder (which carries the comment saying so).  No env variable, no kernel change.

**gfx1201 gate set (2026-10-21, this session — the gates the gfx1151 session could not run):**

| gate | result |
|---|---|
| launches, Gemma4 12B Q8_0 `-ctk/-ctv q8_0` `-n 8` | total **13500 → 12540**; `k_set_rows_q8_0_group` **1152 → 192**; `rms_norm_mul_rope_f32<…,kv=true>` **0 → 480**; rotation-only 960 → 480 |
| same, 2 GPU `-sm layer` | identical counts (480 / 480 / 192 / 192) — the fusion fires under a split |
| greedy text (prefix / fix / fix+`GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1`) | 12B n64 `c740be0e131a` in **all** arms at 1, 2 and 3 GPU `-sm layer`; 26B-A4B n120 `34475f215443` both arms |
| `test-logits-width-probe` `W=1..8`, q8_0 **and** f16 | identical hashes fix / kill-switch / r99-prefix build; `width_purity=PASS (maxdiff 0)` |
| prefill-logit KLD (issue #113 gate) | Gemma4 12B f16 **0.000000 / 100.000 %**; Gemma4 12B **q8_0 rotated** **-0.000000 / 99.990 %**; Qwen3.8 27B standard gate **0.000000 / 100.000 %** |
| graph splits / nodes / buffers | **unchanged** at 1, 2 and 3 GPU `-sm layer` (2549 nodes; splits 1/3/4; 2607/2631 realloc; same compute buffer sizes) |
| `llama-bench -p 512 -n 128 -r 3` (3 interleaved rounds) | 1 GPU tg128 **40.50 → 40.76 (+0.6 %)**, pp512 +0.4 %; 2-GPU layer tg128 **37.29 → 37.52 (+0.6 %)**, pp flat |
| `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` | tg B=1/4/8 **+0.1 / +0.2 / +0.4 %** |
| `test-llama-archs -s 42` | **375/375** (== the r99 record, so no iSWA arch regressed) |
| `test-backend-ops` SET_ROWS / ROPE / FLASH_ATTN | OK, 2/2 backends |

**Non-iSWA inertness:** the changed builder is only called by iSWA archs; the q8_0 K/V-write fold needs
NORMAL/NEOX rope + a rotated cache, neither of which a Qwen3.x (`rope_multi`) config has.  The Qwen3.8 27B
prefill gate above is the direct check.

**Known gap (documented, deliberate):** Gemma4's full-attention layers (head_dim 512) stay outside the
fused kernel because its `fwht_n` instantiation set is `{64,128,256}` and the fused kernel's V-row write
assumes `ncols <= block_size (256)`.  Cost: 4 extra launches/layer/step on 8 of 48 layers ≈ -32 launches
per decode step.  Analysis and the 4-step change list are in `ISWA-WRITE-FUSION.md` §5.6.

Patch: `patches/../*.patch` — see `patch-state/` in this directory (regenerated at the end of the session).

## 2. #77 — the expert cache never sizes on Gemma4 (`#77-expert-cache`)

### 2.1 The symptom as filed

`gemma4` MoE + host-resident experts (`-ncmoe`/`-cmoe`) + `-sm tensor` ran at ~5.5-9 t/s against ~23 t/s
for the cache-less CPU-expert path.  The campaign's hypothesis was that r54's correctness fix routed the
per-op segmented expert upload through the **synchronous** setter, and that making it async would recover
the lost overlap.

### 2.2 What is actually wrong (measured 2026-10-21, gfx1201)

The sync upload is real but it is the *symptom*: **the expert cache never creates an arena on Gemma4**, so
the run pays a full per-op expert upload with **zero residency** — the state the code itself warns is
"~8x slower than the cache-less path".

Instrumented counters (temporary, `GGML_DELIV_STAT`), gemma4 26B-A4B Q8_0 `-ncmoe 40 -sm tensor`:

```
scheduler: prune_branch=716  copy_experts_calls=8174  copy_experts_bytes=28.5 GiB   (≈2.5 GB/token)
meta:      segmented sync set  4087 calls  19.0 GiB  3.80 s of host-block time in an 8-token run
cache:     enabled=1 sized=0 n_tables=60 registered=7713.8 MiB expected=23141.3 MiB complete=0
```

**Root cause A — a 15 KiB accounting drift disables an 11 GB arena.**  `moe_cache_tables_complete_locked()`
gates the deferred arena sizing on `registered >= expected`.  `expected` comes from
`moe_host_expert_account()`, which counts **every** host tensor whose name contains `exps`; Gemma4 has a
per-layer **`ffn_down_exps.scale`** (128 x F32 = **512 B**).  30 layers x 512 B = **15360 B** — exactly the
measured shortfall under `-sm layer` (`registered=24265359360` vs `expected=24265374720`).  The comparison
can therefore never hold, `alloc_all_locked()` defers forever, no arena is allocated, and every per-op
upload is a miss.

Proof that this — not the sync setter — is the dominant cost (same binary, `MOE_EXPERT_CACHE_SLOTS`
forces an arena):

| gemma4 26B-A4B Q8_0 `-ncmoe 40` | gen t/s |
|---|---|
| `-sm layer`, cache auto (inert: the shipped state) | **7.9** |
| `-sm layer`, slots=32 | 24.3 |
| `-sm layer`, slots=64 (≈ what auto would size) | **46.9** |
| `-sm tensor`, cache auto (inert) | 5.5 |
| `-sm tensor`, slots=64 (only `down` cacheable, see B) | 6.5 |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=0` (CPU MoE) | 22.9 |

**Root cause B — the fused expert table is never cacheable under `-sm tensor`.**
`ggml_backend_meta_moe_cache_update()` bails out when the split state is not a single segment
(`ss.n_segments != 1 || ss.nr[0] != 1`), because "a mirrored or multi-segment layout has no single
per-device contiguous slice".  Gemma4's `ffn_gate_up_exps` is exactly that: 2 segments (gate, up) each
split on axis 1, so one device's expert slot is `[seg0_slice | seg1_slice]` — two ranges of the host
expert, at two different offsets.  The table never registers (7.7 of 23.1 GB), so even a fixed
completeness check cannot size the `-sm tensor` arena.

### 2.3 Fix, as implemented (all three landed; see `patch-state/`)

- **A (sizing)** — `moe_host_expert_account()` now counts only routed expert **banks**
  (`ne[2] >= 2`), which removes the `ffn_down_exps.scale` residue exactly rather than hiding it, and the
  completeness check keeps a small slack (`min(max(1 MiB, expected/4096), 16 MiB)`) for meta-split
  rounding.  A table that a backend can never register is reported through the existing
  `moe_cache_note_unregisterable()` so it cannot block the sizing.
- **B (async segmented upload)** — a row-aligned segmented `set_tensor_async` splice: one
  `set_2d_async` per (segment, repeat, device) for the full rows and `set_async` for the MMQ tail pad.
  Mid-row ranges (the loader's chunked upload, whose staging ring may be reused on return) keep the
  synchronous setter.  Kill switch `GGML_META_SEG_ASYNC=0`; measured neutral on the miss path (the win is
  A/C), so it ships as the campaign's stated deliverable with an A/B reference.
- **C (multi-slice cache tables)** — a table may declare its device slot as the concatenation of up to 8
  ordered host ranges, which is exactly `ffn_gate_up_exps` under `-sm tensor`
  (`[gate_slice | up_slice]`).  One shared `fill_slot_from_host()` now serves the eager fill **and** the
  identity fill, so the two cannot disagree (the identity path was the first bug found here: it copied a
  contiguous `expert_bytes` window and produced coherent-but-wrong text at 71 t/s).
  Fused tables have no cold (UVA) read and no host-pool source; every miss is filled from the master.

**Result** (`evidence/MEASUREMENTS-2026-10-21.md` has the full tables):

| gemma4 26B-A4B Q8_0 `-ncmoe 40`, `-n 120` | prefix (r99) | now | text |
|---|---:|---:|---|
| `-sm layer` | 7.9 | **79.0** | identical to prefix |
| `-sm tensor` | 5.0 | **74.7** | identical to prefix |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=0` (CPU MoE) | — | 23.7 | (different kernel, expected) |

Long-context `-sm tensor` run: cache `h=1.0000`, identity path, arena 22.9 GiB (the whole expert set
resident), 71.9 t/s gen / 172.3 t/s prompt, wikitext PPL identical to prefix, no GPU faults.

### 2.4 Non-regression requirement — measured

Every change here touches shared machinery (`ggml-backend.cpp`, `ggml-backend-meta.cpp`,
`moe-expert-cache.cu`, the backend iface), so the contract is: other models unchanged in text **and**
throughput.  Result (**all green**, tables in `evidence/MEASUREMENTS-2026-10-21.md` §2.4): Qwen3.8 27B
dense (1 GPU and 3-GPU tensor, q8_0 KV) text identical and tg128 flat (+0.07 %); Qwen3.5 4B 3-GPU tensor
text identical; Gemma4 12B dense text identical; Qwen3.6 35B-A3B `-ncmoe 40` 2-GPU tensor text identical,
tg128 flat (119.3 vs 119.3), cache tables/slots/arena **byte-for-byte the same configuration**;
`test-llama-archs` 375/375; both prefill-logit gates PASS.

**The trap this change invites, recorded:** an earlier draft of A used a `max(1 MiB, expected/128)` slack,
which on 35B-A3B (one table = 36.4 MiB) silently accepted a sizing pass with one table unregistered
(240 -> 239 tables) and cost **-3.6 % tg128**.  That is why A filters the account instead of leaning on a
wide tolerance, and why the slack is capped at 16 MiB.

## 2.5 MTP with the new Gemma4 drafter (checked 2026-10-21)

`/llm/models/Gemma4/26B-A4B/Q8_0/mtp-gemma-4-26B-A4B-it.gguf` (arch `gemma4-assistant`) works with the
patched tree: text identity vs plain holds on a short greedy run, acceptance is healthy (~0.66 at
`-n 200`, mean accepted length 2.97), and it is faster than plain once the context is pinned:

| arm (`-ncmoe 40 -sm layer --fit false -c 16384`, prose prompt) | gen t/s |
|---|---:|
| plain | 75.1 |
| `draft-mtp` n_max **2** | **86.2** (2/2 identical text) |
| `draft-mtp` n_max 3 | 85.3 |
| plain, `-n 3000` | 60.0 |
| `draft-mtp` n3, `-n 3000` | **82.8 (+38 %)** |

Two caveats, both recorded in the handover: (a) at the **default** (max) context `--fit` gives the KV 19 GiB
and the arena only 13.8 GiB, which costs the cache its residency and makes MTP *slower* than plain
(50.2 vs 79.3) — pin `-c` when measuring; (b) **MTP is not run-to-run deterministic on 2 devices with
captured graphs**, and the differing values are the *draft model's* logits (a single context with the cache
is bit-identical 6/6) — see `HANDOVER-2026-10-21b.md`; the text-based "deterministic with X" claims in this
section are retracted.  Judge determinism on the logit-level instrument (`tools/fp.sh`), never a text hash.
The drafter is correctly rejected for `-sm tensor` (`gemma4-assistant` is not in
`llm_arch_supports_sm_tensor`), so gemma4 MTP is `-sm layer`-only until TODO #68 is fixed.

## 2.6 History: why the *previous* async attempt hung (do not repeat)

Preserved from the absorbed `wip/meta-segmented-upload-async/` (its own record is
`archive/work/moe-expert-cache/item3-findings-session20.md`): an earlier partial fix implemented
"segmented port + tail walk + per-row 1-D async sets" and **hung non-deterministically** (one GPU
spinning).  r54's correctness fix plus this campaign's sized arena give that work the deterministic
reference it lacked (an A/B arm and `GGML_META_SEG_ASYNC=0`), and the shipped B implementation is the
narrower shape: **row-aligned only**, per-(segment, repeat, device) `set_2d_async` for the full rows and
`set_async` for the tail pad, with the mid-row loader path left synchronous.  Rules that still hold:

- Never debug this path under `HIP_LAUNCH_BLOCKING=1` or `AMD_SERIALIZE_KERNEL=3` — both deadlock it.
- A device fault here is asynchronous: bisect with a targeted `ggml_backend_synchronize`.
- The meta compute container is double-buffered and cleared per `graph_compute`; a split intermediate
  that crosses a `graph_compute` boundary must stay alive for the consumer.
- The async arm's source pointer must outlive the queued copies.  `copy_experts` uploads from the
  immutable pinned host master (safe); the loader's chunked upload owns a ring with events (safe), but a
  mid-row chunk still takes the synchronous setter so the contract stays one-line-auditable.

## 3. Configs under test

| model | config | why |
|---|---|---|
| Gemma4 12B Q8_0 | 1/2/3 GPU, `-sm layer` and `-sm tensor`, f16 + `-ctk/-ctv q8_0` | #76 (fusion) and #68 (aborts) |
| Gemma4 26B-A4B Q8_0 | `-ncmoe 40`, `-sm layer` / `-sm tensor`, cache auto / `MIB=0` | #77 (cache + segmented table) |
| Qwen3.8 27B Q8_0 (dense) | 1 GPU + 3 GPU tensor, q8_0 KV | non-regression, prefill gate |
| Qwen3.6 35B-A3B Q4_K_M (MoE, single-segment tables) | 2 GPU tensor `-ncmoe`, MTP n3 | non-regression of the cache path |
| Qwen3.5 4B Q8_0 | 3 GPU tensor | coherence/probe non-regression |

## 3.5 Maintainer decisions (2026-10-21)

Recorded in `HANDOVER-2026-10-21.md` §0 — settled, not to be re-litigated:

1. The stray `(file*).moe-heatmap` artifact is **dropped in block 09 at the promotion cut** (not before).
2. **MTP determinism is a must-fix** and is the session's first work item.
3. The **`--fit` arena shortfall under MTP is investigated with a view to fixing it**, same session.
4. (2026-10-21 session b) **A text hash is not a determinism instrument here** — judge run-to-run
   determinism on the draft-logit fingerprint (`tools/fp.sh`) or `decodestart.sh`.  The text-based
   "deterministic with `DISABLE_GRAPHS` / `ASYNC_INPUT=0` / `n_max 2`" claims are retracted.
5. (2026-10-21 session b) The **ids-staging band-aid is dropped** and the **bounded scheduler barrier is
   removed from the tree** (it fixed nothing; 3-run coincidence).  The **per-origin async-input ring**
   (`0005`) stays — a genuine cross-context bug — and is not folded until the determinism item is closed.
6. (2026-10-21 session b) The `-b`/spec verify-batch abort is **in this campaign** (`0006`), not a new TODO.

Priority order after that: #68 1b (f16 `handle_per_row`) → #68 1a (rotation `mul_mat split states`) →
promotion checklist (`HANDOVER-2026-10-21.md` §6).

## 4. Records in this directory

| file | what |
|---|---|
| `README.md` | this file: the campaign, its items, the plan and the gate contract |
| `HANDOVER-2026-10-21b.md` | **current cold-start record**: the MTP-determinism instrument, reproduction commands, what is established/retracted, the 7-patch set, the next bisect, and the deferred items |
| `evidence/MEASUREMENTS-2026-10-21c.md` | determinism numbers + the two **retractions** (read before `-21b`) |
| `tools/fp.sh`, `tools/clihash.sh` | the determinism instruments (draft-logit fingerprint; SSE delta hash) |
| `ISWA-WRITE-FUSION.md` | the complete #76 record (symptom, code facts, fix, gfx1151 + gfx1201 evidence, the 512-wide gap) |
| `HANDOVER-2026-10-21.md` | handover for the next session (with the maintainer decisions in §0): **TODO #68** (the two `-sm tensor` aborts), the MTP determinism finding, and three smaller items (the stray block-09 artifact, the `--fit` arena note, the `--spec-draft-n-max 2` observation) |
| `patch-state/` | the exact patches this campaign produced, with hashes + apply instructions |
| `NEXT-SESSION-PROMPT.md` | the prompt handed to the next session (scope, order, done criteria) |
| `evidence/` | raw measurement logs/tables referenced above |
