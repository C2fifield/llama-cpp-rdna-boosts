# RESOLUTION — `fix-gemma-4` (closed 2026-10-21, released in **v12-097cf33d9-r101**)

Umbrella campaign for every Gemma4 defect on this delivery.  Seeded 2026-10-21, absorbed the former
`wip/iswa-kv-write-fusion/` and `wip/meta-segmented-upload-async/`, and was **promoted whole** in r101:
three block amendments, no new block.  This file is the closing record; the measurements are in
`evidence/MEASUREMENTS-2026-10-21f.md` (session d, the cut) plus `-21e`, `-21d`, `-21c`, `-21b`, `-21.md`
for the earlier sessions, and the release note is `WORKLOG.md` **261021.003**.

## What was wrong, and what fixed it

| item | defect | fix | where it shipped |
|---|---|---|---|
| **#77** | the gemma4 expert cache **never sized**: `moe_host_expert_account()` counted every `*exps*` host tensor, so the per-layer `ffn_down_exps.scale` (512 B x 30 = 15360 B) left the completeness check permanently unsatisfiable → no arena, every per-op expert upload a miss (`-sm layer` 7.9 t/s, `-sm tensor` 5.5) | count only routed expert **banks** (`ne[2] >= 2`) + a slack capped at 16 MiB; **multi-slice cache tables** (a device slot assembled from up to 8 host ranges, which is what makes the fused `ffn_gate_up_exps` cacheable under `-sm tensor`); a row-aligned segmented `set_tensor_async` splice (`GGML_META_SEG_ASYNC=0`) | block 10 |
| **#76** | the r81 rope+FWHT+KV-write fusion was **inert on every iSWA arch**: `build_attn(llm_graph_input_attn_kv_iswa*)` expanded `q, k, v`, so the V block sat between the K rotation and the K cache write and the matcher declined | expand `q, v, k` (5 lines), mirroring the non-iSWA builder | block 10 |
| **#68 1b** | gemma4 dense + `-sm tensor` aborted in `handle_per_row`: the single-KV-head full-attention row (512) lands **whole on one device** (`ne_per_dev=[0,512]`), and any axis-0 split was rejected | accept an axis-0 split whose every per-device extent is empty or the whole row; `handle_bin_bcast` accepts an elementwise op against a MIRRORED same-shape operand | block 10 |
| **#68 1a** | inflated by the rotated cache (`-ctk q8_0`): `mul_mat(attn_inp_k_rot MIRRORED, Kcur)` has its **contraction on axis 0** and `handle_mul_mat` had no such pair | accept (MIRRORED, axis-0) when the contraction is empty-or-whole per device and `out->ne[0] == src1->ne[0]`; the owning device computes the whole dot product, nothing is reduced.  A straddling contraction still aborts, by name | block 10 |
| **#68 1c** (found while closing 1a) | gemma4 dense + `-sm tensor` aborted in `handle_flash_attn_ext` when one ubatch carried **more than one sequence** (`llama-server -np > 1`, `llama-batched-bench -npl > 1`, `llama-perplexity --kl-divergence`): with `n_stream > 1` the FA cache view spans the stream dimension, so gemma4's whole-on-one-device KV arrives split on the cache's **embd** axis instead of the kv-head axis.  Pre-existing code, newly *reachable* (1b/1a removed the earlier aborts) | accept an empty-or-whole KV slice when the query is empty-or-whole on the same devices | block 10 |
| **`--fit`** | the draft reserve charged a full `n_ctx` KV (18022 of 19152 MiB/device) to a drafter that allocates none (gemma4-assistant shares the target's KV); MTP at the default context was **slower than plain** (41.0 vs 74.6 t/s) | charge only the KV layers the draft owns (`attention.shared_kv_layers`) | block 09 |
| **`-md` trap** | `-md <drafter> --spec-type none` loaded the model, created and warmed up its context, and **aborted** (`GGML_ASSERT(ubatch->embd_state)` — the drafter's graph needs the target's hidden state), so a "plain baseline with the drafter loaded" was impossible | `uses_dft()`/`has_aux()`: an unused draft is neither loaded nor given a context (one WRN); no wasted VRAM either | block 09 + 11 |
| **ni** | `--fit`'s expected `… requires ctx_other to be set` probe failure logged as **E** | **W** (any other init failure stays E) | block 11 |
| **ni** | the stray `(file*).moe-heatmap` artifact shipped in block 09 and was rewritten by `test-llama-archs` | dropped — it is no longer in the set at all | block 09 |

**MTP run-to-run determinism** was the campaign's first item and closed as **external**: ROCm's hipBLASLt
per-start solution pick for gemma4's F32 MoE router (issue #67) — `GREEDY-PURITY.md` §44, invariant 9.  The
per-origin async-input ring fix (`0005`) that the hunt produced is a real cross-context fix and shipped in
block 10.

## Result (all `ROCBLAS_USE_HIPBLASLT=0`)

| what | before | after |
|---|---|---|
| gemma4 26B-A4B `-ncmoe 40 -sm layer` | 7.9 t/s (cache never sized) | **79.0** |
| gemma4 26B-A4B `-ncmoe 40 -sm tensor` | 5.0-5.5 (aborted on the size check) | **74.7** |
| gemma4 26B-A4B MTP n3, default 262144 ctx | 41.0 t/s (arena 13752 MiB) | **93.2** (arena 23141 MiB, slots 4560 -> 7680) |
| gemma4 dense 12B `-sm tensor` (f16 **and** rotated q8_0) | aborted | runs, byte-identical to `-sm layer` (`c740be0e131a`, 2 and 3 GPU) |
| gemma4 dense + `-sm tensor` + a multi-sequence batch | aborted (`-np 2` server, `-npl 4` bench) | runs (`-npl 4`: 3353 t/s; server: two concurrent 400-token requests, both 200) |
| gemma4 q8_0 KV launches (`-n 8`) | 13500 (fusion inert on iSWA) | **12540** (`k_set_rows_q8_0_group` 1152 -> 192) |
| `-md X --spec-type none` | abort | plain baseline + one WRN |

Non-regression (35B-A3B / 27B dense / 4B / gemma4): text identical within each split mode, the 35B-A3B
cache canary byte-identical (240 tables / 61440 slots / 18662.0 MiB), tg128 flat, `test-llama-archs`
644/644, all three prefill-logit gates PASS, and the four-axis adaptive-MTP gate at `-n 3000` green on both
stock models (`none` == `draft-mtp` text on all eight pairs; dense hashes identical to `build-r99`).

## The two rules this campaign leaves behind

* **`GREEDY-PURITY.md` §45** (+ invariant 9): a split the Meta backend cannot serve **locally** must be
  refused by name, never approximated.  "Locally" means *no device straddles the reduction* —
  empty-or-whole per device — and that is now the shape of all three accepted cases (per-row ops, the
  rotated-K `mul_mat`, the multi-sequence FA K/V).  Each came with its straddling-case refusal.
* **`AGENTS.md` invariant 9 / §44**: a fresh-process comparison is only a purity measurement with the
  external solution selection pinned (`ROCBLAS_USE_HIPBLASLT=0`), and never on a text hash.

## Where the record is

| file | what |
|---|---|
| `README.md` | the campaign as it stood (items, plan, the §2.4 non-regression contract) |
| `HANDOVER-2026-10-21c.md` | session c: determinism closed, `--fit` + `-sm tensor` 1b fixed, 1a + promotion left |
| `HANDOVER-2026-10-21b.md`, `HANDOVER-2026-10-21.md` | sessions a/b: the determinism instrument, #68 scope, the `--fit` candidates, the promotion checklist |
| `ISWA-WRITE-FUSION.md` | the complete #76 record (incl. the deliberate 512-wide full-attention gap) |
| `evidence/MEASUREMENTS-2026-10-21f.md` | **session d**: 1a, the multi-sequence defect, the nits, and the full gate set on the 12-patch and the folded tree |
| `evidence/MEASUREMENTS-2026-10-21{e,d,c,b,}.md` | the earlier sessions' numbers (the `--fit`/`-sm tensor` fixes, the hipBLASLt root cause and the retractions) |
| `patch-state/` | the campaign's 12 patches against `972f3b757`, with sha256s.  **Historical**: they are folded now — the delivery is `patches/0009..0011` |
| `tools/` | the determinism instruments (`fp.sh`, `clihash.sh`) and the throwaway graph-gate diff |
| `NEXT-SESSION-PROMPT.md` | the prompt this session ran (session d) |

**Re-opening this:** the campaign is closed.  A new gemma4/`-sm tensor` defect is a new TODO entry and a new
campaign — do not edit these records, and do not re-apply `patch-state/` to a current tree (its base is
`972f3b757`, two releases old).
