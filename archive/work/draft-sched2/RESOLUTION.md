# RESOLUTION — `draft-sched2` (PR #154, @briansp2020)

**Status: RESOLVED (2026-10-15), promoted into the delivery as part of `v12-097cf33d9-r65` (blocks 09 and
10 amended).**  No `wip/` entry; the PR's format-patch and write-up are preserved in this directory.

## What landed

`llama_context` gains a second scheduler (`sched_alt`, created on first use) for the MTP draft context's
graphs with outputs, swapped into `sched` for the duration of a `decode`/`encode` and swapped back by a
scope guard.  Supporting changes: a per-scheduler meta view slot
(`ggml_backend_meta_set_view_slot`) so two live graphs do not clear each other's external views, and a
new optional device-interface entry `slab_compute_plain` so the second scheduler's small COMPUTE buffer
bypasses the slab and the chunk quantization.  Kill switch **`LLAMA_DRAFT_SCHED2=0`** (default on).

## Validation (gfx1201, 2x R9700, ROCm 7.14.1, r65 candidate)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS + shared `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `-sm tensor`,
q8_0 KV, `-lm none`, `-ot per_layer_token_embd.*=CPU`, `--spec-type draft-mtp --spec-draft-n-max 3`,
`-n 3000`, prose prompt, 2 GPUs:

| arm | Generation t/s | acceptance | sched reallocs |
|---|---:|---:|---:|
| r64 reference (x2) | 132.7 / 132.8 | 0.81673 | 1068 |
| r65 default (x2) | 139.6 / 140.1 | 0.81673 | 34 |
| `LLAMA_DRAFT_SCHED2=0` | 137.8 | 0.81673 | 1068 |

The draft-mtp graph allocations drop 1068 -> 34 and the arm adds ~+1.5 % on top of `output-pinned`;
acceptance and the generated text (`5d51438c7a26`, 512-token arm) are identical.  On a dense
Qwen3.8-27B UD-Q4_K_XL `-sm tensor` 3-GPU MTP run the arm was within noise (the Flash-Next
ingest/step alternation is the shape it targets).

## Record

`WORKLOG.md` 261015.002 (r65).  Original PR description and patch: `README.md`, `0001-draft-sched2.patch`.
