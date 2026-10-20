# `mtp-3gpu-decode` — CLOSED (r55, 2026-10-13): no code change, the symptom is gone on r54

> **SUPERSEDED (2026-10-16).**  This closure is kept as history.  The campaign was **revived** as
> [`wip/three-gpu-performance/`](README.md) after a 2026-10-16 measurement found the opposite (3-GPU
> decode slower than 2-GPU, including at 100 % residency).  The closure's A/B below is not comparable to
> that arm (different model and prompt) and must be re-run on the current tree before it is trusted.

## Result

Step 0 (a fresh 3-GPU vs 2-GPU A/B on the current tree) was run first, exactly as the campaign required.
The r38 deficit is gone:

| arm | prefill (t/s) | decode (t/s) | acceptance |
|---|---:|---:|---:|
| 2-GPU MTP | 1402 | 65.0 | 0.832 |
| 3-GPU MTP | 1250 | **74.3** | 0.854 |
| 3-GPU plain | 1593 | 55.2 | — |

3-GPU MTP is **+34.6 % over plain decode** and **+14.3 % over 2-GPU MTP** (r38 anchor: 3-GPU MTP 20.1 vs
3-GPU plain 53.0).  The verify-step cost that motivated the campaign is gone, so no profiling of the
3-device internal all-reduce was needed.

Harness: `archive/work/moe-verify-fusions/tools/{field3.sh,field_nospec.sh}`; prompt
`archive/work/kv-cell-offset-nondeterminism/prompts/mixed30k.txt` (sha256 `69624f4d...`, unchanged); model
Qwen3.8-Flash-Next IQ4_NL + the shared Q8_0 MTP head, `-ncmoe 48 --spec-draft-n-max 3`, 32 358-token
prompt, 1000 generated.

**TODO #51 closed.**
