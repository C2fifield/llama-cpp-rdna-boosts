# RESOLUTION — `0013-meta-hot-dispatcher`: PROMOTED into `v12-097cf33d9-r86` (block 00)

**Status: folded into block 00 (2026-10-19).**  The candidate was decided on 2026-10-18 (issue #184
follow-up: bank it, no standalone release) and promoted with the r86 revision.

## What was promoted

`ggml_backend_meta_get_split_state` returns to the exact pre-#170 shape (meta buffer first, then the
buffer context and the recursive call) and the whole issue #170 resolution — the view-chain buffer, the
host test (`ggml_backend_buffer_is_host`) and the `[TAG_META_HOST_VIEWS]` assert — moves into a `static
__attribute__((noinline, cold))` helper.  No behaviour change, no new environment variable; the
dispatcher drops 454 -> 287 B and the cold helper sits at 164 B.

## Validation (promotion gates, tree `bb825861`)

- 4-concurrent-stream A/B (2x R9700, only `libggml-base.so` swapped): **-0.06 %** vs the delivery tree,
  greedy text identical (`archive/work/issue-184-meta-r80r82/`).
- issue #170 regression: the 24 GiB-parity reproduction (8 GiB held per card, dense 27B Q8_0, `-sm
  tensor`, `--ctx-size 196608`) **LISTENING**; a genuine weight OOM (20 GiB held, `--fit off -ngl 99`)
  **exits rc=1 cleanly**, no `GGML_ASSERT`.
- dense and host-expert MoE greedy text byte-identical to the r86-candidate build; prefill-logit PASS
  (0.000707 / 98.755 %); adaptive-MTP 5 axes exactly equal; `llama-batched-bench` TG equal.

Full record: `WORKLOG.md` 261018.010 and 261019.001.
