# kv-pad-prefill: one graph for several prompt ubatches (`LLAMA_KV_N_PAD_PREFILL`)

One format-patch on top of `v12-097cf33d9-r71` (tree `dc2cc150`; applied tree `0cc58912`), for `git am`.  It also applies cleanly to
`v12-097cf33d9-r72` (tree `43115425`; applied tree `cd368ea6`); the tests below are on r71.  It touches
`src/llama-kv-cache.cpp` and `src/llama-kv-cache.h` (+25 / -4).

## What we saw

During prompt processing in 2048-token ubatches, each ubatch grows the KV cache by 2048 cells.  `n_kv` is padded to
`max(256, LLAMA_KV_N_PAD_MIN)`, which is smaller than a ubatch, so it is different for every ubatch.  The KV views
in the graph change shape with it, so the previous graph can't be reused: every chunk of a long prompt builds,
splits and allocates a new graph while the GPUs wait.

## The patch

- `get_n_kv(sinfo, n_tokens)`: prompt-sized ubatches (`n_tokens > 64`) are padded to `LLAMA_KV_N_PAD_PREFILL`, rounded
  up to a power of two (`GGML_PAD` is a bit mask), once the decode-padded occupancy exceeds
  `LLAMA_KV_N_PAD_PREFILL_FROM` (default 4096).  With 8192, four consecutive 2048-token ubatches share one `n_kv`.
- Below the threshold the decode pad stays, so qwen4exp's dense attention for the first ~2051 cells still applies.
- Decode and verify batches (64 tokens or fewer) keep the decode pad.  In an earlier test on our own build, an 8192
  pad for every batch slowed decode by about 5 % and changed greedy output, which is why the coarse pad is limited
  to prompt ubatches.
- `n_kv` is still capped at the cache size, and the extra cells are masked as before.
- Unset (or 0), nothing changes.  Not for training, whose graphs assert `n_kv == n_tokens`.

## Tests (2 x R9700 32 GB, gfx1201)

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM, + MTP draft (shared Q8_0, `--spec-draft-n-max 3`), no other
tuning env.  Each run is a fresh server after `llama-warmup.py`, in the order r71, patch, two rounds.

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor --n-cpu-moe 0 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0
```

| t/s (round 1 / round 2) | r71 | r71 + patch, `LLAMA_KV_N_PAD_PREFILL=8192` |
|---|---|---|
| 31.5k-token prompt, 1st | 2578.3 / 2537.5 | 2682.0 / 2648.3 |
| 31.5k-token prompt, 2nd | 2505.4 / 2558.1 | 2686.3 / 2648.6 |
| 9.4k-token natural text (round 1) | 2434 | 2558 |
| 32.6k-token natural text (round 1) | 2493 | 2629 |
| greedy 400 tokens, 2nd request | 142.3 / 142.1 | 142.1 / 142.2 |

The 31.5k-token prompt goes from 2545 to 2666 t/s on average (+4.8 %), and decode is unchanged.

Outputs are the same with and without the patch: greedy `a7a81f01b3bc` (draft acceptance 274/374 in every run), and
the 150-token greedy completions after the two natural-text prompts `17429c757874` (9.4k) and `59981b3ce35d` (32.6k).
No faults or aborts.

No urgency on our side.  Happy to re-run it on your config or with other pad sizes.
