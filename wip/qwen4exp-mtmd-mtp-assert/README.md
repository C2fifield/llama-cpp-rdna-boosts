# wip/qwen4exp-mtmd-mtp-assert — drop the `ubatch.token` assert that SIGABRTs every image request under draft-mtp

One `git am` file, made on `v12-097cf33d9-r61` (net tree `047807d7…`; with the commit applied, tree `fd8f289f…`). It touches `src/models/qwen4exp.cpp` only: one line removed, the `GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input")` in the `graph_mtp` constructor (line 1044 of the applied tree). `src/models/qwen4exp.cpp` is byte-identical between r61 and r62 (blob `e76f29ada` in both release diffs), so the patch applies unchanged to the current release, `v12-097cf33d9-r62`. Nothing under `ggml/`. Text-only outputs are unchanged.

**The problem.** With `--spec-type draft-mtp` on, every image request to Qwen3.8-Flash-Next kills the server while the mtmd chunk is decoded:

```
GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input")
  src/models/qwen4exp.cpp:1038        (our capture's line number: r50-lineage build; 1044 on r61/r62)
```

mtmd feeds the projector's output as an embedding-only ubatch (`token == nullptr`, `embd` set), and with an MTP draft attached that decode reaches `graph_mtp`. Our production host logged 14 SIGABRTs of exactly this assert on 2026-10-14 — one per image request, and llama-server down takes the rest of the loaded models with it on our stack.

**The assert is the only thing blocking the path.** `graph_mtp` is already built for embedding input: it registers `inp->embd` with `ggml_set_input`, and its ubatch branch reads `tok_embd = inp->embd` when `ubatch.token == nullptr` (llama.cpp #29622); the feeder, `llm_graph_input_embd_h::set_input`, fills that tensor from `ubatch->embd` end-to-end (llama.cpp #30257). The assert predates the mtmd port and rejects a shape the code around it already handles. The same assert also exists in upstream llama.cpp at `097cf33d9` (`qwen4exp.cpp:529`), so it may want an upstream twin; we report it here, where the image path actually runs.

**The change.** Remove the assert. Nothing is added and no control flow changes: token-carrying ubatches behave exactly as before, embedding-only ones now take the else-branch that was already there.

**Evidence** (gfx1151 / Strix Halo, ROCm payload, Qwen3.8-Flash-Next Q5_K_M + shared MTP head, ctx 131072, `--spec-type draft-mtp --spec-draft-n-max 3`):

- The unpatched server aborts on the repro request. The patched one answers correctly, twice byte-identical, with `usage.prompt_tokens = 322` against a 12-token text prompt — the image was applied, and MTP stayed enabled.
- Control: an older same-lineage engine (r50) reproduces the abort on the same request.
- Gated production cutover against the previous engine: text-only greedy byte-identical; logit fingerprint (truncated top-5 KL, top-1 agreement) 0.0 / 1.0; MTP decode x0.9966 with acceptance unchanged (0.63415); primary and coder llama-bench within x0.994-1.004.

**Repro note.** The legacy `/completion` `image_data` field is silently ignored on this base — the request proceeds text-only and never reaches the assert. A repro must use OpenAI-style `/v1/chat/completions` with an `image_url` content part, and should confirm `usage.prompt_tokens` is well above the text-only count before trusting a "no crash".

**Same pattern elsewhere.** `src/models/deepseek4.cpp:1369` carries the same `ubatch.token` assert in the DEEPSEEK4 MTP graph constructor. We could not test it (no deepseek4 checkpoint on this host); flagging it for whoever has one.

*Patch and note prepared agent-assisted.*
