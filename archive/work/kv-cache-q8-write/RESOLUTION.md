# RESOLUTION -- `kv-cache-q8-write` (PR #173, @briansp2020)

**Status: Resolved in two stages.  REJECTED for `v12-097cf33d9-r78` (2026-10-17), then SHIPPED in
`v12-097cf33d9-r81` (2026-10-17) as part of block 10 -- the re-entry conditions below were met by the
rebased patch (`0001-...-r80.patch` here) on top of PR #175.**

The contributor's original format-patch (r71), the r80-rebased one, and both write-ups are preserved
here.  The r78 rejection text below is kept as written; read it as `why it was rejected at r78`, not as
the current state.

## r81 (shipped): how the re-entry conditions were met

1. **Rebase** on the r78 set: done, `0001-...-r80.patch` applies on `v12-097cf33d9-r80` + #175 (tree
   `67245faf`), unchanged in substance.
2. **Matcher fires.** The Qwen3.x models are not reachable via the iSWA builder; qwen35 / qwen35moe use
   the **non-iSWA** `build_attn`, which expands K last, so the K cache write follows the K rotation.  The
   iSWA ordering for Gemma4 is unchanged and still not matched (see `TODO.md`).  Traced on gfx1201
   35B-A3B 1 GPU, 120-token decode: `k_set_rows_q8_0_group` **2500 -> 20**, and
   `rms_norm_mul_rope_f32<256,false,float,256>` splits into 1220 Q writes and 1240 K + cache writes.
3. **Re-measured** on models where it fires: same-seed text byte-identical to r80 and to r80 + #175 in
   six arms (35B-A3B `sha ecb4e0796ba5`, 27B `sha 980719527364`), width probe PASS maxdiff 0 with the r80
   hashes at P=256/3584, prefill-logit **0.000000 / 100.000 %** on both models, `SET_ROWS` 159/159,
   `ROPE` 450/450, `RMS_NORM_MUL_ROPE` 298/298.  Decode on top of #175 (2-GPU tensor, 35B-A3B): tg256
   115.67 / 115.46 -> 116.17 / 115.83 (**+0.4 %**), `draft-mtp` flat; dense 27B MTP decode flat (53.1 both
   arms, so their +0.7 % does not reproduce).
4. Pairing it with the M-RoPE extension (#175) is exactly what made it reachable; the note in `TODO.md`
   was the right call.

---

# r78 record (kept as written at rejection)

**Status at r78: REJECTED for `v12-097cf33d9-r78` -- not shipped.  Re-entry conditions below.**
The contributor's format-patch and write-up are preserved here.

## What it does

With a q8_0 KV cache it folds the two `SET_ROWS` nodes that write the K and V cache rows into the fused
`rms_norm_mul_rope + FWHT` kernel from PR #172: warp 0 quantizes the rotated K row from its FWHT
registers and each block quantizes its slice of the (already rotated) V row, both through the
`k_set_rows_q8_0_group` quantization (moved into a helper in `set-rows.cuh`).  Two launches per layer
would go away; inside a concurrent-stream region only the K write is folded.
Kill switch `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS`.

## Why it was not taken

**The matcher cannot fire on the current tree, on any model available here, so it is unmeasurable and
would ship as dead code.**

1. It requires the fused rope kernel from PR #172, i.e. `ROPE` mode NORMAL/NEOX with the
   `RMS_NORM, MUL, ROPE` chain.  On this box that is **Gemma4 only** (every Qwen3.x model here is
   `rope_multi`, and gpt-oss-20b has no qk-norm).  Verified by kernel traces: `rms_norm_mul_rope_f32<...,
   256, false>` (the #172 `fwht_n` variant) appears on Gemma4 and on no Qwen3.x model.
2. It requires the K cache `VIEW, SET_ROWS` to immediately follow the K rotation's `RESHAPE`.  On
   Gemma4 it does not: the **ISWA graph builder** (`llm_graph_context::build_attn(llm_graph_input_attn_kv_iswa*)`)
   expands `q_cur`, then `k_cur`, then `v_cur`, and only afterwards the two `cpy_k` / `cpy_v` cache
   writes -- so the V projection/norm/rotation nodes sit between the K rotation and the K write.
   Node dump of a Gemma4 decode graph (r78):

   ```
   15 RMS_NORM 16 MUL 17 ROPE 18 RESHAPE 19 MUL_MAT (K rotation) 20 RESHAPE
   21 MUL_MAT[Vcur] 22 RESHAPE 23 RMS_NORM 24 RESHAPE 25 MUL_MAT (V rotation) 26 RESHAPE
   27 VIEW 28 SET_ROWS[cache_k_l0] 29 VIEW 30 SET_ROWS[cache_v_l0]
   ```

   The matcher (rope at `i`, `i+5` reshape, `i+6` view, `i+7` set_rows) therefore rejects with
   `k_view_src_is_reshape=0`, confirmed with an instrumented build.  The non-iSWA builders already use
   `q_cur, v_cur, k_cur` ("expand k later to enable rope fusion which directly writes into k-v cache"),
   which is why the pattern worked on the reporter's tree (a non-iSWA NEOX model).

No performance regression is implied by the rejection: the two `SET_ROWS` launches still run exactly as
in r77 (`k_set_rows_q8_0_group` counts identical), and the full r78 gate suite (coherence, width probe,
prefill-logit, backend-ops, batched-bench) is green without it.

## Re-entry conditions

1. Rebase onto the r78 set (the #172 fused rope kernel is in block 10 now; the `k_set_rows_q8_0_group`
   helper refactor is a no-op for this purpose).
2. Handle the **ISWA ordering**: either teach the matcher to tolerate the interposed V block (scan a
   bounded window and require single-use/non-output nodes throughout), or change the ISWA builder's
   expand order to `q_cur, v_cur, k_cur` like the other builders -- the latter is a `llama-graph.cpp`
   change that must be measured separately for graph splits, and it is not part of this PR.
3. Re-measure on a model where the matcher actually fires.  Report the launch counts (the trace, not
   only t/s) so it is provable the two `set_rows` launches are gone, plus the greedy hash and the
   `test-logits-width-probe` result at the verify widths.
4. Worth doing alongside PR #172's mrope gap: on this box #172 (and therefore #173) is inert on all
   Qwen3.x models.  Extending the fused rope kernel to `GGML_ROPE_TYPE_MROPE` is the change with the
   larger reachable win here; see `TODO.md`.
