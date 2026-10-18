# RESOLUTION -- `gdn-conv-snapshot-fold` (PR #177, @briansp2020)

**Status: NOT SHIPPED (reviewed 2026-10-17, for `v12-097cf33d9-r81`) -- rejected on purity, not on
speed.  Re-entry conditions below.**  The contributor's format-patch and write-up are preserved here
unchanged.  Nothing was left behind as a `wip/` stub (the campaign arrived as a PR branch, never on
`main`'s `wip/`).

## What it does

With the direct GDN conv path active, a GDN layer makes three launches per multi-token step
(`gdn_concat_tail`, the batched snapshot `CPY`s, and `gdn_conv_direct_kernel`).  The patch has the direct
conv kernel also write the 3-column recurrent snapshot tails, read straight from `state` / `x`, so the
`CONCAT` and the snapshot `CPY`s launch nothing.  It is taken only after a runtime byte-span check over the
nodes between `CONCAT` and `SSM_CONV` (no writes over `state` / `x`, no reads/writes of a snapshot
destination, destinations disjoint from the inputs and from each other, and a no-op node right after each
skipped `CPY`).  Kill switch `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`.

## Why it was not taken

**On 35B-A3B the patched path is not run-to-run deterministic, and it does not reproduce the stock
logits.**  The delivery's purity rules do not allow either, regardless of the throughput win.

Reproduced on gfx1201 (3x R9700, ROCm 10.1.0, r80 against r80 + the patch, tree `84ea4301`),
`test-logits-width-probe` on Qwen3.6-35B-A3B Q4_K_M, **1 GPU**, P=256:

| arm | runs | `W8` hash | row-0 hash |
|---|---|---|---|
| stock r80 | 5+ | `dfcb48a5be95c5ac` (stable) | `a44efecb25966236` |
| r80 + #177 | 4 | `48272966cab954fb` x2, **`cb29e653528bddcf` x2** | `907b3e9c40b8c546` / **`361d76c6d5ff877e`** |
| r80 + #177, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` | 2 | `dfcb48a5be95c5ac` | `a44efecb25966236` |
| r80 + #177, `GGML_CUDA_DISABLE_GRAPHS=1` | 2 | `48272966cab954fb` | `907b3e9c40b8c546` |

- `width_purity` still reports **maxdiff 0 within a run** (W=1..8 agree), so the widths are consistent
  with each other; the difference is **between runs**, and it is already in row 0, i.e. in the prefill.
- The kill switch reproduces stock exactly, so the patch is the cause.
- Disabling HIP graphs makes the patched arm stable but at a **third** value, so capture is a second,
  independent divergence on top of the base one.
- The dense 27B is identical to stock at both P=256 and P=3584, and the 400-token greedy text on
  35B-A3B is stable and identical to r80 across 3 runs.  So this is a logits-level prefill effect that
  the same-seed text gate cannot see; the prefill-logit KLD gate also passes (0.000000 / 100.000 % on
  both models), which is why the probe -- not the text or KLD -- is the instrument that caught it.

**The speed win is real and should not be lost:** dense 27B Q4_K_XL MTP decode 1 GPU 52.8 -> 53.3-53.4 t/s
(**+1.0 %**, two rounds, 2 runs each), reproducible; MoE 2-GPU tensor tg256 and pp31500 flat.

## Re-entry conditions

1. **Determinism first.**  The probe's `W8` / row-0 hashes must be identical to stock r80 across at least
   5 runs, with HIP graphs on, on 35B-A3B **and** the dense 27B, at P=256 and P=3584.  A same-seed text
   match is not sufficient evidence: the current patch has it.
2. Name the mechanism behind the graph-on divergence.  The prime suspect is the write moving from the
   `CPY` node to the `SSM_CONV` node while the graph allocator's lifetime for the snapshot destination is
   still anchored at the (now empty) `CPY` node, so a captured graph can replay against a different
   `snap.dst` address than the one it was captured with -- the same class as "a Q8_1 arena must outlive
   captured graphs" and "a graph is recaptured when the pool memory it captured is freed".  Confirm or
   refute before re-submitting.
3. Keep the kill switch and the byte-span check.  Both worked.
4. Report the launch counts from the trace (`gdn_concat_tail` / `cb_kernel` gone) next to the t/s, plus
   the probe hashes per run, so determinism is visible in the report.

## Record

`WORKLOG.md` 261017.008 (r81); review reply on the PR.
