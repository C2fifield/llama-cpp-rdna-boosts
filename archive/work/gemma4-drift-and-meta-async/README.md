# `gemma4-drift-and-meta-async` — r63 residuals (seeded 2026-10-14)

**Status: ARCHIVED (2026-10-15) — promoted into `v12-097cf33d9-r64` (block 09 + block 10).**  Both
residuals are root-caused: (A) is a proven near-tie (accepted as MoE drift), (B) and the loader-async
ring bug are fixed and folded into the delivery.  See `FINDINGS-2026-10-15.md` and `WORKLOG.md`
261015.001 (r64).  The `tools/*.patch` files are the cut candidates, kept for provenance.

Two independent questions:

* **(A) Is the Gemma4 26B-A4B front-vs-tail greedy divergence acceptable, or does it hide a real
  placement-dependent numeric difference?**
* **(B) Can the `ggml_backend_meta_set_tensor_async` segmented mid-row fast path be made robust, rather
  than merely unreachable from the loader?**

They share one root: the host-expert upload path for a **segmented** Meta tensor.  (B) is a latent
hazard in how that path classifies a write; (A) is the question of whether the classification / serving
choice also perturbs numerics.

---

## A. Gemma4 26B-A4B front vs tail is not bit-identical

### Observed (r63, gfx1201, 2×R9700)

`llama-cli -m /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -f
prompts/prose-rdna-boosts.txt -n 32 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048
-ub 512 -ctk f16 -ctv f16 -fa auto -ngl 99 -sm tensor -ncmoe 8 -t 8`, hashed with
`scripts/extract-generated.py`:

| placement | expert cache | output |
|---|---|---|
| tail (default) | `MOE_EXPERT_CACHE_MIB=4096` | `125 chars sha=0d2717074ba2` |
| front (`LLAMA_HOST_EXPERTS_FROM_END=0`) | `MOE_EXPERT_CACHE_MIB=4096` | `132 chars sha=468a953af132` |
| tail (default) | `MOE_EXPERT_CACHE_MIB=0` | `132 chars sha=468a953af132` |
| front (`LLAMA_HOST_EXPERTS_FROM_END=0`) | `MOE_EXPERT_CACHE_MIB=0` | `125 chars sha=0d2717074ba2` |

Both outputs are coherent (`////` = 0) and the hosted set is correct (`NCMOE_TAIL host blk.22..blk.29`,
8 layers × 3 tensors × 2 devices = 48).  The 35B-A3B Q8_0 is **bit-identical** front vs tail
(`139 chars sha=e2dde701fc04`).

### Reading

The (placement, cache) pairs map onto exactly **two** outputs, and toggling the **cache alone** flips
between them — so the two configurations differ only at a near-tie logit (any tiny perturbation flips
the argmax early, at token ~5: "A README" vs "A long, technical README").  The MTP methodology already
accepts "fusion-ordering numerics drift" for MoE models ("On MoE this is not required ... sane output is
the bar"), and r63 shipped on that basis.

### Why it is worth a look anyway

1. It is the **default flip**: a user upgrading from r62 to r63 gets different Gemma4 text.  Decide
   explicitly whether that is acceptable, or whether the placement should be gated/scoped.
2. The 35B is bit-identical, so the drift is model-specific — a near-tie is the benign explanation, but
   the alternative (a placement-dependent **kernel/reduction** difference, e.g. the op-offload pruned
   upload's reduction order, or the cache identity-vs-remap serving) has not been excluded.

### Investigate

* At the first divergent token, compare the **logits**: top-1/top-2 margin for each arm with
  `--logits-file` (or `llama-logits-dump`).  If the margin is ~0 for the divergent token, (A) is closed
  as drift; if not, the placement is changing a reduction, and the owning kernel is the bug.
* Narrow the variable: is it the placement or the **cache**?  Run `-ncmoe 8` with
  `MOE_EXPERT_CACHE_MIB=0` front vs tail (both non-cache) and with the cache on; the table above
  suggests each is a flip, but confirm with logits, not hashes.
* Check whether the divergence disappears with `-sm layer` (different upload path) or with the host
  experts forced to device (`-ncmoe 0`, no placement).
* Key files: `ggml/src/ggml-cuda/moe-expert-cache.cu` (identity fast path vs remap, `alloc_table_locked`),
  `ggml/src/ggml-backend.cpp` (`copy_experts` pruned upload + `ggml_backend_tensor_set_async`),
  `ggml/src/ggml-backend-meta.cpp` (`ggml_backend_meta_set_tensor_async`), `src/llama-model-loader.cpp`.

### Acceptance

Either a logits-level proof that the divergence is a near-tie (document it, keep the default), or a
one-line/kernel fix with `plain == tail` restored on 35B **and** Gemma4, gates re-run.

---

## B. `ggml_backend_meta_set_tensor_async` segmented mid-row fast path

### The code (r63)

`ggml/src/ggml-backend-meta.cpp`, the segmented branch
(`n_segments != 1 || nr[0] != 1 || axis == PARTIAL`):

```c
size_t row_stride = (axis == AXIS_0) ? tensor->nb[1] : (axis == AXIS_1) ? tensor->nb[2] : 0;
if (row_stride > 0 && offset % row_stride == 0) {
    ggml_backend_tensor_set(tensor, data, offset, size);   // -> synchronous splitter
    return;
}
// else accumulate into ctx->async_set; flush only when offset + size == total
```

The intent: a **row-aligned** range (the scheduler's per-op upload of a pruned expert group, with the
`min(expert_size, 512)` MMQ tail) goes to the sync splitter; a **mid-row chunk** (the loader's pinned
1 MiB pieces) is accumulated and flushed whole.

### The bug

The predicate checks only **offset**, not **size**.  The sync splitter's axis-0 path asserts
`size % row_stride == 0`; its axis-1 path treats `size % row_stride` as a ≤512-byte tail.  A range that
starts on a row boundary but is neither a whole number of rows nor a ≤512 tail (e.g. the loader's first
1 MiB chunk of `blk.0.ssm_out.weight`, q8_0, `nb[1] = 4352`: `1048576 % 4352 = 4096`) is misrouted and
aborts (axis 0) / writes a bogus tail (axis 1).

This was the **r54 `-lm none` + `-sm tensor` regression**.  r63 fixes it on the **loader** side (the Meta
device takes whole-tensor uploads, so the loader never sends a mid-row chunk), which makes the bad path
unreachable *from the loader* but leaves the classifier itself wrong for any other caller.

### The failed attempt (do not repeat)

Requiring the range to be expressible by the sync splitter:

```c
const bool whole_rows = (size % row_stride == 0);
const bool tail_pad   = (axis == AXIS_1) && (size % row_stride > 0) && (size % row_stride <= 512);
if (whole_rows || tail_pad) { ggml_backend_tensor_set(...); return; }
```

**Hung the rear placement** (2/3 `llama-bench -ncmoe 20 -p64 -n20` runs; front stable 3/3; reverting
restored 3/3).  Mechanism (not fully pinned): the tightened predicate diverted some **scheduler** per-op
range into the accumulator, which only flushes at `offset + size == total`; a partial per-op upload
never reaches `total`, so its bytes were never written and a later pass blocked.  The
`size % row_stride <= 512` heuristic is **not** a reliable discriminator between "scheduler tail" and
"loader chunk".  A patch of the attempt is in `tools/meta-async-fastpath-attempt.patch` for reference.

### Fix directions (pick one, with a rationale)

1. **Make the sync splitter a real partial-row writer.**  Generalize the axis-0/axis-1 segmented paths to
   place an arbitrary `[offset, offset+size)` byte range per device/segment (the same way
   `write_partial_row` already does for the non-segmented switch).  Then the async fast path is simply
   "any range" and the accumulator disappears for these tensors.  Highest confidence, most code.
2. **Fix the classifier exactly.**  Derive the expressible shapes from the split state
   (`n_segments`, `nr[]`, `ne[]`), not `size % row_stride`: the sync splitter can express exactly the
   per-op range `first_expert*expert_size` .. `+n*expert_size (+ tail)`.  Pass/compute
   `expert_size`/`n_experts` rather than guessing from the row stride.
3. **Make the accumulator per-split, not per-tensor.**  If the accumulator flushed at the *scheduler's*
   split boundary (or carried a declared range), a diverted partial upload would still be written; this
   makes the heuristic safe even when it misfires.  Touches the scheduler/meta contract.
4. **Keep the loader guard and leave the path latent** (status quo): acceptable only if every current
   and near-term caller is row-aligned ± tail.  Document it as a contract on `set_tensor_async`.

### Investigate

* Instrument the segmented branch: log `(tensor->name, offset, size, row_stride, axis)` for every call on
  a rear placement, and identify which caller sends the range the failed heuristic diverted.  That is
  the missing fact.
* Confirm the scheduler's per-op ranges (`ggml_backend.cpp` `copy_experts`) are always
  `n*expert_size (+ tail)` for the segmented tensors, and whether `expert_size % row_stride == 0`.
* Cross-check with `wip/meta-segmented-upload-async/`: that campaign wants to make the segmented per-op
  upload async **again** for perf (gemma4 + `-ncmoe` + `-sm tensor` is ~9 t/s synchronously).  A proper
  fix here should land together with it, or at least not conflict.

### Acceptance

* 35B and Gemma4, `-ncmoe` front/tail, `-lm auto`/`-lm none`, repeated 5×: no hang, no abort.
* `gate-qwen4exp-quant-coherence.sh` green; `gate-prefill-logits.sh` green.
* `meta-segmented-upload-async` perf not regressed (or improved).
* A comment on `set_tensor_async` stating the exact range contract it accepts.

---

## Reproducers (copy-paste)

```bash
G=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
P=prompts/prose-rdna-boosts.txt
CLI=~/llama.cpp/build-rocm-hybrid/bin/llama-cli
E=scripts/extract-generated.py
for arm in "tail:" "front:LLAMA_HOST_EXPERTS_FROM_END=0"; do
  name=${arm%%:*}; env=${arm#*:}
  env HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_HEATMAP=0 MOE_EXPERT_CACHE_MIB=4096 $env \
    $CLI -m $G -f $P -n 32 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto -ngl 99 -sm tensor -ncmoe 8 -t 8 > /tmp/g-$name.txt 2>/dev/null
  echo "$name: $($E /tmp/g-$name.txt)"
done

# the hang repro for any retried classifier fix (35B is the sensitive model):
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
for i in 1 2 3; do timeout 240 env HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_HEATMAP=0 MOE_EXPERT_CACHE_MIB=4096 \
  ~/llama.cpp/build-rocm-hybrid/bin/llama-bench -m $M -ngl 99 -sm tensor -ncmoe 20 -p 64 -n 20 \
  -b 4096 -ub 512 -r 1 -t 8 -fa 1 -o md >/dev/null 2>&1; echo "run $i rc=$?"; done
```

## Artifacts

| file | what |
|---|---|
| `tools/meta-async-fastpath-attempt.patch` | the reverted classifier tightening (for reference; it hangs the rear placement) |
| `tools/meta-async-partial-row-fix.patch` | **FIX (B)**: generalized segmented sync splitter (arbitrary range) + remove the async accumulator/classifier — cut for block 09 |
| `tools/splice-gather-ring-owner-fix.patch` | **FIX (B-extra)**: per-context CUDA splice-gather ring — cut for block 10; not needed with the loader guard on |
| `FINDINGS-2026-10-15.md` | **the result**: (A) logits near-tie proof; (B) caller/range instrumentation + fix + gate results |
| `README.md` | this write-up |
| `HANDOVER-2026-10-14.md` | cold-start handover |

Promoted parent: `archive/work/moe-upload-overlap/` (`RESOLUTION.md`, `FIX-2026-10-14.md`).
Sibling: `wip/meta-segmented-upload-async/`.
