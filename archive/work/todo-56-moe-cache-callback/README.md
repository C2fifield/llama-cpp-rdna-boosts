# `todo-56-moe-cache-callback` — TODO #56: the scheduler never invokes the copy callback, so the upstream MoE cache's slot map is never written

**Status: ROOT-CAUSED, FIXED and VALIDATED on gfx1151 (Strix Halo, 1 device).  The patch is included here
(`todo-56-moe-cache-callback.patch`); the fix is also a commit on the fork branch
`todo56-moe-cache-fault` in `~/llama.cpp` (`39f1e0b93`, on top of r97 `396376617`).  NOT integrated:
nothing is folded into `patches/`, `release.json` is untouched, no release/tag was cut, no `WORKLOG.md`
entry exists, and the fork's `rdna-boosts` branch was not pushed.  Handed over for integration by the main
development host (see §9); not part of the delivery until that happens.**

Self-contained: this README plus the patch is everything needed.  Read `AGENTS.md` ("Report findings before
shipping", "WIP and promotion rules") before integrating.

## 0. TL;DR

`--moe-cache-mib` (the **upstream** `llama_moe_cache`, `src/llama-moe-cache.cpp`) has been broken on every
block from 09 on.  It is **not** a defect in the upstream cache engine, **not** deepseek2-specific and
**not** the pinned host-expert loader: the delivery's rewritten scheduler
(`ggml_backend_sched_copy_input`) **never calls `sched->callback_copy`**, and that call is the only thing in
the tree that drives `llama_moe_cache::copy()`.

Without it the cache's **slot map is never written** (the host buffer is cleared with `0xff`, so it stays
`-1`), while the graph has already replaced the routed op's `src0` with the cache **bank** view and its ids
with `GET_ROWS(slot_map, ids)`.  The routed `MUL_MAT_ID` then indexes the bank with `-1` — one expert
*below* the bank.  Where that address happens to be mapped the answer is silently wrong; where it is not,
it is an HSA memory fault.  Both recorded #56 manifestations are that one bug.

Fix: restore the base `callback_copy` call in the scheduler, and delete the dead per-op staging/pruning copy
that duplicated the scheduler's own work.  Two files, **+20/−97**.  `test-llama-archs` goes
**373/375 -> 375/375**.

## 1. Reachability, severity, and why it is still worth landing

**The switch is the whole story.**  `llama_moe_cache` is constructed only when
`llama_context_params.moe_cache_size > 0` (`--moe-cache-mib`; the default is 0 and nothing in the delivery
auto-enables it — `src/llama-context.cpp`, `.moe_cache_size = 0` in the default params).  With it unset,
`build_moe_cache_slots` returns null, no `GET_ROWS(slot_map, …)` node is built, and no `-1` id ever reaches
a kernel.

So:

* **No documented delivery config is or was affected.**  `-ncmoe`/`-cmoe`, `MOE_EXPERT_CACHE_MIB`,
  `-sm tensor`/`-sm layer`, MTP, server — none of them are on this path.  That is why it survived from
  block 09 to r97 with every release gate green.
* **The fix is inert when the switch is unset** — `sched_copy_experts` is then `lctx->moe_cache && …` ->
  false, and the scheduler falls through to exactly the copy it made before.  Verified, not assumed
  (byte-identical greedy text, pp/tg within noise: §7).
* Correct framing of the item: *"the upstream `--moe-cache-mib` engine has been unusable since block 09
  (hard GPU fault / silent corruption)"*, **not** *"a delivery config is broken"*.

Severity is therefore low in user terms, and the reasons to land it are different ones: it is a
**crash-class latent hazard** on a switch the CLI still advertises; it is a **broken upstream contract the
delivery created** (`callback_copy` is assigned and read nowhere, so its only consumer is starved); it
**reddens a test that ships in the tree** (`test-llama-archs` 373/375, `deepseek2` 5/7), which masks any
future regression in those rows; and it removes the orphaned duplicate of the staging/pruning code — the
thing that would silently double-stage every host expert table if anyone re-wired the callback later.

## 2. The patch

| | |
|---|---|
| file | `todo-56-moe-cache-callback.patch` (234 lines, one commit) |
| sha256 | `1cf516563cad1e9cbd23e5a8193572f75f78f1f151d214fdcc5e1d496cd5a054` |
| base | r97, `release.json` tip `396376617`, tree `f81213c4dcb057636ef6afdfef7e4c4907a4d83f` |
| result | tree `520c848952603e88850196b95e8e67b95bafa4cf` (verified by `git am` on a pristine r97 worktree) |
| files | `ggml/src/ggml-backend.cpp` (+13), `src/llama-context.cpp` (deletions + rewrite), `src/llama-context.h` (−15) |
| applies with | `git am` (`git apply` of this single commit is also fine — it is one commit, not a series) |

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp   # or reuse a checkout
git checkout 396376617                                            # r97 fork point tip
git am /path/to/wip/todo-56-moe-cache-callback/todo-56-moe-cache-callback.patch
git rev-parse HEAD^{tree}   # must be 520c848952603e88850196b95e8e67b95bafa4cf
```

## 3. Root cause, exactly

`build_moe_cache_slots` (`src/llama-graph.cpp`) is graph-build side and was never broken: for a layer whose
expert tensors are all bound in the cache and whose batch fits (`n_tokens <= 32` and
`n_tokens*n_expert_used <= n_slots`) it emits `slots = GET_ROWS(slot_map, ids)`, and `build_lora_mm_id`
then builds `MUL_MAT_ID(bank_view, cur, slots)` against the **cached** expert tensor.

The map is filled on **copy side** by `llama_moe_cache::copy()`: it reads the routing back, runs the LRU
`plan()`, uploads the misses into the bank, and finally
`ggml_backend_tensor_set_async(backend, dst, src->data, …)` with `src` = the slot map.  Base (`097cf33d9`)
reaches it through `ggml_backend_sched_copy_input`:

```c
if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
    sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
    return;
}
```

The delivery's rewritten `ggml_backend_sched_copy_input` (blocks 09-11: staging ring, expert-cache
take-over, device gather, per-op upload, slab) **does not contain that call**.  `sched->callback_copy` is
assigned in `ggml_backend_sched_set_copy_callback` and **read nowhere**
(`grep callback_copy ggml/src/ggml-backend.cpp` -> the two struct fields and the setter only), so
`llama_context::sched_copy_experts` — and with it `llama_moe_cache::copy()` — is unreachable.

Evidence, from a temporary print in `ggml_cuda_mul_mat_vec_f` (`ggml/src/ggml-cuda/mmvf.cu`, all fields as
printed):

```
before: src0=moe_cache.blk.1.ffn_down_exps.weight ne=[192,128,4] nb2=98304
        ids=ffn_moe_slots-1 (reshaped) idsne=[2,2] idmin=-1 idmax=-1
after : ids=ffn_moe_slots-1 (reshaped) idsne=[2,2] idmin=0  idmax=3
```

`idmin=idmax=-1` over the whole map is the signature: `buf_host` is cleared with `0xff` at construction and
nothing ever overwrote it.  With `ids=-1` the kernel computes `channel_x = ids[...] = -1` and reads
`x + (-1)*stride_channel_x`, i.e. one expert below the bank (the output lane offset stays inside `dst`).
Hence silent wrong values where the page below the bank is mapped, an HSA page fault where it is not.

It also explains why #56's whole kill-switch matrix (`MOE_EXPERT_CACHE_MIB=0`, `GGML_CUDA_SLAB=0`,
`GGML_SCHED_STAGE=0`, `GGML_SCHED_MOE_DUP_SPLIT=0`, `GGML_MOE_CACHE_STAGE=0`, `GGML_MOE_CACHE_MAX_TOK=1`,
`GGML_CUDA_DISABLE_FUSION=1`) was completely inert: the defect is one *missing call* feeding the upstream
engine, not any of the delivery's engines.  And why the `Host experts` row passes while only `MoE cache`
fails: the fault needs `moe_cache_size > 0`.

## 4. The fix, file by file

**`ggml/src/ggml-backend.cpp`, `ggml_backend_sched_copy_input`.**  Restore the base call, in the branch
that has **no `MUL_MAT_ID` consumer** (the slot map is read by a `GET_ROWS`, so it lands there) and after
the existing `wait_before_overwrite()` (the base position; no new wait is introduced):

```c
} else {
    wait_before_overwrite();
    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
        return;
    }
    ...
```

Placing it there rather than at the top of the function (where base has it) is deliberate: it leaves the
delivery's expert branch — device gather, cache take-over, prune-and-upload, and its intentional **absence**
of the wait on the gather path (block 09's `SCHEDSYNC 1.12 -> ~0.1 ms/call` win) — byte-for-byte
unchanged, and guarantees no host expert table can reach the callback (a cached layer's `src0` is the bank,
a device weight buffer; a delivery-served expert table has a `MUL_MAT_ID` consumer).

**`src/llama-context.cpp` / `src/llama-context.h`.**  `sched_copy_experts` now answers *only* for the
upstream cache; the duplicated per-op staging/pruning body and its `copy_experts_info` state are deleted.
That body has been dead code since the scheduler took the pruning over, and now that the callback is wired
again it **must not** survive — it would stage every host expert table a second time.

## 5. What this does NOT change (the constraint it was written under)

* **The delivery's own prefill machinery**: staging ring / op-offload H2D, device gather, expert-cache
  prefill seeding, slab — untouched.  No kernel file is in the patch (no MMQ/MMB/FA/GDN/MoE kernel change).
* **`GGML_CUDA_ASYNC_INPUT`** (delivery-owned, contributor PR #125): untouched, and structurally
  unreachable from the new code — the ring stages `set_tensor` writes into **COMPUTE** buffers, while the
  new predicate requires a `WEIGHTS`-usage **host** buffer.  It matters only because it changes *upload
  ordering*, which is why #56 surfaced as "wrong values" on one arm and "HSA fault" on the other; keep it as
  a bisect arm, not a suspect.
* **Default configs**: no effect (§1).  With `--moe-cache-mib` unset the added code is one predicate plus a
  call that returns false.
* The one intentional behavioural change: with `--moe-cache-mib` **on**, a ≤32-token ubatch (a short prompt
  is exactly that) now takes upstream's small-batch slot-map path instead of faulting.  That is upstream's
  own design on an off-by-default switch.  If the maintainer prefers the upstream engine to be
  *structurally* unable to enter a prefill-shaped batch, the small follow-up is a host gate in
  `build_moe_cache_slots` (refuse when the ubatch is not a decode-shaped batch).  **Offered and not
  taken** — it changes a graph-build path, so it was deliberately left out of this patch.

## 6. Implementation notes worth knowing before re-cutting

* The patch is cut against the **r97 tip** (`396376617`), not against block 09, so it applies as a
  standalone commit on the current tree.  Integration must **fold it into block 09** (both files are
  block-09-owned) and then regenerate the set — see §9.
* `llama_context::sched_copy_experts` in base also called `moe_cache->copy_experts()` (the large-batch arm)
  from inside the pruning loop.  The delivery's scheduler owns that pruning, so with this patch
  `copy_experts()` is again unreachable — as it already was before the patch, since the whole callback was
  dead.  See finding 1 in §8.
* The upstream `src/llama-moe-cache.cpp` is stock: the only thing any block did to it is block 00's
  `-Wshadow` parameter renames (`n_expert` -> `n_expert_l`, 14+/14−, behaviour-neutral).  The delivery's own
  cache is a different file (`ggml/src/ggml-cuda/moe-expert-cache.cu`, absent at base) behind
  `MOE_EXPERT_CACHE_MIB`.

## 7. Validation (gfx1151, Strix Halo, 1 device, ROCm 10.1.0, r97 tree + this patch)

| check | before | after |
|---|---|---|
| `test-llama-archs -a deepseek2 -s 42` | `MoE cache` row: **HSA fault** in `mul_mat_vec_f<float,float,1,64,false,true>` (the gfx1201 arm instead reads NMSE 4.02e-03) | **OK (2.10e-10)** = the upstream base's own value; all 4 rows pass |
| same, `GGML_CUDA_ASYNC_INPUT=0` | fault (the r97 gfx1201 signature) | **OK (2.10e-10)**, all 4 rows pass |
| `test-llama-archs -s 42` (whole suite) | 373/375 | **375/375** |
| 35B-A3B Q4_K_M `-sm layer -ncmoe 40 --moe-cache-mib 3072`, `-n 64` | **HSA fault** in `mul_mat_vec_q_moe<(ggml_type)12 = Q5_K, 2, false>` at the first decode | runs, coherent, **62.08 % hit rate** (7343 hits / 4485 misses, 8183 MiB uploaded) |
| 35B-A3B `-ncmoe 40` (no `--moe-cache-mib`), `-n 128` | tg 61.5 / pp 143.5 t/s | tg 61.4 / pp 144.0 t/s, **greedy text byte-identical** |
| 35B-A3B `-ncmoe 40`, 16 KB prompt (`prompts/prose-rdna-boosts.txt`) | — | pp **1805** (no cache) vs **1831** (cache); tg 50.1 vs 37.8 |

The 35B-A3B generic text (`-t 8`, `--seed 42 --temp 0 --no-display-prompt --single-turn`) was also
byte-identical in the fully-resident config (`GGML_CUDA_ASYNC_INPUT=0`, `-n 200`).

The last row is the honest characterisation of `--moe-cache-mib` **after** the patch: correct, but still
**slower than the delivery's own engine** here, because upstream's design synchronises the whole split
backend once per cached layer per pass to plan slots on the host.  Recommend documenting
`--moe-cache-mib` as "works, not recommended"; `MOE_EXPERT_CACHE_MIB` stays the path to use.

## 8. Findings to decide on (reported, not acted on)

1. **The upstream cache's large-batch `copy_experts()` arm is still unreachable.**  In base the pruning loop
   called `moe_cache->copy_experts()` for contiguous runs, so for a batch too large for the slot map
   (`n_tokens > 32`) *cached* experts were copied from the device bank instead of uploaded from the host
   master.  The delivery's scheduler owns that pruning and knows nothing about the upstream bank, so the
   arm is skipped.  **Not a correctness gap** (the bank holds the same expert bytes, so the staged copy is
   identical); it is a missing upstream optimisation on a path the delivery serves with its own cache and
   gather.  Wiring it would change the delivery's prefill path and needs its own measurement — out of scope
   here by the maintainer's constraint.
2. **`--moe-cache-mib` is correct but not competitive on this box** (§7): recommend "works, not
   recommended" in `ENVIRONMENT.md`/`README.md` rather than removing the engine.
3. **The `MoE cache, layer split` (multi-device) row is still unmeasured.**  This box has one device, and
   the upstream engine rejects `LLAMA_SPLIT_MODE_TENSOR` outright (`"MoE cache does not support tensor
   parallelism"`).  Nothing in the patch is device-count dependent (the callback is per-scheduler, closed
   over one context's `moe_cache`), but the row needs a 2-GPU host.
4. **Doc nits found on the way (not fixed here, both pre-existing):** `TODO.md`'s table still lists **#57**
   as open, though `WORKLOG.md` 261020.008 (r97) says it is closed and there is no `## 57.` body section;
   and `wip/CAMPAIGNS.md` calls `dflash-meta-split-impl/` a "TODO #58 follow-up" where #58 is closed and the
   live handle is #78.

## 9. Integration instructions (for the main development host)

The change belongs to **block 09** (host-resident MoE: iface, scheduler, Meta, slab, cache engine); both
files are block-09-owned (`ggml/src/ggml-backend.cpp`, `src/llama-context.{cpp,h}`).  Standard flow per
`AGENTS.md` -> "Regenerate the patches":

1. In a canonical `~/llama.cpp` rebuilt at `release.json.base` via `scripts/apply-all.sh`, **amend block 09**
   with this change (the r97 chain is `057ddf56f` block 09 -> `994f429e4` block 10 -> `396376617` block 11;
   `llama-context.cpp` is touched by blocks 09/10/11, so expect to resolve a small conflict in 10 if you
   rebase the plain patch rather than editing block 09 directly).
2. Re-apply blocks 10 and 11, then `scripts/make-patches.sh` -> `scripts/make-release.sh` ->
   `scripts/validate-set.sh` (strict apply on a fresh tarball, tree must match).
3. Gates on the result:
   * `test-llama-archs -s 42` — must be **375/375** on gfx1151 (the r97 baseline is 373/375, with only the
     two `MoE cache` rows failing; gfx1201 before this patch read 4.02e-03 on them instead of faulting);
   * the coherence gate on a host-expert model (`-ncmoe`, same-seed greedy diff) — must be **byte-identical**
     to r97;
   * `scripts/gate-prefill-logits.sh` — no prefill kernel is touched, so expect identical;
   * `scripts/validate-set.sh`; then the tag, the fork's `rdna-boosts` refresh and the worker container
     build at the maintainer's release discretion.
4. Then close `TODO.md` **#56** with a `WORKLOG.md` entry (newest first) and move this directory to
   `archive/work/todo-56-moe-cache-callback/` with a `RESOLUTION.md` (no `wip/` stub left behind) — per the
   WIP rules.  Fold finding 4's two doc nits into the same pass if you want them.

Do **not**: hand-edit `patches/` or `release.json`; apply this patch on top of the release set as if it were
a beta item; push anything from `~/llama.cpp` except the release-time `rdna-boosts` refresh to the personal
fork.

## 10. Repro / instrumentation (recreate if this has to be re-bisected)

```bash
# minimal repro (synthetic models, no /llm/models file needed, ~1 min per run)
test-llama-archs -a deepseek2 -s 42                  # fails: `MoE cache` row, GPU fault
GGML_CUDA_ASYNC_INPUT=0 test-llama-archs -a deepseek2 -s 42   # the other arm

# real-model repro / validation (35B-A3B Q4_K_M)
llama-cli -m .../Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 -sm layer -ncmoe 40 \
  --moe-cache-mib 3072 -p "The capital of France is" -n 64 \
  --seed 42 --temp 0 --no-display-prompt --single-turn -t 8 -lv 4
#   `-lv 4` shows the engine's own lines:
#   "MoE cache size = 3070.80 MiB for 18662.00 MiB of host experts" / "37 layers, q5_K: 1558 slots (16.4%)"
#   and, on exit, "llama_moe_cache: ubatch <= 8: hits = ... hit rate = 62.08%"
```

The instrumentation used to find it: a temporary `fprintf` in `ggml_cuda_mul_mat_vec_f`
(`ggml/src/ggml-cuda/mmvf.cu`, beside the `ncols_dst = ids ? ne2 : ne1` block) printing `dst->name`,
`src0->name`/`ne`/`nb[2]`/`ggml_nbytes`/buffer name, `ids->name`/`ne`, and the min/max of a
`hipMemcpyAsync` + `hipStreamSynchronize` readback of `ids`.  It **must be gated off during graph capture**
(a synchronize inside a capture aborts — the decode path is captured), so use it on a prefill-side launch or
with `GGML_CUDA_DISABLE_GRAPHS=1`.

## 11. Provenance of the switches involved (so the next session does not confuse them again)

| mechanism | owner | switch | where |
|---|---|---|---|
| upstream MoE expert cache — what this patch wires | **upstream** (present at `097cf33d9`) | `--moe-cache-mib` (**default off**) | `src/llama-moe-cache.cpp`, `sched_copy_experts`, `build_moe_cache_slots` |
| delivery MoE expert cache + host-expert staging/gather/slab | delivery (block 09) | `MOE_EXPERT_CACHE_MIB`, `GGML_SCHED_STAGE`, `GGML_MOE_CACHE_*` | `ggml/src/ggml-cuda/moe-expert-cache.cu` (absent at base), `ggml-cuda-slab.cu`, the expert branch of `ggml_backend_sched_copy_input` |
| async graph-input ring | **delivery** (contributor PR #125, `@briansp2020`; promoted r45 as pre-V2.0 block 16 -> blocks 09/10) | `GGML_CUDA_ASYNC_INPUT` (default on; `=0` disables) | `ggml/src/ggml-cuda/ggml-cuda.cu` |
