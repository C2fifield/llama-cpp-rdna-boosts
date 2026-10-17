# Cold-start prompt — unified-KV cell-offset non-determinism (TODO #52 part C)

> **PARKED 2026-10-11.** The campaign is parked, not active: the residue is upstream-inherent and a
> proper fix is an upstream unified-KV redesign, so it is documented as a gotcha instead. Do **not**
> resume this prompt unless the maintainer explicitly reopens it. See the disposition block in
> [`README.md`](README.md) and [`FINDINGS-offset-mechanism.md`](FINDINGS-offset-mechanism.md).

Paste the block below into a fresh session. It is self-contained; the campaign README has the full detail.

---

You are picking up the residual multi-sequence non-determinism in `/home/stew675/llama-cpp-rdna-boosts`
(the rdna-boosts delivery repo for the `stew675/llama.cpp` fork), tracked as **TODO #52 part C** and
reported upstream as issue #134. Parts A and B are **fixed and delivered** (`v16-a55e952b8-r42` and
`-r43`); do not reopen them. This is the last piece.

**The symptom.** `llama-server -np 2 --kv-unified`, greedy (`temperature 0, top_k 1, seed 1,
cache_prompt false`), two concurrent requests: identical repetitions are not byte-reproducible, and a
concurrent run differs from the same request run solo. `--no-kv-unified` is fully green.

**The finding already established.** It is the **unified KV cache** and nothing else. It reproduces on
stock upstream `a55e952b8`, on two different model files, with all experts on GPU (`-ncmoe 0` and with no
`-ncmoe` at all), with `-sm layer` and `-sm tensor`, and with the expert cache off, GPU-thread changes,
`LLAMA_KV_N_PAD_MIN`, the derived kq mask and the mmvq band all neutral. It is **not** model-, quant-,
expert-cache-, split- or fork-specific. And the mechanism is now pinned: **a sequence's attention result
depends on the physical offset of its KV cells in the shared stream, modulo the flash-attention reduction
tile (quantum 64)** — a companion fill shifts seq0's cells; at a multiple of 64 seq0 is byte-identical, at
100 or 300 it changes at the rounding level (`3c074b69bddb612b` / `8509e85d4a2343b2` vs `375b3dce0622dce5`).

**Your job.** Turn that mechanism into a **performance-neutral fix** — shape-uniform kernels with
cross-sequence batching and unified KV both intact, not a feature stand-down. Likely directions
(anchoring the FA tile grid to the sequence, per-sequence contiguous/aligned cell ranges, or an
offset-invariant reduction) are in the campaign README §4; pick one, implement it in WIP, and validate.

Read `wip/kv-cell-offset-nondeterminism/README.md` **first, in full**, then `AGENTS.md`
("Report findings before shipping", "Pushing policy", "WIP and promotion rules", "Default-on policy").
The predecessor campaign is archived at `archive/work/moe-verify-fusions/` — read its
`FINDINGS-B-readback-ordering.md` (the offset sweep and the condition matrix) and `HANDOVER-multiseq-bc.md`.

Reproduce on a canonical build (any recent release tree; `release.json` names the base/tip, or a fresh
`release.json.base` + `scripts/apply-all.sh`, then
`BUILD_DIR=build-rocm-hybrid ~/bin/build-llama-rocm`). The
server reproducer is `archive/work/moe-verify-fusions/tools/multiseq.sh` (and its `_plain` / `_np2`
variants; note it never passes `-np`, so the server auto-sets `-np 4` and forces `kv_unified=true`). The
in-process offset probe is `archive/work/moe-verify-fusions/tools/test-multiseq-same.cpp` (copy into
`tests/`, add `llama_build(test-multiseq-same.cpp)`); its `MSQS_ONLY0`/`MSQS_BFILL` sweep is the fastest
way to see the effect.

Report the root cause and the proposed fix to the maintainer **before** any
`patches/`/`release.json`/docs-header change, and wait.

Constraints: WIP stays WIP; never push to upstream `ggml-org/llama.cpp` (only this repo and the personal
fork `git@github.com:stew675/llama.cpp.git`); no parallel GPU benches, one arm at a time; `llama-cli`
needs `--single-turn`; `scripts/gate-prefill-logits.sh` and coherence `1c5d32ac537d` must stay green.
