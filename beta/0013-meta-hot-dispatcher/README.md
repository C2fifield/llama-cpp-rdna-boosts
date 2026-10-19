# `0013-meta-hot-dispatcher` — BETA candidate: pre-#170 hot path with a cold alias path

**Status: STAGED AND DECIDED (2026-10-18).**  Maintainer decision: **fold into block 00 at the next
revision** (no standalone release; the promotion gates run with that revision).  This is a pure code
motion in the block-00 `ggml/src/ggml-backend-meta.cpp` hunk — no behaviour change — so it needs no
external beta test; it is here because it edits a file block 00 owns and must not sit in `wip/`.

| | |
|---|---|
| patch | `0013-meta-hot-dispatcher.patch` |
| sha256 | `d4303170c5e0c50defbc113c9d98beeb32026549d838f3e5e9c036fb353ff727` |
| cut against | release tree `f6f55ffc8628792e6b0d8a6a2988d80641b187d9` (`v12-097cf33d9-r86` candidate) |
| applied tree (this patch alone) | `82360b483ca2de0a9c99703e2335b4d8f19e0b97` |
| applied tree (0012 + 0013) | `bb8258610763da19d03d375865838aac7e79ed30` |
| applies | strict `git am`, blocks 12/12 + beta 12 + beta 13, verified on a fresh `--include-beta` checkout |

## What it does

`ggml_backend_meta_get_split_state(const ggml_tensor *, bool)` goes back to the **exact pre-#170 shape**
— `if (ggml_backend_buffer_is_meta(tensor->buffer))` first, then the buffer context and the recursive
call, nothing else — and everything the issue #170 fix added moves into a `static
__attribute__((noinline, cold))` helper that only host tensors and the unallocated-alias case reach:

* the view-chain buffer resolution (`ggml_backend_meta_view_chain_buffer`);
* the host-buffer test (`ggml_backend_buffer_is_host`);
* the `[TAG_META_HOST_VIEWS]` assert (now `tensor->buffer == nullptr || ...`, i.e. it is not asserted on
  the unresolved-alias path, exactly as before).

Same three outcomes, same order, same assert condition — the #170 fix (issue #170: `-sm tensor` with a
host layer; the `ggml_view_4d` -> `ggml_permute` alias) is preserved.  Provenance: `WORKLOG.md`
261018.010 / `archive/work/issue-184-meta-r80r82/`.

## Why (issue #184)

`ggml_backend_meta_get_split_state` is on the graph-allocation path and runs on every tensor of every
graph allocation (`-sm tensor`, ~10k lookups per decode step in the field config).  The r82 #170 fix put
a null test plus an `is_host()` evaluation *in front of* the classification; `is_host()` on a meta buffer
type walks every sub-buffer type, so it is two indirect calls per lookup.  `@briansp2020` reported that as
a 3.5 % loss at 4 concurrent decode streams (issue #184).  It did **not** reproduce here — 224 bytes of
code, ≤0.4 % with no stable sign, text identical (`archive/work/issue-184-meta-r80r82/`) — but the
arrangement is the right one regardless: the hot path costs one comparison, and the #170 path costs a
branch.

## Measured

* 2x R9700 (gfx1201), ROCm 10.1, r86-candidate tree, **only `libggml-base.so` swapped**, 4 concurrent
  greedy streams with pre-warmed prompt caches (GSQ-IQ3_XXS + shared MTP n3, `-sm tensor -ncmoe 47`):
  **-0.06 %** vs the delivery tree, i.e. nothing measurable; greedy text identical.
* Code: the dispatcher drops from **454 B to 287 B** (below the 385 B of the pre-#170 build) and the
  `noinline, cold` helper is a separate 164 B function outside the hot path.

## Promotion checklist (run with the next revision, not now)

1. Fold into `patches/0000` as a block-00 amendment (or keep the hunk in block 00's commit body), keep the
   `[TAG_META_UNALLOC_ALIAS]` wording so the tag grep still finds the #170 story.
2. Re-cut `upstream/UPSTREAM-PR-meta-split-unalloc-alias-and-alloc-fail.patch` and its `.md` (the staged
   upstream PR carries the same function), and confirm it still applies to `097cf33d9` and builds.
3. `scripts/make-patches.sh` + `scripts/make-release.sh`; `scripts/validate-set.sh` green on a fresh
   tarball (strict 12/12, tree matches).
4. Coherence: same-seed greedy text byte-identical to the previous revision (dense 4B Q8_0 `-sm tensor`
   and the MoE `-ncmoe` config) — the 4-stream harness above already matched every stream's text.
5. `scripts/gate-prefill-logits.sh` (expect mean KLD 0 / same-top-p 100 %: no kernel or approximate path
   is touched; run it anyway because the graph allocator path is inline).
6. The **issue #170** regression check: the 24 GiB-parity reproduction
   (`archive/work/issue-170-meta-alloc/`) — the alias path must still be resolved, not asserted, and a
   genuine KV/compute/weight OOM must still fail cleanly with no VRAM leak.
7. Adaptive-MTP 4-axis + phase-switching gate (decode path change by association), and
   `llama-batched-bench -npl 1,4,8` stock-relative.
8. Then delete this directory (a promoted candidate leaves `beta/`; no stubs) and record it in
   `WORKLOG.md`.

## Apply it (if a tester wants it before then)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 097cf33d9
bash <delivery-repo>/scripts/apply-all.sh --include-beta .
# 14 commits (blocks 00-11, beta 12, beta 13); the tree is NOT release.json's and that is expected
```
