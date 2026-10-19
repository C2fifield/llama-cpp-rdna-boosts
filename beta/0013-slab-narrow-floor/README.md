# `0013-slab-narrow-floor` — the narrow band is measured from the wrong graph (candidate, NOT in the delivery)

**Status: 2026-10-18 — fix implemented and measured on gfx1201 (2× R9700). Staged against the r83 tree
(`v12-097cf33d9-r83`, tree `a1309717…`) to be folded into the block set after r84 lands. Nothing here is
in `patches/`, `release.json` or `rdna-boosts-all.patch`.**

| | |
|---|---|
| patch | `0013-slab-narrow-floor.patch` (one commit; `git am`-able, also `--include-beta`) |
| sha256 | `16cee76feb2e9dc52a23a0361e687ea2409c5dd5f4644c1e578499a51e5014fb` |
| cut against | `release.json` r83: base `097cf33d9`, tree `a1309717 54b5abfb 4a153e6f d1144fd5 78f1154a` |
| files | `ggml/src/ggml-cuda/ggml-cuda-slab.cu` (+39/−3), `src/llama-context.cpp` (+15/−3) |
| block amendment (when promoted) | **block 10** (creates `ggml-cuda-slab.cu`, adds `set_narrow_floor`) + **block 09** (owns `slab_narrow_boundary` / `GGML_DROP_RESERVE_SLACK_BYTES` in `llama-context.cpp`) |
| per-block diffs | `block10-slab-narrow-band.diff`, `block09-llama-context-drop-reference.diff` (named `.diff` so `--include-beta` does not apply them) |

## The failure (measured, 2026-10-18)

Field config: `Qwen3.8-Flash-Next IQ4_NL`, 2× R9700, `-sm tensor -ncmoe 46`, 131k ctx, q8_0 K/V,
`-b/-ub 6144`, MTP `draft-mtp-adaptive` n-max 6, `GGML_CUDA_ALLREDUCE=ce`, stock slab defaults
(headroom 4096 + aux 4096).  A ~74k-token prompt then a decode produced a **degenerate eviction loop**:
every ~165 ms, `moe_cache_rearm: re-armed 66 stood-down expert-cache tables` followed by
`moe_cache_evict_slab_range: evicted 3654.0 / 3664.0 MiB`, decode **17–25 t/s** instead of ~45, until it
self-terminates (18 cycles ≈ 3 s here; the reporter saw 13 s+).

**It is not new in r83** — the r82 build (`7d238ae80`) reproduces the identical 18 cycles / 36 evictions.
r83 is only a catalyst: its prefill is faster (1305 → 1506 t/s in the same run) so the same 3 s of
churn costs relatively more.

## Root cause

`ggml_cuda_slab_set_narrow_floor()` is raise-only **and returns silently while `ring_used > 0`** (the
ring's VA must not move once it holds blocks).  The floor is therefore frozen at the estimate made
before the first carve: `graph_reserve(cparams.n_rs_batch, …, split_only=true)` = **1792 MiB**.
The real decode graph — the MTP verify/draft layout — needs **2304 MiB**:

```
WORK_ALLOC dev=0 need=2304.0 MiB narrow=0 ret_off=3968.0 MiB narrow_floor=1792.0 ring=2176.0 boundary=1792.0
moe_cache_evict_slab_range: evicted 3654.0 MiB ... (arena now 29620.9 MiB)
```

so `2304 > 1792` classified a *decode* view as wide, placed it above the ring
(`narrow_floor + ring_region = 3968`), and grew the boundary to 6272 — evicting 3.65 GiB/device from the
arena.  The prefill→decode drop then re-reserved the `n_rs_batch` shape (1792) and recorded *that* as the
reference (`slab_narrow_boundary`), so the next step was again "materially wider" (512 MiB slack), and
the drop repeated every step.  Scaling confirms the MTP attribution: n-max 6 → 18 cycles, n-max 2 → 2,
`--spec-type none` → 0.

## The fix (three parts, no band-aid)

1. **`work_alloc` learns the band from the real graph.**  A view that is not a requested wide (prefill)
   pass is *narrow* even when it exceeds the floor, provided no wide view is live (`boundary <=
   narrow_floor` and the ring is disarmed — the ring region is only carved while a wide pass is live).
   The floor and ring are re-carved from that size (the ring is idle, so its VA may move).  With a wide
   view live, the old above-the-ring placement is kept — that is the invariant the `ring_used` guard
   protects.
2. **A refused raise is reported**, not silently dropped (`set_narrow_floor` logs when the ring holds
   blocks; a no-op raise stays silent).
3. **The drop reference is `max(re-reserved shape, work region observed before the drop)`**, so a
   re-reserve that is narrower than the real graph can never make every following pass drop again.

## Measured on the fix (same config unless stated)

| | before | after |
|---|---:|---:|
| decode `re-armed` cycles | 18 | **1** (the normal prefill→decode drop) |
| decode t/s (same 74k prompt, 90 tokens) | 23.6 (r82) / 24.0 (r83 s0) / 34.2 (r83 s1) | **45.4** (tensor), **35.8** (`-sm layer`) |
| prefill t/s, 74k tokens | 1506 (r83) | 1506 |
| `-ub 4096` variant | 16.4 t/s, 18 cycles | **1 cycle** |
| `--spec-type none` | 0 cycles | 0 cycles |
| band learned | 1792 MiB (never corrected) | `WORK_ALLOC need=2304 narrow=1 ret_off=0 narrow_floor=2304` |

**Cost (documented, not hidden):** a correctly sized band moves the ring up by 512 MiB, so the wide
(prefill) layout starts 512 MiB higher and the arena is 512 MiB smaller during a prefill: 16k prefill
**1443 → 1411 t/s (`-ub 4096`, −2.2 %, two runs)**, `-ub 2048` unchanged.  That is the honest price of
disjoint regions; overlapping the band into the ring VA was rejected (it is the same class of hazard as
the MTP-drafter/ring-sizing incident).

**Regression gates on the fixed build (gfx1201):** dense `Qwen3.5-4B Q8_0` coherence `1c5d32ac537d`
(unchanged); Flash-Next 16k prefill hash `a60a39e11133`, staging off/on 1083.7 → 1411.2 t/s;
prefill-logit **mean KLD −0.000000 / same-top-p 100.0 %**; adaptive-MTP 4 axes + phase-switching
**exactly equal** to r83 (R 0.66461, P 0.51005, C 0.76070, K 0.96262, phase 0.62752; mean len
4.80/2.62/5.88/6.57/2.93); `-sm layer` and `-sm tensor` both clear.

## Rebase onto r84 (when it lands)

r84 carries a block-00 fix; this candidate is cut against r83.  To fold it in:

```bash
# canonical tree at the r84 base, then the r84 block set
git clone https://github.com/ggml-org/llama.cpp ~/llama-r85 && cd ~/llama-r85
git checkout <r84 release.json base> && bash <delivery>/scripts/apply-all.sh .
# amend block 10 (slab) and block 09 (llama-context), then rebuild + re-gate
git rebase -i <block09>   # edit block 10: git apply block10-slab-narrow-band.diff
git rebase -i <block08>   # edit block 09: git apply block09-llama-context-drop-reference.diff
```

Both diffs are against the r83 tree; block 00's change is not expected to touch either hunk, but if it
does, re-cut them by hand (the two hunks are small and self-contained).

## Testing it now (without promoting)

```bash
cd <fresh apply-all checkout at r83>
git apply --check <this>/0013-slab-narrow-floor.patch && git am -3 <this>/0013-slab-narrow-floor.patch
# or: scripts/apply-all.sh --include-beta .
~/bin/build-llama-rocm
```

Then follow `BETA-TESTING.md` (the loop repro is the primary signal; the gates below guard the rest).
