# RESOLUTION — `rdna2` (TODO #66 / issue #158) — CLOSED: externally validated on a real gfx1030

**Status: CLOSED 2026-10-20 on external confirmation.**  TODO #66 is removed from `TODO.md`; the closure is
recorded in `WORKLOG.md` 261020.012.  The code itself shipped earlier: the compile guards in **r70** (block
06 `mmb.cu` + block 10 `fattn-qsa3.cu`, so the image stops dying on a first-kernel `invalid kernel file`)
and `@slavap`'s tuned RDNA2 paths in **r76** (PR #165), later joined by the r88/r99 RDNA2 FA work (PR #181,
PR #198).  This file records the *platform validation* that was the only open half.

Not part of the delivery set; nothing here is applied to the fork.

## What closed it

Three separate pieces of evidence, all from a **Radeon Pro V620 (`gfx1030`, Navi 21)** owned by
`@slavap` — the maintainer's fleet still has no RDNA2 device:

1. **`test-backend-ops` on the V620** — `FLASH_ATTN_EXT` **6389/6389**, `MUL_MAT` **2444/2444**.  This is the
   "re-run both ops on the current tree" item.  Recorded with the r99 gate (PR #198 promotion) in
   `WORKLOG.md` 261020.011.
2. **End-to-end generation in a mixed-architecture tensor split.**  `@slavap`, PR #181 comment
   [6280629660](https://github.com/stew675/llama-cpp-rdna-boosts/pull/181#issuecomment-6280629660)
   (2026-10-20), r91 delivery, 2x RX 7900 XTX (`gfx1100`) + 1x V620 (`gfx1030`):

   ```
   GGML_CUDA_AR_ALLOW_NON_RDNA4=0 GGML_CUDA_ALLREDUCE=ce llama-bench -m Qwen3.8-27B-Q8_0 \
     -lm dio -b 8192 -ub 2048 -ngl 999 -fa 1 -d 0,32768 -p 2048 -n 256 \
     -sm tensor -ts 1.5/1.5/1 -dev Rocm0/Rocm1/Rocm2
   ```

   | test | t/s |
   |---|---:|
   | pp2048 | 1549.39 ± 8.20 |
   | tg256 | 30.09 ± 1.36 |
   | pp2048 @ d32768 | 1207.12 ± 9.53 |
   | tg256 @ d32768 | 28.90 ± 0.71 |

   Three devices of **two different GPU generations** in one `-sm tensor` split, `-fa 1`, both prefill and
   deep-context decode, no fault or abort.  That exercises the RDNA2 kernel set, the hybrid all-reduce
   (block 01, with `GGML_CUDA_AR_ALLOW_NON_RDNA4=0` selecting the internal path) and the mixed-arch
   scheduler/Meta split state — i.e. more than the unit ops alone prove.
3. **`-fa on` is usable.**  The r76 kv=113 partial-batch fault class that the RDNA2 add-on fixed is not
   observed: the run above uses `-fa 1` at 32K depth.

## What is NOT claimed

* **No performance claim and no purity gate.**  The numbers above are the reporter's box (all three cards on
  PCIe 4.0 x8 per the comment, so link-bound) and are quoted for "it runs correctly", not as a delivery
  figure.  Nothing on this host validates RDNA2 speed or bit-purity.
* **RDNA2 stays best-effort, not supported.**  No RDNA2 device in the maintainer's fleet, so no gate in
  `scripts/` or CI runs on it, and `README.md`/`CONTAINERS.md` keep saying so — they now say *externally
  validated* instead of *not validated on hardware*.
* **The deferred halo sparse-tile walk (`60ecdf8b + e53cee3e`) was never run and is dropped** on the
  maintainer's decision (2026-10-20).  It was the third, optional item in #66; the platform is validated
  without it.  If a sparse-tile defect ever surfaces on RDNA2, that is a new item, not a reopening of #66.

## Follow-up that is still real

`TODO.md` #72 was the last RDNA2-specific *correctness* concern from this line of work (an exact-size
operand could over-read past a natively-read K/V view).  It is **already fixed** — r99 / PR #198 gives the
`ncols2 > 1` last tile the checked (`oob_check = true`) form on RDNA2 and keeps the 511-row converted-path
guard tail, with the derivation documented
(`archive/work/rdna2-fa-followups-todo71-72/RESOLUTION.md`).  Nothing RDNA2-shaped remains open.
