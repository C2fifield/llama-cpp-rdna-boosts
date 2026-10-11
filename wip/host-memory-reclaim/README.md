# `host-memory-reclaim` — reclaiming the host memory of GPU-resident weights (and the `-ncmoe` pool)

**Status: OPEN (seeded 2026-10-11, re-scoped from `archive/work/host-memory-footprint/`).**  This is a
`wip/` tree; nothing here is part of the delivery (`patches/`).  It applies only to `~/llama-fingon`
(campaign) / `~/llama.cpp` (delivery) on `fingon` (RX 7900 XTX, 24 GiB VRAM, ~30.5 GiB RAM, gfx1100,
ROCm 7.14), which is never pushed.

This campaign is the open remainder of the archived
[`../../archive/work/host-memory-footprint/`](../../archive/work/host-memory-footprint/README.md)
(seed measurement 2026-10-02).  Two of its hypotheses were overtaken by the delivery; the rest are still
unmeasured.

## What the delivery already changed (do not re-litigate)

* **r33:** the pageable `--host-experts mmap` mode and the legacy `LLAMA_MMAP_HOST_EXPERTS=0` alias were
  **removed** (issue #116).  gfx1201 reports `XNACK enabled: NO`, so a kernel cannot read a pageable host
  master (fatal "page not present"), and the pageable 2-D H2D fill faults on ROCm 7.14.  The old
  mitigation "keep a staging ring and read the master from the pageable mmap" is therefore **dead**.
* **r35:** the replacement shipped (from `archive/work/host-expert-dio-cache/`): the **bounded pinned
  `--host-experts pool`** (`MOE_HOST_POOL_MIB`, default `MOE_HOST_POOL_FRAC` = 25 % of the MoE host
  expert bytes), a *budgeted* pinned host tier instead of the full-master `ROCm_Host` allocation.  This
  is the direct answer to hypothesis 2 (the unreclaimable ~18.7 GB pinned `RssShmem` pool).

So the remaining question is narrower: **given a pinned master/pool, how much host RAM is still held by
weights that are already GPU-resident, and can it be reclaimed safely?**

## The phenomenon (seed measurement, 2026-10-02)

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.66 GB), gfx1100, `-ngl 99 -fa 1 -t 8 -p 256 -ub 256`, peak RSS:

| config | peak `VmRSS` | `RssShmem` | `RssFile` | `RssAnon` | `MemFree` |
|---|---:|---:|---:|---:|---:|
| `-ncmoe 0` (all GPU) | 16.8 GB | ~0.02 GB | 16.4 GB | 0.2 GB | — |
| `-ncmoe 40` (all experts host) | **29.5 GB** | **18.68 GB** | ~10.7 GB | ~0.1 GB | **~0.25 GB** |

`kswapd0` churns though `Mlocked: 0` / `VmLck: 0` everywhere.  The resident set is host pool (18.7) +
touched mmap (`RssFile` 10.7) ≈ 29.4 GB on a 30.5 GB box.

## Open work (in order)

1. **Hypothesis 1 — GPU-resident weights keep their mmap pages.**  After the loader copies a weight to
   VRAM, the backing mmap pages stay in `RssFile` page cache.  Test an
   `madvise(MADV_DONTNEED)` / `posix_fadvise(POSIX_FADV_DONTNEED)` on the mapped ranges of tensors that
   are **GPU-resident** once loading is complete (candidate home: end of `llama_model_load` /
   `llama-mmap.cpp`).  Quantify the `RssFile` drop and confirm the pages are re-read (from disk) if the
   tensor is ever needed host-side — i.e. that no in-place GPU read depends on the mmap (it does not for
   GPU-resident tensors, but confirm per path, especially the MTP head / `output`).
2. **Re-measure the field with `--host-experts pool`.**  Run the seed matrix on `fingon` with
   `--host-experts pool` (`MOE_HOST_POOL_MIB` 25 % vs explicit) vs `pinned`, and compare peak `VmRSS` /
   `RssShmem` / `RssFile` / `VmLck` / `MemFree` and kswapd CPU.  Does the bounded pool remove the
   starvation?  What is the prefill/decode cost on gfx1100?
3. **Hypothesis 3 — host/GPU duplication.**  Confirm per-tensor that `token_embd` / `output` / router
   weights are not held host-side in a second buffer (`RssAnon` looked tiny, so this is expected to be a
   no-op — verify, do not assume).
4. **Hypothesis 4 — `--no-mmap` / `--mlock` accounting.**  Confirm they move resident pages to
   unreclaimable `RssAnon`, and document the resulting footprint.
5. **Decide:** if (1) is a safe, measurable win, propose a loader-side reclamation with a kill-switch and
   the full gate set; otherwise park with the numbers.

## Method / repro

```sh
cd ~/llama-fingon     # or ~/llama.cpp for the delivery build
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1100/lib HIP_VISIBLE_DEVICES=0
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# monitored load: samples /proc/<pid>/status + /proc/meminfo; dumps the largest smaps region at peak
./memwatch.sh 60 ./build-rocm/bin/llama-bench -m "$MQ4" -ngl 99 -ncmoe 40 -fa 1 -t 8 \
    -p 256 -n 0 -b 256 -ub 256 -r 1 -o jsonl
# now also with --host-experts pool (MOE_HOST_POOL_MIB unset / 4096 / 0) and --no-mmap / --mlock
```

`memwatch.sh` lives in the archived campaign
([`../../archive/work/host-memory-footprint/memwatch.sh`](../../archive/work/host-memory-footprint/memwatch.sh)).

| axis | values |
|---|---|
| `-ncmoe` | 0, 12, 40 |
| host experts | `--host-experts pinned` (default) vs `pool` (`MOE_HOST_POOL_MIB` 25 % / explicit) |
| mmap | default (mmap) vs `--no-mmap` |
| locking | default vs `--mlock` |
| cache | `MOE_EXPERT_CACHE_MIB` unset vs armed |

Record per config: peak `VmRSS` / `RssAnon` / `RssFile` / `RssShmem` / `VmSwap` / `VmLck`,
`MemFree`/`Cached` at peak, kswapd CPU (`pidstat -p $(pgrep kswapd0) 1`), and the largest
`/proc/<pid>/maps` regions with backing.

## Gates

- `scripts/validate-set.sh`; coherence `1c5d32ac537d` (Qwen3.5-4B-Q8_0, seed 42);
  `scripts/gate-prefill-logits.sh` (KLD <= 0.005 / same-top-p >= 98 %) if the loader is touched;
  `MUL_MAT_ID` 931/931; `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (decode must not regress).
- gfx1100 (`fingon`) is the target box; gfx1201 coherence is the portability check.

## Pointers

- Archived seed campaign: [`../../archive/work/host-memory-footprint/`](../../archive/work/host-memory-footprint/README.md).
- The pool: `archive/work/host-expert-dio-cache/` (PROMOTED, r35); `ENVIRONMENT.md` §1.2.1; r33's
  pageable removal and r35's pool are in `WORKLOG.md` 2026-10-08 (r34/r35).
- `ENVIRONMENT.md` §"host experts" for the current `--host-experts` modes.
