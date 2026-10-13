# BETA-TESTING - FA prefill staging arena preallocation (issue #138)

This is the beta test checklist for `fa-stage-prealloc.patch`.  It is NOT part of the delivery.  We
would like you to apply it on top of the current delivery, rebuild, and re-run your Windows q8_0
prefill measurement.  The patch is at:

```
beta/fa-stage-prealloc/fa-stage-prealloc.patch
sha256 f1a0f0b1c87dc77e6bce796af94b63262c109dce5a8a93d09685d262137d7f2a
```

It is cut against release `v16-a55e952b8-r48` (tree `3b9f403544169544f312bd24e4de31a554bf189d`)
and verified to apply clean on that tree.  It changes only `ggml/src/ggml-cuda/common.cuh` and
`ggml/src/ggml-cuda/fattn-common.cuh`.

## 1. Get the current delivery tree

```
git clone https://github.com/stew675/llama.cpp
cd llama.cpp
git checkout rdna-boosts
git rev-parse HEAD
```

The `rdna-boosts` branch is the r48 delivery (the fork is refreshed at every release).  If your
`HEAD` is not the r48 tip, run `scripts/apply-all.sh` from the delivery repo on a clean checkout at the
`release.json.base` commit instead (see the delivery README).

## 2. Apply the beta patch

`git am` (keeps a commit, recommended):

```
git checkout -b beta-fa-stage-prealloc
git am /path/to/fa-stage-prealloc.patch
```

or `git apply` (leaves the change uncommitted):

```
git apply /path/to/fa-stage-prealloc.patch
```

The applied tree should be `24fdc3c7f641e7b37a0d4b0a6d4024622bde94ce`.

## 3. Build

Build as usual (your Windows HIP toolchain).  The patch needs a full rebuild of the `ggml-hip`
backend because both changed files are headers.

## 4. Run your measurement

Use your existing 119k to 164k single-request q8_0 prefill case with `-np 2` and unified KV.  For each
run, record the same dedicated and committed VRAM counters you used before, plus pp and MTP
acceptance.

A/B it in the one binary with the kill switch:

* prealloc on (the new default): run normally.
* prealloc off (the old ladder): `GGML_CUDA_FA_STAGE_PREALLOC=0`.

Expected direction, from your own report: the prealloc run should leave roughly one arena's worth of
committed VRAM (about 512 MiB plus the other ~80 MiB) instead of the multi-GiB ladder growth, with pp
at or slightly above the old number.  The greedy output must be unchanged.

Also worth repeating, if you have time:

* the largest context that no longer spills into shared memory with `-np 2`.
* `GGML_CUDA_FA_STAGE_MAX_MB=0` (unbounded) and `GGML_CUDA_FA_STAGE_MAX_MB=1` as smoke checks; both
  should complete with the same greedy output.

## 5. What to report back

On the issue, please include:

* the delivery release and the `git rev-parse HEAD` of your checkout,
* the sha256 of the patch you applied,
* the dedicated and committed VRAM growth for both arms (prealloc on and `GGML_CUDA_FA_STAGE_PREALLOC=0`),
* pp and MTP acceptance for both arms,
* whether greedy output is unchanged,
* any warning the run logged (the arena logs a one-time message if it falls back to the native read).

## How it works (brief)

`launch_fattn` now passes the F16 size of the K/V view roots as a hint, capped per operand by
`GGML_CUDA_FA_STAGE_MAX_MB`.  The arena allocates that size once instead of walking the 1.25x ladder.
The hint is only tried when it fits the existing free-VRAM cap (the slab guard), and the patch falls
back to the old growth step on a refusal or an allocation failure, so nothing about the previous
behaviour is lost.  `GGML_CUDA_FA_STAGE_PREALLOC=0` disables the hint.
