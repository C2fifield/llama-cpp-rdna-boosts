# `archive/work/rdna2` — RDNA2 (`gfx1030`) bring-up (compile guards, promoted into r70)

**Status:** **PROMOTED into r70 (2026-10-16)** — folded into blocks 06 and 10, and `gfx1030` added to
`ROCM_DOCKER_ARCH`; `patches/`, `release.json` and `rdna-boosts-all.patch` now carry it.  This
directory is the archived record, and `rdna2.patch` is kept for provenance only (cut against the r69
tree before the fold).

**Follow-on (r76):** `@slavap`'s tuned RDNA2 add-on (PR #165) was folded into blocks 05 and 07, and
the original r68-based add-on files are archived under [`pr165/`](pr165/README.md).

## What

Two compile guards that let the delivery build for `gfx1030` (RDNA2):

* `ggml/src/ggml-cuda/mmb.cu` — the `mmb_wmma_bf16` / `mmb_wmma_f16` helpers (block 06).
* `ggml/src/ggml-cuda/fattn-qsa3.cu` — the `qsa3_wmma_f16` helper (block 10).

Each "portable WMMA dispatch" helper ends `#if defined(RDNA4) ... #else <gfx11 builtin> #endif`.  On
`gfx1030` the `#else` arm is taken and the first-gen `__builtin_amdgcn_wmma_*_w32` intrinsic is not
available:

```
fattn-qsa3.cu:68:12: error: '__builtin_amdgcn_wmma_f32_16x16x16_f16_w32' needs target feature wmma-256b-insts,wavefrontsize32
```

The patch changes `#else` to `#elif defined(RDNA3)` and adds an `#else return c;` arm.  The RDNA2 arm
is **dead**: both kernels are runtime-gated off on RDNA2 (`mmb_enabled()` and
`ggml_cuda_flash_attn_qsa3_supported()` return false for `gfx1030`).  RDNA3/RDNA4 preprocessed source
is unchanged, so their codegen is byte-identical.

The full `gfx1030` compile was audited here (build-only, `/opt/rocm-7.14.1-gfx120X` clang):
**these two TUs were the only failures**; every other HIP TU compiles clean and `libggml-hip.so`
carries only `amdgcn-amd-amdhsa--gfx1030` code objects.

## Why

Issue #158.  The published container is built for `gfx1100;gfx1150;gfx1151;gfx1200;gfx1201` and has no
`gfx1030` code object, so every kernel launch on an RDNA2 device fails at the first op:

```
ROCm error: invalid kernel file
  current device: 2, in function ggml_cuda_kernel_launch at .../ggml-cuda/common.cuh:2527
  hipGetLastError()
```

It surfaces in `ggml_cuda_op_rms_norm_fused` only because that is the first kernel `ggml_cuda_try_fuse`
launches — not because the fused norm is itself broken.  Adding `gfx1030` to `ROCM_DOCKER_ARCH` fixes
the image, but the image cannot be *built* for `gfx1030` until these two guards are in the delivery
set, so the two changes land together.

## Patch

* `rdna2.patch` — sha256 `35551d2b01042f0bd560beec8d01f7a950b567301c6d8ba5e22c2d62065dd21e`
* base: `release.json` tree `9e2cdef733e2afee7e3eb54151593e1ac3fdfdb2` (r69)
* applied tree: `592a097cd6cef3ef1d179f1d79441fa798a88439` (verified with both `git apply` and `git am` on top of `ee1fe82f2`)
* mailbox-format, so it applies with `git apply` and `git am` on a fresh `scripts/apply-all.sh` checkout.

## Promotion (done)

Folded into **block 06** (`mmb.cu`) and **block 10** (`fattn-qsa3.cu`) for **r70** (tip
`2afc3a04135bf0057a61e76054ca56eb98dc1015`, tree `592a097cd6cef3ef1d179f1d79441fa798a88439`), together
with the `gfx1030` line in `.devops/rdna-rocm.Dockerfile` and the README/CONTAINERS notes.  It was
folded on the compile audit alone (it is a compile prerequisite and behaviour-preserving on
RDNA3/RDNA4); the external window validates the *resulting RDNA2 platform*, not the guard.  That
validation is `BETA-TESTING.md` and issue #158.
