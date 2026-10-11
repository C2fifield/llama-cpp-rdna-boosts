# Prebuilt ROCm containers (GHCR)

The ROCm container images are built and published **locally** by
[`scripts/build-containers.sh`](scripts/build-containers.sh)
(`ghcr.io/stew675/llama-cpp-rdna-boosts`).  A tag-push CI build took ~5 h for
two ROCm lines, and because a failed/timed-out run wrote no registry cache,
every retry started cold; the same build here is minutes-to-tens-of-minutes
and fails with a usable log.  The old matrix lives on in
[`.github/workflows/docker-ghcr.yml`](.github/workflows/docker-ghcr.yml) as a
**manual `workflow_dispatch` fallback only**.

The script does **not** rebuild or ship the patches themselves; it re-creates
the patched tree the same way the consumer workflow does:

1. download upstream `ggml-org/llama.cpp` at the fork point (read from
   `release.json`; currently `a55e952b8`)
   as a tarball (no full history) — unless the canonical tip
   (`release.json.tip`) is already in the local fork checkout, in which case it
   materialises that tip directly and verifies its git tree,
2. `git init` + one base commit (`git add -A -f`, so upstream-tracked files
   that match `.gitignore` are kept and the base tree is canonical), then
   `scripts/apply-all.sh` applies `patches/0000..0015` with strict `git am`,
3. write the build-context `.dockerignore` (from `.devops/dockerignore`) and
   build with [`.devops/rdna-rocm.Dockerfile`](.devops/rdna-rocm.Dockerfile),
   an adaptation of upstream llama.cpp's `.devops/rocm.Dockerfile`
   (adds `-DGGML_HIP_RCCL=ON`, an RDNA-only `AMDGPU_TARGETS`, and a
   `ROCM_IMAGE_SUFFIX` switch for the `-complete`/`-full` base images),
4. tag `full`, `light` and `server` per ROCm line and, with `--push`, push
   them to GHCR.

## Images

Registry path: `ghcr.io/<owner>/<repo>` (here
`ghcr.io/stew675/llama-cpp-rdna-boosts`).

| ROCm | base image | tags |
|------|-----------|------|
| 7.2  | `rocm/dev-ubuntu-24.04:7.2.4-complete`  | `rocm-7.2`, `server-rocm-7.2`, `light-rocm-7.2`, `full-rocm-7.2` *(manual dispatch only)* |
| 7.14 | `rocm/dev-ubuntu-24.04:7.14.1-full`     | `rocm-7.14`, `server-rocm-7.14`, `light-rocm-7.14`, `full-rocm-7.14` |
| 10.0 | `rocm/dev-ubuntu-24.04:10.0.0-full`     | `rocm-10.0`, `server-rocm-10.0`, `light-rocm-10.0`, `full-rocm-10.0`, `latest` |

**ROCm 7.2 is not built by the automatic (tag-push / weekly) release matrix.**  Its `-complete`
base image compiles several times more slowly than the `-full` lines, and from release r10 onward its
full image no longer fitted the build job's 6 h `timeout-minutes`; a timed-out build also never writes
the registry build cache, so every retry started cold and failed the same way, which blocked the
`release` job (it needs all builds) and left every tag release uncut.  The 7.2 line is still buildable by
a manual `workflow_dispatch` asking for `7.2`, but expect it to be slow and possibly to time out.

Each tag also has an immutable `<tag>-<fork-point>` variant pinned to the fork
point (currently `<tag>-a55e952b8`; earlier releases used
`<tag>-84e76d8a2`, `<tag>-ebbb18522`, `<tag>-d1d3c3396` and `<tag>-790cf51aa`). `rocm-<version>` is an alias of `server-rocm-<version>` (the serving
image); `latest` points at the newest ROCm (10.0) server image.

The binaries are built for `gfx1100;gfx1150;gfx1151;gfx1200;gfx1201` (RDNA3 /
RDNA3.5 / RDNA4) with runtime dispatch, so one image serves every targeted GPU
family. `server` exposes the HTTP API on `8080`, `light` is CLI-only, `full`
adds the Python conversion tooling.

### Strix Point (gfx1150)

`gfx1150` (Strix Point: Ryzen AI 9 HX 370, Radeon 890M) is part of the target
set.  It is the same RDNA3.5 family as `gfx1151` but different silicon, and it
was added after a validation on a Strix Point host (issue #88): a
`gfx1150;gfx1151` build is clean, the same-seed greedy text is byte-identical
between the native `gfx1150` code and the `gfx1151` code, and `GET_ROWS`
220/220, `MUL_MAT_ID` 931/931, `FLASH_ATTN_EXT` 6358/6358, `FLASH_ATTN_QSA`
26/26, `INDEXER_TOPK` 3/3, `HC_MIX` 30/30, `GATED_DELTA_NET` 46/46 and
`RMS_NORM` 51/51 are green.

Images published **before this change** (for example `server-rocm-10.0` built at
`2dbece4`) carry no `gfx1150` code object and fail on the first kernel launch
with:

```
ggml_cuda_compute_forward: GET_ROWS failed
ROCm error: device kernel image is invalid
  current device: 0, in function ggml_cuda_compute_forward at .../ggml/src/ggml-cuda/ggml-cuda.cu:3172
```

For those, either build from source with `gfx1150` in the target list, or make
the runtime present the device as `gfx1151` with
`HSA_OVERRIDE_GFX_VERSION=11.5.1`:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri --group-add video \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-7.14 \
  -m /models/<model>.gguf -ngl 999
```

(compose equivalent: `environment: [HSA_OVERRIDE_GFX_VERSION=11.5.1]`.)  The
override is a workaround for the older images, not a substitute for the native
target.  On the validation host `llama-bench` (9B Q8_0) measured 577.27 vs
580.30 t/s `pp512` and 10.61 vs 10.65 t/s `tg128` for native `gfx1150` vs the
`gfx1151` override, within noise.  Reported and reproduced in
[#88](https://github.com/stew675/llama-cpp-rdna-boosts/issues/88).

> **Toolchain caveat (7.2).**  ROCm 7.14.1 is the toolchain the delivery's claims
> and validation records are measured on.  ROCm 7.2 has been **reported** (not yet
> reproduced here) to break greedy purity where 7.14 is clean, see the "Separate,
> untriaged" note in `TODO.md` item 22; it was also dropped from the automatic release
> matrix because its image build outgrew the runner timeout (see the note above).
> Prefer `rocm-7.14` (or `rocm-10.0`) for speculative decoding and for anything
> compared against the recorded hashes; if a 7.2 run disagrees with a 7.14 run,
> suspect the toolchain before the delivery.

The ROCm `>= 7.14` `-full` base images do not register `/opt/rocm/lib` with the
dynamic loader (no `/etc/ld.so.conf.d` entry, no `LD_LIBRARY_PATH`), so the
image sets `LD_LIBRARY_PATH=/opt/rocm/lib` in its `base` stage — without it the
HIP backend cannot dlopen and llama.cpp reports "no usable GPU found".

## Running

```bash
# server, all RDNA families, ROCm 7.14
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-7.14 \
  -m /models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -sm tensor -mg 0
```

If the package is private, authenticate first:
`echo "$GHCR_TOKEN" | docker login ghcr.io -u <user> --password-stdin`.

## Triggering and releases

Releases are tagged by hand and the images are built locally; only the cheap
asset packaging stays in CI:

- **tag push** (`v16-<base>-r<N>`, the release identity) →
  [`.github/workflows/release.yml`](.github/workflows/release.yml) attaches
  `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and `SHA256SUMS`
  to the GitHub Release (about a minute; no compiler, no Docker).  The tag is
  checked against `release.json.release` and the `v16-<base>-r<N>` shape.
- **local image build** → `scripts/build-containers.sh --rocm "7.14 10.0"
  --push` builds and pushes the images.  Run it before or after the tag; it
  reads the tree from `release.json`, not from the tag.
- `docker-ghcr.yml` no longer runs automatically (no tag or schedule trigger);
  it is a manual `workflow_dispatch` fallback.

Ordinary commits to `main` (docs / `WORKLOG.md` / `benchmarks/`) run
[`.github/workflows/validate.yml`](.github/workflows/validate.yml) instead,
which applies the patch set and checks it against `release.json` in about a
minute.  Building the six images (2 ROCm lines x 3 targets) for a docs commit
was wasted runner time, and it let a docs push fail at `git am` when the
workflow's fork point had gone stale — the breakage this split exists to
prevent.

### Cutting a release

`release.json` is the single source of truth: the fork point, the canonical
upstream tree of that fork point, the canonical tip/tree of the applied set,
the block count, and the sha256 of every shipped artifact.  `apply-all.sh`,
`validate-set.sh` and the workflows all read it, so the fork point cannot
drift out of sync in one place while another stays stale.

To cut a release after a re-base or a block amendment:

```bash
# refresh the artifact hashes and stamp the release name (the git tag is the
# release identity; it MUST equal release.json.release)
./scripts/make-release.sh --release v16-<base-sha>-r<N>
# or, on a re-base, set all four metadata values together:
./scripts/make-release.sh \
  --release v16-<base-sha>-r1 \
  --base <new-base-sha> \
  --base-tree "$(git -C ~/llama.cpp rev-parse <new-base-sha>^{tree})" \
  --tip  <canonical-block-15-tip> \
  --tree "$(git -C ~/llama.cpp rev-parse <canonical-block-15-tip>^{tree})"

./scripts/validate-set.sh          # strict apply + tree/hash/checksum gate

# build + push the release images locally (see "Build the release container
# images" in AGENTS.md).  Do this before the tag so a broken image never gets a
# release name, or after it — the script reads the tree from release.json, not
# from the tag.
./scripts/build-containers.sh --rocm "7.14 10.0" --push

# freeze it: annotated tag on the commit that carries this release.json
git tag -a v16-<base-sha>-r<N> -m "rdna-boosts v16-<base-sha>-r<N>"
git push origin main v16-<base-sha>-r<N>
```

**Release naming.**  One tag per release, `v16-<base-sha>-r<N>`, where `N` is
that base's revision: `r1` is the release that lands the re-base and `r2`,
`r3`, ... each later release on the same base.  `release.json.release` must be
exactly the tag — `validate.yml` checks the manifest, and
[`release.yml`](.github/workflows/release.yml) refuses to package a tag that
disagrees with it or does not match `v16-<base-sha>-r<N>`.  Only a **tag push**
cuts a release; the local image build never bumps the revision and never
creates a Release.  (The historical `v16-790cf51aa` tag predates the explicit
`-rN`; it is the `r1` of its base.  Two revisions were never tagged —
`v16-790cf51aa-r5` was built by a dispatch and `v16-d1d3c3396-r1` is the
re-base — so they exist only in `release.json` history.)

The `v*` tag push runs [`release.yml`](.github/workflows/release.yml), which
attaches `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and
`SHA256SUMS` to the GitHub Release.  Consumers can pin the tag and verify the
checksums instead of tracking a moving `main`.  The tag's name is checked
against `release.json.release` before anything is packaged, so a release can
never ship under a name the manifest does not record.  The images themselves
are pushed separately by `scripts/build-containers.sh` (locally).

The `tree` field in `release.json` is the strongest check available: CI
rebuilds the patched source from a tarball and asserts the resulting git tree
is byte-for-byte the recorded canonical tree.  If it fails, the patch set
and/or the recorded base are not the delivery.

To build the images for a branch before tagging, run
`scripts/build-containers.sh` against `release.json` — it builds from the
canonical tree, so it does not need the branch to be merged or tagged.  The
manual fallback remains available via `workflow_dispatch` on
`.github/workflows/docker-ghcr.yml` (merge first, since GitHub only exposes it
once the workflow is on the default branch).

## Local build

This is the normal path — build and push the release images without CI:

```bash
gh auth refresh -h github.com -s write:packages            # once; podman needs write:packages
gh auth token | podman login ghcr.io -u stew675 --password-stdin

./scripts/build-containers.sh --prepare-only               # reconstruct + verify the release tree
./scripts/build-containers.sh --rocm "7.14 10.0" --push    # build + push the release images

# fast smoke: one ROCm line, one GPU arch, one target, no push
./scripts/build-containers.sh --rocm 10.0 --arch gfx1201 --targets server
```

`--src DIR` overrides the build-context tree (default `$HOME/llama-src`);
`--llama DIR` is the local fork checkout the canonical tip is borrowed from
(default `$HOME/llama.cpp`; when the tip is absent the script falls back to the
tarball + `apply-all.sh`).  `--arch LIST` trims `AMDGPU_TARGETS` for a fast
smoke build — the published image uses the Dockerfile's full RDNA set
(`gfx1100;gfx1150;gfx1151;gfx1200;gfx1201`).
