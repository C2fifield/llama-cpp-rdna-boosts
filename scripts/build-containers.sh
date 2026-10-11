#!/usr/bin/env bash
# Build (and optionally push) the rdna-boosts ROCm container images LOCALLY.
#
# The release images are built on the maintainer's box, not on GitHub Actions.
# The old tag-driven CI matrix took ~5 h for two ROCm lines and, because a
# failed/timed-out build never wrote the registry cache, every retry was cold
# again.  This box has 16 cores and the ROCm toolchains are already present, so
# the same build is minutes-to-tens-of-minutes and a failure logs immediately.
#
# The script reconstructs the canonical patched tree from release.json (the
# single source of truth), asserts its git tree hash, builds the `full`,
# `light` and `server` targets per ROCm line with the SAME build args the CI
# matrix used, applies the SAME registry tags, and (with --push) pushes them to
# GHCR.  Nothing here builds on GitHub Actions; the fast CI workflows
# (validate.yml + the tag-triggered Release assets) are unaffected.
#
# Usage:
#   ./scripts/build-containers.sh [options]
#
#   --src DIR          source tree / build context (default $HOME/llama-src)
#   --llama DIR        llama.cpp checkout to borrow the canonical tip from
#                      (default $HOME/llama.cpp); used for a fast worktree
#   --image IMAGE      registry image base (default
#                      ghcr.io/stew675/llama-cpp-rdna-boosts)
#   --rocm "7.14.1 10.1" ROCm release lines to build (7.2 is manual-only/slow)
#   --arch LIST        AMDGPU_TARGETS override, e.g. gfx1201 for a fast smoke
#                      build; default = the Dockerfile's full RDNA set
#   --targets "full light server"   image targets to build
#   --push             push the built tags to the registry
#   --prepare-only     prepare + verify the source tree, then exit
#   --force            recreate --src if it is not the release tree
#   --no-cache         pass --no-cache to the container engine
#   -h, --help         this help
#
# Typical release:
#   gh auth refresh -h github.com -s write:packages   # once
#   gh auth token | podman login ghcr.io -u stew675 --password-stdin
#   ./scripts/build-containers.sh --rocm "7.14.1 10.1" --push
#
# Fast smoke (no push):
#   ./scripts/build-containers.sh --rocm 10.1 --arch gfx1201 --targets server
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"
DOCKERFILE="$REPO_DIR/.devops/rdna-rocm.Dockerfile"
DOCKERIGNORE="$REPO_DIR/.devops/dockerignore"
UPSTREAM_TARBALL="https://codeload.github.com/ggml-org/llama.cpp/tar.gz"

SRC="${LLAMA_SRC_DIR:-$HOME/llama-src}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
IMAGE="${CONTAINER_IMAGE:-ghcr.io/stew675/llama-cpp-rdna-boosts}"
ROCM_LINES="7.14.1 10.1"
ARCH=""
TARGETS="full light server"
PUSH=0
PREPARE_ONLY=0
FORCE=0
NO_CACHE=""

usage() { sed -n '2,44p' "$0"; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --src)          SRC="$2"; shift 2 ;;
        --llama)        LLAMA_DIR="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --rocm)         ROCM_LINES="$2"; shift 2 ;;
        --arch)         ARCH="$2"; shift 2 ;;
        --targets)      TARGETS="$2"; shift 2 ;;
        --push)         PUSH=1; shift ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        --force)        FORCE=1; shift ;;
        --no-cache)     NO_CACHE="--no-cache"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v jq >/dev/null || die "jq is required"
[ -f "$MANIFEST" ] || die "$MANIFEST not found"
[ -f "$DOCKERFILE" ] || die "$DOCKERFILE not found"
[ -f "$DOCKERIGNORE" ] || die "$DOCKERIGNORE not found"

if command -v podman >/dev/null; then
    ENGINE=podman
elif command -v docker >/dev/null; then
    ENGINE=docker
else
    die "neither podman nor docker is installed"
fi

base="$(jq -r .base "$MANIFEST")"
tip="$(jq -r .tip "$MANIFEST")"
tree="$(jq -r .tree "$MANIFEST")"
n_blocks="$(jq -r .n_blocks "$MANIFEST")"
for x in base tip tree; do
    v="$(eval "echo \$$x")"
    [ -n "$v" ] && [ "$v" != "null" ] || die "release.json is missing .$x"
done

echo "==> release ${base} -> tip ${tip}, ${n_blocks} blocks, tree ${tree}"
echo "==> engine=${ENGINE}  image=${IMAGE}  src=${SRC}"

remove_src() {
    if [ -e "$SRC/.git" ] && [ -d "$LLAMA_DIR/.git" ]; then
        git -C "$LLAMA_DIR" worktree remove --force "$SRC" 2>/dev/null && return 0
    fi
    rm -rf "$SRC"
}

verify_src() {
    local have
    have="$(git -C "$SRC" rev-parse HEAD^{tree} 2>/dev/null || true)"
    [ "$have" = "$tree" ] || die "source at $SRC has tree ${have:-<none>}, expected $tree"
}

prepare_src() {
    if [ -d "$SRC" ]; then
        if git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
            local have
            have="$(git -C "$SRC" rev-parse HEAD^{tree} 2>/dev/null || true)"
            if [ "$have" = "$tree" ]; then
                echo "==> reusing prepared source at $SRC (tree $tree)"
                return 0
            fi
            [ "$FORCE" -eq 1 ] || die "$SRC is not the release tree (have ${have:-none}); --force to recreate"
        else
            [ "$FORCE" -eq 1 ] || die "$SRC exists and is not a git checkout; --force to recreate"
        fi
        echo "==> removing stale $SRC"
        remove_src
    fi

    if [ -d "$LLAMA_DIR/.git" ] && git -C "$LLAMA_DIR" cat-file -e "${tip}^{commit}" 2>/dev/null; then
        echo "==> creating worktree $SRC at $tip (borrowed from $LLAMA_DIR)"
        git -C "$LLAMA_DIR" worktree add --detach "$SRC" "$tip"
    else
        echo "==> reconstructing $SRC from the ${base} tarball + apply-all.sh"
        mkdir -p "$SRC"
        curl -fsSL -o "$SRC/.llama-base.tar.gz" "${UPSTREAM_TARBALL}/${base}"
        tar -xzf "$SRC/.llama-base.tar.gz" -C "$SRC" --strip-components=1
        rm -f "$SRC/.llama-base.tar.gz"
        git -C "$SRC" init -q .
        git -C "$SRC" add -A -f
        git -C "$SRC" -c user.email=ci@rdna-boosts.invalid -c user.name="rdna-boosts" \
            commit -q -m "llama.cpp ${base} (base)"
        RDNA_BRANCH=rdna-boosts bash "$REPO_DIR/scripts/apply-all.sh" "$SRC"
    fi
    verify_src
}

rocm_full() { case "$1" in 7.2) echo 7.2.4;; 7.14.1) echo 7.14.1;; 10.1) echo 10.1.0;; *) return 1;; esac; }
rocm_suffix() { case "$1" in 7.2) echo complete;; 7.14.1|10.1) echo full;; *) return 1;; esac; }

validate_rocm() {
    local v
    for v in $ROCM_LINES; do
        rocm_full "$v" >/dev/null || die "unsupported ROCm release line: $v (7.2, 7.14.1, 10.1)"
        if [ "$v" = "7.2" ]; then
            echo "WARNING: ROCm 7.2 (-complete) builds are much slower; expect a long build" >&2
        fi
    done
}

prepare_src
cp -f "$DOCKERIGNORE" "$SRC/.dockerignore"
echo "==> wrote build-context .dockerignore"

if [ "$PREPARE_ONLY" -eq 1 ]; then
    echo "==> prepare-only: $SRC is ready (tree $tree)"
    exit 0
fi

validate_rocm

avail_kb="$(df -Pk "$SRC" | awk 'NR==2 {print $4}')"
if [ "${avail_kb:-0}" -lt 60000000 ]; then
    echo "WARNING: only $((avail_kb/1024/1024)) GiB free on the source filesystem; the two full lines plus build layers can need ~80-120 GiB" >&2
fi

BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

tags_for() { # $1=target  $2=rocm-line
    case "$1" in
        full)   echo "$IMAGE:full-rocm-$2 $IMAGE:full-rocm-$2-$base" ;;
        light)  echo "$IMAGE:light-rocm-$2 $IMAGE:light-rocm-$2-$base" ;;
        server)
            local t="$IMAGE:server-rocm-$2 $IMAGE:server-rocm-$2-$base $IMAGE:rocm-$2"
            [ "$2" = "10.1" ] && t="$t $IMAGE:latest"
            echo "$t" ;;
        *) die "unknown target: $1" ;;
    esac
}

for v in $ROCM_LINES; do
    rocm_version="$(rocm_full "$v")"
    suffix="$(rocm_suffix "$v")"
    for t in $TARGETS; do
        tags="$(tags_for "$t" "$v")"
        targs=()
        for tag in $tags; do targs+=(-t "$tag"); done

        barg=(
            --build-arg "ROCM_VERSION=$rocm_version"
            --build-arg "AMDGPU_VERSION=$rocm_version"
            --build-arg "ROCM_IMAGE_SUFFIX=$suffix"
            --build-arg "BUILD_DATE=$BUILD_DATE"
            --build-arg "APP_VERSION=$base"
            --build-arg "APP_REVISION=$base"
        )
        [ -n "$ARCH" ] && barg+=(--build-arg "ROCM_DOCKER_ARCH=$ARCH")

        echo
        echo "==> [$v/$t] building (ROCm $rocm_version-$suffix${ARCH:+, arch $ARCH})"
        echo "    tags: $tags"
        time "$ENGINE" build $NO_CACHE \
            -f "$DOCKERFILE" --target "$t" \
            "${barg[@]}" "${targs[@]}" "$SRC"

        if [ "$PUSH" -eq 1 ]; then
            for tag in $tags; do
                echo "==> pushing $tag"
                "$ENGINE" push "$tag"
            done
        fi
    done
done

echo
if [ "$PUSH" -eq 1 ]; then
    echo "==> pushed:"
else
    echo "==> built (not pushed; pass --push to publish):"
fi
for v in $ROCM_LINES; do
    for t in $TARGETS; do
        for tag in $(tags_for "$t" "$v"); do echo "    $tag"; done
    done
done
