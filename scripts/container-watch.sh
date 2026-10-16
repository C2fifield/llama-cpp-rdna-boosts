#!/usr/bin/env bash
# Release container-image watcher for a dedicated build worker (e.g. `radiance`).
#
# Why this exists: the ROCm release images are built and pushed LOCALLY, not in CI
# (a tag-driven matrix took ~5 h and a failed run wrote no cache).  On the primary
# box that ties up the machine for tens of minutes per release.  This script lets a
# separate, beefier worker poll the delivery repo for a new release tag and run
# `scripts/build-containers.sh --push` in the background (cron or a systemd timer).
#
# It fetches tags, finds the newest `v12-*` release BY TAG CREATION DATE, and if it
# differs from the last successfully pushed tag, checks that tag out and runs the
# tag's own `scripts/build-containers.sh`.  The creation-date sort matters: a
# version sort (`-v:refname`) mis-orders tags across a re-base, because the base SHA
# is part of the tag name (`v12-<base>-rN`) and the next base need not sort higher.  A `flock` prevents two builds at once and a state
# file stops a tag being rebuilt.  See CONTAINERS.md, "Offload the image build".
#
# Env (all optional):
#   RDNA_REPO     delivery repo checkout      (default $HOME/llama-cpp-rdna-boosts)
#   LLAMA_DIR     llama.cpp checkout          (default $HOME/llama.cpp; passed as LLAMA_CPP_DIR)
#   STATE_DIR     state + logs                (default $HOME/.cache/rdna-containers)
#   ROCM_LINES    ROCm release lines          (default "7.14.1 10.1")
#   BUILD_EXTRA   extra args for build-containers.sh (e.g. "--no-cleanup")
#   FORCE_TAG     build this tag even if it equals the last-built marker
#
# Exit: 0 = nothing to do or a build succeeded; 1 = a build/push failed.
set -euo pipefail

# cron has a minimal PATH and no login session; rootless podman needs XDG_RUNTIME_DIR.
export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# GHCR credential: keep a PERSISTENT copy so cron never depends on the login
# session's temporary $XDG_RUNTIME_DIR/containers/auth.json (it is lost on
# logout/reboot).  Refresh it from the gh token on every run.  `gh auth status`
# must be authenticated with write:packages (push) and delete:packages (prune).
export REGISTRY_AUTH_FILE="${REGISTRY_AUTH_FILE:-$HOME/.config/containers/auth.json}"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    mkdir -p "$(dirname "$REGISTRY_AUTH_FILE")"
    gh auth token | podman login --authfile "$REGISTRY_AUTH_FILE" ghcr.io \
        -u "${GHCR_USER:-stew675}" --password-stdin >/dev/null 2>&1 || true
fi

RDNA_REPO="${RDNA_REPO:-$HOME/llama-cpp-rdna-boosts}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/rdna-containers}"
ROCM_LINES="${ROCM_LINES:-7.14.1 10.1}"
BUILD_EXTRA="${BUILD_EXTRA:-}"
FORCE_TAG="${FORCE_TAG:-}"

log() { echo "$(date -Is) $*"; }

[ -d "$RDNA_REPO/.git" ] || { log "ERROR: not a git checkout: $RDNA_REPO"; exit 1; }
mkdir -p "$STATE_DIR"

# One build at a time.
exec 9>"$STATE_DIR/.lock"
if ! flock -n 9; then
    log "another watcher/build is running; skipping"
    exit 0
fi

cd "$RDNA_REPO"

if [ -n "$FORCE_TAG" ]; then
    tag="$FORCE_TAG"
else
    git fetch --quiet --tags origin
    # -creatordate, not -v:refname: the base SHA is in the tag name, so a re-base to a
    # "lower" base would otherwise be missed.  Release tags are annotated, so this is
    # the release order.
    tag="$(git tag -l 'v12-*' --sort=-creatordate | head -1)"
    [ -n "$tag" ] || { log "no v12-* release tag found"; exit 0; }
fi

built="$(cat "$STATE_DIR/last-built" 2>/dev/null || true)"
if [ "$tag" = "$built" ]; then
    log "up to date ($tag)"
    exit 0
fi

log "new release $tag (last built: ${built:-none}); building"
git fetch --quiet origin
git checkout --quiet --force --detach "$tag"

LOG="$STATE_DIR/build-$tag.log"
# LLAMA_DIR is forwarded so build-containers.sh can borrow the canonical tip when
# present; otherwise it reconstructs the tree from the base tarball + the patches.
# --prune keeps the local disk bounded: after a successful push it removes the
# images just built and any dangling build layers, but keeps the ROCm base images
# and the ccache store (both bounded: ~20 GB/base and CCACHE_MAXSIZE=20G).  A
# failed push aborts before the prune (build-containers.sh is `set -e`), so a
# broken image is left for inspection.
if LLAMA_CPP_DIR="$LLAMA_DIR" ./scripts/build-containers.sh --rocm "$ROCM_LINES" --push --prune $BUILD_EXTRA >"$LOG" 2>&1; then
    echo "$tag" > "$STATE_DIR/last-built"
    log "$tag pushed (log: $LOG)"
else
    log "ERROR: $tag build/push failed (log: $LOG); not marking it built"
    tail -20 "$LOG" >&2 || true
    exit 1
fi
