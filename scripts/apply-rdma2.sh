#!/usr/bin/env bash
# Apply the RDNA2 (gfx1030) FA follow-up patch set (patches-rdna2/) on top of
# the full 12-block rdna-boosts set (r93).
#
# Usage: ./apply-rdma2.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   a llama.cpp checkout with the 12-block set applied
#                        (scripts/apply-all.sh already run); default: current dir
#   rdna-boosts-repo     path to the rdna-boosts repo; default: parent of scripts/
#
# The base is the recorded 12-block delivery tree (release.json.tree).  The
# add-on is applied with strict `git am` (one commit per patch); if the strict
# apply ever fails on a close-but-drifted base, the whole series is retried
# with `git am -3` (3-way merge against the blob ids).  On the recorded base
# the strict path is exact.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches-rdna2"

# release-rdma2.json is the add-on's source of truth (12-block base tree,
# add-on tree, patch hashes).
RELEASE_JSON="$RDNA/release-rdma2.json"
RELEASE_TREE=""
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    RELEASE_TREE="$(jq -r '.tree // empty' "$RELEASE_JSON")"
fi

cd "$LLAMA"

if [ ! -f CMakeLists.txt ] || [ ! -d ggml ]; then
    echo "ERROR: $LLAMA does not look like a llama.cpp checkout" >&2; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2; exit 1
fi

BRANCH="${RDNA_BRANCH:-rdna-boosts-rdma2-followups}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BRANCH already exists; delete it first (git branch -D $BRANCH)" >&2
    exit 1
fi
git checkout -q -b "$BRANCH"

APPLIED_WITH_3WAY=0
if ! git am "$PATCHES"/0[0-9][0-9][0-9]-*.patch; then
    echo "strict 'git am' failed at this base; aborting and retrying the series with 'git am -3' (3-way merge)" >&2
    git am --abort >/dev/null 2>&1 || true
    git am -3 "$PATCHES"/0[0-9][0-9][0-9]-*.patch
    APPLIED_WITH_3WAY=1
fi
echo
N_BLOCKS=1
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    _nb="$(jq -r '.n_blocks // empty' "$RELEASE_JSON" 2>/dev/null || true)"
    [ -n "$_nb" ] && N_BLOCKS="$_nb"
fi
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: applied with 'git am -3' (hunks merged against the recorded blob ids)."
    echo "The applied tree may differ from release-rdma2.json; report it to the maintainer."
fi
echo "Applied $N_BLOCKS add-on patch(es) on top of the 12-block rdna-boosts delivery."
if [ -n "$RELEASE_TREE" ]; then
    APPLIED_TREE="$(git rev-parse HEAD^{tree})"
    if [ "$APPLIED_TREE" = "$RELEASE_TREE" ]; then
        echo "Applied tree matches release-rdma2.json ($RELEASE_TREE)."
    else
        echo "ERROR: applied tree $APPLIED_TREE != release-rdma2.json tree $RELEASE_TREE" >&2
        exit 1
    fi
fi
git log --oneline -"$N_BLOCKS"
