#!/usr/bin/env bash
# Apply the RDNA2 (gfx1030) add-on patch set (patches-rdna2/, blocks 00-03) on
# top of the full 12-block rdna-boosts set.
#
# Usage: ./apply-rdna2.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   a llama.cpp checkout with the 12-block set applied
#                        (scripts/apply-all.sh already run); default: current dir
#   rdna-boosts-repo     path to the rdna-boosts repo; default: parent of scripts/
#
# The base is the recorded 12-block delivery tree (release.json.tree).  The
# add-on is applied with strict `git am` (one commit per block, subjects are
# the block subjects); if the strict apply ever fails on a close-but-drifted
# base, the whole series is retried with `git am -3` (3-way merge against the
# blob ids).  On the recorded base the strict path is exact.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches-rdna2"

# release-rdna2.json is the add-on's source of truth (12-block base tree,
# add-on tip/tree, patch hashes); see scripts/make-release.sh for the main set.
RELEASE_JSON="$RDNA/release-rdna2.json"
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

BRANCH="${RDNA_BRANCH:-rdna-boosts-rdna2}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BRANCH already exists; delete it first (git branch -D $BRANCH)"
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
N_BLOCKS=4
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    _nb="$(jq -r '.n_blocks // empty' "$RELEASE_JSON" 2>/dev/null || true)"
    [ -n "$_nb" ] && N_BLOCKS="$_nb"
fi
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: applied with 'git am -3' (hunks merged against the recorded blob ids)."
    echo "If this checkout was not at the recorded 12-block base, diff the applied tree"
    echo "against the canonical add-on tip before building."
else
    echo "All $N_BLOCKS add-on patches applied cleanly (strict git am) and committed on branch $BRANCH:"
fi

# On the strict path, assert the reconstructed tree is byte-for-byte the
# canonical add-on tree recorded in release-rdna2.json. ("PENDING" = the
# manifest is not finalized yet; record the tree, fill the manifest, re-run.)
if [ -n "$RELEASE_TREE" ] && [ "$RELEASE_TREE" != "PENDING" ] && [ "$APPLIED_WITH_3WAY" -eq 0 ]; then
    applied_tree="$(git rev-parse HEAD^{tree})"
    if [ "$applied_tree" != "$RELEASE_TREE" ]; then
        echo "ERROR: applied tree $applied_tree != release-rdna2.json tree $RELEASE_TREE" >&2
        echo "ERROR: the 12-block base and/or the add-on patch set do not match the recorded delivery" >&2
        exit 1
    fi
    echo "Applied tree matches release-rdna2.json ($RELEASE_TREE)."
fi

git log --oneline -$N_BLOCKS
echo
echo "Build (RDNA2 = gfx1030):"
echo "  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "  The set also compiles alongside the main set's defaults (gfx1100;gfx1151;gfx1201);"
echo "  every block is gated to gfx1030 at compile time, so the other families are untouched."
echo
echo "Gates (gfx1030, e.g. Radeon AI PRO R9700):"
echo "  ./build/bin/test-backend-ops --op=FLASH_ATTN_EXT   # full pass; without block 01 the kv=113 cases abort"
echo "  ./build/bin/test-backend-ops --op=MUL_MAT          # full pass"
