#!/usr/bin/env bash
# Per-block build gate -- the r2 "every one of the 16 commits builds the `all`
# target" standard.
#
# WHY THIS EXISTS:
#   The 16-block delivery is bisected and reviewed block by block, so every block
#   commit must build on its own.  Historically this regressed silently (the r2
#   fix landed 2026-10-05; by r49 blocks 06-14 no longer built) because only the
#   *tip* was ever built.  Fixes kept being folded into the last block instead of
#   the block that owns the broken code.  This gate is the instrument that makes
#   the regression impossible to miss.
#
# METHOD (do NOT use an incremental/reused build dir):
#   A reused build cache hides configure/header errors and reports a *wrong*
#   per-block list (the r2 lesson: the reused cache kept the tip's
#   GGML_CUDA_FA_QUANTS and made blocks 00-07 look like configure failures).  A
#   fresh `rm -rf "$BUILD_DIR"` + configure + `all` build per commit is the only
#   reliable method.  ccache makes it fast (only the per-block delta recompiles).
#
# Usage:
#   ./gate-per-block-build.sh [options]
#
# Options / environment:
#   --repo DIR       REPO        llama.cpp checkout to walk (default: $LLAMA_REPO
#                                or ~/llama.cpp).  Must contain release.json's
#                                base..tip commits.
#   --base REV       BASE        base commit (default: release.json .base)
#   --tip REV        TIP         tip commit  (default: release.json .tip)
#   --build-dir NAME BUILD_DIR   build directory, relative to REPO (default
#                                build-per-block)
#   --jobs N         JOBS        build parallelism (default 16)
#   --rocm DIR       ROCM_714    ROCm install (default /opt/rocm-7.14.1-gfx120X)
#   --gpu-target T   GPU_TARGETS (default gfx1201)
#   --keep-going                 continue past the first failure (default: stop)
#
# Exit: 0 = every commit built, 1 = at least one failed, 2 = usage/prereq error.
#
# ccache is used when present (disable with CCACHE=0).  This is a local,
# compiler-only gate: it does not run the model or touch a GPU.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"

REPO="${LLAMA_REPO:-$HOME/llama.cpp}"
BUILD_DIR="build-per-block"
JOBS="${JOBS:-16}"
ROCM_714="${ROCM_714:-/opt/rocm-7.14.1-gfx120X}"
GPU_TARGETS="${GPU_TARGETS:-gfx1201}"
KEEP_GOING=0
BASE=""
TIP=""

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)       REPO="$2"; shift 2 ;;
        --base)       BASE="$2"; shift 2 ;;
        --tip)        TIP="$2"; shift 2 ;;
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --rocm)       ROCM_714="$2"; shift 2 ;;
        --gpu-target) GPU_TARGETS="$2"; shift 2 ;;
        --keep-going) KEEP_GOING=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

if [ -z "$BASE" ] || [ -z "$TIP" ]; then
    command -v jq >/dev/null || { echo "ERROR: jq is required to read release.json" >&2; exit 2; }
    [ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found" >&2; exit 2; }
    [ -n "$BASE" ] || BASE="$(jq -r '.base' "$MANIFEST")"
    [ -n "$TIP" ]  || TIP="$(jq -r '.tip' "$MANIFEST")"
fi
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: $REPO is not a git checkout" >&2; exit 2; }
case "$BUILD_DIR" in /*|..*|*..) echo "ERROR: --build-dir must be a plain relative path" >&2; exit 2 ;; esac
[ -d "$ROCM_714" ] || { echo "ERROR: ROCm install not found: $ROCM_714" >&2; exit 2; }

command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
git -C "$REPO" cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "ERROR: base $BASE not in $REPO" >&2; exit 2; }
git -C "$REPO" cat-file -e "$TIP^{commit}"  2>/dev/null || { echo "ERROR: tip $TIP not in $REPO" >&2; exit 2; }

CCACHE_LAUNCHERS=()
if [ "${CCACHE:-1}" != 0 ] && command -v ccache >/dev/null 2>&1; then
    CCACHE_LAUNCHERS=(-DCMAKE_HIP_COMPILER_LAUNCHER=ccache
                      -DCMAKE_C_COMPILER_LAUNCHER=ccache
                      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rboosts-per-block.XXXXXX")"
ORIG_REF="$(git -C "$REPO" symbolic-ref -q --short HEAD || git -C "$REPO" rev-parse HEAD)"
echo "==> per-block build gate: $BASE..$TIP in $REPO"
echo "    logs: $LOG_DIR   build dir: $REPO/$BUILD_DIR   jobs: $JOBS"

total=0; ok=0; failed=0; declare -a fail_list=()
for c in $(git -C "$REPO" rev-list --reverse "$BASE..$TIP"); do
    total=$((total+1))
    subj="$(git -C "$REPO" log -1 --format=%s "$c")"
    printf '==> [%2d] %s %s\n' "$total" "${c:0:9}" "$subj"
    git -C "$REPO" checkout -q "$c" || { echo "    CHECKOUT FAILED"; failed=$((failed+1)); fail_list+=("$c checkout"); [ "$KEEP_GOING" = 0 ] && break; continue; }

    rm -rf "$REPO/$BUILD_DIR"
    if ! HIPCXX="$ROCM_714/lib/llvm/bin/clang" HIP_PATH="$ROCM_714" \
        cmake -S "$REPO" -B "$REPO/$BUILD_DIR" \
            -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 \
            -DHIP_PLATFORM=amd -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS="$GPU_TARGETS" \
            -DCMAKE_C_COMPILER="$ROCM_714/lib/llvm/bin/clang" \
            -DAMDGPU_TARGETS="$GPU_TARGETS" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER="$ROCM_714/lib/llvm/bin/clang++" \
            -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_714/lib" \
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_HIP_FLAGS= \
            "${CCACHE_LAUNCHERS[@]}" \
            > "$LOG_DIR/${total}.configure.log" 2>&1; then
        echo "    CONFIGURE FAILED (log: $LOG_DIR/${total}.configure.log)"
        tail -5 "$LOG_DIR/${total}.configure.log" | sed 's/^/      /'
        failed=$((failed+1)); fail_list+=("$c configure")
        [ "$KEEP_GOING" = 0 ] && break
        continue
    fi
    if cmake --build "$REPO/$BUILD_DIR" --target all -j "$JOBS" -- -k \
            > "$LOG_DIR/${total}.build.log" 2>&1; then
        echo "    BUILD OK"
        ok=$((ok+1))
    else
        echo "    BUILD FAILED (log: $LOG_DIR/${total}.build.log)"
        grep -E "error:" "$LOG_DIR/${total}.build.log" \
            | sed -E 's/^.*(error:)/      \1/' | sort -u | head -12
        failed=$((failed+1)); fail_list+=("$c build")
        [ "$KEEP_GOING" = 0 ] && break
    fi
done

git -C "$REPO" checkout -q "$ORIG_REF" 2>/dev/null || true

echo
echo "==> result: $ok/$total built, $failed failed"
if [ "$failed" -ne 0 ]; then
    printf '    failed: %s\n' "${fail_list[*]}"
    echo "    full logs kept in $LOG_DIR"
    exit 1
fi
rm -rf "$LOG_DIR"
echo "    all $total block commits build the all target"
exit 0
