#!/usr/bin/env bash
# GPU fault check for a ROCm run: authoritative kernel faults, plus an optional
# llama log scan that ignores echoed prompt text.
#
# WHY THIS EXISTS
#   `grep 'page fault' server.err` is not a fault check.  A long test prompt is
#   echoed into the log on a single `launching slot : {...}` line, and the model
#   can generate those words too, so "page fault", "slab" and "aperture" all turn
#   up in a completely clean log (this cost real time in the #127 investigation).
#   The only authoritative source for a real GPU fault is the kernel log
#   (amdgpu / KFD) plus the runtime error strings.  This script reads the kernel
#   log, and when `--log` is given it also scans a llama log with the
#   echoed-prompt lines removed.
#
# Usage:
#   ./check-gpu-faults.sh [options]
#
# Options:
#   --since SPEC   kernel-log window, e.g. "30 min ago" (journalctl only)
#   --log FILE     also scan FILE (repeatable); echoed prompt lines are skipped
#   --quiet        print nothing unless a fault is found
#   -h, --help     this help
#
# Environment:
#   MAX_ECHO_LEN   a log line longer than this many bytes is treated as an
#                  echoed prompt/model dump and skipped (default 4096)
#
# Exit: 0 = clean, 1 = fault(s) found, 2 = usage error.
#
# Typical gate use:
#   HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-server ... >run.log 2>&1 &
#   ... drive the workload ...
#   ./scripts/check-gpu-faults.sh --since "1 hour ago" --log run.log
set -uo pipefail

MAX_ECHO_LEN="${MAX_ECHO_LEN:-4096}"
SINCE=""
LOGS=()
QUIET=0

usage() {
    cat >&2 <<'EOF'
Usage: check-gpu-faults.sh [--since SPEC] [--log FILE] [--quiet]

  --since SPEC   kernel-log window, e.g. "30 min ago" (journalctl only)
  --log FILE     also scan FILE (repeatable); echoed prompt lines are skipped
  --quiet        print nothing unless a fault is found
  -h, --help     this help

Exit: 0 clean, 1 fault found, 2 usage error.
EOF
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --since)   [ $# -ge 2 ] || usage; SINCE="$2"; shift 2 ;;
        --log)     [ $# -ge 2 ] || usage; LOGS+=("$2"); shift 2 ;;
        --quiet)   QUIET=1; shift ;;
        -h|--help) usage ;;
        *) echo "check-gpu-faults: unknown argument '$1'" >&2; usage ;;
    esac
done

say() { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }

# Real GPU-fault signatures.  `amdgpu` is required for the generic kernel lines
# so an unrelated userspace string cannot match; the runtime signatures are
# already unambiguous.
FAULT_RE='amdgpu.*(page fault|GCVM|PROTECTION_FAULT|GPU reset|device wedged|sq_intr)|HSA_STATUS_ERROR|MEMORY_APERTURE_VIOLATION|GPU reset\([0-9]|device wedged'

status=0
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# ---- kernel log ---------------------------------------------------------------
kernel_src=""
if command -v journalctl >/dev/null 2>&1 && timeout 20 journalctl -k -n 1 >/dev/null 2>&1; then
    kernel_src="journalctl"
    args=(-k --no-pager -o cat)
    [ -n "$SINCE" ] && args+=(--since "$SINCE")
    timeout 60 journalctl "${args[@]}" >"$tmp" 2>/dev/null
elif command -v dmesg >/dev/null 2>&1 && dmesg >/dev/null 2>&1; then
    kernel_src="dmesg"
    [ -n "$SINCE" ] && say "check-gpu-faults: note: --since is ignored for dmesg (no time filter)"
    timeout 60 dmesg >"$tmp" 2>/dev/null
else
    warn "check-gpu-faults: WARN: no readable kernel log (journalctl/dmesg); kernel check skipped"
fi

if [ -n "$kernel_src" ]; then
    if hits="$(grep -aE "$FAULT_RE" "$tmp")"; then
        status=1
        warn "check-gpu-faults: FAIL: kernel GPU fault(s) in $kernel_src:"
        printf '%s\n' "$hits" | sed 's/^/  /' | head -40 >&2
    else
        say "check-gpu-faults: kernel ($kernel_src${SINCE:+ since '$SINCE'}): clean"
    fi
fi

# ---- llama log(s) -------------------------------------------------------------
for log in ${LOGS[@]+"${LOGS[@]}"}; do
    if [ ! -f "$log" ]; then
        warn "check-gpu-faults: WARN: log not found: $log"
        status=2
        continue
    fi
    # Drop the echoed-prompt slot lines and any over-long line (both are prompt or
    # model dumps, not a fault), then match the fault signatures.
    if hits="$(awk -v m="$MAX_ECHO_LEN" 'length($0) <= m && $0 !~ /launching slot/ {print}' "$log" | grep -aE "$FAULT_RE")"; then
        status=1
        warn "check-gpu-faults: FAIL: fault signature(s) in $log (echoed prompt lines filtered):"
        printf '%s\n' "$hits" | sed 's/^/  /' | head -40 >&2
    else
        say "check-gpu-faults: log $log: clean (echoed prompt lines filtered)"
    fi
done

[ "$status" -eq 1 ] || say "check-gpu-faults: OK: no GPU faults detected"
exit "$status"
