#!/usr/bin/env python3
"""heatmap-audit.py -- health-check and coverage analysis for a `<base>.moe-heatmap` sidecar.

Why this exists
---------------
The MoE heat map (`MHT1` sidecar, block 09 / r60) is the only routing-distribution evidence we
have on disk, and it feeds three decisions that are invisible at runtime:
  * the **seed** (`apply_heat_seed_locked`: the top-`slots` experts per table),
  * the **placement order** (`alloc_all_locked`: layers allocated hottest-first, so the coldest
    sit nearest the slab's bottom-up boundary),
  * the campaign's coverage arithmetic (how many pairs an arena holds, and what hit rate that buys).

A poisoned or degenerate map does not error: it silently ranks by expert id and every consumer
keeps "working".  `--health` is the check for that.

Sections
--------
  health    saturation / zeros / distinct values, plus a coverage-curve collapse test.  A healthy
            map has a rising, concave H(f); a map whose entries are all tied degenerates to the
            uniform line H(f) ~= f (see the doubling bug, wip/moe-arena-defrag/FINDINGS-*).
  curve     H(f): the share of routed mass held by the top f of (layer, expert) pairs, and the f
            needed for 90/95/99 %.  This is the number a hit rate is bounded by.
  alloc     global-top-f vs per-table-uniform(q) at the same held fraction -- i.e. what our
            "uniform slots per table" sizing gives up against Strata's global ranking.
  arena     what `MOE_EXPERT_CACHE_MIB=<N>` (or the measured arena) can hold, and the coverage
            that implies.
  window    per-token hit rate vs per-verify-window-distinct hit rate for the same resident set.
            Strata counts the latter (one lookup per distinct (layer, expert) in the window); we
            count the former.  Strata's own docs imply this is a large effect; it is not -- the
            union probability 1-(1-p)^T is nearly linear in p because p_e ~ 1e-4, so the gap is
            ~0.0x points.  Measure it before blaming the metric for a residency gap.

Usage
-----
  ./heatmap-audit.py --health <sidecar>...
  ./heatmap-audit.py --all <sidecar>                        # health + curve + alloc + window
  ./heatmap-audit.py --arena-mib 15666 <sidecar>           # + what that arena holds
  ./heatmap-audit.py --window 2.6 <sidecar>                # measure tokens/verify-window

No dependencies (stdlib only).  Read-only: it never writes the sidecar.
"""
import argparse
import math
import struct
import sys

MAGIC = b"MHT1"
HDR = struct.Struct("<4sIIIQII")        # magic, version, n_tables, -, model_bytes, n_tensors, -
REC = struct.Struct("<i64sIQ")          # layer, role[64], n_experts, host_bytes (per expert!)
SAT = 0xFFFFFFFF


def parse(path):
    """-> (tables, header) where tables = [(layer, role, n_experts, expert_bytes, heat[list[int]])]."""
    with open(path, "rb") as f:
        blob = f.read()
    magic, ver, n_tables, _r0, model_bytes, n_tensors, _r1 = HDR.unpack_from(blob, 0)
    if magic != MAGIC:
        raise SystemExit("%s: not an MHT1 sidecar (magic %r)" % (path, magic))
    if ver != 1:
        raise SystemExit("%s: version %d, this tool reads version 1" % (path, ver))
    off = HDR.size
    tables = []
    for _ in range(n_tables):
        layer, role, n_exp, host_b = REC.unpack_from(blob, off)
        off += REC.size
        heat = list(struct.unpack_from("<%dI" % n_exp, blob, off))
        off += 4 * n_exp
        tables.append((layer, role.split(b"\0")[0].decode(), n_exp, host_b, heat))
    return tables, dict(version=ver, model_bytes=model_bytes, n_tensors=n_tensors, n_tables=n_tables)


def role_short(role):
    for k in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        if k in role:
            return k.replace("ffn_", "").replace("_exps", "")
    return role[:16]


def coverage(values, fracs):
    """H(f) for each f in `fracs` over a flat list of heat values."""
    total = sum(values)
    if total <= 0:
        return [(f, 0.0) for f in fracs]
    ordered = sorted(values, reverse=True)
    n = len(ordered)
    out, cum, i = [], 0, 0
    for f in fracs:
        want = max(1, int(round(n * f)))
        while i < want:
            cum += ordered[i]
            i += 1
        out.append((f, cum / total))
    return out


def f_for(values, target):
    total = sum(values)
    ordered = sorted(values, reverse=True)
    cum = 0
    for i, v in enumerate(ordered, 1):
        cum += v
        if cum / total >= target:
            return i / len(ordered)
    return 1.0


# ---------------------------------------------------------------- sections
def show_health(path, tables, hdr):
    vals = [v for t in tables for v in t[4]]
    n = len(vals)
    n_sat = sum(1 for v in vals if v == SAT)
    n_zero = sum(1 for v in vals if v == 0)
    distinct = len(set(vals))
    total = sum(vals)
    print("  pairs=%d tables=%d  total_heat=%.3g  distinct_values=%d" % (n, len(tables), total, distinct))
    print("  saturated=%.2f%%  zero=%.2f%%  nonzero=%.2f%%" % (
        100.0 * n_sat / n, 100.0 * n_zero / n, 100.0 * (n - n_zero) / n))
    # the collapse test: a degenerate (all-tied) map gives H(f) == f.  Healthy routing is much
    # steeper.  Compare H(0.5) with the uniform line and with a plausible floor.
    h50 = dict(coverage(vals, [0.5]))[0.5]
    verdict = []
    if n_sat / n > 0.02:
        verdict.append("POISONED: %.1f%% of entries are 0xffffffff -- the ranking is destroyed "
                       "(ties break by expert id).  See the heat-map doubling bug." % (100.0 * n_sat / n))
    if h50 < 0.55:
        verdict.append("DEGENERATE: H(0.5)=%.3f is at/near the uniform line (0.5) -- this map cannot "
                       "rank anything." % h50)
    elif h50 < 0.80:
        verdict.append("WEAK: H(0.5)=%.3f is well below a real routing distribution (~0.93 on the "
                       "35B-A3B sidecars); suspect a short run or partial accumulation." % h50)
    if total <= 0:
        verdict.append("EMPTY: no tally at all.")
    print("  H(0.5)=%.3f  %s" % (h50, "OK" if not verdict else ""))
    for v in verdict:
        print("  !! " + v)


def show_curve(tables):
    vals = [v for t in tables for v in t[4]]
    fracs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.637, 0.70, 0.80, 0.90]
    print("  " + "  ".join("f=%.2f" % f for f, _ in coverage(vals, fracs)))
    print("  " + "  ".join("%5.1f%%" % (100 * c) for _, c in coverage(vals, fracs)))
    print("  pairs needed: 90%% at f=%.3f, 95%% at f=%.3f, 99%% at f=%.3f" % (
        f_for(vals, 0.90), f_for(vals, 0.95), f_for(vals, 0.99)))
    print("  (Strata's fitted curve H(f)=1-(1-f)^3 would give 90% at f=0.536, 99% at f=0.785)")


def show_alloc(tables, fracs=(0.20, 0.25, 0.381, 0.50, 0.637)):
    """global top-f vs the same number of pairs spread uniformly over the tables (our sizing)."""
    total = sum(sum(t[4]) for t in tables)
    allv = [v for t in tables for v in t[4]]
    n, nt = len(allv), len(tables)
    ordered = sorted(allv, reverse=True)
    print("  per-table heat share: min %.3f%%  max %.3f%%  (uniform = %.3f%%)" % (
        100 * min(sum(t[4]) for t in tables) / total,
        100 * max(sum(t[4]) for t in tables) / total, 100.0 / nt))
    for f in fracs:
        held = int(n * f)
        glob = sum(ordered[:held]) / total
        q = held // nt
        uni = sum(sum(sorted(t[4], reverse=True)[:q]) for t in tables) / total
        print("  held %5.1f%% (%6d pairs): global-top %6.2f%%   per-table q=%3d %6.2f%%   loss %+.2f pts" % (
            100 * f, held, 100 * glob, q, 100 * uni, 100 * (uni - glob)))


def show_arena(tables, arena_mib):
    """Reproduce alloc_all_locked's arithmetic: slots = budget / sum(expert_bytes over tables)."""
    per_slot = sum(t[3] for t in tables)          # one slot in every table
    nt = len(tables)
    q = int(arena_mib * 1024 * 1024 // per_slot)
    pairs = q * nt
    allv = [v for t in tables for v in t[4]]
    held = min(pairs, len(allv))
    cov = sum(sorted(allv, reverse=True)[:held]) / sum(allv)
    print("  arena %.1f MiB, %d tables, %.3f MiB per (slot x table) -> q=%d slots/table, %d/%d pairs = %.1f%%" % (
        arena_mib, nt, per_slot / 1048576, q, pairs, len(allv), 100.0 * pairs / len(allv)))
    print("  truncation loss: %.1f MiB (%.2f%% of the arena)" % (
        arena_mib - q * per_slot / 1048576, 100 * (arena_mib - q * per_slot / 1048576) / arena_mib))
    allv_sat = sum(1 for v in allv if v == SAT) / len(allv)
    note = "  (!! map is %.0f%% saturated: the coverage figure is meaningless)" % (100 * allv_sat) if allv_sat > 0.02 else ""
    print("  coverage at that residency: %.2f%% (global top-%d; what a static hot set would hold)%s" % (
        100 * cov, held, note))


def show_window(tables, toks):
    """per-token vs per-window-distinct hit rate for the same resident set (global top-f)."""
    vals = [v for t in tables for v in t[4]]
    total = sum(vals)
    p = [v / total for v in vals]
    n = len(vals)
    print("  window = %.2f verified tokens (MTP); p_e = heat_e/total" % toks)
    print("  %-8s %-14s %-14s %-8s" % ("held%", "per-token", "per-window", "delta"))
    allp = sorted(p, reverse=True)
    print("  (hottest per-token p = %.4f, so 1-(1-p)^T differs from T*p by %.1f%%)" % (
        allp[0], 100 * abs((1 - (1 - allp[0]) ** toks) / (toks * allp[0]) - 1)))
    for f in (0.20, 0.381, 0.50, 0.637):
        held = int(n * f)
        idx = sorted(range(n), key=lambda i: -p[i])[:held]
        res = set(idx)
        tok = sum(p[i] for i in idx)
        num = sum(1 - (1 - p[i]) ** toks for i in idx)
        den = sum(1 - (1 - p[i]) ** toks for i in range(n))
        win = num / den
        print("  %-8.1f %-14.4f %-14.4f %+.4f pts" % (100 * f, 100 * tok, 100 * win, 100 * (win - tok)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecar", nargs="+")
    ap.add_argument("--health", action="store_true", help="only the saturation/degeneracy check")
    ap.add_argument("--all", action="store_true", help="every section")
    ap.add_argument("--arena-mib", type=float, default=None, help="measured/sized arena to price")
    ap.add_argument("--window", type=float, default=2.6, help="verified tokens per window (default 2.6)")
    a = ap.parse_args()

    for path in a.sidecar:
        tables, hdr = parse(path)
        print("== %s" % path)
        if a.health:
            show_health(path, tables, hdr)
            print()
            continue
        show_health(path, tables, hdr)
        if a.all or a.arena_mib is not None:
            print(" coverage (H(f) = share of routed mass in the top f of pairs):")
            show_curve(tables)
        if a.all:
            print(" capacity allocation (global ranking vs our uniform slots per table):")
            show_alloc(tables)
            print(" window accounting (our per-token counter vs Strata's per-window-distinct):")
            show_window(tables, a.window)
        if a.arena_mib is not None:
            print(" arena:")
            show_arena(tables, a.arena_mib)
        print()


if __name__ == "__main__":
    main()
