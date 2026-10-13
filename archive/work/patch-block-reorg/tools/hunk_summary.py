#!/usr/bin/env python3
"""hunk_summary.py -- print each hunk header in a unified diff with a tag made
from the first few added lines (for triaging which target a hunk belongs to).

Usage: git show <commit> [-- path] | hunk_summary.py [keyword ...]
"""
import re
import sys

KEYS = sys.argv[1:] or [
    "band", "Band", "BAND", "GQA", "gqa", "ncols", "two_byte", "WIDE", "wide",
    "kv_native", "native", "derived", "kq_mask", "tok_lo", "cell_pos",
    "sync_readback", "readback", "view_src", "n_views", "slab", "moe_cache",
    "MMB", "mmb", "Q6_K", "J_MAX", "VFME", "shexp", "hc_mix", "qsa", "QSA",
]

def main() -> int:
    lines = sys.stdin.read().splitlines()
    hdr = None
    body = []
    idx = 0
    def flush():
        if hdr is None:
            return
        adds = [l for l in body if l.startswith("+") and not l.startswith("+++")]
        kw = []
        for a in adds:
            for k in KEYS:
                if k in a and k not in kw:
                    kw.append(k)
        tag = (" <<< " + " | ".join(kw[:4])) if kw else ""
        first = next((a[:90] for a in adds if a.strip() not in ("+",)), "")
        print(f"  [{idx}] {hdr}{tag}")
        if not kw:
            print(f"        e.g. {first}")
    for l in lines:
        if l.startswith("@@ "):
            flush()
            hdr = l
            body = []
            idx += 1
        else:
            body.append(l)
    flush()
    return 0

if __name__ == "__main__":
    sys.exit(main())
