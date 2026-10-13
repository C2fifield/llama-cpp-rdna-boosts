#!/usr/bin/env python3
"""select_hunks.py -- filter a unified diff to the hunks matching a regex.

Phase-1 tooling for the patch-block re-org: a source block's diff is often split
across several target blocks.  This selects the hunks whose body matches a
regex and emits a valid patch containing only those hunks (file headers are
kept only for files that contribute a matching hunk).

Usage:
    git show <commit> [-- path] | select_hunks.py '<regex>'
    select_hunks.py '<regex>' < patchfile

Exit 2 if no hunk matches (so a scripted move fails loudly).
"""
import re
import sys

def split_files(lines):
    """Yield (header_lines, [hunk_lines, ...]) per file in a git diff."""
    n = len(lines)
    i = 0
    while i < n:
        if not lines[i].startswith("diff --git "):
            i += 1
            continue
        header = [lines[i]]
        i += 1
        while i < n and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git "):
            header.append(lines[i])
            i += 1
        hunks = []
        while i < n and not lines[i].startswith("diff --git "):
            if lines[i].startswith("@@ "):
                hunk = [lines[i]]
                i += 1
                while i < n and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git "):
                    hunk.append(lines[i])
                    i += 1
                hunks.append(hunk)
            else:
                i += 1
        yield header, hunks

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    pat = re.compile(sys.argv[1])
    lines = sys.stdin.read().splitlines(keepends=True)

    out = []
    kept = 0
    for header, hunks in split_files(lines):
        matched = [h for h in hunks if pat.search("".join(h))]
        if not matched:
            continue
        out.extend(header)
        for h in matched:
            out.extend(h)
        kept += len(matched)

    if kept == 0:
        print("select_hunks: no hunk matched %r" % sys.argv[1], file=sys.stderr)
        return 2
    sys.stdout.write("".join(out))
    return 0

if __name__ == "__main__":
    sys.exit(main())
