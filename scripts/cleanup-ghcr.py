#!/usr/bin/env python3
"""Prune old versions of a GHCR container package, keeping a rolling window.

GHCR keeps every manifest ever pushed: a multi-arch image is an index version
plus one version per architecture, and a moving tag leaves its previous
manifest behind as an untagged version.  A year of releases is therefore
thousands of versions, most of them unreachable (untagged).  This prunes them.

A version is KEPT if any of:
  * its created_at is within --keep-days of now, or >= --keep-since;
  * one of its tags ends in "-<sha>" for a --keep-base;
  * one of its tags matches a --keep-tag glob.

Everything else is deleted.  GHCR lists the index and its children as separate
versions, so a release must be kept or deleted as a whole; the date window does
that because one release is pushed in a single run.

Dry-run by default: prints the keep set, the delete summary and the tagged
versions that would go.  --apply performs the deletions (parallel, via `gh`).
Bulk deletes are throttled by GitHub's secondary rate limit; the script retries
--attempts times with a --wait-seconds pause and then reports how many are left
(it never treats a throttle as a hard failure, so it is safe to run from the
build/push sequence).

Requires a token with the `delete:packages` scope (once):
    gh auth refresh -h github.com -s delete:packages

Usage:
    ./cleanup-ghcr.py --package llama-cpp-rdna-boosts --keep-days 7 [--apply]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone


def fetch_versions(package: str) -> list[dict]:
    url = f"/user/packages/container/{package}/versions?per_page=100"
    out = subprocess.run(["gh", "api", "--paginate", url],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh api failed: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout)


def tags_of(v: dict) -> list[str]:
    return v.get("metadata", {}).get("container", {}).get("tags") or []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default="llama-cpp-rdna-boosts")
    ap.add_argument("--keep-days", type=int, default=7,
                    help="keep versions created within the last N days (default 7)")
    ap.add_argument("--keep-since", default=None, metavar="YYYY-MM-DD",
                    help="absolute cutoff; overrides --keep-days")
    ap.add_argument("--keep-base", action="append", default=[], metavar="SHA",
                    help="always keep versions tagged '<tag>-<SHA>' (repeatable)")
    ap.add_argument("--keep-tag", action="append", default=[], metavar="GLOB",
                    help="always keep versions with a tag matching the glob (repeatable)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel delete jobs (default 6)")
    ap.add_argument("--attempts", type=int, default=1,
                    help="delete attempts when GitHub throttles (default 1 = do not wait)")
    ap.add_argument("--wait-seconds", type=int, default=300,
                    help="pause between attempts after a throttle (default 300)")
    args = ap.parse_args()

    if args.keep_since:
        keep_since = datetime.fromisoformat(args.keep_since).replace(tzinfo=timezone.utc)
        window = args.keep_since
    else:
        keep_since = datetime.now(timezone.utc) - timedelta(days=args.keep_days)
        window = f"last {args.keep_days} days"
    keep_since_s = keep_since.strftime("%Y-%m-%dT%H:%M:%S")

    def is_kept(v: dict) -> str | None:
        if v["created_at"] >= keep_since_s:
            return f"age < {window}"
        for b in args.keep_base:
            if any(t.endswith("-" + b) for t in tags_of(v)):
                return f"keep-base {b}"
        for g in args.keep_tag:
            if any(fnmatch.fnmatch(t, g) for t in tags_of(v)):
                return f"keep-tag {g}"
        return None

    print(f"==> {args.package}: fetching versions (gh api)")
    versions = fetch_versions(args.package)
    print(f"    {len(versions)} versions; keeping created >= {keep_since_s}Z"
          + (f"; bases: {', '.join(args.keep_base)}" if args.keep_base else "")
          + (f"; tags: {', '.join(args.keep_tag)}" if args.keep_tag else ""))

    keep = [(v, is_kept(v)) for v in versions if is_kept(v)]
    delete = [v for v in versions if not is_kept(v)]
    tagged_del = [v for v in delete if tags_of(v)]
    byday = Counter(v["created_at"][:10] for v in delete)

    print(f"\n== KEEP   : {len(keep)} versions")
    for v, why in sorted(keep, key=lambda x: x[0]["created_at"]):
        print(f"   {v['created_at'][:19]}  {','.join(tags_of(v)) or '<untagged>'}   [{why}]")
    print(f"\n== DELETE : {len(delete)} versions ({len(tagged_del)} tagged, "
          f"{len(delete) - len(tagged_del)} untagged)")
    print("   by day: " + " ".join(f"{d}:{byday[d]}" for d in sorted(byday)))
    if tagged_del:
        print("   tagged versions to delete:")
        for v in sorted(tagged_del, key=lambda x: x["created_at"]):
            print(f"     {v['created_at'][:19]}  {','.join(tags_of(v))}")

    if not args.apply:
        print("\n(dry-run; pass --apply to delete)")
        return 0
    if not delete:
        print("\nnothing to delete")
        return 0

    print(f"\n==> deleting {len(delete)} versions ({args.jobs} jobs) ...", flush=True)
    left = versions
    throttled = False
    for attempt in range(1, args.attempts + 1):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as fh:
            for v in delete:
                fh.write(f"/user/packages/container/{args.package}/versions/{v['id']}\n")
            fh.flush()
            # xargs runs `gh api -X DELETE <url>` once per line, N in parallel
            proc = subprocess.run(
                ["xargs", "-a", fh.name, "-P", str(args.jobs), "-n", "1",
                 "gh", "api", "-X", "DELETE"],
                capture_output=True, text=True)
        left = fetch_versions(args.package)
        throttled = "rate limit" in proc.stderr.lower()
        print(f"    attempt {attempt}/{args.attempts}: {len(left)} versions remain "
              f"(was {len(versions)})", flush=True)
        if proc.stderr.strip() and attempt == 1:
            print("    stderr:", proc.stderr.strip()[:300])
        if not left or not throttled:
            break
        if attempt < args.attempts:
            print(f"    throttled; waiting {args.wait_seconds}s ...", flush=True)
            time.sleep(args.wait_seconds)
        delete = [v for v in left if not is_kept(v)]

    print(f"cleanup: deleted={len(versions) - len(left)} remaining={len(left)}")
    if left and throttled:
        print("    (GitHub throttled bulk deletes; re-run later to finish)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
