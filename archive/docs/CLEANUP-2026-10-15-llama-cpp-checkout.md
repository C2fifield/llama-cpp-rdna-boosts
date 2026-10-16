# Cleanup of the `~/llama.cpp` checkout (2026-10-15)

`~/llama.cpp` is disposable hosting for the block commits; the delivery (`patches/`) plus this repo's
records are the source of truth.  On 2026-10-15 the checkout was cleaned of its accumulated build/test
worktrees and experiment branches (disk was at 95 %).

Kept: the `~/llama.cpp` worktree itself, the `rdna-boosts` release branch (`fc6dcad9f`, r68), the
`master` ref at the base `097cf33d9`, and the `origin` remote (personal fork).

Removed: 15 registered worktrees and ~35 local-only experiment/backup branches (the release-time
`scripts/apply-all.sh` flow recreates everything it needs from `patches/`).

Preserved here because they were uncommitted and not captured anywhere else:

* `archive/work/issue-151-host-path-regression/FINDINGS-2026-10-15-review.md` -- a parallel-session
  review of issue #151 (GDN walk + async path), the env-read audit and the upstream `MUL_MAT_ID` dup
  finding, from the `llama-issue151` worktree.
* `wip/kv-cell-offset-nondeterminism/kv-offset-r43-wip.patch` -- the uncommitted fixed-alignment
  experiment from the parked `kv-offset-r43` worktree (the campaign README already records it as
  insufficient for the decode/server case).
* `archive/docs/llama-wip-r54-snapshot-2026-10-15.patch` and
  `archive/docs/llama-wip-r54-pairbench-2026-10-15.cpp` -- uncommitted `llama-wip-r54` changes
  (`ggml-backend-meta.cpp`, `llama-model.cpp`, `llama.cpp`) and its `pairbench.cpp`.

Everything else was either on the `origin` remote, reachable from `rdna-boosts`/`master`, or
superseded by the delivered block set.
