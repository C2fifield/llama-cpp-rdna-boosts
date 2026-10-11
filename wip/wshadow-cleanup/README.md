# `wshadow-cleanup` — enable `-Wshadow` for `src/` (TODO item 15)

**Status: PARKED (seeded 2026-10-11).** Not part of the delivery; nothing here may be applied to the fork
without the maintainer's go-ahead (see the WIP and promotion rules in `AGENTS.md`). This is the parked TODO
item *"`-Wshadow` cleanup for `src/` (item 15)"*.

> **Cold start.** Read this file, then
> [`archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`](../../archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md)
> (the full audit + method + disposition). Report a plan to the maintainer before any change and wait.

## 0. Why it exists

The Block-15 dense-arm blocker (2026-09-11, `GREEDY-PURITY.md` §23) was a **one-token shadowing bug**:
`ggml_tensor * kq_mask_top_k = ...` declared inside a block that already had an outer declaration of the
same name, which made a whole mask chain dead code. The code compiled, the chain was still built, and the
only detector was a node dump. `-Wshadow` reports that class outright; it is not enabled anywhere.

## 1. Current reality (measured 2026-09-12 on the delivery)

**128 `-Wshadow` warnings in 27 files:**

| class | count |
|---|---:|
| `declaration shadows a local variable` — **the dangerous class** | **46** |
| `declaration shadows a field of '…'` (constructor params / locals vs members) | 82 |

Top files: `llama-graph.cpp` 32, `llama-batch.cpp` 14, `llama-context.cpp` 12, `llama-model.cpp` 12,
`llama-sampler.cpp` 10, `llama-kv-cache.cpp` 9, then a long tail. `src/models/qwen4exp.cpp` is clean.

**Note for a r48 session:** the audit's line numbers are from the 2026-09-12 tree; re-run the (cheap)
replay method below rather than trusting the old numbers. The delivery's own files are clean — this is
upstream `src/` hygiene, not a delivery defect.

## 2. Proposal (from the audit disposition)

1. Enable `-Wshadow -Wno-shadow-field-in-constructor` for `src/` on the CI/build path.
2. Fix the **46 "shadows a local variable"** sites (mechanical renames) — that is the class that silently
   kills a chain.
3. The 82 field-shadow sites can follow later (or be suppressed by `-Wno-shadow-field-in-constructor`).
4. Do it in a **dedicated block/commit**, not folded into a code block: it touches ~20 upstream `src/`
   files and will collide on every re-base otherwise (which is exactly why it is parked).

## 3. Measurement method (cheap; no rebuild)

Replay the tree's own host compile commands for the `llama` target from `build-rocm-hybrid/compile_commands.json`,
replacing `-o <obj>` with `-fsyntax-only` and appending `-Wshadow`:

```bash
cd ~/llama.cpp/build-rocm-hybrid
python3 - <<'PY'
import json, shlex, subprocess
cc = json.load(open('compile_commands.json'))
n = 0
for e in cc:
    f = e['file']
    if '/src/' not in f or 'llama.dir' not in e.get('command',''):
        continue
    cmd = shlex.split(e['command'])
    out = []
    skip = False
    for i, a in enumerate(cmd):
        if skip:
            skip = False; continue
        if a == '-o':
            skip = True; continue
        out.append(a)
    out += ['-fsyntax-only', '-Wshadow']
    r = subprocess.run(out, capture_output=True, text=True)
    if 'warning:' in r.stderr:
        print(r.stderr.strip()); n += 1
print('TUs with warnings:', n)
PY
```

## 4. Pointers

- `TODO.md` → "Parked (not planned now)" → item 15.
- [`archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`](../../archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md)
  (full 46-site list, the method, and the disposition).
- `GREEDY-PURITY.md` §23 (the Block-15 mask-chain shadowing bug that motivated it).
