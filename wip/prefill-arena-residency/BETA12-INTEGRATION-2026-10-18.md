# Beta-12 integration + gate record — R2 prototype, 2026-10-18 (session 2)

Everything here is **WIP**: `wip/` only, nothing applied to `patches/`, no delivery change.
Companion records: [`README.md`](README.md) (design/measurements), [`HANDOVER.md`](HANDOVER.md)
(checkpoint + next steps), [`CONCURRENCY-2026-10-18.md`](CONCURRENCY-2026-10-18.md) (two-slot stress,
75/25 freeze negative result).

Box: 2x R9700 gfx1201, ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), `Qwen3.8-Flash-Next IQ4_NL`
(176.9B A3B, ~100 GB), field loader `--load-mode none --lazy-mode off`, 3 GPU idle.
Env for every perf run: `HIP_VISIBLE_DEVICES=0,1 LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1
GGML_CUDA_MMB_DOWN16=1 GGML_CUDA_SLAB_STATS=1`.

## 0. Checkpoint verification (as found)

| item | result |
|---|---|
| `r2-prototype.patch` sha256 | `2bfdb8263b0b64191baa67cd4293d4ad292475936c7067a4bf0b56ad4302f7a2` ✔ (matches `2bfdb826…`) |
| `git diff` in `~/llama.cpp` | byte-identical to the patch file (`diff -u` clean, same sha256) |
| `~/llama.cpp` HEAD / tree | `24380e5c7` / `dc48b4a9…` (= `release.json` r82 tree), R2 applied uncommitted |
| build current | `moe-expert-cache.cu.o` / hooks `.o` 12:39:00, `bin/llama-cli` 12:39:02, sources 12:38:56 ✔ |
| R2-only | one env var (`GGML_MOE_CACHE_STAGE`), no freeze / activity / device-map / `slot_dev` / `res_gen` code; the only freeze reference is a `TODO(freeze, phase 2)` comment (decision: the freeze is dropped — see §7) |
| patch size | 227 insertions / 3 files |

## 1. Beta-12 apply check (both orders)

Beta candidate `beta/0012-slab-transient-absorption/…beta-12…patch`, sha256 `9facba49…` ✔.
On a clean `rdna-boosts` worktree (release tree `dc48b4a9…`):

* `beta → R2` : `git apply --check` clean, applies.
* `R2 → beta` : `git apply --check` clean, applies.
* **The two orders produce byte-identical content** (the three changed files hash
  `bf14e1d5…` / `6bd5e609…` / `4dfaed2d…` in both orders — the hunks in
  `ggml-cuda-moe-cache-hooks.cu` are disjoint; only `git diff` line numbering differs).
  Consequence: the reverse order needs **no separate build**; the canonical order is the build.

## 2. Canonical tree built for the measurements

`~/llama-beta12-r2` = fresh clone of `~/llama.cpp` at `release.json.base` (`097cf33d9`) →
`scripts/apply-all.sh` (12 blocks, strict `git am`, tree `dc48b4a9…` = `release.json.tree`) →
`git am -3 beta 12` (commit `ab3af95ee`) → `git apply r2-prototype.patch` (uncommitted).
Build: `~/bin/build-llama-rocm`, 6m21s (ccache warm), `build-rocm`. Build id `ab3af95ee5f5`.

Control tree: `~/llama.cpp` (r82 + R2, build id `24380e5c7eb4`). Stage 0 on either tree is the
whole-table-staging arm (R2 inert).

## 3. Prefill A/B (stage 0 = plain staging, stage 1 = R2), 2 GPU `-sm tensor -ncmoe 48`

Prompts: `/tmp/prefill16k.txt` (48 268 B, sha256 `d8795003…`, 15 800 tok) and
`/tmp/prefill32k.txt` (96 558 B, 31 555 tok) — temp, **not** delivery prompts.

| prompt / `-ub` | r82 s0 | r82 s1 | **b12 s0** | **b12 s1** | R2 gain (b12) | b12 s1 vs r82 s1 |
|---|---:|---:|---:|---:|---:|---:|
| 16k / 2048 | 582.8 | 839.2 | 579.9 | **918.1** | +58.3 % | **+9.4 %** |
| 16k / 4096 | 1092.4 | 1436.1 | 1080.5 (2 runs) | **1554.3 (2 runs)** | +43.8 % | **+8.6 %** |
| 32k / 4096 | 1181.1 | 1585.4 | 1173.7 | **1737.6** | +48.0 % | **+9.6 %** |

Greedy hashes (`extract-generated.py`, `-n 64`, seed 42, temp 0) — **identical in every pair and to
the campaign record**: 16k/ub2048 `7171139467ce`, 16k/ub4096 `a60a39e11133`, 32k/ub4096 `98800fdba634`.

* **Beta 12 is not neutral for R2: +8.6 … +9.6 %** on top of R2 (same-session interleaved A/B, each
  arm repeated at 16k/ub4096: b12 1559.1 / 1549.5 vs r82 1436.1 / 1433.4).
* **Beta 12 costs ~1 % with R2 off** (stage 0: -1.2 % / -1.1 % / -0.6 %, consistent sign, three
  widths; b12 s0 16k/ub4096 repeated 1079.6 / 1081.4). Small but directionally consistent — worth
  recording for the beta's own validation, not a blocker for either change.

`-sm layer` was untested before; 16k/ub4096, `-sm layer -ncmoe 48`, b12: **741.7 → 1269.2 t/s
(+71.1 %)**, hash `37ea05798d5d` identical, staged `ok=448 geom_decline=0 unusable=116`,
H2D 15.6 % of staged. No `pipeline_parallel` / `n_copies>1` line in the log (the path stays
single-copy).

Other modes, 16k/ub4096, b12: `GGML_SCHED_STAGE_MODE=1` (redirect) s1 **1547.0** (vs 1549-1559
default) and s0 1080.7, hash `a60a39e11133` in both — redirect is neutral.
`--host-experts pool` (16k/ub2048): s0 472.7 → s1 **662.9 t/s (+40.2 %)**, hash `7171139467ce`
identical to the pinned-master run — **pool mode is compatible** (the master is pinned, the cold half
reads the same bytes; the campaign's open question #4 is answered: no decline needed).

## 4. Why beta 12 helps R2 (diagnostics, 16k/ub2048, `-lv 4`)

| quantity | r82 (no beta) | beta 12 |
|---|---:|---:|
| arena total (sizing) | 46 604.0 MiB | **52 309.9 MiB (+12.2 %)** |
| slots/table (per device) | 368 / 512 = 71.9 % | **413 / 512 = 80.7 %** |
| staged table bytes: arena / host | 272 625 / 101 595 MiB → **H2D 27.1 %** | 312 593 / 66 667 MiB → **H2D 17.6 %** |
| staged `ok` / `geom_decline` / `unusable` / `nomatch` | 1659 / 0 / 315 / 0 | 1687 / 0 / **287** / 0 |
| PRS rebuilds | 1 per staged call (1659) | 1 per staged call (1687) |
| decode hit rate `h` (after the prefill) | 0.9427 | 0.9450 |
| slab: evicted during the prefill | (campaign record 4.2 GiB) | 4260 + 4266 + 150 + 150 MiB |
| slab: re-armed after the drop | 54 tables | 54 tables |
| H2D calibration | 14.5 GB/s | 14.5 GB/s **from the slab arena** ✔ (runs before `alloc_all_locked`, arena sizes normally: `SLAB_INIT` dev1 free 29.15 GiB → `per-device sizing: 2 device(s), arena total 49 485.9 MiB` in the 131k config) |

So the beta's transient absorption (ring + FA/GDN/QSA prefill transients out of the outside reserve,
wide band released at the drop) leaves ~5.7 GiB more per-run VRAM to the expert-cache arena, which
buys ~9 points of residency and cuts the staged H2D by a third — that is the +9 %.

## 5. Two-slot concurrency (`-np 2`, `-c 131072`, `-b/-ub 4096`, field loader, harness as shipped)

Same-session arms: b12 s0 / b12 s1 / r82 s1.

| scenario | metric | r82 s1 | b12 s0 | **b12 s1** |
|---|---|---:|---:|---:|
| A (15.7k prefill + 5.3k prefill/decode) | long prefill | 882.5 | 750.8 | **927.9** (+5.2 % vs r82 s1) |
| A | short prompt | 331.4 | 285.9 | **345.1** |
| B (60 s decode window + 3.9k prefill stream) | short prefill median | 1431.7 | 1091.4 | **1526.0** (+6.6 % vs r82 s1) |
| B | long decode alone → under the stream | 49.2 → 4.2 | 49.7 → 3.7 | 49.3 → 4.2 |

* R2-on vs R2-off (b12): A +23.6 % long / +20.8 % short, B +39.8 % short prefill.
* **The long-decode starvation is unchanged** (≈4.2 t/s under the prefill stream with R2 off and on,
  r82 and beta) — the serialized two-slot scheduler, as recorded in `CONCURRENCY-2026-10-18.md`.
* Cache/arena in the 131k config: b12 **49 485.9 MiB / 112 608 slots**, H2D **23.6 %**; r82
  43 410.9 MiB / 98 784 slots, H2D 33.3-33.7 %. PRS rebuilds == staged calls in both (no freeze —
  by design).
* Slab churn per transition (server logs): **re-armed 54 tables with beta vs 60 without** (scenario A
  and B), evicted ~3.88 + 3.94 GiB per short-prefill request + 138/208 MiB at the drop. Beta 12
  absorbs the transients as designed; it does not remove the stand-down/re-arm cycle.

## 6. Gates (both arms unless noted)

| gate | b12 + R2 (stage 1) | r82 + R2 | r82 reference |
|---|---|---|---|
| coherence, dense `Qwen3.5-4B Q8_0` (seed 42, temp 0) | `1c5d32ac537d` | `1c5d32ac537d` | stage 0 also `1c5d32ac537d` |
| coherence, Flash-Next host-expert stage 0 == stage 1 | ✔ 3 widths + `-sm layer` | ✔ 3 widths | ✔ |
| `gate-prefill-logits.sh` (dense Gemma4-12B, base recorded from r82 s0, PPL 225.1593) | **mean KLD -0.000000, same-top-p 100.0 %** | same | self-test 0 / 100 % |
| prefill-logit, **host-expert MoE path** (Flash-Next IQ4_NL 1 GPU `-ncmoe 48`, ctx 512, 40 chunks, `--save-all-logits`/`--kl-divergence`) | **max KLD 5.9e-5, same-top-p 100.0 %** (base = r82 s0, PPL 2.7445) | — | — |
| `gate-qwen4exp-quant-coherence.sh` | PASS (IQ3_XXS / IQ4_NL / Q4_K_M transparent; IQ4_XS absent) | PASS (Q4_K_M needed one re-run — known `-sm tensor` near-tie nondeterminism, both coherent) | — |
| adaptive MTP, 4 axes + phase probe (`-n 3000`, reasoning pinned, `draft-mtp-adaptive --spec-draft-n-max 7`, `-sm layer -ncmoe 20`) | 0.66461 / 0.51005 / 0.76070 / 0.96262 / 0.62752 — **identical counts, mean len and t/s** | identical | identical to the recorded r82 table |
| MTP prose text hash (no `-lv 4`) | `7759fa4a7c8d` | `7759fa4a7c8d` | `7759fa4a7c8d` |
| `test-backend-ops -o MUL_MAT_ID` | 947/947, 3/3 backends | 947/947, 3/3 backends | — |
| `GGML_SCHED_STAGE_MODE=1` | ✔ hash + perf neutral | ✔ | — |
| `scripts/check-gpu-faults.sh` | clean (kernel journal 3 h window + 6 logs) | — | — |
| `////` runs | 0 in every log | 0 | — |

## 7. Findings / open questions for the maintainer (before any delivery work)

1. **All gates are green** (with and without beta 12) and the checkpoint verifies. The R2 prototype
   on the **r82 tree (no beta)** reproduces the campaign record: +44 % / +31 % / +34 % prefill,
   byte-identical greedy text, KLD 0 / 100 %.
2. **The +9 % from beta 12 is not part of the delivery**: beta 12 is a beta candidate awaiting
   independent validation, so shipping R2 now delivers the r82 numbers (+31-48 %), and the extra ~9 %
   arrives if/when beta 12 promotes. Worth stating in the block-09 note so the two are not conflated.
3. **Beta 12 shows a small (~1 %) stage-0 prefill cost** (three widths, consistent sign). Not caused
   by R2; flag it to the beta's own validation.
4. **`--host-experts pool` is compatible** (byte-identical, +40 %) — no decline path needed, which
   closes campaign open question #4 and removes a delivery caveat.
5. **`-sm layer` + `-ncmoe` now measured** (untested in session 1): +71 %, byte-identical, single-copy
   path.
6. **Two stale `ENVIRONMENT.md` rows** must be fixed whichever way the campaign goes:
   `GGML_MOE_CACHE_INPLACE` (r36, removed at r40 — the row documents a knob that no longer exists) and
   `GGML_MOE_CACHE_STAGE` (documented "on / kill-switch r36"; the r82 tree has only a stub and the
   prototype default is `0`).
7. **The `TODO(freeze, phase 2)` comment** in `moe_cache_stage_table` references the dropped 75/25
   design; it should be removed (or reworded) in the delivered patch so the block does not advertise
   a design the maintainer rejected.
8. No new risk surfaced: no GPU fault, no `////`, no decline counters beyond the expected
   `unusable` (slab stand-down mid-prefill), `geom_decline=0`, `nomatch=0`.

## 8. Repro

```bash
# trees
cd ~/llama.cpp && git checkout -- . && git apply ~/llama-cpp-rdna-boosts/wip/prefill-arena-residency/r2-prototype.patch
git clone -q ~/llama.cpp ~/llama-beta12-r2 && cd ~/llama-beta12-r2 && git checkout -q 097cf33d9 \
  && git branch -D rdna-boosts && bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh . \
  && git am -3 ~/llama-cpp-rdna-boosts/beta/0012-slab-transient-absorption/0012-rdna-boosts-beta-12-slab-transient-absorption.patch \
  && git apply ~/llama-cpp-rdna-boosts/wip/prefill-arena-residency/r2-prototype.patch \
  && ~/bin/build-llama-rocm

# 16k A/B (see HANDOVER §4); diagnostics need -lv 4 (the report is INFO and llama-cli hides INFO by default)
HIP_VISIBLE_DEVICES=0,1 GGML_MOE_CACHE_STAGE=1 GGML_CUDA_SLAB_STATS=1 ... -lv 4
# then: moe_cache_report "prefill cache staging (R2): ok=… geom_decline=… unusable=… nomatch=…; arena=… host=… (H2D …%)"
#       SLAB_STATS wide-arm / wide-drop / "moe_cache_rearm: re-armed N stood-down"

# gates
./scripts/gate-prefill-logits.sh --record  --base-dir /tmp/gates-r2 --model <gemma4-12b>   # on the r82 s0 build
./scripts/gate-prefill-logits.sh           --base-dir /tmp/gates-r2 --model <gemma4-12b>   # candidate
LLAMA_CLI=<tree>/build-rocm/bin/llama-cli ./scripts/gate-qwen4exp-quant-coherence.sh
python3 wip/prefill-arena-residency/tools/concurrency_check.py --scenario a --stage 1     # LLAMA_DIR=<tree>
```
