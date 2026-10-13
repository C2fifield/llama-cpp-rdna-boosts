# NEXT-SESSION-PROMPT — WIP cleanup (copy-paste into the fresh session)

We are doing a WIP cleanup session on the delivery repo (`~/llama-cpp-rdna-boosts`). Read `AGENTS.md`
first, then the full handover at `wip/dflash-meta-split/HANDOVER-wip-cleanup.md`. The delivery is
`v12-097cf33d9-r53` (`release.json`: tip `4b27ac644`, tree `4083fd0ebed7c4659bd3254b2ab5a9f9906a209e`);
the fork `~/llama.cpp` is on `rdna-boosts` (tip `f74905dea`, same tree). `wip/dflash-meta-split/README.md`
also exists as the campaign's own cold-start note.

Goal: drive these four campaigns to a **recorded resolution** in this one session if possible:

1. `wip/dflash-meta-split/` - the `-sm tensor` + DFlash abort (TODO #58)
2. `wip/host-expert-master-drop/` - drop the full pinned `ROCm_Host` master under `--host-experts pool` (TODO #55)
3. `wip/meta-segmented-expert-upload/` - finish the segmented async upload for gemma4 + host experts + `-sm tensor`
4. `wip/ssm-alpha-beta-fusion/` - single-walk `ssm_alpha` + `ssm_beta` (TODO item 5(b))

"Resolved" means the decision is recorded, not that a fix necessarily lands: promote (give it an env
kill-switch, re-validate the combination, fold into the owning block, regenerate + validate, archive to
`archive/work/<name>/` with a `RESOLUTION.md`, record in `WORKLOG.md`, delete the `wip/` entry with no
stub and update `wip/CAMPAIGNS.md`) or close/park/refute (same archive + records, decision recorded).

Work in a disposable worktree, **never** on the fork's `rdna-boosts` branch, and never fold anything into
`patches/` without my explicit go-ahead for that item. Prefer 2 GPUs (`HIP_VISIBLE_DEVICES=0,1`) and pin
`-t` to leave the GPU IRQ cores free; run GPU work sequentially (never parallel/background benches).

Start by reading `AGENTS.md` and the handover, then **triage all four first** (confirm each repro and the
current r53 state), and tell me the plan and the priority order **before** doing any delivery-affecting
commit. Report all findings and design questions to me before any commit/tag/push, and stop to ask on
anything that is not a straightforward change.
