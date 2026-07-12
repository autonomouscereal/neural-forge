# Canonical source consolidation

## Decision

`neural-forge` is the one active source tree and Git repository for the model
folding project.

- Local checkout: `C:/Users/cereal/Documents/ai training/neural-forge`
- Remote checkout: `/home/cereal/model-surgery-lab/neural-forge-src`
- Active branch: `codex/lrc-blank-compression`
- Git remote: `https://github.com/autonomouscereal/neural-forge.git`

No experiment may create another copied source tree. Launchers must execute from
the canonical remote checkout. Run outputs, parent checkpoints, child
checkpoints, caches, and datasets remain external to Git and are referred to by
absolute path plus cryptographic hash.

## What was merged

The active working tree contains the newest integrated implementation. Exact
source bytes from every discovered historical implementation copy were imported
as orphan Git commits so a superseded variant can be recovered without keeping
another directory alive.

Remote snapshots preserved under `refs/archive/remote-20260712/`:

- `neural-forge-behavior-vo-tuner-20260711` (`eb9c37a`)
- `neural-forge-cached-attention-smoke-20260711` (`db34a1f`)
- `neural-forge-pivot` (`fa9ca7c`)
- `neural-forge-residual-attn-20260711` (`ce49229`)
- `neural-forge-residual-attn-functional-20260711` (`0d7d915`)
- `neural-forge-role-scale-functional-20260711` (`c24fa1f`)
- `neural-forge-src` (`c537058`)
- `neural-forge-src-981feb9` (`ff9755d`)
- `neural-forge-unified-20260710` (`bd9f807`)

Local deployment packets preserved under
`refs/archive/local-packets-20260712/`:

- `neural-forge-unified-20260711` (`512b74c`)
- `neural-forge-unified-v4-patch-20260711` (`4205d44`)
- `neural-forge-unified-v5-patch-20260711` (`dff3d21`)
- `neural-forge-unified-v6-patch-20260711` (`4b0f35a`)
- `neural-forge-unified-v7-patch-20260711` (`c75dd31`)

Adjacent project histories preserved under `refs/archive/projects/`:

- `chimera-omni-main` at `925a083296630aec3477367a15419af74f79fe8d`
- `ai-frontier-research-2026-main` at
  `bafd73231a057312bda986c93f5e46b78b5887e7`

`chimera-omni` is a modality/causal-ABI proof project, not a second folding
implementation. The frontier repository is the curated evidence corpus, not a
second compiler. Their complete histories are retained for reuse without
placing their generated proof media or large paper datasets in the active
compiler checkout.

The source archive downloaded from the remote host is
`runs/neural-forge-source-copies-20260712.tar.gz`, SHA-256
`3db5463950c22bd991a758ceb5102a77e8030e98cb221b7415699fe414846c8a`.
It is an ignored verification artifact, not another maintained source tree.

## Unique material recovered

Content hashing found the current active tree already contained every remote
source path except one historical path-conjugate launcher and a `.orig` backup.
The launcher was restored as
`experiments/run_qwen4b_blank_qwen146m_path_conjugate_c3fc503.sh`. The backup's
content is preserved in the archive ref and is intentionally excluded from the
active tree.

The compensated-fragile selection input was recovered from the pivot snapshot
into
`experiments/qwen3_compensated_fragile_selection_v5c_20260711.json` with
SHA-256
`c3418192d5d988edf286920453a22db30ff8a578b589563b9d636b468236beef`.

Ten obsolete deployment/copy-management scripts were removed from the active
tree after their exact bytes were preserved in archive refs. Rewriting those
scripts to target the canonical checkout would have made self-copy or cleanup
operations unsafe.

## Operational rule

Before launching an experiment:

1. Confirm the active commit and a clean or explicitly documented worktree.
2. Reuse `/home/cereal/model-surgery-lab/neural-forge-src`; never make a
   date-named code copy.
3. Put outputs under the external run/checkpoint roots and record hashes in the
   result ledger.
4. Search the experiment catalog and Git archive refs before implementing a
   method.
5. Treat a new runner name as insufficient novelty. Its mechanism, target,
   training signal, and promotion gate must differ from closed experiments.

That retirement is complete. The two adjacent local Git worktrees and eight
remote source copies were removed only after their pushed archive tags were
verified. Unique ignored research/media data moved inode-for-inode under
`runs/legacy-projects`; remote run evidence moved under
`runs/legacy-source-copies`. The local receipt SHA-256 is
`26e3b8eb1242aeaee624e1eaaf8c61713b3ace6610f5b80205653551756b5f09` and
the remote receipt SHA-256 is
`eaa7d55e54b1eb1d4aad17b3f06bfeaf6f837598ada1813ee13d1032071ca268`.
The selected residual-attention evidence remains available through the
canonical run path, and its eight authentication tests pass after retirement.
Checkpoint staging and model artifacts were outside the deletion set.

## Execution frontier after consolidation

The next model-building path is not another one-shot projection, numeric
weight-matching pass, V/O-only repair, or uniform codebook sweep. It is the
exact-target progressive compiler transaction:

1. compile `4.022B -> 1.410567B` in a target-native blank Qwen child;
2. install assignment-local Q/K/V/O and role-complete Gate/Up/Down residuals;
3. optimize only causal output actions and parent-/child-driven free rollouts;
4. bake accepted corrections into native child matrices;
5. require exact target parameter count, ordinary state keys, save/reload, no
   runtime parent or sidecar, and broad parent-pass retention;
6. promote the same transaction through `600.628M` and then `134.515M`, yielding
   a `29.903489x` final target.

This combines the first positive assignment-local QKVO result with the missing
role-complete MLP operator at the target size. It is distinct from the closed
p35-only and one-shot blank-fold lanes.
