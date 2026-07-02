# Validation Results - 2026-07-02

## Activation-Quantile Transport Support

Added a structural fold lane for blank-target compression: `transport_support="quantile"`.

What changed:

- `hf-donor-project-init --transport-support quantile`
- `hf-xfamily-chain-anchor-compress --init-transport-support quantile`
- Activation transport can now seed target support axes from activation-weight strata instead of only clustered centroids or top-importance rows.
- For `importance` and `quantile` support, transport target masses are estimated from nearest-support activation load and used in both dense Sinkhorn transport and sparse balancing.
- Transport diagnostics now include `target_masses` so reports can prove whether support-energy balancing was active.
- `experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh` now accepts `CALIBRATION_FILE` so quantile runs can use the held-out-safe drill mix.
- The same launcher now accepts `EVAL_SUITE_FILE` so queued structural lanes can use the exact-generation behavior suite.

Why this matters:

- `importance` can over-focus on already-dominant channels.
- `centroid` can wash rare high-signal channels into cluster averages.
- `quantile` is intended as a middle lane: preserve low/mid/high activation strata before folding, then let transport merge donor axes into the target width.

Local validation:

- Focused xfamily transport/CLI tests: `3 passed, 3 skipped, 207 deselected`
- Full local suite: `140 passed, 125 skipped`

Current status:

- This is a fold/merge implementation improvement, not a proof of intelligence retention by itself.
- The next required evidence is a queued Qwen3-4B -> blank SmolLM2-135M quantile-support run with parent/before/after behavior and manual sample comparison.

## Activation Quantile-Medoid Transport Support

Added a second structural support lane for blank-target width folding:
`transport_support="quantile-medoid"`.

What changed:

- `hf-donor-project-init --transport-support quantile-medoid`
- `hf-xfamily-chain-anchor-compress --init-transport-support quantile-medoid`
- The support selector first partitions donor axes by activation-weight strata,
  then picks the real donor axis nearest the weighted activation centroid within
  each stratum.
- This combines the two useful transport ideas tested so far:
  `quantile` avoids collapsing all target axes onto the most active channels,
  while `medoid` avoids using only synthetic centroid axes.
- Like `importance`, `quantile`, and `medoid`, the new mode estimates transport
  target masses from nearest-support activation load and records
  `target_support_indices` / `target_masses` diagnostics.

Why this matters:

- The current failure mode is structural retention, not tensor shape fitting.
- A top-k-only support set can lose low/mid activation channels that still
  encode useful local behavior.
- A pure centroid support set can blur real donor axes before the recovery stage
  ever sees them.
- `quantile-medoid` is an adapter-free fold-first option: every donor axis still
  contributes through the transport plan, but compressed target axes are anchored
  on real donor channels across the activation distribution.

Validation:

- Local full suite: `142 passed, 133 skipped`
- Remote focused Torch tests: `6 passed, 217 deselected`
- Remote full suite: `275 passed`

Current status:

- This is ready to queue as a strict Qwen3-4B -> blank SmolLM2-135M structural
  lane using the existing chain launcher with
  `INIT_TRANSPORT_SUPPORT=quantile-medoid`.
- It still needs behavior/manual evidence against the cached parent, blank, and
  recovered baselines before it can be promoted.

## Weighted Projected-Vocab Recovery

Added weighted cross-tokenizer projected-vocab maps for recovery losses.

What changed:

- `_candidate_to_source_token_groups` now returns normalized source-piece weights in addition to ids, masks, and validity.
- OMP token-piece weights are preserved for digit, mixed-digit, and code-like tokens instead of discarded after sparse surface matching.
- Teacher-to-student projected logits now use weighted source-token combinations.
- Student-to-teacher projected probabilities now scatter candidate probability mass using the same source-piece weights.
- Projected-vocab group summaries report weighted-row counts and weight-sum ranges.

Why this matters:

- Prior projected-vocab KD treated multi-piece donor token groups as uniform averages.
- That weakens cross-tokenizer transfer for numbers, code surfaces, punctuation-heavy tokens, and composed tokens where one donor piece explains more of the candidate surface than another.
- The change keeps existing callers backward-compatible; weights are only required when projected-vocab KD paths are active.

Validation:

- Local full suite: `140 passed, 127 skipped`
- Remote focused weighted projection/recovery tests: `12 passed, 203 deselected`
- Remote full suite: `267 passed`

## Module-Specific Residual Compensation

Added opt-in module-specific pre-compensation modes for block recovery:

- `pre-module-linear`
- `pre-module-lowrank`
- `pre-module-eora`
- Aliases: `module-linear-pre`, `module-lowrank-pre`, `module-eora-pre`

What changed:

- Broad modes such as `lowrank-pre` still fit one block residual map and absorb it into both attention and MLP output projections.
- Module modes now capture folded donor attention/MLP module outputs across the full donor layer group, project them into the target hidden space, and fit separate maps for the candidate attention output projection and MLP output projection.
- The fitted maps are validation-gated before mutation using the same held-out row settings as pre-reconstruction.
- The direct block recovery CLI and chain launcher both accept the new modes.

Why this matters:

- The old broad residual map could force attention and MLP outputs through the same hidden-to-hidden correction even when their module geometries diverged after cross-architecture folding.
- The new path is closer to the desired parent-to-child fold: every donor module output in the assigned layer group is folded first, then each child module output is compensated independently.

Validation:

- Local focused module compensation/parser tests: `2 passed, 4 skipped, 213 deselected`
- Local full suite: `142 passed, 129 skipped`
- Remote focused module compensation/parser tests: `6 passed, 213 deselected`
- Remote full suite: `271 passed`

Current status:

- This is a structural folding improvement. It still needs a GPU behavior A/B against the existing `lowrank-pre` / `eora-pre` lanes on the exact-generation suite before it can be called a retention win.

## Cached Baseline Eval Context

Existing July 1 reports were parsed before queuing the module-specific lane so parent/blank/recovered baselines do not need to be rerun:

| Model/report | Behavior | Choice | Generation health | Manual check |
|---|---:|---:|---:|---|
| Qwen3-4B parent | `400/1000` | `250/250` | `161/1000` | `6+4` correct; simple Python add correct. |
| Random SmolLM2-135M blank | `0/1000` | `28/250` | `0/1000` | Nonsense output. |
| Cleanopen recovered 135M | `803/1000` | `149/250` | `1000/1000` | Python add correct; `6+4` wrong/malformed. |
| Mathdrill recovered 135M | `805/1000` | `150/250` | `1000/1000` | Python add correct; `6+4` wrong/malformed. |

Interpretation:

- The recovery stack is clearly implanting non-random language/code behavior into the blank 135M target.
- Broad behavior and generation-health scores can overstate exact math/code retention.
- New lanes must be compared against parent and recovered baselines on exact-generation/manual probes, not only broad generation health.

## Active Full-Run Status Snapshot

Polled AI server state at `2026-07-02T00:43:54-06:00`.

The three long full runs occupying usable GPUs were still progressing, not
stalled:

| Run | Latest block status | Best observed in-loop behavior |
|---|---:|---:|
| `qwen4b_cleanopen_anchor_newloss_full_gpu4_20260701` | layer `28`, step `4` | `803/1000`, generation health `400/400` |
| `qwen4b_mathdrill_anchor_newloss_full_gpu6_20260701` | layer `26`, step `4` | `807/1000`, generation health `400/400` |
| `qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701` | layer `16`, step `0` | `807/1000`, generation health `400/400` |

Interpretation:

- These jobs are preserving the current recovered-child plateau while folding
  through the full block stack.
- They have not yet produced final exact-generation reports, so they are not
  completion evidence.
- The queued exact-generation comparisons and drill selector should remain the
  promotion signal once these jobs finish.
