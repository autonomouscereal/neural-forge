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
