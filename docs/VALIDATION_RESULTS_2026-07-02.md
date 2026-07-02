# Validation Results - 2026-07-02

## Attention-Influence Cross-Tokenizer Recovery

Added an optional AIM-style recovery loss for cross-tokenizer and cross-family
blank-target compression.

What changed:

- `hf-xfamily-recover --attention-influence-weight`
- `hf-xfamily-recover --attention-influence-layer {last,middle}`
- `hf-xfamily-chain-anchor-compress --alm-attention-influence-weight`
- `hf-xfamily-chain-anchor-compress --alm-attention-influence-layer {last,middle}`
- The ALM recovery stage can now request teacher/student attentions and compare
  row-normalized attention-influence submatrices over span-aligned tokens.
- The loss is default-off, so existing queued runs and historical evidence keep
  their exact recipes unless a queue script opts in.
- The Qwen4B -> blank SmolLM, LFM230M -> blank SmolLM, and Qwen4B -> blank
  GPT-2 launchers now expose `ALM_ATTENTION_INFLUENCE_WEIGHT` and
  `ALM_ATTENTION_INFLUENCE_LAYER` for A/B GPU lanes.

Why this matters:

- The latest blank-target failures look less like tensor-shape failures and more
  like a decoder-manifold/tokenizer-alignment failure after structural folding.
- Projected-vocab KD aligns token distributions, and span hidden KD aligns local
  hidden states, but neither explicitly asks the target to preserve the donor's
  token-to-token interaction pattern.
- This lane follows the 2025-2026 direction of model-aware tokenizer transfer
  and cross-architecture transport: align structure and behavior across
  tokenizer/family boundaries without adding adapters or assuming a pretrained
  child.

References used for direction:

- MATT, Model-Aware Tokenizer Transfer, 2025:
  https://arxiv.org/html/2510.21954v1
- Transport and Merge, cross-architecture merging, 2026:
  https://arxiv.org/abs/2602.05495
- Chain of Merges, staged compression/merging, 2025:
  https://arxiv.org/html/2508.21421v1

Validation:

- Local focused suite:
  `2 passed, 9 skipped, 216 deselected`
- Local full suite:
  `142 passed, 137 skipped`
- Remote focused Torch suite:
  `11 passed, 216 deselected`
- Remote full suite:
  `279 passed`
- Follow-up eager-attention hardening for enabled attention-influence recovery:
  local focused `2 passed, 1 skipped, 224 deselected`; remote focused
  `3 passed, 224 deselected`; remote full `279 passed`.

Current status:

- This is a recovery objective improvement, not yet proof of intelligence
  retention.
- The next proof step is an A/B GPU lane against the cached Qwen3-4B parent and
  existing recovered 135M baselines, using exact-generation/manual probes, not
  PPL alone.

Queued A/B proof lane:

- Tag:
  `qwen4b_blank_smollm_module_eora_quantile_medoid_aim004_alm_gpu4_20260702`
- Queue PID: `2712676`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_module_eora_quantile_medoid_aim004_alm_gpu4_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_module_eora_quantile_medoid_aim004_alm_gpu4_20260702.queue.log`
- Initial revision: `f9098088d6786dbe39701b5f3b233429f93fa2b2`
- GPU: index `4`, RTX 3090 UUID
  `GPU-2ce6a95c-7b1f-644a-3808-d781732a0582`
- Upstream dependency: waits for the existing quantile-medoid proof queue PID
  `2588435`.
- Recipe: ALM-only recovery A/B from
  `qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702/models/01_block`
  with `--alm-attention-influence-weight 0.04` and
  `--alm-attention-influence-layer middle`.
- Final eval target:
  `eval_qwen4b_blank_smollm_module_eora_quantile_medoid_aim004_alm_gpu4_20260702_exactgen_20260702_1000x1000x250.json`

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

Queued proof lane:

- Tag:
  `qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702`
- Queue PID: `2588435`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702.queue.log`
- Initial revision: `a013d18bc39a0f290931d7f534348ef3c11e480d`
- GPU: index `4`, RTX 3090 UUID
  `GPU-2ce6a95c-7b1f-644a-3808-d781732a0582`
- Upstream dependency: waits for the existing GPU4 quantile proof queue PID
  `2441342`.
- Recipe mirrors
  `qwen4b_blank_smollm_module_eora_quantile_drill_lrc_gpu0_20260702` and
  changes the structural init support from `quantile` to `quantile-medoid`.
- Final eval target:
  `eval_qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702_exactgen_20260702_1000x1000x250.json`

## Gated MLP Activation Profiles For Folding

Updated the donor activation profile used for MLP intermediate-axis transport.

What changed:

- Gated MLP channel scores/features now come from the actual intermediate
  neuron stream `activation(gate_proj(x)) * up_proj(x)`.
- The old path collected `gate_proj` and `up_proj` outputs independently and
  averaged their summaries before building the merge/transport map.
- Linear input/output hooks are still retained for activation-fit rows; only
  the channel signature used to decide which MLP neurons are merged changed.

Why this matters:

- For SwiGLU/Gated-MLP models, the folded neuron is not the gate row or the up
  row in isolation. It is the multiplicative path consumed by `down_proj`.
- Cross-family width folding already reuses the same assignment cache for
  `gate_proj`, `up_proj`, and `down_proj` intermediate axes. This change makes
  that shared assignment correspond to the functional neuron being merged.
- This is a fold-first structural improvement. It does not add adapters or a
  post-hoc benchmark gate.

Validation:

- Local full suite: `142 passed, 134 skipped`
- Remote focused Torch tests: `6 passed, 218 deselected`
- Remote full suite: `276 passed`

Current status:

- Queued lanes that have not yet launched their Python process will consume this
  updated source mirror when they start.
- Existing running full jobs are unchanged because they already imported their
  code before this patch.

## Semantic MLP Intermediate-Axis Routing

Tightened how MLP activation profiles are attached to tensor axes during
cross-family width folding.

What changed:

- MLP neuron transport is now routed by tensor suffix, not by dimension match
  alone.
- `gate_proj`, `up_proj`, `w1`, `w3`, `dense_h_to_4h`, and `fc1` weights use
  MLP intermediate profiles on axis `0`.
- `down_proj`, `w2`, `dense_4h_to_h`, and `fc2` weights use MLP intermediate
  profiles on axis `1`.
- GPT-style `mlp.c_fc.weight` and `mlp.c_proj.weight` are handled with their
  transposed Conv1D conventions.

Why this matters:

- The previous dimension-only rule worked for Qwen/SmolLM-style shapes where
  hidden size and intermediate size are different.
- It could misroute MLP neuron transport in compact or custom target shapes
  where hidden and intermediate dimensions happen to match.
- This is directly aligned with arbitrary target architecture support: the
  hidden residual axis and MLP intermediate-neuron axis can now coexist even
  when their lengths are equal.

Validation:

- Local full suite: `142 passed, 135 skipped`
- Remote focused Torch tests: `4 passed, 221 deselected`
- Remote full suite: `277 passed`

## Transport Cache Isolation By Support Mode

Updated the shared axis-assignment cache key used during width folding.

What changed:

- Transport cache keys now include `transport_backend` and
  `transport_support` whenever `fold_mode="transport"`.
- Non-transport mean/RMS assignment keys are unchanged.

Why this matters:

- Gate/up/down MLP tensors deliberately share a semantic axis key so the same
  merged neuron map is used across the whole MLP path.
- A future chain can also evaluate multiple transport supports for the same
  semantic axis. Without support/backend in the key, a `quantile` map could be
  silently reused by a `quantile-medoid`, `medoid`, or Sinkhorn variant.
- This keeps fold-first A/B lanes honest: a reported support mode now really
  owns its transport plan.

Validation:

- Local full suite: `142 passed, 136 skipped`
- Remote focused Torch tests: `5 passed, 221 deselected`
- Remote full suite: `278 passed`

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

## Compact Exact-Generation Metric Snapshot

Polled and parsed the AI-server reports again at
`2026-07-02T01:33:36-06:00`.

| Model/report | Params | Behavior | Choice | Generation health | PPL mean |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B parent | `4,022,468,096` | `400/1000` | `250/250` | `161/1000` | `882.31` |
| Random SmolLM2-135M blank | `134,515,008` | `0/1000` | `28/250` | `0/1000` | `102037.06` |
| Cleanopen recovered 135M L4 ALM | `134,515,008` | `803/1000` | `149/250` | `1000/1000` | `36704.57` |
| Mathdrill recovered 135M L4 ALM | `134,515,008` | `805/1000` | `150/250` | `1000/1000` | `36407.37` |
| Cleanopen full block checkpoint | `134,515,008` | `803/1000` | `150/250` | `1000/1000` | `37026.94` |
| Qwen3-4B -> blank GPT-2 init | `124,439,808` | `0/200` | `11/100` | `0/100` | `43116.57` |

Manual sample check still shows the key weakness:

- Parent: `6 + 4 -> 10`
- Blank: nonsense continuation
- Recovered SmolLM lanes: fluent answers, but `6 + 4 -> 30`
- GPT-2 cross-architecture init: repeated/function-token collapse

Interpretation:

- The SmolLM blank-target compression path is not merely fitting tensor shapes;
  it is implanting broad language behavior into a random 135M target.
- The gap is exact-answer and forced-choice retention. The parent's forced
  choice score is `250/250`, while the best recovered 135M lanes are only
  around `150/250`.
- New recovery lanes should target exact answer grounding, not only lower PPL or
  generic fluent generation.

Queued exactness combo lane:

- Tag:
  `qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu1_20260702`
- Queue PID: `2729211`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu1_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu1_20260702.queue.log`
- Initial revision: `1a470325178ecc2d7521c7e750b6cd50ef806785`
- GPU: index `1`, Tesla P40 UUID
  `GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d`
- Upstream dependency: waits for quantile-medoid proof queue PID `2588435`.
- Recipe: ALM-only recovery from the quantile-medoid `01_block` checkpoint with
  attention influence, answer contrastive candidates, teacher-ranked candidates,
  number-focused token KD, first/last answer token LM, teacher-generate LM,
  stepwise donor KD, and on-policy hidden/anchor KD.
- Final eval target:
  `eval_qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu1_20260702_exactgen_20260702_1000x1000x250.json`

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
