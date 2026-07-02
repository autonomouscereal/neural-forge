# Validation Results - 2026-07-02

## Activation-CKA Donor Layer Grouping

Added a merge-first donor layer grouping strategy for blank-target compression:
`activation-cka`. Instead of only slicing parent layers by depth or partitioning
by scalar residual energy, the new mode clusters donor residual-update
activation directions by pairwise linear CKA, allowing similar non-contiguous
parent layers to merge into the same child layer group before any recovery or
pruning-style cut.

What changed:

- `hf-donor-project-init --donor-layer-group-strategy activation-cka`
- `hf-xfamily-block-recover --donor-layer-group-strategy activation-cka`
- `hf-xfamily-chain-anchor-compress --init-donor-layer-group-strategy activation-cka`
- `hf-xfamily-chain-anchor-compress --block-donor-layer-group-strategy activation-cka`
- Donor init uses calibration hidden input/output rows already collected for
  activation-energy and activation-geometry grouping.
- Block recovery computes the same CKA groups from teacher hidden-state deltas
  on paired calibration batches.
- Reports include CKA similarity, seed layers, assignment similarity,
  per-layer delta scores, and rows per layer.

Why this matters:

- Current evidence and 2025-2026 research notes keep pointing at
  similarity-guided folding, DOT/transport merging, and FFN/submodule
  similarity as better fit than prune-first deletion.
- `activation-geometry` still partitions by contiguous energy bins after
  scoring. `activation-cka` can actually merge similar non-contiguous donor
  layers before folding them into a smaller blank target.
- This is still a structural operator, not proof of intelligence retention by
  itself. It must beat the activation-geometry controls under broad behavior
  and manual exact-generation evals.

Validation:

- Local parser/import check with `PYTHONPATH=src`:
  `1 passed, 3 skipped, 229 deselected`
- Local compile check:
  `python -m py_compile src/neural_forge/hf_xfamily.py src/neural_forge/hf_recovery.py src/neural_forge/cli.py`
- Remote focused Torch-backed tests:
  `4 passed, 229 deselected`
- Remote full suite:
  `285 passed`

Queued proof lane:

- Tag:
  `qwen4b_blank_smollm_activation_cka_geomean_full_gpu4_20260702`
- Queue PID: `3136217`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_activation_cka_geomean_full_gpu4_20260702.queue.log`
- GPU target: RTX 3090 UUID
  `GPU-2ce6a95c-7b1f-644a-3808-d781732a0582`
- Dependency wait PIDs:
  `2803948 2712676 2588435 2441342 596237`
- Recipe: geomean RMS true-LRC/EoRA/GRAIL chain with both init and block
  donor grouping set to `activation-cka`.
- Current status at queue start:
  waiting for prior PID `2803948`.

## Signed Activation Similarity Folding

Added an opt-in width/channel folding assignment mode:
`signed-similarity`. The existing activation similarity path grouped source
axes by cosine direction, so opposite-but-equivalent residual coordinates could
be treated as unrelated or cancel when folded. The signed mode groups by
absolute activation similarity and stores a per-source sign that is applied
when folding both incoming and outgoing axis contributions.

What changed:

- `hf-donor-project-init --activation-fold-assignment signed-similarity`
- `hf-xfamily-chain-anchor-compress --init-activation-fold-assignment signed-similarity`
- Sparse/Sinkhorn activation transport can now use absolute similarity for the
  non-negative transport plan while applying a stored sign matrix to folded
  source contributions.
- Non-transport activation folding caches both target assignment and sign
  vector, so paired incoming/outgoing folds use the same signed coordinate map.
- `experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh` now exposes
  `INIT_ACTIVATION_FOLD_ASSIGNMENT` while keeping the previous default
  `similarity`.

Why this matters:

- 2026 signed-permutation transport notes point out that RMSNorm transformer
  coordinates can match under sign flips, not just raw permutation.
- This is still merge/fold-first: source axes are not dropped; they are routed
  into target axes with sign correction before recovery.
- The mode is default-off because MLP activation axes are not always sign
  symmetric. It needs behavior A/B before promotion.

Validation:

- Local parser-focused check:
  `1 passed, 2 skipped, 233 deselected`
- Local compile check:
  `python -m py_compile src/neural_forge/hf_xfamily.py src/neural_forge/cli.py`
- Remote launcher syntax check:
  `bash -n experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh`
- Remote focused Torch-backed tests:
  `3 passed, 233 deselected`
- Remote full suite:
  `288 passed`

Queued proof lane:

- Tag:
  `qwen4b_blank_smollm_signed_cka_geomean_full_gpu4_20260702`
- Queue PID: `3168176`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_signed_cka_geomean_full_gpu4_20260702.queue.log`
- Recipe: same geomean RMS + activation-CKA true-LRC/EoRA/GRAIL chain as the
  activation-CKA control, but with
  `INIT_ACTIVATION_FOLD_ASSIGNMENT=signed-similarity`.
- Dependency: waiting behind the activation-CKA queue PID `3136217`.

## Fold RMS Stat Repair

Added an opt-in magnitude repair for merge-first donor projection:
`fold_stat_repair=rms`. It runs after many donor axes have been assigned or
transported into fewer target axes. Plain mean folding can damp useful
orthogonal donor rows by averaging them; this repair rescales each folded target
axis toward the RMS power of the donor axes that fed it.

What changed:

- `hf-donor-project-init --fold-stat-repair rms`
- `hf-xfamily-chain-anchor-compress --init-fold-stat-repair rms`
- The repair is threaded through generic tensor folding, attention
  head/head-dim partition folding, non-layer tensors, token/lm-head hidden-axis
  projection, and activation-fit row folding.
- Assignment folding and sparse/Sinkhorn transport both use the same repair
  path. Mean, sum, and sqrt-sum reductions compute desired RMS with matching
  energy semantics so consumer-path summation is not blindly converted back to
  a mean.
- The main Qwen-4B to blank-SmolLM launcher accepts
  `INIT_FOLD_STAT_REPAIR`, defaulting to `none` for reproducibility.

Why this matters:

- The current best structural lanes already use activation/signed similarity,
  transport, CKA depth folding, per-module reductions, and projection RMS
  scaling. This patch targets the remaining local failure mode where a good
  fold assignment can still shrink the signal before recovery sees it.
- It is default-off because over-amplifying folded axes can hurt some layers.
  The proof lane is a controlled A/B against the signed activation-CKA recipe.

Validation:

- Local focused tests:
  `3 passed, 1 skipped, 234 deselected`
- Local full suite:
  `146 passed, 144 skipped`
- Remote launcher syntax check:
  `bash -n experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh`
- Remote focused tests:
  `4 passed, 234 deselected`
- Remote full suite:
  `290 passed`

Queued proof lane:

- Tag:
  `qwen4b_blank_smollm_signed_cka_rms_repair_geomean_full_gpu4_20260702`
- Queue PID: `3218886`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_signed_cka_rms_repair_geomean_full_gpu4_20260702.queue.log`
- Recipe: same signed activation-CKA geomean true-LRC/EoRA/GRAIL chain, plus
  `INIT_FOLD_STAT_REPAIR=rms`.
- Dependency: should run after signed activation-CKA queue PID `3168176`.
- Current status at queue start:
  waiting for prior PID `3168176`.

## Answer-Boundary Margin Recovery

Added an opt-in ALM recovery loss aimed at the current exact-answer failure:
the compact model often emits a prompt number or nearby arithmetic distractor
as the first answer token.

What changed:

- `hf-xfamily-recover --answer-first-token-margin-weight`
- `hf-xfamily-recover --answer-first-token-margin`
- `hf-xfamily-recover --answer-first-token-max-negatives`
- `hf-xfamily-chain-anchor-compress --alm-answer-first-token-margin-weight`
- `hf-xfamily-chain-anchor-compress --alm-answer-first-token-margin`
- `hf-xfamily-chain-anchor-compress --alm-answer-first-token-max-negatives`
- The loss builds numeric or multiple-choice distractors from the same
  answer-candidate set used by answer contrastive recovery, including copied
  prompt numbers such as `6` and `4` for `6 + 4 -> 10`.
- It applies a direct hinge margin at the first response token: the gold answer
  token must beat distractor token logits by the configured margin.
- Token IDs are selected with tokenizer offset mappings when available so BPE
  tokens that absorb the pre-answer space are handled correctly.

Why this matters:

- The latest exact-generation reports show useful broad behavior transfer but
  repeated exact arithmetic failures. For example, the cleanopen full ALM
  exact-gen report answered `6+4 -> 6`.
- Existing answer contrastive and teacher-rank losses compare whole candidate
  sequence likelihoods. This new objective directly attacks the first-token
  prompt-copy failure without replacing the broader recovery stack.
- The loss is default-off, so historical runs and queued recipes remain
  reproducible until a queue explicitly opts in.

Validation:

- Local focused xfamily tests:
  `3 passed, 1 skipped, 225 deselected`
- Local full xfamily module:
  `96 passed, 133 skipped`
- Local full suite:
  `143 passed, 138 skipped`
- Remote focused xfamily tests after deployment:
  `4 passed, 225 deselected`
- Remote full xfamily module after deployment:
  `229 passed`
- Remote full suite after deployment:
  `281 passed`

Current status:

- This is an implementation improvement, not proof of retention yet.
- Next proof step is a margin-enabled exactness probe from the known
  mathdrill recovered checkpoint and, if positive, a follow-up ALM lane after
  the quantile-medoid blank-target structural checkpoint finishes.

Queued margin-enabled proof lanes:

- Immediate A/B probe tag:
  `qwen4b_mathdrill_l4_answer_margin_probe_gpu0uuid_20260702`
- Queue PID: `2905156`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_answer_margin_probe_gpu0uuid_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_answer_margin_probe_gpu0uuid_20260702.queue.sh.log`
- GPU: RTX 3090 UUID `GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561`
- Dependency: waits for control queue PID `2830714`
  (`qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702`).
- Anchor:
  `qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701/models/02_alm`
- New objective settings:
  `--alm-answer-first-token-margin-weight 0.08`,
  `--alm-answer-first-token-margin 1.5`,
  `--alm-answer-first-token-max-negatives 10`.
- Final eval target:
  `eval_qwen4b_mathdrill_l4_answer_margin_probe_gpu0uuid_20260702_exactgen_20260702_1000x1000x250.json`

- Quantile-medoid follow-up tag:
  `qwen4b_blank_smollm_quantile_medoid_answer_margin_alm_gpu6uuid_20260702`
- Queue PID: `2905158`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_answer_margin_alm_gpu6uuid_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_answer_margin_alm_gpu6uuid_20260702.queue.sh.log`
- GPU: Quadro RTX 8000 UUID
  `GPU-f5a09535-fc85-a027-50f7-734b344747dd`
- Dependency: waits for control queue PID `2830665`
  (`qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702`).
- Anchor:
  `qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702/models/01_block`
- New objective settings:
  `--alm-answer-first-token-margin-weight 0.07`,
  `--alm-answer-first-token-margin 1.5`,
  `--alm-answer-first-token-max-negatives 10`.
- Final eval target:
  `eval_qwen4b_blank_smollm_quantile_medoid_answer_margin_alm_gpu6uuid_20260702_exactgen_20260702_1000x1000x250.json`

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
| Cleanopen full ALM exact-gen | `134,515,008` | `804/1000` | `154/250` | `926/1000` | `29460.00` |
| Qwen3-4B -> blank GPT-2 init | `124,439,808` | `0/200` | `11/100` | `0/100` | `43116.57` |

Manual sample check still shows the key weakness:

- Parent: `6 + 4 -> 10`
- Blank: nonsense continuation
- Recovered SmolLM lanes: fluent answers, but exact arithmetic remains weak.
  The July 1 L4 lanes produced malformed/wrong `6+4` continuations, and the
  July 2 cleanopen full ALM exact-gen report answered `6+4 -> 6`.
- GPT-2 cross-architecture init: repeated/function-token collapse

Interpretation:

- The SmolLM blank-target compression path is not merely fitting tensor shapes;
  it is implanting broad language behavior into a random 135M target.
- The gap is exact-answer and forced-choice retention. The parent's forced
  choice score is `250/250`, while the best recovered 135M lanes are only
  around `150/250`.
- New recovery lanes should target exact answer grounding, not only lower PPL or
  generic fluent generation.

Cleanopen full ALM exact-generation report:

- Report:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/evals/eval_cleanopen_full_02_alm_exactgen_20260702_1000x1000x250.json`
- Source checkpoint:
  `qwen4b_cleanopen_anchor_newloss_full_gpu4_20260701/models/02_alm`
- This run improved PPL and forced-choice versus the earlier L4 recovered
  baselines (`154/250` choice vs. `149/250` and `150/250`), but exact
  generation health fell to `926/1000`.
- Manual exact sample from the report:
  `Question: What is 6 + 4? ... Answer: 6`, failed against expected `10`.
- Qualitative samples remain useful for non-exact prompts: pruning, model
  merge risk, Python `add(a, b)`, tokenizer mismatch, and rain-safety prompts
  generated coherent text/code. This is retained behavior, not random noise,
  but it is not enough for the compression goal because exact arithmetic and
  forced-choice retention still lag the parent.

Queued exactness combo lane:

- Previous P40 attempt:
  `qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu1_20260702`
  was canceled. The installed PyTorch build does not support the Tesla P40
  `sm_61` cards and raises `CUDA error: no kernel image is available for
  execution on the device`.
- Replacement tag:
  `qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702`
- Queue PID: `2830665`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702.queue.sh.log`
- GPU: Quadro RTX 8000 UUID
  `GPU-f5a09535-fc85-a027-50f7-734b344747dd`
- Upstream dependency: waits for quantile-medoid proof queue PID `2588435`.
- Recipe: ALM-only recovery from the quantile-medoid `01_block` checkpoint with
  attention influence, answer contrastive candidates, teacher-ranked
  candidates, number-focused token KD, first/last answer token LM,
  teacher-generate LM, stepwise donor KD, and on-policy hidden/anchor KD.
- Final eval target:
  `eval_qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702_exactgen_20260702_1000x1000x250.json`

Superseded P40 queue metadata retained for audit:

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

Immediate exactness probe from existing recovered checkpoint:

- Tag:
  `qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702`
- Queue PID: `2751973`
- Child PID: `2752007`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702.queue.log`
- Run log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702.log`
- Initial revision: `a34f48e6f180dd20c8895d61a505d2c5f2e0e14e`
- GPU: index `2`, Tesla P40 UUID
  `GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4`
- Anchor:
  `qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701/models/02_alm`
- Reason: the quantile-medoid structural proof lanes are still waiting behind
  upstream queue PID `2441342`; this immediate probe tests the exact-answer
  recovery mix on a known non-random recovered 135M checkpoint.
- Recipe: ALM-only recovery with the same exactness-focused ingredients as the
  queued combo lane, shortened to `160` ALM steps.
- Final eval target:
  `eval_qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702_exactgen_20260702_1000x1000x250.json`

Replacement immediate exactness probe:

- Previous P40 attempt:
  `qwen4b_mathdrill_l4_exact_combo_probe_gpu2_20260702` was canceled for the
  same unsupported `sm_61` kernel reason.
- Replacement tag:
  `qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702`
- Queue PID: `2830714`
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702.queue.sh.log`
- GPU: RTX 3090 UUID `GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561`
- Anchor:
  `qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701/models/02_alm`
- Final eval target:
  `eval_qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702_exactgen_20260702_1000x1000x250.json`

## Active Full-Run Status Snapshot

Polled AI server state again at `2026-07-02T02:07:59-06:00`.

The cleanopen full run has finished and produced the exact-generation report
summarized above. The remaining long and queued runs are still progressing or
waiting behind explicit dependencies:

| Run/queue | Current status | Latest observed signal |
|---|---|---:|
| `qwen4b_cleanopen_anchor_newloss_full_gpu4_20260701` | done; exact eval parsed | `804/1000` behavior, `154/250` choice, `926/1000` generation health |
| `qwen4b_mathdrill_anchor_newloss_full_gpu6_20260701` | ALM stage active | step `1` started from `807/1000`, generation health `1000/1000` |
| `qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701` | block stage active around layer `19` | `807/1000`, generation health `400/400` |
| `qwen4b_blank_smollm_eora_grail_source_rms_full_gpu4_20260702` | active on RTX 3090 UUID `GPU-2ce...` | model load / early chain active |
| `qwen4b_blank_smollm_quantile_geomean_drill_lrc_eora_gpu4_20260702` | queued | waits behind upstream GPU4 lane |
| `qwen4b_blank_smollm_module_eora_quantile_medoid_drill_lrc_gpu4_20260702` | queued | waits behind quantile proof lane |
| `qwen4b_blank_smollm_module_eora_quantile_medoid_aim004_alm_gpu4_20260702` | queued | waits behind quantile-medoid proof lane |
| `qwen4b_blank_smollm_quantile_medoid_exact_combo_alm_gpu6uuid_20260702` | queued | waits behind quantile-medoid proof lane |
| `qwen4b_mathdrill_l4_exact_combo_probe_gpu0uuid_20260702` | queued | waits for RTX 3090 UUID `GPU-4ead...` to clear |

Interpretation:

- The cleanopen full result improves PPL and forced choice, but exact arithmetic
  is still not recovered. It is evidence of useful behavior transfer, not goal
  completion.
- The currently queued replacement exactness lanes are on supported RTX/RTX8000
  UUIDs. The older P40 queue artifacts are audit history only and should not be
  used as live proof.
- The queued exact-generation comparisons and drill selector should remain the
  promotion signal once the active blank-target structural lanes finish.

## Answer-Top-Negative Recovery Probe

Implemented an opt-in first-answer-token top-negative path for exact-answer
recovery:

- `hf-xfamily-calibrate-head --head-calibration-mode answer-margin`
  now supports `--answer-first-token-top-negatives`.
- `hf-xfamily-recover` now supports
  `--answer-first-token-top-negatives`.
- `hf-xfamily-chain-anchor-compress` now supports
  `--alm-answer-first-token-top-negatives`.
- The default remains `0`, so existing runs keep their previous behavior unless
  the flag is explicitly set.

Validation:

- Local parser/handler checks:
  `4 passed`.
- Remote focused tests:
  `6 passed`.
- Remote full suite after the loss-side change:
  `283 passed`.

Head-only calibration from the cleanopen full compact checkpoint:

- Candidate:
  `qwen4b_cleanopen_anchor_newloss_full_gpu4_20260701/models/02_alm`
- Output:
  `qwen4b_cleanopen_full02_answer_margin_topneg24_a035_20260702`
- Calibration report:
  `512` answer rows, `12,984` margin pairs, `143` touched vocab rows,
  hinge loss `1.535 -> 0.838`, violations `2305 -> 2010`.
- Full exact eval:
  `eval_qwen4b_cleanopen_full02_answer_margin_topneg24_a035_20260702_exactgen_1000x1000x250.json`
- Result versus pre-calibration cleanopen full:
  behavior `804/1000 -> 804/1000`, choice `154/250 -> 154/250`,
  generation health `926/1000 -> 800/1000`, PPL
  `29460.00 -> 28841.49`.
- Manual read: not a promotion winner. Arithmetic stayed wrong and general
  manual responses became less coherent despite the modest PPL improvement.

Loss-side recovery with top-negative answer margin:

- Output:
  `qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702`
- Recipe:
  `80` recovery steps, all trainable params, `answer-first-token-margin-weight
  0.04`, `answer-first-token-margin 2.0`, `answer-first-token-max-negatives
  12`, `answer-first-token-top-negatives 24`, no intermediate behavior gate
  because the first gated attempt started an oversized `1000`-generation step-0
  eval before training.
- Full exact eval:
  `eval_qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702_exactgen_1000x1000x250.json`
- Result versus pre-recovery cleanopen full:
  behavior `804/1000 -> 804/1000`, choice `154/250 -> 146/250`,
  generation health `926/1000 -> 980/1000`, PPL
  `29460.00 -> 30729.52`.
- Manual read: mixed/negative. The Python prompt improved from a wrong
  `multiply` name to `def add(a, b): return a + b`, but arithmetic stayed
  wrong (`6+4` still generated malformed text ending in `88`) and safety /
  general answers remained confused.

Interpretation:

- Top-negative answer pressure is useful as a diagnostic and can improve
  generation-health gates, but head-only and short all-param recovery do not fix
  exact arithmetic or broad manual coherence on this compact checkpoint.
- Do not promote these lanes as the current best model. Keep the implementation
  because it gives future ALM/block-recovery lanes a sharper exact-answer loss,
  but pair it with behavior-selected structural recovery rather than standalone
  head correction.
