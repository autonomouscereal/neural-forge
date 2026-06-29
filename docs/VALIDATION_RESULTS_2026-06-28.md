# Validation Results 2026-06-28

This note records the gate expansion and rerank done after the small six-probe
layer gates proved too weak for blank-target Qwen3-4B to SmolLM2-135M folding.

## Tooling Changes

- `hf-donor-project-init` activation-fitted linears now report every attempted
  fit, not only applied fits. Reports include train rows, validation rows,
  folded-reference validation MSE, fitted validation MSE, relative improvement,
  cosine movement, applied/rejected counts, and rejection reasons.
- `--activation-fit-gate validation` now rejects fits that cannot be checked
  against enough held-out activation rows. Use `--activation-fit-gate none`
  only for explicit unsafe experiments.
- `hf-rank-evals` now writes `summary_rows`, behavior/generation category
  breakdowns, and suite mismatch warnings. This makes 1000-probe broad reports
  directly comparable without trusting PPL or tiny smoke probes.
- `hf-rank-evals` expands report globs itself, which is useful from PowerShell
  where native wildcard expansion is inconsistent for Python CLIs.

## Test Status

- Local: `PYTHONPATH=src python -m pytest -q` -> `106 passed, 88 skipped`.
- Remote AI server focused torch-backed tests:
  `11 passed, 166 deselected`.

## Broad Existing Rerank

Artifacts:

- Broad rank:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/broad_eval_1000_20260628/rank_broad_existing_all15_with_categories_20260628.json`
- All-report schema-filtered rank:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/broad_eval_1000_20260628/rank_all_existing_schemafiltered_with_categories_20260628.json`

The corrected broad-only rank includes 15 reports. The first script was too
aggressive and accidentally filtered names containing `lowrank`, `rank`, or
`gate`; the final rank uses report schema fields instead.

| Rank | Checkpoint | Behavior | Generation | Choice | PPL |
|---:|---|---:|---:|---:|---:|
| 1 | `qwen4b_tied29_lowrank_prerecon_blockinterface_a001_suitegate_b100g40_l1_20260628_201550` | 600/1000 | 160/200 | 25/250 | 14191.81 |
| 2 | `qwen4b_tied29_residual_lowrank_r16_a001_l2_20260628_192318` | 600/1000 | 160/200 | 25/250 | 14191.81 |
| 3 | `qwen4b_tied29_block_byte_behaviorbest_l2s8_20260628_141222` | 600/1000 | 160/200 | 25/250 | 14259.01 |
| 4 | `qwen4b_tied29_residual_comp_l2_20260628_181118` | 600/1000 | 160/200 | 25/250 | 14259.01 |
| 5 | `qwen4b_tied29_rsd_behaviorbest_60s` | 600/1000 | 160/200 | 25/250 | 14290.37 |
| 10 | `qwen4b_tied29_focused_lm_answer_rank_120s_20260628_123426` | 600/1000 | 80/200 | 30/250 | 19012.19 |
| 14 | `qwen4b_mean_cached_cka_mixed8192_1200s_best` | 2/1000 | 80/200 | 83/250 | 7783.28 |
| 15 | `qwen4b_activation_fit_inputproj_a0005_smollm135m_20260628_204104` | 0/1000 | 0/200 | 19/250 | 79179.90 |

Category signal from the top two broad candidates:

- Behavior `600/1000` is not broad intelligence. It is `200/200`
  commonsense, `200/200` tokenizer, and `200/200` model-surgery definition
  probes, with `0/200` math and `0/200` code.
- Generation `160/200` is the same pattern: commonsense `40/40`, tokenizer
  `40/40`, model-surgery `80/80`, and code `0/40`.
- The focused answer-rank run flips a different slice: code generation reaches
  `40/40`, but model-surgery and tokenizer generations drop to `0`, so it is
  not a better general candidate.
- The mixed8192 CKA run has the best broad choice score in this set
  (`83/250`) and much lower PPL, but only `2/1000` behavior. This is the clearest
  current example of PPL/choice movement not meaning usable behavior transfer.
- The activation-fit input-projection initializer is a hard negative:
  `0/1000` behavior, `0/200` generation, and `79,179.90` broad PPL.

## Interpretation

The current narrow leader is the suite-gated block-interface low-rank candidate,
but it is only a tie-preserving checkpoint over the previous low-rank residual
leader. The suite gate prevented a broad-generation regression; it did not
recover missing math or code behavior.

Direct activation-fitted weight replacement should be treated as a diagnostic
and small structural repair primitive, not the core blank-target compression
method. The current 2025-2026 source trail points to representation and
tokenizer bridges before folding or absorbing weights:

- CKA hidden-state matching can compare teacher and student hidden states with
  different dimensions: [OpenReview ICLR 2025](https://openreview.net/forum?id=IcVSKhVpKu).
- Approximate Likelihood Matching targets cross-tokenizer distillation without
  requiring identical vocabularies: [arXiv 2503.20083](https://arxiv.org/abs/2503.20083).
- Byte-Level Distillation gives a common byte interface for mismatched
  tokenizers, while still warning that consistent gains are not solved:
  [arXiv 2604.07466](https://arxiv.org/abs/2604.07466).
- Cross-architecture Transport and Merge uses feature/layer optimal transport
  over activations to infer neuron correspondences before weight fusion:
  [arXiv 2602.05495](https://arxiv.org/abs/2602.05495).

Next work should combine ALM/byte-token bridges, CKA or Procrustes-style hidden
geometry, task-focused feature selection, and only then activation-aware folding
or low-rank absorption. The broad gate remains the promotion authority.

## 2026-06-29 Category-Aware Gate Update

The broad ranker now uses hard category tie breakers after aggregate behavior:
`math` and `code` behavior coverage, then minimum behavior category pass rate,
then generation health and downstream metrics. Recovery commands with
`--save-best-behavior` use the same category-aware selector when aggregate
behavior ties.

New artifact:

- `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/broad_eval_1000_20260628/rank_broad_existing_all15_categoryaware_20260629.json`

The current top order did not materially change because the leading candidates
all have the same missing hard capabilities:

| Rank | Checkpoint | Behavior | Hard Behavior | Generation | Hard Generation | Choice | PPL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `qwen4b_tied29_lowrank_prerecon_blockinterface_a001_suitegate_b100g40_l1_20260628_201550` | 600/1000 | 0/400 | 160/200 | 0/40 | 25/250 | 14191.81 |
| 2 | `qwen4b_tied29_residual_lowrank_r16_a001_l2_20260628_192318` | 600/1000 | 0/400 | 160/200 | 0/40 | 25/250 | 14191.81 |
| 10 | `qwen4b_tied29_focused_lm_answer_rank_120s_20260628_123426` | 600/1000 | 0/400 | 80/200 | 40/40 | 30/250 | 19012.19 |
| 14 | `qwen4b_mean_cached_cka_mixed8192_1200s_best` | 2/1000 | 2/400 | 80/200 | 40/40 | 83/250 | 7783.28 |
| 15 | `qwen4b_activation_fit_inputproj_a0005_smollm135m_20260628_204104` | 0/1000 | 0/400 | 0/200 | 0/40 | 19/250 | 79179.90 |

This confirms that some previous attempts produced isolated code-generation or
math-choice movement, but none combine those gains with broad behavior. Future
recovery sweeps should be judged on hard-category movement explicitly, not only
aggregate `600/1000` behavior.

## 2026-06-29 Broad Recovery Gate Update

`hf-recover`, `hf-lm-recover`, and `hf-xfamily-recover` now accept
`--behavior-eval-suite` and `--behavior-max-generation-prompts`. This moves the
1000-probe category-aware gate into `--save-best-behavior` checkpoint selection
instead of using it only after a run finishes. `--behavior-max-probes` controls
the behavior slice. `hf-build-eval-suite` now defaults to 1000 behavior probes
and 1000 unique generation prompts. Broad generation in `hf-eval` and recovery
behavior checks is batched, so recovery can now be run with broad probes:

```powershell
--behavior-eval-suite broad_behavior_suite_1000_20260628.json --behavior-max-probes 1000 --behavior-max-generation-prompts 1000
```

Smaller generation slices such as `--behavior-max-generation-prompts 200` remain
useful for quick diagnostic intervals.

`hf-xfamily-reconstruct`, `hf-sweep-xfamily-mlp`, and `hf-xfamily-compress`
now also expose broad eval-suite controls so layer gates, sweep gates, pipeline
baseline/candidate gates, and recovered-candidate gates can be judged on the
same held-out suite instead of a tiny smoke slice.

Remote validation on the Qwen3-4B-Base into random SmolLM2-135M target run:

- Recovery run:
  `qwen4b_tied29_broadgate_noproj_logged_xrecover_20260628_215158`
- Recovery gate suite:
  `broad_behavior_suite_1000_20260628.json`
- Step 0: behavior `600/1000`, generation health `120/200`
- Step 1: behavior `600/1000`, generation health `120/200`
- Step 40: behavior `600/1000`, generation health `40/200`
- Best behavior checkpoint: step 1

Post-recovery `hf-eval` on the same 1000-behavior/200-generation suite ranked
the recovered model third of 29 broad reports:

| Rank | Report | Behavior | Hard Behavior | Generation | Hard Generation | Choice | PPL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `eval_prerecon_blockinterface_a001_suitegate_b100g40_l1.json` | `600/1000` | `0/400` | `160/200` | `0/40` | `25/250` | `14191.81` |
| 2 | `eval_residual_lowrank_r16_a001_l2.json` | `600/1000` | `0/400` | `160/200` | `0/40` | `25/250` | `14191.81` |
| 3 | `qwen4b_tied29_broadgate_noproj_logged_xrecover_20260628_215158_broad.json` | `600/1000` | `0/400` | `160/200` | `0/40` | `25/250` | `14240.93` |

A stricter new suite with 1000 behavior probes and 1000 unique categorized
generation prompts was built as
`broad_behavior_generation_suite_1000_20260629.json`. Re-evaluating the prior
top candidate and the recovered checkpoint showed the recovery improved broad
generation-health counts while leaving behavior intelligence unchanged:

| Model | Behavior | Hard Behavior | Generation | Hard Generation | Choice | PPL |
|---|---:|---:|---:|---:|---:|---:|
| Prior top candidate | `600/1000` | `0/400` | `199/1000` | `1/200` | `25/250` | `14191.81` |
| Broad-gated recovery | `600/1000` | `0/400` | `212/1000` | `14/200` | `25/250` | `14240.93` |

Manual inspection:

- Tokenizer-mismatch, rain-safety, and neuron-merging prompts produce
  recognizable but repetitive answers.
- Arithmetic behavior still fails: answers are long numeric noise instead of the
  requested result.
- Code behavior still fails: Python completions are syntactically invalid.
- The generation-health improvement is real under the gate but should not be
  mistaken for recovered code/math intelligence.

## 2026-06-29 Covariance Fold Feature Mode

`hf-donor-project-init` now accepts `--activation-feature-mode covariance` in
addition to `stats` and `subspace`. The new mode standardizes each donor
activation channel, computes a low-rank covariance/PCA signature, and appends
that signature to the existing scalar activation statistics before assignment
or transport. This gives `fold-mode=transport` and
`--activation-fold-assignment similarity` a co-activation geometry signal for
hidden axes, MLP channels, attention heads, and token/head projection folding.

This is an implementation response to the 2025-2026 activation-folding source
trail already tracked in `docs/RESEARCH_PATHS_2026.md`: DOTResize-style
transport, FLAT-LLM activation-space folding, Transport-and-Merge activation
alignment, and GRAIL/Gram-style reconstruction. It is still default-off. The
previous broad gates showed that local activation fits and PPL improvements can
hide broken arithmetic/code behavior, so covariance folding is only a new
candidate generator until remote broad-gate and manual-output results prove it
helps.

The same patch fixed an activation-fit collection trap: passing
`--activation-fit-linears` with the default `--activation-fit-max-rows 0`
previously collected no donor parent input/output rows, so the linear fit stage
had no real parent function signal unless another flag was remembered. It now
defaults the effective fit row cap to `--hidden-basis-max-rows` when linear
fitting is enabled and records both requested and effective caps in the
initializer report.

Remote candidate validation:

- Candidate:
  `qwen4b_tied29_covariance_transport_fit_mlpout_20260628_225904`
- Donor/target: `Qwen/Qwen3-4B-Base` folded into blank
  `HuggingFaceTB/SmolLM2-135M`, tied output head, `134,515,008` params
  (`~29.74x` compression using a 4B donor estimate).
- Settings: `fold-mode=transport`, `activation-feature-mode=covariance`,
  `activation-fold-assignment=similarity`, `donor-layer-fold-mode=fold`,
  `donor-layer-group-strategy=activation-energy`,
  `attention-fold-mode=activation-coupled`, `token-init-mode=donor-tokenized-weighted`,
  `projection-scale=target-rms`, and validation-gated `mlp-output`
  activation fits.
- Activation-fit row bug check: requested fit rows `0`, effective fit rows
  `512`; fit attempts `30`, applied `26`, rejected `4`; axis assignment cache
  entries `122`.
- Smoke broad eval against 200 behavior and 200 generation prompts from
  `broad_behavior_generation_suite_1000_20260629.json`: behavior `0/200`,
  generation health `0/200`, choice `11/64`, PPL `232,017,789.60`.
- Manual output is rejected. Example prompts produce repeated fragments such as
  `THE`, fenced-code ticks, and stray tokens rather than pruning/model-merge
  explanations or valid Python.

Interpretation: covariance features and the fixed parent activation-row
collection are mechanically active, but this specific schedule is worse than
the current broad leaders. The likely issue is not missing donor rows anymore;
it is that target-RMS plus 26 down-projection fits over only 8 calibration texts
creates an unstable autoregressive manifold. Next candidate should either use a
much smaller fit alpha with behavior-gated init, or separate covariance
transport from activation-fitted linears so their failure modes can be isolated.

Follow-up ablations isolated those failure modes:

| Candidate | Feature Mode | Activation Fits | Behavior | Generation | Choice | PPL | Manual Output |
|---|---|---:|---:|---:|---:|---:|---|
| `qwen4b_tied29_covariance_transport_nofit_20260628_230821` | covariance | no | `0/200` | `0/200` | `25/64` | `303,241,410.14` | repeated `THE`, `Fried`, fenced-code ticks, and stray fragments |
| `qwen4b_tied29_stats_transport_fit_mlpout_a003_20260628_231320` | stats | MLP output, alpha `0.03` | `0/200` | `0/200` | `7/64` | `258,154,649.12` | repeated token fragments; one merge-risk prompt produced empty output |

These ablations reject two tempting explanations: the collapse is not only from
covariance features, and it is not fixed by reducing the activation-fit alpha
from `0.15` to `0.03`. The shared risky ingredients are direct target-RMS
transport into the blank tied target and un-gated raw initializer promotion.
`hf-donor-project-init` now has optional `--post-init-eval-suite` controls so
raw initializer schedules can immediately write a broad behavior/generation
report and return compact metrics. Next work should avoid treating
donor-projected init as a final artifact. It should either:

- run initializer schedules through a behavior-gated ablation harness before
  recovery;
- split tokenizer/embedding transplant, attention/MLP transport, and
  reconstruction into separately ranked artifacts; or
- move to staged parent compression / chain anchors before the final blank
  SmolLM2 target, because one-jump raw transport is not producing a usable
  autoregressive manifold.
