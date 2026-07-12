# P35 -> p34 behavior-first residual-attention prompts (2026-07-11)

This package converts the preserved p35 and p34 broad generation evaluations
into exact behavioral targets for residual-attention selection and held-out
evaluation. No prompt was written or paraphrased for this package. Every prompt
is copied byte-for-byte from the shared 1,000-case generation inventory embedded
in both run reports.

## Exact transition denominators

The comparison joins on generation case ID and requires the prompt and source
category to match in both reports before classifying a transition.

| p35 -> p34 outcome | Total | Code | Tokenizer | Model-surgery | Math | Commonsense |
|---|---:|---:|---:|---:|---:|---:|
| lost: pass -> fail | 228 | 125 | 87 | 15 | 1 | 0 |
| gained: fail -> pass | 12 | 0 | 0 | 11 | 1 | 0 |
| stable pass | 405 | 158 | 0 | 106 | 141 | 0 |
| stable fail | 355 | 2 | 56 | 154 | 0 | 143 |

Thus p35 passes 633/1,000 cases, p34 passes 417/1,000, and the net change is
-216 passes. The complete 228 lost and 12 gained records, including exact ID,
category, prompt text, and prompt SHA-256, are in
`experiments/p35_p34_generation_transitions_20260711.json`.

Source bindings:

- p35 report SHA-256:
  `eb6fbaee748e03a70a6e78f6e82a087f655fca38b0cbab2c41fb95d84451b9c8`
- p34 report SHA-256:
  `2f9ef2a7887874b8e15119f304e272d889d93d71ae69ef205f46a100f5d5653b`
- Shared embedded generation inventory SHA-256:
  `048b27443a4313218a12f6c73db59616ced8542b184b3424752c963a6e9a398c`
- Transition inventory SHA-256:
  `212ebf3d9350c1d48ad77a71616d39d5d163dd022a7ac95b9bb0d4c03f0b929a`

The source category `model_surgery` is displayed as `model-surgery`; the source
spelling remains bound in the prompt manifest.

## Curated sets

The prompt files remain compatible with the existing probe loaders: each is a
plain JSON array of exact prompt strings. IDs, roles, category strata, output
hashes, selection ranks, and provenance live in the sidecar manifest.

### Calibration-selection: 6 prompts

This set is evenly split between three p35 -> p34 degradation targets and three
stable-positive controls.

| Role | Category | Suite ID |
|---|---|---|
| degradation target | code | `generation_python_add_0569` |
| degradation target | tokenizer | `generation_tokenizer_mismatch_0542` |
| degradation target | model-surgery | `generation_model_merge_risk_0309` |
| stable-positive control | code | `generation_python_add_0233` |
| stable-positive control | math | `generation_math_answer_0467` |
| stable-positive control | model-surgery | `generation_model_merge_risk_0890` |

File:
`experiments/p35_p34_residual_attention_calibration_selection_prompts_20260711.json`

SHA-256:
`a2eaea993c7f7f434e30744eeaf4b4da750b9a327d835deedb385dbaf154e406`

### Heldout-evaluation: 8 prompts

This set is evenly split between four degradation targets—one from every actual
loss category—and four stable-positive controls.

| Role | Category | Suite ID |
|---|---|---|
| degradation target | code | `generation_python_add_0345` |
| degradation target | tokenizer | `generation_tokenizer_mismatch_0241` |
| degradation target | model-surgery | `generation_model_merge_risk_0995` |
| degradation target | math | `generation_math_answer_0474` |
| stable-positive control | code | `generation_python_add_0744` |
| stable-positive control | code | `generation_python_add_exact_0713` |
| stable-positive control | math | `generation_math_answer_0936` |
| stable-positive control | model-surgery | `generation_model_merge_risk_0449` |

File:
`experiments/p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json`

SHA-256:
`cebf98bdbdfc1154c45758cdeed25e15141b6aeb910f06fa6cf349efdaf6d04f`

Across both sets, degradation targets contain two code, two tokenizer, two
model-surgery, and the sole math loss. The math loss is held out rather than
duplicated into calibration. There are no tokenizer stable-pass cases in the
source denominator, so this package does not mislabel a stable tokenizer failure
as a positive control.

## Selection and leakage contract

Cases are ranked deterministically using seed
`neural-forge-residual-attention-behavior-first-20260711` and a SHA-256 over the
split, transition, source category, optional prompt family, suite ID, and exact
prompt. Calibration is selected first; heldout is then selected without
replacement by ID or exact prompt text.

Verified leakage counts:

- calibration/heldout ID overlap: 0
- calibration/heldout exact-prompt overlap: 0
- overlap with the current tiny V-probe sets: 0 exact and 0 normalized prompts
- selected IDs: 14/14 unique
- selected exact prompt texts: 14/14 unique

The full machine-readable contract is
`experiments/p35_p34_residual_attention_prompt_manifest_20260711.json`, SHA-256
`d6d4d326cc55d864d388bda6220fe09df70a6ec7ffe4664d4277af53857a00d1`.

This is a prompt-curation artifact only. It performs no model run, fitting,
checkpoint mutation, or promotion decision.
