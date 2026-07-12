# P35 behavior-space V/O tuner integration boundary

The behavior tuner lives in
`src/neural_forge/hf_residual_attention_behavior_tuner.py`. It operates on an
already-installed, tentative one-redirect registry and returns selected scale
data without installing, committing, or saving it.

## Required split contract

The public entrypoint requires three parent functional-trace mappings:

```python
tune_exact_p35_residual_attention_vo(
    runtime,
    tentative_registry,
    parent_fit_traces,
    parent_calibration_traces,
    parent_heldout_traces,
    controls=controls,
    trace_limits=limits,
)
```

Each mapping must contain both real cached greedy traces and forced traces
driven by the corresponding parent-greedy tokens. Prompt token rows must be
pairwise disjoint across all three sets.

- Fit outputs supply the two-scalar V/O gradients and nothing else.
- Calibration outputs choose between the untouched tentative incumbent and
  the fitted V/O candidate using the existing functional scorer.
- Heldout output values are first read after the calibration winner and
  candidate set are hashed and locked. Heldout produces an ordering but no
  winner.

The current six-prompt calibration and eight-prompt heldout files must not be
reused as fit data. A separate fit manifest is required.

## Actual p35 integration

The execution wrapper is
`experiments/run_qwen_p35_behavior_vo_tuner.py`. It reuses the same exact
tentative-codebook construction as
`experiments/run_qwen_p35_vo_role_scale_search.py` and owns the bounded
transaction that:

1. Resolves the exact `L19/G4 -> L21/G6` proposal.
2. Constructs the `280 -> 279` tentative codebook.
3. Collects the three parent trace sets once from untouched p35.
4. Installs the tentative registry and calls the differentiable tuner.
5. Restores the tentative role scales and then the original p35 runtime.

The tuner independently verifies:

- 36 logical layers and eight physical GQA groups;
- removed source `156` and survivor source `174`;
- affected physical assignments `L2/G1`, `L19/G4`, and `L21/G6`;
- exact one-prototype `280 -> 279` accounting;
- immutable physical-to-canonical mapping for target `L19/G4`;
- full role-scale and non-scale registry identity after rollback.

The wrapper returns the calibration-selected scales as report data only. It
does not leave them installed, commit the fold, or save a child checkpoint.

## Completed p35 learned-V/O run

The completed report is
`runs/residual-attention-20260711/p35_L19G4_to_L21G6_behavior_vo_tuner_h4_s8_fp16_p40_gpu2.json`.
It used horizon `4`, eight manual Adam steps, eight fit prompts, six
calibration-selection prompts, and eight heldout prompts. Q/K stayed `1.0`;
only the existing V/O scalars were trainable.

The optimizer did move in fit space:

- incumbent: `V=-0.0012493133544921875`,
  `O=-0.0005393028259277344`;
- fitted candidate: `V=0.040374755859375`, `O=-0.03515625`;
- fit loss: `1.3433064222` initially, `1.3409405947` at its minimum,
  and `1.3419187069` at the final recorded step.

That fit-loss movement produced **zero discrete behavioral gain**.
Calibration gave both candidates forced top-1 `19/24`, forced top-5 `24/24`,
and exact greedy/token/LCP retention `2/6`. The incumbent had the slightly
better mean parent-action log-probability, `-0.8790305615` versus
`-0.8793858894`, so calibration selected the incumbent
`behavior-vo-85714f5bd0528b152513`.

Heldout evaluation was winnerless and could not alter that lock. Both
candidates scored forced top-1 `28/32`, forced top-5 `32/32`, and exact
greedy/token/LCP retention `5/8`. The fitted point had only a tiny heldout
mean-log-probability edge, `-0.6842885071` versus `-0.6844520694`; no discrete
token outcome changed. This is not a repaired redirect and not promotion
evidence. The fitted point was rejected, the incumbent was retained, and no
fold or checkpoint was committed.

Exact accounting and integrity evidence:

- prototype lineage: `280 -> 279`, removed source `156`, survivor `174`;
- attention bytes: `1,835,011,432 -> 1,828,460,132`, saving `6,551,300`;
- one prototype: `3,276,800` parameters / `6,553,600` bytes;
- role-scale registry: `1,152` parameters / `2,304` bytes;
- net learned-parameter reduction: `3,275,648`;
- tentative model parameter count remained `3,992,294,528` throughout tuning;
- runtime parameters added: `0`; saved optimizer state: none;
- tentative-child rollback and incumbent-p35 rollback: exact;
- file SHA-256:
  `86b0d9c500315d867a8097142956fd5aee22d3c207b9f683060ca9eaf93f8632`;
- top-level report hash:
  `7d48b24a089c3bd85915a71bfa6f362c8500eb027d111cf31c4a8dfed52f9d2e`;
- nested tuner report hash:
  `dc7242445d2ef529d92be42ae9e0a251afcaaa684718a15d3756a615a07bab31`;
- training-history hash:
  `9281673a98b4f5f046158f3f7a74123351053ca3132c9c541c634c470c0e09cf`;
- calibration ranking hash:
  `61514add8801ff26cc8c173ffd113b7e52ab1900c826ff8cf128df16e3d6f1be`;
- calibration selection-lock hash:
  `8d8999b784e75cdcd4a0f4fc2e16f49c7250f16590e7cdde2e5223182379be77`;
- heldout ordering hash:
  `00594215fa490ab94a39100575311270e56a35e674a60c9513b430e48fb23072`.

## Promotion boundary

The search result is evidence, not a checkpoint. It contains selected Q/K/V/O
scale data, but Q/K are unchanged and only V/O were searchable. A later,
explicit commit workflow may install the calibration winner only after its
heldout result is acceptable. That workflow must then save/reload and run the
broad generation, behavior, choice, and perplexity evaluations.

This report covers one redirect. The full p35-to-p34 transition requires eight
such redirects; one prototype must never be reported as the entire roughly
26.2-million-parameter transition.
