# Behavioral Evidence Decision Policy

Updated: 2026-07-23

## Scope

This policy governs the active dynamic blank-target folding sequence, beginning
with the exact-4x LFM2 control and extending to larger ratios, cross-family
adapters, tokenizer bridges, and additional modalities.

## What may stop an artifact transaction

Structural and provenance failures remain hard failures. Examples include an
incorrect dense parameter count, a nonblank target seed, missing or mutated
cached evidence, NaN/Inf parameters, topology mismatch, use of training,
pruning, quantization, sparsity, sharing, a parent sidecar, or failure to save
and freshly reload a standalone child. These checks establish that an artifact
is the requested fold; they do not estimate intelligence.

## What may not stop or decide a folding lane

None of the following has automatic accept, reject, rollback, deletion,
publication, or stop authority:

- exact token or text match;
- first-token agreement;
- perplexity, RMSE, cosine similarity, rank, activation fit, or attention-score
  reconstruction;
- task-score aggregates or a computed intelligence-retention percentage;
- repetition, empty-output, EOS, or generation-limit heuristics in isolation;
- the approximately 90% retention objective; or
- one binary human or agent verdict.

These measurements remain useful diagnostics. They can prioritize inspection,
identify a failing circuit, order the next experiment, or motivate a different
folding mechanism. They cannot substitute for the complete behavioral record.

## Primary behavioral evidence

Every structurally valid candidate in the active frontier is retained and its
complete standalone decoded responses are preserved. Parent-relative review
considers the full responses, not wording identity, across at least these
dimensions:

- coherence across the complete continuation;
- task correctness and factual behavior;
- reasoning and useful intermediate structure;
- instruction following;
- practical usefulness;
- safety and appropriateness;
- context retention and truncation effects;
- retained parent-demonstrated capability; and
- concrete failure modes such as hallucination, repetition, context loss,
  premature termination, or code/arithmetic failure.

Observations may summarize strengths, regressions, and hypotheses, but worksheet
completion is not a continuation gate and no candidate winner is required.
Roughly 90% useful intelligence retention remains the directional objective.
It is reported as a holistic target, not certified by crossing one threshold.

## Active implementation boundary

The active path is:

1. `experiments/run_lfm2_semantic_fold_to_review.py` folds and retains every
   structurally eligible dense child;
2. `experiments/review_lfm2_semantic_frontier.py` generates and preserves all
   complete decoded responses; and
3. `experiments/prepare_lfm2_semantic_qualitative_review.py` renders the evidence
   and emits an optional multidimensional observation worksheet.

Historical modules and reports containing names such as `gate`, `accepted`,
`rejected`, `promotion`, or `verdict` preserve older experiments and artifact
schemas. They are not decision-authority dependencies of this active path.
Compatibility field or filename aliases do not restore their former gating
semantics.

## Exact-4x direct-review evidence retained on 2026-07-23

The `fit-rank128-scale1` whole-response dossier has file SHA-256
`5ca1744e0d8ac14bf4b74343c21d7ce556a834121688bfd1034171b6a68adec5`.
Two independent review passes read all 37 candidates and all 16 responses per
candidate (592 complete continuations). No response formed a usable task
answer or a sentence-level coherent island. Earlier candidates showed hard
token/subtoken attractors and repetition; middle candidates showed
prompt-conditioned lexical soup; later candidates produced more recognizable
English fragments but remained non-compositional, malformed, and incapable of
conversation, instruction following, arithmetic, logic, context use, coding,
or reliable safety guidance. Alpha and ridge changes altered surface form but
did not restore intelligence.

This is a mechanism diagnosis, not a numeric rejection threshold. The dossier
and every candidate checkpoint remain evidence. It rules out spending another
run on the same alpha/ridge surface and directs the next fold toward a causal
mechanism change. The active change is full-history residual-attention folding:
true donor FIR context is computed before tokenizer-owner expansion, inserted
child subtokens remain in causal warmup, and the FHVR hybrid uses the compiled
residual-writer candidate rather than a structural control.

The audit specifically classified the threshold-driven engines in
`dynamic_behavior_gate.py`, `hf_lfm2_intelligence_gate.py`,
`hf_lfm2_parent_relative_manual_gate.py`,
`dynamic_lfm2_cpcf_reviewed_gate.py`, `hf_lfm2_macro_cell_attractor_gate.py`,
and the historical Qwen compensated/unified/native-stage runners as legacy
evaluation or transaction machinery. The active three-file path above imports
none of those behavioral decision engines. They remain readable so earlier
reports can be reproduced, but they must not be registered as the behavioral
decision policy for new dynamic-fold experiments.
