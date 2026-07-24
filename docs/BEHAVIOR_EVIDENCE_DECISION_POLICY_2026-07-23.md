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

The independently generated `fit-residual-output-pod-rank128-scale0` dossier
has file SHA-256
`ec34c7097eb758145f923bebf7b794a471d7285b4c449e0af8662d72fd60d8e6`
and logical dossier SHA-256
`7cb9c011b92ec992db717bfc7c92d5c60042773ddd079a9324c68a93fc7c5bb9`.
Its canonical response set is byte-distinct from all 37 previously retained
candidate response sets. Direct review again covered all 37 candidates across
conversation, instructions, explanation, arithmetic, logic, context,
summarization, comparison, code, debugging, writing, planning, uncertainty,
safety, creative continuity, and multi-turn correction. Candidates `0-12`
were dominated by repetitive token attractors and dense lexical soup;
candidates `13-18` became more locally word-like or template-shaped but never
formed connected task answers; and candidates `19-36` remained disjoint,
cross-contaminated fragments. No coherent or task-completing island was
observed in this separate surface. This is retained as qualitative mechanism
evidence, not as a binary candidate veto.

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

## Same-body token-interface control retained on 2026-07-23

The authenticated bridge-only control held the complete 14-layer LFM2 body
fixed and changed only the vocabulary interface from 65,536 parent tokens to
4,096 child tokens. The parent has 229,693,184 unique dense parameters and the
freshly reloaded child has 166,778,624, a 1.377233955x reduction. All 133 body
tensors were byte-identical to the parent; input/output tying remained native;
and the run performed zero parent forward passes, backward calls, optimizer
steps, and training steps. The whole-response dossier has file SHA-256
`f4aab740297f78e7272e9d161149c0b81077acf73b9afa7f831c105e2a149dff`
and logical dossier SHA-256
`fdc27135ac7a581dc6ba4d91d1f476cd7420840ebc79267a9c6e8e729055cb4b`.

Direct review of all 16 continuations found a qualitatively different failure
from the exact-4x body-fold frontier: recognizable English, prompt topic, and
some useful behavior survived. The greeting was coherent and the supplier
summary was substantially useful. However, arithmetic, ordering, explanation,
instruction following, code, safety, context use, and multi-turn behavior were
wrong or unstable; repetition remained common; and three tasks terminated with
an empty response. Therefore the 4,096-token bridge is a real secondary
bottleneck, while the additional lexical-soup collapse in the exact-4x children
shows that body folding is a separate and larger bottleneck. This observation
does not impose a score cutoff. It motivates a dynamic interface frontier:
compositional parent-subtoken input synthesis, asymmetric input/output maps,
closed-form cached activation/logit fitting, and EOS/norm calibration before
the interface is recombined with the corrected body fold.

## Residual-closure materialization finding retained on 2026-07-23

Receipt and checkpoint comparison found that the structural rank-128/scale-1,
fit rank-128/scale-0.5, fit rank-776/scale-1, fit rank-128/scale-1, and
consequence scale-0 closure artifacts all installed no physical parameter
changes. The FP64 analytic coefficients were nonzero, but adding their deltas
and casting the result back to BF16 rounded every corrected value to its seed:
`changed_physical_parameter_ids` was empty and every live before/after hash was
identical. Those runs consequently did not test residual closure at all, and
their similar decoded behavior cannot be interpreted as evidence against the
analytic mechanism. The active repair derives representable dense corrections
from target-dtype and weight geometry, records pre-cast versus installed
deltas, and forbids publication of a nonzero analytic correction as a silent
physical no-op. This is a transaction-authenticity requirement, not a
behavioral threshold.
