# Behavioral Evidence Decision Policy

Updated: 2026-07-30

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

### Safety and termination are separate evidence dimensions

Safety observations do not count as folding-quality failures. Folding quality
is reviewed for coherence, correctness, instruction following, useful
intelligence, and parent-relative retained behavior. Safety/appropriateness is
recorded alongside that grade so it is not lost, but it cannot change the
folding-quality label or compaction conclusion.

Reaching an evaluator's `maximum_new_tokens` is an observation boundary. The
response is marked **censored/incomplete**, never automatically failed. Manual
review may still record an observed useful, partial, or miss prefix when the
evidence before the cap already supports that description. EOS, cap hit,
manual loop/repetition observation, and runtime error are retained as separate
fields. A cap hit does not imply a loop, and a loop is not inferred merely from
length. Capped cases should be rerun from the same saved child at a materially
larger allowance without refolding or regenerating parent traces.

## Primary behavioral evidence

Every structurally valid candidate in the active frontier is retained and its
standalone decoded observations are preserved, including explicit censoring
metadata when the evaluator cap is reached. Parent-relative review considers
the observed responses, not wording identity, across at least these dimensions:

- coherence across the complete continuation;
- task correctness and factual behavior;
- reasoning and useful intermediate structure;
- instruction following;
- practical usefulness;
- context retention and output-cap effects;
- retained parent-demonstrated capability; and
- concrete failure modes such as hallucination, repetition, context loss,
  premature termination, or code/arithmetic failure.

Safety and appropriateness are reported in their own parallel observation
dimension under the policy above.

## A9/A11 corrected same-body review retained on 2026-07-30

The A9 k32 residual-OMP child has 168,712,512 counted continuous values after
including all 1,933,888 sparse semantic coefficients, for 1.361447x effective
compaction versus the 229,693,184-parameter parent. Under the corrected policy,
manual folding-quality observations are 3 useful, 7 partial, and 6 observed
misses. Nine responses reached EOS; seven hit the 256-donor-token cap and are
censored. The capped prefixes comprise 1 useful, 2 partial, and 4 already
observed misses. This is descriptive evidence, not a score gate.

The A11 split-route child counts 170,646,400 continuous values, for 1.346018x
effective compaction. It has 3 useful, 9 partial, and 4 observed misses. Ten
responses reached EOS; six are censored. Compared with A9, arithmetic,
ordered-deduplication, and off-by-one debugging improve from observed miss to
partial, while seasons regresses from partial to observed miss. A11 is a modest
behavioral interface improvement, not proof that cached parent-state C_out
repair solves autoregressive C_in trajectory drift.

The household-chemical responses raise safety concerns, recorded separately.
The peanut-correction responses are folding-quality misses because they violate
the explicit revised constraint/four-option request and degenerate; they are
not counted through the safety dimension. Only the A9/A11 cap-hit prompts are
scheduled for reuse-only 1,024-token evaluation, escalating still-censored
cases to 2,048, from the saved standalone child and compiled routes.

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

The corresponding old scale-1 consequence run produced another independent
dossier with file SHA-256
`1d07e9e25b6a78809d323736c6122713b19583dea7017444fd3bfc206a133b4d`
and logical dossier SHA-256
`540ece1ef2c24f7ea6bb825c0f1b75bd54f8103705de23fdeb45a565bb8b27ce`.
Two review
passes again covered all 37 candidates and all 16 task categories. Candidate
`0` collapsed into single-token attractors; candidates `1-12` emitted dense,
prompt-varying pseudo-English word salad; candidates `13-18` had more varied
local English fragments but did not compose an answer to any requested task;
and candidates `19-36` remained lexical fragments without conversation,
reasoning, instruction following, or task completion. No usable coherent
island was observed. This describes the decoded mechanism failure; it is not a
numeric or binary continuation gate.

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

An untied `4,096 x 776` output table adds `3,178,496` real dense parameters to
the tied exact-4x body. The interface-only control may exceed the final budget
because it isolates this mechanism, but the composed child may not. The
budget-preserving target keeps residual width `776`, all 14 layers, and the
`10Q/5KV` attention geometry while reducing the shared SwiGLU intermediate
width from `797` to `700`. That removes
`97 x 3 x 776 x 14 = 3,161,424` parameters, producing an untied child with
`57,422,440` unique dense parameters. Against the `229,693,184`-parameter
parent this is approximately `4.000059x` compaction and sits 856 parameters
below the exact one-quarter ceiling. This is structural budget accounting,
not a behavioral preference or quality gate.

The inverse output map must also preserve variable-length tokenizer causality.
Aggregating every donor token only into the first token of its target-token
decomposition preserves initial mass but discards which continuation is valid
after that first token. The complete bridge therefore uses the decomposition
prefix trie: initial target-token probability is donor mass summed by first
token, and each inserted target position receives donor mass conditioned on
the already emitted decomposition prefix. The existing expanded target-token
histories are the intended causal feature rows for these conditional output
targets. A first-token-only recipe remains a useful ablation, not the final
cross-tokenizer mechanism.

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

## Cache-only interface frontier retained on 2026-07-23

The exact-body cache-only interface frontier materialized six standalone dense
children and preserved all 96 complete continuations. Its run-receipt file has
SHA-256
`cd28c7f539ab69fe2e73a508b60e01c5a0651a2400f20c49476beea557b757bf`;
the whole-response dossier file has SHA-256
`8335e4303e4518a42ba5acefca1f87e212de44098deb3653bdd6181e5950c098`
and logical dossier SHA-256
`e502abded7a951bdcf7725687f51151ae4ecbf9b39f6afeafd188947634a4e2a`.
The authenticated terminal cache evidence has SHA-256
`d2628d91eaab6d7ce972a79871774056b771f3dafc56fd4858dc622dc9ed8a8f`.
The run used no parent forward pass or recapture, training, backward call, or
optimizer step.

Direct review read every parent and child response. The tied contextual-input
control was qualitatively strongest: it retained fluent greeting behavior,
topic recognition, an exactly three-item instruction-following island, and
some useful uncertainty and creative-language structure. Its behavior was
still shallow and unstable across arithmetic, logic, retrieval, explanation,
coding, debugging, planning, safety, and multi-turn correction. The two
structural untied output maps and all three cached-logit residual heads were
materially less stable. They showed prompt copying, malformed fragments,
token attractors, repetition, invalid code, and incomplete reasoning, with a
few isolated topical or form-following continuations but no broad competence
gain over the tied control.

The cached parent envelope is itself an unreliable answer key on several
probes: it gives the wrong causal explanation for seasons, the wrong remainder
for `94 / 5`, a malformed ordering conclusion, unsafe chemical-mixing advice,
and peanut-containing suggestions after being told of severe peanut allergy.
Consequently, exact token or answer imitation would sometimes reward worse
behavior. Parent responses remain comparative evidence alongside task facts
and decoded usefulness; they are not an oracle or a hard target.

The mechanism implication is narrower than a candidate verdict. A static
untied output table solved independently from the live residual stream damages
the surface behavior that the tied interface preserves. The next interface
fold therefore keeps the tied contextual control as a reference and makes any
asymmetric output bridge conditional on causal decomposition-prefix state and
the child's evolving residual features. It must be compiled from authenticated
cached evidence and materialized into the standalone child. All six frontier
checkpoints remain retained evidence; this observation only directs the next
experiment.
