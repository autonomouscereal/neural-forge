# LFM2 donor-clock A0 result and the dual-channel bridge gap

## Result

The completed A0 control evaluated the existing same-body, 4096-row child on
the donor semantic-token clock.  It used the exact raw ByteLevel span bridge,
one body step per donor token, 256-token response bounds, all 16 established
whole-response prompts, and the cached parent response envelope.

- Parent parameters: `229,693,184`
- Child model parameters: `166,778,624`
- Realized model-only compaction: `1.3772339554x`
- Parent model loads/forwards during review: `0`
- Training, optimizer, gradient, pruning, and quantization operations: `0`
- Undefined donor rows scored: `0`
- Target whole-string roundtrips: exact on all 16 responses
- Child state before/after review: unchanged

The child produced genuine English and sometimes tracked the prompt, so this
was not a random or wholly dead model.  It gave a fluent generic greeting and
several related fragments.  It did not retain useful broad intelligence:
numbered-list prompts frequently collapsed into repeated `1.` tokens, code and
debugging answers were malformed, arithmetic and context answers were wrong,
the creative response was empty, and the multi-turn response was `G catch`.
The result is therefore a completed negative control, not a successful child.

Primary evidence:

- `reports/semantic-clock-a0-same-body-v4096-20260730/whole-response-dossier.json`
- `reports/semantic-clock-a0-same-body-v4096-20260730/run-receipt.json`

## What the control proved

The raw span bridge and donor clock are now operational.  Prompt prefill uses
donor tokens, each generated donor action advances the body once, target
spelling spans never become body steps, and the exact output string is
retokenized only at the external boundary.  Dtype relocation and checkpoint
authentication also completed on CUDA BF16 without changing model bytes.

The remaining token failure is not a clock or spelling failure.  The current
runtime reuses the exact spelling operator `S` as the semantic linear operator:

```text
E_hat = S T
logits_hat = S (T h)
```

That equation assumes the learned embedding of a donor token is the positional
average of the target tokens that spell it.  Exact orthography does not imply
that learned geometric relationship.

Existing cached evidence isolates the consequence:

- direct-anchor parent winners retained: `293 / 293`
- multi-code parent winners retained: `0 / 72`
- mean cosine for uniform multi-piece semantic composition: about `0.1209`

Changing positional weights inside `S` or rerunning a single global residual
map does not address this defect.

## Next fold

Keep two separate channels:

```text
S: donor token -> exact target spelling span       (external text only)
C: donor token -> sparse target-anchor semantic code (body input/readout)
```

For each direct donor anchor, `C` remains an exact one-hot row.  Each remaining
donor embedding is folded locally over the existing 4096 planted anchor rows
using cached cosine candidates followed by a bounded residual/ridge solve.
The resulting tied algebra is:

```text
E_hat = C T
logits_hat = C (T h) = E_hat h
```

Candidate discovery is cached independently of active sparsity so `k=8`,
`k=16`, and `k=32` reuse the same expensive similarities.  Every continuous
coefficient is counted as child state; only integer offsets and indices are
structural metadata.  The exact raw span operator remains unchanged.

This dual-channel bridge runs before any protected/free table surgery because
it attacks the observed failure while preserving all 4096 known-good anchors.
If it is insufficient, the next bounded lane is a protected/free target-table
solve under the winning `C`, first `3072/1024`, then `2048/2048`.

After same-body coherence, the same `C` transfers through a right-side linear
width fold and will be installed in the already-proven exact-4x blank.  Its
continuous coefficients must be reserved inside that total parameter budget.

## A1/A2 sparse-code result

The dual-channel implementation was completed with exact one-hot direct rows,
all-4096 cosine candidate discovery, local ridge/refit coefficients, independent
candidate/coefficient caches, and FP32 coefficient accounting.

| Variant | C scalars | Effective parameters | Ratio | Cached hidden top-1 | Direct top-1 | Multi-code top-1 |
|---|---:|---:|---:|---:|---:|---:|
| A1, `k=8` | 486,544 | 167,265,168 | 1.37323x | 533/648 | 532/535 | 1/113 |
| A2, `k=16` | 968,992 | 167,747,616 | 1.36928x | 533/648 | 532/535 | 1/113 |

`k=16` improved embedding reconstruction relative to `k=8` but did not move
the cached hidden top-1 result.  Manual review rejected both.  A1 had a few
plausible outputs (notably the three-action and planning prompts), while A2
was mixed or worse, with empty responses, repetition, wrong arithmetic and
context, malformed code, and unsafe or irrelevant cleaner advice.

The A2 reconstruction receipt exposed the important geometry:

- non-direct median cosine: `0.42905`
- non-direct median reconstructed/source norm ratio: `0.42862`
- non-direct median relative L2 error: `0.90328`

The near equality of cosine and norm ratio is the expected orthogonal-
projection identity.  This is not primarily ridge shrinkage; most donor rows
contain semantic direction outside the span selected by one-shot top-cosine
neighbors.

## A3 coefficient-only norm reclosure result

A3 reused the authenticated A2 codebook byte-for-byte and performed no
candidate discovery or coefficient solve.  It baked
`scale=min(3, rho^-0.5)` into each existing non-direct coefficient, added zero
state, preserved direct one-hots bit-exact, and completed in `153.61 s`.

- non-direct median norm ratio: `0.42862 -> 0.65469`
- median applied scale: `1.52743`
- multi-code top-1: `1/113 -> 21/113`
- multi-code top-5: `44/113 -> 69/113`
- direct top-1: `532/535 -> 492/535`
- total top-1: `533/648 -> 513/648`
- total top-5: `579/648 -> 601/648`
- total MRR: `0.85048 -> 0.84535`
- non-direct rows with worse reconstruction L2: `60,253/60,306`

This is a useful negative diagnostic.  More energy makes multi-code tokens
competitive, but indiscriminate positive scaling also lets incorrectly
oriented multi-code rows displace protected direct anchors.  Whole-response
review was worse overall: greeting, email, and creative prompts remained
empty; arithmetic, logic, planning, uncertainty, and safety remained wrong or
repetitive; a few fragments became more specific but not reliably intelligent.
Do not run stronger global norm scaling as the next experiment.

Primary local evidence:

- `reports/sparse-semantic-a2-k16-rank-retro-20260730/cached-hidden-diagnostic.json`
- `reports/sparse-semantic-a3-k16-norm-gamma05-cap3-20260730/run-receipt.json`
- `reports/sparse-semantic-a3-k16-norm-gamma05-cap3-20260730/cached-hidden-diagnostic.json`
- `reports/sparse-semantic-a3-k16-norm-gamma05-cap3-20260730/whole-response-dossier.json`

## Revised next fold: residual-aware support selection

The next change must improve direction before gain.  For every non-direct
donor row, select the first anchor by residual correlation, refit the selected
support, recompute the residual, and search all 4096 anchors again.  Repeat to
the same bounded `k`, then perform the existing explicit FP64 final refit.
This is residual-aware orthogonal matching pursuit, not a larger one-shot
cosine pool and not trace fitting.  Cache each row chunk so support discovery
is reusable across downstream coefficient/runtime experiments.

Only after the residual-aware code improves manual whole responses should a
bounded row-specific gain calibration be considered.  The global A3 rule is
retained as a diagnostic control, not a candidate solution.

## A4 residual-OMP and four-lane isolation

A4 replaced one-shot candidate selection with residual-aware OMP at `k=16`.
It improved cached hidden behavior to `544/648` top-1 (`532/535` direct and
`12/113` multi-code), but manual review still rejected the child: none of the
16 responses completed the requested task, three were empty, and six looped.

A four-lane first-token control then separated prompt embedding and output
readout effects:

| Prompt input | Readout | Matches out of 16 |
|---|---|---:|
| exact | exact | 14 |
| exact | folded | 7 |
| folded | exact | 3 |
| folded | folded | 2 |

Mean terminal prompt-hidden cosine under folded input was `0.5528`, with mean
relative L2 error `0.8554`. Among the 14 exact-model-reproducible prompts,
folded input lost 11 first-token decisions while folded readout lost seven.
This falsified the assumption that one shared global semantic map was adequate
and identified `C_in` as the larger immediate failure.

## A5 asymmetric output refit

A5 kept A4 supports and fitted a separate `C_out` with per-row closed-form
solves on a cached functional feature pack. It performed no parent forward,
support search, backward pass, optimizer step, or training step. The selected
`0.001` functional weight retained `570/648` cached-hidden top-1 decisions,
including `527/535` direct and `43/113` multi-code decisions; held-out cached
rows retained `138/155`.

That was a useful output-side gain, but it did not repair the main prompt-side
failure. Exact-input first-token agreement with the exact-model winner remained
`9/16`; folded-input agreement was only `4/16`.

The runtime now binds authenticated, role-separated sparse maps:

```text
C_in  -> reconstruct embeddings for body input
C_out -> project target-table scores to donor logits
S     -> preserve exact external spelling only
```

Both coefficient vectors are counted as child state. BF16 and FP16 model
tables route through explicit FP32 accumulation; compile-time local solves use
FP64 only where required and publish FP32 coefficients.

## A6 semantic carrier replant diagnostic

A6 exploited the separation of spelling from semantics. A4 had 356 physical
target rows that no non-direct sparse route used. The diagnostic reassigned
191 such rows to the non-direct donor tokens occurring in the review prompts,
made those `C_in` routes exact one-hots, and recomputed only the 191 displaced
donor routes. Every other sparse route remained reusable.

- Reassigned physical rows: `191`
- Full-dictionary OMP searches: `191 * 16 = 3,056`
- Localized OMP time: about `0.12 s`
- Displaced-row mean relative L2: `0.000490989`
- Displaced-row mean cosine: `0.999999873`
- Parent forwards, training, backward calls, optimizer steps: `0`
- Effective child parameters: `168,716,608`
- Effective compaction at this same-body diagnostic stage: `1.361414x`
- Whole-response generation time after compilation: `84.71 s`

Manual review found coherent, non-looping responses on all 16 prompts. Five
were substantially useful, three were relevant but incomplete, and eight were
incorrect, broken, or unsafe. This is the first strong qualitative recovery
from the folded bridge, but not a completed child.

The selector deliberately used the review-prompt tokens, so A6 is a causal
upper bound rather than a generalization result. It proves that carrier
allocation can recover coherent behavior from the existing body; it does not
prove unseen-input coverage. The next fold must choose semantic carriers from
authenticated broad evidence, freeze the artifact, and evaluate on disjoint
prompts. Only that mechanism—not the prompt-specific selector—should be
promoted into the dynamic compiler and transferred to the blank exact-4x
child.

Primary evidence:

- `reports/sparse-semantic-a4-k16-residual-omp-20260730/`
- `reports/functional-output-refit-a5-20260730/result.json`
- `reports/carrier-replant-a6-prompt191-20260730/carrier-artifact.json`
- `reports/carrier-replant-a6-prompt191-20260730/whole-response-dossier.json`
- `reports/carrier-replant-a6-prompt191-20260730/manual-whole-response-review.md`

## A7 and A8 disjoint-corpus carrier selection

A7 froze carrier demand from 942 repository files (11,596,867 donor-token
occurrences, 16,302 unique donors). A8 froze it independently from a
deterministic sample of 1,024 year-coded 2025-2026 paper-text files
(80,198,516 bytes, 23,218,838 occurrences, 42,149 unique donors). Neither
selector used a review prompt, a cached parent response, a parent forward, or
training.

Both runs were constrained to the 356 target rows with zero non-direct
incidence across `C_in ∪ C_out`. Their held-out review-prompt coverage was:

| Bridge | Exact occurrences | Exact unique donors |
|---|---:|---:|
| Inherited A4 anchors | 635/885 | 246/437 |
| A7 repository corpus | 647/885 | 255/437 |
| A8 natural-language corpus | 646/885 | 256/437 |

Manual whole-response review rejected both children. A7 produced three
substantially useful responses and two partial responses; A8 produced two
substantially useful and three partial responses. Each had three empty
responses, and many nonempty responses were incorrect or entered repetition
loops. A8 therefore falsifies the idea that merely replacing the repository
corpus with much larger natural-language evidence can solve this bridge while
carrier capacity remains fixed at 356.

The result sharpens, rather than overturns, the A6 causal proof:

1. Exact demand-specific semantic carriers can restore sustained behavior
   without altering the folded body.
2. The inherited 4,096-row table plus only 356 free carriers cannot represent
   enough broad input semantics accurately at `k=16`.
3. Another 356-carrier selector is not an informative next experiment.
4. The next compiler step must either reassign additional low-incidence rows
   while recomputing every affected input/output donor route, or extend the
   cached residual-OMP prefix beyond `k=16` if reassignment fan-out is too
   expensive.

A7's localized repair took about `0.13 s`; A8's took about `0.15 s`. In A8,
the 356 displaced routes retained mean cosine `0.999999737` and mean relative
L2 `0.000718951`. These numbers prove that localized route reclosure itself is
cheap and accurate. They do not imply broad semantic quality.

Primary evidence:

- `reports/carrier-replant-a7-repository-corpus356-20260730/`
- `reports/carrier-replant-a8-natural-language-corpus356-20260730/`
