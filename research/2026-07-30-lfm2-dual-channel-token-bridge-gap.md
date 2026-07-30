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
