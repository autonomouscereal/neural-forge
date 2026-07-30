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
