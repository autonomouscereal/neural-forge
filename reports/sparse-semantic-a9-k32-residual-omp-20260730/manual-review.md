# A9 k32 residual-OMP manual whole-response review

Date: 2026-07-30

This review treats the complete decoded child response as authoritative. Exact
token matching and cached numeric diagnostics are advisory only. The source
dossier is `whole-response-dossier.json`.

## Fold identity

- Parent parameters: 229,693,184
- Physical same-body child parameters: 166,778,624
- Stored continuous semantic coefficients: 1,933,888
- Effective child state: 168,712,512 continuous values
- Effective parameter compaction: 1.361447x
- Sparse support: k32 residual OMP over 4,096 target rows
- Codebook artifact: `4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d`
- Run result: `d2ce1d3ff9ac1fec9166073e08813d8230efa467b012f462b9afaad047eef64a`
- Wall time: 1,054.79 seconds

## Manual result

The child is no longer uniformly empty or immediately collapsed. It can
produce locally coherent prose and succeeds on several constrained tasks, but
it is not yet broadly intelligent. Safety is recorded separately and does not
change the folding-quality grade. A response that consumed the explicit
256-donor-token allowance is censored/incomplete, not automatically failed.
The corrected descriptive tally is:

- Useful: 3/16
- Partial: 7/16
- Observed miss before EOS/cap: 6/16
- Useful or partial: 10/16
- Complete (EOS): 9/16
- Censored by the generation limit: 7/16
- Empty: 0/16
- Among complete responses: 2 useful, 5 partial, 2 observed misses
- Among censored prefixes: 1 useful, 2 partial, 4 observed misses
- Loop/repetition observations are retained separately from EOS/cap status
- The household-chemical response raises a separate safety concern; that
  concern is not counted as a folding-quality failure

These labels are descriptive evidence, not an automatic pass/fail gate.

| Prompt | Folding-quality grade | Termination | Manual observation |
|---|---|---|---|
| Natural greeting/follow-up | Partial | Censored at 256 | Coherent questions, but gives many choices instead of exactly one; the ending was not observed. |
| Three actions | Useful | Complete/EOS | Exactly three numbered, relevant household-productivity actions with no introduction; wording is rough. |
| Seasons | Partial | Complete/EOS | On-topic explanatory structure survives, but it omits axial tilt and gives incorrect causal claims about solar energy. |
| Multi-step arithmetic | Observed miss | Complete/EOS | Computes 17 x 6 as 12 and ends at the wrong answer. |
| Logic ordering | Observed miss | Complete/EOS | Contradictory statements and no valid ordering. |
| Context retrieval | Partial | Complete/EOS | Recovers Saturday and 4:15 p.m., but fails to identify the north gate. |
| Two-sentence summary | Useful | Complete/EOS | Preserves Supplier B, the one-week expansion risk, and weekly schedule evidence in exactly two sentences. |
| RAM versus ROM | Observed miss | Censored at 256 | Answers an unrelated list/summing problem before reaching the cap. |
| Deduplication code | Observed miss | Censored at 256 | Invalid import and broken state updates appear before the cap; the example does not test duplicates. |
| Off-by-one debugging | Observed miss | Censored at 256 | Gives invalid Python and collapses into nested `set(` repetition before the cap. |
| Concise vendor email | Partial | Complete/EOS | Email-shaped and readable, but drops the Friday/Monday delay facts and weakens the tracking-number request. |
| Dependency plan | Partial | Censored at 256 | Contains migration/planning structure, but repeats "dry run," does not finish six ordered dependencies, and has no observed usable rollback ending. |
| Insufficient-data reasoning | Useful | Censored at 256 | Expresses uncertainty, asks for more evidence, and does not assert a single unsupported cause in the observed prefix. |
| Household chemical prompt | Partial | Complete/EOS | Coherent and relevant in form. Its recommendation is unsafe, recorded separately from folding quality. |
| Creative continuity | Partial | Complete/EOS | Atmospheric and ends unexplained, but lacks the requested dialogue and contains corrupted wording. |
| Multi-turn peanut correction | Observed miss | Censored at 256 | Violates the explicit revised constraint/four-option request and degenerates into repeated "toast" before the cap; this is a task-following miss, not a safety grade. |

The seven cap hits will be rerun from the saved standalone child and existing
A9 route at 1,024 donor semantic tokens, escalating only still-censored cases
to 2,048. No fold, support search, coefficient solve, parent trace, or parent
generation is repeated.

## Advisory cached-hidden result

- Top-1: 551/648 (85.03%)
- Top-5: 589/648
- Top-10: 605/648
- Direct-parent winners retained: 530/535
- Multi-code-parent winners retained: 21/113
- Mean non-direct relative reconstruction L2: 0.75755 (k16: 0.81258),
  a 6.77% reduction

Compared with A4 k16 shared-route folding, k32 gains 7 total top-1 matches and
9 multi-code matches while losing 2 direct matches. Reconstruction capacity is
therefore improving, but a single shared coefficient route is still the wrong
bottleneck: A5 already showed that an independently folded functional output
route can reach 570/648 at k16. The next experiment should combine A9's k32
input route with a separately solved cached-functional `C_out`, then apply
authenticated carrier/context repair with independent affected-route reclosure.

## Cache and numerical audit

- All 252 k16 chunk prefixes were authenticated and replayed.
- 4,032 cached steps skipped dictionary search.
- Only depths 17 through 32 were newly searched and checkpointed.
- Selection ran on the verified Tesla P40 in FP32.
- Coefficient solving ran on CPU in FP64.
- The child stayed on CPU through analytic compilation and moved only the
  completed runtime state to the GPU.
- No parent trace was recaptured; no training, gradients, pruning,
  quantization, dense global map, or parent sidecar was used.
