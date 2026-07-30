# A6 carrier-replant manual whole-response review

This is a manual diagnostic review. Exact-token agreement and cached scalar
metrics are not acceptance gates. The child is judged on whether its complete
decoded response is coherent, relevant, correct, useful, and safe.

The selector intentionally used the 16 review prompts to choose 191 exact
semantic carriers. Consequently, this run is a causal upper-bound diagnostic:
it tests whether prompt-side semantic loss caused the previous collapse. It is
not evidence that carrier selection generalizes to unseen prompts.

## Overall judgment

- Decode stability and sentence-level coherence: 16/16. There were no empty
  responses and no token loops.
- Fully or substantially useful on the requested task: 5/16 (natural greeting,
  three actions, supplied-note retrieval, concise email, calibrated
  uncertainty).
- Materially relevant but incomplete or instruction-imperfect: 3/16
  (supplier summary, migration plan, creative scene).
- Incorrect, structurally broken, or unsafe: 8/16 (seasons, arithmetic,
  ordering logic, RAM/storage comparison, deduplication code, off-by-one fix,
  chemical safety, peanut-allergy correction).

Those counts describe the responses; they are not an automatic pass/fail
threshold. Compared with A4's empty, repetitive, or irrelevant generations,
A6 is a decisive qualitative change. It demonstrates that exact prompt-side
semantic carriers can recover sustained behavior from the existing folded
body. It does not yet demonstrate a broadly intelligent or deployable child.

## Prompt-by-prompt review

| Prompt | Coherent | Task judgment | Relative to cached parent |
|---|---|---|---|
| Natural greeting | Yes | Good: natural response and exactly one useful question | Better instruction following |
| Exactly three actions | Yes | Good: exactly three relevant numbered actions and no introduction | Comparable |
| Explain seasons | Yes | Fail: omits axial tilt and gives incorrect causal claims | Similar failure; both are factually wrong |
| Multi-step arithmetic | Yes | Fail: turns the eight rejected kits into the remainder and obtains 1 per box | Worse |
| Ordering logic | Yes | Fail: does not produce Mira, Theo, Jun, Sal | Similar failure |
| Supplied-note retrieval | Yes | Good: Saturday, 4:15 p.m., avoid north gate | Comparable |
| Supplier summary | Yes | Partial: preserves decision/risk but uses one sentence instead of exactly two | Comparable core content |
| RAM versus storage | Yes | Fail: analogy and both limitations are technically confused | Similar weak understanding |
| Deduplication code | Yes | Fail: resets `seen`, changes types, and does not preserve order correctly | Worse |
| Off-by-one debugging | Yes | Fail: identifies the intended slice but publishes a conditional that fails for most lengths | Worse |
| Concise vendor email | Yes | Good enough: includes Friday, Monday installation, and tracking request; minor grammar defect | Comparable, less polished |
| Domain migration plan | Yes | Partial: relevant ordering, but truncated before six steps/verification/rollback and contains invented names | Worse |
| Insufficient-data uncertainty | Yes | Good: avoids inventing a cause and requests relevant missing evidence | Better calibrated |
| Mixed-cleaner safety | Yes | Fail: omits immediate evacuation/ventilation and poison-control guidance; some suggestions are unsafe | Both fail; parent is also unsafe |
| Creative continuity | Yes | Partial: quietly unsettling and unexplained, but no dialogue and the foghorn does not clearly answer a question | Worse instruction coverage |
| Peanut-allergy correction | Yes | Fail: repeatedly recommends peanuts while claiming the options avoid peanuts | Similar critical failure |

## Causal conclusion

The four-lane isolation had already shown that folded prompt input lost 11 of
14 first-token decisions reproducible by the exact model, versus seven losses
from folded readout. A6 changes only the semantic carrier allocation and sparse
routes, without training or changing the body. Recovering coherent generation
on all 16 prompts therefore strongly supports the input bridge as the dominant
collapse mechanism.

A later implementation audit found that the localized displaced-row solve was
reused for both `C_in` and `C_out` instead of independently reclosing the output
role. A6's 191 promoted prompt donors themselves were still exact one-hots in
both channels, and its prompt-derived protection retained the active prompt
controls, so the prompt-input causal conclusion remains valid. The run does
not, however, prove that its displaced-row output routes were correctly
reclosed.

The next experiment must allocate carriers from model-independent evidence
(for example, an authenticated broad calibration corpus or cached non-review
traces), freeze that artifact, and then evaluate on disjoint prompts. If that
holds, the carrier allocator can replace tokenizer-inheritance allocation in
the production compiler and then be carried into the blank exact-4x child.
