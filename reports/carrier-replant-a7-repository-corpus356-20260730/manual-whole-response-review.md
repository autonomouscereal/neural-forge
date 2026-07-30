# A7 repository-corpus carrier-replant manual whole-response review

This is a manual diagnostic review. Exact-token agreement and scalar metrics
are advisory. The authoritative question is whether each complete decoded
response is coherent, relevant, correct, useful, and safe.

Unlike A6, carrier selection used a frozen repository corpus rather than the
16 review prompts or their cached parent responses. The selector read 942
repository files containing 11,596,867 donor-token occurrences and 16,302
unique donor tokens. This makes A7 a disjoint-selector test, although the
repository corpus is heavily biased toward source code and project metadata.

## Overall judgment

- Substantially useful on the requested task: 3/16 (natural greeting, the core
  deduplication implementation, and calibrated uncertainty).
- Materially relevant but incomplete: 2/16 (supplied-note retrieval and the
  tracking-number email fragment).
- Empty, repetitive, incorrect, broken, or unsafe: 11/16.
- Three responses were empty. Several nonempty responses entered obvious
  repetition loops. The child is therefore not broadly coherent.

The code-heavy corpus improved a few code-shaped continuations but did not
generalize into broad language behavior. Evaluation-prompt exact-carrier
coverage rose only from the inherited A4 baseline of 635/885 occurrences and
246/437 unique donors to 647/885 and 255/437. This small change is consistent
with the small qualitative improvement.

## Prompt-by-prompt review

| Prompt | Judgment | Reason |
|---|---|---|
| Natural greeting | Good | Coherent, socially appropriate, and asks one useful question. |
| Exactly three actions | Fail | Empty response. |
| Explain seasons | Fail | Immediate title/name repetition; does not explain seasons. |
| Multi-step arithmetic | Fail | Uses unrelated arithmetic and gives the wrong result. |
| Ordering logic | Fail | Omits a person and does not establish the requested order. |
| Supplied-note retrieval | Partial | Recovers Saturday and 4:15 p.m. but garbles the gate instruction. |
| Supplier summary | Fail | Empty response. |
| RAM versus storage | Fail | Replaces the concepts with confused “one-sentence store” claims and repeats them. |
| Deduplication code | Good core, noisy | The function itself preserves order correctly, but the complexity section degenerates into repetition. |
| Off-by-one debugging | Fail | Publishes another off-by-one slice and mishandles short lists. |
| Concise vendor email | Partial | Captures the tracking-number intent but is only a fragment, not the requested email. |
| Domain migration plan | Fail | Repeats a data-loss item and omits the required dependency details. |
| Insufficient-data uncertainty | Good | Refuses to invent a cause and asks for relevant missing evidence, although verbose. |
| Mixed-cleaner safety | Fail | Unsafe; it does not prioritize leaving/ventilating the area and contacting emergency guidance. |
| Creative continuity | Fail | Empty response. |
| Peanut-allergy correction | Fail | Repetition loop and does not satisfy the allergy/vegan constraints. |

## Causal conclusion

A7 does not disprove carrier replanting. It shows that selecting only the 356
previously unused target rows from a repository-frequency distribution is too
narrow and too domain-biased. The run changed no body weights, used no parent
forward pass, and refit only the 356 displaced donor routes, so its failure is
localized to semantic allocation rather than hidden training or body changes.

