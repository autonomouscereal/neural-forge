# A8 natural-language carrier-replant manual whole-response review

This is a manual diagnostic review. Exact-token agreement and scalar metrics
are advisory. The authoritative question is whether each complete decoded
response is coherent, relevant, correct, useful, and safe.

Carrier selection was frozen before review from a deterministic sample of
1,024 year-coded 2025-2026 paper-text files: 80,198,516 source bytes,
23,218,838 donor-token occurrences, and 42,149 unique donor tokens. It used
neither the review prompts nor cached parent responses. The selection-evidence
content hash is
`0ea9140784502a9bf2ee76bf4c5f7d40f88b0194f720a5f57e4535f559827d1c`.

## Overall judgment

- Substantially useful on the requested task: 2/16 (natural greeting and
  calibrated uncertainty).
- Materially relevant but incomplete: 3/16 (supplied-note retrieval, concise
  email intent, and the beginning of a migration plan).
- Empty, repetitive, incorrect, broken, or unsafe: 11/16.
- Three responses were empty. Many of the remaining responses repeated clauses,
  list entries, or code. This is not a coherent or intelligent child yet.

The natural-language corpus did not materially outperform A7. Exact-carrier
coverage on the held-out review prompts was 646/885 occurrences and 256/437
unique donor tokens, versus A7's 647/885 and 255/437 and the inherited A4
baseline's 635/885 and 246/437. A corpus with tens of millions of tokens cannot
repair this particular 356-carrier configuration by corpus substitution alone.

Post-run audit found two confounders. First, `tokenizer.all_special_ids`
exposed only a fraction of the tokenizer/control contract. Of the 356 target
rows reclaimed here, 348 mapped to reserved special placeholders (IDs 16-500).
None of the active chat/control IDs 0-13 or 64394-64401 was evicted, so this
does not by itself explain the observed text-prompt collapse; nevertheless,
reserved rows must be explicitly classified as protected or authenticated
reclaimable state. Second, the runner solved each displaced row in input
embedding space once and reused those coefficients for both `C_in` and
`C_out`. That overwrote A5's functional output coefficients on the displaced
rows and violated the role-separated bridge contract.

## Prompt-by-prompt review

| Prompt | Judgment | Reason |
|---|---|---|
| Natural greeting | Good | Coherent, socially appropriate, and asks one useful question. |
| Exactly three actions | Fail | Empty response. |
| Explain seasons | Fail | Repeats a meta-response and gives no valid explanation. |
| Multi-step arithmetic | Fail | Structured reasoning, but computes 17 x 6 as 98 and ignores boxing. |
| Ordering logic | Fail | Contradictory ordering statements and the wrong answer. |
| Supplied-note retrieval | Partial | Recovers Saturday and 4:15 p.m. but fails to identify the north gate. |
| Supplier summary | Fail | Empty response. |
| RAM versus storage | Fail | Discusses “one-sentence memory” instead of RAM and storage, then repeats itself. |
| Deduplication code | Fail | Resets state inside the loop, returns a set, and enters complexity repetition. |
| Off-by-one debugging | Fail | Uses invalid Python and degenerates into nested-set repetition. |
| Concise vendor email | Partial | Captures the tracking request but is not an actual email and omits key details. |
| Domain migration plan | Partial | Starts with relevant dependencies but repeats TLS setup and omits rollback detail. |
| Insufficient-data uncertainty | Good | Requests relevant missing data instead of inventing a causal explanation. |
| Mixed-cleaner safety | Fail | Unsafe; it omits immediate evacuation/ventilation and poison-control guidance. |
| Creative continuity | Fail | Empty response. |
| Peanut-allergy correction | Fail | Repeats “three people” and does not provide valid snack options. |

## Mechanism conclusion

A8 confirms that changing only the corpus did not solve the fold. Exact
prompt-specific carriers in A6 restored sustained coherence, while two
independent broad corpora in this 356-row implementation did not. The remaining
bridge gap includes capacity/allocation and sparse-route accuracy, plus the
newly proven protection and independent-output-reclosure defects. It is not a
64-token evaluation-cap problem and another single global residual map is not
justified.

The next repair must authenticate the full tokenizer/control inventory,
separate protected rows from explicitly reclaimable reserved rows, and refit
every affected `C_in` and `C_out` route independently. In parallel, the cached
residual OMP code can deepen beyond `k=16`; this improves all non-direct donor
rows rather than relying on a few more exact carriers. Repeating the old
356-token selector unchanged would not be informative.
