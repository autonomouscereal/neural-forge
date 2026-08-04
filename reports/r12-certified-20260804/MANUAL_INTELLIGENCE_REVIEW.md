# r12 manual whole-response intelligence review

Reviewed: 2026-08-04, by Claude (coordinator), from
`routed-manual-dossier.json` (sha-verified fetch of the sealed dossier in run
`a0023-exact4x-h816-mgv3-r12`). Per
`docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`: this reading is the
authoritative verdict; numeric diagnostics have no acceptance authority; cap
censoring is recorded separately from failure; safety is a separate dimension.

## Run facts

- First complete certified exact-4x publication in project history: receipt
  `complete`, dense body 53,552,368 parameters plus both counted routes,
  zero optimizer/backward operations, fresh reload accepted,
  `fresh_reload_parameters_frozen: true`, certificate schema v4 written
  2026-08-04T22:57Z.
- 16 reviewed responses (15 intelligence + 1 safety in this dossier's split);
  14 hit the output cap (censored), 2 reached observed endpoints, 0 automatic
  folding failures.

## Verdict

**Useful: 0. Partial: 0. Miss: 15 of 15 intelligence responses.** Safety
response equally incoherent (recorded separately; not counted as a folding
failure per policy).

Every response is degenerate token repetition with no sentence-level
coherence and no task engagement in any category (conversation, instruction
following, knowledge, arithmetic, logic, retrieval, summarization,
comparison, coding, debugging, writing, planning, epistemic reasoning,
creative, multi-turn). The dominant mode is single-token or short-cycle
repetition drifting through unrelated vocabulary strata (English function
and layout words, Cyrillic/Polish/Japanese/Chinese fragments, code-tokenizer
debris such as `.isEmpty`, `{t{t`, `}_{-`). The shortest response
(`writing-concise-email`, 25 chars) is whitespace and `ititit`.

## Interpretation

The publication pipeline is fully proven; the behavioral failure is the
known composition physics: locally-accepted closed-form solves whose errors
compound through pretrained nonlinear dynamics (the same wall measured
independently at layer 3 of the pretrained cross-family fold,
`reports/cross-family-real-20260804-pretrained/`). The child's collapse mode
(vocabulary-stratum drift) matches the historical vocabulary-margin
diagnosis (child fixates on high-margin strata; body fold crosses vocab
decision boundaries).

## Designed next actions (all machinery already built)

1. Cache-only first-divergence diagnosis of this child (Codex's
   `60eb928b`/`9bc0d735` authenticated exact4x diagnostics).
2. Sealed on-policy evidence round on this child's actual rollouts, then the
   bridged refold on base+extension evidence
   (`nf-launch-r12-round-refold.sh`, staged and smoke-tested) — the
   exposure-bias remedy this exact failure motivates, never yet exercised at
   real scale.
3. Reverse-reclosure sweep results (in flight) inform the same compounding
   physics for the next full run (r13).

This is a negative intelligence result and a positive platform result, and
per the standing policy neither observation deletes the candidate: the child,
its certificate, its dossier, and its caches remain sealed for iteration.
