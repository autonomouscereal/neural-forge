# A0018 source-local residual fold failure

- Run: `fp32-source-local-residual-rho025-a0018`
- Deployed source: `5e430f30`
- Terminal status: `failed-without-behavior-promotion`
- Failure: `Lfm2SemanticPostfoldError: Source-local short-conv layer 0 composite receipt is incomplete or inconsistent.`
- Decoded output: none; the run failed while authenticating the folded body before checkpointing or decode.
- Canonical failure digest: `337d8e0978a0b21a914c66d36f9d965f35a073ff2a5f60f1da4b6e077b6c6585`
- Exception-chain digest: `56a4ebbba7d94d8a224ce42e3844687eac445383db4a3076c3e0d8ac03503d6b`
- Archived `failure-receipt.json` SHA-256: `aea9e1208645bf01d2cd2ad6e4e28d0cf54c9c13fa9181dae9c7620cc5bda02c`
- Archived `fp32-source-local-residual-rho025-a0018.log` SHA-256: `b9880d8df93742f6fa094d1f964bc2eab8410023db2f6acd386d99631e715985`

## Diagnosis

This was a producer/consumer receipt-contract failure, not a GPU crash, folding
math failure, or decoded-behavior result. The body compiler completed all 14
persisted normalized-input request capsules and constructed a sealed layer-0
source-local short-conv composite. The later postfold consumer rejected that
composite before checkpointing or decode.

The surviving A0018 failure artifact did not retain the individual failed
predicate, so the deployed `5e430f30` producer and validator were reproduced on
the CUDA path without rerunning the full fold. That reproduction isolated one
and only one failed clause:

- the CUDA structural-clustering producer seals `target_model_forward_calls: 0`;
- the aggregate postfold validator incorrectly required
  `child_model_forward_calls: 0` for that cluster receipt; and
- the structural writer and source-local compiler correctly use the separate
  `child_model_forward_calls` name.

Changing only the cluster predicate to require its producer-issued
`target_model_forward_calls` field made the same production-shaped composite
authenticate. Current `c3b50a14` did not contain this fix; its candidate-archive
change was unrelated.

## Implemented correction and failure-proofing

The nested execution authenticator now requires an explicit counter-field name:
the structural writer and source-local compiler authenticate
`child_model_forward_calls`, while the CUDA structural cluster authenticates
`target_model_forward_calls`. There is no permissive fallback. Synthetic CUDA
regressions prove that a correct zero target counter passes and that a missing
or nonzero target counter fails closed.

The exact-4x runner also captures the canonically sealed layer-stack receipt
immediately after the compiler boundary and persists
`semantic-layer-stack-pre-auth.json` on either success or downstream failure.
Persistence failures add diagnostic context without masking the original fold
exception. This prevents another expensive result from losing its exact
pre-authentication evidence.

Combined validation after both changes: 169 focused tests passed, including the
source-local short-conv composite, semantic postfold, and exact-4x runner suites;
Ruff and `git diff --check` also passed.
