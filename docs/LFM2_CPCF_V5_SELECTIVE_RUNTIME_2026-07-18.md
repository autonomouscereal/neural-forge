# LFM2 CPCF v5 Selective Runtime

## Scope

`hf_lfm2_cpcf_v5_model_compiler.py` is an opt-in fresh-fold runtime. The
existing v4 compiler remains the default and its report, byte, and provenance
contracts are unchanged.

The v5 runtime:

- consumes an already sealed local LFM2 circuit-trace cache;
- constructs tensor-free canonical phase shells;
- holds only the current source layer and its prefetched successor;
- fits each boundary chart immediately before compiling that layer;
- rebinds phase-state banks without copying archived child states;
- releases source evidence before compiling the following layer;
- loads final-normalization evidence separately and reauthenticates it;
- optionally injects an authenticated candidate-replay evaluator;
- optionally invokes a tensor-blind candidate selector while every sealed
  short-conv candidate state is still retained;
- samples process RSS through the whole compile; and
- rolls the exact target blank back if compilation or the RSS gate fails.

It does not run the parent, train, prune, quantize, or use a fallback loader.

## Authenticated memory claim

The configured full-tree byte count must equal:

```text
sealed final-normalization bytes
+ sum(sealed aggregate bytes for every source layer)
```

The value is derived from the authenticated selective-evidence catalog. A
caller cannot inflate the baseline to make the RSS comparison easier.

Logical residency and physical RSS are separate receipts. A logical layer
release is not treated as proof that mapped pages left the process. The
runtime continuously samples process RSS and refuses the fold if its measured
increase is not materially below the authenticated full-tree baseline.

Commit `ccea0f7` changed the selective cache session so each safetensors
mapping is closed after one bounded selection while the authenticated raw file
guard remains open.

## Executed local proof

A real fresh test captures five batches from a two-layer LFM2 parent with
hidden width 8, then folds them into an independently created zero blank with:

- hidden width 4;
- 484 unique dense parameters;
- one short-conv layer and one full-attention layer;
- selection, three fit, and validation batches;
- complete four-token prefill history plus one forced-decode step;
- a 43,390-byte authenticated full evidence tree;
- a 40,990-byte current-plus-next layer peak; and
- injected CPU-reference candidate replay.

Both layers compile and replay in order. The nested short-conv artifact uses
the explicit pre-install-selector v5 schema. An authenticated test selector
chooses a different retained state from the default v4 ranking, and the
installed state hash exactly matches that choice. Injected CPU-reference
candidate replay remains active. This proves the selector is part of the real
all-layer compile path instead of a post-install report rewrite.

The integration test freezes the RSS probe so unrelated pytest allocations do
not dominate a 43 KB toy tree; independent tests exercise the real process-RSS
sampler and its failure path. A large production cache run is still required
for a meaningful physical reduction receipt.

## Pre-install selector contract

The hook is additive and defaults to `None`. With that default, the committed
v4 report bytes and compiled state tensors are unchanged.

When enabled, the compiler:

1. seals the complete candidate-ID-to-state-hash inventory and every sealed
   selection report;
2. gives the selector only a detached canonical request, never live tensors,
   fit reports, or validation reports;
3. re-hashes every retained state and the request after the callback;
4. requires a sealed v5 report naming an existing candidate ID and its exact
   live-state hash; and
5. installs the already-retained state for that ID before releasing the
   frontier.

The selector receipt binds the request, selector report, default v4 choice,
chosen ID and state hash, and before/after immutability checks. Selector-backed
success and enforced-gate failure use explicit v5-only schemas and never claim
to be legacy v4 artifacts.

`Lfm2CpcfV5EndpointFirstCandidateSelector` exposes the committed source-bound
endpoint-first policy through this dynamic protocol. A caller may provide
another authenticated selector, but the model report distinguishes that
generic hook from the source-bound numerical-policy adapter.

## Explicit remaining gaps

1. Disk publication and resume are not integrated. Any resume input is
   rejected before model access so v4 and v5 fingerprints cannot mix.
2. The short-conv v5 endpoint-first policy introduced in commit `6fd6f61`
   and numerically hardened in commit `8cf34a3` now selects at the correct
   pre-install frontier, but it only rescales and ranks the candidates the
   v4 solver already generated. Ridge, basis-residual, and interpolation
   candidate generation are not active and are not claimed.
   On the current tiny real-cache fixture, the fixed source-bound gates
   correctly fail closed because no existing candidate is feasible; the gates
   were not weakened to manufacture a success.
3. The two-GPU authenticated replay backend has its own parity proof, but the
   small local all-layer test uses the CPU-reference evaluator. A production
   run may inject the GPU evaluator through the same runtime argument.

## Regression evidence

After the selector hook and numerical hardening were combined, the canonical
root verification completed with `263 passed, 2 skipped` across every
`test_hf_lfm2_cpcf_*` test plus the dynamic replay-scheduler and
evidence-residency suites. Ruff and `git diff --check` also passed.
