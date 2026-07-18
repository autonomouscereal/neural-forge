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
the scheduled-replay v5 schema and proves that replay scheduling did not
change the closed-form solve. The integration test freezes the RSS probe so
unrelated pytest allocations do not dominate a 43 KB toy tree; independent
tests exercise the real process-RSS sampler and its failure path. A large
production cache run is still required for a meaningful physical reduction
receipt.

## Explicit remaining gaps

1. Disk publication and resume are not integrated. Any resume input is
   rejected before model access so v4 and v5 fingerprints cannot mix.
2. The short-conv v5 numerical primitives from commit `6fd6f61` are
   source-bound but not active. Honest integration requires a pre-install
   selector inside candidate-frontier construction, while every candidate
   state is still available. Post-install rescoring is explicitly refused.
3. The two-GPU authenticated replay backend has its own parity proof, but the
   small local all-layer test uses the CPU-reference evaluator. A production
   run may inject the GPU evaluator through the same runtime argument.

The minimum pre-install hook should receive the complete mapping of sealed
selection reports and the matching authenticated candidate-state hashes,
return one candidate ID plus the sealed v5 rescore receipt, and let the
existing compiler install the already-retained state for that exact ID. The
caller must verify every candidate state before and after the callback.

## Regression evidence

The focused CPCF and runtime suites completed with:

- 233 passed, 2 skipped for all `test_hf_lfm2_cpcf_*` tests;
- 18 passed, 1 skipped for circuit-trace/mapping/selective-residency tests;
- 86 passed for replay-scheduler and resume-artifact tests; and
- 18 passed for the final v5 runtime plus v5 numerical-policy check.
