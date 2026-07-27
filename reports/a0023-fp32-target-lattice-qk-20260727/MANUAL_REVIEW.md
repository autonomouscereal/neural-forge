# A0023 FP32 target-lattice Q/K treatment

Date: 2026-07-27

## Verdict

A0023 is mechanically valid but behaviorally unsuccessful. Manual review found
**0 coherent or useful responses out of 96 decoded responses**. This run does
not demonstrate retained intelligence.

The run did prove that the target-lattice treatment can alter only the expected
Q/K tensors, preserve FP32, save a standalone child, and reproduce the treated
state after a fresh reload without training or parent inference.

## Manual whole-response review

Six independently materialized heads were each evaluated on all 16 prompts.

| Candidate | Coherent/useful | Reviewed |
|---|---:|---:|
| `exact4x-body-control-a000-r1e2-norm1000` | 0 | 16 |
| `exact4x-causal-a500-r1e2-norm1000` | 0 | 16 |
| `exact4x-causal-a1000-r1e2-norm250-sealed-background` | 0 | 16 |
| `exact4x-causal-a1000-r1e2-norm1000` | 0 | 16 |
| `exact4x-causal-a1000-r1e1-norm1000` | 0 | 16 |
| `exact4x-causal-a1000-r1e3-norm1000` | 0 | 16 |

All 96 responses reached the 256-token generation limit. None ended with EOS
and none was empty. The body control remained word salad dominated by fragments
such as `pay`, `nose`, and `check`. The active causal heads instead collapsed
into repeated subword/pseudocode fragments such as `ip`, `ult`, `plies`, and
`ract`; none answered its prompt. The sealed-background control was exactly
identical to the body control for all 16 prompts.

These numerical diagnostics support the manual reading but do not replace it.
The decisive evidence is the decoded text retained in the dossier.

## Integrity evidence

- Status: `qk-treated-body-saved-reloaded-and-six-head-frontier-decoded`
- Source and treated checkpoint dtype: `float32`
- Standalone fresh reload: verified
- Changed tensors: the 12 expected `q_proj.weight` and `k_proj.weight` tensors
  in attention layers 2, 4, 6, 8, 10, and 12
- All six heads and all raw decoded responses: retained
- Parent model loads, parent forward/generate calls, trace recaptures, training
  steps, backward calls, optimizer steps, pruning, quantization, and
  sparsification: zero

This establishes implementation and persistence integrity. It also disproves
behavioral rescue for this particular Q/K treatment plus six-head combination.

## Artifact identities

Canonical document/result hashes recorded inside the receipts:

- Outer run result: `9b1e53c3a68d071d792d977629bc7b17989b1fb7f54649e487d80a400ad1eb31`
- Six-head frontier result: `68f1c61f65014aa5654c4248e031951c7486cd5c544977f447740a29b5f2e471`
- Whole-response dossier: `68f872dca935beaf57ee94e111ea60cd709d05937fb8b91d24adc525d81f4e4d`
- Target-lattice Q/K treatment: `107c9e360d2c6367744ce76c9a44a392f98763d49c82969aba58abe944991fd0`
- Treated-body provenance: `6d8b29a799051c22e1666ea726541cf59396ae848eb22afc14dcac06654faa63`
- Treated checkpoint inventory: `8c8c765d2e92e0dccbcd6455968a2198e4fec39ae2cbd4c6f33384997f9b52bd`

Downloaded file SHA-256 values, independently verified after transfer:

- `outer/run-receipt.json`: `853d62ee6094f72a694db2ef389c0cfdfd7c4826f144244680eed462ff4dfabb`
- `six-head-frontier/run-receipt.json`: `cb81ddd217e25ff9c07ec6e662fe7ca4ebc107d25f3518bf319595ecfbc8a26e`
- `six-head-frontier/whole-response-dossier.json`: `57105cfc5cbe9865d40aa2dee837a53f70a0451515c9398285fe557761c4ef40`

Remote source artifacts were read from:

`/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/neural-forge-b6332131/a0014-fp32-target-lattice-qk-a0023`
