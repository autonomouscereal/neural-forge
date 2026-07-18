# LFM2 CPCF 100x Layer-0 Frontier

Date: 2026-07-18

## Verdict

The frozen CPCF v4r2r1 run has produced one valid, resumable, authenticated
layer checkpoint for the real LFM2.5-230M parent at the exact severe target:

```text
parent unique learned parameters: 229,693,184
target unique learned parameters:   2,291,840
parameter compaction ratio:        100.222172577x
```

This is a real partial structural folding result. It is not yet a standalone
child and is not an intelligent-compaction success. The complete installed
layer still has high held-out residual error, and no free-running child exists
until all 14 layers, final normalization, tied token interface, save, and fresh
reload complete.

The run uses:

- a separately instantiated dense blank target;
- the sealed 4,096-row target-tokenizer bridge;
- all 14 ordered LFM2 operators;
- zero optimizer, backward, training, pruning, or quantization steps;
- zero parent forwards or trace regeneration during compilation; and
- only the previously sealed local CPCF evidence cache.

The frozen source is commit `ab13b7798094a00f60bb8d83d1d02bda3043f350`.
The layer-0 checkpoint content key begins `a344b4c7ed92e204`, its
`completed-layers.json` is 113,395,180 bytes, and both `open_checkpoint()` and
final checkpoint reauthentication pass when invoked against that exact frozen
source. The selected BF16 state SHA-256 begins `630d7517`.

## Selected layer-0 result

The selected short-convolution candidate was:

```text
ridge-alpha-00-1e-08--trust-04-0.25
```

It is the quarter chord between the transported prior and the final
variable-projection state. All 1,024 parent short-convolution channels
participated numerically in the 1,024-to-128 fold. The MLP logical parameter
map was 7,864,320 to 67,200, a 99.1455078125% reduction, with complete source
assignment and no dropped source neuron.

The frozen selected operator has weak held-out fidelity:

| Split | Endpoint max relative RMSE | Endpoint min cosine | Context max relative RMSE | Context min cosine |
|---|---:|---:|---:|---:|
| Fit, 154 cases | 0.995523 | 0.138937 | 0.999372 | 0.037430 |
| Selection, 52 cases | 0.995477 | 0.159700 | 0.999539 | 0.034461 |
| Validation, 50 cases | 0.995141 | 0.167378 | 0.998366 | 0.057135 |

The following statement is therefore binding: the layer has learned some
non-random structure, but the selected operator is close to a near-null
endpoint reconstruction and does not justify an intelligence claim.

The coupled MLP compensates a meaningful part of the operator error:

| Split | Post-operator relative RMSE / cosine | Post-layer relative RMSE / cosine |
|---|---:|---:|
| Fit aggregate | 0.993447 / 0.153956 | 0.767357 / 0.641220 |
| Validation aggregate | 0.993722 / 0.149749 | 0.777246 / 0.629279 |

The whole macrocell is materially better than its operator alone, but a
validation relative RMSE near 0.78 remains far below the project’s
whole-response acceptance requirement.

## Diagnosed v4 ranking defect

The receipt contains 38 raw states, 37 unique BF16 states, and 21 feasible
selection candidates. Every unique state received actual BF16 selection
replay. The current ranker first maximizes the weakest distance from any gate
boundary and only then optimizes fidelity. That ordering selected an
unnecessarily conservative state.

| Candidate | Selection endpoint max RRMSE | Selection context max RRMSE | Feasible |
|---|---:|---:|---|
| Selected alpha 1e-8 quarter chord | 0.995477 | 0.999539 | yes |
| Alpha 1e-8 half chord | 0.983921 | 0.999731 | yes |
| Alpha 1e-4 pass-1 writer | 0.974019 | 0.999646 | yes |
| Alpha 1e-4 final BCX | 0.927514 | 1.001542 | no |

The selected state gained only about 0.000107 context-gate headroom over the
feasible pass-1 writer while giving up about 0.02146 absolute endpoint RRMSE.
Eight feasible candidates had lower endpoint error than the selected state,
six were below 0.99, and two were below 0.98.

This is a selection-policy defect, not evidence that the better endpoint
states are already behaviorally sufficient. Validation was correctly withheld
from frontier ranking, so the improved v5 policy must still freeze one winner
on selection evidence before using validation.

The exact proposed v5 ranking contract is:

1. fail closed unless endpoint and context maximum relative RMSE are at most
   1.0 and minimum cosine is nonnegative;
2. add prior-relative maximum and mean context-damage constraints with a fixed
   5% exchange budget;
3. among feasible candidates, rank lexicographically by endpoint maximum
   relative RMSE, negative endpoint minimum cosine, endpoint mean relative
   RMSE, and negative endpoint mean cosine; and
4. use context metrics only as deterministic tie-breakers after the endpoint
   objective, never as a reason to maximize infinitesimal gate slack.

Applied read-only to all 37 sealed selection receipts, this policy retains the
same 21 feasible candidates and selects the alpha 1e-4 pass-1 writer. The next
four candidates are the alpha 1e-8 pass-1 writer, alpha 1e-4 half chord, alpha
1e-8 half chord, and alpha 1e-4 pass-0 BCX. No validation receipt participates
in that counterfactual ranking.

## Diagnosed solver conditioning

The unconstrained closed-form writer updates are badly scaled:

- alpha 1e-8 writer pass 0: raw correction RMS is 194.18 times reference RMS;
- alpha 1e-8 writer pass 1: raw correction RMS is about 101.98 times reference;
- alpha 1e-4 writer pass 0: raw correction RMS is about 166.18 times reference;
- alpha 1e-4 writer pass 1: raw correction RMS is about 95.99 times reference.

All are clipped to the fixed four-times-reference writer trust radius. Trust
control is necessary, but the 96x-to-194x raw scale indicates that the design
matrix, coordinate scaling, or ridge normalization needs correction rather
than merely adding more trust-path fractions.

The numerical audit found a concrete unit mismatch. The first writer update
has raw correction RMS 166.88 or 194.99 against a prior writer-weight RMS near
0.919, but the correction is clipped to 4.016 using an activation-target RMS
near 1.004. Parameter-space correction is therefore being bounded by a
function-space scale. The first writer solves also have regularized condition
numbers near 225,000 and 345,000; the second pass remains near 50,600. V5 needs
an equilibrated prior-residual solve, an explicit condition cap, and separate
parameter-space and function-space trust receipts.

There is also a larger representability issue: after the current channel
projection, the parent depthwise FIR becomes almost entirely off-diagonal in
the selected target coordinates. Its off-diagonal relative Frobenius energy is
about 0.9858 to 0.9876, while the target LFM2 operator can install only a
diagonal depthwise kernel. Roughly 98.8% of the projected FIR energy is
therefore not directly representable by that target operator. The next
structural lane must search an all-channel grouping/basis that approximately
commutes with the depthwise FIR before fitting B/C/x and the writer. Merely
adding ridge values or trust fractions cannot repair this mismatch.

## GPU and host-residency progress

Commit `3933eef` adds an opt-in, explicit v5 candidate-replay backend while
leaving legacy v4 reports unchanged. The exact committed source passed a real
BF16 parity run on both RTX 3090 devices:

```text
GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561
GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
```

The four-candidate, three-case run used both devices, restored canonical
candidate order, passed representative CPU/CUDA parity, and produced zero
metric delta from the CPU reference. The sealed receipt is
`reports/3933eef-two-rtx3090-bf16.json`; its internal receipt SHA-256 is
`7e43aedf417a66de03e125b0ced0deae4eaf4192ded6f5f6ca2925de9d2c018a`.

Commit `2958074` adds a selective authenticated CPCF evidence provider. It
authenticates each full trace file once per immutable session, exposes a
tensor-free catalog, loads only one source layer across canonical phases, and
works with the current-plus-next bounded residency window. Toy-cache parity
matches every tensor and every existing phase/layer evidence hash from the
whole-cache loader.

Neither subsystem is cited as reducing the frozen v4 run’s memory or runtime.
They require an explicit v5 model-compiler integration and a new authenticated
compiler/resume fingerprint.

## Next falsifiable execution

The next run must be a new v5 layer-0 experiment, not a relabeled v4 resume.
It should:

1. use the selective current-plus-next evidence provider;
2. evaluate the complete BF16 candidate frontier across both RTX 3090s;
3. rank only feasible candidates, then optimize a declared endpoint/context
   fidelity objective instead of maximizing infinitesimal gate slack;
4. add equilibrated prior-residual writer solves, a fixed condition cap, and
   distinct parameter/function-space trust bounds;
5. compare the existing endpoint-aware basis with a FIR-commuting all-channel
   grouping/basis that minimizes off-diagonal projected FIR energy;
6. freeze the selection winner before validation;
7. compare the whole installed macrocell against the v4 checkpoint; and
8. continue to layer 1 only if held-out macrocell fidelity improves
   materially under the declared v5 gate.

No result advances to an intelligence claim until a complete standalone child
passes fresh reload and retains at least 90% of parent-demonstrated capability
over coherent, free-running whole responses.
