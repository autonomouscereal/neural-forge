# LFM2 Exact-4x Self-Dual Control and Tied-Congruence Gap

Date: 2026-07-21

This record preserves the result of the first complete production attempt
after the global self-dual tied gauge and the weight-complete MLP structural
signature were combined. It also records the cached component ablations that
identified the next folding change. It is not a coherent-child claim.

## Frozen control

- Runtime commit:
  `c19d8f273b7faca430b83908cd9392a93f298c24`
- Parent: local `LFM2.5-230M`, `229,693,184` unique parameters
- Blank target: `57,405,368` unique dense parameters
- Exact ratio: `4.001249x`
- Child topology: hidden `776`, intermediate `797`, 14 layers, target
  vocabulary `4096`
- Semantic evidence cache:
  `dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b`
- Foundation artifact:
  `2cf4e9d2875922a25b3f9543d943c4b1408d77045535616aa60df2c4defe3153`
- Remote run:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_postfold_20260721/c19d8f2/a0001`
- Runtime: approximately 59 minutes 36 seconds

The run performed no parent or child training, optimizer step, backward call,
pruning, quantization, sparsification, or trace recapture.

## Control result

The transaction failed at the layer-0 short-convolution producer before the
layer-0 MLP was compiled:

```text
Layer 0 short_conv coverage rejected:
blank-or-partial-producer-chain-was-not-densely-planted.
```

The short-convolution candidate selector retained the exact-zero incumbent.
The later structural-coverage contract correctly rejected that no-op because
an all-zero operator is not a valid folded producer. The complete layer-stack
transaction restored its exact entry bytes. No checkpoint was published.

This is a policy contradiction specific to blank targets: an incumbent no-op
is a legitimate rollback candidate for an already functional model, but not
for a target whose incumbent operator is entirely zero. Exact-zero
short-convolution targets must receive the authenticated finite full-source
structural plant; held-out metrics remain diagnostics rather than permission
to retain no model.

## Confirmed runtime defect

The farthest-first source-channel selector in
`hf_lfm2_cpcf_short_conv_fold._cluster_all_source_channels` recomputes every
distance to every previously chosen seed.

For the real `1024 -> 776` geometry it performs:

- `152,454,900` tiny distance reductions and host scalar reads;
- `5,031,011,700` scalar feature-distance terms.

The algebraically equivalent running-nearest-distance recurrence requires:

- `775` distance-vector updates;
- `793,600` source-distance rows;
- `26,188,800` scalar terms.

That is `192.1x` fewer distance evaluations. The old loop alone accounts for
roughly 41-47 minutes per short-convolution layer and would have repeated
eight times. This was folding-orchestration waste, not evidence that the
closed-form operator needs that runtime.

The replacement was then exercised twice on the production Quadro RTX 8000
at the exact `1024 -> 776`, feature-width-33 geometry:

- warm execution: `5.336` seconds
- repeat execution: `3.250` seconds
- source-distance rows: `793,600`
- legacy-equivalent rows: `152,454,900`
- reduction: `192.105x`
- source-channel coverage: `1024 / 1024`
- repeated seed, group, weight, feature, and receipt bytes: exact
- deterministic CUDA state restored after execution: yes
- execution receipt:
  `8468bc3afde35e87dbb587345f33c009ebec6dad9abc0889357510035b4df05c`

The semantic structural-cluster and closed-form writer routes were also
exercised together on that GPU. This validates the production compute seam;
it does not by itself claim child coherence.

## Historical-coordinate MLP correction ablation

Artifact:

```text
/home/cereal/model-surgery-lab/artifacts/
  lfm2_layer0_current_normalized_20260721/c19d8f2/a0003/
  structural-ablation.json
```

This diagnostic used the trusted historical residual chart `E`, not the new
production congruent candidate. It separated the prompt-independent
gamma-aware structural decoder from the FIT-only decoder and W2 corrections.

| Partition | Decoder | W2 | Relative RMSE | Mean cosine |
|---|---|---|---:|---:|
| FIT | corrected | fitted | 0.01244 | 0.99996 |
| FIT_CHECK | corrected | fitted | 1.13648 | 0.44817 |
| SELECTION | corrected | fitted | 1.12273 | 0.44437 |
| VALIDATION | corrected | fitted | 1.14512 | 0.38262 |
| FIT_CHECK | corrected | structural | 1.13350 | 0.44506 |
| SELECTION | corrected | structural | 1.11440 | 0.44155 |
| VALIDATION | corrected | structural | 1.13816 | 0.38357 |
| FIT_CHECK | structural | structural | 1.13773 | 0.44194 |
| SELECTION | structural | structural | 1.11571 | 0.44151 |
| VALIDATION | structural | structural | 1.13968 | 0.38237 |

The W2 correction was underdetermined: 80 FIT rows for input width 797, with a
correction norm `14.20` against structural-prior norm `20.22`. Removing that
correction mildly improved held-out magnitude and maximum error, but did not
repair the bridge. Removing the rank-32 decoder correction also did not repair
the bridge. Therefore a correction-cap sweep is not the missing mechanism.

## Strong tied-congruence diagnosis

The self-dual control proves:

```text
B.T B = I
B T_selfdual.T = E T_base.T
```

It did not prove the condition required for the same coordinates to enter the
model through the tied input embedding:

```text
T_child = S_bridge B
```

Here `S_bridge` is the complete 4096-row mapped parent tied table. The current
factorization preserves the fitted output operator, but its child embedding
rows can begin outside the residual manifold assumed by every folded body
operator.

Production diagnostic artifact:

```text
/home/cereal/model-surgery-lab/artifacts/
  lfm2_tied_congruence_20260721/c19d8f2/a0001/
  tied-congruence.json
```

The measured current self-dual target differs from `S_bridge B` by:

- relative Frobenius error: `0.1533192692`
- mean row cosine: `0.9894266478`
- minimum row cosine: `0.9432734289`

Replacing only the tied table with the exact congruent table
`T_congruent = S_bridge B`, while keeping the already authenticated global
basis `B`, produced these algebraic diagnostics:

| Diagnostic | Current self-dual | Exact congruent |
|---|---:|---:|
| Full-vocabulary normalized-input decode relative error | 0.32356 | 0.29053 |
| Full-vocabulary normalized-input mean cosine | 0.94820 | 0.95824 |
| Bridged-vocabulary self-operator parent top-1 agreement | 0.87646 | 0.99976 |
| Bridged-vocabulary self-operator centered retained energy | 0.96678 | 0.98893 |

The current output factorization remains machine-accurate to its fitted
operator. The congruent operator differs from that old factorization by
relative Frobenius error `0.15515`, while retaining mean source-row cosine
`0.98810`. A cached causal-output evaluation is still required; the
vocabulary self-operator diagnostic is not a substitute for whole-model
behavior.

## Exact-zero blank-target policy gaps

The frozen `c19d8f2` control exposed two independent policy contradictions
after the tied-interface diagnosis:

1. Attention could plant Q/K but still retain exact-zero V/O when the
   established nonregression gates preferred the blank incumbent.
2. The terminal W2 frontier measured every candidate's relative weight delta
   against an exact-zero incumbent. Useful closed-form candidates therefore
   appeared `16,700x` to `662,000x` over the refinement cap, so zero was
   republished and the final structural-coverage check rejected the run.

Neither failure demonstrates that the closed-form plant is poor. Both are
cold-start policy errors: a blank tensor is not a meaningful incumbent
candidate, and a relative-to-zero refinement radius is undefined.

The exact-zero branches now use this common contract:

- Authenticate exact zero from the live incumbent bytes; no caller flag can
  enable the exception.
- Compile and rank the candidate from FIT evidence only.
- Require finite signal plus complete output-row and input/source-axis
  structural coverage.
- Freeze the candidate before opening FIT_CHECK, SELECTION, or VALIDATION.
- Record held-out results honestly as diagnostics; they cannot restore a blank
  incumbent.
- Use no hash signs, epsilon fills, randomness, gradients, optimizers,
  pruning, quantization, or topology changes.
- Install through a transaction that remains guarded until a valid receipt is
  sealed, with exact rollback on any later failure.

For attention this produces a complete Q/K/V/O bundle: FIT-only route ridge
for Q/K, FIT-only joint value/output factorization, and a FIT-only output
writer. Missing axes may be completed only by coefficients identifiable from
the real FIT linear system; otherwise the layer fails before mutation.

For terminal W2 the entire frontier is evaluated on FIT, the minimum eligible
nonincumbent is frozen, and only that winner plus the incumbent are evaluated
on the three held-out partitions. Publication independently recomputes the
seed, evidence rows, split roles, policy, frontier keys, FIT winner, held-out
metrics, coverage, and zero-operation audit. A correctly self-sealed arbitrary
dense tensor is therefore insufficient to enter the exception.

## Compact authenticated foundation replay

The original semantic-foundation interface solve reopens an approximately
`85 GiB` compiler trace cache even though the later postfold needs only the
closed-form tied-interface sufficient statistics and its three immutable
anchor splits.

The compiler now retains:

- the packed FP64 source second moment,
- exact metric witnesses,
- BF16 anchor rows,
- the original selection/FIT/validation split objects,
- the exact source-table, receipt, runtime, policy, and implementation
  bindings.

These are published once as a content-addressed safetensors sidecar. Backfill
opens the old compiler cache exactly once and must reproduce the saved
interface, report, and self-dual gauge byte-for-byte before publishing.
Postfold then replays from the compact sidecar with zero compiler-cache,
trace-safetensor, or selective-provider opens. A requested invalid sidecar
fails closed and never falls back to the full cache.

The sidecar deliberately binds the exact Python/PyTorch solver runtime,
including Torch thread counts. Backfill and postfold must therefore run from
the same immutable commit with the same
`OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, and
`OPENBLAS_NUM_THREADS=16` environment.

## Next bounded experiment

The next exact-4x candidate keeps the same global `B`, target topology,
parameter count, parent cache, and tokenizer bridge. It changes only the
interface/body contract and the blank cold-start policy. The implementation
is complete locally; the remaining step is the immutable remote execution and
whole-response evaluation:

1. Install the native tied table as `S_bridge B`, for exact input/residual/head
   coordinate congruence.
2. Use the geometry-derived gamma-aware decoder rather than depending on a
   rank-32 FIT rowspace to define missing coordinates.
3. Plant the complete source-derived dense short-convolution operator when
   its incumbent is exact zero.
4. Use the deterministic running-nearest CUDA clustering recurrence and bind
   its execution receipt.
5. Retain full-source W1/W3/W2 structural planting and the one global residual
   gauge.
6. Plant all exact-zero Q/K/V/O and terminal W2 roles under the authenticated
   FIT-only cold-start contract.
7. Rehydrate the tied interface from the compact authenticated sidecar.
8. Run all 14 layers, publish the standalone 57,405,368-parameter checkpoint,
   and evaluate long free responses against the 229,693,184-parameter parent.

This is not another PCA, balanced-basis, tokenizer, quantization, pruning, or
training experiment. It corrects an algebraic incompatibility between the
existing tied input table and the residual coordinates used by the body.
