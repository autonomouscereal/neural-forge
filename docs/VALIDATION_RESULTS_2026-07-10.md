# Neural Forge Pivot Validation - 2026-07-10

## Promotion Contract

- The child starts from seeded random weights.
- Folding may use deterministic merging, projection, SVD/eigendecomposition,
  clustering, and closed-form solves.
- Recovery training, backward calls, and optimizer steps are zero.
- A fold is not successful unless free-running text is coherent and retains
  parent behavior. Hidden-state cosine, PPL, or tensor shape alone cannot
  promote a checkpoint.
- Parent outputs are cached once and reused for every comparison.

The cached Qwen3-4B parent has `4,022,468,096` parameters and scored PPL
`877.579654`, behavior `400/1000`, exact generation `388/1000`, and forced
choice `250/250`.

## Completed 29.9x Control

Run:

`qwen3_4b_to_blank_qwen134m_jointvocab_sourcel2_delta0p1_gpu4_r3_20260710`

- Child: blank Qwen3, `134,445,568` parameters.
- Compression: `29.918934x`.
- Residual map: joint activation plus tied-vocabulary sparse fold.
- Vocabulary row repair: donor L2 norm.
- Recovery/optimizer/backward: zero.
- Full evaluation completed: behavior `0/1000`, exact generation `0/1000`,
  forced choice `96/250`, PPL `153,086,609.22`.
- Parent-pass behavior and generation retention were both zero.

The local held-out gate accepted the fold because final-state relative RMSE
improved by `3.13%` and logit relative RMSE improved by `36.90%`. Candidate
final-state cosine was `0.4511`, logit cosine was `0.7900`, but teacher top-1
agreement remained exactly zero. This is another direct demonstration that
local cosine/RMSE improvement is not a behavioral promotion signal.

Manual inspection across math, code, tokenizer, commonsense, and model-surgery
prompts found repeated `Slee`, `hog`, and mixed-language fragments. It did not
produce a correct number, usable Python function, or coherent explanation.

Evidence bundle SHA-256:

`25eb51d849201d6266c3cdf05209c4d0f7f6a997b9f5beae3b860b808579e2f4`

## Native Raw-RGB Parent Baseline

Parent: `Paranioar/NEO1_5-2B-SFT`, `2,896,618,496` parameters.

The model consumes raw RGB patches through its integral vision path. The
evaluation uses 256 deterministic images covering color, count, spatial, and
OCR tasks. It does not use a frozen external vision tower or a post-hoc
adapter.

| Slice | Result |
|---|---:|
| Overall | `239/256` |
| Calibration | `180/192` |
| Held out | `59/64` |
| Color | `64/64` |
| Count | `63/64` |
| Spatial | `48/64` |
| OCR | `64/64` |

Manual output is direct and coherent. The sole count miss answered `6` for an
expected `5`; spatial failures consistently confused `below` with `right`.
The held-out baseline therefore records what the modality parent actually
knows before any fold.

- Output-cache SHA-256:
  `d02b28bbc91b5206dc7f86b48ba7743aa66a334a037ea390fc702259fd2d9e7f`
- Inference fingerprint:
  `d8dd8e98f5d271f7970a7841b1a22d3b2acd589a765420be3474a671f0ca47b8`

## Capacity Reallocation

A strict Qwen ByteLevel-BPE reducer now creates ordinary Hugging Face
tokenizers whose target rows map one-to-one to exact donor rows. The first
production tokenizer has `50,982` rows, retains `50,712` BPE merges, and was
selected using all `2,906` calibration records.

- Transport fingerprint:
  `d305ef401e219ce0c420cc6625d6290424a3def1d98559404e240a6a9d2c5973`
- Selected-row fingerprint:
  `dbf0e9acac9aefca4ad7e2ed4935114c574fe5c70562af5769ccaccf947fea04`

This allows an exact `134,445,568` parameter Qwen3 child with hidden size
`768`, 19 layers, intermediate size `1,408`, 6 query heads, 3 KV heads, and
128-dimensional heads. Compared with the old 151,936-row child, roughly 52
million parameters move from tied vocabulary rows into the transformer body.

A second strict tokenizer was built specifically for full-width topology
sharing. It has `8,192` rows, retains `7,922` complete BPE merges, and maps
every child row to one exact Qwen parent row. It was also selected from all
`2,906` calibration records.

- Alignment fingerprint:
  `c8c89d4e53a6ca76897470495f16255a73d52b2aa7e80692c5648cfe51df0e1a`
- Transport fingerprint:
  `7a27c04928e4616bab2f774ce37567906552d6e30a2937aa253036003831e8da`

## Token-Row Activation-Subset Fold

Run:

`qwen3_4b_to_blank_qwen134m_targetrows_subset_dual_gpu0_r1_20260710`

- Child: blank Qwen3, exactly `134,445,568` parameters.
- Compression: `29.918934x`.
- Calibration: 1,882 exact target-token trajectories from 96 prompts; 24
  held-out prompts.
- Width merge: 1,408 real coupled SwiGLU neurons selected from the parent,
  with a joint full-residual consumer fit.
- Attention merge: complete real GQA head groups.
- Residual coordinates: dual-aware orthonormal basis.
- Recovery/optimizer/backward: zero.

This candidate passed the stricter hidden-state plus logit trace gate and was
therefore subjected to the complete parent-relative evaluation. It failed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `87/250` | `34.8%` |
| PPL | `877.579654` | `189,291,282.955` | n/a |

Manual inspection covered 64 behavior and 64 generation examples. The child
returned an empty continuation for 126/128 samples. One nonempty answer was
`certain?`; the other repeated `reason`, `tokenizer`, and `mismatch` without a
valid answer. It retained no coherent math, code, tokenizer, commonsense, or
model-surgery behavior.

Comparison SHA-256:

`988b694c6cceca0d878637df1c9feda5a6839529c470d9ea04af8b230a2b7121`

A matched joint-sparse-coordinate run was rejected at the held-out trace gate.
The neuron/head subset is therefore not sufficient: removing roughly 96% of
the active parent circuit destroys autoregressive stability even when
single-pass target-token traces fit well.

The matched PCA-coordinate run also completed the full evaluation. It reduced
PPL to `164,240.637`, much lower than the dual run, but retained behavior
`0/400`, exact generation `0/388`, and forced choice `71/250`. All 128 manual
continuations were nonempty, yet they consisted almost entirely of repeated
`1`, `7`, spaces, and punctuation. The PCA comparison SHA-256 is
`4ac53ba75cb343323d2a054298654b3f57a8faa4c38ac922a39ff74894ed5e7e`.
This is direct evidence that optimizing PPL would have selected a still
behaviorally collapsed candidate.

## Same-Width Real-Medoid Control

Run:

`qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_tokenrows_outputgate_gpu4_r1_20260710`

This control retained the parent hidden width and tokenizer rows while reducing
the model to `1,130,007,552` parameters (`3.559682x`). It used coupled real
SwiGLU neurons and complete GQA groups. Its held-out gate was deliberately
changed to treat logit improvement as authoritative after the earlier hidden
cosine gate rejected a candidate whose logits improved.

The complete parent-relative evaluation still failed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `88/250` | `35.2%` |
| PPL | `877.579654` | `67,570,658.930` | n/a |

All 128 manually inspected continuations were nonempty and byte-distinct, but
none was coherent. Typical continuations repeatedly mixed `chairs`, `Here`,
`which`, `Answer`, parentheses, and prompt fragments. It retained no usable
math, code, or explanation behavior. This falsifies activation-selected width
folding as a standalone solution even at only 3.56x compression.

Comparison SHA-256:

`63d30f51dde11380de51870234556e9a68e6e4b8180d44d7ec73e1a2bdebbb6f`

## Mathematical Pivot

The completed controls show four noncommuting errors that must not be hidden by
local regression metrics:

1. Independent Q/K rotations change attention dot products.
2. Averaging Q/K paths before softmax introduces cross terms absent from the
   parent.
3. Averaging gate/up paths before SwiGLU does not equal merging their
   post-nonlinearity contributions.
4. Averaging two residual blocks does not equal their ordered composition.

The activation-subset control falsified the sparse-circuit hypothesis as a
standalone 29.9x solution. The two completed full-topology controls below also
falsify extreme depth sharing and low-rank depth bases as standalone solutions.

## Standalone Full-Topology Basis Result

Run:

`qwen3_4b_to_basis134443364_fulltopology_gpu0_r1_20260710`

The checkpoint has `134,443,364` unique parameters (`29.919425x`). It preserves
the parent's logical residual width, MLP width, attention topology, and all 36
executions while storing shared depth bases and a factorized tied vocabulary.
Export completed atomically, the parent was unloaded, and parent-free reload
passed before evaluation.

The complete evaluation failed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `9/250` | `3.6%` |
| PPL | `877.579654` | `141,540.313` | n/a |

Of 128 manually sampled continuations, 125 were nonempty and 67 were unique.
They repeated fragments such as `111`, `mek`, and `mate` and did not contain a
correct answer, usable code, or coherent explanation.

Comparison SHA-256:

`8f52c1c70cf0e67661e631f0b618bd2eba8c1b728b00ea4cbdd9470b42dabd28`

## Full-Depth Tied-Projection Result

Run:

`qwen3_4b_to_tieddepth122m_a18_m18_gpu4_r1_20260710`

This child has `122,093,056` unique parameters (`32.945920x`), exact selected
token rows, all 36 logical layers, one real parent attention projection set,
one real parent MLP projection set, and all layer-specific norms. It used zero
optimizer, backward, or recovery steps.

The complete evaluation also failed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `15/250` | `6.0%` |
| PPL | `877.579654` | `8,073,711.456` | n/a |

All 128 manual continuations were nonempty and 119 were unique, but none
passed. They repeatedly mixed fragments such as `raw`, `limited`, `questions`,
`compar`, and `vice` rather than producing coherent math, code, or prose.

Comparison SHA-256:

`86d969491b11b5fddbe808e1d5b79308247bc43bd7988475326bf985db3e4c14`

## Full-Vocabulary Rank-One Result

Run:

`qwen3_4b_depth_basis_allrank1_fullvocab_gpu6_r1_20260710`

This control retained the full parent vocabulary, residual width, MLP width,
attention topology, and all 36 logical executions while representing every
projection role with one depth basis. It had `490,077,948` unique parameters
(`8.207813x`) and used zero recovery.

The complete evaluation failed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `28/250` | `11.2%` |
| PPL | `877.579654` | `244,468,857.739` | n/a |

All 128 manual continuations were nonempty and 120 were distinct, but none
passed. Outputs mixed unrelated multilingual fragments and repeated tokens such
as `派` and `表格`; they contained no usable math, code, or explanation.

Comparison SHA-256:

`1982c67a64c717f0e44f7ed2ffcd6d98fc12db3fb76c396cabac84b5c6c05be9`

This closes the hypothesis that full logical topology plus one basis per
projection role is sufficient. The next codebook path uses a continuous 1-36
real-prototype schedule with activation/Jacobian-guided arbitrary assignments.

## Real-Activation Codebook Identity Control

Run:

`qwen3_4b_codebook_pn36_pa36_gpu0_r3_20260710`

This control retained all 36 real depth variants for every activation-aligned
canonical MLP neuron and GQA group. It proved exact reconstructed projection
tensors, a dense-free in-memory registry, zero optimizer/backward calls, and
bounded FP16 logit drift (`1.94e-5` mean absolute error, `4.31e-5` relative L2,
and `24/24` top-1 agreement on the no-op probe).

The run stored `4,021,784,576` parameters instead of `4,022,468,096`. That
`683,520`-parameter difference is only the removal of 267 padded model rows
that are not tokenizer IDs, so this is not a meaningful compression result.
It also means the control is a trimmed-vocabulary identity rather than a pure
full-softmax identity.

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `400/1000` | `100.0%` |
| Exact generation | `388/1000` | `384/1000` | `382/388` (`98.45%`) |
| Forced choice | `250/250` | `250/250` | `100.0%` |
| PPL | `877.579654` | `882.311915` | n/a |

Manual inspection found six parent-pass losses and two gains. Three pruning
answers became one sentence instead of the requested two. Three Python prompts
collapsed to repeated zeroes. One gained Python prompt produced a correct
addition function. This demonstrates that tiny logit/softmax changes can alter
autoregressive paths even when short-probe top-1 agrees. Future same-tokenizer
controls must retain padded logits and compare the full softmax.

Comparison SHA-256:

`c66e66433d620d0e2d2cf50610cf37061fc53d249331d4bacca534a48b5e2a22`

## Uniform 32-of-36 Codebook Failure

Run:

`qwen3_4b_codebook_pn32_pa32_gpu4_r1_20260710`

This was neuron/head merging, not pruning or layer deletion. The compiler kept
all 36 logical residual blocks, inferred neuron and GQA permutations from real
post-SwiGLU and attention-output activations, and then retained 32 real depth
medoids independently for every canonical MLP neuron and GQA group. Chosen
medoids could be nonadjacent. Four of 36 depth variants were merged for every
canonical feature.

The deployable runtime used `3,618,082,816` learned parameters plus `3,256,020`
index bytes (`1.111768x` versus the source). The complete zero-recovery result
collapsed:

| Signal | Parent | Child | Parent-pass retention |
|---|---:|---:|---:|
| Behavior | `400/1000` | `0/1000` | `0.0%` |
| Exact generation | `388/1000` | `0/1000` | `0.0%` |
| Forced choice | `250/250` | `18/250` | `7.2%` |
| PPL | `877.579654` | `90,846.287214` | n/a |

Manual inspection confirmed real collapse. Math, code, tokenizer, commonsense,
and model-surgery prompts produced repeated multilingual `答案` fragments,
punctuation, blank lines, and unrelated tokens. No parent-pass behavior or
generation survived. Uniformly removing the same depth capacity from every
feature compounds destructively even though each replacement is a real,
activation-aligned parent medoid.

Comparison SHA-256:

`80f23a5a50dbd01aae4f9e9e64f35dbbd845000e34a508d04dd10b150f82565f`

## Uniform Codebook Survival Boundary

The matched `35/34/32/28` sweep completed with zero recovery, the same cached
parent, `1000/1000/250` scoring, and captured outputs. Ratios in this table add
back the 267 padded vocabulary rows removed by the old runner, so they measure
the codebook transform rather than the unrelated softmax trim.

| Prototypes per feature | Adjusted ratio | Behavior | Forced choice | Generation | PPL |
|---:|---:|---:|---:|---:|---:|
| `36/36` | `1.000000x` | `400/1000` | `250/250` | `384/1000` | `882.312` |
| `35/35` | `1.025736x` | `255/1000` | `204/250` | `185/1000` | `1,851.378` |
| `34/34` | `1.052832x` | `0/1000` | `80/250` | `0/1000` | `13,890.088` |
| `32/32` | `1.111558x` | `0/1000` | `18/250` | `0/1000` | `90,846.287` |
| `28/28` | `1.251131x` | `0/1000` | `20/250` | `0/1000` | `1,480,582.096` |

At `35/35`, parent-pass retention was `255/400` (`63.75%`) for behavior,
`204/250` (`81.6%`) for forced choice, and `184/388` (`47.42%`) for
generation. Behavior retained all `200/200` executable-code checks but only
`55/200` arithmetic checks. Generation retained `142/232` parent-passing code
items and `42/142` math items, but `0/14` parent-passing model-surgery
explanations.

Manual inspection agrees with the scores. The child still answered some math
questions correctly and emitted an executable `add(a, b)` function. However,
it often repeated that function many times, turned correct arithmetic into
nearby wrong numbers, and replaced prose explanations with repeated zeroes or
empty output. This is partial circuit survival, not coherent general retention
and therefore not a promoted fold.

Removing two uniform depth variants per feature (`34/34`) crosses a sharp
failure boundary: behavior and generation both reach zero. Deeper uniform
merges remain collapsed. The result rejects a global per-feature depth budget;
the compiler must pin high-damage paths and spend prototypes unevenly according
to measured held-out damage.

Comparison SHA-256 values:

- `35/35`: `955b59cea06aa09ed1272e952548c2f59faa3fac32c3efa1129e97dbdb0b7c24`
- `34/34`: `29ace7ca773c3da77d46aab766ad5adea71c4ab27ba3129cb0712f43cbf2ed57`
- `28/28`: `6a8155624909e6d6474dcc378b66bd55520fed2522903eeabda6e02da5f343c3`
- downloaded evidence archive:
  `f6650f57a09be9ba6b9510304593affd169bff59e2b0a0247982026f20667443`

The contract-checked aggregate and manual-output summaries are stored in
`docs/qwen3_codebook_boundary_evidence_20260710.json` and the matching text
render. Their SHA-256 values are
`e82834b541e64c037337af7741c2fefe52fba7bac611de2b77c1c9b7d3685e3d`
and `0d69fd6111781acf137774badeac1684c9dfb50f810445107e689b8497f4c96f`.

The earlier depth-deletion controls did not produce capability measurements.
The 34-layer run failed checkpoint reload equality, the 32-layer run lacked
the required calibration token rows, and the 28-layer run failed exact
transaction rollback. Those are infrastructure failures, not fold scores.

## Goal And Method Decision

The goal is unchanged: transform one or more parents from arbitrary supported
families, tokenizers, and native modalities into a user-selected child shape,
including a blank child, while retaining as much capability as possible and
removing most scratch-training cost.

Research papers are treated as implementation clues, not proof and not limits
on that goal. Neural Forge's cached-parent comparisons, free-running output,
manual inspection, and controlled before/after learning curves decide which
ideas survive.

The evidence does require a method pivot. More scale-only variants of the
failed one-shot PCA, medoid, tied-depth, and rank-one folds will not be queued.
Instead, three lanes proceed in parallel:

1. Find the first compression boundary where autoregressive behavior breaks,
   then identify and preserve the specific coupled paths responsible.
2. Build parent-owned executable anchors and trajectories into blank children,
   and measure how much training they remove versus the identical random child.
3. Generalize those anchors through byte-aligned token interfaces and explicit
   operator layouts for cross-family, simultaneous multi-parent, and native
   raw-modality transfer.

Any target shape remains accepted as a requested transformation. Reports must
distinguish a mechanically valid transformation from retained capability and
must count all stored assignments, indices, metadata, and runtime costs.

### Adaptive Surgery Requirement

The compiler must not assume that a target column receives the matching source
column or that only adjacent layers can combine. It must construct candidates
over arbitrary rows, columns, heads, residual directions, token rows, and depth
positions. Valid actions include preserving an operator exactly, sharing it,
merging nonadjacent or interleaved paths, projecting it through a retained
subspace, composing residual steps, or skipping a proven-low-impact step.

Selection is sequential. Each accepted transaction changes the child rollout,
so activation, Jacobian/rollout, and output-sensitivity evidence must be
recomputed before choosing the next transaction. Exact anchors may be pinned;
soft anchors may be projected or filtered; anchors may also be omitted when
their measured retention-per-byte is worse than another path. An external
parent-relative evaluator must be able to reject and roll back every proposed
transaction.

A learned surgery policy is a parallel implementation path. Its training rows
will contain parent and child weight/layout features, tokenizer-byte mappings,
activation and rollout sketches, the proposed transform, and the measured
post-transform behavior. The policy is useful only if it predicts safer target
weights or action sequences on held-out model families; it cannot be trained or
ranked on shape validity or PPL alone.
