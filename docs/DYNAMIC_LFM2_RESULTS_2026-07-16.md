# Dynamic LFM2 Folding Results

Date: 2026-07-16

## Corrected packed-value controls

The packed-value loaders originally rebuilt the LFM2 rotary runtime buffers in
`float32`, while the direct local parent used `bfloat16`. That changed real
logits before generation and contaminated every earlier packed-checkpoint
behavior score. The loaders now reconstruct the buffers in the source runtime
dtype. Two independent fresh reloads agree bit-for-bit on the authenticated
probe, and the packed all-eight-bit runtime agrees bit-for-bit with the intended
dense quantized model on real prompts.

The corrected authoritative results are:

| Control | Checkpoint bytes | Physical reduction | Parent-correct tasks | Demonstrated capabilities | Verdict |
|---|---:|---:|---:|---:|---|
| mixed `8/7`-bit exact-50 baseline | `232,063,998` | `50.0000034473%` | `7/11 = 0.63636` | `5/8 = 0.625` | reject |
| role-protected mixed `8/7`-bit exact-50 | `232,064,001` | `50.00000280095%` | `9/11 = 0.81818` | `6/8 = 0.75` | reject |
| uniform-eight-bit lossless container | `214,133,416` | `53.863287%` | `8/11 = 0.72727` | `5/8 = 0.625` | reject |
| old mixed `4/3`-bit exact-75 | `116,031,978` | `75.0000062483%` | not re-promoted | not re-promoted | collapsed: `56/64` responses empty |

The corrected exact-50 baseline also scored arithmetic `2/4`, instruction
compliance `3/5`, code syntax `4/8`, EOS/generation-limit retention `61/64`,
and repetition retention `62/64`. The corrected protected candidate improved
the main task and capability results, with arithmetic `3/4`, instruction
compliance `4/5`, code syntax `4/8`, EOS/generation-limit retention `61/64`,
and repetition retention `63/64`. This is clean evidence that asymmetric
allocation helps, but neither candidate approaches the required approximately
`90%` parent-relative intelligence and coherence.

All three are value-compaction controls. They preserve every logical layer,
head, neuron, and tied-vocabulary role and therefore do not prove structural
folding into a smaller target architecture. Logical/topological size and
physical checkpoint bytes remain separate promotion dimensions.

### Superseded pre-correction semantic packet

The manual packet and statistics below are retained as historical audit
provenance only. They were produced with the wrong rotary-buffer dtype and are
not current folding evidence. In particular, the old `9/11` mixed-control
score must not be compared with corrected candidates.

Before that loader correction, the 50% control appeared to retain `9/11 =
0.81818` parent-correct tasks and `6/8 = 0.75` parent-demonstrated capability
contracts. It retained every parent-nonempty response (`47/47`) and every
parent-nonfragment response (`46/46`), but failed arithmetic (`3/4`),
instruction compliance (`4/5`), code syntax (`3/8`), three strict
EOS/generation-limit cases, and one repetition case.

The six open-ended cases have now received packet-bound semantic review.
Scores use the sealed `0..4` rubric in the order correctness, coherence,
instruction following, and capability retention.  A case passes only if
every dimension is at least `3` and its mean is at least `3.6`; exact wording,
token identity, and sentence-structure matching were not used.

| Case | Scores `C/Co/I/R` | Mean | Pass | Reason |
|---|---:|---:|:---:|---|
| `holdout_v1_manual_00` | `0/0/0/0` | `0.00` | no | Parent and candidate are blank; neither supplies the requested empathetic help. |
| `holdout_v1_manual_01` | `3/4/3/4` | `3.50` | no | Coherent lighthouse story with the unexpected-purpose idea and at least parent-level quality, but only three sentences rather than the requested four. |
| `holdout_v1_manual_02` | `1/2/2/4` | `2.25` | no | Candidate exactly preserves the parent response, but that response incorrectly says sharing the basis reduces the number of layers and never explains private per-layer coefficients/operators. |
| `holdout_v1_manual_03` | `4/4/4/4` | `4.00` | yes | Correctly explains that one token cannot test contextual, multi-token coherence; fully useful and instruction-compliant. |
| `holdout_v1_manual_04` | `1/2/1/2` | `1.50` | no | Supplies no analogy and gives a circular, technically weak account of compounding errors; it preserves only part of an already weak parent response. |
| `holdout_v1_manual_05` | `0/1/0/1` | `0.50` | no | Drifts into an irrelevant multiple-choice template, truncates, and provides neither two useful diagnostic questions nor a next step. |

The strict manual pass rate is therefore `1/6 = 0.1666667`, below the required
`0.90`.  The normalized capability-retention dimension is `15/24 = 0.625`;
the normalized totals for correctness, coherence, and instruction following
are respectively `9/24 = 0.375`, `13/24 = 0.54167`, and
`10/24 = 0.41667`.  Across all 96 available rubric points the candidate scores
`47/96 = 0.489583`.  Those normalized rubric summaries are diagnostics, not a
certified intelligence-retention percentage.  The reviewed gate's
conservative lower bound is `1/6`, and the candidate remains rejected because
the manual, task, capability, arithmetic, instruction, code-syntax, EOS, and
repetition components all fail their individual contracts.

The most actionable compression-induced manual failure is
`holdout_v1_manual_05`: loss of prompt-role binding, long-horizon constraint
tracking, and termination is exactly the kind of behavior a structural
residual-attention recovery path must protect.  `manual_01` is a weaker
sentence-plan/counting target, although it is not a clear regression because
the parent also misses the four-sentence constraint.  `manual_04` exposes
analogy-plan drift but is largely inherited from the parent.  `manual_00` is
an inherited blank, `manual_02` is an exact copy of a parent misconception,
and `manual_03` needs no repair.  The folding allocator should therefore
prioritize parent-demonstrated prompt constraints, long-horizon residual
state, EOS margins, and template-drift competitors; it should not spend scarce
structural capacity trying to “recover” abilities absent from the parent.

The review is bound to packet
`156b73f862dbab676ccbf29c3de9cd1106f0983725de0e47b811cef899e9dea3`.
The ratings seal is
`60d28f92ee484a8fd4f180cc2e21edeb50733ca9257571889b614d348e8e880d`
and their exact file SHA-256 is
`995067d020da831c4a8425e1c593ead6cb6303eaaca3552a6549fdfeca440895`.
The cached-only reviewed report is
`/home/cereal/model-surgery-lab/artifacts/lfm2_value_fold_control_20260717/`
`lfm2-230m-sealed64-b0a2c815b744-src-3b9fa3145aef-run-fda4a6553719-v1/`
`target-50pct/intelligence_report_manual_reviewed.json`; its internal seal is
`e4f131b2ab805f7c9e4d27e628107da38cdc6ca8a01b9f1e0cff4bc17a191a71`
and exact file SHA-256 is
`08a263b6f44f779502c967b875a9ed4c9ac678ef25e0f162a67bde5baa62898a`.
The review replay loaded no model, generated no tokens, and performed no
forward, backward, optimizer, gradient, network, or Python-candidate-code
execution.

## Protected-weight-error exact 50% ablation

The first role-protected exact-50% packed ablation completed on 2026-07-17.
It restored normalization, convolution state, and local special/chat token
rows exactly; prevented embeddings, attention Q/K/V/O, convolution state,
and normalization from receiving seven-bit groups; and spent the remaining
budget on FFN producers and residual writers. It used no causal frontier, no
new parent execution, and no training.

The artifact met the physical target:

- checkpoint bytes: `232,064,001`;
- ceiling: `232,064,014`;
- realized reduction: `50.00000280095%`;
- exact-overlay source bytes: `1,140,224`;
- base64 overlay characters in the manifest: `1,521,024`;
- lower-bit groups: `192,965`;
- additional lower-bit groups required to pay for exact overlays: `51,272`;
- parent forwards, backward calls, and optimizer steps: all zero.

Under the corrected `bfloat16` rotary runtime, behavior improved over the
corrected mixed-width baseline but did not reach the project contract. It
retained `9/11 = 0.81818` parent-correct tasks versus `7/11 = 0.63636`, and
`6/8 = 0.75` demonstrated capabilities versus `5/8 = 0.625`. Other corrected
automatic components were arithmetic `3/4`, instruction compliance `4/5`,
code syntax `4/8`, EOS/generation-limit retention `61/64`, repetition
retention `63/64`, and nonempty retention `47/47`.

This result keeps role-aware/asymmetric allocation open and closes only this
specific broad-overlay/base64 implementation. Paying for broad exact anchors
by forcing another `51,272` FFN groups down to seven bits is too blunt, and
base64 consumes bytes without improving the runtime representation. The next
lane uses actual binary costs, causal/activation evidence, exact outlier
peeling, attention-aware mismatch correction, and residual writers.

The rejected artifact root is
`/home/cereal/model-surgery-lab/artifacts/`
`lfm2_protected_weight_error_20260717/`
`lfm2-230m-exact50-independent64-v1`. The corrected reevaluation is under
`/home/cereal/model-surgery-lab/artifacts/`
`lfm2_packed_rotary_reeval_20260717/`
`protected50-corrected-rotary-v1`; its report seal is
`6578fbb6a6dce860cc0217eed65d79e9142e77561810cbbacb964e9127f9d768`.

## Clean uniform-eight-bit control and quantizer geometry

The standalone all-eight-bit control stores every logical weight value with an
eight-bit code and a `float16` group scale. It has no seven-bit values, no
dense hidden checkpoint, and no hub dependency. Lossless compression is used
only around the exact code, scale, tokenizer, and configuration streams.

The corrected `v2-rotary-exact` artifact has:

- source directory bytes: `464,128,028`;
- checkpoint bytes: `214,133,416`;
- realized physical reduction: `53.863287%`;
- persistent packed runtime bytes: `231,490,112`;
- logical eight-bit codes: `229,695,488`;
- group scales: `897,248`;
- rotary runtime bytes: `128`, restored exactly in `bfloat16`;
- seven-bit weight codes: zero.

An independent real-model audit authenticated every code and scale, all 64
tokenizations, and the tokenizer package. The corrected packed runtime and the
intended dense quantized model produced bitwise-identical logits on real
probes (`max_abs_logit_delta = 0`). The remaining behavioral loss is therefore
caused by the quantized representation, not the container or loader.

Behavior remains below target: `8/11 = 0.72727` parent-correct tasks,
`5/8 = 0.625` capability groups, arithmetic `4/4`, code behavior `1/2`, code
syntax `3/8`, instruction compliance `3/5`, and EOS/generation-limit retention
`62/64`. The candidate is rejected. Numerical closeness and a correct container
are necessary diagnostics, not intelligence retention.

The artifact root is
`/home/cereal/model-surgery-lab/artifacts/lfm2_exact50_all8_20260717/`
`lfm2-230m-all8-independent64-v2-rotary-exact`.

A separate local safetensors audit then compared row-aligned group sizes
`32/64/128/256`, bit widths `4..8`, exact top-`k` outlier peeling, and
per-group clipping. It sampled equal coverage over `229,668,608 / 229,693,184`
learned values (`99.9893%`). For valid group size `256`, best-clipped relative
RMSE was:

| Bits | no exact outliers | exact top 2/group | exact top 4/group |
|---:|---:|---:|---:|
| `8` | `0.004229` | `0.003502` | `0.003286` |
| `7` | `0.008452` | `0.007084` | `0.006604` |
| `6` | `0.017156` | `0.014186` | `0.013431` |
| `5` | `0.033883` | `0.029113` | `0.027161` |
| `4` | `0.066239` | `0.056525` | `0.053228` |

This does not promote any child. It establishes useful option geometry for the
causal byte allocator: approximately eight, seven, six, and five average bits
correspond to the `50/55/60/65%` physical ladder, while outlier handling and
weighted rounding materially improve each operating point. The planner must
select those choices by downstream activation/attention damage per actual
encoded byte.

The sealed diagnostic is
`reports/lfm2_quantizer_geometry_20260717/report-v2.json`, SHA-256
`4cd315d1a2a3b15604aaa224118620d7fe6ccd85df9d7c1510201a1d400d175d`.
Its focused remote test suite passed `4/4`; it performs no generation, parent
runtime call, optimization, or training.

## Exact 50% producer-first structural fold

The first exact large-target producer-first structural child has now completed
real scoring. It was installed into the blank target transactionally, used
only cached parent evidence, performed zero parent forwards and zero training,
reloaded as the actual child, and was then rejected on behavior:

- parent: `229,693,184` unique learned parameters;
- child ceiling: `114,846,592`;
- child: `114,846,080`;
- realized parameter reduction: approximately `50.00022%`;
- topology: six folded layers, `14Q/7KV`, FFN width `1458`;
- parent-to-child spans:
  `0..1`, `2..3`, `4..5`, `6..7`, `8..10`, and `11..13`;
- teacher-forced top-1 agreement: `0.009375`;
- free-running token agreement: `0`;
- mean residual cosine: `0.501341`;
- mean residual relative RMSE: `1.027765`;
- trajectory score: `0.174522`;
- retained parent-correct tasks: `0`;
- all `28/28` outputs hit the generation limit;
- `23` new generation-limit saturations and `28` repeated-ngram regressions.

The outputs were nonempty but repetitive fragments such as recurring
`"ound"`, `"ogue"`, and `"Mac"` continuations. This is decisive evidence
against this particular depth-six/uniform-width balance, not evidence that a
50% child is impossible. The run is closed: it must not be promoted or
blindly repeated with the same topology.

The failed-closed artifact root is
`/home/cereal/model-surgery-lab/artifacts/lfm2_hybrid_50pct_20260717/`
`direct-large-v6-span-mapped-evaluator`. The exact receipt-file SHA-256 is
`ebee2fa61a225234c786deeee7e4b135ebab538fa06ab705b4ed5c4860d38556`;
the diagnostics-file SHA-256 is
`e81904e3dde4b565c660d5d87b167d425035ed643b0cb93b085b198f75c53933`.

This result also validates two pieces of infrastructure that earlier attempts
did not reach. The trajectory evaluator now maps each folded child layer to
the endpoint of its authenticated parent span instead of incorrectly
requiring identical parent/child layer kinds. Transaction rollback restores
both parameter state and owned runtime metadata exactly. Neither correction
rescued the candidate; it merely ensured that the rejection measures the real
half-size child rather than an evaluator or rollback artifact.

The next 50% experiment therefore keeps all logical layers and spends the
physical byte budget selectively. It preserves embeddings, token anchors,
normalization, Q/K state, and convolution state, then ranks lower-precision
FFN and attention V/O groups by cached causal damage. This directly tests
whether the coherent packed-value control can move from `75–82%` demonstrated
retention to the required parent-relative `>= 90%` without training.

## Large-ratio producer-first update

The producer-first residual-attention compiler has now moved the real LFM2
folding frontier well beyond the historical sub-percent ladder.  The current
quality reference is a full-depth, zero-training, MLP-only fold:

- parent: `229,693,184` unique learned parameters;
- child: `206,726,912`;
- removed: `22,966,272` (`9.998673709%`);
- attention topology: unchanged `16Q/8KV`;
- recurrent/short-convolution operators: unchanged;
- target FFN width: `2026` in all 14 layers;
- accepted sequential residual refinements: layers
  `0,1,2,3,4,5,7,9,10,12`;
- rejected and exactly rolled back refinements: layers `6,8,11,13`;
- teacher-forced top-1: `0.953125`;
- eight-token free-run match: `0.8125`;
- mean residual cosine: `0.9903684841`;
- mean residual relative RMSE: `0.1326511611`;
- trajectory score: `0.9072707566`.

The standalone final child is stored on the AI server under
`artifacts/lfm2_large_ratio_20260716/wave2-producer-first-r2/`
`lfm2_10pct_mlp_q16_kv8_ffn2026_producer_first/output/final-child`.
Its state identity is
`433f925c94fb8c508d7624db1c970e9e2d84c8af383fe54aa27a594159077271`.

The fresh-reload 28-case whole-output evaluation rejects this reference.  It
retains all 24 nonempty parent responses and records weak task-relevance
retention `22/24 = 0.9166667`, but that number is explicitly **not an
intelligence score**.  Manual output review finds coherent arithmetic,
factual, and short creative islands, but malformed code, wrong implementation
logic, repetitive conversation/explanations, and eight new 64-token
saturations.  This checkpoint is a mechanically real and partially coherent
folding reference, not a `90%`-intelligence result.

A new sealed post-hoc intelligence gate evaluated the same cached 28 parent
and child responses without loading either model or generating any new traces.
It confirms that the old relevance score materially overstated retained
ability:

| Strict component | Result | Required |
|---|---:|---:|
| Absolute reliable correctness retention | parent `10/23`, child `6/23`; `0.60000` retained | `>= 0.90` of parent |
| Parent-demonstrated correct-task retention | `5/10 = 0.50000` | `>= 0.90` |
| Parent semantic-capability retention | `18/23 = 0.78261` | `>= 0.90` |
| Code-syntax rate | `0.62500` | `>= 0.90` |
| Arithmetic exactness | `0.62500` | `>= 0.90` |
| EOS/generation-limit retention | `0.71429` | no regressions |
| Repetition retention | `0.89286` | no regressions |

The all-case topical diagnostic is only `22/28 = 0.78571`, and it is
non-gating.  The seven-case semantic review is now complete and bound to the
exact packet.  No case passed: the candidate was repetitive or missed the
instruction on chat, fact, code, model-compression explanation, summary, and
folding prompts; its robot/key response was partly coherent but still below
the per-case threshold.  The manual pass rate is therefore `0/7`, and the
final intelligence-retention claim is rejected rather than merely withheld.
The reviewed report's internal seal is
`cd21e845138cb16126e4a062dbcf591522b259f8872b3d9ac3f48b19a533c925`;
the exact report-file SHA-256 is
`0eea8db9dc54d9fac236809a2c0b77471a2a5ff9aa9b337dd0d1f211564d2544`.
The gate reports the child's raw absolute correctness
`6/23 = 0.26087` alongside the parent's `10/23 = 0.43478`, but correctly
certifies retention against the parent's demonstrated ability rather than
requiring the child to solve tasks its parent cannot solve.  The original
absolute-90%-accuracy policy report and the later manual-pending report are
retained only as superseded audit artifacts.
The local sealed evidence is under
`runs/lfm2_dynamic_20260716/intelligence-gate-10pct/`.

Two much larger same-mechanism frontier probes were then executed instead of
continuing tiny width steps:

| Target FFN width | Unique child parameters | Removed | Accepted layer refinements | Weak task retention | Reliable correct cases | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| `1492` | `183,760,640` | `19.9973474%` | `2/14` | `0.3750` | `0/18` | incoherent; reject |
| `1225` | `172,277,504` | `24.9966843%` | `1/14` | `0.1667` | `0/18` | collapsed; reject |

The 20% child introduced three blanks/fragments, eight new generation-limit
saturations, and sixteen repetition regressions.  Its code outputs degenerated
into repeated fences/examples and its arithmetic was mostly wrong.  The 25%
child introduced eleven blanks/fragments, ten new saturations, and fifteen
repetition regressions; manual outputs are overwhelmingly blank or repetitive.
Both are real standalone parameter reductions with no sidecars, but neither is
behavioral progress.

This establishes a sharp result: unilateral FFN-width folding is viable near
10% overall removal on this parent, but simply pushing the same operation to
20-25% crosses a nonlinear circuit-capacity cliff.  The next large-ratio lane
therefore composes mechanisms:

1. repair the 10% reference with cached 64-token termination, code,
   arithmetic, instruction, rare-symbol, and long-position anchors;
2. retain that folded circuit and replace only highly redundant cross-layer
   coupled SwiGLU units with a physical partial-neuron bank;
3. apply sequential residual/attention correction and whole-output rollback
   after each material compaction stage;
4. use independent correctness, EOS, repetition, code behavior, instruction,
   and manual-coherence gates rather than exact token matching or topical
   relevance.

Whole-FFN sharing is closed for this target.  Three activation-aligned
shared-base/residual variants removed `11.13-12.50%`, but weak whole-output
retention was only `0.125-0.333`; increasing residual rank did not repair the
collapse.  The reusable lesson is transactionality and evidence sealing, not
the whole-layer sharing hypothesis.

Partial coupled-neuron banking was also tested as a physical large-ratio
runtime.  Each pair stored one shared `w1` row, `w3` row, and `w2` column and
removed exactly `3 * 1024 = 3,072` unique parameters; every unselected unit
remained byte-exact.  Both runs used zero parent forwards and rolled back
exactly:

| Total removal | Shared pairs | Candidate parameters | Parent trajectory score | Teacher top-1 | Weak task retention | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| `15.0006854%` | `3,740` | `195,237,632` | `0.36050` | `0.30625` | `0.29167` | reject |
| `19.9986849%` | `7,477` | `183,757,568` | `0.24532` | `0.134375` | `0.00000` | reject |

The selected 15% pairs had mean activation correlation `0.48528`, but mean
coupled-weight cosine only `0.01434`; the 20% values were `0.44384` and
`0.01007`.  Thus similarly firing neurons at different depths generally store
different operators.  Averaging those operators destroys the trajectory even
when individual runtime sharing and accounting are exact.  Close this
activation-permutation-plus-average formulation; keep the indexed bank runtime
for future dictionary/residual parameterizations that do not assume the two
operators are numerically interchangeable.

Activation-aware low-rank factorization of the residual-writing `w2`
projections was then tested at the requested large targets.  The nonlinear
`w1` and `w3` producers, every attention operator, and every recurrent
operator remained exact.  Ranks were allocated globally from disjoint
activation-fit and held-out-validation paths, and every candidate was
physically installed, evaluated against the original parent and 10% incumbent,
then exactly rolled back:

| Total removal | Candidate parameters | Per-layer `w2` ranks | Parent trajectory score / cosine | Strict reliable correctness | Whole-output failure | Verdict |
|---:|---:|---|---:|---:|---|---|
| `15.0005357%` | `195,237,976` | `680,680,680,680,680,680,233,185,202,172,195,220,216,253` | `0.36085 / 0.81004` | child `0/23`; parent `10/23`; retention `0` | `11` new empties, `21` new saturations, `26` repetition regressions | reject |
| `19.9985943%` | `183,757,776` | `127,144,147,146,173,131,139,140,134,112,144,130,196,129` | `0.36915 / 0.79122` | child `0/23`; parent `10/23`; retention `0` | `10` new empties, `12` new saturations, `23` repetition regressions | reject |

Activation weighting materially beat raw weight-only SVD on the held-out
projection diagnostic: mean validation relative RMSE was `0.25860` versus
`0.52218` at 15%, and `0.28287` versus `0.75398` at 20%.  That local gain did
not compose through 14 residual writes.  Both children retained zero of the
parent's ten reliably correct cases, failed arithmetic, code syntax,
parent-capability, EOS, and repetition gates, and produced no certifiable
intelligence-retention headline.  The seven-case manual packet remains
unrated, so the headline is also formally withheld rather than inferred from
the older topical-relevance score.

The 15% allocation is especially diagnostic: its first six layers reached
rank `680`, while later layers fell to ranks `172-253`, yet it was no more
coherent than the much lower-rank 20% allocation.  The missing component is
therefore not merely a few more singular directions or a better local
matrix-reconstruction objective.  Independent `w2` factorization accumulates
off-manifold residual error across depth; a future factorized lane must
allocate and solve ranks jointly against end-to-end residual/attention
trajectories, or retain an explicit cross-layer residual dictionary, before
whole-output generation.

Both runs used the existing parent caches only, performed zero training and
zero new parent forwards/generations, restored source state
`433f925c94fb8c508d7624db1c970e9e2d84c8af383fe54aa27a594159077271`,
and published no checkpoint.  Authenticated artifacts are:

- 15%:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_activation_factorization_20260717/target15-v1/report.json`,
  file SHA-256
  `97023d823695abf6739b8cfa083f35f73b0164a8abe27b8b558f75b6df3501dc`,
  run seal
  `48b884495197c8398d2381f8b789651b6bb857e57b26dac81a232fe25f69224f`;
  strict-v2 report file SHA-256
  `70998e070a31d09441145bdc3c522a9aab853b788740bb96b9715f3764dd1e54`,
  report seal
  `977ef8b3a0debc622d4ec21db745bb590b421da6284e4caf62e7ff136b118cc6`;
- 20%:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_activation_factorization_20260717/target20-v1/report.json`,
  file SHA-256
  `ad6dcaf6eda6b1dc207edd89aba02d080a612d140ad1a4559b8b9f4b3150e82b`,
  run seal
  `ee5e13adbf4c8a465634c7cf146fc142421a242d1678872af73e44063c131d70`;
  strict-v2 report file SHA-256
  `8aade115e0f1f4f9d49d4b920b2d0ccd6c78cf5f4081128b2a06cb5426b55ff8`,
  report seal
  `82d183b4c0cb15cbd4d4141089782fa6d4e8693d3ef51e3211c9da350344c8e0`.

A shared residual-output-basis variant then tested whether the fourteen
residual-writing `w2` projections could retain distinct private operators
while sharing only their output subspace.  The installed runtime used one
physical `1024 x 597` basis and fourteen private `597 x 2026` coefficient
matrices, fitted with layer-balanced activation evidence.  It reached
`195,226,812` unique learned parameters, or `15.0053960678%` total removal
from the original parent.

The local projection diagnostic looked strong: mean held-out layer cosine was
`0.9625460007` and the worst layer was `0.9451569071`.  Those independent
fits did not compose through depth.  Against the original parent, the
end-to-end trajectory score fell to `0.3876031917` and teacher-forced top-1
agreement to `0.4192872117`.  The whole-output gate found zero reliable
correct answers, task retention `0.291667`, fourteen new empty responses,
fourteen new fragments, seventeen semantic/task regressions, and fourteen
repetition regressions.  The candidate was rejected, exactly rolled back,
and published no checkpoint.  It performed no training, optimizer step,
backward call, new parent forward, or new parent response generation.

This closes one shared linear output basis by itself at the 15% target.  A
high per-layer reconstruction cosine is not enough when the same basis
constrains every residual write: small directional errors still compound
through attention and the residual stream.  Future basis sharing would need
trajectory-aware multiple dictionaries, sparse exact residual directions, or
an explicit downstream correction mechanism rather than a single global
subspace.

The authenticated report is
`/home/cereal/model-surgery-lab/artifacts/lfm2_shared_residual_basis_20260717/target15-layer-balanced-v1/report.json`;
its file SHA-256 is
`2f65583b75d49994d136e4fa2cfb8cf41761769261bffc1cffafe74bcd3b7d6a`
and its internal run seal is
`160bfc7fd5169b06e13d8cc1186472c8997829e42e8df55c388033393b258b00`.

The exact 20% boundary made the same conclusion stronger.  Rank `206`
produced a real `183,736,104`-parameter runtime
(`20.0080294938%` removed).  Local held-out layer cosine had already fallen
to mean `0.830233`, worst `0.787634`; the composed parent trajectory fell to
score `0.225268`, residual cosine `0.565732`, RRMSE `0.955992`, and
teacher-forced top-1 `0.073375`.  Whole-output behavior retained zero task
score, added sixteen empty/fragment responses, fifteen generation-limit
saturations, twenty-four semantic/task regressions, and nineteen repetition
regressions.  It exactly rolled back with no checkpoint, training, gradients,
optimizer, or parent calls.  The report file SHA-256 is
`8e45d8af63e5f791e24948684c6db8d02b0f1d5ae7cc2a6f0aef9db507490a44`
and its internal run seal is
`2447ce9549822764851d384f1408446969d0ad0ce8f9ded79d5c3a0e48467b07`.

A separate variable-width lane removed the assumption that every layer should
pay the same FFN contraction.  It measured a held-out contribution-loss curve
for every layer, solved the exact global budget by dynamic programming, then
recaptured the live child and attempted one prior-preserving `w2` residual
correction at a time.  The corrected 15% rerun used
`full_prefix_canonical` traces: every child trace used `use_cache=False`,
accepted zero cached child logits, and retained the original numeric
tolerances.

| Total removal | Candidate parameters | Accepted `w2` corrections | Parent residual cosine / RRMSE | Whole-output task retention | Failure | Verdict |
|---:|---:|---:|---:|---:|---|---|
| `14.9993480%` | `195,240,704` | `12/14` | `0.904757 / 0.416276` | `0.125` | `1` new empty, `1` fragment, `2` new saturations, `7/12` semantic regressions, `9` repetition regressions | reject |
| `20.0000223%` | `183,754,496` | `13/14` | `0.883406 / 0.470165` | `0.000` | nonempty retention `0.625`, `3` new empties, `3` fragments, `5` saturations, `8` repetition regressions | reject |

The 15% allocation was strongly nonuniform:
`1100,1100,2560,2560,2560,1250,1536,1371,2560,2560,1372,2560,768,768`.
It is materially less damaged than the 20% candidate, proving that uneven
sensitivity allocation and live child recapture help.  It is still nowhere
near coherent: only one of twelve promotion responses was automatically
relevant, and the outputs remain repetitive or instruction-wrong.  This
closes layer-local contribution curves plus sequential `w2` correction as a
standalone route.  The next structural allocator must score proposed removals
against downstream final-logit margins, EOS/repetition behavior, and the
residual-attention path globally.

The canonical 15% report is
`/home/cereal/model-surgery-lab/artifacts/lfm2_variable_width_20260717/measured-v2-full-prefix-canonical-15pct/report.json`;
its file SHA-256 is
`8d72d1326ef18b7e7f2978a1d8acc9c170e414fd17316c3e309bcddc8fb99899`,
its internal report seal is
`ef5e06d225d10628ac808f75c04694eb3a6d81c93a6622fd17976bda6e04e146`,
and its sealed manual packet identity is
`111f90c2b643cb0b570b1fce6c9cf71e51a0c81cfc12bb9618c33f1854494ce4`.
Both candidates exactly rolled back and published no checkpoint.  They used
zero training, gradients, optimizer steps, new parent forwards, or new parent
responses.

A tied vocabulary-interface frontier then isolated another possible large
compression surface.  LFM2 uses one exact shared `65,536 x 1,024` Parameter
for both token embedding and the LM head, so the experiment replaced that
single `67,108,864`-parameter matrix with one physically shared factor pair.
One deterministic CPU-float64 SVD was reused for all ranks and both source
allocations.  No dense vocabulary matrix remained, every non-vocabulary
module/Parameter/buffer stayed object- and byte-exact, and all six candidates
rolled back exactly:

| Source allocation | Rank | Candidate parameters | Total removal | Retained spectral energy | Parent trajectory score / top-1 | Strict reliable correctness | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 10% child, then vocab | `835` | `195,195,648` | `15.01896%` | `0.893978` | `0.37517 / 0.209375` | child `0/23`; retained `0/10` | reject |
| 10% child, then vocab | `663` | `183,747,328` | `20.00314%` | `0.777547` | `0.23241 / 0.096875` | child `0/23`; retained `0/10` | reject |
| 10% child, then vocab | `490` | `172,232,448` | `25.01630%` | `0.644659` | `0.20550 / 0.043750` | child `0/23`; retained `0/10` | reject |
| 10% child, then vocab | `145` | `149,269,248` | `35.01364%` | `0.313132` | `0.18313 / 0.012500` | child `0/23`; retained `0/10` | reject |
| parent, vocab only | `490` | `195,198,720` | `15.01763%` | `0.644659` | `0.20960 / 0.053125` | child `0/23`; retained `0/10` | reject |
| parent, vocab only | `318` | `183,750,400` | `20.00181%` | `0.493980` | `0.22438 / 0.081250` | child `0/23`; retained `0/10` | reject |

The rank-835 result is decisive: preserving `89.40%` of raw matrix energy did
not preserve the few vocabulary rows and logit margins that control
autoregressive decisions.  It added `22` generation-limit saturations,
`24` repetition regressions, three empty responses, and four fragments.
Outputs collapsed toward repeated spaces, repeated `1` tokens, and looping
arithmetic fragments.  The distributed 10%-width-plus-vocabulary allocation
was less damaged than direct vocabulary-only compression at the same total
size, but neither retained coherent capability.  Close global unweighted
tied-matrix SVD as a standalone mechanism.

The next justified vocabulary lane is a low-rank shared base plus exact sparse
residual rows.  The embedding and LM head must share one selected-token list
and one residual Parameter: embedding lookup adds the selected exact row
residual, while the head scatters `hidden @ residual.T` only into the same
selected logits.  Selection must prioritize special/EOS, observed prompt and
output tokens, parent top-k/small-margin competitors, occurrence, and actual
base-residual output energy using calibration evidence only.  It must be
judged by disjoint logit top-1, margin, and worst-case validation rather than
Frobenius energy.

That successor mechanism is now implemented in the canonical repository.  Its
runtime stores one shared low-rank base, one sorted integer token-ID buffer,
and one shared residual-row Parameter; it never materializes or retains a
dense candidate vocabulary matrix.  The selector is restricted to the
authenticated 16-case recovery horizon cache, with a deterministic
12-case-fit/4-case-validation split.  BOS, EOS, PAD, and every fit prompt token
are mandatory.  The old 12 manual prompts and the independent 64-case holdout
cannot influence row or rank selection.

The exact strong-child budget grid is deliberately much larger than the old
sub-percent probes:

| Total-parent target | Base ranks | Exact residual-row budgets |
|---:|---|---|
| `15%` | `768 / 640 / 512` | `4,397 / 12,717 / 21,037` |
| `20%` | `640 / 576 / 512` | `1,502 / 5,662 / 9,822` |
| `25%` | `448 / 384 / 320` | `2,766 / 6,926 / 11,086` |
| `35%` | `128 / 96 / 64` | `1,135 / 3,215 / 5,295` |
| `50%` | `64 / 32 / 16` | impossible in this isolated vocabulary lane |

The `50%` refusal is an exact structural result, not a quality judgment: the
strong child's unchanged non-vocabulary parameters alone total
`139,618,048`, already exceeding the `114,846,592`-parameter 50% target.
Reaching 50% therefore requires composing this vocabulary mechanism with a
non-vocabulary fold; the runner records every failed budget solution instead
of pretending the vocabulary surface can satisfy it.  A direct-parent 15%
control also reuses the same SVD with ranks `448 / 384 / 320`.

The frontier runner is
`experiments/run_lfm2_sparse_residual_vocab_frontier.py`.  It authenticates an
exact GPU UUID and the independent holdout file before any model load, ranks
allocations only by held-back top-1/margin/worst-case logit behavior, then
transactionally evaluates one selected allocation per feasible target on the
independent 64-case whole-response gate.  The independent parent envelope has
now been generated exactly once: `64` cases and `64` parent generation
invocations, manifest
`5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690`,
envelope seal
`b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2`,
and exact file SHA-256
`a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b`.
The reusable cache is
`/home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json`.

The focused `35%` sparse-residual run has now finished.  All three allocations
were screened on the disjoint four-case/76-row logit validation set.  Rank
`128` plus `1,135` exact residual rows won: it retained `0.9342105`
teacher-forced top-1 agreement, `0.9342105` positive parent-winner margins,
and `0.8684211` top-16 overlap.  The installed runtime was real:
`206,726,912 -> 149,299,968` unique learned parameters, or
`35.0002619%` removal from the original parent.  It retained no dense
vocabulary matrix, shared one factor core between embedding and head, left
every non-vocabulary module byte-exact, and exactly rolled back.

That strong local logit screen did not preserve autoregressive intelligence.
On the independent 64-case gate the candidate retained `0/11` parent-correct
tasks and `0/8` demonstrated capabilities.  It added `15` empty responses and
one fragment; nonempty retention was `0.6808511`, EOS/limit retention
`0.671875`, and repetition retention `0.453125`.  Code syntax, executable code
behavior, arithmetic exactness, and parent-demonstrated instruction compliance
all retained zero successes.  The candidate was rejected and no checkpoint
was published.

This closes low-rank vocabulary plus exact sparse token rows as the primary
route to `35%`.  The experiment is especially useful because it proves that
high teacher-forced next-token agreement on selected traces can coexist with
complete whole-task collapse.  Further large structural work must distribute
the budget over residual, attention, FFN, and vocabulary circuits and score
causal multi-token damage during allocation.

The authenticated result is
`/home/cereal/model-surgery-lab/artifacts/lfm2_sparse_residual_vocab_frontier_20260717/strong35-rtx3090-gpu4-v4-teacher-forced/report.json`.
Its exact file SHA-256 is
`06a659fdef7b23f4d21e0af3802740f887d7611a52bc921979635fd67524e4d6`
and its internal run seal is
`a617213cb906c16be737bf7baf7825bed098b3a7990b953bc68fe0b4781512c0`.
The independent holdout report file SHA-256 is
`12698908039f8b93dcf6b631057c548c43851f74669672c3f067a0e7ba7cee16`.
No training, gradients, optimizer step, new parent trace, new parent forward,
or new parent response generation was used.

The authenticated raw-SVD report is
`/home/cereal/model-surgery-lab/artifacts/lfm2_vocab_interface_frontier_20260717/raw-svd-v2/report.json`;
its file SHA-256 is
`f4213f585bf991f7a711dba6850c364a4c0d7a546732d32ed7023c47a881c989`
and its internal run seal is
`ebfe94d3d7373688392613f645da2f645f6984d4998bc4bb5c9f0cc714805150`.
No checkpoint was published.

The cached long-horizon residual-attention repair lane also completed two
termination-heavy arms against the 10% reference.  Each arm repaired the four
previously rejected layers `6,8,11,13` as one atomic tranche, using the sealed
64-token residual/attention paths.  Both reused parent horizon cache
`5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656`;
they generated no new parent responses, executed no new parent horizon
forwards, and performed no training, backward call, or optimizer step:

| Candidate | Ridge | Manual task retention | Semantic regressions | New saturations | Repetition regressions | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| `3` | `1e-8` | `0.625` | `3` | `2` | `5` | reject; exact rollback |
| `4` | `1e-6` | `0.375` | `5` | `3` | `4` | reject; exact rollback |

Candidate 3 regressed `manual_code_00`, `manual_ai_00`, and
`manual_fold_00`; candidate 4 additionally regressed `manual_chat_00` and
`manual_compare_00`.  Both retained every nonempty parent response, so their
failure is not blank-output collapse: the repairs still produced semantically
wrong, repetitive, or nonterminating continuations.  The tranche state hashes
returned exactly to the 10% source-child identity
`433f925c94fb8c508d7624db1c970e9e2d84c8af383fe54aa27a594159077271`,
and neither arm published a checkpoint.

The authenticated remote reports are:

- candidate 3:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_long_horizon_parallel_20260717/candidate-3/report.json`;
  file SHA-256
  `1125af234338bdb2aa686a414435aa5532ec3dd629419b543596212e36e9239b`,
  report self-seal
  `501d46bfdba6248d09f76b34b4df1d25e8121932f76d916fdef613e7992b2374`,
  tranche seal
  `c3b5da12db949ea2780bf786552e4d045bab6599e2ab39ea8c12098d03f5c60b`;
- candidate 4:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_long_horizon_parallel_20260717/candidate-4/report.json`;
  file SHA-256
  `1473b9fb39a38b192c4bd2aa8c6f745be996967404c149e245ff612ea5718afa`,
  report self-seal
  `abe5b183373a094126489dc5bc9c18032464ee2d4a784d6be99dd185da582f8d`,
  tranche seal
  `3fc053316ff6e78f4bd0f7e88a326eff13ab675a9b36932bac5a25793b2a4d3b`.

The report, candidate, tranche, and behavior-event seals all recompute
exactly.  This closes these two termination-heavy ridge settings as repair
strategies.  It does not close residual-attention-aware folding generally:
the evidence shows that endpoint repair alone cannot recover coherent
whole-output behavior after the current four-layer mutation.

The complete serial residual-attention grid has also finished.  It evaluated
all three ridges for both anchor profiles from a clean, authenticated 10%
source state.  Every candidate retained all nonempty parent responses, but
every candidate failed semantic, termination, or repetition gates and was
exactly rolled back:

| Candidate | Anchor profile | Endpoint ridge | Manual task retention | Semantic regressions | New saturations | Repetition regressions | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| `0` | balanced | `1e-8` | `0.375` | `5` | `3` | `5` | reject |
| `1` | balanced | `1e-6` | `0.500` | `4` | `2` | `3` | reject |
| `2` | balanced | `1e-4` | `0.500` | `4` | `3` | `5` | reject |
| `3` | termination-heavy | `1e-8` | `0.625` | `3` | `2` | `5` | reject |
| `4` | termination-heavy | `1e-6` | `0.500` | `4` | `2` | `2` | reject |
| `5` | termination-heavy | `1e-4` | `0.500` | `4` | `2` | `2` | reject |

This final grid is authenticated at
`/home/cereal/model-surgery-lab/artifacts/lfm2_long_horizon_repair_20260716/report.json`.
The exact report-file SHA-256 is
`ee234d60553f8aca17f35a7dc9e3cf61a2dc4bf7db1f939f54644ce2a736968b`;
the recomputed internal report seal is
`2525bbe6c689a3774f971a35b6341cfc8720b821459e5583f1f634ae373ae0c9`.
The file is canonical JSON, the seal verifies, no candidate was published,
and no parent forwards, training, gradients, or optimizer steps were used.
The earlier parallel candidate reports remain separate authenticated
executions; small scalar differences between separate rejected executions do
not change the closed-lane conclusion.

## Current production verdict

There is no production-accepted LFM2 fold-only child yet.

The authenticated `run-005-final` width-2552 attempt is rejected fail-closed
and publishes no artifact. It compacts `229,693,184 -> 229,349,120`, removing
344,064 learned parameters (0.1497928646%) with zero training. Its trajectory
sub-gate accepts:

- teacher-forced top-1: `0.9625`;
- free-run token match: `0.8125`;
- mean residual cosine: `0.998978597287314`;
- mean residual relative RMSE: `0.04214396084440369`;
- trajectory score: `0.9117237446914597`.

The strict 28-case whole-output sub-gate rejects:

- semantic regression on `manual_chat_00`;
- new generation-limit saturation on `recovery_code_00238`;
- task-score retention `23/24 = 0.9583333333333334`.

Human inspection confirms that all eight recovery-code answers are
nonfunctional, four manual cases are empty, and multiple long-form answers are
contradictory, inaccurate, or truncated. Because the in-memory gate rejects,
there is no artifact certificate, no freshly reloaded child, and no
fresh-reload output set.

The receipt seal is
`1862be5f31e0a2067ddb919bf5b37ab9f8bba2e192eb57ba91f3eaf4dfe63e25`;
the diagnostics seal is
`736d69bc7bc9940d8a88bf4aff6dd7207f813664b23893e59a49262410bf0c6b`;
the rejected ledger fingerprint is
`2b9e4e0e41fc0456eb4e04b2976701112541d1583ff6f55aee10161c44de103b`;
and the in-memory output-list hash is
`529f002e5c90efb145d796b5d3fa3db0bc9d6b07a424c89feb7301735722ca70`.

## Historical mechanical and trajectory ladder

The progressive coupled-SwiGLU folding lane has now produced eleven consecutive
standalone LFM2.5-230M children with real learned-parameter removal, no recovery
training, exact save/reload equivalence, and passing parent-relative trajectory
gates. These older reports establish mechanically valid compaction and a
useful failure frontier under the earlier contract; they do not override the
stricter production verdict above.

| FFN width | Learned parameters removed | Removed | Gate score | Teacher top-1 | Free-run token match | Residual cosine | Residual RRMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2559 | 43,008 | 0.018724% | 0.916373 | 0.990625 | 0.806250 | 0.999787 | 0.018922 |
| 2558 | 86,016 | 0.037448% | 0.912139 | 0.981250 | 0.806250 | 0.999681 | 0.023314 |
| 2556 | 172,032 | 0.074896% | 0.911011 | 0.978125 | 0.806250 | 0.999442 | 0.031156 |
| 2552 | 344,064 | 0.149793% | 0.922667 | 0.975000 | 0.831250 | 0.999036 | 0.040665 |
| 2544 | 688,128 | 0.299586% | 0.902995 | 0.959375 | 0.793750 | 0.998095 | 0.056888 |
| 2536 | 1,032,192 | 0.449379% | 0.872227 | 0.953125 | 0.725000 | 0.997086 | 0.071217 |
| 2528 | 1,376,256 | 0.599171% | 0.872518 | 0.931250 | 0.750000 | 0.995924 | 0.084759 |
| 2496 | 2,752,512 | 1.198343% | 0.854771 | 0.912500 | 0.718750 | 0.991999 | 0.119006 |
| 2464 | 4,128,768 | 1.797514% | 0.861303 | 0.918750 | 0.731250 | 0.987539 | 0.147960 |
| 2432 | 5,505,024 | 2.396686% | 0.852657 | 0.903125 | 0.725000 | 0.982884 | 0.172616 |
| 2368 | 8,257,536 | 3.595029% | 0.828502 | 0.846875 | 0.718750 | 0.975278 | 0.208409 |

All eleven reports record:

- exact target parameter accounting;
- a standalone freshly reloaded child;
- trajectory equivalence between the saved in-memory child and fresh reload;
- no parent or parameter sidecar at inference;
- no optimizer, backward, or recovery-training step in the audited runner;
- no parameter gradients.

The local evidence copies are under `runs/lfm2_dynamic_20260716/`.

## Earlier exploratory whole-output review

The observations below remain useful historical evidence. Any language that
previously “promoted” or “accepted” width 2552 is superseded by the binding
strict verdict above.

Final-only output packets were generated for the 2559 and 2558 children using
28 held-out prompts. Cached parent responses were reused with zero new parent
forwards.

- Both children produced coherent conversational, arithmetic, explanatory,
  and creative text on multiple prompts.
- Both exactly matched the parent on 13/28 outputs and matched each other on
  23/28.
- Parent and both children contained the expected answer on 7/8 held-out
  arithmetic prompts, including preservation of the parent's existing wrong
  answer for `86 + 73`.
- The 2559 child completed a coherent three-sentence robot story; the 2558
  child truncated its version.
- Parent and children all remained weak on several code, factual, safety, and
  instruction-following prompts. Folding preserved the small parent's
  limitations; it did not create new intelligence.

The second review compared widths 2536, 2528, and 2496 on the same sealed
28-prompt set:

- All three answered the eight arithmetic prompts coherently. Width 2528 was
  the only model to answer all eight correctly; it corrected the parent's
  incorrect `86 + 73 = 153` to `159`.
- Width 2528 produced the strongest palindrome implementation and the cleanest
  RAM-versus-disk answer.
- Width 2496 produced the clearest explanation of why one-token agreement is
  insufficient and a coherent robot/key story, but its factual rain answer and
  several generated code continuations degraded more.
- All three lost the parent's greeting response and, like the parent, emitted
  nothing on the safety, exact-one-word, syllogism, and summary prompts.
- The recovery-code prompts remained malformed across parent and children.
  This is a weakness of the tiny source model/evaluation set, not evidence that
  the child has reached broad assistant intelligence.

Width 2528 is the best round-three research checkpoint: it removes 1,376,256
parameters (0.599171%) and has the best whole-output tradeoff of that group.
It is not promoted, because the new empty greeting is a critical regression
and all eight recovery-code tasks fail their intended two-argument operation
semantics. Width 2496 proves that 1.198343% fold-only compaction can retain
islands of intelligence, but it marks a worse quality boundary rather than a
new incumbent. In that exploratory packet, width 2559 remained the preferred
manual-coherence reference; it was not a hardened production certificate.

The mid-frontier review then compared widths 2556, 2552, and 2544. The
automatic critical-regression gate accepted all three: each has exactly the
same four empty and four fragment-like response IDs as the parent, with no new
empty or fragment IDs. Cached parent responses were reused and parent forward
passes remained zero.

Manual whole-output review initially nominated width 2552 as the candidate safe
frontier:

- 344,064 learned parameters removed (0.149793%);
- the strongest internal gate score (`0.922667`) and free-run token retention
  (`0.83125`) in the reviewed group;
- coherent greeting and conceptual responses;
- a valid case- and space-insensitive palindrome function;
- 7/8 inherited arithmetic answers;
- no additional blank or fragment-like response;
- five maximum-length responses, equal to the parent.

Width 2556 is dominated by 2552: it compresses less, produces seven
maximum-length responses, and gives a nonsensical rain explanation. Width 2544
is the deepest reviewed child that passes the automatic critical gate, but it
produces eight maximum-length responses, a repetitive/truncated RAM answer, an
unfinished story, and a weaker folding explanation. It remains a useful
0.299586% research checkpoint, not the exploratory width-2552 reference.

The exploratory whole-output gate encoded this manual lesson. With cached
generation-length and repeated-ngram metadata enabled, that older packet replay
accepted 2552 at parent-level total saturation and rejected 2544 for three
newly saturated outputs (`recovery_code_00238`, `recovery_code_00145`, and
`manual_compare_00`). The later binding `run-005-final` gate is stricter and
rejects 2552 itself.

The parent and all three children still fail all eight malformed recovery-code
prompts and remain blank on four task types. These are inherited source/eval
weaknesses, not capabilities created by folding.

No flattering aggregate score was assigned. Exact outputs and sealed packets
are in `runs/lfm2_dynamic_20260716/review/` and
`runs/lfm2_dynamic_20260716/review-round3/`, with the new safe-frontier
comparison in `runs/lfm2_dynamic_20260716/review-midfrontier/`.

## Closed-loop refinement failure

The 2559 full closed-loop refinement run did not complete. During the first
candidate child-trace refresh, cached decoding and a no-cache full-prefix
replay differed by a maximum logit error of `1.375`. The trace verifier failed
closed before accepting or rejecting a layer. The tolerance was not weakened.

This does not invalidate the provisional coupled-neuron folds above. It shows
that the current endpoint-refinement transaction or its cached decoding path
needs diagnosis before it can be promoted.

## Current frontier

The largest mechanically and trajectory-gated child is FFN width 2368, with
8,257,536 learned parameters removed (3.595029%). No child currently passes the
hardened production intelligence and fresh-reload publication contract. Width
2552, with 344,064 parameters removed (0.149793%), is the closest strict
failure-boundary candidate. Width 2544 is mechanically deeper but has worse
truncation/repetition, and width 2528 remains a deeper failure-boundary
checkpoint with a blank-greeting regression.

Attention parameters remain exact during this lane so MLP compaction and
attention-parameter compaction are not conflated. Cached residual and
attention-context evidence still informs the coupled SwiGLU fold.
