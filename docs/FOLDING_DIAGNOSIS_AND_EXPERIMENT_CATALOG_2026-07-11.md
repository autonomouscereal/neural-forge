# Folding Diagnosis And Experiment Catalog

Date: 2026-07-11

This document answers the recurring questions about blank initialization,
anchors, folding, and why the project has not yet produced a coherent 25x
zero-recovery child. It also preserves the materially distinct 2026 experiment
families so a method pivot does not become a restart or an accidental repeat.

The authoritative Qwen3-4B parent cache has `4,022,468,096` parameters and
scores PPL `877.579654`, behavior `400/1000`, exact generation `388/1000`, and
forced choice `250/250`. Results from older, smaller suites are identified as
such and must not be compared as if their denominators were identical.

## Direct answers

### Is random initialization too spread out or not spread out enough?

No single weight-scale or random-spread explanation fits the evidence.

Target-RMS, source-RMS, geometric-mean RMS, unscaled mean, RMS-preserving
folds, tied and untied heads, multiple hidden widths, same-tokenizer blank Qwen
children, and cross-tokenizer blank SmolLM children have all been tested. They
change numerical scale and sometimes PPL substantially, but they do not turn a
collapsed 25-30x fold into a coherent autoregressive model.

Scale is a stability control. It is not the root missing mechanism.

### Are we failing to plant enough parent weights or anchors?

Not in the simple sense of anchor count.

The project has tried tens of thousands of exact token anchors, complete real
GQA and SwiGLU paths, activation-selected subsets, real medoids, PCA bases,
oblique real-axis anchors retaining `98.24%` of sampled energy, full-vocabulary
head fitting, sparse all-path coefficients, pair compensation, and sparse
multi-anchor compensation. Several of those methods make the parent signal
measurably decodable, yet the free-running child still visits the wrong hidden
states and generates loops or fragments.

More anchors can help only if they are the right coupled, reachable, and
observable circuits and if the rest of the parent function is folded into them
without breaking their algebra.

### What is the main blocker?

The missing object is the parent's coupled sequential closed-loop function.

A transformer is not a bag of independently useful weights. Q and K jointly
define a softmax decision; V and O jointly move the residual stream; gate, up,
post-SwiGLU activation, and down jointly define one nonlinear MLP path; residual
blocks act in a specific order. The child then consumes its own generated token
and changes the next state distribution. Independent averaging, locally good
regressions, or teacher-forced endpoint fits can damage those relationships.

In child-friendly terms: we have repeatedly copied many good puzzle pieces and
even proved that the picture is hidden in the pile, but we have not preserved
the hinges that make the pieces move together after the child takes its first
step by itself.

### Has any compaction been proved?

Yes, in three deliberately separate senses.

| Proof | Size / ratio | Strongest result | Honest meaning |
|---|---:|---|---|
| Useful recovered blank child | `134,515,008` parameters, about `29.9x` | `200/200` generation, `161/200` behavior, `40/64` choice; code, tokenizer, commonsense, and model-surgery `40/40`; math `1/40` | A useful target-size child exists after recovery. It is not a fold-only or donor-equivalent result. |
| Fold-only attention codebook | `3,995,570,176` parameters, `1.006732x`; saves `26,897,920` | Parent-pass behavior `400/400`, choice `250/250`, strict generation `372/388`, PPL `1,012.671` | Real zero-recovery compaction is proved, but only at about `0.67%`, not 25x. |
| Fold-only individual MLP paths | Saves `7,680` parameters per edit | Layers 35, 34, and 33 each retained behavior `400/400`, choice `250/250`, and parent generation `383/384`; net pass-count changes were `0`, `+1`, and `+2` | High-retention uneven Pareto steps exist. Their safe composition is not yet proved. |

No experiment has yet proved a coherent `>=25x` zero-recovery blank child.

## Experiment catalog: merge, bootstrap, and one-shot folding

| Method | Explanation for a child | Target and strongest result | Failure, proof, and reuse decision |
|---|---|---|---|
| Same-shape weight merge and task arithmetic | Mix two related books page by page. | Qwen 596M 50/50 merge: PPL `383.395`, choice `1.0`. Task-vector alpha `0.25` and `0.50` passed the early gate. | Related, same-shaped trained parents can merge. This is not compression. Keep as a merge primitive. |
| Net2Wider expansion | Copy a neuron and split its outgoing work. | Merged 596M Qwen expanded 5% to 609M with PPL `383.855` versus `383.395`. | Function-preserving expansion works much better than blind shrink. Reuse for the inverse operation. |
| Blind FFN shrink | Push neighboring neurons together because their indices fit. | A 10% shrink reached PPL `1,615,475` and choice `0`. | A dimension-valid checkpoint is not a functional child. Do not rerun unchanged. |
| Coupled small width/depth folds | Match whole gate/up/down paths and combine selected layers. | Width `0.995` kept choice `1.0`; layer `13+14` saved `2.64%`; the combined control saved `2.85%`. | Modest folds can survive, but layer location matters and later or multiple groups failed. Retain only as low-ratio controls. |
| Tiny-alpha trained-host transplant | Sprinkle donor token rows or MLP channels into an already trained child. | SmolLM vocabulary transplant improved PPL `347.405 -> 334.584`; Qwen hosts from Pythia/LFM preserved choice `1.0` with small PPL changes. | Safe signal injection is possible, but donor capability was not measurably inherited. Not a blank-child answer. |
| Tiny-alpha random-target transplant | Sprinkle a tiny amount of parent signal into random weights. | Blank SmolLM `134.5M`, `29.9x`: benchmark-identical to random and choice `0`. | Planting at tiny alpha is far too weak. Close unchanged. |
| Direct mean fold | Average all parent coordinates assigned to each child coordinate. | Blank `134.5M`, `29.9x`: PPL `63,726.945`, choice `1/4`; WinoGrande moved `0.46 -> 0.64` on a limit-50 slice. | It implanted some task statistics, but generation repeated unrelated words. Keep only as a raw-fold baseline. |
| RMS, source-RMS, target-RMS, and geomean-RMS scaling | Change how loudly folded weights speak. | Each mode behaved numerically as designed; RMS fold PPL was `68,513.184`. | None restored behavior. Do not sweep scale by itself. |
| Activation-weighted and activation-similarity fold | Put frequently or similarly firing parent neurons in the same child buckets. | Activation-similarity PPL `59,140.629`; after LM-only recovery `32,577.546`. | Better numeric fit still produced repeated-token collapse. Closed as a standalone method. |
| Same-tokenizer blank Qwen stress | Remove tokenizer mismatch to isolate the internal fold. | `125-142M`, `28-32x`; raw behavior `0/6`. After as many as 2,400 KD steps, behavior remained `0/6`. | Tokenizer mismatch is not the primary blocker. Do not repeat unchanged. |
| PCA hidden basis | Keep the parent's largest activation directions. | Per-layer PCA plus HLD reached PPL `3,401.72`, choice `4/4`, behavior `1/6`. July 10 blank `134M` PCA reached PPL `164,240.637`, behavior/generation `0`, choice `71/250`. | High retained variance is not an executable circuit. Closed as a standalone fold. |
| Dual-coordinate basis | Build paired producer and consumer coordinates. | Blank `134M`, `29.9x`: PPL `189,291,282.955`, behavior/generation `0`, choice `87/250`. | Better coordinate duality still lost the closed-loop function. Closed unchanged. |
| Real-neuron activation subset | Keep 1,408 complete real SwiGLU paths and complete real GQA groups. | Blank `134M`, `29.9x`: behavior/generation `0`, choice `87/250`. | Removing roughly 96% of the active circuit cannot be repaired by importance selection alone. Close the standalone sparse-subset hypothesis. |
| Same-width real-medoid | Preserve parent width and real representatives while reducing capacity. | `1.130B`, only `3.56x`: PPL `67,570,658.930`, behavior/generation `0`, choice `88/250`. | Collapse occurs well before 25x. Do not rerun unchanged. |
| Dense Sinkhorn optimal transport | Let every source neuron distribute mass globally under a balanced load. | Blank 141M raw PPL `376,549.34`, behavior `0/6`; after HLD PPL `3,862.52`, behavior `1/6`. | Balanced transport is useful infrastructure, but the tested cost/features did not preserve function. Reuse only inside a stronger coupled pipeline. |
| Quantile, medoid, quantile-medoid, and domain-diverse support | Reserve child buckets across activation levels or domains and anchor them on real neurons. | The support selectors and diagnostics worked; completed one-shot descendants remained collapsed. | Better support alone is insufficient. Do not rerun unchanged scale-only variants. |
| Signed, balanced-signed, CountSketch, superposition, PCA-extrema, volume, and operator-product support | Try to cancel collisions or preserve rare/full-path directions. | Implemented as structural candidate families; no certified coherent winner exists in the evidence. | Preserve as hypothesis generators. Only compare them through the same parent-relative gate. |

## Experiment catalog: conjugate paths, depth, and closed-form folding

| Method | Explanation for a child | Target and strongest result | Failure, proof, and reuse decision |
|---|---|---|---|
| Path-conjugate coordinate/grouped fold | Use exactly the same map on both sides of every pathway. | Width-only `139.7M`, `28.8x`: PPL `189k-278k`, behavior/generation `0`, choice `16-33/250`. Attention-rich PCA variant reached choice `92/250`. | Correct pairing alone does not preserve softmax, nonlinear MLP, and residual composition. Close unchanged. |
| Oblique QR/ridge real-axis basis | Keep real anchor coordinates and reconstruct discarded coordinates from them. | `156.1M`, `25.77x`; corrected run retained `98.24%` sampled energy, cosine `0.98059`, and applied all `180` closed-form input fits. Behavior/generation `0`, choice `88/250`. | Excellent basis reconstruction can still destroy rollout. Never promote or rerun solely for a better basis metric. |
| Sequential functional output fitting | Refit each attention and MLP output without gradients and recompute downstream activations. | `156.1M`, `25.77x`; all 36 attention/MLP output fits accepted. Behavior/generation `0`, choice ranged `24-98/250`. | Local output fits did not compose into a working model. Close unchanged. |
| Ordered pair composition | Put intact paths from two ordered parent layers into one child block. | `158.3M`, `25.41x`: choice `146/250`, behavior/generation `0`. | It preserved more choice signal, but too little nonlinear producer capacity and high absolute block error caused incoherence. Keep as a comparison anchor. |
| Full-vocabulary head isolation | Ask whether the folded body still contains linearly decodable parent state. | Teacher top-1 agreement `11.67% -> 65.75%`, teacher top-1 in child top-5 `30.55% -> 85.67%`, KL `10.41 -> 1.20`; behavior/generation stayed `0`. | Parent information can be present teacher-forced while the free-running state trajectory is wrong. Retain as a diagnostic. |
| Sparse all-path and transactional path gates | Add discarded paths through sparse consumer coefficients, reverting harmful fits. | `158.3M`, `25.41x`; transactional run reached choice `154/250`, behavior/generation `0`; accepted pair fits `4/18`, attention aggregates `0/18`, MLP aggregates `0/18`. | Rollback preserves signal but does not create coherence. Keep transactionality; close the unchanged sparse fit. |
| Activation-PCA depth folding | Average or project nonlinear layer pairs into fewer executions. | `146.1M`, `27.53x`: PPL `285,161.12`, behavior/generation `0`, choice `23/250`. | Averaging parameters is not ordered function composition. Closed. |
| Full-topology shared depth basis | Keep all logical blocks but store shared low-rank projection bases. | `134.44M`, `29.92x`: PPL `141,540.313`, behavior/generation `0`, choice `9/250`. | Logical topology without distinct functional operators is insufficient. Closed. |
| One projection set reused at every depth | Make all 36 layers reuse one real attention set and one real MLP set. | `122.1M`, `32.95x`: PPL `8,073,711.456`, behavior/generation `0`, choice `15/250`. | Repeated average operators cannot replace ordered residual steps. Closed. |
| Rank-one depth sharing | Store one depth basis for each projection role. | `490.1M`, `8.21x`: PPL `244,468,857.739`, behavior/generation `0`, choice `28/250`. | Even much milder extreme sharing collapsed. Closed. |
| Child-action-only rollout refit | Ask the parent whether the child's current actions are plausible, then refit. | Path-gated child stayed behavior/generation `0`; legacy rollout choice fell `146 -> 88/250` and PPL rose to `485M`. | Generic or repetitive child actions can look plausible to the parent. This selection rule is rejected. |

## Experiment catalog: codebooks and compensated local surgery

| Method | Explanation for a child | Strongest result | Failure, proof, and reuse decision |
|---|---|---|---|
| Executable `36/36` codebook identity | Store every real depth variant in a compact registry and reconstruct it at runtime. | Behavior `400/400`, choice `250/250`, parent generation `382/388`, PPL `882.312`. | It proved runtime, assignment, save/reload, and evaluator correctness. Parameter reduction was only padded vocabulary rows. This is the correct starting state. |
| Uniform `35/35` codebook | Merge one of 36 depth variants for every feature. | `1.025736x`: behavior `255/400`, choice `204/250`, generation `184/388`. | Some code/math survives, but prose, model-surgery, and much math collapse. Keep as the sharp failure boundary. |
| Uniform `34/34`, `32/32`, and `28/28` | Remove two or more depth variants everywhere. | `34/34` already falls to behavior/generation `0`; `32/32` choice `18/250`; `28/28` choice `20/250`. | Global uniform allocation crosses a cliff. Never rerun unchanged. |
| MLP `p35`, attention `p36` | Merge one MLP depth prototype while leaving attention exact. | `3.947B`, `1.019101x`: behavior `245/400`, choice `203/250`, generation `175/388`. | Uniform MLP merging is far more fragile than attention. Preserve as factorial negative evidence. |
| MLP `p36`, attention `p35` | Merge one attention depth prototype while leaving MLP exact. | Saves `26,897,920`: behavior `400/400`, choice `250/250`, generation `372/388`, PPL `1,012.671`. | This is the strongest fold-only structural frontier, not a target-size result. Preserve immutably. |
| Raw V-cache regression plus global O refit | Numerically fit the removed attention path's V cache, then refit its shared O write. | Calibration V RMSE improved `0.658301 -> 0.015534`; held-out functional risk worsened `0.841695 -> 7.153297`, forced top-1 mismatch and top-k damage were `1.0`, and all six greedy paths diverged. V+O risk was `8.785512`. | A strong internal tensor fit can be a functional false positive. Reject this repair and never use weight/cache/residual resemblance as a promotion gate. Retain only as initializer evidence. |
| Nine real Q/K transplant anchors | Keep V/O fixed and replace the shared survivor's entire Q+K pair with nearby active parent anchors. | The unchanged Q/K was best (`0.859123` held-out risk); all eight alternatives were `10.6-13.8%` worse, and four lost calibration actions. | No alternative anchor repaired addressing. Close discrete raw Q/K transplantation for this redirect; use a learned functional correction instead. |
| Per-assignment V/O functional search | Keep the exact p35 -> tentative p34 redirect and vary only its already-accounted V/O influence while Q/K and all other parameters remain fixed. | The tentative initializer nearly mutes the removed site (`V=-0.001249`, `O=-0.000539`). A signed output-function sweep is in progress. | Candidate selection uses parent actions and child rollouts only; held-out prompts cannot select. A favorable search result is still not a committed or broadly certified compaction. |
| Behavior-only learned V/O scales | Fit only the two existing `L19/G4` V/O assignment scales against fixed parent actions and output logits; use calibration only to choose between the fitted point and the incumbent. | Eight manual Adam steps moved `V/O` from `-0.001249/-0.000539` to `0.040375/-0.035156`, and fit loss moved `1.343306 -> 1.340941` at its minimum (`1.341919` final). Calibration behavior did not move: both candidates scored forced top-1 `19/24`, top-5 `24/24`, and greedy retention `2/6`; calibration kept the incumbent because its mean parent-action log-probability was slightly better (`-0.879031` versus `-0.879386`). Heldout behavior was also identical: forced top-1 `28/32`, top-5 `32/32`, and greedy retention `5/8`. | This is real fit-loss movement with **zero discrete behavioral gain**. The heldout-only ordering gave the fitted point a tiny log-probability edge, but it could not select and changed no token outcome. Reject the fitted point; retain the incumbent; no fold or checkpoint was committed. |
| MLP `p34`, attention `p36` | Merge two MLP depth variants. | `3.872B`, `1.038763x`: PPL `9,132.496`, behavior/generation `0`, choice `82/250`. | The MLP capacity cliff is between p35 and p34. Closed. |
| Pair neuron compensation | Remove one neuron and pour its output into one survivor. | Layers 35/34/33 each saved `7,680`, kept behavior `400/400`, choice `250/250`, and parent generation `383/384`; net pass changes `0`, `+1`, `+2`. | High-retention uneven steps exist. Reuse as transaction candidates and refresh evidence after every accepted edit. |
| Sparse multi-anchor compensation | Spread one removed neuron across several survivors. | Layer-18 validation error was `52-60%` lower than pair compensation. Single-layer savings were `491,520` for 64 neurons and `1,966,080` for 256. | Better local fit did not beat pair folding in full-model generation: pair-64 lost 4 parent generations, multi-anchor-64 lost 7. Reuse only for new candidates. |
| Full-model compensated controls | Install pair or multi-anchor edits and certify the whole model. | Tested candidates kept behavior `400/400` and choice `250/250`. | The old zero-lost-ID rule rejected every edit; the late strict retry accepted `0/64`. Do not repeat identical proposals. Use a parent-relative Pareto floor and test composition explicitly. |

The completed learned-V/O report is
`runs/residual-attention-20260711/p35_L19G4_to_L21G6_behavior_vo_tuner_h4_s8_fp16_p40_gpu2.json`.
It verified the exact `280 -> 279` edit, removing source `156` in favor of
survivor `174`. Attention storage changed from `1,835,011,432` to
`1,828,460,132` bytes, a physical reduction of `6,551,300` bytes. One
prototype accounts for `3,276,800` parameters and `6,553,600` bytes; the
existing role-scale registry accounts for `1,152` parameters and `2,304`
bytes, leaving a net reduction of `3,275,648` learned parameters. Tuning added
zero runtime parameters, saved no optimizer state, and restored both the
tentative child and incumbent p35 exactly. The file SHA-256 is
`86b0d9c500315d867a8097142956fd5aee22d3c207b9f683060ca9eaf93f8632`;
the verified top-level report hash is
`7d48b24a089c3bd85915a71bfa6f362c8500eb027d111cf31c4a8dfed52f9d2e`,
and the nested tuner report hash is
`dc7242445d2ef529d92be42ae9e0a251afcaaa684718a15d3756a615a07bab31`.
The calibration selection lock is
`8d8999b784e75cdcd4a0f4fc2e16f49c7250f16590e7cdde2e5223182379be77`.

## Experiment catalog: recovery, tokenizer bridges, and gates

Recovery results are kept separate because they demonstrate trainability and
parent-informed advantage, not zero-recovery folding.

| Recovery family | Explanation for a child | Best signal | Why it did not solve fold-only compaction |
|---|---|---|---|
| LM-only, anchor KD, span KD, and hidden KD | Practice text, shared tokens, aligned spans, and parent hidden states. | Reduced PPL from tens of thousands to low thousands and recovered partial language. | Random controls showed some gains came from ordinary training; loops remained. |
| CKA, HLD, RSD, Gram, and geometry losses | Make parent and child representation shapes agree without requiring equal width. | Per-layer PCA plus HLD reached choice `4/4` and one behavior pass; RSD preserved the existing 29.9x behavior branch with slightly lower PPL. | Hidden geometry can match while greedy decoding remains wrong. |
| Sparse top-k KD | Copy only the parent's preferred token set. | Choice sometimes reached `3/4` or `4/4`. | Behavior remained `0-1/6`, with malformed free generation. |
| Dual-space KD plus ALM | Compare states through opposite heads and match likelihood over common character chunks. | At `29.9x`, PPL `2,721.31` and behavior `4/6` on the old small gate. | Arithmetic and executable Python remained malformed; external tasks did not improve. |
| Blockwise hidden or residual-delta recovery | Train one child block at a time against a parent block or residual update. | Delta targeting slightly improved local alignment/PPL while preserving the incumbent. | It preserved the plateau but did not create missing math/code circuits. |
| Module-output and consumer-interface reconstruction | Match attention/MLP outputs or absorb a map into the next block. | Module-output CKA improved `0.4216 -> 0.5029`; many local errors fell sharply. | Local gains often harmed broad generation. Suite-backed rollback was necessary. |
| On-policy hidden and stepwise donor KD | Let the parent correct prefixes actually visited by the child. | Mechanically active and behavior-preserving under strict checkpoint selection. | Small corrections did not escape the child's collapsed attractors. |
| Teacher-generated continuation replay | Train on correct parent continuations retokenized by the child. | The donor supplied correct arithmetic/code strings and all intended losses were active. | Light schedules could not redirect the child's internal closed-loop state. |
| Answer rank, contrastive, first-token margin, and top-negative losses | Push the right answer above nearby distractors. | Occasionally recovered one executable add function or `2/40` math items. | It overfit answer surfaces, hurt prose/generation, and failed held-out arithmetic. |
| Focused CE, EOS, and response-end loss | Train only digits, code, response starts, or stop tokens. | Focused CE could emit `def add(a, b): return a + b`. | Broader generation fell; EOS stopped loops without fixing correctness. |
| Output-head ridge and mean-shift calibration | Move output rows toward parent logits. | Local projected-logit fit and PPL sometimes improved. | Greedy behavior degraded, especially when a tied head also mutated embeddings. |
| Weighted projected-vocab, byte KD, and byte-hidden probes | Bridge tokenizers through source pieces or byte buckets. | Mechanically active and often preserved topical behavior. | They did not remove numeric/code attractors. Temporary probes are recovery scaffolding, not published child components. |
| Residual RMS, full-rank, and low-rank compensation | Match parent/child residual magnitude or absorb an adapter-free map. | Rank-16 alpha `0.01` preserved broad `600/1000` behavior, `160/200` generation, `25/250` choice while improving broad PPL `14,259 -> 14,192`. | It added no math/code capability; full-rank maps could improve PPL while halving generation health. |
| Broad behavior-first and category-aware gates | Save states by behavior/category/generation before PPL and roll back losses. | Prevented many false promotions and retained stronger incumbents. | Essential experiment control, not a compression algorithm. |

The strongest recovery-assisted product checkpoint remains
`qwen4b_mathboost_answer_cleanopen_800_gpu4`: a target-sized child with useful
text and code, but only `1/40` math behavior in its 200-item comparison. It is
the product-quality frontier, not the fold-only frontier.

## Experiment catalog: cross-family, multi-parent, and native modality

| Experiment | Strongest result | What it proves and does not prove |
|---|---|---|
| Qwen host from Pythia or LFM MLP channels | Tiny-alpha 596M candidates preserved choice `1.0` and sometimes slightly improved PPL. | Cross-family tensor/layout translation works. Large capability transfer is not proved. |
| Qwen plus LFM merge-compress | `596M -> 564.6M`, `5.28%`; the early small text gate passed. | A trained host can be cross-family merged and modestly compressed. It is not blank-child compilation. |
| Qwen4B into trained SmolLM 135M | Folded artifacts were benchmark-identical to the host; the host-only recovery control gained the same amount. | Tiny-alpha donor transplantation did not transfer measurable Qwen4B capability. |
| GPT-2/Qwen cross-architecture diagnostic | A blank `134.5M` SmolLM checkpoint loaded with all 270 layer tensors initialized and finite. | Cross-architecture checkpoint construction works, not intelligence retention. |
| Qwen plus GPT-2 into blank SmolLM smoke | `134,515,008` parameters; both donors projected, all tensors finite. | A+B->C construction works mechanically. Sequential overwrite/recovery is not simultaneous attributable fusion, and no certified retention result exists. |
| LFM230M into blank SmolLM | Parent cache: behavior `382/1000`, generation `622/1000`, choice `250/250`, PPL `1,245.7`. | A different-family donor baseline exists. The evidence read contains no promoted compact child. |
| Native toy early-fusion model | `350,227` parameters; `1,024/1,024` exact synthetic answers after 1,600 training steps. | Native raw-patch causal architecture is trainable. It is not parent folding. |
| Raw-patch graft on merged/compressed Qwen | Added `66,560` parameters; synthetic vision reached `64/64` or `192/192`. | Native pixel-prefix mechanics work. It used training and a graft, so it does not meet the final compiled-child proof alone. |
| Vision graft on recovered 135M child | Frozen LM: vision `108/192`, text unchanged. Integrated: vision `192/192`, text behavior `147/200`, generation `117/200`. | Modality can be added, but integrated training currently trades away text quality. |
| NEO native raw-RGB parent | `2,896,618,496` parameters; overall `239/256`, held out `59/64`. | A real native-vision parent and reusable cache exist. Its compression has not been proved. |

## Adjacent project evidence that must not be mistaken for folding

| Evidence | Result | Correct reuse |
|---|---|---|
| Chimera semantic capability proof | Synthetic route, class, and answer accuracy `1.0`; media cosine `0.9999766`, MSE `2.81e-5`. | Reuse packet ABI, masking, provenance, causal tests, and fixtures. It is not compression evidence. |
| Chimera real-artifact proof | Route/class accuracy `1.0`; real PNG, WAV, and MP4 artifacts; media cosine `0.9574`, MSE `0.00413`. | Reuse artifact manifests and native-modality test contracts. |
| Chimera multitask interference | Mixed model kept code execution and tool dispatch `1.0`; media cosine fell `0.0644`; piano MSE `0.03024` exceeded its `0.018` threshold. | Reuse interference reporting; do not claim settled multimodal integration. |
| OmniCoder workspace | Provider routing, teacher records, byte estimation, fail-closed manifests, and contamination gates. | Reuse packaging and recovery infrastructure only after fold coherence. It is not a folding engine. |

## What the successful methods have in common

The highest-retention structural results are the methods that disturb the
smallest number of complete functional paths and validate the entire model:

1. attention `p35` rather than uniform MLP merging;
2. single, late, compensated MLP paths rather than broad neuron deletion;
3. exact rollback and cached parent-relative gates rather than PPL promotion;
4. real parent operators and assignments rather than fitting random child
   coordinates;
5. uneven allocation rather than the same compression quota everywhere.

The strongest target-size result used a much larger recovery stack. It proves
that the fold contains useful training signal, but its arithmetic gap and the
random-control comparisons show that recovery cannot be used to hide an
incoherent fold.

## Next experiment: combined closed-loop circuit folding

This is a new combined experiment. Several ingredients exist separately, but
the complete sequence below has not been tested as one end-to-end method.

### What is already implemented or previously tested

- executable `36/36` codebook identity and attention `p35` frontier;
- real-axis activation profiles, medoids, PCA/QR bases, and sensitivity data;
- pair and sparse multi-anchor MLP compensation;
- signed assignment, GQA-aware attention layouts, and coupled gate/up/down
  tensor routing;
- sparse and dense OT/Sinkhorn transport;
- sequential closed-form reconstruction and GRAIL-style downstream absorption;
- candidate rollback, cached parent reports, exact save/reload, and broad
  behavior/generation/choice gates;
- progressive `4B -> 1.4B -> 600M -> 134M` target planning.

Those bullets are not evidence that the proposed combination has already run.
They are reusable components.

### What is new in the proposed combination

#### 1. Seed-erasure audit

Before evaluating intelligence, prove how much of the published child's learned
state still comes from its random seed.

- Build the same fold from at least three blank seeds.
- Record provenance for every learned tensor and every retained random degree
  of freedom.
- Require all semantically active output to be parent-derived or explicitly
  reported as residual random features.
- Compare child function and tensor hashes after parent materialization across
  seeds.
- Reject a method whose result depends materially on untouched random child
  coordinates.

This directly tests the question "are we planting hard enough?" instead of
inferring the answer from PPL.

#### 2. Reachable-observable balanced-Gramian real-axis anchors

Choose exact parent axes by both sides of the control problem:

- **Reachability:** how strongly calibration and parent-rollout inputs excite a
  path or residual direction.
- **Observability:** how strongly changing that path affects held-out logits,
  parent actions, and free-running behavior.
- Form a bounded balanced-Gramian approximation from activation covariance and
  output/Jacobian sensitivity sketches.
- Use pivoted Cholesky/QR only to rank coverage, then select real parent axes as
  anchors instead of publishing synthetic basis vectors.
- Allocate anchors unevenly by measured retained behavior per byte.

Activation importance alone and output sensitivity alone were tried in weaker
forms. Their balanced intersection over real axes is new.

#### 3. Signed SwiGLU triplet segments and joint QK/VO attention groups

Treat nonlinear and attention circuits as indivisible candidate units.

- A SwiGLU segment contains gate row, up row, post-SwiGLU activation trace, and
  down column together.
- Enumerate only functionally legal signs. For example, an up-row sign flip can
  be canceled in its down column; an arbitrary gate sign flip is not assumed
  symmetric through SiLU.
- An attention unit contains a complete GQA group. Q and K share a
  dot-product-preserving signed permutation or common orthogonal subspace; V
  and O share the matching value/output map.
- Keep high-damage units exact. Fold only units whose complete coupled edit
  passes held-out parent-action probes.

The repository has coupled routing and signed assignments, but this exact
functional segment representation and legality check are new.

#### 4. All-to-survivor optimal transport

After pinning exact anchors, transport every non-anchor parent path into the
survivor set.

- Build costs from balanced-Gramian features, post-nonlinearity traces, module
  outputs, and operator-product contribution rather than raw weight cosine.
- Constrain mass jointly for each SwiGLU triplet or QK/VO group.
- Preserve signed contribution and load balance without averaging Q/K before
  softmax or gate/up before SwiGLU.
- Count all transport indices, scales, assignments, and runtime costs.

OT itself was tried. OT over these new coupled functional units with exact
anchors and closed-loop costs is new.

#### 5. Sequential GRAIL / Chain-of-Merges compensation

Apply one bounded transaction at a time.

1. Materialize the coupled anchor-plus-transport edit.
2. Rerun parent anchors and child free-running prefixes.
3. Fit a closed-form attention, MLP, residual, or downstream interface
   compensation on the new state distribution.
4. Re-evaluate on a disjoint fixed gate.
5. Accept or restore the exact incumbent.
6. Refresh all evidence before proposing the next edit.

Sequential compensation exists in parts. Combining it with the new circuit
units and refreshed closed-loop evidence is new.

#### 6. Functional error-feedback with OMP selection

Do not minimize each layer independently. Carry the remaining end-to-end
functional error forward.

- Maintain residual error traces for parent actions, logits, residual deltas,
  attention outputs, and post-SwiGLU outputs.
- Generate a bounded dictionary of exact paths, transported paths, and legal
  compensation operators.
- Use orthogonal matching pursuit or a related greedy residual selector to
  choose the next operator that removes the most held-out functional error per
  serialized byte.
- Subtract accepted contribution from the residual before selecting again.

OMP token initialization was tried and failed. OMP over end-to-end functional
operator residuals is a different, new use.

#### 7. Identity-to-progressive targets

Begin at the executable `36/36` identity, not at a random 25x endpoint.

- First reproduce the attention `p35` frontier and the known single-neuron
  Pareto edits under the new evaluator.
- Compress in small accepted budget steps while retaining a Pareto frontier.
- Once same-shape compaction is stable, compile through approximately
  `4B -> 1.4B -> 600M -> 134M` blank targets.
- A later target inherits only a certified functional predecessor, never an
  incoherent one-shot fold.

Progressive targets were planned and some chain infrastructure exists. This
identity-driven certified sequence is not yet complete.

#### 8. Fold-only promotion gates

Every frontier must satisfy all of the following before recovery:

- zero optimizer steps, backward calls, or gradient updates;
- exact target checkpoint budget and complete artifact accounting;
- parent-free save, reload, and byte-inventory verification;
- cached identical parent/child rows;
- `1,000` behavior, `1,000` exact-generation, and `250` forced-choice probes;
- PPL/NLL as diagnostics, not promotion authority;
- paired manual inspection across math, executable code, commonsense,
  tokenizer/model-surgery, and every transferred modality;
- approximately `90%` retained useful parent behavior, with explicit
  per-category floors;
- exact rollback on rejection and evidence refresh after acceptance.

Only after a coherent fold passes may recovery be compared against an
identically seeded random child to measure the claimed training reduction.

## Decision

The next useful work is not another random-scale sweep, another global PCA
rank, another uniform medoid count, or a larger pile of unstructured anchors.
It is the combined closed-loop experiment above: audit seed dependence, anchor
real paths that are both reachable and observable, fold complete nonlinear and
attention circuits into those anchors, feed end-to-end residual error forward,
and advance from identity only through behavior-certified transactions.

Existing results remain immutable comparisons. A new method must beat the
attention `p35` fold-only frontier and the recovered `29.9x` product frontier
on the dimensions each one actually proves.
