# Research Paths For Model Folding And Recovery

Current as of 2026-07-09. This file tracks implementation directions for the
blank-target, cross-family compression goal. It deliberately focuses on
2025-2026 work and actionable hooks for Neural Forge.

## Path-Conjugate No-Recovery Fold (2026-07-09)

The strict blank-target lane now has a `path-conjugate` initializer. It keeps one
global residual coordinate system by selecting real donor coordinates, preserves
RoPE head dimensions, selects real attention/MLP producer channels, and sums the
paired consumer paths with the same cached assignment. This avoids three invalid
operations in the older independent fold: changing PCA coordinates between
residual blocks, mixing channels before SwiGLU, and independently assigning
gate/up versus down or Q/V versus O channels.

The first proof target is a blank Qwen3-shaped `146,097,664` parameter model
(`hidden=512`, `intermediate=2048`, `layers=18`, `heads=4`, `kv_heads=1`,
`head_dim=128`), a `27.53x` reduction from Qwen3-4B Base. The reproducible
fold-only lane is
`experiments/run_qwen4b_blank_qwen146m_path_conjugate_20260709.sh`; promotion
still requires useful held-out generation and broad parent-relative behavior,
not PPL alone.

The first volume-medoid path run was a negative control: it retained `0/400`
parent behavior passes and `0/388` parent generation passes, with repeated
`.ifold` output despite `94/250` choice matches. Its assignment report exposed
an attention dispatch bug: Q/K/V used an anchor-only cache key while O used a
path cache key. The corrected initializer keeps both sides on one
`path-conjugate` map. It also couples grouped-query attention structurally: a
selected KV head brings its complete query-head group, discarded KV groups map
to the same within-group query offsets, and the matching O columns reuse that
map. This preserves Q-to-K/V identity instead of independently selecting query
and KV heads.

`activation-grouped` is the merge-first residual-width companion to the initial
coordinate-selection control. It assigns every donor residual coordinate to one
target coordinate, weights each assignment by activation mass, and normalizes
the disjoint groups so the global basis has orthonormal columns. RMSNorm gains
use the squared group coefficients, while embeddings, linears, residual outputs,
and the tied head use the same global basis. This retains all hidden-coordinate
paths without introducing a per-layer bridge or adapter.

The first true width-only `36 -> 36` control used a blank Qwen3 target with
`hidden=384`, `intermediate=1536`, `Q/KV=4/1`, and `139,673,856` evaluated
parameters (`28.80x`). Keeping every donor layer did not rescue the RMS-repaired
coordinate fold: PPL was `215,032.38`, behavior `0/1000`, generation `0/1000`,
and choice `31/250`, versus parent `877.58`, `400/1000`, `388/1000`, and
`250/250`. Manual math/code generations were multilingual fragments.

That run exposed a mathematical error in combining selection with RMS repair.
Producer paths use real selected rows, but the old repair treated a selected
anchor as if it carried the summed power of its entire cluster. For eight-way
attention-head compression this multiplies both Q and K RMS by about
`sqrt(8)`, making attention logits roughly eight times too large. RMS repair is
now a no-op for `select` reductions, and unrepaired l36 coordinate/grouped
controls are the required comparison. Both controls are now complete and are
negative. The unrepaired coordinate fold reached PPL `189,167.47`, behavior
`0/1000`, generation `0/1000`, and choice `16/250`; the unrepaired grouped fold
reached PPL `278,009.71`, behavior `0/1000`, generation `0/1000`, and choice
`33/250`. Parent-pass retention was zero for all `400` scored behavior items and
all `388` scored generation items. Paired manual inspection showed repeated
multilingual fragments for math and Python prompts. Removing the erroneous RMS
scaling therefore helps one local statistic but does not restore a usable
function at `28.80x`.

The next zero-gradient lane is `activation-oblique`. It selects real residual
coordinates by pivoted subspace QR, fits a ridge reconstruction of all discarded
coordinates from those anchors, applies that reconstruction to incoming
operators, and keeps exact coordinate selection on residual outputs. Because a
tied embedding/head cannot simultaneously implement different input and output
maps, the proof target uses a standard untied head with `hidden=320`,
`intermediate=1280`, and 36 layers: `156,254,016` projected parameters, or
`25.74x` compression. This is SVD/QR/ridge materialization only, with no
optimizer steps, adapters, or recovery training.

`hf-functional-fold` is the sequential compensation companion to this basis.
It fits attention-output and MLP-output operators with closed-form ridge solves,
recomputes activations in sequence, and records `zero_gradient=true` and
`optimizer_steps=0`. The detached runner
`experiments/launch_hf_functional_fold_run.py` snapshots the operation, then
runs the exact `1000/1000/250` suite, seven manual generations, and cached-parent
retention comparison. This is classified as fold materialization rather than
recovery because it performs no gradient or optimizer update; behavioral
promotion criteria remain unchanged.

## Highest-Priority Paths

No 2025-2026 primary source found so far proves arbitrary "large parent into a
totally random blank child at 25x+ with tiny recovery" as a solved recipe. The
practical path remains staged: tokenizer transplantation, activation transport,
merge/fold width and depth, local reconstruction compensation, then short
behavior-gated recovery. Each stage needs its own ablation because local PPL or
activation reconstruction gains have repeatedly failed to imply math/code
behavior recovery in the strict blank-target runs.

| Path | Current Source Trail | Neural Forge Hook |
|---|---|---|
| On-policy representation distillation | OPRD, arXiv `2606.06021`; Improving Language Model Distillation through Hidden State Matching, ICLR 2025 | Extend `hf-xfamily-block-recover` so child rollouts and donor-generated completions are decoded, fed through both models, and matched through donor/child hidden projections at selected layers. Existing `--on-policy-hidden-kd-weight` and `--teacher-generate-hidden-kd-weight` are the seeds; the next version should use stronger rollout scheduling and layer/interface projectors. |
| Cross-tokenizer behavior transfer | Breaking the Tokenizer Barrier, arXiv `2606.09456`; X-Token, arXiv `2605.21699` | Use chunk/span alignment and sparse tokenizer projections for behavior-level recovery. `--teacher-generate-lm-weight` is the current lightweight primitive: donor completions become child-token training text while the folded block is active. |
| Byte-level cross-tokenizer interface | Cross-Tokenizer LLM Distillation through a Byte-Level Interface, arXiv `2604.07466`; Cross-Tokenizer Likelihood Scoring, arXiv `2512.14954` | `--byte-kd-weight` is the logit-space probe. `--byte-hidden-kd-weight` now routes donor first-byte targets directly into active block hidden states through a fixed child byte probe, avoiding tied-head mutation. The stronger version should fit/task-select byte probes instead of only averaging output rows. |
| Low-rank clone initialization | Low-Rank Clone, arXiv `2505.12781` | Add a teacher-to-student low-rank projection initializer for attention/MLP linear weights, train only projection factors/RMSNorm on activation clone + logit KL + CE, then materialize a normal child checkpoint. This is a larger follow-up to `pre-linear`: it constrains activation maps so they are not full-rank behavior-damaging edits. |
| Activation-space folding | FLAT-LLM, arXiv `2505.23966` | Collect calibration activations, PCA-fold attention value/output heads, use activation-correlation channel selection for MLPs, and allocate ranks by decoder input/output dissimilarity. This is the next training-free folding path if full-rank residual maps keep hurting generation. |
| Activation-space optimal transport folding | DOTResize, arXiv `2507.04517`; Transport and Merge, arXiv `2602.05495` | Replace simple rank/depth grouping with entropic transport over activation signatures for MLP channels, attention heads, and residual dimensions before any pruning. `--activation-feature-mode covariance` is now implemented as an opt-in co-activation signature for this path; it still needs broad-gate validation before promotion. |
| Sequential activation compensation | GRAIL, arXiv `2602.23795`; Chain of Merges, arXiv `2508.21421`; RegMean++, arXiv `2508.03121` | After folding each donor layer group into the target block, solve a local ridge/Gram compensation map, absorb it into downstream projections, then recompute activations before moving to the next block. Gate promotion by behavior, not just local reconstruction error. |
| Cross-tokenizer likelihood matching | ALM, arXiv `2503.20083`; Cross-Tokenizer Likelihood Scoring, OpenReview 2026 | Add byte-offset chunk likelihood regression and tokenizer-bias debiasing as a regression test for cross-tokenizer recovery. Useful when projected-vocab and byte losses disagree with greedy behavior. |
| Geometry-preserving hidden recovery | CKA-QAD, arXiv `2606.05682`; Flex-KD / task-tangent geometry, arXiv `2507.10155`; geometry-aware representational alignment, arXiv `2509.25253` | `gram`, `projected-gram`, `geometry`, and `projected-geometry` are now implemented. The first broad runs preserved a recovered child but did not make a donor-projected blank child useful, so the next version needs task-focused feature selection for math/code spans instead of average hidden geometry alone. |
| Submodule-level replacement | SubFit, arXiv `2606.02559`; Merging Feed-Forward Sublayers, arXiv `2501.06126` | Fold and fit attention and MLP residual outputs independently, instead of treating each transformer block as one indivisible unit. |
| Fold-first diagnostics | Cut Less, Fold More, arXiv `2602.18116`; Model Folding, arXiv `2502.10216` | Add projection-error, variance-preservation, and functional-perturbation diagnostics to folded candidates before recovery. This should reject brittle folds that only look good by tensor shape or PPL. |
| Learned tokenizer-neutral byte bridge | Byte-Level Distillation, arXiv `2604.07466`; X-Token, arXiv `2605.21699` | Upgrade fixed first-byte buckets into a temporary learned byte/codepoint head during block recovery, and report digit/code/token coverage for projected-vocab KD. This directly targets the current arithmetic and code-token gap. |
| Staged intermediate anchors | Chain-based Distillation, arXiv `2605.07783`; Reasoning-Aware Compression, arXiv `2509.12464` | Test donor-generated reasoning/code traces and intermediate shape anchors instead of only one-hop `4B -> 135M`. This may provide a recoverable activation manifold before strict blank-target compression. |

## Current Evidence

The 2026-06-28 and 2026-06-29 runs show that output-interface losses alone are not enough for
the strict Qwen3-4B to blank SmolLM2-135M `29.90x` case:

- static projected-vocab KD, byte KD, and optional head training are active and
  behavior-gated;
- behavior-best per-layer selection prevents lower-loss states from replacing
  better-generation states;
- donor-generated completions are now active as block-coupled LM/hidden
  recovery streams, but the first strict 29.90x runs preserved baseline behavior
  rather than recovering arithmetic/code;
- arithmetic and Python outputs still collapse, so the next useful work is
  structural activation compensation and stronger on-policy representation
  recovery.
- per-layer input activation projections in `hf-xfamily-block-recover` prevent
  the worst global-projection collapses, but the best 2026-06-29 run still held
  math at `1/40`;
- response-token LM losses were added to blockwise folding. Tied output-head
  response training lowered PPL while collapsing behavior, and no-head response
  training preserved the conservative profile without improving arithmetic. The
  next attempt should avoid tied-head mutation and use staged hidden/interface
  transfer before response CE.
- untied-head blockwise KD avoids mutating input embeddings, but it still did
  not transfer useful behavior into pure random or donor-projected blank
  targets. It only preserved the existing recovered-child plateau.
- `projected-geometry` and closed-form head calibration showed a sharp PPL
  trap: the donor-projected blank headfit run improved mean PPL from about
  `89,632` to `4,516`, but behavior stayed at `1/200` and generation stayed
  `0/200`. PPL/head reconstruction cannot be used as the promotion signal.
- manual generations confirmed the failure: donor outputs were coherent, the
  best recovered child remained semi-coherent but arithmetically wrong, and the
  blank headfit model repeated answer-like phrases rather than solving tasks.
- chain runs now expose the full block-interface pre-reconstruction and residual
  compensation controls, so GRAIL/Ghosted-layer-style ridge/cosine boundary
  repairs can be swept from the staged launcher instead of being locked behind
  hidden defaults. The next lane should start from the strongest clean text
  anchor and combine low-rank/EoRA residual repair with block-interface
  pre-reconstruction before any ALM/math drill cleanup.

## 2026-07-01 Research Refresh

Fresh 2025-2026 search and sub-agent review point to three next changes with
the best fit to current failures:

1. Sparse cross-tokenizer projection for numbers/code. X-Token (`2605.21699`)
   explicitly calls out multi-digit and uncommon-token failures under
   tokenizer mismatch and uses a sparse projection matrix plus top-ranked token
   mappings. Neural Forge already has projected-vocab and byte KD; the missing
   piece is reporting and preserving sparse mappings for digits, code
   punctuation, and near-equivalent surface forms instead of relying only on
   strict decoded-token matches.
2. Byte-level and on-policy cross-tokenizer recovery. Byte-Level Distillation
   (`2604.07466`) and cross-family on-policy distillation (`2606.09456`) both
   reinforce the direction already started by `--byte-kd-weight`,
   `--byte-hidden-kd-weight`, `--on-policy-hidden-kd-weight`, and
   teacher-generated recovery streams. The next implementation should make byte
   supervision a temporary learned interface and add coverage diagnostics.
3. Fold/transport before prune. DOTResize (`2507.04517`), Transport and Merge
   (`2602.05495`), GRAIL (`2602.23795`), Model Folding (`2502.10216`), and Cut
   Less, Fold More (`2602.18116`) all support the same experimental rule:
   compute activation/geometry-aware transport or folding first, absorb local
   reconstruction maps, and prune only after measuring fold perturbation. This
   supports extending the existing sinkhorn/activation geometry path rather
   than adding more discard-only pruning.

Immediate repo implication:

- `hf-xfamily-chain-anchor-compress` now accepts multi-donor init through
  comma-separated `source_model` plus `--donor-weights`/`--source-weights`.
  Recovery defaults to the first donor for backward compatibility, but
  `--recovery-source-mode sequential-all` runs block and ALM recovery
  sequentially against every comma-separated donor so multi-donor folding is
  no longer init-only.
- Projected-vocab KD now uses sparse surface-piece fallback for fragile
  digit/code/mixed-digit child tokens and reports category coverage in recovery
  and head-calibration artifacts. This is the first concrete X-Token-inspired
  coverage fix; it still needs behavior evidence from the queued blank-target
  lanes.
- Projected-vocab recovery now also maps candidate special tokens through the
  same role/surface rules used by initialization, so EOS/BOS/PAD/additional
  rows are not masked out during cross-tokenizer KD. The post-fix proof queue is
  `qwen4b_blank_smollm_beststack_specialvocab_qrsinkhorn_gpu4_20260707`, waiting
  behind the active pre-fix GPU4 Qwen beststack run.
- Two follow-up queues preserve the best known work instead of replacing it:
  `qwen4b_blank_smollm_beststack_nonfinite_retry_qrsinkhorn_gpu4_20260707`
  lowers block LR, tightens residual alpha, and raises non-finite skip budget
  behind the post-fix proof run; `qwen4b_lfm230m_multidonor_beststack_seqall_qrsinkhorn_gpu0_20260707`
  replaces weak GPT-2 as the second donor with `LiquidAI/LFM2.5-230M` and keeps
  sequential-all recovery.
- The Qwen launcher now honors `ANCHOR_MODEL`, matching the LFM launcher. This
  enables `qwen4b_basisaligned_anchor_beststack_recover_gpu6_20260707`, which
  waits for the basis-aligned init-only artifact and then runs `block,alm`
  recovery from that anchor instead of redoing or discarding the init.
- Recovery now targets next-token behavior more directly: projected-vocab
  teacher-to-student KD uses the first source piece for multi-piece target rows
  by default, and chain ALM exposes `--alm-span-kd-weight` so aligned span-logit
  KD can run alongside span-hidden KD. Beststack queues default this to `0.04`.
- Beststack block recovery now defaults to untied output-head KD through
  `BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD=1` and
  `BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD=1`, because the strict blank-target failures
  showed hidden/interface repair can move while the child logits remain dead.
- Broad eval generation prompts now support behavior-style exact expectations
  and newly built suites include exact math/code generation checks. This was
  added after manual inspection showed old `generation_health` could report
  `1.0` while generated arithmetic was visibly wrong.
- The queued full lane
  `qwen4b_gpt2_multidonor_blank_smollm_eora_full_gpu6_20260702` tests
  Qwen3-4B plus GPT-2 folded into blank SmolLM2-135M, then applies the current
  EoRA/GRAIL-style block-interface recovery with 1000 behavior probes and 1000
  generation prompts.

## 2026-07-02 Research/Experiment Refresh

Additional 2025-2026 source refresh keeps the main implementation path pointed
at merge/fold/recombine before pruning:

1. DOTResize (`2507.04517`) remains the closest primary-source match for
   width reduction by neuron recombination instead of discard-only pruning. Its
   practical implication for Neural Forge is that every compressed target layer
   should preserve an all-source-to-survivor transport map, not just top-k
   survivor rows.
2. Cross-Tokenizer Distillation via Approximate Likelihood Matching
   (`2503.20083`) supports the current ALM/chunk likelihood losses, especially
   when source and target tokenizers differ. The new recovery drill calibration
   file is intentionally prompt/response JSONL so ALM and child-token LM losses
   can supervise exact arithmetic/code spans without reusing held-out eval rows.
3. Cross-Architecture Merging for LLMs (`2602.05495`) directly supports
   activation-space optimal transport for heterogeneous architectures and
   capacities. This reinforces the current `activation-pca`,
   activation-geometry layer grouping, and Sinkhorn transport lanes.
4. EoRA-style post-compression error recovery remains useful as a lightweight
   local correction layer, but the current experiments keep it behavior-gated
   because local reconstruction/PPL gains have already proven insufficient.
5. DERN-style expert/neuron recombination (`2025.findings-emnlp.820`) is useful
   conceptually even outside MoE: recombine transferable neuron segments before
   dropping anything. The analog in Neural Forge is MLP gate/up/down triplet
   handling and module-output KD, not independent row pruning.

New queued experiment:

- `hf-build-recovery-calibration` generated `1800` non-eval drill rows:
  arithmetic, tiny Python functions, tokenizer mismatch, model-surgery, and
  commonsense.
- `qwen4b_blank_best_drill_recover_gpu6_20260702` waits for the three
  Qwen3-4B -> blank SmolLM2-135M EoRA/GRAIL lanes, exact-evals each candidate,
  selects the best by behavior-first exact metrics, applies drill recovery, and
  exact-evals the recovered model.

Immediate next interpretation rule:

- If drill recovery improves exact math/code but hurts commonsense/tokenizer
  categories, split recovery batches by domain and lower domain-destructive LM
  weights.
- If exact behavior does not move while ALM/projected-vocab losses improve,
  the next code change should be stronger activation interface reconstruction,
  not more PPL optimization. The first learned-byte-probe lane is now queued as
  `qwen4b_blank_smollm_cka_weighted_block_weighted_trainable_byteprobe_full_gpu4_20260702`.
- If one RMS/transport initialization wins clearly before drill recovery, use
  it as the new default for the next multi-donor and modality-graft lanes.

## 2026-07-07 Source Refresh And Queue Policy

Latest 2025-2026 source checks still point to merge/fold/recombine first, then
targeted recovery, not discard-first pruning:

- DOTResize (`2507.04517`) argues that width reduction should transport all
  source neurons into the reduced layer instead of keeping only top survivors.
- Transport and Merge (`2602.05495`) reinforces activation-space optimal
  transport as the most relevant cross-architecture merge primitive.
- ALM (`2503.20083`) remains the practical cross-tokenizer recovery objective
  when source and child tokenizers disagree.
- DERN (`2509.10377`, Findings EMNLP 2025) supports segment/neuron
  recombination before dropping expert or neuron structure.
- Linearity-based compression (`2506.21146`) is a useful path for layer/depth
  folding, but should be gated by behavior because local linearity/PPL alone
  has repeatedly failed to predict arithmetic/code retention.

Immediate experiment policy:

- Do not abandon the strongest completed anchors while testing newer
  beststack folds. The restored queue
  `queue_qwen4b_prior_anchor_pair_20260707.sh` runs the old
  `tied29x_dualalm240_last4` and
  `alpha05_ffn1496_folded_xreconstruct_mlp_a05` anchors through the current
  exact-generation/top-negative recovery and retention comparison path.
- The paired prior-anchor lane is an A/B against current beststack structural
  runs, not a replacement for them. Promotion requires parent-vs-child,
  baseline-vs-child, and manual output inspection from the exactgen suite.
- Top-negative answer pressure remains an auxiliary cleanup after structural
  folding. The queue now reuses its exactgen child report for retention compare
  so GPU time is not wasted re-evaluating the same child checkpoint.

Implementation note:

- `hf-rank-evals`, retest matrix summaries, and ALM/block save-best behavior now
  treat math, code, tokenizer, and model-surgery categories as critical
  cross-family proof domains. Broad fluent behavior is still the first signal,
  but ties are broken by critical-domain coverage before generation, choice, or
  PPL. This is intended to prevent the old failure mode where a fold looks good
  on generic text while exact arithmetic/code or tokenizer/model-merge behavior
  has collapsed.
- `experiments/run_tiny_cross_family_proof.py` is the local CPU proof harness.
  It builds a small eval suite, evaluates donor and random blank target,
  folds the donor into the blank target with the current transport/tokenizer
  defaults, now using `domain-diverse-medoid` support so keyword calibration
  domains reserve real donor-axis representatives while still covering
  activation geometry. It compares donor-vs-folded and donor-vs-recovered item
  retention, optionally runs lightweight cross-tokenizer recovery, then writes
  `proof_summary.json` plus donor/blank/folded/recovered eval JSONs, retention
  JSONs with manual sample pairs, and a rank report. Example:
  `python experiments/run_tiny_cross_family_proof.py --output-dir runs/local_tiny_cross_family_proof --device cpu --dtype float32 --recover-steps 0`.
- The next GPU A/B wrappers for this fold-first selector are
  `queue_qwen4b_lfm230m_multidonor_beststack_domain_diverse_20260707.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_domain_diverse_20260707.sh`. They
  keep the current beststack recovery stack and change only
  `TRANSPORT_SUPPORT=domain-diverse-medoid` for comparison against the queued
  quantile-medoid lanes.
- A separate first-principles collision-cancellation lane is now available via
  `queue_qwen4b_lfm230m_multidonor_beststack_countsketch_20260708.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_countsketch_20260708.sh`. These use
  `INIT_ACTIVATION_FOLD_ASSIGNMENT=countsketch` with centroid transport, so
  many-to-one donor-axis collisions get deterministic signed hashes instead of
  unsigned averaging. The mathematical bet is that paired incoming/outgoing
  folds reuse the same signs, making collision cross-terms cancel in
  expectation before recovery.
- A new pure superposition lane is available via
  `queue_qwen4b_lfm230m_multidonor_beststack_superpose_20260708.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_superpose_20260708.sh`. These set
  `INIT_FOLD_MODE=mean`, disable init activation-PCA basis projection, and use
  `INIT_ACTIVATION_FOLD_ASSIGNMENT=signed-superpose`. The mathematical bet is
  that, at high compression ratios, each child axis must carry a deliberately
  mixed superposition of donor extremes with signs chosen to reduce bucket
  moment bias, rather than one similarity cluster.
- A PCA-extrema transport lane is available via
  `queue_qwen4b_lfm230m_multidonor_beststack_pca_extrema_20260708.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_pca_extrema_20260708.sh`. These keep
  QR-Sinkhorn transport but use `TRANSPORT_SUPPORT=pca-extrema-medoid`, choosing
  real donor axes at principal activation extremes before filling the remaining
  support for coverage. The mathematical bet is that rare capability directions
  live on activation-cloud boundaries and are lost by density-only medoids.
- A combined operator-product lane is available via
  `queue_qwen4b_lfm230m_multidonor_beststack_operator_product_20260708.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_operator_product_20260708.sh`. These
  enable `INIT_OPERATOR_PRODUCT_AXIS_WEIGHTS=1`, so MLP intermediate fold mass
  is multiplied by adjacent expansion/contraction operator norms before the
  signed-superpose/PCA-extrema fold. The mathematical bet is that a neuron should
  be merged by its full path contribution, not by activation usage alone.
- A volume-operator lane is available via
  `queue_qwen4b_lfm230m_multidonor_beststack_volume_operator_20260708.sh` and
  `queue_qwen4b_gpt2_multidonor_beststack_volume_operator_20260708.sh`. These
  keep the operator-product stack but set `TRANSPORT_SUPPORT=volume-medoid`,
  selecting real donor axes by greedy residual activation volume instead of
  PCA extremes. The mathematical bet is that a child axis budget should cover
  independent subspace directions, even when duplicated high-mass donor axes
  dominate raw activation weight.
- A donor-only hidden-basis lane is now available through
  `INIT_HIDDEN_BASIS_MODE=donor-token-pca` or `activation-token-pca`, with
  queue wrappers
  `queue_qwen4b_lfm230m_multidonor_beststack_activation_token_volume_operator_20260709.sh`
  and
  `queue_qwen4b_gpt2_multidonor_beststack_activation_token_volume_operator_20260709.sh`.
  Unlike `shared-token-ridge`, this does not fit the donor into random target
  weights. It builds the compression basis from donor embedding/head rows and,
  optionally, donor activation rows. The mathematical bet is that blank-child
  compression should choose the child hidden subspace from parent semantics
  before layer, MLP, attention, token, and head tensors are folded through it.
  The activation-token wrappers force `INIT_PROJECTION_SCALE=source-rms` so
  blank target tensor RMS does not steer the fold scale.
- Pure fold-only comparison wrappers are available for 0-recovery ranking:
  `queue_qwen4b_lfm230m_beststack_volume_operator_initonly_20260709.sh`,
  `queue_qwen4b_lfm230m_beststack_pca_operator_initonly_20260709.sh`,
  `queue_qwen4b_lfm230m_beststack_countsketch_volume_operator_initonly_20260709.sh`,
  and
  `queue_qwen4b_lfm230m_beststack_activation_token_volume_operator_initonly_20260709.sh`.
  They set `STAGES=init`, keeping the benchmark signal on folding/merging
  rather than recovery.

## First-Principles Experiment Ideas

These are intentionally not source-led; they are mathematical mechanisms to
test against the current failure mode.

1. Signed collision cancellation. Compress donor axes with CountSketch-style
   target buckets and signs, reusing the exact assignment for paired incoming
   and outgoing tensors. This targets the destructive cross terms created when
   unrelated donor paths are averaged into one child axis.
2. Domain-diverse support. Allocate reduced axes across calibration domains,
   then choose diverse real donor medoids inside each domain. This targets
   broad-text dominance where math/code/tokenizer/model-surgery signals lose
   representative axes.
3. Operator-product folding. Score and merge neurons by their full path
   contribution, such as `down[:, i] * up[i, :]` or attention value/output
   products, instead of clustering incoming or outgoing weights separately.
   This should preserve functions better than preserving individual tensors.
4. Volume/leverage support. Select survivor axes by residual activation volume
   under a weighted Gram-Schmidt score, not by density medoids or extrema alone.
   This targets the failure case where many high-mass donor neurons are
   collinear and rare independent capability directions get folded away.
5. Random-feature teacher emulator. Treat a blank child block as a fixed random
   feature map and solve layer-local ridge maps to reproduce donor residual
   deltas, then materialize those maps into normal weights. This abandons
   weight similarity and asks only for functional operator matching.
5. Error-feedback folding. Fold one module at a time, measure residual function
   error on calibration states, and explicitly add the error into the next
   available fold rather than minimizing each layer independently. This is a
   control-theory view: conserve the end-to-end residual, not each component.
6. Superposition buckets. Stop trying to make one child axis mean "one
   concept". Seed each child axis with a diverse real donor axis, then add
   opposite activation extremes with signs chosen to minimize each bucket's
   feature moment. This tries to turn destructive collisions into controlled
   superposition.
7. Principal-extreme anchors. Select support axes from both ends of the top
   activation principal directions before medoid fill-in. This protects
   boundary/rareness signals that weighted centroids can wash out.
8. Operator-product axis mass. For MLP intermediate axes, multiply activation
   mass by adjacent input and downstream projection norms. This approximates
   the product contribution of a path and should reduce cases where a frequently
   active but weakly connected channel displaces a less frequent high-impact
   channel.
