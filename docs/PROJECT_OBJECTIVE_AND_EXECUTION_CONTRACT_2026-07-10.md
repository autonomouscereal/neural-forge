# Neural Forge Project Objective And Execution Contract

## Product objective

Neural Forge is an arbitrary model-shape compiler. It accepts one or more
trained parents, an explicit child architecture, and an exact target parameter
or deployable-byte budget. It produces a standalone child checkpoint whose
learned function comes from the parents rather than from scratch training.

Parents may differ in family, architecture, tokenizer, and native modality. The
default proof child begins from seeded random weights; a pretrained child is an
optional input, never a prerequisite. The first minimum scale proof is at least
25x compression into a blank child. The longer-term objective includes ratios
in the hundreds or thousands, multi-parent A+B->C compilation, native modality
transfer, and the inverse expansion operation.

The economic objective is to avoid at least 90-95% of ordinary scratch-training
work. A mechanically valid checkpoint is not enough. Folding must preserve a
coherent, useful baseline before cleanup training is allowed to claim success.

## Transform contract

1. Direction is parent(s) into the already target-sized child.
2. Preserve, merge, compensate, project, compose, interleave, or share measured
   paths before destructive pruning.
3. Operate on coupled functional circuits rather than independent tensors:
   Q/K/V/O and complete GQA groups; gate/up/post-SwiGLU/down paths; ordered
   residual compositions; norms; embeddings and token rows; output heads; and
   native modality I/O.
4. Permit nonlocal mappings across rows, columns, heads, layers, and modalities.
   Important paths may remain exact while low-damage paths merge unevenly.
5. Recompute parent-relative activation, rollout, and output evidence after each
   accepted transaction. Rejected transactions restore the exact prior state.
6. Multi-parent compilation fuses parent evidence simultaneously and records
   attributable contribution from every parent. Applying donors sequentially
   and overwriting earlier capability is not the product contract.
7. The published child has no runtime dependency on a parent, training adapter,
   frozen modality tower, projector sidecar, or external neural decoder.

## Budget truth

Every report separates:

- logical and dense-equivalent parameters;
- unique learned parameters;
- actual serialized learned-weight bytes;
- assignment, scale, index, tokenizer, metadata, and auxiliary bytes;
- FLOPs/token, KV or recurrent state, peak load bytes, and runtime memory.

The deployable artifact set is authoritative. Source-sized hidden bases or index
buffers cannot be omitted from the claimed compression ratio.

The current Qwen codebook checkpoint planner is deliberately narrower: it
accounts exactly for its canonical four-file checkpoint (`model.safetensors`,
config, tokenizer-contract hashes, and manifest). Its surgery target is a
maximum checkpoint-artifact budget, not exact equality and not a standalone
deployment-size claim. Actual tokenizer files and the Neural Forge runtime must
be packaged and counted before a frontier can satisfy this product-level
deployable-byte contract.

## Promotion ladder

### Fold-only gate

- zero optimizer steps, backward calls, or gradient recovery;
- exact requested child budget and parent-free save/reload;
- cached identical parent/child evaluation rows;
- at least 1,000 behavior prompts, 1,000 exact-generation prompts, 250
  forced-choice probes, PPL/NLL, and paired manual inspection;
- coherent free-running output with meaningful parent-pass retention across
  math, executable code, commonsense/reasoning, tokenizer/model-surgery, and
  every transferred modality.

PPL, hidden cosine, endpoint RMSE, local activation error, or teacher-forced
top-1 agreement are diagnostics. None can promote a checkpoint alone. The
desired fold threshold is approximately 90% retained useful parent behavior;
perfect retention is not assumed.

### Recovery gate

Recovery begins only after the fold-only child is coherent. Fold-only and
recovered results remain separate. The recovered child is compared with an
identical random child by training-to-quality curves to measure the claimed
90-95% training reduction.

### Generality gate

Completion requires all of the following, not merely one Qwen artifact:

- a blank-child result at 25x or greater;
- a different-family and different-tokenizer result;
- a multi-parent A+B->C result;
- a native-modality result without a permanent external neural sidecar;
- more than one child budget or shape.

## Preserved incumbents

Two checkpoints are immutable comparison anchors:

1. Product baseline: Qwen3-4B into blank SmolLM-shaped 134,515,008-parameter
   child at approximately 29.9x. The best recovered checkpoint is
   `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_mathboost_answer_cleanopen_800_gpu4`.
   It reached 200/200 generation, 161/200 behavior, and 40/64 choice. Code,
   tokenizer, commonsense, and model-surgery were 40/40; math was 1/40. It is
   useful target-size evidence, not a fold-only success.
2. Fold-only structural baseline: Qwen3-4B attention `p35`, MLP `p36` at
   `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child`.
   It removes 26,897,920 parameters while retaining all parent-passing behavior
   and choice items and 372/388 parent-passing strict generations. It is a safe
   boundary component, not the target-size solution.
3. Uneven MLP Pareto steps: one-path pair folds in layers 35, 34, and 33 each
   saved `7,680` parameters, retained behavior `400/400`, choice `250/250`, and
   `383/384` bound-parent strict generations. Their net generation pass-count
   changes were 0, +1, and +2 respectively. They were refused only by the old
   zero-lost-ID publication rule and are evidence for refreshed, approximately
   90%-retention Pareto allocation; they are not yet proven safe in composition.

None of these anchors may be replaced merely because a new lane has better
local loss, PPL, or tensor reconstruction.

## Settled negative results

Do not rerun unchanged scale-only variants of:

- one-shot PCA, dual-coordinate, mean, medoid, activation-subset, dense OT, or
  sparse-circuit 25-30x folding;
- uniform depth codebooks below the sharp 35/35 survival boundary;
- full-topology rank-one or low-rank depth sharing as a standalone solution;
- child-action-only rollout selection;
- shared-token ridge maps fitted against random child coordinates;
- head-only, answer-ranking, arithmetic-drill, or stronger-PPL recovery on an
  incoherent fold;
- tiny Chimera routing, artifact-exists, or shared-MSE canaries already covered
  by its semantic, artifact, and interference reports.

These experiments established that Q/K rotations and pre-softmax averaging,
gate/up averaging before SwiGLU, and averaging ordered residual blocks introduce
noncommuting errors that local regressions do not expose.

## Active execution path

1. Begin from the executable 36/36 identity and retain the attention-p35 and
   recovered-134M checkpoints as separate frontiers.
2. Use the existing sensitivity collector and adaptive surgery graph to score
   uneven neuron and attention capacity. Pin exact high-damage anchors.
3. Let pair, sparse multi-anchor, complete-attention, and ordered-depth proposals
   compete under the same disjoint parent-relative gate. Retain a bounded Pareto
   frontier instead of demanding one brittle all-or-nothing endpoint.
4. Refresh evidence after accepted edits and rollback rejected edits exactly.
5. Restore progressive blank-child compilation at approximately
   `4B -> 1.4B -> 600M -> 134M`; no stage may be promoted by local metrics alone.
6. Generalize the accepted operator path through canonical byte boundaries and
   explicit layouts for cross-family and simultaneous multi-parent parents.
7. Add the validated NEO raw-RGB parent and Neural Forge raw-token codec only
   after text-path promotion. Chimera's media packet and OmniCoder's provider
   registry are optional parent-trace and artifact interfaces, not replacement
   folding engines.

## Current truth

The project has already built a useful 29.9x target-size child with recovery and
identified a high-retention zero-recovery boundary. It has not yet produced a
coherent 25x zero-recovery child or proved arbitrary cross-family/multimodal
compilation. The unified Qwen surgery runner also remains an adaptive,
four-file-checkpoint frontier rather than a standalone deployment certificate.
Reports and user-facing claims must keep those distinctions explicit.
