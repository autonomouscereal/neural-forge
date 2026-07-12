# P35 output-trained functional residual adapter

Date: 2026-07-11

## The objective is behavior, not numerical weight matching

Projected child weights are not expected to numerically match parent weights,
especially after a redirect, width change, tokenizer bridge, or architecture
change. Numerical similarity is therefore not a loss, selection feature, or
promotion gate in this experiment.

The fixed structural proposal removes the `L19/G4` attention prototype and
redirects that physical assignment to the surviving `L21/G6` prototype. The
child then learns one assignment-local correction:

```text
child residual write = shared_O(z_all) + U @ (R @ z_physical_group_4)
```

`z_physical_group_4` is the real post-softmax, context-dependent group value
immediately before the O projection and residual add. This is the residual
attention folding operator: it preserves shared attention storage while giving
the removed assignment a small context-aware residual channel of its own.

The implementation is split between:

- `src/neural_forge/hf_functional_residual_adapter.py`, which owns the sparse
  operator, exact accounting, output losses, disjoint trainer, and rollback;
- `experiments/run_qwen_p35_functional_residual_adapter.py`, which owns the
  exact p35 transaction, prompt provenance, output-target construction,
  calibration lock, post-lock heldout reveal, and final report.

## Fixed first-run design

The exact p35 mode deliberately has one narrow first-run configuration:

```text
redirect                 L19/G4 -> L21/G6
prototype transition     280 -> 279
rank                     8
adapter parameters       24,576
initial U                exact zeros
initial R                deterministic random, seed 0
role-scale initializer   untouched incumbent
incumbent V              -0.0012493133544921875
incumbent O              -0.0005393028259277344
fit updates              32
micro-batch size         2
update order             parent, student, parent, student, ...
update rule              manual Adam
learning rate            0.0015
gradient norm clip       1.0
calibration gates        0, 8, 16, 32
```

No fitted V/O scalar point is used as a warm start. Starting `U=0` makes the
new module an exact zero functional perturbation of the tentative 279-prototype
child, while random `R` gives `U` a non-degenerate gradient direction. Only
`U` and `R` may change.

The eight fit prompts at horizon four produce 32 parent-prefix actions and 32
student-prefix actions. They are packaged as sixteen batch-2 parent micro-
batches and sixteen aligned batch-2 student micro-batches, then interleaved.
Exactly 32 updates therefore consume one complete deterministic fit pass.

## Data and selection boundary

The runner reuses the cryptographically verified, prompt-disjoint fixtures:

```text
fit                    8 prompts
calibration-selection  6 prompts
heldout-evaluation     8 prompts
```

Two prefix families are fixed before training:

1. Parent-greedy prefixes with the parent's next-token top-1 action and output
   logits.
2. Baseline-child-greedy prefixes re-evaluated by the parent, again using the
   parent's top-1 action and output logits as the target.

Fit batches alone receive gradients. Calibration batches receive no gradients
and are evaluated only at steps `0/8/16/32`. A state can replace the incumbent
only when discrete parent-action behavior improves: higher top-1 retention, or
lower parent-action rank as the tie-break. Better log probability or lower KL
alone cannot promote a state. Those continuous values remain diagnostics.

The selected adapter is locked in memory and hashed. Only after the
`winner-locked` event may any heldout model forward occur. Heldout then compares
the zero-adapter tentative child with the locked adapter on parent actions and
independent greedy rollouts. Heldout is winnerless and cannot change the lock.
When the two children generate different student prefixes, their combined
action summary is explicitly labeled an own-rollout-domain comparison rather
than a common-prefix causal delta; the greedy-rollout outcomes remain directly
comparable.

## Exact compaction accounting

For Qwen3-4B, one removed coupled attention prototype has `3,276,800`
parameters. The redirect's role-scale table has `1,152` parameters, and the
rank-8 residual adapter has `24,576` parameters:

```text
net parameters saved versus p35
  = 3,276,800 - 1,152 - 24,576
  = 3,251,072

FP16/BF16 learned payload saved = 6,502,144 bytes
exact tensor payload saved      = 6,502,132 bytes
```

That is proven structural compaction for this one redirect. It is not yet a
claim that the compact child is behaviorally successful. The report keeps
`compaction_certified=false` until the locked adapter demonstrates adequate
heldout and broad-suite behavior, is explicitly committed in a separate
workflow, saved/reloaded, and verified parent-free.

## Rollback and non-claims

The runner verifies that:

- the tentative child retains the exact incumbent Q/K/V/O role scales;
- base codebook tensors, identities, versions, gradients, and parameter flags
  do not change during adapter training;
- the locked adapter can be reconstructed exactly in memory for heldout;
- the adapter is removed and incumbent p35 is restored exactly on success;
- the same p35 restoration occurs when training raises an exception.

The runner exposes no save or commit flag. It writes only a JSON experiment
report. It does not claim a checkpoint, deployable artifact, completed p35 to
p34 transition, or broad functional certification.

## Prepared GPU 3 launch

The local launch script is:

```text
experiments/remote_run_p35_functional_residual_adapter_rank8_h4_s32_fp16_p40_gpu3_20260711.sh
```

It targets GPU UUID `GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8`, mounts the
staged repository, Hugging Face model, and p35 checkpoint read-only, and writes
only the report/log/PID metadata to the experiment output directory. The
authoring task did not upload or launch it.

Interpret the eventual report in this order:

1. Confirm exact `280 -> 279` lineage, rank-8 accounting, and p35 rollback.
2. Confirm the best step is nonzero only if calibration top-1 or rank improved.
3. Inspect heldout top-1/rank and greedy-rollout deltas against the zero-adapter
   child.
4. Treat continuous loss/log-probability improvement without discrete output
   improvement as a negative result, not as evidence of successful compaction.
