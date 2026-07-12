# Neural Forge current result and next execution

This is the resume ledger for the current compiler frontier. A pivot changes
the next experiment; it does not discard these results or restart old tests.

## Objective

Compile one or more arbitrary parent models into a standalone child at an
exact target parameter budget, across architecture/tokenizer/modality when
needed, while avoiding scratch-scale retraining. Promotion is determined by
held-out outputs and free-running behavior. Weight, activation, cache, or
latent numerical resemblance cannot by itself certify compaction.

The first target remains an exact `134,515,008`-parameter child (at least
`25x`; the concrete Qwen3-4B source ratio is `29.903489x`).

## Filesystem and disk correction

- Missing historical checkpoints were deliberately pruned by the user to
  recover disk space. They were not evidence that saving was broken.
- CIFS directory publication is fixed: try atomic replace, then use an
  incomplete-marker + byte/tree/hash-verified copy fallback on `EACCES` or
  `EXDEV`, followed by scoped cleanup.
- Do not write another full 8+ GB checkpoint for probes. Persist small JSON
  evidence and factor packs until a candidate passes broad promotion.
- Reuse the existing remote checkout. Do not create another source clone for
  each experiment.

## Preserved frontiers

- Exact small-child frontier: `134,515,008` parameters.
- Fold-only Qwen frontier: p35, `3,995,570,176` parameters.
- Raw Qwen3-4B source: `4,022,468,096` parameters.

## Residual-attention findings

### Rejected approaches

- Raw V/O internal regression fit calibration V RMSE (`0.6583 -> 0.01553`)
  while held-out functional risk worsened (`0.8417 -> 7.1533`) and all six
  greedy examples diverged. This is the clearest false-positive from internal
  numerical matching.
- Residual-target V basis improved fit by only about `0.04%`.
- Q/K anchor alternatives were all `10.6%` to `13.8%` worse than the incumbent.
- O-only rank-8 functional residual moved parameters but produced no discrete
  held-out gain.
- Global signed V/O scales improved the small calibration set but regressed
  held-out behavior and were vetoed.

### Promoted method-development result

The missing attention operator was assignment-local residual Q/K/V/O folding:

- Q/K corrections occur before Q/K norm, RoPE, and cache append.
- V correction occurs before cache append.
- O correction occurs on post-softmax physical-group context.
- Training uses output actions/distributions only; there is no weight,
  activation, attention-map, residual, or cache matching objective.
- Initialization is exact-zero output factors with random nonzero input
  factors. This preserves p35 exactly at step zero.

For p35 `L19/G4 -> L21/G6`, rank 8, step 16:

- net parameters removed: `3,183,488`;
- calibration top-1: `38/48 -> 45/48` (`0.791667 -> 0.9375`);
- held-out top-1: `56/64 -> 64/64` (`0.875 -> 1.0`);
- held-out exact rollouts: `5/8 -> 8/8`;
- held-out rollout tokens: `20/32 -> 32/32`;
- all eight Q/K/V/O factor tensors received gradients;
- p35 was restored after the experiment.

Exact selected state:

`f4cad9b277826a6724d76c9727999dc614ae5c54ef6197b9902d8d4c26c20240`

Verified factor-only artifact:

- directory:
  `runs/residual-attention-20260711/p35_L19G4_to_L21G6_qkvo_selected_s16_factor_pack_fp16_p40_gpu3_v3`;
- factors: 8 FP16 tensors, `92,160` values, `184,320` raw tensor bytes,
  `185,184` file bytes;
- factor file SHA-256:
  `5807fc789516ed5548a78fb4657f369a0f1c67c45a0b9e1626aa5813dd0832a8`;
- manifest SHA-256:
  `6c6e979712e67f862e4574121e500e9c823778f6774ce81647752158ccb2886d`;
- reproduction report SHA-256:
  `69d5127eec861dae5deb45c5e14087d09eaa185f59e93a79043f67af91c360d3`;
- full-vocabulary calibration and held-out action logits match byte-for-byte;
- held-out traces match exactly; p35 restores exactly; no checkpoint exists.

Parallel P40 reductions changed continuous calibration aggregates by roughly
`1e-6` despite byte-identical logits. Exact state, full-vocabulary logits,
discrete actions, and rollout traces are identity gates; redundant floating
reductions are diagnostics only.

### Broad gate status

The first broad run spent about two hours producing a valid fresh p35 baseline
before failing a stale hardcoded pass-set check. It never installed/evaluated
the residual candidate.

Fresh P40/Torch-2.4 baseline:

- report SHA-256:
  `403329fea0ec53510ec88449b8512de5331a05665e5229d78ca1ffa8ff29003c`;
- behavior: `400/1000`, pass hash `98fe6b18...63ad`;
- choice: `250/250`, pass hash `317bb1e8...abaf`;
- generation: `544/1000`, pass hash `642bd06b...5afd8`;
- perplexity: `1011.1708944515992`.

The historical `546` generation set came from an RTX-3090/Torch-2.11
in-memory evaluation. Forty outputs differ across those environments, with
seven historical passes lost and five new passes gained. Suite order and the
current scorer are unchanged. The historical hash is drift telemetry, not a
hardware-independent acceptance baseline.

The fresh dynamic, prompt-decontaminated p35-pass subset is:

- behavior `333`, hash `2c528c1e...47ae4`;
- choice `250`, hash `317bb1e8...abaf`;
- generation `386`, hash `212a81f6...da864`;
- canonical suite hash `b718f680...0a689`.

A resume consumer is being hardened to authenticate the preserved baseline,
rescore saved behavior outputs, isolate it from writable output mounts, and
pin evaluator/runtime sources. It must require zero candidate losses against
the fresh dynamic subset. Do not rerun the two-hour baseline.

## Multi-site attention path

An audited ragged all-eight p35->p34 experiment is staged but unrun. The first
schedule is `(4,4,4,8,4,4,4,4)`, preserving rank 8 on the proven anchor and
using rank 4 elsewhere.

- structural parameters removed: `26,214,400`;
- residual factors added: `414,720`;
- role scales added: `1,152`;
- net saving if behavior passes: `25,798,528` parameters;
- resulting size: `3,969,771,648` parameters.

Seven sites remain behaviorally unproven. Do not launch this ahead of the
one-site dynamic broad zero-loss result.

## MLP findings

p36->p35 removes `9,728` MLP prototypes, or `74,711,040` parameters. A
layer-local L1 width-64 residual SwiGLU adds `491,520`, leaving a theoretical
`74,219,520` net saving versus p36 if behavior is recovered.

Three behavior-only runs were rejected:

1. V1 selected step 8 on action top-1/rank. Held-out top-1/rank improved, but
   exact rollouts regressed `2/8 -> 1/8`. This exposed action-only selection as
   a false positive.
2. V2 added Pareto rollout-safe calibration selection and a fresh sealed
   held-out set. Step 8 improved actions but regressed calibration rollout
   tokens/LCP, so step 0 was correctly locked. Fresh held-out matched baseline.
3. V3 added fit-only on-policy/DAgger prefix refresh at steps 8/16/24 with
   persistent Adam moments and another fresh sealed held-out set. Step 8 still
   improved actions while regressing rollout tokens/LCP; later states
   collapsed further. Step 0 was again locked and fresh held-out matched
   baseline.

Therefore random initialization and an inability to train are not the MLP
blockers. All residual tensors learned, but this small L1 aggregate repair did
not preserve sequence behavior. Do not repeat width-64 L1 with more selection
or prefix-refresh variants.

MLP reports are under `runs/mlp-behavior-residual-20260711/`. No MLP
checkpoint, factor pack, bake, or committed candidate exists.

## Exact target-native compiler path

The structural ladder is now explicit:

`4,022,468,096 -> 1,410,567,168 -> 600,628,224 -> 134,515,008`

The final child is SmolLM2/Llama style:

- hidden `576`, intermediate `1,536`, layers `30`;
- query heads `9`, KV heads `3`, head dimension `64`;
- tied vocabulary `49,152`;
- exact total `134,515,008` (`29.903489x`).

The target bridge can install temporary target-sized Q/K/V/O residuals,
require external held-out actions plus free rollouts, then bake into plain
`nn.Linear` projections with unchanged state-key topology, exact count, and no
adapter/parent/source-width sidecar. This proves publication structure, not
useful behavior at the target sizes.

## Next ordered execution

1. Finish audit of the authenticated broad-resume consumer.
2. Run the one-site QKVO candidate against the preserved fresh P40 baseline's
   dynamic decontaminated zero-loss subset.
3. Only if step 2 passes, run the full broad candidate regression and then the
   all-eight ragged attention experiment.
4. Stop iterating on L1 width-64 MLP residuals. Next MLP work must change the
   capacity/location operator (for example output-gradient-selected layers or
   target-reserved multi-layer paths), not merely the selector.
5. Compose promoted attention and MLP operators into the exact
   `1.410B -> 600.628M -> 134.515M` native ladder, with a fresh held-out seal at
   every algorithm-changing run.

## Truth boundary

There is now real proof of one-site behavioral compaction: `3,183,488` net
parameters removed with improved disjoint held-out actions and rollouts, plus
an exact tiny replay artifact. There is not yet broad certification, an
all-eight live result, or a standalone `134,515,008` useful child. Do not
describe structural plans or rejected MLP runs as those missing proofs.
