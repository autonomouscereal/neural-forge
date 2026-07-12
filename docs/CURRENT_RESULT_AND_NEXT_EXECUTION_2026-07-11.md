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
- Keep exactly one immutable canonical blank and the shared compiler cache;
  rollback remains an integrity check and does not discard the selected factor
  pack. Retain at most three behavior-ranked candidates per comparable
  evaluation contract. A rejected-but-improved pack is labeled
  `research-factor-only-candidate`, never a standalone compaction. Only a child
  that passed bake, output gates, save, and fresh reload is labeled
  `accepted-standalone`.
- The post-run retention frontier is implemented by
  `src/neural_forge/hf_exact_candidate_frontier.py` and
  `experiments/retain_qwen_exact_target_candidates.py`. It archives
  content-addressed factor packs or certified children plus hashed report and
  preregistration evidence. Conservative v1 only emits a durable
  `pending_prune` plan: actual recursive deletion is disabled and
  `--apply-prune` fails closed. Construction and inspection are zero-write;
  registration requires caller-provided report/preregistration hashes and
  canonical source authentication before creating the frontier. Source runs,
  the blank, and the compiler cache are protected and never deleted.
- Reuse the existing remote checkout. Do not create another source clone for
  each experiment.

## Preserved frontiers

- Exact small-child frontier: `134,515,008` parameters.
- Fold-only Qwen frontier: p35, `3,995,570,176` parameters.
- Raw Qwen3-4B source: `4,022,468,096` parameters.

## Current intelligence and coherence boundary

The strongest aggressive exact-stage child is not coherent yet. At its
selected lexical-attention state it matches `31/192` fresh candidate first
actions (`16.15%`) and `27/192` candidate LCP tokens, with `0/24` exact
eight-token rollouts and `0/24` prompts reaching `LCP >= 4`. This is measurable
parent-derived lexical/phrase reflex, not sustained reasoning or useful chat.

Two different frontiers must not be conflated with that result:

- the recovery-assisted `134,515,008`-parameter child is selectively coherent
  on its narrow project suite: generation `200/200`, behavior `161/200`, choice
  `40/64`, and code/tokenizer/commonsense/model-surgery `40/40` each, but math
  only `1/40`. It can emit an executable two-argument Python add function and
  sensible short pruning/tokenizer answers, but observed arithmetic includes
  `6+4 -> 4`, `41*30 -> 1250`, and `35+27 -> 85`, and broader manual outputs
  include off-topic continuations. Broader retests reached behavior/generation/
  choice `803/1000`, `861/1000`, `150/250` for this checkpoint; another
  recovered 134M lane reached `804/1000`, `926/1000`, `154/250`. Recovery
  trained all child parameters, so usefulness does not prove donor-transfer
  causality or reduced training requirement;
- the clean fold-only p35 attention child is the strongest coherent result. It
  retains useful raw-Qwen3-4B-base behavior including correct `41+30 -> 71`, an
  executable Python add function, and coherent merge-risk/pruning/tokenizer
  answers. Its broad totals are behavior `400/1000`, generation health
  `633/1000`, and choice `250/250`; it retains `372/388` strict generations
  that the parent passed. The often-cited `400/400` behavior number is a
  parent-pass retention denominator, not broad accuracy. Some prompts still
  produce zero loops, prompt echoes, or unrelated answers. It removes only
  `26,897,920` parameters, about `0.67%` of the source, so it is genuine but
  mild zero-recovery compaction.

The missing intersection is still: aggressive reduction, coherent free-running
behavior, and attributable parent-derived function in one standalone artifact.
The output-Jacobian-guided tranche below directly targets that boundary.

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

The target bridge now installs one atomic temporary transaction containing
target-sized Q/K/V/O plus role-complete Gate/Up/Down residuals. It freezes the
native child, exposes only those factors to an output/free-rollout optimizer,
requires external held-out actions plus free rollouts, and bakes all seven
roles into plain `nn.Linear` projections. Rejection and partial-publication
failure restore the exact pre-transaction child. Publication restores the
original state keys and parameter count with no adapter, optimizer, parent, or
source-width sidecar. This proves publication structure, not useful behavior.

The first exact FP16 seed-0 stage target has now been materialized once in
checkpoint staging. Its report proves:

- vocabulary capacity `151,936`, tokenizer length `151,669`, reserved rows
  `267`;
- h1536/i5120/L30/Q16/KV4/d128, untied embeddings/head;
- exact unique parameters `1,410,567,168`;
- source-to-child ratio `2.851667x`;
- `model.safetensors` SHA-256
  `4147674b29180d75e8f5bb518b5fc69349d475bb5c8bd49a8f5e7e0ecf364a45`.

It is still blank and staging-only. Fresh-reload certification and a behavior
transaction are required before it is a result.

### First exact-stage transaction result

The authenticated FP32-factor/eager transaction completed all 24 optimizer
steps against the exact stage-1 target. It was correctly rejected and wrote no
standalone checkpoint.

- run:
  `/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_fp32factors_eager_20260712`;
- report SHA-256:
  `ff0bb2c78e29301b53a129a39753a81949b48495e19f3779c061943cf6b2bc58`;
- structural count: `4,022,468,096 -> 1,410,567,168`, or `2.851667x`
  and `64.9328%` fewer parameters;
- selected checkpoint: step 16;
- candidate first-action agreement: `5/96 -> 12/96` (`5.2083% -> 12.5%`);
- parent-action mean rank: `18,630 -> 2,923.7`;
- parent-action mean log probability: `-9.369 -> -6.5735`;
- exact candidate rollouts: `0/12 -> 0/12`;
- repetition excess: `0.3646 -> 0.7708`;
- adjacent-repeat excess: `0.3452 -> 0.9762`;
- all 14 QKVO/Gate/Up/Down factor tensors received finite nonzero gradients;
- exact rollback passed; no model checkpoint was written.

This proves real output-behavior acquisition inside the 64.93%-smaller
structure, but not a useful standalone compaction. The blocker is now the
autoregressive rollout frontier and repetitive attractor, not numerical weight
matching, random-target spread, missing gradients, or the child architecture.

The selected factor-only continuation is preserved at
`rejected_factor_pack.pt` (file SHA-256
`6958c061048e8ff7344148a92e4de2802b5e348fe3679b5982dcb6a0cb8900e1`,
factor-state SHA-256
`ba23bd7343c52cefb96b3219f5d3aecfb0015dd62c78e0a929b14c4184b474ae`).
It contains no parent, native child, optimizer state, activation, or hidden
state.

### Residual-attention continuation result

The preregistered continuation restored that exact selected state, loaded the
same compiled child from the authenticated shared cache, and completed all 32
scheduled steps on the real Qwen parent and exact-size child. It was rejected
by the fit gate, rolled the child back exactly, kept the sealed split locked,
and wrote no model checkpoint.

- run:
  `/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step16_residual_attention_continuation_20260712`;
- report SHA-256:
  `cc261ee91c20235e19633ce8627b634dd80af9e8dd4058fd584dafbca834d9c4`;
- selected checkpoint: continuation step 24 from inherited step 0;
- candidate first-action agreement: `24/192 -> 32/192`
  (`12.5% -> 16.6667%`), a `33.3%` relative gain;
- candidate longest-common-prefix tokens: `24/192 -> 31/192`
  (`12.5% -> 16.1458%`), a `29.2%` relative gain;
- parent-action mean log probability: `-6.7136 -> -6.3533`;
- repetition excess: `0.7760 -> 0.515625`, a `33.6%` reduction;
- adjacent-repeat excess: `0.9762 -> 0.64881`, a `33.5%` reduction;
- exact candidate rollouts remained `0/24`, and LCP-at-least-four remained
  `0/24`;
- parent-action mean rank regressed `2,908.9 -> 4,744.7`, and top-5 agreement
  moved `0.2917 -> 0.2813`;
- all 14 QKVO/Gate/Up/Down factor tensors again received finite nonzero
  gradients; exact rollback passed.

The selected continuation evidence is preserved as
`rejected_factor_pack.pt` (file SHA-256
`ccda13cf9462228b729c5b8f54be73206a89a8769de35d4c57e63f34406c4612`,
factor-state SHA-256
`dff1e77fa0d769f0fca2932700b127b1367dd7060efd410876393bdbca69fc0d`).
This is meaningful residual-attention progress, but not a promoted standalone
child. The simultaneous action/LCP/repetition gains and rank/top-5 regression
support a narrower next test: retain the learned attention state and add a
bakeable embedding/LM-head lexical residual to repair vocabulary ranking.

### Lexical-attention continuation result

The corrected continuation restored the exact selected 14-factor state above,
required a hit from the same authenticated compiled-child cache, and added four
rank-32 embedding/LM-head factors as an exact-zero functional extension. It
completed all 48 optimizer steps. The fit gate rejected it, so the sealed split
remained locked, no bake or checkpoint occurred, and the exact compiled child
was restored.

- run:
  `/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_lexical_attention_continuation_hotpath_v2_20260712`;
- report SHA-256:
  `1a73192a103e3faffd529063a1bd1dbd87b847afff5ef1bc7792867168ec07b8`;
- selected checkpoint: step 24; steps 24 and 32 were on the Pareto frontier;
- fresh-candidate first-action/token agreement: `27/192 -> 31/192`
  (`14.0625% -> 16.1458%`);
- fresh-candidate LCP tokens: `26/192 -> 27/192`
  (`13.5417% -> 14.0625%`);
- parent-action mean log probability: `-6.4030 -> -6.2254`;
- parent-action mean rank: `4,664.9 -> 3,910.9`;
- parent-action top-5: `23.4375% -> 28.6458%`;
- repetition excess: `0.5052 -> 0.3750`;
- adjacent-repeat excess: `0.6667 -> 0.5298`;
- exact candidate rollouts and LCP-at-least-four remained `0/24`;
- selected fit behavior was `162/960` actions, `159/960` token matches,
  `130/960` LCP tokens, and zero exact rollouts across 120 prompts;
- all 18 QKVO/Gate/Up/Down/embedding/LM-head factors received finite nonzero
  gradients across 48 backward observations;
- the authenticated cache disposition was `hit`, exact rollback passed to
  state `39c57212...a606e`, and sealed evaluation was not unlocked.

Step 48 reached `36/192` actions and `38/192` token matches, but LCP and
repetition regressed, so rollout-first selection correctly retained step 24.
The lexical boundary repaired vocabulary ranking and reduced repetition on a
fresh split, but it did not unlock multi-token autoregressive coherence.

The selected 18-factor continuation is preserved at `rejected_factor_pack.pt`
(file SHA-256
`37869a624efe7fc68feeff55fca1470cf83464c446080be5fbc6aef3a0eff545`,
factor-state SHA-256
`bd3f2b305e160b1fe1aecb1d57e2a736b831b19cf1158aaf705d9023f42ed319`).
It contains no parent, native child, optimizer state, activation, or hidden
state. The live transaction used `21,741,568` temporary FP32 factor parameters;
publication would still contain exactly `1,410,567,168` native parameters.

## Completed lexical execution contract

1. Require an authenticated hit from the already-published exact compiled-child
   cache; do not reconstruct or recompile the structural base.
2. Restore the selected continuation factor state
   `dff1e77f...c69fc0d`; do not restart QKVO/Gate/Up/Down from zero and do not
   reconstruct optimizer state.
3. Add rank-32 bakeable lexical residual factors to the untied token embedding
   and LM head inside the same atomic transaction. Start with an LM-head-only
   ranking ablation, then rejoin QKVO at lower learning rates; unfreeze the
   embedding and Gate/Up/Down only in the later phase.
4. Train on the prior 96 fit prompts plus the now-exposed 24 candidate prompts.
   Evaluate promotion on 24 entirely fresh candidate prompts. Reuse the exact
   still-locked 32-prompt sealed split. Historical candidates are diagnostics,
   never new promotion evidence.
5. Select on free-running action/LCP/repetition behavior, not weight or hidden
   similarity. On rejection, retain the strongest authenticated factor/report
   evidence and restore exactly. On acceptance, bake all residuals, save once,
   reload fresh, certify the exact native state, and run broad parent-pass
   retention with no sidecar.
6. Keep the p35 broad consumer as certification evidence for its already
   selected factor pack; do not make it another prerequisite method-development
   loop. Do not run the eight-site p34 or width-64 L1 branches ahead of the
   exact-target transaction.
7. If the first stage passes, repeat the promoted transaction at `600,628,224`
   and then `134,515,008`, with a new sealed gate at every stage.

The lexical continuation contains 18 temporary factor tensors: the inherited
14 QKVO/Gate/Up/Down tensors plus four rank-32 embedding/LM-head tensors. The
lexical portion is `9,822,208` FP32 parameters and the complete temporary
transaction is `21,741,568` parameters. These are training-time factors only;
the baked child retains exactly `1,410,567,168` parameters and no adapter
state. The preregistered 48-step schedule is head-only through step 8,
head+QKVO through step 24, then joint lexical/QKVO/Gate/Up/Down training, with
a final half-rate consolidation after step 40. Acceptance requires the native
baked child—not merely the live FP32 residual wrappers—to pass the same
output/free-rollout behavior gates before its one allowed save and fresh
reload certification.

The first launch of that exact contract was intentionally stopped before the
first optimizer step or checkpoint. Profiling exposed a pure execution bug:
every token forward revalidated the complete factor registries independently
at every Q/K/V/O, Gate/Up/Down, embedding, and LM-head hook. That performed
about `1,306,198,016` redundant factor-element finiteness checks per child
forward. The source factor pack, compiled-child cache, behavior splits, and
methodology were not changed. Hot-path validation is now cached against factor
identity, storage, topology, and `Parameter._version`; optimizer updates and
restores force a new full scan, while unchanged factors validate once. The
corrected continuation used a new source-pinned preregistration and retained
the same 120/24/32 behavior contract.

## Selected-factor bottleneck audit

The authenticated selected v2 pack contains 18 finite FP32 tensors and exactly
`21,741,568` values. Every tensor hash and the declared factor-state hash
`bd3f2b30...ed319` reproduce. Compared with its inherited state:

- Q/K/V/O left factors moved `2.66%` to `4.05%` in global L2 norm, while their
  right/input factors moved only `0.165%` to `0.299%`;
- all six Gate/Up/Down factors are byte-identical to the inherited state;
- the lexical embedding left factor remains exact zero and its right factor is
  byte-identical to initialization;
- the LM-head left factor learned, but its right factor moved only `2.98%` and
  retained cosine `0.999557` with its random initialization;
- every Q/K/V/O and Gate/Up/Down product has numerical rank `8/8`;
- the rank-32 LM-head product is severely concentrated: entropy-effective rank
  `4.5633`, stable rank `1.2637`, and smallest/largest singular ratio
  `0.0004038`.

This makes the next hypothesis narrower. The corrections are not failing to
receive gradients, and simply planting more random anchors is not justified.
They fill the available product rank while remaining nearly locked to their
random input subspaces. Later checkpoints improved local log probability and
rank but damaged LCP/repetition, which is consistent with locally useful
directions outside the rollout-safe manifold. A direct output-Jacobian audit is
required to test that inference before another training run.

### Authenticated output-Jacobian audit result

That preregistered diagnostic has now run against the real parent, exact
`1,410,567,168`-parameter child, authenticated compiled-child cache, and
selected v2 factor state. It passed decisively without an optimizer step,
factor update, bake, or checkpoint publication.

- preregistration:
  `/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712`;
- preregistration SHA-256:
  `ee3fc282b25f28dc9c8452df9196f16b63cd7e19b1868e4c9e2e2dd4fed6718e`;
- run:
  `/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712`;
- run-report SHA-256:
  `5f95a8db4b1e62574279a87e1497902cdd8221f004cd4826323985365f35d77c`;
- terminal status/reason:
  `diagnostic-passed` / `output-jacobian-audit-passed`;
- aggregate normal-energy fraction: `0.7091664`, versus the preregistered
  `0.20` gate;
- aggregate guided/random capture ratio: `92.85442`, versus the
  preregistered `1.5` gate;
- guided-basis pack SHA-256:
  `68d4ccf6b9a1cbac711cec15d8aedd6848b2fa087a3e62a20cbeb54c571916ad`;
- guided-basis manifest SHA-256:
  `b3bd1619aa226892d893dc3981097484201e3aa7231db7ab6c6f9cc44b3e6e8e`;
- four role tensors, `4,915,200` FP32 values total: Q/K/V each
  `[120, 8, 1536]`, O `[120, 8, 512]`.

Every attention role independently passed: Q normal fraction `0.90957` and
guided/random ratio `200.13`; K `0.80522` and `204.62`; V `0.62588` and
`184.24`; O `0.82414` and `62.36`. All four roles are therefore authorized for
the guided tranche. The randomized energy estimator can exceed one, so its
`1.0405` aggregate guided-capture estimate is not described as a literal
deterministic percentage.

The execution replayed 263 blocks per pass and 63,120 wrapper callbacks,
recomputed parent logits under `no_grad`, retained token paths rather than
cached full-vocabulary logits, authenticated every D/X element through 16
probabilistic lanes, and stayed below the 256 MiB auxiliary-memory cap. The
probabilistic authenticator is explicitly not collision-free; its value is
reproducible execution evidence, not a cryptographic tensor digest. Before and
after native, factor, inherited-source, replay, and cache identities matched;
hooks and gradients were removed; exact rollback returned to
`39c57212e804280b67e9706fc2935a038c41c0f8491ab950af6b45f4a21a606e`.

This validates the narrow hypothesis: most behavior-useful output-gradient
energy at the selected state lies outside the inherited random Q/K/V/O row
subspaces, and the output-guided directions are dramatically more informative
than equal-rank random controls. It does not itself prove a coherent child;
that requires the guided behavioral continuation below.

### Guided residual-attention continuation result

The authenticated 64-step guided Q/K/V/O continuation completed on the exact
`1,410,567,168`-parameter child. An initial execution failed before step 1
because JSON publication alphabetized the `per_role` mapping while the adapter
incorrectly treated insertion order as semantic. The audit itself authorized
all four roles. Commit `8c66e18` changed that gate to require the exact semantic
role set independent of JSON key order, with missing/extra/substituted-role
regressions, and the corrected run passed installation, all optimizer stages,
all checkpoints, and exact rejection rollback.

Corrected run evidence:

- run:
  `/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_schemafix_v2_20260712`;
- run-report SHA-256:
  `4eb8804ba257f27926fc8f0891145061f43eeb01710c4e4a94ac0ee3df4bb1d1`;
- preregistration SHA-256:
  `ca2dbce54592c99c669f3c2044cf0af2f36268fe35d50fbf151da8fdfebcc57f`;
- terminal status/reason: `rejected` / `fit-gate-rejected`;
- all `64` optimizer steps completed and every authorized factor received
  finite nonzero output-loss gradients;
- exact rollback restored
  `39c57212e804280b67e9706fc2935a038c41c0f8491ab950af6b45f4a21a606e`;
- no standalone checkpoint was written and sealed evaluation remained locked;
- the retained selected step-0 v3 pack is `115,492,213` bytes, SHA-256
  `2524d5224a09193f6561d9108576c422988ad267461df4c0e897d509433d32e9`;
- trainer-evidence SHA-256:
  `60610af5f17e110aab0ac5be37655b304ccb193707b9009dbda66d24e697c536`.

The strict behavior-only selector retained step 0 because every trained
checkpoint had `0/24` exact rollouts, `0/24` prompts with `LCP >= 4`, and at
least a small aggregate-LCP regression. That does not mean training was flat:

- step 0: action `29/192`, token `30/192`, LCP tokens `28/192`, mean parent
  action rank `4026.5`, repeat excess `0.41146`, adjacent excess `0.57143`;
- step 12: action `48/192`, token `50/192`, LCP tokens `26/192`;
- step 40: best action `50/192` (`26.04%`, versus `15.10%` at step 0), token
  `49/192`, LCP tokens `26/192`, mean rank `1062.60`;
- step 56: action `46/192`, LCP tokens `24/192`, best degeneration control:
  repeat excess `0.22396`, adjacent excess `0.35119`, longest-run excess `2`;
- step 64: action `47/192`, token `40/192`, LCP tokens `27/192`, repeat excess
  `0.23958`, adjacent excess `0.38095`.

This is real first-action, ranking, and repetition progress from guided
attention, but not trajectory coherence. The aggressive child remains
incoherent. The run also exposed a retention defect relative to the project
policy: steps 40/56/64 were authenticated only by factor-state hashes
(`98b5a29d...c72b4`, `12b9428b...c9`, `1a045927...c3118`) and their tensors
were not persisted because the strict selector restored step 0. The candidate
frontier correctly refused to register the step-0 pack as behavior-acquired;
the source run/report/pack remain preserved in place.

## Next ordered execution

1. Add a bounded checkpoint sink to the existing trainer/runner. At each
   preregistered checkpoint, persist a content-addressed v3 factor pack only if
   it enters the top-three behavior frontier. Keep strict standalone selection
   unchanged. A rejection must retain the strongest research packs, then
   restore exactly; it must not save only the strict selector's step 0.
2. Deterministically replay the authenticated schedule on the same hardware
   only far enough to reproduce and persist the proven step-40/56/64 states.
   Require the exact historical factor-state and behavior-report hashes before
   treating replay output as recovery; abort on any mismatch.
3. Move the exposed 24-prompt candidate into fit (`144 -> 168`), create one new
   prompt/prefix-disjoint 24-prompt candidate, and reuse the exact still-locked
   32-prompt sealed split. Candidate and sealed data cannot select sites,
   bases, ranks, schedules, or learning rates.
4. Add candidate-only execution so a fixed finite arm tournament cannot open
   sealed independently. Preregister all arms, run them against the same new
   candidate, select at most one winner, and allow only that winner to access
   sealed. Keep all existing standalone thresholds and post-bake/fresh-reload
   gates.
5. Run three bounded arms in parallel from preserved evidence:
   - divergence-centered horizon-16 on-policy continuation from recovered step
     40 (best actions/rank);
   - the same continuation from recovered step 64 (near-step0 LCP with much
     better repetition);
   - a Jacobian-energy sparse-site guided tranche from the authenticated v2
     source, selecting the smallest fit-only site set covering 80% of eligible
     normal energy (8 to 60 sites per role).
   Keep GUD and lexical factors frozen so the experiment tests trajectory
   propagation and attention-site interference rather than readout repair.
6. Require selected step greater than zero, at least one candidate rollout with
   `LCP >= 4`, and repetition non-regression before any arm can win. Lexical
   readout repair remains conditional on trajectory acquisition; do not run it
   while all arms remain `LCP >= 4 == 0`.
7. On the single winner, run sealed once. On rejection preserve the top-three
   authenticated research packs and restore exactly. On acceptance bake every
   inherited/new product, save once, reload fresh, certify exactly
   `1,410,567,168` native parameters with no adapter/parent/sidecar, and run
   broad parent-pass retention.
8. After stage 1 passes, repeat the promoted transaction at `600,628,224` and
   then `134,515,008`, with a new sealed gate at every stage.

## Truth boundary

There is now real proof of one-site behavioral compaction: `3,183,488` net
parameters removed with improved disjoint held-out actions and rollouts, plus
an exact tiny replay artifact. There is not yet broad certification, an
all-eight live result, or a standalone `134,515,008` useful child. Do not
describe structural plans or rejected MLP runs as those missing proofs.
