# Residual-Attention Folding

Date: 2026-07-11

This is the next zero-gradient folding branch for Neural Forge.  It is not a
restart and it does not replace the preserved codebook, compensated-neuron,
cross-tokenizer, or recovered-child results.

## Why this branch exists

The full-vocabulary head-isolation control improved teacher-forced top-1
agreement from `11.67%` to `65.75%`, top-1-in-top-5 from `30.55%` to `85.67%`,
and KL from `10.41` to `1.20`, while free-running behavior and generation stayed
at zero.  This proves that useful parent information can remain linearly
readable when the folded body is supplied a valid token trajectory.  It does
not prove that the first generated token is correct on 65.75% of prompts, and
it does not by itself isolate attention as the only fault.  It does sharply
localize the missing invariant: the compact model does not preserve the state
transition that produces the next residual state and the next autoregressive
prefix.

Attention and the residual stream are the most direct place to test that
invariant.  Attention routes information from the prefix and KV cache; the
attention output is added to the residual stream; the changed residual stream
then changes every later layer and every later token.  A small first error can
therefore compound even when a teacher-forced output head can read the state.

## What the original Model Folding implementation contributes

The 2025 Model Folding paper and its official LLaMA code do not fold Q, K, V,
and O independently.  They build one head-clustering feature from Q/K/V rows
and the corresponding O input columns, average clustered Q/K/V head rows, sum
the matching O input columns, and reuse a consistent hidden/residual merge map
for O outputs.  That is the correct structural lesson: an attention head is a
coupled path, not four unrelated matrices.

Its released LLaMA path is not the complete operator needed here:

- it clusters weights rather than causal attention distributions on reachable
  decode trajectories;
- its grouped-query-attention branch contains explicit unimplemented paths;
- it does not score the KV-cache transition;
- it does not use on-policy multi-token decode trajectories;
- its published LLM compression is modest structured sparsity, not a 25x
  arbitrary-parent-to-blank-child proof.

Neural Forge already goes beyond that structural baseline in several ways.  It
has atomic GQA Q/K/V/O bundles, activation-coupled and path-conjugate maps,
whole-bundle rollback, residual/output fits, and a fold-only attention
codebook.  However, the current sensitivity collector runs teacher-forced
forwards with `use_cache=False`, records Q/K/V/O energy and downstream
residual/logit damage, and does not record attention probabilities or the
stepwise KV-cache state.  That is the concrete missing experiment, not merely
another switch from independent to coupled QKVO.

## Operator to preserve

For a pre-norm decoder layer, define the attention residual update at layer
`l` and decode step `t` as

```text
u_l,t       = RMSNorm(r_l,t)
q_l,g,t     = QNorm(Wq_l,g u_l,t)
k_l,g,t     = KNorm(Wk_l,g u_l,t)
v_l,g,t     = Wv_l,g u_l,t
A_l,g,t     = softmax(q_l,g,t K_l,g,<=t^T / sqrt(d) + mask)
delta_l,t   = sum_g Wo_l,g (A_l,g,t V_l,g,<=t)
s_l,t       = r_l,t + delta_l,t
```

The fold must preserve the complete map

```text
(r_l,t, K_l,<=t-1, V_l,<=t-1)
    -> (A_l,t, delta_l,t, s_l,t, K_l,<=t, V_l,<=t)
```

on states the parent and child can actually reach.  Matching Q/K/V/O weights,
one-token activations, or `delta_l,t` on teacher-forced rows is not enough.

## Residual-attention fold

For a cluster of parent attention paths assigned to one child path:

1. **Choose Q/K as an addressing anchor, not an average.**  Select a real
   parent Q/K path whose causal attention distributions best represent the
   cluster over reachable prefixes.  Averaging Q/K before softmax creates cross
   terms that did not exist in any parent head.
2. **Fit V/O to the routed residual contribution.**  With Q/K and therefore
   the causal attention matrices fixed, solve bounded ridge systems for V and O
   to reproduce the sum of the clustered parent attention residual deltas.
   Alternate the two closed-form solves a small deterministic number of times;
   no optimizer, backward pass, or gradient is permitted.
3. **Fit the residual branch scale.**  Fit a scalar or bounded diagonal scale
   from the candidate attention delta into the parent delta, then absorb it
   into O.  This addresses the known post-pruning residual-magnitude shift
   without adding a neural sidecar.
4. **Preserve cache evolution.**  K and V errors are measured after RoPE and in
   the exact cache layout used during incremental decoding.  The same candidate
   must agree in full-prefix `use_cache=False` and incremental `use_cache=True`
   execution.
5. **Commit atomically.**  Q, K, V, O, Q/K norms, logical assignments, residual
   compensation, and accounting are one transaction.  A partial attention edit
   is not representable.

For fixed Q/K, the V/O subproblem is an attention-weighted residual regression.
At a high level, the shared child bundle minimizes

```text
sum_(trajectory,site,step)
    ||Delta_parent_cluster - Delta_child_bundle||^2
  + lambda_v ||V - V_anchor||^2
  + lambda_o ||O - O_anchor||^2
```

over every logical site that will share that physical prototype.  The target is
   the parent's *residual contribution*, not its raw V activations.

When a physical codebook prototype is already shared, refitting it changes
every logical `[layer, group]` assigned to that prototype.  The regression set
must therefore contain every affected logical site, including the survivor's
original sites.  Fitting only the removed source site would silently repair one
path by damaging the existing consumers of the survivor.

## Candidate evidence and score

Each candidate is evaluated over a union of:

- parent greedy prefixes;
- current-child greedy prefixes, including prefixes after the first
  divergence;
- held-out natural text, code, math, tokenizer, and behavior prefixes;
- bounded token perturbations around those prefixes.

This is not the rejected child-action-only method.  Parent trajectories remain
anchors, child-visited states expose recovery from drift, and the two sets are
kept separately in the report.

The candidate damage vector contains:

```text
attention_js       causal attention-distribution Jensen-Shannon divergence
attention_topk     retained attention-mass and top-k source agreement
query_temporal     temporal query-direction/self-similarity error
key_score_error    centered QK score error (softmax-invariant shifts removed)
value_readout      attention-weighted V readout error
key_cache_error    post-RoPE incremental key-cache diagnostic
value_cache_error  incremental value-cache diagnostic
delta_error        attention residual-contribution error
post_residual      post-attention residual error and angle
margin_damage      parent next-token and protected-token margin loss
rollout_survival   exact-prefix survival at horizons 1, 2, 4, 8, 16, and 32
```

Candidates are ranked by a Pareto/worst-quantile gate.  A good mean may not hide
a catastrophic domain or late-horizon failure.  Pair and small-group
interventions also measure synergy:

```text
synergy(i,j) = damage(i+j) - damage(i) - damage(j)
```

Large positive synergy pins the pair or requires it to be folded as one
hyperedge with its compensating paths.

Raw K/V reconstruction is retained only as a diagnostic.  HeadQ's 2026 result
shows why it cannot be the main objective: attention observes keys through QK
scores modulo row-constant softmax shifts, and it observes values through the
attention-weighted readout.  GRKV independently supports fitting the compressed
state against the actual attention output rather than storage-space MSE.
KVarN's pseudo-decode result supplies the other missing control: rewritten or
compressed cache state must be consumed at every subsequent step so accumulated
decode error is visible.

## First falsifiable experiment

The first run uses the preserved Qwen3-4B codebook frontier because it provides
a small, real, zero-recovery attention mutation whose baseline is already
known.

1. Start from the exact `attention p36 / MLP p36` parent-equivalent codebook.
2. Re-evaluate the known `attention p35 / MLP p36` boundary with residual-
   attention evidence and joint V/O residual fitting.
3. Compare against the existing p35 result:
   `3,995,570,176` parameters, behavior `400/400`, choice `250/250`, strict
   generation `372/388`, PPL `1,012.671`.
4. Attempt one additional attention prototype fold only after the p35 result is
   reproduced or improved.  This directly tests whether residual-attention
   fitting moves the p34 cliff rather than merely changing a local metric.
5. Recompute all evidence after every accepted fold.  Save/reload and parent-
   free evaluation are mandatory.

The method passes its first stage only if it meets all of these conditions:

- exact physical parameter/byte savings are nonzero;
- no recovery optimizer, backward call, or gradient is used;
- Q/K/V/O and cache evidence are complete and provenance-bound;
- full-prefix and cached incremental logits agree within the declared numeric
  tolerance before candidate comparisons;
- the p35 broad behavior and choice scores do not regress;
- strict generation is at least the existing `372/388` result;
- no hidden sidecar or parent dependency remains after reload.

If it cannot rank the known safe p35 candidate above destructive candidates,
or it improves teacher-forced metrics without improving horizon survival, the
method is rejected before a target-size run.

## Progression after the boundary test

If the boundary test passes:

1. apply the same operator progressively at 4B -> 1.7B/1.4B -> 0.6B -> target;
2. use the trained Qwen3-0.6B sibling as a coordinate and attention-pattern
   atlas, while requiring a second compile into a random same-shape child to
   prove that sibling weights are not the product dependency;
3. cluster parent heads by causal attention pattern, then solve child V/O paths
   against parent residual contributions for true width reduction;
4. combine attention hyperedges with complete SwiGLU gate/up/post-activation/
   down hyperedges and whole-block residual gates;
5. extend trajectories to byte-aligned prefixes for cross-tokenizer parents and
   children;
6. retain the same operator for native modality streams, with modality-specific
   prefix/cache lanes and a shared residual/action boundary.

## Current implementation status

- The prior atomic GQA surgery and exact rollback infrastructure is preserved.
- A seed-erasure audit now exists to prove whether compiled tensors and behavior
  are independent of the nominal blank initialization seed; it has synthetic
  tests but has not yet certified a real compile.
- Residual-attention evidence/gating, softmax-Fisher score-space diagnostics,
  fixed-attention value/readout diagnostics, bounded residual scaling, and
  zero-gradient V/O ridge primitives are implemented with synthetic tests.
- The cached collector now retains the exact full-prefill and incremental
  `v_proj` input chunks and rejects a trace unless their cumulative history,
  multiplied by the materialized V weight, reproduces the complete post-update
  V cache.  A global anchored ridge fitter and rollback-safe installer now fit
  one shared V survivor across every assigned logical site while applying each
  site's V role scale and GQA group permutation.
- The live Qwen cached pseudo-decode collector and transactional probe runner
  are the active integration step; no real-model result is claimed until that
  path runs.
- The previously missing isolated Qwen3-4B `MLP p36 / attention p34` medoid
  control is complete as `qwen3_4b_codebook_pn36_pa34_gpu6_r1_20260711`.
  Its parent-free checkpoint passed exact save/reload verification; its broad
  generation regression is recorded below.
- The failed unified v6 import mutated nothing.  Candidate identity drift is
  being repaired without weakening provenance checks before v7.

## First live Qwen boundary result

The live evidence path is now proven on the preserved Qwen3-4B p35-attention /
p36-MLP incumbent.  An isolated FP32 P40 greedy-cache smoke at layer 35/group 0
preserved cached/full-prefix top-1 exactly, with maximum logit drift
`1.8596649e-05` and reconstructed-context drift `1.6212463e-05`.  That certifies
the collector, not a new fold.

The p35 and p34 checkpoint registries also have an unusually clean lineage:
the group permutations are byte-identical, every p34 medoid set is a strict
subset of p35, and p34 removes exactly one real source prototype from each of
the eight canonical GQA groups.  This permits a granular p35-to-p34 experiment
without reconstructing either checkpoint from scratch.

The first live transaction redirected physical prototype `L19/G4` to the
existing `L21/G6` survivor.  The survivor was already executed at logical sites
`L2/G1` and `L21/G6`, so the global O solve correctly included those sites plus
the new `L19/G4` assignment.  The tentative registry changed from 280 to 279
attention prototypes and saved exactly `6,551,300` tensor bytes
(`3,276,800` learned parameters, plus changed index/provenance storage).

Measured over two prompts and four cached actions per prompt:

| Variant | Risk | O-delta error | Residual-state error | Action-margin damage | Top-1 mismatch |
|---|---:|---:|---:|---:|---:|
| Uncompensated | 0.903619 | 0.246419 | 0.085616 | 0.097040 | 0/8 |
| Scalar O | 0.884994 | 0.249970 | 0.086466 | 0.074947 | 0/8 |
| Global ridge O | 0.886460 | 0.243666 | 0.085462 | 0.050915 | 0/8 |

The ridge fit therefore reduced action-margin damage by about `47.5%` and
slightly improved the attention residual write.  It did not repair the main
addressing/cache fault: mean score-geometry error remained about `0.504` and
mean cache-transition error about `0.719`.  All three variants kept aligned
greedy prefixes for this small horizon, but all failed the intentionally strict,
uncalibrated local canary gate.  No checkpoint was committed, broad generation
was not evaluated, and the incumbent was restored exactly.

This is evidence for the residual-attention hypothesis, but also a falsification
of an O-only version of it.  The next causal test must independently substitute
parent K, V, attention delta, K+V, and K+V+delta.  The next installable fit must
run the newly implemented alternating sequence: globally fit and install the
shared V prototype, recollect the real child `A @ V`, and only then refit O.
The implementation enforces this order with materialized-V trace hashes.  It is
an experiment, not compaction proof: success still requires closed-loop gates,
broad generation, a saved/reloaded smaller checkpoint, and parent-free replay.

## Completed p34 broad boundary control

The zero-recovery `MLP p36 / attention p34` medoid control is now complete.  It
is a parent-free, saved/reloaded checkpoint with exact runtime equivalence after
reload.  It contains `3,969,355,776` learned parameters, a `1.013380589x`
source-to-child parameter ratio.  Relative to p35 it removes one additional
prototype in each of the eight canonical GQA groups.

The broad p35-to-p34 comparison is not a total coherence collapse; it is a
specific generation regression:

| Boundary | Behavior absolute | Parent behavior retained | Choice retained | Generation absolute | Parent generation retained | PPL |
|---|---:|---:|---:|---:|---:|---:|
| p35 | 400/1000 | 400/400 | 250/250 | 633/1000 | 372/388 | 1012.671 |
| p34 | 598/1000 | 398/400 | 250/250 | 417/1000 | 296/388 | 1235.360 |

The p34 child gained passes on some prompts the dense parent failed, which is
why its absolute behavior count rises, but it lost `92` parent-passing
generation cases versus `16` at p35.  Raw medoid selection therefore preserves
most short-form behavior and every choice probe while damaging multi-token
sequence continuation.  A teacher-forced or one-step gate would miss this
boundary.

The direct p35-versus-p34 generation comparison sharpens the diagnosis.  Of
the p35-passing cases that p34 lost, `125` are code prompts, `87` are tokenizer
prompts, `15` are model-surgery prompts, and `1` is math.  P34 gained only `12`
cases over p35 (`11` model-surgery and `1` math).  The extra attention fold is
therefore disproportionately breaking exact symbolic/context continuation,
not uniformly erasing every capability.

## First dense-parent causal intervention

The strict FP32 oracle replayed one code prompt for four recurrent cached
actions at the exact newly removed `L19/G4` site.  The p35 and p34 children were
separately compared with the same dense parent, token-row chart, driver tokens,
and attention coordinates.  The v2 oracle binds RoPE, Q/K normalization,
layer/head geometry, cache semantics, and the 151,936-to-151,669 vocabulary
chart before permitting raw substitution.

| Child | Intervention | Top-1 | Centered-logit RMSE | Parent→child KL | KL recovery vs child baseline |
|---|---|---:|---:|---:|---:|
| p35 | none | 4/4 | 0.820874 | 0.075446 | — |
| p35 | parent K | 4/4 | 0.820723 | 0.076711 | -1.68% |
| p35 | parent V | 4/4 | 0.815979 | 0.071226 | +5.59% |
| p35 | parent K+V | 4/4 | 0.814941 | 0.071470 | +5.27% |
| p35 | parent residual write | 4/4 | 0.810114 | 0.069604 | +7.74% |
| p34 | none | 4/4 | 1.029378 | 0.155213 | — |
| p34 | parent K | 4/4 | 1.066804 | 0.189554 | -22.13% |
| p34 | parent V | 3/4 | 1.034352 | 0.190618 | -22.81% |
| p34 | parent K+V | 3/4 | 1.064639 | 0.225014 | -44.97% |
| p34 | parent residual write | 4/4 | 0.952474 | 0.115764 | +25.42% |

The extra p34 fold therefore more than doubles baseline KL at this path and
changes parent-like V from mildly helpful to actively harmful.  Directly
restoring the parent residual attention write remains helpful.  This rejects a
raw-cache-MSE objective: parent K/V content cannot simply be transplanted into
the child's changed routing state.  The installable objective must fit the
child's own `A_child @ V_child @ O_child` residual contribution, while K/V
reconstruction remains an auxiliary recurrent-state diagnostic.

`K+V+residual-write` is intentionally omitted from the causal table.  Once the
complete `o_proj` return is clamped to the parent, same-step K/V effects are
overwritten; the v2 oracle proves this control is bit-identical to the residual
write alone and labels it a redundancy check rather than independent evidence.
This is a one-prompt localization result, not broad repair proof.

Local evidence:

- `runs/residual-attention-20260711/p35_redirect_L19G4_to_L21G6_h4_p2.json`
- `runs/residual-attention-20260711/p35_unshared_forced_cache_smoke.json`
- `runs/residual-attention-20260711/p34_forced_cache_smoke.json`
- `runs/qwen3_4b_codebook_p35_20260710/run_report.json`
- `runs/qwen3_4b_codebook_p34_20260711/run_report.json`
- `runs/residual-attention-20260711/p35_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`
- `runs/residual-attention-20260711/p34_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`

The exact four-control V -> O executable, fixed prompt splits, output contract,
and documented command are in
`docs/P35_RESIDUAL_ATTENTION_VALUE_PROBE_2026-07-11.md`.  Its implementation is
locally tested but no live run is claimed here.

## 2025-2026 primary sources used for this design

- Model Folding, ICLR 2025: <https://arxiv.org/abs/2502.10216>
- Official Model Folding implementation:
  <https://github.com/nanguoyu/model-folding-universal>
- Reasoning-Aware Compression / on-policy reconstruction:
  <https://arxiv.org/abs/2509.12464>
- TAPPA temporal attention-pattern analysis:
  <https://arxiv.org/abs/2601.21709>
- HeadQ model-visible key score/value readout distortion:
  <https://arxiv.org/abs/2605.03562>
- GRKV attention-output global regression:
  <https://arxiv.org/abs/2605.31105>
- KVarN pseudo-decode cache-error accumulation:
  <https://arxiv.org/abs/2606.03458>
- High-Layer Attention Pruning with residual rescaling:
  <https://arxiv.org/abs/2507.01900>
- GRAIL sequential zero-gradient compensation:
  <https://arxiv.org/abs/2602.23795>
- Chain of Merges sequential activation statistics:
  <https://arxiv.org/abs/2508.21421>
