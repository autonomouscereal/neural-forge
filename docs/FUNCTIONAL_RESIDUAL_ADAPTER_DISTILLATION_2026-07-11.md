# Functional residual-adapter distillation

Date: 2026-07-11

This is the next no-repeat residual-attention experiment after the exact p35
`L19/G4 -> L21/G6` V/O scale search. It is explicitly designed for projected
child weights that need not numerically resemble parent weights.

## Why this operator

The current causal evidence localizes the missing information to the complete
assignment-specific attention residual write:

- transplanting parent K/V directly was harmful;
- replacing the complete residual write improved the local KL error by
  `25.42%`;
- raw V regression fit its calibration cache extremely well but destroyed
  held-out actions and rollouts;
- global O fitting damages the survivor's existing L2 and L21 consumers;
- all eight alternate real Q/K pairs were worse than the existing Q/K;
- a two-vector residual-targeted V basis changed fit error by only about
  `0.04%` relative and produced no useful repair.

The next candidate should therefore preserve the shared survivor and add a
small correction only where physical assignment `L19/G4` executes.

## Operator

Let `z[l,g,t]` be the actual post-softmax GQA group context. The compact child
keeps the shared O projection and adds one low-rank site correction:

```text
residual_write[l,g,t] = O_shared z[l,g,t] + U[l,g] (R[l,g] z[l,g,t])
```

For Qwen3-4B at this site:

```text
hidden width h        = 2560
group-context width m = (32 query heads / 8 KV heads) * 128 = 512
U shape               = [2560, r]
R shape               = [r, 512]
```

The correction is attention-aware because it consumes the real causal,
post-softmax context. It is assignment-local, so it does not alter the shared
prototype's other consumers. No bias is required; the existing assignment O
scale can act as a gate.

## Exact parameter accounting

For `n` corrected assignments:

```text
adapter parameters = n * r * (h + m)
```

For this one site:

```text
adapter parameters = 3072r
```

One physical attention-prototype redirect removes `3,276,800` parameters.
The tentative codebook introduces its full role-scale tensor of
`36 * 8 * 4 = 1,152` parameters. Therefore the one-redirect net is:

```text
net parameters saved versus p35 = 3,276,800 - 1,152 - 3,072r
```

At rank 8, the adapter has `24,576` parameters and the redirect remains
`3,251,072` parameters smaller than p35 before exact index/header accounting.
The complete p35 -> p34 step contains eight separate redirects; the roughly
26-million-parameter full step must not be confused with one prototype.

In a dense target architecture, `U @ R` can be baked into that site's O
matrix and the factors discarded. In the shared codebook, retaining sparse
site factors is cheaper than unsharing the complete O tensor. Either way, the
final child must own the correction inside its checkpoint; no parent runtime,
external sidecar, or optimizer state is allowed.

## Output-only training

Use three prompt-level-disjoint sets: adapter fit, calibration selection, and
held-out evaluation. The current 6/8 scalar-search split is an evaluation
fixture, not a sufficient optimizer corpus.

Each micro-round uses both:

1. parent-greedy prefixes, forced through the child;
2. current child-greedy prefixes, evaluated by the frozen parent;
3. newly regenerated child prefixes after an accepted round.

The teacher target is always the parent's next action/distribution:

```text
loss =
    lambda_parent * KL(parent(.|parent prefix), child(.|parent prefix))
  + lambda_dagger * KL(parent(.|child prefix), child(.|child prefix))
  + mu * CE(child, parent top-1 action)
```

This differs from the failed child-action-only rollout method. A child's
current token is never made the target merely because the parent assigns it
nonzero probability. Fixed parent trajectories prevent a collapsed child
attractor from becoming the whole training distribution, while DAgger-style
student prefixes train recovery on states the child actually visits.

Only U/R (and an explicitly counted optional gate) may receive gradients.
Every other parameter must remain byte-identical. Parent/child weight,
activation, residual, attention, and cache similarity are prohibited as loss,
selection, or promotion inputs. OT, PCA, SVD, or residual fits may initialize
U/R, but action behavior is free to move the factors anywhere.

## Promotion contract

1. Run the V/O scalar search and retain its calibration winner only as the
   baseline initializer.
2. Train a bounded rank ladder such as `r = 1, 2, 4, 8, 16` on the fit split.
3. Select on calibration parent-action retention and independent child
   rollouts only.
4. Keep held-out evaluation winnerless; it cannot rename the calibration
   winner.
5. Commit one net-smaller candidate, save/reload it, and verify parent-free
   inference plus exact artifact bytes.
6. Run the full `1000/1000/250` behavior/generation/choice suite and PPL.
7. If successful, repeat transactionally for the other seven p35 -> p34
   redirects before proceeding toward intermediate and target-size children.

The clean reusable output-only components are
`hf_functional_decode_trace.py`, `hf_residual_attention_functional.py`, and
`hf_rollout_gate.py`. Output KD math can be extracted from `hf_recovery.py`,
but the new runner must not couple itself to hidden-state recovery paths.
