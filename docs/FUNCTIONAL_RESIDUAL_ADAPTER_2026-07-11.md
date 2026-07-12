# Functional residual adapter proof path (2026-07-11)

## What this tests

The tentative `p35` redirect assigns physical site `L19/G4` to the surviving
prototype at `L21/G6`. A sparse adapter now provides one assignment-specific
correction without changing the shared survivor for its other consumers:

```text
base_O(z_all) + U[L19,G4] @ (V[L19,G4] @ z_physical_group_4)
```

`z` is the post-softmax group context immediately before the O projection and
residual add. The correction is therefore attention-aware and context-aware,
but its factors are not required to resemble any parent tensor numerically.

The implementation is in
`src/neural_forge/hf_functional_residual_adapter.py`. It is a registered,
sparse child module with sorted `int32[site,2]` indices and learned
`left[site,h,rank]` / `right[site,rank,m]` factors. It does not allocate dense
zero factors for every layer/group and retains no parent runtime reference.

## Exact one-site accounting

For Qwen3-4B, `h=2560`, `m=512`, and `head_dim=128`.

```text
removed coupled group prototype = 2*h*(m+head_dim) = 3,276,800
new redirect role-scale table     = 36*8*4             =     1,152
rank-r adapter                    = r*(h+m)             =   3,072*r
net saved versus p35              = 3,276,800 - 1,152 - 3,072*r
```

At rank 8, the adapter adds 24,576 learned parameters and the tentative
redirect still saves exactly 3,251,072 parameters. At FP16/BF16 this is
6,502,144 learned payload bytes. Including the redirect's four-byte prototype
index reduction, the adapter's eight-byte site index, and its eight schema
buffer bytes gives 6,502,132 tensor-payload bytes before safetensors header and
manifest deltas. The artifact planner remains authoritative for a checkpoint
claim.

The previously observed `26,897,920` saving is not one L19/G4 prototype and is
not used in this formula. Full-step totals must come from the live artifact
report rather than multiplying or hardcoding that result.

## Functional-only learning contract

Training accepts only:

- fixed parent logits and their verified top-1 action;
- prefixes from both fixed parent-greedy and regenerated student-greedy paths;
- the real child model's next-token logits.

The objectives are hard parent-action cross entropy and optional parent-output
distribution KL. There is no API for a weight, Q/K/V/O, activation, attention,
residual, or cache target. The best in-memory factor state is selected only on
the calibration split. Heldout batches are rejected by the training boundary
and cannot be marked as selection inputs.

This is learned/recovery-assisted compaction, not a zero-gradient fold-only
claim. The proof path constructs no optimizer object or optimizer state, saves
no checkpoint, and exposes no commit operation yet.

## Transaction and proof status

The installer:

1. converts physical groups through the live permutation and verifies that
   L19/G4 and L21/G6 share survivor provenance `174`;
2. verifies removed provenance `156` is absent;
3. hashes every base attention-registry state tensor and all parameter/buffer
   identities;
4. installs one unique sparse registry owner and a parameterless O view;
5. starts with `U=0`, so the tentative child output is bit-exact before
   learning;
6. permits only adapter-factor changes and verifies the base registry remains
   identical;
7. restores the original projection object and parameter count on rejection,
   even if adapter factors themselves were corrupted.

The synthetic codebook proof covers physical/canonical permutation traps,
false redirect provenance, duplicate/unsorted sites, non-finite factors, exact
zero behavior, exact rollback, native state roundtrip, action-only improvement,
both rollout-prefix families, and heldout leakage rejection.

Local verification:

```text
15 adapter-specific tests passed
65 adapter + codebook + functional contract tests passed
Ruff clean
```

No remote deployment, candidate save, or full-Qwen result is claimed by this
primitive proof.
