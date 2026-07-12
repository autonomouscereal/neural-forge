# P35 multi-site ragged Q/K/V/O residual fold — 2026-07-11

## Decision

The next experiment is a behavior-trained, joint eight-site p35→p34 fold. It
does **not** try to make child weights numerically resemble parent weights.
The child may use completely different factor values. The only selection and
acceptance signals are parent output behavior: action logits/ranks and greedy
decode tokens under parent-greedy and child-greedy prefixes.

The recommended first live schedule is `anchor-economy-r36`:

```text
(4, 4, 4, 8, 4, 4, 4, 4)
```

It preserves rank 8 at the one empirically successful L19/G4 site and spends
rank 4 at each still-unproven site. The runner also evaluates rank-sum 40 and
50 schedules on calibration before locking a winner; heldout data cannot
choose the schedule.

## Exact structural lineage

The eight redirects below were extracted read-only from the preserved p35 and
p34 codebook registries. Together they remove eight physical attention
prototypes, taking 280 p35 prototypes to 272 p34 prototypes while preserving
the group-permutation tensor exactly.

| Site | Removed physical source | Surviving physical target | Flat source → target | Economy rank |
|---:|---|---|---:|---:|
| 0 | L10/G2 | L15/G5 | 82 → 125 | 4 |
| 1 | L13/G2 | L35/G0 | 106 → 280 | 4 |
| 2 | L15/G2 | L19/G5 | 122 → 157 | 4 |
| 3 | L19/G4 | L21/G6 | 156 → 174 | 8 |
| 4 | L21/G3 | L7/G6 | 171 → 62 | 4 |
| 5 | L25/G1 | L26/G4 | 201 → 212 | 4 |
| 6 | L25/G5 | L20/G6 | 205 → 166 | 4 |
| 7 | L30/G1 | L14/G4 | 241 → 116 | 4 |

The exact p35 base manifest is pinned to:

```text
f62927114d5bc40fa10d239f7eb5d628ba217cd3fe2b7dcac0bdb5fbb68e9b1f
```

The successful single-site anchor state is pinned to:

```text
f4cad9b277826a6724d76c9727999dc614ae5c54ef6197b9902d8d4c26c20240
```

## What is being learned

Each removed site gets independent low-rank residual functions for Q, K, V,
and O. Ranks are truly ragged: a rank-4 site stores half as many learned values
as a rank-8 site; there is no hidden max-rank padding.

- Q/K/V corrections execute from the physical site's pre-normalization input,
  before RoPE and cache use.
- O corrections execute from that physical site's post-softmax context.
- The residuals are added only at the removed physical assignments.
- Training uses output loss only. It never scores weight distance, cache-weight
  similarity, or numerical agreement with the removed tensors.
- All 64 learned factor tensors must receive nonzero gradients.

The selected L19/G4 rank-8 pack seeds only that site. Every other site begins
as an exact zero correction with deterministic nonzero input factors and must
learn its behavior contribution from the fit split.

## Candidate budgets

Eight removed Q/K/V/O prototype slices contain 26,214,400 parameters. The
explicit role-scale table costs 1,152 parameters once. Each unit of total rank
costs 11,520 learned factor parameters.

| Schedule | Site ranks | Rank sum | Factor parameters | Raw FP16 factor bytes | Net parameters saved | Net FP16 bytes saved | Installed unique parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| `anchor-economy-r36` | 4,4,4,8,4,4,4,4 | 36 | 414,720 | 829,440 | 25,798,528 | 51,597,056 | 3,969,771,648 |
| `anchor-targeted-r40` | 4,4,6,8,4,4,4,6 | 40 | 460,800 | 921,600 | 25,752,448 | 51,504,896 | 3,969,817,728 |
| `anchor-conservative-r50` | 6,6,6,8,6,6,6,6 | 50 | 576,000 | 1,152,000 | 25,637,248 | 51,274,496 | 3,969,932,928 |

These are real parameter savings versus the exact 3,995,570,176-parameter
p35 incumbent, not file compression. The economy child removes about 0.646%
of p35 in this one incremental fold. It is a scale-up step toward repeated
compaction, not the final target-size child.

## Search and acceptance protocol

1. Capture exact p35 installation, tensor identities, and runtime snapshot.
2. Collect p35 fit/calibration greedy targets.
3. Install the joint 280→272 child and collect child-prefix drivers.
4. Restore p35 and collect parent outputs on those child-driven prefixes.
5. For each fixed rank schedule, train only its Q/K/V/O factors for 32 steps,
   calibrate at steps 0/8/16/32, snapshot the locked state, then roll back.
6. Re-evaluate post-training fit behavior and exclude any schedule that regresses
   versus the no-adapter joint child or leaves a factor tensor without a
   gradient. Among fit-passing schedules, select by higher calibration
   parent-action top-1, lower mean parent-action rank, then fewer parameters.
   Calibration must also not regress versus the no-adapter joint child.
7. Only after schedule lock, enter the heldout stage. Its required baseline,
   parent-target, winner, manifest, and reload forwards all occur after that
   lock. Require 100% parent-action top-1, mean rank 1, exact greedy
   rollout/token matches, and no regression versus the joint child.
8. Write only the winner's factor bundle. Reload it into a fresh ragged
   registry and require the exact installed state, exact raw fit/calibration/
   heldout action-output manifests, identical discrete evaluation contracts,
   and identical greedy traces. Continuous loss/log-probability reductions are
   recorded as diagnostics rather than treated as artifact identity because
   repeated GPU reductions can differ despite byte-identical raw logits.
9. Roll back every candidate and restore the exact p35 installation on success
   and failure.

The bundle contains eight small safetensors files and manifests. It explicitly
contains zero optimizer, parent, base-model, or full-checkpoint tensors and is
hard-capped at 2 MiB. A failed post-write verification removes only a
byte-verified bundle owned by that invocation. Loading rejects undeclared
files, symlinks, and nested directories; installation rebinds the saved
geometry and role-scale-table hash to the live child.

## Current status and honest limitations

The local implementation is complete and its focused suite passes 18 tests.
No multi-site GPU result exists yet, and the prepared P40 wrapper has not been
uploaded or launched.

Only L19/G4→L21/G6 has prior positive single-site evidence. The other seven
ranks are disciplined hypotheses, not proven capacities. Joint corrections
can interact non-additively across layers, so eight individually reasonable
sites may still fail together. The heldout split is disjoint from fit and
calibration, but historical broad-suite overlap means this small gate is not
an independent compaction certification. Any passing bundle still needs the
separate broad behavior consumer before it can be called a successful
eight-site compaction.

This experiment proves or rejects one same-family attention fold. It does not
by itself prove arbitrary architecture, tokenizer, modality, or target-size
translation. Those remain the larger compiler objective, with this operator
serving as a behavior-preserving primitive if it scales.

## Files

- `src/neural_forge/hf_ragged_assignment_local_qkvo_residual_adapter.py`
- `src/neural_forge/hf_ragged_qkvo_factor_bundle.py`
- `experiments/run_qwen_p35_multisite_ragged_qkvo_residual.py`
- `experiments/remote_run_p35_multisite_ragged_qkvo_factor_bundle_p40_20260711.sh`
- `tests/test_hf_ragged_assignment_local_qkvo_residual_adapter.py`
- `tests/test_hf_ragged_qkvo_factor_bundle.py`
- `tests/test_run_qwen_p35_multisite_ragged_qkvo_residual.py`
