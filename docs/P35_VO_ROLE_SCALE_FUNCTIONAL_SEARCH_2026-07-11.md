# Exact p35 per-assignment V/O role-scale functional search

Date: 2026-07-11

`experiments/run_qwen_p35_vo_role_scale_search.py` tests one narrow hypothesis
without changing the structural redirect: the tentative p35
`L19/G4 -> L21/G6` fold may be functionally muted by its automatically fitted
per-assignment V/O role scales.

The raw least-squares projection observed for the removed L19 assignment was
approximately:

```text
V scale = -0.001249
O scale = -0.000539
```

Those values are close enough to zero to nearly erase both the assignment's
value-cache contribution and its residual write. That does not prove the scale
fit is the cause of the behavioral failure, but it creates a direct,
low-dimensional test: keep the exact same 280 -> 279 prototype redirect, keep
the assignment's current Q/K addressing scales unchanged, and search only its
two already-existing V/O scalars by actual cached output behavior.

## Exact edit boundary

The runner is hard-wired to physical prototype redirect and executable scale
site:

```text
removed prototype:       L19/G4, flat source 156
surviving prototype:     L21/G6, flat source 174
scale target:            executable physical L19/G4
expected shared sites:   L2/G1, L19/G4, L21/G6
prototype transition:   280 -> 279
```

Physical L19/G4 is resolved through layer 19's group permutation to its unique
canonical assignment row. Every candidate copies that row's current Q and K
scales exactly and changes only columns V and O. The role-scale parameter
already exists in the tentative compact registry: the search adds no adapter,
parameter, layer, optimizer state, or runtime sidecar.

Repeated `--vo-scale V:O` arguments declare the bounded search set. Values are
rounded once to the registry dtype, duplicate materialized pairs are removed,
and candidate IDs/order are content-derived rather than input-order-derived.
The tentative redirect's incumbent V/O pair is always included, even when the
user does not repeat it explicitly.

## Functional evidence and leakage boundary

The parent p35 model is traced once per prompt and lane on two token-row-disjoint
sets:

- calibration parent greedy rollout plus its matching parent-driven forced
  lane;
- held-out parent greedy rollout plus its matching parent-driven forced lane.

For every V/O pair the actual tentative child is installed, then the lightweight
functional collector runs the real cached decode path. It records only token
paths and output logits. It does not reconstruct or compare Q/K/V, attention
matrices, activations, residuals, or cache tensors.

Candidate selection uses calibration function only, with fixed parent-action
agreement/likelihood/rank ahead of independent greedy rollout retention and
degeneration. Selection fails closed unless every calibration forced driver is
the parent's actual top-1 action. Held-out function receives a separate
evaluation ordering, names no winner, and cannot alter the calibration winner.

The report retains each candidate's installation and trace hashes, but these
are provenance and transaction evidence, not similarity features in the
ranking key.

## Transaction proof

Each candidate installation proves:

- physical-to-canonical assignment mapping;
- Q/K scale equality with the tentative incumbent;
- changed coordinates are a subset of only target V/O;
- all non-scale registry tensors and parameter identities are unchanged;
- functional collection leaves the installed candidate live and exact;
- rollback restores the full tentative registry snapshot exactly.

After all candidates, the tentative 279-prototype registry is discarded and
the original 280-prototype p35 runtime is restored by identity and content.
The runner cannot save or commit a checkpoint and never calls backward.

## Prepared invocation

This is a local implementation command only. It has not been deployed or
launched remotely.

```bash
ROOT=/home/cereal/model-surgery-lab/neural-forge-src
MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
P35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child

PYTHONPATH="$ROOT/src:$ROOT/experiments" \
python "$ROOT/experiments/run_qwen_p35_vo_role_scale_search.py" \
  --codebook-checkpoint "$P35" \
  --tokenizer "$MODEL" \
  --vo-scale=-0.001249:-0.000539 \
  --vo-scale 0.25:0.25 \
  --vo-scale 0.5:0.5 \
  --vo-scale 1.0:1.0 \
  --vo-scale 1.0:-1.0 \
  --calibration-prompts-json /path/to/calibration.json \
  --heldout-prompts-json /path/to/heldout.json \
  --horizon 8 \
  --top-k 5 \
  --max-candidates 16 \
  --max-abs-scale 4 \
  --max-prompts-per-split 4 \
  --device cuda:0 \
  --dtype float32 \
  --output /path/to/p35_vo_role_scale_search.json \
  --local-files-only
```

The CLI explicitly recognizes a negative colon-delimited pair as the value of
`--vo-scale`; both `--vo-scale -0.001249:-0.000539` and the shown
`--vo-scale=-0.001249:-0.000539` form are accepted.

## Interpretation limit

A favorable winner would show that the same one-prototype p35 redirect retains
more parent output function when the removed assignment is not nearly muted.
It would not yet prove a deployable compact child. No checkpoint is written;
promotion would still require a separately authorized commit, save/reload
identity, parent-free replay, and broad predeclared behavior/generation tests.
