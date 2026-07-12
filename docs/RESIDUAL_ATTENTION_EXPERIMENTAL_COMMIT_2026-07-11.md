# Residual-attention experimental commit path (2026-07-11)

`experiments/run_qwen_residual_attention_commit.py` turns a measured redirect
into a new parent-free Qwen codebook checkpoint without modifying the
incumbent. It is deliberately fail-closed and permanently labels every saved
child `experimental-uncertified`.

The transaction does the following in order:

1. Requires an existing canonical codebook checkpoint and a new, absent output
   directory.
2. Previews explicit redirects atomically through the proven attention-surgery
   graph and coupled Q/K/V/O compactor.
3. Resolves physical provenance coordinates through each layer's group
   permutation.
4. Includes every executable `[layer, group]` assigned to the selected
   survivor slot, including the survivor's earlier assignments.
5. Collects parent and tentative-child `use_cache=True` lanes for a disjoint
   calibration split and held-out split.
6. Uses calibration only to fit either:
   - one bounded scalar on the existing survivor O block, or
   - one FP64 ridge replacement for that existing survivor O block.
7. The design is the candidate's actual `A_child @ V_child` context. The target
   is the isolated matching parent group contribution, not the full attention
   delta.
8. Proves non-O tensors and every unaffected O slot were unchanged by the
   compensation edit.
9. Requires forced-token and greedy cached lanes to pass independently on both
   calibration and held-out data. There is no CLI gate bypass.
10. Saves into a staging directory, reloads it without a dense parent, verifies
    the full tensor state, parameter report, attention snapshot, zero parent
    runtime references, and canonical artifact accounting, then performs one
    final non-overwriting directory rename.

Any failure before that final rename restores the incumbent registry/config
exactly, deletes staging, and leaves the requested output absent. The source
checkpoint's complete file inventory is SHA-256 checked before and after.

## Build the L19G4 -> L21G6 scalar and ridge siblings from p35

Run these as two independent transactions from the same p35 incumbent. The
commands below are templates only; they were not launched while implementing
this path.

```bash
ROOT=/home/cereal/model-surgery-lab/neural-forge-src
PY=/home/cereal/model-surgery-lab/venv/bin/python
MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
P35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child
OUT=/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711

CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 \
PYTHONPATH="$ROOT/src" "$PY" \
  "$ROOT/experiments/run_qwen_residual_attention_commit.py" \
  --codebook-checkpoint "$P35" \
  --tokenizer "$MODEL" \
  --candidate 19:4=21:6 \
  --calibration-prompt "The capital of France is" \
  --calibration-prompt "Complete the Python function: def add(a, b):" \
  --heldout-prompt "What is 19 plus 26?" \
  --heldout-prompt "Write one sentence explaining why ice floats." \
  --compensation scalar \
  --ridge 0.001 \
  --horizon 8 \
  --max-prompts-per-split 2 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --device cuda:0 \
  --dtype float16 \
  --reload-device cpu \
  --local-files-only \
  --output-checkpoint "$OUT/p35_L19G4_to_L21G6_scalar_h8_child" \
  --report "$OUT/p35_L19G4_to_L21G6_scalar_h8_commit.json"

CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 \
PYTHONPATH="$ROOT/src" "$PY" \
  "$ROOT/experiments/run_qwen_residual_attention_commit.py" \
  --codebook-checkpoint "$P35" \
  --tokenizer "$MODEL" \
  --candidate 19:4=21:6 \
  --calibration-prompt "The capital of France is" \
  --calibration-prompt "Complete the Python function: def add(a, b):" \
  --heldout-prompt "What is 19 plus 26?" \
  --heldout-prompt "Write one sentence explaining why ice floats." \
  --compensation ridge \
  --ridge 0.001 \
  --horizon 8 \
  --max-prompts-per-split 2 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --device cuda:0 \
  --dtype float16 \
  --reload-device cpu \
  --local-files-only \
  --output-checkpoint "$OUT/p35_L19G4_to_L21G6_ridge_h8_child" \
  --report "$OUT/p35_L19G4_to_L21G6_ridge_h8_commit.json"
```

The coordinate means prototype provenance layer 19, physical group 4 redirects
to prototype provenance layer 21, physical group 6. Executable sites are then
derived through the saved group permutations; the command does not assume
those physical provenance coordinates equal a layer's materialized group axis.

## Promotion boundary

A saved sibling is not a result yet. Compare scalar, ridge, uncompensated p34,
and p35 on the existing untouched broad protocol: 1,000 behavior prompts,
1,000 exact-generation prompts, 250 choice probes, perplexity, strict reload,
and multi-token manual generation. Keep p35 as the incumbent unless one child
passes that broad gate. The embedded checkpoint marker cannot be converted to
`certified` by this local commit runner.

The current commit runner's cached-lane gate is a conservative transaction
guard, not the definition of model quality. Cache/full-prefix or activation
numerical closeness must be recorded separately from broad functional
behavior; an internal-tolerance rejection must not be interpreted as proof
that a functionally better candidate is invalid. Such a candidate needs a
record-only evaluation path before any commit decision, while provenance,
rollback, artifact accounting, and absence of parent dependencies remain hard
requirements.
