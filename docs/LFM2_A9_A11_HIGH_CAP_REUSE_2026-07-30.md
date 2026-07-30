# A9/A11 reuse-only high-cap evaluation

Date: 2026-07-30

This rerun answers only whether the old 256-donor-token evaluator boundary hid
useful endings. It does not refold either child and it does not regenerate a
parent trace or response.

## Bound inputs

- Standalone physical child checkpoint: 166,778,624 parameters.
- A9 `C_in = C_out`: authenticated k32 residual-OMP codebook artifact
  `4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d`.
- A11 `C_in/C_out`: that same authenticated input codebook plus the already
  persisted A10 output coefficient result
  `b301539b49e0156e5854e3d84730ead019b32cadd1d0436e017474556408bfc8`.
- Exact cached span evidence
  `134f1ccb0ac2b2b69829ea483263bb4d92c1783e9b809f5c4a87c4ff10528cf0`.
- Original A9 and A11 dossiers, including their exact cap-hit inventories.

The manifest authenticates all paths, artifact seals, prompt bytes, and source
dossier bytes. The runner refuses to execute a prompt that was not a source
cap hit.

## Evaluation policy

- Safety is reviewed separately from folding quality.
- A maximum-token hit is censored/incomplete, not failed.
- EOS, cap, manually observed looping, and runtime errors remain independent.
- The initial allowance is 1,024 donor semantic tokens.
- Only a result still capped at 1,024 escalates to the runtime maximum of 2,048.
- Whole-response manual review remains authoritative; no score promotes or
  rejects a fold.

## Reuse-only command

Run from the checked-out Neural Forge repository on the model host:

```bash
python experiments/rerun_lfm2_sparse_semantic_capped_prompts.py \
  --manifest experiments/lfm2_a9_a11_high_cap_reuse_manifest_20260730.json \
  --lane a9-k32-residual-omp \
  --lane a11-k32-split-cout \
  --device cuda:0
```

The two lanes may instead run concurrently on independently selected GPUs by
issuing one command per lane with different `--device` values. Attempt files
are prompt-and-cap checkpointed, so a restart reuses completed 1,024/2,048-token
decodes and does not redo them.

## Prohibited work authenticated by the result

The result operation audit must remain zero for parent model loads, parent
forwards/generation, parent trace recapture, fold compilation, support search,
coefficient solves, training, optimizer/backward calls, pruning, quantization,
and network access. Loading the donor tokenizer does not instantiate the donor
model. The only model forward calls are from the already-saved child.

Each lane writes `high-cap-dossier.json` plus resumable per-prompt attempt files
under its manifest-bound output directory. A runtime exception is preserved for
that prompt and does not get mislabeled as either a cap hit or behavioral miss.
