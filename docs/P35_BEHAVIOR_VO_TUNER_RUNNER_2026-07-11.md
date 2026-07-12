# P35 behavior-only V/O tuner runner

`experiments/run_qwen_p35_behavior_vo_tuner.py` is the local execution wrapper
for `hf_residual_attention_behavior_tuner.py`. It reuses the exact tentative
codebook construction used by the prior role-scale search and is hard-wired to
the one-prototype redirect `L19/G4 -> L21/G6`.

## What it optimizes

The proposed child already has the structural `280 -> 279` edit installed.
The tuner changes only the two existing V/O role-scale scalars at physical
assignment `L19/G4`. Q/K, every other role-scale row, and every other model
parameter are frozen.

The manual two-scalar Adam objective reads only causal-model outputs:

- cross-entropy on the fixed parent top-1 action;
- a hinge loss on that action's child output margin;
- optional KL distillation over the parent's top-k output tokens.

It does not compare parent and child weights, Q/K/V tensors, attention maps,
hidden activations, residual streams, or cache values. The cached decoder is
used only to produce real token paths and output logits.

## Three immutable prompt roles

The runner requires three JSON files and verifies their raw SHA-256 values
before loading the model:

| Role | Count | Use | SHA-256 |
|---|---:|---|---|
| fit | 8 | gradients only | `b67d140eb3e5f4bbaaa01a0e47ef755818a3379d4d25692d805bcb5c252003ca` |
| calibration-selection | 6 | select incumbent or fitted candidate | `a2eaea993c7f7f434e30744eeaf4b4da750b9a327d835deedb385dbaf154e406` |
| heldout-evaluation | 8 | evaluation after the selection lock; no winner | `cebf98bdbdfc1154c45758cdeed25e15141b6aeb910f06fa6cf349efdaf6d04f` |

The six calibration and eight heldout prompts are the existing audited files
and were not edited. The new fit file is
`experiments/p35_p34_residual_attention_fit_prompts_20260711.json`.

Its manifest,
`experiments/p35_p34_residual_attention_fit_prompt_manifest_20260711.json`,
derives eight degradation targets from the existing p35/p34 transition
inventory. It excludes every audited calibration/heldout ID and exact prompt
before ranking, then takes deterministic per-category quota winners: three
code, three tokenizer, and two model-surgery cases. The sole math loss remains
heldout-only. The runner independently repeats this derivation and fails if an
entry, rank, prompt hash, file byte, or overlap count changes.

After tokenization, the runner performs a second pairwise-disjointness check.
This catches distinct source strings that truncation or tokenizer behavior
might collapse to the same token row.

## Transaction order

1. Verify all three text fixtures and provenance files.
2. Construct, but do not install, the exact tentative redirect.
3. Collect parent forced and real-greedy functional traces once for all three
   splits from untouched p35.
4. Install the tentative 279-prototype registry.
5. Fit on the fit split only.
6. Select on calibration only and hash the selection lock.
7. Evaluate heldout after that lock without naming a heldout winner.
8. Restore the tentative child scales exactly.
9. Restore the original p35 runtime exactly in a `finally` block.

The report includes parent/proposed attention snapshots, affected assignment
lineage, byte accounting, the nested behavior-tuner audit, and both rollback
proofs. It returns selected scales as data only. It has no save or commit CLI
option and explicitly reports `checkpoint_saved: false` and
`fold_committed: false`.

## Local invocation

```powershell
.venv\Scripts\python.exe experiments\run_qwen_p35_behavior_vo_tuner.py `
  --codebook-checkpoint <p35-checkpoint-directory> `
  --tokenizer <qwen-tokenizer-directory-or-id> `
  --fit-prompts-json experiments\p35_p34_residual_attention_fit_prompts_20260711.json `
  --calibration-selection-prompts-json experiments\p35_p34_residual_attention_calibration_selection_prompts_20260711.json `
  --heldout-evaluation-prompts-json experiments\p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json `
  --device cuda:1 `
  --dtype float32 `
  --horizon 8 `
  --steps 16 `
  --output runs\residual-attention-20260711\p35_l19g4_behavior_vo_tuner.json
```

This is a one-redirect experiment, not a p34 proof. A successful candidate
would still need an explicit commit/save/reload workflow, broad promotion
evaluation, and eventually the other seven p35-to-p34 redirects.
