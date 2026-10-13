# Structured text recovery trainer review — 2026-10-13

This is a source audit and a small CPU integration protocol. It is not an
OMNI1 recovery run, an intelligence measurement, or promotion evidence.

The source inspected was the root recovery worktree's `train_text.py`, SHA256
`ca3f933a607e11732606a6ed10c2ca85f42271aa550ff52ebea002045ff123f0`, against
the exact audited `streamed_kd_trainer.py` required by `training.py`'s baseline
pin. Later fixes necessarily change the helper and candidate pins.

The assistant mask owns token t+1 when predicting from token t. Right padding
is cropped to the longest actual row in each microbatch. Each accumulation
window uses the total assistant prediction positions across all its rows as
its loss denominator, including a final partial window. The source and exact
candidate, helper, amendment, corpus and row-plan identities are incorporated
in the checkpoint data identity. The actual production command, layerwise
engine and AdamW optimizer are retained.

Two source issues were reported to the root agent and fixed during review:
duplicate native corpora aliased under different source names could be sampled
twice, and checkpoint cursor/counters could disagree with optimizer progress.
The root now rejects duplicate corpus hashes, verifies the cursor against the
applied accumulation windows and checks recorded progress and Adam step
counts. Nonfinite windows now restore the sampler cursor and retain the
applied-step/token counters, so they retry the same data. The inherited skip
guard bounds repeated failures. Attempted and applied counts remain separate.

An additional integrity gap was reported: the original production `QTextCache`
checks the corpus and shard JSON seals but does not verify raw top-k target file
hashes before mapping them. A same-size target mutation can leave the recorded
data identity unchanged. The real cache peer already provides `verify_shard`
for all five raw files and structural/finite checks. Structured source admission
must perform these checks before updates/resume and pin the actual verifying
peer. A data preparation docstring claiming these checks already occur in the
production reader was also identified as inaccurate.

Recorded progress alone cannot prove its own source-row counts: checking
`progress.real_tokens == tokens_seen` compares two saved values. Reconstructing
the actual row-plan prefix length/mask totals supplies independent evidence
when rejecting an internally inconsistent but resealed checkpoint. This is
checkpoint integrity hardening, not evidence a normal saved run consumed the
wrong rows.

`tests/test_chimera_recovery_train_text_integration.py` uses seven synthetic
2026 rows of unequal lengths and two real small Qwen3.5 full-attention layers.
It invokes the actual patched command parser and `command_train`, source plugin,
real cache writer/reader/verifier, layerwise forward/backward engine, AdamW and
safety checkpoint writer/loader. Only the preregistration peer is replaced by
an explicit synthetic-document pin; actual campaign preregistration admission
requires its separate production check.

The three integration checks require:

- uninterrupted versus interrupted/resumed final adapter tensors and Adam
  state to be exactly equal, with all seven rows consumed once, true token
  counts, assistant positions and a cropped final accumulation window;
- an injected single nonfinite accumulation window to replay the same real
  rows and produce exactly the clean-run adapter and Adam states;
- a saved, resealed wrong-cursor checkpoint and a same-size raw teacher target
  mutation to be refused before any new adapter export.

Set `CHIMERA_STREAMED_BASELINE` to the exact original and
`QTEXT_TEACHER_CACHE_PATH` to its actual cache peer, then run this test file in
the existing CPU Torch runtime. The local Python lacks Torch: the integration
checks were skipped explicitly, while the existing corpus, media admission and
evaluation tests passed (36 tests). Remote CPU execution results must be
recorded before treating the integration protocol as completed validation.
