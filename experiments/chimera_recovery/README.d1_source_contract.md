D1 source successor contract, 2026-10-13
======================================

`d1_candidate.py` prepares an additive candidate from the exact reviewed v2
launcher/trainer. It does not deploy, allocate GPUs, run training or register a
checkpoint. The preserved original queue and corpus remain evidence. The new
candidate uses distinct run, delta, status and preflight paths.

`d1_source_contract.py` admits the pinned successor before importing model code
or opening checkpoints. It checks the manifest SHA and canonical seal, every
source/document/amendment/dataset file SHA, exact fit/diagnostic groups and
document spans, safetensors shapes/dtypes/byte ownership, every real token ID,
zero padding, real lengths and both explicit row masks. The trainer refuses
different IDs, lengths keys, width, held count, amendment or row truncation.
The source record enters the actual run identity and initial receipt before
updates. The launcher repeats source/code checks after its resource wait.

Actual CPU-prepared successor v2 (parent receipt): 5,855 fit documents,
3,752,131 fit real tokens, 917 fit rows; 331 diagnostic documents, 250,280 real
tokens, 62 diagnostic rows. Total 979 rows and 4,002,411 real tokens, 7,573
padding positions and 619 whole-document overshoot. Its canonical seal is
`20b45eef4d5ebf8ca300191482bdd753761d3bab90698f9f8ec327e4d36dac1d`;
manifest file SHA is
`a488e75b46a667ebe91c23a08e5ce50a07a49be8208520f9af2d71bc060bbf88`.

The actual D1 teacher is the immutable parent prefix hidden-state forward.
`qtext_teacher_cache.py` logits are not a required D1 input. Optional later KD
must use the successor corpus with `--ids-key input_ids --lengths-key lengths`
and a new source-bound cache namespace; an old cache cannot silently be reused
against the new dataset. An optional argument receipt is included for this
purpose; it is not a cache seal or a teacher-runtime proof.

The local candidate artifact is under `reviews/2026-10-13-folding/
recovery-preparation/d1-candidate-v3-source-successor`. Parent must run the
source validator against actual host paths and the CPU dry run before promotion
or production use. Preparing these files is not evidence of recovery quality.

Verification:

    python -B -m pytest tests/test_chimera_d1_source_contract.py tests/test_chimera_d1_source_successor.py tests/test_chimera_d1_source_audit.py -q

Fourteen local tests passed, including invoking the actual patched trainer and
proving a changed source rejects before checkpoint loading/output creation.
Generated Bash passed `bash -n`. Genuine parent-runtime CPU validation remains
a separate receipt; it is not inferred from these tests.
