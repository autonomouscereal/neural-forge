# P35 QKVO Factor-Pack Broad Evaluation

`experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py` is the read-only
promotion consumer for the exact one-site rank-8 Q/K/V/O factor pack. It does
not replay training, alter the pack, bake weights, save a checkpoint, or commit
the redirect.

## Required immutable inputs

- Exact p35 checkpoint whose manifest SHA-256 is
  `f62927114d5bc40fa10d239f7eb5d628ba217cd3fe2b7dcac0bdb5fbb68e9b1f`.
- Exact Qwen3-4B-Base tokenizer snapshot used by p35.
- Verified eight-factor pack and canonical manifest produced by
  `reproduce_pack_qwen_p35_qkvo_selected.py`.
- Broad suite file SHA-256
  `fa6066980faf4677b194e00951076dc477cc4243d185449aa6590ca4c8768166`.
- Raw-Qwen strict report SHA-256
  `3333bc54e9170dc8a65e8a53179c9a20a2ab7d1931e2da5795f6bc5ecf031395`.
- The exact 8 fit, 6 calibration-selection, and 8 heldout fixture files and
  manifests already bound into the factor-pack provenance.

The consumer verifies file hashes, factor inventory, selected adapter state,
fixture provenance, redirect site, role-scale table, registry geometry, P40
environment, and tokenizer scoring paths before candidate evaluation.

## Staged evaluation

1. Evaluate untouched p35 over the full suite with the current
   `SuiteExecutor` and 96-token strict-generation contract, or resume from an
   explicit file-and-SHA-256-bound report produced by that exact path. The
   legacy p35 report claiming 633 generation passes is never reused. The live
   P40 baseline has 400 behavior, 250 choice, and 544 generation passes. The
   earlier 546-generation snapshot and raw-Qwen deltas are historical drift
   diagnostics, not acceptance conditions.
2. Exclude any broad row whose prompt has the same token IDs as any of the 22
   factor fit/calibration/heldout prompts. Keep only fresh p35 passes.
3. Canonically hash and record the dynamically derived decontaminated subset.
   The authenticated fresh-544 report derives exactly:

   | Domain | Rows | Ordered-ID SHA-256 |
   |---|---:|---|
   | Behavior | 333 | `2c528c1ed36196e00f2c422eb135eb91dcc9a27441948b5092709b12a6d47ae4` |
   | Choice | 250 | `317bb1e8815eef3cb0a0ef6f4266c0405c31d7b3aad57b601d387dba2231abaf` |
   | Generation | 386 | `212a81f6111c78b8b61c7d520ea7e63419742ea8e8a59667eb74ab35c04da864` |

   Its canonical suite SHA-256 is
   `b718f6806f22cc37be3dd850287d84ffcaf148a68b666d11e9fb86ad6190a689`.
   These rows, derived from the fresh live report, define the zero-loss
   acceptance gate. The earlier snapshot was:

   | Domain | Rows | Ordered-ID SHA-256 |
   |---|---:|---|
   | Behavior | 333 | `2c528c1ed36196e00f2c422eb135eb91dcc9a27441948b5092709b12a6d47ae4` |
   | Choice | 250 | `317bb1e8815eef3cb0a0ef6f4266c0405c31d7b3aad57b601d387dba2231abaf` |
   | Generation | 388 | `6b8e537bb44cb7b7a39d4489264fdbe6b158de7a08adadf977081db262eccfae` |

   Its historical canonical suite SHA-256 is
   `51f65b3f11dd402b6b27fd39657d04cfb6cb73b417490a226b2d4088a971aa00`.
   Those historical counts and hashes are reported for drift analysis only.
4. Install the exact `L19/G4 -> L21/G6` structural redirect, create the fresh
   rank-8 adapter registry, and transactionally load the eight verified factor
   tensors.
5. Evaluate the candidate on the decontaminated parent-pass subset. Any loss
   rejects the promotion and suppresses the full candidate evaluation.
6. Only after a zero-loss pass, evaluate the complete 128-PPL, 250-choice,
   1,000-generation, 1,000-behavior suite. This reports candidate-versus-p35
   losses/gains, raw-Qwen pass retention, and PPL ratios. The full suite is not
   used for acceptance because it overlaps the factor fixtures.
7. Roll back the adapter and restore the incumbent p35 attention registry by
   exact snapshot and state identity. Rehash the pack and manifest to prove
   they were not changed.

The overlap proof is expected to find 67 behavior rows, zero choice rows, and
163 generation rows sharing fixture prompts. Among current p35 passes, those
counts are 67, zero, and 158 respectively.

## Outputs

The fresh output directory contains JSON only:

- `p35_current_strict_full.json` for a fresh run; a resume instead records the
  immutable external report path and file hash
- `decontaminated_parent_pass_suite.json`
- `decontamination_provenance.json`
- `p35_decontaminated_projection.json`
- `candidate_decontaminated.json`
- `candidate_full_broad.json` only after the zero-loss gate passes
- `broad_evaluation_report.json`

No model, checkpoint, optimizer, parent tensor, factor tensor, or binary
artifact is written.

## Expected cost

The full suite deduplicates to 1,064 generation/behavior prompt texts and
performs 1,000 serial choice-option forwards plus 128 PPL forwards. The
decontaminated subset has approximately 588 unique generation/behavior prompts, still has
1,000 choice-option forwards, and omits PPL.

- Rejected candidate: one full p35 evaluation plus the subset candidate,
  approximately 1.55 full generation workloads and two choice workloads.
- Passing candidate: add one full candidate evaluation, approximately 2.55
  full generation workloads, three choice workloads, and two PPL workloads.

Budget several hours on one Tesla P40. FP16 model weights are approximately
8 GB; the 24 GB P40 has sufficient memory for batch-8 greedy evaluation. JSON
disk usage should remain below roughly 25 MB even when the full stage runs.

The prepared P40 wrapper is
`experiments/remote_evaluate_p35_qkvo_factor_pack_broad_p40_gpu3_20260711.sh`.
The initial consumer completed and preserved its live baseline but failed
closed on the stale 546-pass historical contract. The review-only resume
wrapper
`experiments/remote_resume_p35_qkvo_factor_pack_broad_p40_gpu3_20260711.sh`
hash-locks that 4,993,045-byte report at
`403329fea0ec53510ec88449b8512de5331a05665e5229d78ca1ffa8ff29003c` and
skips the expensive baseline forward pass.

The resume also requires
`experiments/p35_qkvo_broad_fresh544_resume_certificate_20260711.json`, SHA-256
`365768ab766310d62afd7de141546eeb10a3576bd4bc8e79a3b212ea326e10e9`.
The Python gate consumes exact report/certificate bytes once, verifies the
checkpoint state manifest, tokenizer, producer sources, choice rows, all 1,000
behavior rows by scorer replay, live pass hashes, and bounded seven-loss / five-gain
generation churn. The audited wrapper passes its physical GPU UUID, compute
capability, driver, container image ID, Transformers tree hash, and
Hugging Face Hub tree hash into Python; those values must equal the certificate
and are copied into the final audit report. The only writable container mount
is the separate resume run root; the baseline and certificate are distinct
read-only file mounts and are not reachable through that writable tree. It
must not be launched before review.
