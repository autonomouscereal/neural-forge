# Exact p35 QKVO Selected-State Factor Pack Replay

`experiments/reproduce_pack_qwen_p35_qkvo_selected.py` is the deterministic
artifact step after the successful assignment-local Q/K/V/O residual run. It
does not restart the search and does not compare parent and child weights.

The runner replays only updates 1 through 16 with rank 8, horizon 4, batch 2,
seed 0, Q/K learning rate `0.001`, V/O learning rate `0.002`, manual Adam
(`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`), hard-action weight 1,
distribution weight 0.1, temperature 1, and gradient clipping at 1. Calibration
is evaluated only at steps 0, 8, and 16.

It fails closed unless it reproduces all of these existing results:

- selected full adapter state:
  `f4cad9b277826a6724d76c9727999dc614ae5c54ef6197b9902d8d4c26c20240`;
- step-16 calibration evaluation:
  `370701ea2b3f521445d8df9adce5bfd403cdeba6ab3af99ddd718d79189ee130`
  (`45/48`, or `93.75%`, parent-action top-1);
- heldout action evaluation:
  `acdd2449699c25b3ad1eef4a81dcf2f58b774363fb546170113c46c7d4883d49`
  (`64/64`, or `100%`, parent-action top-1);
- heldout greedy comparison:
  `b8af9fb81de9fd40b88eb33044b74c1f46cec68d1a002079bc36f93acff1ae66`
  (`8/8` exact rollouts and `32/32` tokens).

The output safetensors file contains exactly eight FP16 factors: Q-left/right,
K-left/right, V-left/right, and O-left/right. That is 92,160 learned values and
184,320 raw tensor bytes. Immutable site/schema buffers are represented and
hashed canonically in the JSON manifest; they are not smuggled in as additional
tensors. The manifest binds the source evidence, exact fixtures, p35 base
manifest, role-scale table, replay implementation file hashes, environment, and
complete optimizer/loss schedule. Hard file-size limits and exclusive output
reservations prevent the format from turning into a disguised checkpoint.

After writing, the runner loads the safetensors file, constructs a fresh FP16
one-site registry, installs only those eight factors, and requires the complete
registry state to hash back to the selected state. It then proves exact equality
of calibration action logits, heldout action logits, heldout greedy traces, and
both discrete action-evaluation contracts. The full-vocabulary action-logit
hashes are the functional identity gate. Continuous loss and log-probability
aggregates are recorded as diagnostics but are not required to be byte-identical:
parallel reductions on the P40 can differ in their final floating bits even when
the input logits are byte-identical. Those continuous values never select or
promote the pack. The incumbent p35 runtime is restored in a `finally` path on
success or failure.

The prepared, not-uploaded and not-launched command wrapper is
`experiments/remote_reproduce_pack_p35_qkvo_selected_p40_gpu3_20260711.sh`.
It requires the original Tesla P40/Python 3.10.12/torch 2.4.1+cu121/FP16
environment and a new absent output directory.

Broad evaluation is intentionally the next separate pack consumer. It must
first evaluate untouched p35 live under the current strict generation contract,
then evaluate a freshly provenance-bound prompt-decontaminated p35-pass subset
with a zero-loss gate, and only then run the full 1,000 behavior / 250 choice /
1,000 generation / 128-text regression suite. The 22 QKVO prompt fixtures came
from that old broad suite, so the full suite is a regression test, not independent
selection or certification evidence.
