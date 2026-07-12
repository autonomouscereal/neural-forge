# Exact p35 causal Q/K anchor search

This probe tests whether the static medoid `L21/G6` is the wrong addressing
anchor for the tentative p35 redirect `L19/G4 -> L21/G6`.

It does not average Q or K.  Each candidate copies the complete Q+K affine
pair from one active real p35 survivor into the tentative `L21/G6` slot.  V and
O remain the ordinary redirect incumbent.  An optional second control fits the
existing survivor O block with a closed-form ridge solve on calibration rows;
it does not allocate parameters.

The executable is
`experiments/run_qwen_residual_attention_qk_anchor_search.py`.  Its reusable
edit and assignment audit live in
`src/neural_forge/hf_residual_attention_qk_anchor.py`.

## What is actually ranked

Every candidate is executed at all three logical sites that consume the
tentative `L21/G6` survivor in this lineage:

- `L2/G1`
- `L19/G4`
- `L21/G6`

The code derives those physical execution coordinates through each layer's
canonical-to-physical `group_permutations`; it does not assume assignment axis
equals runtime group.  It also records Q/K role scales and the exact real
`q_norm`/`k_norm` weights and epsilon used by each logical layer.  Qwen3 norms
are layer modules, not prototype-owned group tensors, so a shared real Q/K
prototype is deliberately evaluated through each destination layer's existing
norm semantics.  The cached trace observes Q/K after those norms and RoPE.

Calibration and held-out prompt rows must be disjoint. Calibration forced rows
may feed the optional O ridge. Candidate selection then uses calibration output
function only, in this order:

1. fixed-parent-action top-1 agreement;
2. parent-action mean log probability, top-k retention, and rank;
3. independent greedy exact-token, longest-prefix, and full-rollout retention;
4. greedy repetition and longest-run excess;
5. anchor ID only as a deterministic final tie break.

Selection fails closed unless every forced calibration driver is the parent's
actual top-1 action. Held-out function receives a separate evaluation ordering
and names no winner. The prior held-out attention JS, score-geometry,
residual/cache transition, action-margin, top-k, and risk fields remain in each
anchor report as internal diagnostics, sorted by anchor ID for presentation;
they do not feed selection. Static weight distance also remains excluded.

## Bounded exact-p35 invocation

The following nine-candidate neighborhood is explicit and every coordinate is
an active real survivor in the preserved 280-prototype p35 manifest.  It keeps
the incumbent plus adjacent-depth/same-group and nearby addressing controls.

```bash
python experiments/run_qwen_residual_attention_qk_anchor_search.py \
  --codebook-checkpoint /mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child \
  --tokenizer /home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --qk-anchor 21:6 \
  --qk-anchor 18:4 \
  --qk-anchor 20:4 \
  --qk-anchor 22:4 \
  --qk-anchor 19:3 \
  --qk-anchor 19:5 \
  --qk-anchor 18:6 \
  --qk-anchor 20:6 \
  --qk-anchor 22:6 \
  --calibration-prompts-json /path/to/calibration.json \
  --heldout-prompts-json /path/to/heldout.json \
  --horizon 8 \
  --max-anchors 9 \
  --max-prompts-per-split 4 \
  --device cuda:0 \
  --dtype float32 \
  --output /path/to/p35_qk_anchor_search.json
```

Use the process-visible CUDA ordinal established by the isolated runtime; do
not infer it from host `nvidia-smi` numbering.

## Fail-closed boundaries

- Exact mode requires 36 logical layers, eight KV groups, and 280 p35
  prototypes before the redirect, then proves that the proposed source set is
  exactly the same set minus `L19/G4` (280 -> 279).
- Removed or synthetic anchors are rejected before parent traces run.
- The live tokenizer's complete token-to-ID vocabulary and special IDs must
  match the resolved tokenizer contract already bound to the checkpoint.
- Q and K must come from the same unique active source slot, and the assignment
  audit independently re-hashes the installed destination pair against that
  named anchor; V, assignments,
  scales, permutations, and all unrelated Q/K slots remain byte-exact.
- Any installation-time validation failure immediately restores the tentative
  registry before raising; rollback remains retryable after unrelated tamper is
  repaired.
- All shared destination assignments must match the exact expected p35 set.
- Candidate Q/K and optional O edits are restored after every candidate, then
  the original p35 installation is restored exactly.
- No optimizer, backward pass, checkpoint save, fold commit, or broad-suite
  certification occurs.

A local winner is evidence for the next Q/K+V/O transaction, not proof of a
deployable compact child.  Promotion still requires a saved/reloaded
parent-free child and the broad behavior/generation suite.

## Completed nine-anchor result

The strict FP32 horizon-4 P40 run completed in about 71 minutes:

`runs/residual-attention-20260711/p35_L19G4_to_L21G6_qk_anchor_search_a9_h4_p1_fp32_p40_gpu3.json`

Artifact SHA-256:
`d74abdee85f5164514fa4ecbaf4bc6997c568532dcff0bcc2e9d2974ebddf7df`.
Report SHA-256:
`aa7dc6af4b34bb4f835c53da550b8459c34ba1194f90cb997a2028bee65e878f`.

All nine anchors retained the parent's four forced actions and produced the
same four-token greedy path on the single held-out prompt. This is useful
causal evidence but not broad behavior proof. The unchanged redirect-incumbent
Q/K (`L21/G6`) had the lowest raw held-out causal risk, `0.859123008`. Every
alternate real Q/K anchor was worse by `10.588%` through `13.793%`; four also
lost parent actions on calibration. Attention-JS damage rose from `0.455473`
for the incumbent to `0.874702`-`0.987003` for alternate anchors. V was
unchanged, so cache damage remained approximately `1.998` for every candidate.

The result therefore contains no new Q/K repair. Its internal ranking's first
entry is simply the Q/K already used by the tentative fold, and every
promotion flag is false. Exact 280 -> 279 tentative accounting and rollback
passed; no checkpoint was saved or committed. Discrete transplantation of a
different real Q/K pair is closed for this redirect. The next method must fit
a context-conditioned or assignment-specific functional correction rather
than select another raw Q/K tensor.

The local cache/activation evidence is diagnostic, not a veto or ranking key.
If held-out function disagrees with the calibration-selected winner, preserve
both orderings and do not rename the winner after seeing heldout. A later broad,
predeclared promotion suite—not this search report—must decide whether a saved
child is acceptable.
