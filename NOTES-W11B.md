# NOTES-W11B — chimera-omni folds are planned, ledgered transactions now

Proof: `tests/test_chimera_omni_causal_fold_execution.py` (14 passed, 1 dormant
skip), contracts: `tests/test_chimera_omni_native_fold_contract.py` (11 passed).
Predecessor: `NOTES-W10A.md` + `docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`.
Kernel contract: `docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md` §1.3, §2.

## What was actually wrong

`plan_dynamic_fold` returned `ready = false` for a chimera-omni parent → child
pair and the W10A 4x fold ran as driver code with no ledger record. Two causes,
and only one of them was the gate everybody looked at:

1. `_family_neutral_causal_layer_contract` demanded
   `inspector == "hf-causal-decoder"` **and** `position_policy == "rope"` on
   both graphs.
2. Even with that gate open, **nothing else fit**. chimera's layer bundle is
   eight nodes, not six (a `block_attnres` depth selector precedes each branch);
   its ordered-branch labels are `operator`/`feed_forward`, not
   `attention`/`mlp`; its attention output role is `out_weight`, which
   `_projection_role_contract` did not recognize; its MLP owns a fused
   `[2I, H]` tensor, which `_gated_mlp_contract` cannot see; and every global
   interface changes width, which only `copy_exact` was offered for.

So this was never one boolean. It was five contracts.

## The relaxation, and why it is not the thing W10A warned against

W10A and the coverage doc both say: **do not let chimera through the RoPE
gate**, because `CausalGQAAttentionBranchOperator` calls `build_decoder_layout`
and reads `HFCausalCircuitEvidence`, neither of which chimera can produce. That
advice is still correct and is still enforced. `_causal_layer_position_transport`
now names *two* transports and they select **different executables**:

| transport | requires | attention method | MLP method |
|---|---|---|---|
| `rope` | both graphs `hf-causal-decoder` + `rope` | `causal_gqa_rope_residual_attention_layer` | `causal_swiglu_residual_mlp_layer` |
| `nope` | both graphs the *same* native decoder inspector + `nope` | `causal_windowed_nope_attention_layer` | `causal_fused_gate_up_mlp_layer` |

A mixed pair returns `None`. There is no evidence anywhere that says what a
rotated parent's key would have been at an unrotated child's position, so the
crossing keeps its gap instead of getting a plausible answer.

Every NoPE acceptance is checked node by node, not by metadata alone
(`_windowed_nope_attention_contract`): split Q/K/V/O, bias-free, causal,
`position_policy = "nope"` on the operator node, no `rope_theta`/`rope_type`/
`rope_scaling`/`partial_rotary_factor`/`rotary_dim` anywhere, the declared
kernel family is `causal_sliding_window_split_qkv_attention`, and the parent
and child windows are **equal**. A window is a contract constant: changing which
keys exist is a different operator, not a fold coordinate.

The fused MLP contract (`_fused_gate_up_mlp_contract`) requires the node to
*declare* its split (`gate_up_split = "chunk_two_on_last_dimension"`,
`projection_layout = "fused_gate_up_then_down"`, `channel_mixer = "dense"`,
SiLU, bias-free). `situ` — chimera's tanh-capped gate — fails closed; the
SwiGLU solve cannot represent it. A parent/child pair that disagrees on the
layout (one fused, one split) also fails closed: both layouts fold, but no
registered operator crosses between them in one transaction.

## Wiring against the parallel kernel wave

The window/NoPE policy fields landed on `HFCausalGQATopology`
(`attention_window`, `position_encoding`), not on the compile signature, so the
operator binds by inspecting the **dataclass fields** and threading the policy
through both topologies. Two consequences worth knowing:

* The band is the kernel's own mask and it matches chimera exactly
  (`0 <= t - s < window`, counting the query's own position). The banded proof
  (`local_window = 8`, sequence 24 — the band really binds) folds through it and
  beats the no-branch baseline.
* The fallback path is still in the module and still correct: with no
  `position_encoding` field, a NoPE branch runs through the rotary kernel at
  **position zero for every token**, which is the identity map exactly
  (`cos 0 = 1`, `sin 0 = 0` in IEEE double). The operator *proves* that
  numerically on the real head width before it fits anything and seals
  `nope_identity_transport_verified`. Without the band flag it additionally
  requires the sealed evidence to prove the band never binds
  (`window >= sequence_length`) and refuses otherwise, naming the missing field.
  The `xfail-until` marker the brief allowed was removed once the flags landed;
  the refusal test survives as a dormant `skipif` guard.

## What the proof shows

`tests/test_chimera_omni_causal_fold_execution.py`, real
`chimera_omni.modeling.ChimeraOmniModel`, hidden 128 → 32, `local` + dense
blocks, `residual_mode = "block_attnres"`, 12 × 24 tokens plus 16 media
segments:

* **`plan.ready is True`, zero gaps**, 24 actions: 4 ×
  `causal_windowed_nope_attention_layer`, 4 ×
  `causal_fused_gate_up_mlp_layer`, 14 × `native_residual_chart_interface`,
  2 × `copy_exact` (the two parameter-free wiring nodes). Every child tensor is
  owned by exactly one transaction and no tensor is claimed twice. Parent
  unique parameters 1,104,128 → 276,032, **exactly 4x**.
* **`DynamicFoldExecutor` runs it end to end**: `status = "complete"`, 0 parent
  root forwards, 0 parent module forwards, 0 optimizer constructions/steps, 0
  backward calls, 0 autograd-grad calls, 0 pruning/quantization operations,
  `executed_fold_static_zero_training_audit_passed is True`.
* **Rollback is exact**: rejecting the first attention branch undoes that branch
  *and* every interface transaction accepted before it; the child returns to the
  blank state tensor for tensor, not just by fingerprint.
* **Held-out endpoints beat the no-branch baseline** on tokens no solve saw:
  attention 0.0501 / 0.1113 / 0.0886 / 0.0743, fused MLP 0.2117 / 0.1667 /
  0.3019 / 0.3033 relative RMSE against the chart-projected residual write
  (writing nothing scores exactly 1.0). The operators *enforce* that bound and
  roll back rather than install a regression.
* **The receipt is the child.** `test_the_installed_child_reproduces_every_
  reported_branch_endpoint` recaptures the installed child and recomputes every
  branch endpoint from its real forward; each matches the reported number to
  1e-6 relative. If the class the kernel replayed and the class the child runs
  differed by an off-by-one in the band, this test would fail.
* Next-token top-1 agreement with the parent **93.4%** against a blank child's
  **0.0%**. The parent is randomly initialised, so this is a statement about
  mechanism, not capability.

## Three things I changed my mind about while building it

1. **A two-way fit/validation split was not enough.** The fused-MLP solve
   expands 32 residual channels into 320 collinear features; at ridge 1e-6 the
   fit error was 1e-4 and the held-out error was **8.4** — a branch far worse
   than writing nothing, installed without complaint. The bundle now carries a
   third frozen partition and the branch operators pick the ridge from a sealed
   ladder on the *selection* rows only. Validation is reported, never selected
   on. Same evidence, same solver: MLP endpoints went from 1.9–8.4 to 0.17–0.30.
   Every one of the four layers selected ridge 1.0 for that solve, and the
   whole ladder is in the receipt so the choice can be re-read.
2. **`index % period` is not a neutral split.** With `period | sequence_length`
   it assigns whole token *positions* to one partition, and on a structured
   batch (chimera's tiny-proof rows are special tokens, then media placeholders,
   then padding) the held-out rows become a different distribution rather than a
   held-out sample. The lm_head endpoint read 1.86 for that reason alone. The
   split is now a multiplicative hash of the token index — still deterministic,
   still integer-only, no longer resonant with the sequence axis — and the same
   head reads 0.77.
3. **The depth-selector capture does not need the live-list trick.** W10A
   snapshotted `completed_blocks` because the trunk keeps appending to it after
   the call returns. Hooking the selector's *internal* norm instead gives the
   materialized `values` stack in the module's own order, so the parent and the
   child are paired by construction and there is no container to snapshot.

## Named gaps, preserved and sharpened

Uncovered layers no longer collapse into one generic
`cross_family_closed_loop_layer_operator`. The planner reads the IR's own
`fold_kernel_family_required` and emits one gap per missing family:

| declared family | reported capability |
|---|---|
| `bounded_gated_delta_recurrence` | `bounded_gated_delta_recurrence_layer_operator` |
| `compressed_summary_attention` | `compressed_summary_attention_layer_operator` |
| `compressed_latent_attention` | `compressed_latent_kv_attention_layer_operator` |
| `stable_latent_moe_router` / `_expert` | `stable_latent_moe_channel_mixer_operator` |

Proven in the contract suite: a `kda/local/csa/hca` pattern gaps by the
recurrence *and* summary names while its one `local` layer still folds; a
k3-aligned `kda/gated_mla` + `stable_latent_moe` model gaps by all three.
A window change and a `situ` gate both fail closed.

## For whoever wires csa/hca (read before touching the planner)

The kernel wave reports `hf_causal_sparse_latent_fold.py` treats
`SparseLatentAttention` as **one composite attention node with one residual
write**, while `ChimeraOmniCircuitInspector` currently emits **two `ATTENTION`
nodes plus a `RESIDUAL_ADD`** linked by `compound_parent`/`compound_branch`.

Resolution: **fix the inspector, not the contract.** The family-neutral branch
contract admits exactly one norm, one operator and one ordered residual write
per branch, and the design doc chose the composite node deliberately
(`KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`, Trade-offs: "two nodes break
the one-operator-per-branch contract and falsely suggest two independent
residual writes"). Contracting against the two-node emission would encode a
residual write the model does not perform.

I did **not** wire csa/hca in either shape — they stay named gaps — so nothing
here has to be unwound. The gap mapping is keyed on
`fold_kernel_family_required`, which survives the node-shape change unchanged.
`dynamic_model_inspector.py` is outside my ownership; this is the spec, not the
edit.

## Honest gaps in this work

* **One global chart, not a per-boundary chart set.** Transition defects are
  zero by construction; cycle and norm-commutation defects are live and
  unmeasured here. `ResidualBoundaryChartSet` still authenticates against a
  sealed `HFCausalCircuitEvidence` bundle a native trunk cannot produce
  (coverage doc §3.3.1). The chart report says so in its own text.
* **The evidence bundle is module-boundary, not per-family.** It captures what
  a module's inputs and outputs *are*; it does not capture recurrence state
  snapshots, retention rows or per-sub-branch deltas. It is not, and does not
  claim to be, the `hf_chimera_circuit_evidence.py` bundle the KDA/CSA/HCA
  design reserves. When that lands, this stays the family-neutral lane
  underneath it — that is a merge point, and someone should do the merge rather
  than leave two capturers.
* **The parent is untrained.** This file proves planning, transaction, rollback
  and audit. Capability retention is what W10A's trained tiny-proof run measured
  and what whole-response review decides.
* **The even-head-dim gate is conservative.** The kernel now admits an odd head
  dimension under NoPE (there are no rotation pairs to preserve); the planner
  contract still requires an even one. That can only produce a gap, never a
  false accept, and every chimera head dim is even — but it is stricter than the
  executable and should be relaxed deliberately, not forgotten.
* **Interface heads use the base ridge, not the ladder.** Their solves have
  child-residual-width columns against far more rows and no nonlinear expansion,
  so they are well posed; the two branch solves are the ones that needed
  selection. If a head ever reads a wider intermediate, it will need the ladder.
* **The window envelope.** When the band is threaded into the kernel mask this
  is exact. In the fallback (`inactive_on_sealed_evidence`) the fit is only
  valid where the band does not bind, and the report says so in its own field
  rather than in a comment.
* **Depth reduction is still untouched**, and `block_attnres` depth transport is
  still pointwise (a diagonal norm refit plus a ridge-fit pseudo-query per
  selector), not a joint treatment of the checkpoint resets and the softmax mix.

## Files

| path | what |
|---|---|
| `src/neural_forge/dynamic_fold_planner.py` | two position transports, the depth-selector-aware branch contract, the windowed-NoPE and fused-gate/up contracts, the chart-projected global interface arm, per-family named gaps |
| `src/neural_forge/dynamic_causal_operator_promotion.py` | `NativeWindowedNoPEAttentionBranchOperator`, `NativeFusedGatedMLPBranchOperator`, `NativeResidualChartInterfaceOperator`, the shared chart preamble, ridge selection, the no-branch endpoint gate |
| `src/neural_forge/dynamic_fold_registry.py` | three `EXPERIMENTAL` method specs on their own lanes |
| `src/neural_forge/dynamic_native_circuit_evidence.py` | **new** — sealed module-boundary evidence, live child recapture, the global residual chart |
| `tests/test_chimera_omni_native_fold_contract.py` | **new** — 11 planner contract and fail-closed cases |
| `tests/test_chimera_omni_causal_fold_execution.py` | **new** — 14 execution, rollback, audit and endpoint-reproduction cases |
