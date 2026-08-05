# NOTES-W12F — every chimera-omni block family, sequence mixer and channel mixer, now folds as a planned ledgered transaction

Proof: `tests/test_chimera_omni_causal_fold_execution.py` (40 passed, 1 dormant
skip), kernels: `tests/test_hf_causal_latent_kv_attention_fold.py` (18 passed)
and `tests/test_hf_causal_stable_latent_moe_fold.py` (16 passed), contracts:
`tests/test_chimera_omni_native_fold_contract.py` (28 passed), IR:
`tests/test_dynamic_chimera_omni_inspector.py` (18 passed).
Predecessors: `NOTES-W12C.md` (KDA/CSA/HCA), `NOTES-W11B.md` (the planner lane
and the composite-node spec).
Design: `docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`.

W12C closed four of chimera-omni's five sequence-mixer families and left two
*named* gaps: `compressed_latent_attention` (`gated_mla`) and
`stable_latent_moe_router`/`_expert`. Those two are the families of the
k3-aligned reference configuration, so until both closed, the architecture
chimera-omni actually targets could not be folded at all. Both are closed here.

## 1. What the two modules are, read off the executable

**`GatedMultiHeadLatentAttention`** (`layers.py:231-306`) is **not** DeepSeek
MLA. The architecture is NoPE end to end, and `mla_extra_head_dim` is not a
decoupled *rotary* channel set: it is a shared, head-independent key block taken
straight off `kv_down`'s output and broadcast to every head
(`layers.py:290-294`). Assuming otherwise would have invented a rotation the
module does not have. Two structural facts drive the kernel:

* the key side factors through the latent exactly —
  `q_base_h(t)ᵀ W_k_h n(latent_s) = (W_k_hᵀ q_base_h(t))ᵀ n(latent_s)`, so the
  per-head key contribution to any score has rank ≤ `kv_latent_size`
  independently of `key_head_dim`;
* the shared block adds a *rank-`extra_head_dim`* head-independent term to every
  head's score, and its share of the score energy is measurable exactly from
  cached rows. Measured **0.385** (layer 1) and **0.512** (layer 3) on the
  proof crossing, so folding it as a head-independent block is not a vacuous
  claim.

**`StableLatentMoEFeedForward`** (`layers.py:362-445`) routes each token to
`experts_per_token` of `num_experts` latent experts, accumulates their weighted
outputs, normalizes, projects back, and adds one or more full-width shared
experts. `router_bias` is a **non-gradient buffer** that enters *selection*
only; the mixture weights are the unbiased sigmoid scores renormalized over the
selected set.

## 2. The inspector: one composite node for the MoE

`ChimeraOmniCircuitInspector._channel_mixer_nodes` emitted the MoE as a router
`INTERFACE_PROJECTION`, a latent-down projection, one `GATED_MLP` per routed
expert, a `RESIDUAL_ADD` combining them, a `NORM`, a latent-up projection, one
`GATED_MLP` per shared expert and a second `RESIDUAL_ADD`. On the k3-aligned
config that is **16 nodes** where the module makes **one** residual write.

| before | after |
|---|---|
| `layer.N.feed_forward.router` (`INTERFACE_PROJECTION`) | — |
| `layer.N.feed_forward.latent_down` (`INTERFACE_PROJECTION`) | — |
| `layer.N.feed_forward.routed_expert.{000..E}` (`GATED_MLP`) | — |
| `layer.N.feed_forward.routed_combine` (`RESIDUAL_ADD`) | — |
| `layer.N.feed_forward.routed_norm` (`NORM`) | — |
| `layer.N.feed_forward.latent_up` (`INTERFACE_PROJECTION`) | — |
| `layer.N.feed_forward.shared_expert.{000..S}` (`GATED_MLP`) | — |
| `layer.N.feed_forward.combine` (`RESIDUAL_ADD`, `ordered_branch=feed_forward`) | — |
| | `layer.N.feed_forward` (`GATED_MLP`, owns **every** tensor under `blocks.N.ffn`, one `DATA` edge from the norm, one `RESIDUAL` edge into `layer.N.feed_forward_residual`) |

This is W12C's resolution (a) applied on the channel axis, for exactly the same
reason: the old emission encoded two residual adds the module never performs and
could not satisfy the family-neutral one-norm / one-operator / one-ordered-write
branch contract (`dynamic_fold_planner.py:1026`). The sub-branch structure moved
onto attributes of the one operator: `composite_operator`,
`composite_sub_branches = "top_k_routed_latent_experts,full_width_shared_experts"`,
`composite_residual_writes = 1`, `expert_count`, `experts_per_token`,
`shared_expert_count`, `expert_latent_size`, `routed_intermediate_size`,
`router_score_activation`, `routed_weight_normalization`,
`router_bias_is_non_gradient_buffer`, `situ_gate_beta`, `situ_up_beta`.

**`gated_mla` needed no inspector change.** Its node already declared every
attribute the kernel reads and already made one ordered residual write.

`_CHIMERA_STRUCTURALLY_COVERABLE_MIXERS` grew to
`{local, csa, hca, kda, gated_mla}`, so `kernel_gap_block_kinds` on a k3-aligned
graph is now `""`. The dense channel mixer's node is untouched, asserted in
`test_the_stable_latent_moe_is_one_composite_channel_mixer_circuit`.

## 3. Two kernels

### 3.1 `hf_causal_latent_kv_attention_fold.py`

Two evidence gates, both run **once, before the ridge ladder**:

| gate | what it compares | measured (proof crossing) |
|---|---|---|
| `readout_identity` | `kv_up` recomputed from the **captured** normalized latent and the carried weight, against the **captured** `kv_up` output | **9.07e-08** / **8.49e-08** |
| `delta_reconstruction` | `out(sigmoid(gate) · softmax(scores) v)` rebuilt from the derived split, broadcast, mask and gate, against the **captured** branch delta | **1.71e-07** / **1.75e-07** |

On a **float64** parent module (`tests/test_hf_causal_latent_kv_attention_fold.py`)
the readout identity measures **< 1e-12** — it is exact linear algebra, not a
rounding claim. The delta reconstruction cannot: `forward` calls `.float()` on
its scores and context (`layers.py:301`, `:304`) whatever the module dtype, so
that gate measures the parent's own float32 downcast, and the test says so.

Recipe: barycentric complete head assignment, energy-ranked key/value channel
selection on the merged targets, POD latent bases (identity when the width does
not change — asserted), `kv_down`/`kv_up` and `q_down`/`q_up` ridges with the
child's own RMSNorms *replayed* rather than fitted through, installed-probability
replay with an overlap floor, a gate ridge against the captured **pre-sigmoid**
boundary (so this family needs no inverse-sigmoid clamp policy at all), and an
endpoint-coupled writer. **No score chart is taken or accepted**: unlike CSA's
full-width summary score, every MLA score is an inner product in head
coordinates, so a residual-width crossing needs nothing but a chart-projected
endpoint target.

### 3.2 `hf_causal_stable_latent_moe_fold.py`

Two evidence gates, again ridge-independent:

| gate | what it compares | measured |
|---|---|---|
| `routing_identity` | recovered row sets against the **captured per-expert row counts** (an integer equality, checked first), recovered rows against the captured expert inputs, and the rebuilt accumulator against the captured `routed_norm` input | rows **exact**, inputs **0.0**, accumulator **6.2e-08 … 6.7e-08** |
| `channel_split` | `routed_delta + Σ shared_delta` against the captured branch write, on three independently captured values | **3.61e-08 … 3.69e-08** |

On a float64 parent all three measure **< 1e-12**
(`tests/test_hf_causal_stable_latent_moe_fold.py`); nothing in this module
downcasts.

**The sparse-routing row budget is the answer to "which evidence rows does each
expert's solve consume".** Expert `j`'s three solves consume exactly
`rows_j & fit` — the rows the *parent* routed there. A sealed policy carries an
absolute floor (`minimum_expert_fit_rows`) and a relative one
(`minimum_expert_row_fraction` of `N·K/E`); an expert under either makes the
whole channel mixer refuse, naming the expert, its count, the router's share and
the floor. It is not a warning: an expert fitted on a handful of rows is an
interpolation of noise the endpoint gate cannot see, because it is only ever
exercised on those same rare rows. Measured per-expert row counts on the proof
crossing are strikingly unbalanced — `[72, 171, 83, 250]`, `[234, 95, 187, 60]`
— which is exactly why the budget is a gate rather than an assumption.

The experts are fitted on the **parent's** assignment (the captured evidence
belongs to it); the child replays its own. The disagreement is measured, not
assumed away: `router_top_k_agreement` **0.882 – 0.962** across the four layers,
with a sealed floor.

`fit_functional_swiglu` was not reused: it hardcodes SiLU, and the k3-aligned
reference runs `activation_kind = "situ"`. The kernel implements its own
activation-parameterized gated dictionary fit (same ridge-then-pivot-then-writer
rule) so the closure covers both activations. Both are unit-tested.

## 4. Wiring

| what | where |
|---|---|
| `_compressed_latent_kv_attention_contract` | `dynamic_fold_planner.py:1456` |
| `_stable_latent_moe_channel_mixer_contract` | `dynamic_fold_planner.py:1523` |
| attention branch arm admits the latent-KV contract | `:1062-1082` |
| mlp branch arm admits the composite MoE contract | `:1090-1096` |
| `_native_channel_mixer_branch` (the channel-mixer dispatch) | `:1774` |
| per-family width/constant tables (sequence **and** channel) | `:1690`, `:1727` |
| `causal_latent_kv_attention_layer`, `causal_stable_latent_moe_channel_mixer_layer` | `dynamic_fold_registry.py:1527` |
| `CompressedLatentKVAttentionLayerOperator` | `dynamic_causal_operator_promotion.py:4746` |
| `StableLatentMoEChannelMixerLayerOperator` | `dynamic_causal_operator_promotion.py:5135` |

Both operators sit on the shared `_ChimeraMixerBranchOperator` preamble, so they
inherit the chart authentication, the depth-selector fold, the ridge ladder
discipline, the 1.0 no-branch endpoint gate and the kernel-refusal recording the
W12C pair already use. Both assemble their family evidence record from the
**sealed module-boundary bundle** and never call a parent module:
`observed_parent_module_forward_calls == 0` is asserted on the executed run.

`dynamic_native_circuit_evidence.py` needed **no change at all**. W12C's additive
intermediate-module rule already captures `blocks.N.ffn.routed_experts.{j}` and
`blocks.N.ffn.shared_experts.{i}`, which is the only boundary at which a routed
expert's own rows exist — and, because a routed expert is only invoked with the
rows the router selected, that capture *is* the ragged per-expert row block.

Per family, what may contract and what may not:

| family | contraction coordinates | contract constants |
|---|---|---|
| `compressed_latent_attention` | query_heads, key_head_dim, value_head_dim, kv_latent_size (+ query_latent_size, checked in the family arm) | **extra_head_dim**, layer_type, sliding_window |
| `stable_latent_moe_channel_mixer` | intermediate_size, expert_latent_size, routed_intermediate_size | expert_count, experts_per_token, shared_expert_count, activation, situ betas, router_score_activation, routed_weight_normalization, **router_bias** |

### Neither family gets a second method id

`gated_mla`'s dense-query (`mla_query_latent_size == 0`) and compressed-query
variants are the *same class* under one config constant — the csa/hca situation
exactly — so the latent size is a sealed contract *shape* (both zero or both
positive; when positive it is a contraction coordinate) rather than a second
executable. Both variants are unit-tested through the one kernel.

The MoE gets one id, not a router id plus an expert id, because the module makes
one ordered residual write and is therefore one operator. The planner's two old
gap family names (`stable_latent_moe_router`, `stable_latent_moe_expert`)
collapse into one, `stable_latent_moe_channel_mixer`, mapping to the same
capability string the planner already reported.

### `router_bias`: a contract constant, not a folded tensor

It is a buffer, so it is not in `named_parameters`, not in the IR, not owned by
any transaction, and not inside the snapshot/rollback envelope every operator
here relies on. It moves the child's expert-selection boundary. The operator
therefore **requires equality** and refuses otherwise, before any mutation,
naming `blocks.N.ffn.router_bias` and saying it is not a learned tensor.
Exercised in `test_a_router_bias_the_child_does_not_share_fails_closed`.

## 5. The proof

`tests/test_chimera_omni_causal_fold_execution.py`, real
`chimera_omni.modeling.ChimeraOmniModel`, `block_pattern =
("local", "gated_mla", "kda", "gated_mla")`, every channel mixer a Stable
LatentMoE (`moe_first_dense_layers = 0`), `activation_kind = "situ"`,
`residual_mode = "block_attnres"`, 12 × 24 tokens plus 16 media segments.
Two families this wave closed, two the last wave closed, in one trunk.

### Exact 4x: 128 → 31, and the arithmetic that forces it

This topology has **no `[hidden, hidden]` tensor anywhere** (no csa/hca) and
every latent width is pinned as an explicit constant rather than derived from
`hidden_size`, so the unique parameter count is exactly **affine** in the
residual width. Measured at four widths and asserted
(`test_the_latent_topology_is_affine_in_the_residual_width`):

```
total(H) = 13299 * H + 17732
```

`total(H_p) = 4·total(H_c)` therefore reduces to
`13299 · (H_p − 4·H_c) = 3 · 17732 = 53196`, and `53196 / 13299 = 4` exactly, so

```
H_p = 4·H_c + 4
```

At `H_c = 31` that is `H_p = 128`:

```
parent  13299 * 128 + 17732 = 1_720_004
child   13299 *  31 + 17732 =   430_001   ->  exactly 4x, asserted
```

The unround vocabulary (1459) and the MoE widths are what make the coefficient
divide three times the constant. They are the *solution*, not a rounding of one:
the offset `3γ/α` must be a positive integer, and `α = 3·vocab + 24·intermediate
+ A₀` is what makes it so. (Contrast W12C, where csa/hca's four `[H, H]` summary
projections put a genuine `H²` term in the count and forced `128 → 34`.)

### Plan

`plan.ready is True`, **zero gaps**, 24 actions:

| method | count |
|---|---|
| `causal_windowed_nope_attention_layer` | 1 |
| `causal_kda_recurrent_layer` | 1 |
| `causal_latent_kv_attention_layer` | 2 |
| `causal_stable_latent_moe_channel_mixer_layer` | 4 |
| `native_residual_chart_interface` | 14 |
| `copy_exact` | 2 |

Every child tensor is owned by exactly one transaction and no tensor is claimed
twice. One MoE transaction carries the router, both latent projections, all four
routed experts, the accumulator norm and both shared experts, because they are
one operator.

### Execution

`DynamicFoldExecutor`: `status = "complete"`, 0 parent root forwards, **0 parent
module forwards**, 0 optimizer constructions/steps, 0 backward calls, 0
autograd-grad calls, 0 pruning/quantization operations,
`executed_fold_static_zero_training_audit_passed is True`.

### Per-branch fold numbers (held-out tokens, chart-projected target, no-branch baseline = 1.0)

| layer | family | method | ridge | endpoint | BF16 endpoint |
|---|---|---|---|---|---|
| 0 | local | `causal_windowed_nope_attention_layer` | 1e-2 | **0.0651** | — |
| 1 | gated_mla | `causal_latent_kv_attention_layer` | 1e-2 | **0.1632** | 0.1633 |
| 2 | kda | `causal_kda_recurrent_layer` | 10.0 | **0.4461** | 0.4465 |
| 3 | gated_mla | `causal_latent_kv_attention_layer` | 0.1 | **0.2418** | 0.2420 |
| 0 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 1.0 | **0.3475** | 0.3501 |
| 1 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 0.1 | **0.5364** | 0.5357 |
| 2 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 1.0 | **0.5127** | 0.5067 |
| 3 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 1.0 | **0.4845** | 0.4783 |

gated-MLA diagnostics: readout identity **9.07e-08** / **8.49e-08**, delta
reconstruction **1.71e-07** / **1.75e-07**, shared-key score energy fraction
**0.385** / **0.512**, probability overlap **0.9696** / **0.9614**, gate
saturation **0.0** on both.

Stable LatentMoE diagnostics, per layer:

| layer | routing identity | split | router top-k agreement | expert rows | expert fit rows | routed / shared energy |
|---|---|---|---|---|---|---|
| 0 | 6.21e-08 | 3.63e-08 | 0.9618 | 72/171/83/250 | 34/88/44/130 | 0.941 / 0.054 |
| 1 | 6.18e-08 | 3.61e-08 | 0.9201 | 170/122/169/115 | 88/62/84/62 | 0.925 / 0.062 |
| 2 | 6.61e-08 | 3.67e-08 | 0.9132 | 122/73/198/183 | 64/36/103/93 | 0.917 / 0.063 |
| 3 | 6.65e-08 | 3.69e-08 | 0.8819 | 234/95/187/60 | 118/52/99/27 | 0.960 / 0.061 |

`expert_input_relative_rmse` is **exactly 0.0** on all four: the rows recovered
from the captured router output are bit-identical to the rows the parent's
expert boundary actually received.

### The receipt is the child

`test_the_installed_child_reproduces_every_latent_branch_endpoint` recaptures the
installed child and recomputes all **eight** branch endpoints from its real
forward; every one matches its receipt to better than **1e-05** relative. That
is what makes the MoE honest rather than plausible: a top-`k` router is a
*discrete* decision, so if the installed router disagreed with the one the kernel
replayed on a single row, that row would run through different experts and the
recomputed endpoint would not match.

### Rollback, refusals, agreement

* Rejecting the first MoE branch undoes it **and every transaction accepted
  before it**; the child returns to blank tensor for tensor.
* A tightened probability-overlap floor (MLA) and a tightened router-agreement
  floor (MoE) make each kernel refuse on **every** rung; the operator surfaces
  it as its own failure carrying the kernel's measurement, and the child is
  bit-identical afterwards.
* The **sparse expert row budget** is exercised the same way: a floor above the
  achievable count refuses the whole channel mixer on every rung, naming the
  expert.
* Mutating the parent's `router_bias` fails closed before any mutation, as does
  mutating the parent's `extra_head_dim` so the live module disagrees with the
  IR.
* The MoE kernel's **no-branch floor is not hypothetical**: on this crossing it
  refuses the three smallest ridge rungs of every MoE layer
  (`kernel_refused_ridge_rungs == 3`, four times), and the ladder records each
  refusal instead of the operator quietly taking it.
* Next-token top-1 agreement with the parent **70.5%** against a blank child's
  **0.0%**. The parent is randomly initialised, so this is a statement about
  mechanism, not capability.

## 6. Suites

```
.venv/Scripts/python.exe -m pytest tests/test_chimera_omni_causal_fold_execution.py -q
→ 40 passed, 1 skipped in 128s

.venv/Scripts/python.exe -m pytest \
  tests/test_chimera_omni_native_fold_contract.py \
  tests/test_dynamic_chimera_omni_inspector.py \
  tests/test_dynamic_model_inspector.py \
  tests/test_dynamic_fold_planner.py \
  tests/test_dynamic_causal_operator_promotion.py \
  tests/test_dynamic_fold_registry.py \
  tests/test_cross_family_causal_fold_execution.py \
  tests/test_cross_width_causal_fold_execution.py \
  tests/test_hf_causal_kda_fold.py \
  tests/test_hf_causal_sparse_latent_fold.py \
  tests/test_hf_causal_windowed_nope_attention_fold.py \
  tests/test_dynamic_fold_executor.py \
  tests/test_hf_causal_latent_kv_attention_fold.py \
  tests/test_hf_causal_stable_latent_moe_fold.py \
  tests/test_run_chimera_omni_toy_fold.py -q
→ 259 passed in 49s

.venv/Scripts/python.exe experiments/run_chimera_omni_toy_fold.py \
  --train-steps 2 --out <scratch>
→ stage "complete"

.venv/Scripts/python.exe -m ruff check <the eleven owned code/test files>
→ All checks passed
```

**Four failures exist outside this lane** in
`tests/test_dynamic_neo_native_vision.py` and
`tests/test_native_{audio,video,points3d}_executable_lane.py`
(`behavior_gate.evaluation_count == 3`, expected 4). That is the *same*
signature `NOTES-W12C.md` section 5 recorded as pre-existing and reproduced in a
clean worktree of its HEAD; the lane belongs to a parallel agent and nothing
here touches it. I could **not** re-verify it in a fresh worktree this time:
the volume has 353 MB free and `git worktree add` failed with
`No space left on device`. So the claim rests on W12C's verification plus the
fact that the four files, their imports and the failing assertion are all
outside this wave's surface — weaker evidence than a clean-tree reproduction,
and said as such.

**The branch moved and is shared.** `HEAD` advanced from `9e82a886` to
`8e08844c` (four LFM2/report commits) during this work, and the working tree
also carries another agent's *uncommitted* edits to
`src/neural_forge/dynamic_closed_loop_fold.py`,
`src/neural_forge/hf_lfm2_dense_budget_model_fold.py` and their two tests. None
of those are touched here, and nothing here is committed.

**One operational note, stated because it was a real mistake.** The runner's
default `--out` is the *committed* report directory
`reports/chimera-omni-fold-readiness-20260804`, so the first
`--train-steps 2` invocation overwrote a sealed artifact. It was restored
byte-exactly from `HEAD` and re-verified against the report manifest's own
digests — a stronger check than `git status`, which reports clean even when
autocrlf has changed the bytes; three `video_manifest.json` files needed the
CRLF form rather than the blob form. `tests/test_run_chimera_omni_toy_fold.py`
is 16/16 green again, and every later invocation used a scratch `--out`.

## 7. Honest gaps in this work

* **The shared-expert sub-branch endpoint exceeds 1.0 on all four MoE layers**
  (1.69 – 2.80) while the *branch* endpoint is 0.35 – 0.54. The bounded
  alternation trades mass between the routed writer and the shared writers to
  serve the shared endpoint, and the shared half carries only ~6% of the delta
  energy, so its own reconstruction is sacrificed. This is the design's own
  reason for reporting sub-branch endpoints separately, and the receipt does;
  but a caller who wants the shared experts to *be* the parent's shared experts
  needs a per-sub-branch acceptance gate this wave does not add.
* **The exact-4x pair contracts only the residual width.** Head merging, latent
  truncation and expert-dictionary contraction are implemented and unit-tested
  (`test_a_contracted_fold_beats_the_no_branch_baseline` in both kernel suites),
  but the trunk-scale proof does not exercise them.
* **The dense (non-MoE) SiTU channel mixer is still refused**, by the existing
  fused gate/up lane, which admits SiLU only. On the *real*
  `k3_aligned_test_config` that leaves exactly one gapped layer — layer 0, the
  `moe_first_dense_layers` layer — while its four MoE layers and both gated-MLA
  layers plan. That is an existing lane's activation envelope, not a chimera
  family gap, and this wave deliberately does not widen it (the task's rule was
  new lanes only). `test_gated_mla_and_stable_latent_moe_now_plan_on_the_k3_reference`
  asserts exactly that, including that the two named capabilities are gone.
* **The gap machinery names that layer by the recurrence family**, because
  `_native_kernel_family_gaps` reports every family a gapped layer *needs*, not
  which half failed. Unchanged heuristic from W12C, and it is now slightly more
  misleading than it was, since the recurrence does have a kernel.
* **The MoE latent-basis / norm commutation is approximate under truncation.**
  `RMSNorm(routed·U) ≠ RMSNorm(routed)·U` for a truncating `U`; the gap is
  absorbed by the diagonal norm fit and reported as its relative RMSE, not
  assumed away. At the proof crossing `U` is the identity, so the effect is not
  exercised there.
* **`router_bias` is not transported.** Equality is required and refused on
  breach. A curator who wants a rebalanced child must set the buffer before the
  fold; that is a caller action, and it is recorded as one.
* **One global chart, not a per-boundary chart set.** Unchanged from W11B:
  transition defects are zero by construction, cycle and norm-commutation
  defects are live and unmeasured.
* **The parent is untrained.** This file proves planning, transaction, rollback
  and audit across four sequence-mixer families and one composite channel
  mixer. Capability retention is what a trained parent and whole-response review
  decide.

## Files

| path | what |
|---|---|
| `docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md` | the design contract this wave implements |
| `src/neural_forge/hf_causal_latent_kv_attention_fold.py` | **new**: the gated-MLA kernel |
| `src/neural_forge/hf_causal_stable_latent_moe_fold.py` | **new**: the Stable LatentMoE kernel |
| `src/neural_forge/dynamic_model_inspector.py` | the MoE channel mixer emits one composite `GATED_MLP` node; SiTU betas in the geometry; `gated_mla` becomes structurally coverable |
| `src/neural_forge/dynamic_fold_planner.py` | two contracts, two branch arms, the channel-mixer dispatch and its width/constant tables, two method ids, two bundle-kind and four coupled-role arms |
| `src/neural_forge/dynamic_causal_operator_promotion.py` | `CompressedLatentKVAttentionLayerOperator`, `StableLatentMoEChannelMixerLayerOperator` |
| `src/neural_forge/dynamic_fold_registry.py` | two `EXPERIMENTAL` method specs on their own family lanes |
| `tests/test_hf_causal_latent_kv_attention_fold.py` | **new**: 18 stage-0/1 kernel cases |
| `tests/test_hf_causal_stable_latent_moe_fold.py` | **new**: 16 stage-0/1 kernel cases |
| `tests/test_dynamic_chimera_omni_inspector.py` | the composite MoE node contract, the updated coverage metadata |
| `tests/test_chimera_omni_native_fold_contract.py` | the k3-reference closure plus nine new planner contract cases |
| `tests/test_chimera_omni_causal_fold_execution.py` | eleven new execution, receipt, refusal and rollback cases for the two families |
