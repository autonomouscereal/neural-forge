# NOTES-W12C — every chimera-omni sequence mixer with a kernel now folds as a planned, ledgered action

Proof: `tests/test_chimera_omni_causal_fold_execution.py` (26 passed, 1 dormant
skip), contracts: `tests/test_chimera_omni_native_fold_contract.py` (19 passed),
IR: `tests/test_dynamic_chimera_omni_inspector.py` (17 passed).
Predecessors: `NOTES-W11A.md` (the kernels), `NOTES-W11B.md` (the planner lane
and the composite-node spec), `docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`
§1.1, §1.2, §4.

W11A built the KDA and CSA/HCA fold kernels and wired nothing. W11B planned the
`local` + fused-MLP pair and left the other two families as *named* gaps, with
one unresolved discrepancy: the inspector emitted CSA/HCA as two `ATTENTION`
nodes plus a `RESIDUAL_ADD`, and the kernel folds it as one branch with one
residual write. This wave settles that and closes the family coverage.

## 1. The inspector composite fix (resolution (a))

`ChimeraOmniCircuitInspector._mixer_nodes` (`dynamic_model_inspector.py:8580`)
now emits **one** `ATTENTION` node per `csa`/`hca` block.

| before | after |
|---|---|
| `layer.N.operator.local_window` (`ATTENTION`, owns `mixer.local.*`) | — |
| `layer.N.operator.compressed_summary` (`ATTENTION`, owns `summary_{q,k,v}`, `out`, via a bespoke by-name parameter resolver) | — |
| `layer.N.operator.combine` (`RESIDUAL_ADD`, `ordered_branch = operator`, 0 learned tensors) | — |
| | `layer.N.operator` (`ATTENTION`, owns **all eight** tensors under `blocks.N.mixer`, one cache edge, one `DATA` edge from the norm, one `RESIDUAL` edge into `layer.N.operator_residual`) |

Three nodes and two intra-branch edges became one node. The sub-branch structure
did not disappear — it moved to where it belongs, onto attributes of the one
operator: `composite_operator`, `composite_sub_branches =
"local_window,compressed_summary"`, `composite_residual_writes = 1`,
`sub_branch_combination`, `summary_stride`, `local_window`, `summary_heads`,
`summary_head_dim`, `summary_projection_style`, `summary_reduction`,
`mixer_family`.

`_chimera_add_node`, `_chimera_named_parameters` and
`_CHIMERA_SPARSE_LATENT_SUMMARY_PARAMETERS` existed only to split one module
across two nodes; all three are deleted, and `_mixer_nodes` no longer takes the
`groups` map it needed for them. **Net −125 lines in the inspector** (48 added, 173 removed).

Why this and not the contract: `SparseLatentAttention.forward` computes both
sub-branches from the same normalized rows and returns `local + out(global_y)`
— one value, written into the residual stream exactly once. The old emission put
a `RESIDUAL_ADD` between them, which encoded a residual write the model does not
perform, and broke the family-neutral one-operator-per-branch contract
(`dynamic_fold_planner.py:773-786`, design §4). This is W11B's resolution (a),
applied.

`_CHIMERA_STRUCTURALLY_COVERABLE_MIXERS` grew from `{local}` to
`{local, csa, hca, kda}`, so `kernel_gap_block_kinds` on a kda/local/csa/hca
graph is now `""` and on a k3-aligned graph is `"gated_mla"`. That metadata is a
claim about coverage and it would have been false otherwise.

**KDA needed no inspector change.** Its node already carries every attribute the
kernel reads — `heads`, `key_head_dim`, `value_head_dim`, `decay_rank`,
`gate_lower_bound`, plus the conv kernel size under the more precise name
`internal_short_conv_kernel_size` (the design calls it `conv_kernel_size`) — and
already declares the `RECURRENT_STATE` ports and the recurrent `STATE` edge. The
planner keys on the IR's names; nothing was renamed to match a document.

## 2. Two operators, and what they read

`dynamic_causal_operator_promotion.py` gains
`BoundedGatedDeltaRecurrenceLayerOperator` (`:3660`) and
`CompressedSummaryAttentionLayerOperator` (`:4171`), both on the shared
`_ChimeraMixerBranchOperator` preamble under `_NativeChartOperator`, so they
inherit the chart authentication, the depth-selector fold, the ridge ladder
discipline and the 1.0 no-branch endpoint gate the W11B operators already use.

The load-bearing decision is **where the family evidence comes from**. The
kernels want `ChimeraKDALayerEvidence` / `ChimeraSummaryAttentionLayerEvidence`,
and the kernels' own capturers build those by re-running parent modules. Doing
that at fold time would have put nonzero parent module forwards into the run,
which is the one number the whole lane exists to keep at zero. So the operators
**assemble the family record from the sealed module-boundary bundle instead**,
and `run.observed_parent_module_forward_calls == 0` is asserted.

That needed one additive change to `dynamic_native_circuit_evidence.py`:
`native_capture_paths` now also hooks **every module strictly between a node's
own module and each of its tensor owners** (`:167-210`). It is family-neutral —
a node whose tensors are all direct children gains nothing, so every HF-shaped
decoder bundle is byte-identical — but for a composite operator it is the only
boundary at which a sub-branch's own value exists. It buys exactly:
`blocks.N.mixer.local` (the CSA local delta) and
`blocks.N.mixer.{q,k,v}_conv` (the KDA post-convolution, post-SiLU rows).

### KDA rows: observed vs. derived, and the gate that separates them

| field | source |
|---|---|
| `pre_conv_{query,key,value}` | captured `q_proj` / `k_proj` / `v_proj` outputs |
| `query_rows` / `key_rows` / `value_rows` | captured `q_conv` / `k_conv` / `v_conv` outputs (+ the parent's own L2 normalization on the two key-side ones) |
| `recurrent_output_rows` | captured **input** of `output_norm` — the only boundary at which `o_t` exists as a value |
| `attention_delta`, `block_input` | captured mixer output, captured norm input |
| `decay_logit_rows`, `retention_rows`, `beta_rows`, `gate_rows` | **derived** from captured projection outputs plus the carried decay parameters |
| `state_snapshots` | **derived**: the parent's own static `delta_rule_update` stepped over those rows (a method call on cached tensors, not a module forward) |

Deriving retention makes the kernel's decay-inverse-chain check vacuous, and
saying so is the point of the `derived_rows` field in the receipt. The gate that
is *not* vacuous is the one that matters: `kda_effective_weights` rebuilds
`o_t = Σ_{s≤t} w_h(t,s) v_s` from the derived retention/beta and the captured
q/k/v, and compares it against the **captured** `recurrent_output_rows`. Measured
**9.31e-08** against a 1e-06 tolerance. A wrong derivation cannot survive that,
and the receipt names it as the consistency gate
(`derived_row_consistency_gate`). The unroll runs **once, before the ridge
ladder**, so an inconsistent cache fails closed with the kernel's own number
rather than being reported as "no good ridge".

### CSA/HCA rows: both sub-branches are independently captured

`local_delta` is the captured output of `blocks.N.mixer.local`; `summary_delta`
is the captured output of `blocks.N.mixer.out`; `attention_delta` is the
captured output of `blocks.N.mixer`. So `split_summary_branch_delta` compares
two independently observed values against a third — it is a real reconstruction
check, not `a + b == a + b`. Measured **2.51e-08** (csa) and **2.47e-08** (hca).
Pooled summaries and both probability tensors are recomputed with the kernel's
own `summary_pooling_matrix` / `summary_attention_probabilities` /
`causal_attention_probabilities`, which the kernel suite asserts are the
parent's operators.

### The one place the kernel's shape forced a decision

`compile_hf_causal_sparse_latent_fold` fits both sub-branches against the
*parent's* sub-branch deltas, because that is where its split gate captured
them; passing a chart-projected `endpoint_target` while the sub-branch targets
stay parent-width is simply unsupported and raises. So this operator lets the
kernel target parent coordinates and then transports the two sub-branch writers
through the chart:

```
context @ (Eᵀ W)ᵀ  ==  (context @ Wᵀ) @ E
```

which is an identity, not a second fit — projecting the writer *is* projecting
the sub-branch's output. Exactly the tensors whose residual-writing side is the
parent width get the producer fold (`local.out.weight`, `out.weight`); anything
whose shape matches neither the child parameter nor that pattern fails closed.
Both facts are sealed: `sub_branch_writers_transported_through_chart`,
`sub_branch_fit_coordinate_system = "parent_residual_width"`,
`endpoint_gate_coordinate_system = "child_residual_width"`. The honest cost is
in the first of those: the sub-branch solves also fit the component the chart
discards, which is suboptimal for the coordinates the child actually writes.
The gate is applied where it matters — the endpoint the operator refuses on is
the child-coordinate one.

KDA has no such problem: its writer is the only map into the residual, so
`endpoint_target` is passed chart-projected directly.

### Kernel refusals are planner-visible

Both kernels enforce the design's 1.0 no-branch floor after a BF16 replay and
raise rather than emit. `_select_over_ridge_ladder` records each rung as either
its selection endpoint **or** the kernel's refusal message, and a ladder on
which every rung refused raises with all of them listed. This is not
hypothetical: on this crossing the unregularized rung really does lose to
writing nothing, and the receipt says so —

```
csa  ridge 1e-06: "The BF16 validation endpoint is at or above the no-branch
                   baseline (1.374159 >= 1.000000)."
hca  ridge 1e-06: "... (1.505102 >= 1.000000)."
```

— after which the ladder selects ridge 1.0 and 0.1 respectively. Asserted in
`test_the_compressed_summary_receipt_separates_its_two_sub_branches`.

Both kernels' sealed policies (`ChimeraKDAFoldPolicy`,
`ChimeraSparseLatentFoldPolicy`) are now constructor arguments on their
operators, defaulting to the kernels' own, sealed into
`resource_fingerprint` and into the receipt (`kernel_fold_policy`). A caller may
tighten a threshold; nobody reaches past one.

The operators also cross-check the IR against the live module
(`_declared_matches_module`): the graph is what the planner contracted against
and the module is what the kernel will read, so a disagreement on heads, head
dims, decay rank, conv kernel size, gate floor, stride, window or family is
refused before any mutation.

## 3. Planner and registry

| what | where |
|---|---|
| `_bounded_gated_delta_recurrence_contract` | `dynamic_fold_planner.py:1377` |
| `_compressed_summary_attention_contract` | `dynamic_fold_planner.py:1316` |
| `_causal_branch_contract` admits one `RECURRENT_MIXER` for the sequence-mixing branch | `:1026` |
| `_NATIVE_OPERATOR_CONTRACTION_WIDTHS` / `_NATIVE_OPERATOR_CONTRACT_CONSTANTS` | `:1493` |
| `_native_sequence_mixer_branch` (the three-family dispatch) | `:1529` |
| `causal_kda_recurrent_layer`, `causal_summary_attention_layer` method specs | `dynamic_fold_registry.py:1444` |

Admitting a `RECURRENT_MIXER` on the sequence-mixing branch is not a relaxation:
a bounded gated delta rule is exactly one complete coupled operator making one
ordered residual write, and demanding split Q/K/V/O of it would demand softmax
projections it does not have. The two operator classes are proven by different
contracts and select different executables.

Per family, what may contract and what may not:

| family | contraction coordinates | contract constants |
|---|---|---|
| `causal_sliding_window_split_qkv_attention` | query_heads, key_value_heads, head_dim | sliding_window |
| `compressed_summary_attention` | query_heads, key_value_heads, head_dim | local_window, **summary_stride**, mixer_family |
| `bounded_gated_delta_recurrence` | heads, key_head_dim, value_head_dim, decay_rank | internal_short_conv_kernel_size, gate_lower_bound |

### hca does not get its own method id

Honest answer to the question the work order asked: **the stride parameter
suffices.** `csa` and `hca` are literally the same `SparseLatentAttention` class
at two `cfg` strides, and one stride-parameterized kernel executes both. W11B's
rule for separate ids is "the operator that executes them is a different
executable with a different proven envelope"; here the executable is the same
object, so a second id would claim a second one that does not exist. The stride
is instead a **contract constant**: it is checked for equality by the planner,
refused as a kind translation by the kernel, and sealed into the action
configuration and the receipt (`mixer_family`, `summary_stride`), so a csa
parent can never be folded into an hca child by accident.

The measured *envelopes* do differ — W11A recorded an HCA cross-width child
refused where CSA succeeded, because a stride wider than the window opens a
coverage gap — but that is handled where it belongs, by the endpoint gate failing
closed, not by a name.

**One risk, stated rather than hidden:** the two families share a `family_id`
lane in the ledger, so a deliberate LANE- or FAMILY-level closure recorded
against an hca failure would also close csa. `summary_stride` and `mixer_family`
are in the action configuration, so EXACT and CONFIGURATION closures already
separate them. If a curator ever needs to close one stride and not the other,
that is the moment to split the lane, not before.

### Named gaps: what is filled and what is not

| declared family | capability | state |
|---|---|---|
| `bounded_gated_delta_recurrence` | `bounded_gated_delta_recurrence_layer_operator` | **filled** |
| `compressed_summary_attention` | `compressed_summary_attention_layer_operator` | **filled** |
| `compressed_latent_attention` (gated_mla) | `compressed_latent_kv_attention_layer_operator` | **still a named gap** |
| `stable_latent_moe_router` / `_expert` | `stable_latent_moe_channel_mixer_operator` | **still a named gap** |

Both remaining gaps are proven to survive with every registered lane enabled
(`test_gated_mla_and_stable_latent_moe_remain_named_gaps_with_every_lane_on`).
Note one honest subtlety in that test: on the k3-aligned parent the recurrence
gap *also* survives — not because the recurrence has no kernel, but because that
layer's channel mixer is a Stable LatentMoE, and a layer whose feed-forward half
has no owning transaction cannot be closed by folding its sequence mixer alone.
The gap machinery names both missing families rather than one, which is exactly
what it is for.

`NATIVE_CAUSAL_LAYER_BRANCH_METHODS` still means the local/dense pair, so every
W11B contract test is unchanged; `NATIVE_CHIMERA_LAYER_BRANCH_METHODS` is the
new four-method opt-in.

## 4. The proof

`tests/test_chimera_omni_causal_fold_execution.py`, real
`chimera_omni.modeling.ChimeraOmniModel`, `block_pattern =
("local", "kda", "csa", "hca")`, `residual_mode = "block_attnres"`,
window 8 (it binds — sequence is 24), csa stride 4, hca stride 8,
12 × 24 tokens plus 16 media segments.

### Exact 4x is not 128 → 32 here, and that is arithmetic, not a compromise

A `local`-only chimera trunk is linear in the residual width, so 128 → 32 is 4x
by construction. Add `csa`/`hca` and it stops being true: their four summary
projections are `[hidden, hidden]`, which shrinks by **sixteen** while everything
else shrinks by four. Writing the total as `α·H + β·H² + γ` (γ being the KDA
conv taps, decay bias/log-scale and head-norm gain, which do not scale with the
residual at all), `total(H_p) = 4·total(H_c)` has **no** solution at `H_c = H_p/4`
for any β > 0 unless the child's per-width coefficient *exceeds* the parent's —
which contraction forbids.

The exact integer solution used here is **128 → 34 with every other geometry
identical**: vocab 1534, intermediate 448, kda `(9, 8, 4)`, heads 4/2 × 32.

```
parent unique parameters  1_601_760
child  unique parameters    400_440   ->  exactly 4x, asserted
```

### Plan

`plan.ready is True`, **zero gaps**, 24 actions:

| method | count |
|---|---|
| `causal_windowed_nope_attention_layer` | 1 |
| `causal_kda_recurrent_layer` | 1 |
| `causal_summary_attention_layer` | 2 (csa **and** hca) |
| `causal_fused_gate_up_mlp_layer` | 4 |
| `native_residual_chart_interface` | 14 |
| `copy_exact` | 2 |

Every child tensor is owned by exactly one transaction and no tensor is claimed
twice.

### Execution

`DynamicFoldExecutor`: `status = "complete"`, 0 parent root forwards, **0 parent
module forwards**, 0 optimizer constructions/steps, 0 backward calls, 0
autograd-grad calls, 0 pruning/quantization operations,
`executed_fold_static_zero_training_audit_passed is True`.

### Per-family fold numbers (held-out tokens, chart-projected target, no-branch baseline = 1.0)

| layer | family | method | ridge | endpoint | BF16 endpoint |
|---|---|---|---|---|---|
| 0 | local | `causal_windowed_nope_attention_layer` | 1e-2 | **0.0472** | — |
| 1 | kda | `causal_kda_recurrent_layer` | 1.0 | **0.3059** | 0.3007 |
| 2 | csa (stride 4) | `causal_summary_attention_layer` | 1.0 | **0.0680** | 0.1087 |
| 3 | hca (stride 8) | `causal_summary_attention_layer` | 0.1 | **0.0703** | 0.1143 |
| 0 | dense fused MLP | `causal_fused_gate_up_mlp_layer` | 1.0 | 0.2103 | — |
| 1 | dense fused MLP | `causal_fused_gate_up_mlp_layer` | 1.0 | 0.2907 | — |
| 2 | dense fused MLP | `causal_fused_gate_up_mlp_layer` | 1.0 | 0.3106 | — |
| 3 | dense fused MLP | `causal_fused_gate_up_mlp_layer` | 1.0 | 0.2925 | — |

KDA diagnostics: unroll reconstruction **9.31e-08**, conv-tap carry
**5.56e-08**, state snapshots at positions 7/15/23 with relative drift
0.1284 / 0.0423 / 0.0251, drift growth ratio **0.165** against a cap of 4.0, and
the drift gate is **applicable** (the drift is well above the 1e-3 noise floor,
unlike the identity-geometry toy in W11A).

CSA/HCA sub-branch diagnostics:

| family | split reconstruction | local energy | summary energy | local endpoint | summary endpoint | probability overlap |
|---|---|---|---|---|---|---|
| csa | 2.51e-08 | 0.8473 | 0.2093 | 0.0790 | 0.1977 | 0.9422 |
| hca | 2.47e-08 | 0.7938 | 0.1375 | 0.0849 | 0.0961 | 0.9605 |

(The two energy fractions exceed 1 in sum because the sub-branch deltas are not
orthogonal; that is the design's own definition and why it demands they be
reported separately.)

### The receipt is the child

`test_the_installed_child_reproduces_every_family_branch_endpoint` recaptures the
installed child and recomputes all **eight** branch endpoints from its real
forward. Every one matches its receipt to better than **1.5e-07** relative. This
is what makes the recurrence honest rather than plausible: a KDA branch is not a
linear map, and if the class the kernel replayed and the class the child's
`KimiDeltaAttention` runs differed anywhere — a conv tap, the decay chain, the
gate — twenty-four sequential steps would not land on the same number.

### Rollback, refusals, agreement

* Rejecting the recurrence branch undoes it **and every transaction accepted
  before it**; the child returns to blank tensor for tensor, not by fingerprint.
* A tightened kernel policy (KDA `conv_carry_tolerance = 1e-15`, summary
  `minimum_summary_probability_overlap = 0.999`) makes the kernel refuse on every
  rung; the operator surfaces it as its own failure carrying the kernel's
  measurement, and the child is bit-identical afterwards.
* A single absurd ridge rung drives the summary kernel's **no-branch floor**
  specifically. The recurrence kernel enforces the same floor but a
  ridge-collapsed KDA solve lands *just under* it — a huge ridge leaves the
  correlation direction, which still beats writing nothing — so its floor is not
  reachable from the ridge axis and is covered by the policy case. That is
  written into the test's own docstring rather than papered over with a rung
  chosen to trip it.
* Mutating the parent's `gate_lower_bound` so the live module disagrees with the
  IR fails closed before any mutation.
* Next-token top-1 agreement with the parent **85.4%** against a blank child's
  near-zero. The parent is randomly initialised, so this is a statement about
  mechanism, not capability.

## 5. Suites

```
.venv/Scripts/python.exe -m pytest \
  tests/test_chimera_omni_causal_fold_execution.py \
  tests/test_chimera_omni_native_fold_contract.py \
  tests/test_dynamic_chimera_omni_inspector.py \
  tests/test_dynamic_model_inspector.py \
  tests/test_dynamic_fold_planner.py \
  tests/test_dynamic_causal_operator_promotion.py \
  tests/test_dynamic_fold_registry.py -q
→ 110 passed, 1 skipped in 81s

.venv/Scripts/python.exe -m pytest \
  tests/test_cross_family_causal_fold_execution.py \
  tests/test_cross_width_causal_fold_execution.py \
  tests/test_hf_causal_kda_fold.py \
  tests/test_hf_causal_sparse_latent_fold.py \
  tests/test_hf_causal_windowed_nope_attention_fold.py \
  tests/test_dynamic_fold_executor.py -q
→ 115 passed in 33s

.venv/Scripts/python.exe experiments/run_chimera_omni_toy_fold.py --train-steps 2
→ stage "complete"

.venv/Scripts/python.exe -m ruff check <the eight owned files>
→ All checks passed
```

Four failures exist in `tests/test_dynamic_neo_native_vision.py`,
`tests/test_native_{audio,video,points3d}_executable_lane.py`
(`behavior_gate.evaluation_count == 3`, expected 4). They are **pre-existing at
`HEAD` (73d40512)** — reproduced in a clean `git worktree` of HEAD with none of
this wave's edits — and live in the artifact-publishing lane a parallel agent
owns. Not touched.

## 6. Honest gaps in this work

* **`gated_mla` and `stable_latent_moe` are still named gaps.** Nothing here
  approximates them, and the contract suite proves they survive with every lane
  enabled.
* **The summary sub-branch fit is in parent coordinates.** Sealed and explained
  above. Closing it properly means the CSA kernel accepting chart-projected
  sub-branch targets, which is a kernel change and is not owned here.
* **The decay-inverse-chain gate is vacuous through this operator**, because
  retention is derived from the captured decay projection. The unroll gate is
  the one that binds, and both facts are in the receipt. A per-family chimera
  capturer (`hf_chimera_circuit_evidence.py`, design §5) would observe retention
  directly and make the chain gate real again; that module still does not exist.
* **One global chart, not a per-boundary chart set.** Unchanged from W11B:
  transition defects are zero by construction, cycle and norm-commutation
  defects are live and unmeasured.
* **The KDA no-branch floor is not exercised from the ridge axis**, only from
  the policy axis. Stated in the test.
* **The exact-4x pair contracts only the residual width.** Head merging, head-dim
  cuts and decay-rank cuts are what W11A measured as expensive, and this proof
  does not exercise them at trunk scale — the geometry is identity on every
  operator axis and the compaction comes entirely from `hidden_size`.
* **The parent is untrained.** This file proves planning, transaction, rollback
  and audit across four sequence-mixer families. Capability retention is what a
  trained parent and whole-response review decide.

## Files

| path | what |
|---|---|
| `src/neural_forge/dynamic_model_inspector.py` | csa/hca emit one composite `ATTENTION` node; three helpers deleted; coverage metadata updated |
| `src/neural_forge/dynamic_fold_planner.py` | recurrence and composite-summary contracts, the `RECURRENT_MIXER` branch arm, the per-family width/constant tables, the three-family dispatch, two method ids |
| `src/neural_forge/dynamic_causal_operator_promotion.py` | `BoundedGatedDeltaRecurrenceLayerOperator`, `CompressedSummaryAttentionLayerOperator`, the shared `_ChimeraMixerBranchOperator` preamble, the exact sub-branch writer transport |
| `src/neural_forge/dynamic_fold_registry.py` | two `EXPERIMENTAL` method specs on their own family lanes |
| `src/neural_forge/dynamic_native_circuit_evidence.py` | additive: intermediate module boundaries inside a composite node |
| `tests/test_dynamic_chimera_omni_inspector.py` | the composite-node contract, the updated coverage metadata |
| `tests/test_chimera_omni_native_fold_contract.py` | six new planner contract cases (eight with the parametrization) for the two families |
| `tests/test_chimera_omni_causal_fold_execution.py` | ten new execution, receipt, refusal and rollback cases for the whole family |
