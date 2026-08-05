# NOTES-W13J — the dense SiTU channel mixer folds, and the real k3-aligned config now plans with zero gaps

Proof: `tests/test_chimera_omni_causal_fold_execution.py` (52 passed, 1 dormant
skip), contracts: `tests/test_chimera_omni_native_fold_contract.py` (30 passed),
IR: `tests/test_dynamic_chimera_omni_inspector.py` (19 passed), kernel:
`tests/test_hf_closed_form_causal_circuit_fold.py` (13 passed).
Predecessors: `NOTES-W12F.md` (gated-MLA + Stable LatentMoE), `NOTES-W12C.md`
(KDA/CSA/HCA), `NOTES-W11B.md` (the planner lane).
Branch `codex/lrc-blank-compression`, HEAD `8a4ed56f`. Nothing committed.

`NOTES-W12F.md` §7 named the last gap precisely: *"the dense (non-MoE) SiTU
channel mixer is still refused, by the existing fused gate/up lane, which admits
SiLU only. On the real `k3_aligned_test_config` that leaves exactly one gapped
layer — layer 0, the `moe_first_dense_layers` layer."* That is closed here, by
widening the existing family's contract rather than by adding a lane.

---

## 1. Tracing the refusal: where SiLU was pinned

The chain, end to end:

| stage | where | what it did with the activation |
|---|---|---|
| IR | `dynamic_model_inspector.py:8869` (`_channel_mixer_nodes`) | declared `activation` on the dense node; **never declared SiTU's betas** |
| contract | `dynamic_fold_planner.py:1640` (`_fused_gate_up_mlp_contract`) | `_semantic_name(activation) in {"silu", "swish"}` — refusal by name |
| dispatch | `dynamic_fold_planner.py:1950` (`_family_neutral_causal_layer_contract`) | selected the method with **no source/target activation check at all** |
| registry | `dynamic_fold_registry.py:1417` | reason text: *"a non-SiLU gate … fails closed"* |
| operator | `dynamic_causal_operator_promotion.py:3206` | `str(activation).lower() not in _GATED_ACTIVATIONS` → *"implements SiLU-gated targets only"* |
| kernel | `hf_closed_form_causal_circuit_fold.py:1228,1239,1249` + `:1180` | **four hardcoded `torch.nn.functional.silu` call sites** |

So it was pinned in **all three places** — a contract constant refusal, an
operator refusal, and hardcoded arithmetic. The interesting part is that the
first two were refusals *about* the third.

### 1.1 What the kernel's solve actually assumes

`fit_functional_swiglu` is an evidence-row solve against captured tensors, and
its dependence on the activation is narrower than the four call sites suggest.

**Activation-agnostic** (unchanged, and provably so):

* **The two producer ridges.**
  `all_gate_weights = ridge(child_rows → captured gate rows)` and
  `all_up_weights = ridge(child_rows → captured up rows)`. Their targets are
  the halves of the parent's **captured `blocks.N.ffn.up` output** — rows the
  parent emitted *before* `FeedForward._activate` ran (`layers.py:357-359`
  applies `up`, then chunks, then activates). No nonlinearity is anywhere in
  either solve. This is the load-bearing fact: the widening is a contract
  change precisely because the fits consume pre-activation evidence.
* **The chunk-two split** of the fused `[2·intermediate, hidden]` output — a
  tensor view the module performs identically under either activation.
* **The input RMSNorm diagonal fit**, the ridge ladder, the chart transport,
  the depth-selector fold, the re-fuse shape equality against the live child
  parameter, the 1.0 no-branch endpoint floor, the snapshot/rollback envelope,
  the zero-training audit.
* **The writer ridge's *form*** (features → complete target).

**Activation-dependent** — exactly three arithmetic sites and one serialization
site, all inside `fit_functional_swiglu`/`FunctionalSwiGLUFold`:

1. `realizable_features` — the dictionary the deterministic maximal-residual
   pivot ranks. A different nonlinearity ranks different columns and therefore
   **selects a different subset of parent neurons**.
2. `child_features` — the design matrix of the writer ridge, so a different
   nonlinearity gives a different `down_weight`.
3. `FunctionalSwiGLUFold.predict` — the selection score *and* the validation
   endpoint. If it disagreed with the child's module, the receipt would be a
   number the child does not compute.
4. `source_features` → `source_feature_matrix_sha256` — report only, but it is
   an identity claim about which feature matrix was consumed, so it has to
   follow the activation too.

### 1.2 No gate in this lane is SiLU-specific

The task allowed for a gate that genuinely needs SiLU (an exact gate-collapse
identity in kind translation, an explicit linearization, an inverse-sigmoid
clamp). **This lane has none.** Enumerated, its gates are: the fit/selection/
validation token split, the contraction-only intermediate check, the declared
layout/bias/chunk-split checks, the re-fused shape equality, the endpoint
maximum and the no-branch floor. Every one of them is a statement about shapes,
rows or residual error, not about a nonlinearity.

The SiLU-shaped identities that *do* exist in the tree are elsewhere and were
not touched:

* `hf_causal_kind_translation.py:1240` `gate_collapse_detected` — that is the
  ShortConv→attention lane's *convolution* gate dispersion, not a channel-mixer
  activation.
* `hf_causal_latent_kv_attention_fold.py` fits its output gate against a
  captured **pre-sigmoid** boundary, which is why W12F recorded that family as
  needing no inverse-sigmoid clamp policy. Different family, different gate.

So nothing had to be kept SiLU-only inside this lane, and nothing was skipped
for SiTU: **SiTU runs the identical code path with a different sealed
constant.**

### 1.3 What deliberately stays SiLU-only

* `CausalGatedMLPBranchOperator` (`dynamic_causal_operator_promotion.py:929`)
  and `_gated_mlp_contract(require_swiglu=True)`
  (`dynamic_fold_planner.py:2847`) keep the narrow `{silu, swish}` set. That is
  the **split** gate/up lane; nothing has widened its contract or measured its
  envelope, and chimera's dense mixer is fused, not split.
* `fit_functional_swiglu`'s **default** is still SiLU, so every existing caller
  (`hf_causal_semantic_compiler.py:453`, the split-MLP operator) is untouched.

---

## 2. The widening

### 2.1 One sealed constant, one executable, one method id

`GatedActivation` (`hf_closed_form_causal_circuit_fold.py:1148`) is a frozen
kind plus the constants that kind reads:

```
silu :  F.silu(gate) * up
situ :  (β_g·tanh(gate/β_g)) · σ(gate) · (β_u·tanh(up/β_u))
```

The SiTU form is read off `chimera-omni/src/chimera_omni/modeling/layers.py:350-355`,
not assumed. It is a **soft-capped SwiGLU**: `β_g·tanh(gate/β_g) · σ(gate)` is a
capped SiLU and `β_u·tanh(up/β_u)` is a capped up, and both degenerate to the
SwiGLU form as their beta grows. The betas therefore *are* part of the function,
which is why they travel with the kind rather than beside it.

`fit_functional_swiglu` takes `activation: GatedActivation | None = None` and
threads it through exactly the three arithmetic sites above.
`FunctionalSwiGLUFold` carries it so `predict` replays the same constant it was
fitted under.

`ChimeraStableLatentMoEGeometry.activate`
(`hf_causal_stable_latent_moe_fold.py:281`) now **delegates to the same
object** instead of carrying its own copy of the formula. Both channel-mixer
lanes fold the same `FeedForward._activate`; one definition means it cannot
drift from the executable in one lane and not the other.

The method id does **not** change. `causal_fused_gate_up_mlp_layer` executes
both, which is the `hca`-stride rule from `NOTES-W12C.md` §3 applied to a
nonlinearity: a second id would claim a second proven envelope that does not
exist, because the executable is the same object.

### 2.2 Where the constant is checked

| check | where |
|---|---|
| the IR declares SiTU's betas *where SiTU is read* | `dynamic_model_inspector.py:8887` |
| the contract admits `{silu, swish, situ}` and demands a SiTU node declare positive betas | `dynamic_fold_planner.py:1614` (`_fused_gate_up_activation_constant`), `:1649` |
| source and target must carry the **same** constant | `dynamic_fold_planner.py:1990` |
| the fit objective names the constant | `..._fused_gate_up_situ_residual_write` vs the unchanged `..._fused_gate_up_swiglu_residual_write` |
| the operator resolves and compares both sides | `dynamic_causal_operator_promotion.py:3175`, `:3318` |
| both **live modules** must agree with the graph | `dynamic_causal_operator_promotion.py:3222` (`_fused_gate_up_live_activation`) |
| the receipt names what was folded | `gated_activation` on every fused-MLP receipt |

Three deliberate details:

* **`silu` and `swish` collapse to one class, `situ` does not.**
  `hf_arch_layout.py:283` maps the name *swish* onto `nn.SiLU`, so they denote
  one function; comparing raw strings would have newly refused an existing
  silu↔swish crossing. The constant is therefore the *class* tuple
  `("silu",)` / `("situ", β_g, β_u)`, and the equality is on that.
* **The betas are declared only under SiTU.** `FeedForward._activate` reads
  neither under SiLU, so declaring them on a SiLU node would put a constant in
  the contract that no arithmetic consumes — and would move every existing SiLU
  chimera graph's digest for nothing. A missing key on both sides compares
  equal, so the constants table is unaffected. (The composite MoE node still
  declares them unconditionally: that shape is W12F's, and its operator
  cross-checks both betas against the live module on every crossing. The
  asymmetry is real and is recorded here rather than papered over.)
* **The live-module check is skipped only where there is nothing to compare.**
  A module with no `activation_kind` attribute — every HF-shaped fused MLP —
  is left alone, because inventing a default would be an assumption rather than
  a check. A chimera module always has one, so the chimera path is always
  cross-checked, SiLU included.

---

## 3. Byte-neutrality for SiLU

The claim is not "the tests still pass". It is measured against the
implementation at `8a4ed56f`.

**Kernel level.** The pre-widening `hf_closed_form_causal_circuit_fold.py` was
extracted from `HEAD` and both implementations were run on identical evidence:

```
head report_sha256 : 1fec5f69c8a30deb4b585fb1cdc9e67c10253773239aa081ac321fe0322d2d60
live report_sha256 : 1fec5f69c8a30deb4b585fb1cdc9e67c10253773239aa081ac321fe0322d2d60
bodies equal / gate / up / down / predict : True True True True True
```

**Operator level.** `fit_functional_swiglu` was traced through a complete SiLU
dense chimera fold (the file's original 128→32 `local` trunk), replaying every
call through the HEAD implementation:

```
kernel calls traced      : 24        (4 layers x 6 ridge rungs)
all digests match HEAD   : True
all tensors match HEAD   : True
layer.000 swiglu_report_sha256 b55dde9e...  gated_activation {'kind': 'silu'}  endpoint 0.2089
layer.001 b31128a0...  0.2372     layer.002 58e83fd7...  0.2768     layer.003 aad56a0e...  0.2600
```

**Serialization shape.** The SiLU report body gains **no key**; the SiTU one
gains `gated_activation` and nothing else. This is conditional on purpose:
every sealed SwiGLU receipt already in the tree carries the old body, and an
unconditional key would invalidate all of them for a constant SiLU does not
read. The task asked for a pin if the shape moved, so there is one —
`test_the_silu_swiglu_report_body_is_byte_pinned` asserts
`24148a208085a44f2d4b1d1496a17f044b38991b1efeb7e7729ef28e8b0bb751`, a digest
**recomputed from the pre-widening implementation**, not a snapshot of the new
code agreeing with itself.

**Graphs.** A SiLU dense chimera node's attributes are unchanged, so every
existing SiLU graph digest is unchanged.

**The sealed report.** `reports/chimera-omni-fold-readiness-20260804/` is
byte-identical to `HEAD` (`git diff` empty; the `M` flags are the pre-existing
CRLF index-stat artifact `NOTES-W12F.md` §6 described). Every runner invocation
here used a scratch `--out`.

---

## 4. The proof

### 4.1 The real `k3_aligned_test_config`, zero gaps

`chimera_omni.config.k3_aligned_test_config()` at 128 against a 96-wide child,
every registered lane on — the same crossing `NOTES-W12F.md` left with one
gapped layer:

```
parent kernel_gap_block_kinds : ''
plan.ready                    : True
plan.gaps                     : []
methods : native_residual_chart_interface 14
          causal_stable_latent_moe_channel_mixer_layer 4
          causal_kda_recurrent_layer 3      <- was 2; layer 0 now plans
          causal_latent_kv_attention_layer 2
          causal_fused_gate_up_mlp_layer 1  <- the dense SiTU mixer
          copy_exact 2
every child tensor owned exactly once : True
dense action fit_objective : cached_parent_endpoint_coupled_fused_gate_up_situ_residual_write
```

`test_the_real_k3_reference_plans_with_zero_gaps` asserts all of it. Two
companion cases assert the constant is load-bearing: a SiLU child of a SiTU
parent and a β_g=8 child of a β_g=4 parent both keep the gap
(`test_a_gated_activation_kind_translation_keeps_its_gap`,
`test_a_situ_cap_the_child_does_not_share_keeps_its_gap`).

### 4.2 A complete executed exact-4x fold with the dense SiTU layer in it

The k3 reference's own widths cannot be 4x-contracted (its `moe_latent_size` is
64, above a 32-wide child), so the execution trunk is W12F's k3-aligned-*style*
crossing with `moe_first_dense_layers = 1`, which puts a dense SiTU mixer at
layer 0 next to gated-MLA and Stable LatentMoE neighbours:
`block_pattern = ("local", "gated_mla", "kda", "gated_mla")`,
`activation_kind = "situ"` (β_g 4.0, β_u 25.0), 12 × 24 tokens + 16 media
segments.

Swapping one MoE mixer for a dense one only changes the affine law's
coefficients (both are linear in the residual width), so the exact-4x pair
survives with a different vocabulary — 1403 instead of 1459:

```
total(H) = 12135 * H + 16180          (measured at four widths and asserted)
12135 * (H_p - 4 H_c) = 3 * 16180 = 48540,  48540 / 12135 = 4  ->  H_p = 4 H_c + 4
parent  12135 * 128 + 16180 = 1_569_460
child   12135 *  31 + 16180 =   392_365     ->  exactly 4x, asserted
```

Plan: `ready`, **zero gaps**, 24 actions — 1 windowed-NoPE, 1 KDA, 2 latent-KV,
**1 fused gate/up (SiTU)**, 3 Stable LatentMoE, 14 chart interface, 2
`copy_exact`; every child tensor owned once.

Execution: `status = "complete"`, 0 parent root forwards, **0 parent module
forwards**, 0 optimizer constructions/steps, 0 backward calls, 0 autograd-grad
calls, 0 pruning/quantization operations,
`executed_fold_static_zero_training_audit_passed is True`.

#### Per-branch numbers (held-out tokens, chart-projected target, no-branch baseline = 1.0)

| layer | branch | method | ridge | endpoint | BF16 | kernel-refused rungs |
|---|---|---|---|---|---|---|
| 0 | local | `causal_windowed_nope_attention_layer` | 1e-2 | **0.0504** | — | — |
| 0 | **dense SiTU** | `causal_fused_gate_up_mlp_layer` | 1.0 | **0.2329** | — | — |
| 1 | gated_mla | `causal_latent_kv_attention_layer` | 0.1 | **0.1672** | 0.1672 | 0 |
| 1 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 1.0 | **0.3225** | 0.3221 | 2 |
| 2 | kda | `causal_kda_recurrent_layer` | 1.0 | **0.3659** | 0.3648 | 0 |
| 2 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 10.0 | **0.3367** | 0.3245 | 2 |
| 3 | gated_mla | `causal_latent_kv_attention_layer` | 0.1 | **0.2164** | 0.2166 | 0 |
| 3 | Stable LatentMoE | `causal_stable_latent_moe_channel_mixer_layer` | 10.0 | **0.4535** | 0.4533 | 3 |

The dense SiTU rung ladder is monotone into a real interior optimum, not a
degenerate pick: `1e-6 → 0.7697`, `1e-4 → 0.5444`, `1e-2 → 0.3777`,
`0.1 → 0.2591`, **`1.0 → 0.2111`**, `10.0 → 0.2816`.

MoE diagnostics on this trunk: routing identity **6.49e-08 … 6.71e-08**, router
top-k agreement **0.851 / 0.938 / 0.983**, per-expert fit rows
`[108,65,79,44] / [93,80,76,47] / [52,79,59,106]`.

#### The receipt is the child, and the child computes SiTU

`test_the_installed_child_reproduces_every_dense_situ_branch_endpoint`
recaptures the installed child and recomputes all **eight** branch endpoints
from its real forward; each matches its receipt to better than **1e-05**
relative.

`test_the_installed_dense_situ_child_computes_situ_and_not_silu` measures the
activation directly at `blocks.0.ffn`, chunking the installed fused tensor
exactly as the module does and applying both activations to the same halves:

```
SiTU replay relative error : 2.14e-07
SiLU replay relative error : 2.21e-02        (~1e5 x larger)
```

This is stated carefully, because it is the honest shape of the risk. On a
randomly initialised parent most pre-activations sit inside SiTU's caps, so the
two activations are only a couple of percent apart — a lane that had quietly
kept SiLU's dictionary would **still fold, still beat the 1.0 floor, and still
pass every count above**. What it could not do is make the receipt match the
child, because the receipt endpoint is computed through the kernel's activation
and the child computes SiTU; the 1e-05 recomputation gate is three orders below
the 2.2e-02 discrepancy.

#### Rollback, refusals, agreement

* Rejecting the dense SiTU branch undoes it **and every transaction accepted
  before it**; the child returns to blank tensor for tensor
  (`test_a_rejected_dense_situ_branch_rolls_the_run_back`).
* Mutating the parent module's `activation_kind` to `"silu"` so the live module
  disagrees with the IR fails closed **before any mutation**, naming both
  values; mutating the child module's `situ_up_beta` does the same, naming the
  beta and the side.
* The MoE kernel's no-branch floor still refuses **2/2/3** ridge rungs across
  the three MoE layers, recorded per rung rather than silently taken.
* Next-token top-1 agreement with the parent **70.8%** against a blank child's
  **0.0%**. That parent is randomly initialised, so it is a statement about
  mechanism, not capability — which is why the trained crossing below exists.

### 4.3 The same trunk from a *trained* toy parent

`test_a_trained_parent_folds_and_the_child_tracks_its_next_token` trains the
parent first under chimera-omni's own tiny-proof objective (80 steps, lr 5e-4,
`runner.train_parent`, which mirrors
`chimera_omni.proof.tiny_multimodal.train_tiny_proof`) and then folds it with
exactly the same transactions. Training a *parent* is not folding a child, and
the executed run's audits are what separate the two.

```
tiny-proof training : mean_first_10 0.5922 -> mean_last_10 0.2330   status "passed"
fold                : status "complete"
                      0 parent root forwards, 0 parent module forwards,
                      0 optimizer constructions / steps, 0 backward, 0 autograd-grad,
                      executed_fold_static_zero_training_audit_passed = True
dense SiTU endpoint : 0.2074   (no-branch baseline 1.0)
next-token top-1    : folded 83.7%   vs   blank child 0.0%
```

Per-branch endpoints on the trained crossing: local **0.0952**, dense SiTU
**0.2074**, gated-MLA **0.1699** / **0.2131**, KDA **0.4156**, MoE **0.3243** /
**0.2933** / **0.2684**.

One thing the trained run exposed and is worth recording: the tiny-proof loop
does **not** apply the router's auxiliary-loss-free quantile balancing between
optimizer steps (chimera leaves that to training code), so a hot schedule
collapses the routing onto two experts and the MoE kernel's sparse row budget
then refuses the channel mixer outright — measured at lr 3e-3, where the
smallest expert fell to 13 fit rows against the floor of 15. The learning rate
here is chosen so the router stays usable (smallest expert 22 fit rows), not so
the gate stays quiet.

#### One honest note on the parent seed

At this file's default parent seed the trunk's smallest routed expert owns
**9** fit rows against the MoE kernel's sealed floor of **15**, and the kernel
correctly refuses the whole channel mixer on every ridge rung. Lowering the
floor to make a proof run would have disabled the gate the suite exists to
exercise, so the *parent* was changed instead (`DENSE_SITU_PARENT_SEED = 512`,
smallest expert 44 fit rows) and the counts are asserted. The refusal is
recorded here because it is evidence the row budget is live, not an
inconvenience that was engineered away.

---

## 5. Suites

```
.venv/Scripts/python.exe -m pytest tests/test_chimera_omni_causal_fold_execution.py -q
→ 52 passed, 1 skipped in 216s          (was 40 passed, 1 skipped)

.venv/Scripts/python.exe -m pytest \
  tests/test_chimera_omni_native_fold_contract.py tests/test_dynamic_chimera_omni_inspector.py \
  tests/test_dynamic_model_inspector.py tests/test_dynamic_fold_planner.py \
  tests/test_dynamic_causal_operator_promotion.py tests/test_dynamic_fold_registry.py \
  tests/test_cross_family_causal_fold_execution.py tests/test_cross_width_causal_fold_execution.py \
  tests/test_hf_causal_kda_fold.py tests/test_hf_causal_sparse_latent_fold.py \
  tests/test_hf_causal_windowed_nope_attention_fold.py tests/test_dynamic_fold_executor.py \
  tests/test_hf_causal_latent_kv_attention_fold.py tests/test_hf_causal_stable_latent_moe_fold.py \
  tests/test_run_chimera_omni_toy_fold.py tests/test_hf_closed_form_causal_circuit_fold.py \
  tests/test_hf_causal_semantic_compiler.py -q
→ 280 passed in 85s

.venv/Scripts/python.exe -m pytest \
  tests/test_audit_lfm2_cpcf_predictive_rank.py tests/test_hf_lfm2_cpcf_layer_compiler.py \
  tests/test_hf_lfm2_cpcf_model_fold.py tests/test_hf_lfm2_cpcf_resume_bridge.py \
  tests/test_hf_causal_gqa_qk_norm_fold.py tests/test_dynamic_neo_native_vision.py \
  tests/test_native_{audio,video,points3d}_executable_lane.py -q
→ 167 passed in 110s      (the four downstream consumers of the widened kernel,
                           plus the four native lanes W12F recorded as failing —
                           green now, fixed by a8b65e15, not by this work)

.venv/Scripts/python.exe experiments/run_chimera_omni_toy_fold.py --train-steps 2 --out <scratch>
→ stage "complete"; committed report untouched (git diff empty)

.venv/Scripts/python.exe -m ruff check <the ten owned code/test files>
→ All checks passed
```

**One pre-existing failure**, outside this surface and named in the task:
`tests/test_run_lfm2_dynamic_catalog.py::test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`.
Reconfirmed failing here; it belongs to another agent's uncommitted diff to
`dynamic_closed_loop_fold.py` / the LFM2 resume capsule, and nothing in this
wave touches those files.

---

## 6. Honest gaps in this work

* **The operator's own source/target activation-equality guard is unreachable
  through a planner-produced action**, because the planner refuses the crossing
  first (`test_a_gated_activation_kind_translation_keeps_its_gap` proves that).
  It is defence in depth for a hand-built action and is therefore **untested**.
  The reachable operator refusals — graph-vs-live-module disagreement on the
  kind and on a beta — are both exercised.
* **The dense node declares SiTU's betas conditionally and the MoE node
  declares them unconditionally.** Both are correct; they are not consistent.
  Making the MoE node conditional would move its digests and would require
  making its `_declared_matches_module` conditional too, which is churn on a
  sealed W12F path for no proof.
* **The exact-4x execution trunk is not the literal `k3_aligned_test_config`.**
  It cannot be: that config pins `moe_latent_size = 64`, above any 4x child of
  128. The real config is proven at the *planning* layer (zero gaps, every
  tensor owned) and the k3-aligned-style trunk at the *execution* layer.
* **The exact-4x pair still contracts only the residual width.** The dense SiTU
  fold runs at `intermediate 320 → 320`; the SiTU dictionary contraction is
  unit-tested (`test_the_gated_dictionary_compiles_situ_rather_than_approximating_it`
  fits 4 of 4 neurons, and `fit_functional_swiglu`'s contraction is covered by
  the pre-existing SiLU cases) but not exercised at trunk scale.
* **SiTU and SiLU are numerically close on an untrained parent** (2.2% apart on
  the installed tensors here). The widening matters for correctness, not
  because the numbers are far apart; the SiLU-vs-SiTU separation is measured
  only on the untrained crossing, not on the trained one.
* **The trained crossing is a toy.** 80 tiny-proof steps on a 1.57M-parameter
  parent is enough to make top-1 agreement mean something (83.7% against a
  blank child's 0.0%) and nothing more. Whole-response capability review is
  still a separate question this file does not answer.
* **The betas are contract constants, not fold coordinates.** A curator who
  wants a differently-capped child must set the caps before the fold; that is a
  caller action and it is recorded as one.
* **One global chart, not a per-boundary chart set.** Unchanged from W11B.
* **The parent is untrained.** This file proves planning, transaction,
  rollback and audit across five sequence-mixer families and both channel-mixer
  families. Capability retention is what a trained parent and whole-response
  review decide.

## Files

| path | what |
|---|---|
| `src/neural_forge/hf_closed_form_causal_circuit_fold.py` | `GatedActivation` + `GATED_ACTIVATION_KINDS`; `fit_functional_swiglu` and `FunctionalSwiGLUFold` are activation-parameterized, SiLU-default, SiLU-byte-identical |
| `src/neural_forge/hf_causal_stable_latent_moe_fold.py` | `ChimeraStableLatentMoEGeometry.activate` delegates to the one shared definition |
| `src/neural_forge/dynamic_model_inspector.py` | the dense channel-mixer node declares SiTU's betas where SiTU is read |
| `src/neural_forge/dynamic_fold_planner.py` | `_fused_gate_up_activation_constant`; the widened contract; the source/target constant equality and the SiTU fit objective |
| `src/neural_forge/dynamic_causal_operator_promotion.py` | `_FUSED_GATE_UP_ACTIVATIONS`, `_fused_gate_up_activation`, `_fused_gate_up_live_activation`; the operator carries the constant into the kernel and onto the receipt |
| `src/neural_forge/dynamic_fold_registry.py` | the method spec's reason states the two sealed activations instead of "non-SiLU fails closed" |
| `tests/test_hf_closed_form_causal_circuit_fold.py` | the SiLU byte pin, the SiTU dictionary case, the unnamed-activation refusals |
| `tests/test_dynamic_chimera_omni_inspector.py` | the dense node declares SiTU's caps and only SiTU's |
| `tests/test_chimera_omni_native_fold_contract.py` | the real k3 reference plans with zero gaps; two kind-translation refusals |
| `tests/test_chimera_omni_causal_fold_execution.py` | the dense-SiTU exact-4x trunk: twelve plan/execution/receipt/rollback/refusal cases, including one folded from a trained parent; `_ChimeraFold` gains `parent_seed` and `train_steps` |
