# NOTES-W3A — target Q/K normalization and the executable cross-family path

Branch `codex/lrc-blank-compression`, on top of `29939aea`. This wave closes
**NOTES-1F item 1** (a qwen3-shaped child planned through the attention branch
method and then failed closed) and lands **NOTES-2D §5** (the pair-bridge
sibling of the token-interface operator). The cross-family fold is now
executable end to end at toy scale, not just plannable.

Files changed:

- `src/neural_forge/hf_causal_gqa_attention_fold.py` — target per-head Q/K RMS
  normalization.
- `src/neural_forge/dynamic_causal_operator_promotion.py` — the attention branch
  operator installs the Q/K gains; new `PairCodeTokenInterfaceBridgeOperator`.
- `src/neural_forge/hf_causal_circuit_evidence.py` — one added guard (source
  head gains must be direct scale).
- `src/neural_forge/hf_causal_semantic_compiler.py` — comment plus fail-closed
  wording only; no behavior change.
- `tests/test_dynamic_causal_operator_promotion.py` — the obsolete "fails closed
  on target Q/K norm" test is replaced by the install contract plus a broader
  fail-closed test.
- `tests/test_hf_causal_gqa_qk_norm_fold.py` (new), 4 tests.
- `tests/test_cross_family_causal_fold_execution.py` (new), 6 tests.

Deliberately **not** touched: `dynamic_fold_planner.py`,
`dynamic_fold_registry.py`, `dynamic_model_inspector.py`,
`hf_tokenizer_pair_bridge.py`, `hf_tokenizer_bridge.py`, `experiments/*`, every
`hf_lfm2_*` module. Nothing in this wave needed them (see §5).

---

## 1. The Q/K-norm math

A qwen3 attention computes, per head, before RoPE:

```
q = g_q ⊙ ( u / sqrt(mean(u²) + ε) ),      u = W_q x
```

with one gain vector `g_q` of length `head_dim` shared by every query head
(`k_norm` likewise over the KV heads). Two facts drive the fit:

1. The norm is **scale-blind per (token, head)**. Whatever magnitude the
   projection produces is discarded and replaced by the gain. The parent's
   per-token Q/K magnitude is therefore *unrecoverable by construction* — a real
   qwen3 child cannot represent it either.
2. Everything else is unchanged. The desired pre-RoPE target coordinate
   `D = invert_rope(barycentric mean of the parent's post-RoPE heads, contracted
   to the selected RoPE pairs)` is exactly the same object the identity lane
   already fits.

So the compile is a two-stage closed form, both stages ridge least squares:

```
W        = ridge( child_normalized_fit_rows -> D_fit_rows )        # unchanged
u        = reshape_heads(child @ Wᵀ)                               # what the child computes
g        = fit_rmsnorm_diagonal(u_fit_rows, D_fit_rows, ε, ridge)   # per-dim diagonal
replay Q = g ⊙ (u / sqrt(mean(u²)+ε))                              # then RoPE, then V/O
```

`fit_rmsnorm_diagonal` is the same primitive the branch operator already uses
for the block input norm; it normalizes its input rows internally and solves
`g_j = Σ u'_j D_j / (Σ u'_j² + ridge)` per channel. Rows are pooled over all
selected `(token, head)` pairs, because the target owns one gain vector, not one
per head.

Two properties are worth stating because they are easy to get wrong:

- **The gain is fitted against the projection the child actually computes**, not
  against `D`'s own normalization. Whatever the projection solve got wrong is
  absorbed by the diagonal instead of being assumed away. That is what makes it
  a joint fit rather than two independent ones.
- **The projection is fitted against raw `D`, not against `D`'s unit-RMS
  direction.** The direction is the only part the norm keeps, so fitting it
  looks more principled, and I implemented and measured that version first. It
  is *worse*: `D` is a near-linear function of the cached parent state while
  `D/rms(D)` divides by a per-token scalar that a linear projection cannot
  represent. Held-out attention-branch relative RMSE, four topologies, llama →
  qwen3:

  | case | fit raw `D` (kept) | fit unit-RMS direction |
  |---|---|---|
  | same topology, seeds 11/13 | **0.007609 / 0.003599** | 0.013156 / 0.005698 |
  | same topology, seeds 101/103 | **0.003627 / 0.004563** | 0.005895 / 0.007216 |
  | 8→4 q, 4→2 kv, h64 | 0.602188 / **0.453924** | **0.602011** / 0.452289 |
  | 8→4 q, 4→2 kv, h32, d8 | **0.632348** / 0.262179 | 0.645703 / **0.261273** |

  It wins on the same-topology cases and ties on the contractions, and it keeps
  the projection solve byte-identical between the two lanes. Keeping raw `D` is
  the whole reason the identity path is provably unchanged (§2).

The V/O solve is unaffected in structure: it still runs *after* the installed
Q/K are replayed through the softmax, and the replay now includes the fitted
gains, so `qk_installed_before_vo_fit` stays true in the real sense.

**Error budget.** The gain fit's own relative RMSE is reported
(`query_norm_gain_relative_rmse` / `key_norm_gain_relative_rmse`) and is
*expected* to be large — 0.21–0.32 at toy scale. That number is the magnitude
information the target architecture destroys, not a defect in the solve. The
number that matters is the branch's held-out
`validation_attention_delta_relative_rmse`.

### Fail-closed surface

`_target_query_key_norm` in `dynamic_causal_operator_promotion.py` classifies
the target and rejects everything it cannot compile into:

| target | outcome |
|---|---|
| no `q_norm`, no `k_norm` | `identity` — the pre-existing lane, byte-identical |
| bias-free `rms_norm`/`direct_scale` at `query_head`+`key_head`, one shared ε | `rms_norm` — installed |
| only one of Q/K normalized | rejected, "requires a target that normalizes both queries and keys per head, or neither" |
| `weight_kind == "unit_offset"` (gemma3 stores `1 + w`) | rejected, "installs bias-free direct-scale per-head RMS Q/K normalization only" |
| `has_bias`, non-RMS kind, or wrong placement | rejected, same message |
| `q_norm.epsilon != k_norm.epsilon` | rejected, "requires one shared target Q/K normalization epsilon" |

The kernel dataclass independently requires the gains to be present **exactly
when** `target_topology.query_key_norm == "rms_norm"` and to have width
`head_dim`, so a caller cannot construct a fold whose declared semantics and
carried tensors disagree.

On the **source** side, `_assert_recordable_layout` in
`hf_causal_circuit_evidence.py` now also requires `q_norm`/`k_norm` to be
direct-scale RMS. The kernel multiplies those cached weights straight into the
source head coordinates, so a unit-offset source gain would be off by one per
channel. llama (no head norms) and qwen3 (direct scale) are unaffected; the
guard exists so the family allowlist is not the only thing preventing it, per
the NOTES-1F rule.

### Identity-Q/K children are byte-identical

`tests/test_hf_causal_gqa_qk_norm_fold.py::test_identity_query_key_target_fold_is_byte_identical_to_the_old_kernel`
pins digests captured from the kernel **before** this change (llama seed 3 →
llama seed 7, `intermediate_size=16`):

```
evidence cache key          f270c1d80b7452f272846fe6ee743c29a87f6583f38ec9d38e512607fb257b8d
input norm report sha256    fd1c8960204a8747c14bf07a922ac61bddd1b20cf6daafd6d5f526c4c352b83c
attention report sha256     5f0d11c1fe7308ae73dc4f53f8bd40bc0c19cb6f3fb83a4634c40383fa3146e8
candidate state fingerprint 095b4de28616d8cfd210e694e93c9b1c06c77289eb3f74c9cd0af5c4f11a704e
q/k/v/o + input norm payload sha256s: all five pinned
validation_attention_delta_relative_rmse 0.0036692173732920173
```

This is why the new receipt keys are **conditional**. `HFCausalGQATopology.as_dict()`
gained no key, and both the kernel report and the operator report only grow
`target_query_key_norm*` / `*_norm_gain_relative_rmse` / `*_norm_report_sha256`
/ `*_norm_weight_sha256` when the target actually declares a head norm. A
receipt persisted before this wave still reproduces bit for bit. Anyone adding
an unconditional key to either report body breaks that test and should say so
explicitly rather than re-baselining the digests.

### Proof that it is the *target architecture's* norm

`test_installed_qk_norm_child_replays_its_own_attention` installs the branch,
traces the real `Qwen3ForCausalLM` with `collect_hf_transport_trace`, reads the
installed tensors **back out of the child**, and feeds the child's own
pre-attention norm output through `HFCausalCoupledAttentionFold.replay`:

```
replay vs the module's own attention_delta   relative RMSE 1.1e-07   (< 1e-5 asserted)
same tensors read as identity-Q/K            relative RMSE 1.9e-03   (17,000x worse)
centered score relative RMSE between the two 0.484
```

Only per-head, direct-scale, pre-RoPE gains give 1e-07. The control matters:
at toy scale the *delta* is fairly forgiving of Q/K error because the softmax is
near uniform, which is why the score-geometry control is asserted too.

---

## 2. `PairCodeTokenInterfaceBridgeOperator` (NOTES-2D §5)

Sibling class in `dynamic_causal_operator_promotion.py`, same
`method_id = "token_interface_bridge"`, does **not** overload the strict one.
Steps 1–4 and 6 of the strict operator are reused verbatim; only the transport
differs, exactly as §5 predicted. The §4 alias properties were already present
on `TokenizerPairBridgeArtifact`, so `hf_tokenizer_bridge.py` needed no edit —
wiring item 1 in NOTES-2D §4 is still open but was not required here.

**Row-capacity check.** The strict operator compares target rows against
`artifact.target_vocabulary_size`. The pair operator compares against
`profile.target_row_capacity` — the physical matrix height — and records both
`target_row_capacity` and `target_vocabulary_size`. Parent rows are checked
against `source_row_capacity` the same way. Trailing padded rows carry no code
and are structurally uncovered.

**Transports.** `embedding_transport` / `lm_head_transport` accept
`anchor-preferred` (default), `anchor-exact`, `length-normalized`, and
`occurrence-weighted` (which requires `token_frequencies` at construction, not
at prepare time). `anchor-preferred` is §5's option 1: take the byte-exact
anchor row wherever the anchor transport has one, the length-normalized code
average elsewhere. The two agree exactly on rows whose only incidence is a
single direct code, so the blend only ever upgrades a mixed row to its exact
anchor. The operator asserts `anchor_mask ⊆ length_normalized_mask` rather than
assuming it.

**`evidence_mask` is load-bearing — this is the part to not "simplify".** The
transport builder returns exact zeros for uncovered rows. Those zeros are a
claim the bridge cannot make, so `_install_only_covered_rows` writes
`where(mask, compiled, incumbent)`: uncovered rows keep the blank child's own
value. After the install, and *inside* the try block that rolls back, the
operator re-reads every installed tensor and requires `live[~mask] == blank[~mask]`
bit for bit. The incumbent round-trips float32 → float64 → float32, which is
exact.

The report records `covered_output_rows`, `uncovered_output_rows`,
`masked_row_policy`, and per tensor
`{embedding,lm_head}_uncovered_target_rows`,
`_uncovered_target_row_indices` (capped at 64, with an explicit `_truncated`
flag), and `_evidence_mask_sha256` over the full covered-row set so the record
is auditable even when the index list is truncated. `covered_output_rows` /
`uncovered_output_rows` describe the **token embedding**; a caller that gives
the head a different weighting reads the `lm_head_*` keys.

**Tied storage.** One transport, one tensor, reused for both nodes — never two
builder calls compared for equality, because the two are only bit-identical when
the weighting and inputs match. Covered by
`test_pair_bridge_installs_one_tied_storage_group_from_one_transport`.

**Coordinates.** The operator accepts the `DenseVocabularyProjector` protocol
(`source_width`, `target_width`, `map_sha256`, `fold_consumer_weight`) and
defaults to `IdentityVocabularyProjector`, like the strict lane. It deliberately
does **not** accept a `CoordinateMap`; width folding stays where it lives and a
caller that needs one wraps it.

### No registry change was needed — and this is the answer to the coordinator question

The task allowed either a `NOTES-W3A` registry spec or a per-system injected
method spec. Neither turned out to be necessary:

- `token_interface_bridge` is **already** an `ACTIVE` method in
  `default_method_registry()` (`dynamic_fold_registry.py`, the `_method(
  "token_interface_bridge", "token_interface", MethodStatus.ACTIVE, ...)` entry),
  and
- `DynamicFoldPlanner` already selects it for *any* target whose tokenizer
  fingerprint differs from the parent's — the `add_global_targets` closure in
  `DynamicFoldPlanner.plan` sets `method_id = "token_interface_bridge"` on
  exactly that condition. There is no "child tokenizer must be a pruned donor"
  check to relax; NOTES-2D §5's planner/registry note was a conservative guess,
  and auditing both call sites shows the concern does not exist. The bridge kind
  is authenticated by the operator, not the planner.

(Both files are being edited by concurrent waves, so symbols are cited rather
than line numbers.)

So the operator is injected per run into a `FoldOperatorRegistry`, exactly like
`Lfm2DenseBudgetModelOperator` and the two branch operators. **Nothing is
registered globally by this wave.** A run picks the strict or the pair operator
by constructing one or the other; the registry holds one operator per method id,
which is the correct constraint (a single run has a single token interface).

Coordinator action required: **none.** If a future system wants both bridge
kinds selectable inside one run, that needs a *new* method id and a registry
entry, and the planner would need a rule for choosing between them — do not add
that until there is a caller.

---

## 3. Cross-family execution proof (toy scale)

`tests/test_cross_family_causal_fold_execution.py`. Two real ByteLevel-BPE
tokenizers trained on divergent corpora (English technical prose vs a
French/physics mix), 330 and 300 rows, compiled into a real sealed pair bridge.

| | parent | child |
|---|---|---|
| family | `llama` | `qwen3` (blank) |
| vocab | 330 | 300 (unrelated tokenizer) |
| hidden | 32 | 32 |
| intermediate | 48 | 32 |
| query / kv heads | 8 / 4 | 4 / 2 |
| head_dim | 8 | 8 |
| Q/K norm | identity | per-head RMS, direct scale |
| unique parameters | 42,784 | 31,680 (ratio 1.35) |

The blank child is deliberately **non-zero**: the transport builder returns
zeros for uncovered rows, so a zero blank would make "left alone" and "installed
the builder's zeros" indistinguishable and the central assertion vacuous.

Plan (`enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS`):
`ready=True`, **zero gaps**, 7 actions —

```
token_interface_bridge                     model.embed_tokens.weight, lm_head.weight
copy_exact                                 (no learned tensors: position node)
causal_gqa_rope_residual_attention_layer    layer 0: input_layernorm, q/k/v/o_proj, q_norm, k_norm
causal_swiglu_residual_mlp_layer            layer 0: post_attention_layernorm, gate/up/down_proj
causal_gqa_rope_residual_attention_layer    layer 1: (same 7)
causal_swiglu_residual_mlp_layer            layer 1: (same 4)
copy_exact                                 model.norm.weight
```

The action target-path multiset has no duplicate and equals the child graph's
parameter paths exactly, so every learned tensor is owned by exactly one
transaction.

Execution through `DynamicFoldExecutor`:

```
status                                   complete
parent forward hook calls during run     0
observed_parent_root_forward_calls       0
observed_parent_module_forward_calls     0
optimizer/backward/autograd/prune/quant  0 each
executed_fold_static_zero_training_audit_passed  True
completed_target_parameter_paths         == every child parameter path
changed_state_paths across all attempts  no duplicate, all declared
child logits                             (2, 16, 300), finite, |max| 0.459
```

Held-out (validation-mask) branch errors, real 8→4 / 4→2 head contraction into a
Q/K-normalized target:

```
attention layer 0   validation_attention_delta_relative_rmse 0.3597
attention layer 1   validation_attention_delta_relative_rmse 0.2676
  q gain fit rmse   0.2085 / 0.2371        k gain fit rmse 0.2610 / 0.2613
mlp layer 0         validation_mlp_delta_relative_rmse 0.9215
mlp layer 1         validation_mlp_delta_relative_rmse 0.9276
```

The MLP numbers are ~0.92 because the child MLP is 32 wide against a 48-wide
source and the fit is against parent residual endpoints, not a child replay
(NOTES-1F item 3 still stands). **These are structural-completeness figures, not
capability claims.** The behavior gate remains the acceptance authority; the
test asserts bounds, not quality.

Pair bridge coverage at this scale:

```
forward codes   defined 330   anchor-exact 266   multi-code 60   special-role 4
                unresolved 0  byte mismatch 0    max code length 6
                pure byte fallback 47
target rows     covered 273   uncovered 27  (of 300)
```

27 of 300 child rows are reached by no forward code. All 27 keep their blank
value bit for bit, all 27 are listed in
`embedding_uncovered_target_row_indices`, and the test additionally asserts the
blank values are non-zero (so the check cannot pass trivially) and that every
covered row genuinely changed.

Rollback: a second run rejects the *second* attention branch. The rejected
attempt reports `rollback_exact=True` with
`final_state_fingerprint == incumbent_state_fingerprint`, the whole run is
unwound, and every child tensor is `torch.equal` to its blank value — a
partially folded child never survives a rejected run.

---

## 4. Verification

```
.venv/Scripts/python.exe -m pytest tests/test_hf_causal_semantic_compiler.py \
    tests/test_dynamic_causal_operator_promotion.py \
    tests/test_hf_causal_gqa_qk_norm_fold.py \
    tests/test_cross_family_causal_fold_execution.py \
    tests/test_hf_tokenizer_pair_bridge.py tests/test_hf_tokenizer_bridge.py \
    tests/test_dynamic_fold_executor.py tests/test_dynamic_fold_planner.py \
    tests/test_dynamic_fold_registry.py \
    tests/test_hf_causal_blank_target_adapter.py \
    tests/test_hf_closed_form_causal_circuit_fold.py -q
152 passed in 22.87s

.venv/Scripts/python.exe -m ruff check   (all seven changed/added files)
All checks passed!
```

The planner, registry, and inspector are being edited by concurrent waves; the
run above was repeated after those landed and is still green.

`ruff format --check` is **not** clean on these files — but it is not clean on
`dynamic_fold_planner.py`, `hf_causal_circuit_evidence.py`,
`hf_causal_semantic_compiler.py` or `tests/test_hf_causal_semantic_compiler.py`
either. The repo is hand-formatted; `ruff check` is the shared gate and it
passes. I matched the surrounding style rather than reformatting.

---

## 5. Deliberately left out

1. **The whole-model `HFCausalSameDepthSemanticCompiler` still fails closed on
   Q/K norm.** Its guard is now a *compiler* limit, not a kernel one, and the
   message and a comment say so. It never claims the `q_norm`/`k_norm` tensors,
   so a normalized target would leave them uninstalled and only fail the
   exactly-once audit *after* mutation. Filling it means adding two writes per
   layer and threading the fold's gains through `_CompileWorkspace.install_body`;
   that file is also llama→llama only today, so nothing can reach it.
2. **Source Q/K norm + target Q/K norm together is untested.** The kernel
   supports it (both branches are independent and both already exist), but the
   only cross-family case exercised is identity source → normalized target. A
   qwen3 → qwen3 or qwen3 → llama fold would exercise the other pairings.
3. **`unit_offset` head norms are not supported, by choice.** `hf_blank_bootstrap`
   already has `_read/_write_effective_head_norm_scale` that converts `1 + w`
   both ways, so supporting gemma3 here is mechanical — but gemma3 fails
   `_require_rope_transport` first (per-layer-type RoPE variants) and the
   evidence record rejects it for five other reasons (NOTES-1F §3). Adding the
   conversion without the rest would be an unreachable branch.
4. **Cross-width residual crossing** (NOTES-1F item 2) and **sequential child
   replay** (item 3) are untouched. Both branch operators still require
   `target hidden_size == source hidden_size`.
5. **No `authenticated_live_session` for the pair bridge**, per NOTES-2D §8. The
   operator calls `verify_files()` twice per prepare (construction + prepare),
   which at LFM2 scale is payload hashing, not tokenizer recompilation. If a
   caller prepares in a loop, that is the thing to cache.
6. **No reverse-code fallback for uncovered rows** (NOTES-2D §5 option 3). It is
   implementable on top of `artifact.reverse_code(t)` and would reduce the 27
   uncovered rows, but it installs evidence the *forward* direction does not
   support, and option 1 is the only policy that installs nothing unproven. If
   someone adds it, it must be a new named policy with its own
   `masked_row_policy` value, not a change to the default.
7. **The `token_frequencies` source is still a caller decision.** The operator
   hashes the table into the transport report via the pair module; it does not
   estimate frequencies.
8. **The source head-norm epsilon is still the block-norm epsilon.**
   `HFCausalArchitectureEvidence.rms_norm_epsilon` is captured from
   `blocks[0].attention_input_norm.epsilon` and the kernel reuses it for the
   *source* `q_norm`/`k_norm` when the parent is normalized. On llama and qwen3
   both come from `config.rms_norm_eps`, so it is exact today, but the record
   has no separate slot for a distinct head-norm epsilon. Adding one changes the
   sealed dataclass shape and invalidates persisted bundles — the same argument
   as NOTES-1F item 4 — so it was left alone. The **target** side does not have
   this problem: the operator reads `attention.query_norm.epsilon` from the live
   layout and requires `q_norm.epsilon == k_norm.epsilon`.
