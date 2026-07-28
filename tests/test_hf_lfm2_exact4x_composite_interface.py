from __future__ import annotations

import hashlib

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_bridge_only_control import (
    _storage_identity,
    _tensor_sha256,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    install_lfm2_cpcf_neutral_norms,
)
from neural_forge.hf_lfm2_exact4x_composite_interface import (
    LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA,
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
    Lfm2Exact4xCompositeInterfaceError,
    _foundation_tensor_sha256,
    install_lfm2_independent_causal_output_head,
    install_lfm2_independent_structural_interface,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
    LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
    LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
    Lfm2CausalResidualTokenBridge,
    Lfm2ChildCausalResidualEvidence,
    compile_lfm2_contextual_input_table,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
    LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA,
    Lfm2SemanticFoundationInterfaceSolve,
    build_lfm2_semantic_foundation_attention_lineage,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    _tensor_sha256 as _anchor_tensor_sha256,
)
from neural_forge.hf_token_align import (
    StrictTokenRowMapping,
    StrictTokenTransportProfile,
)
from neural_forge.hf_tokenizer_bridge import (
    STRICT_TOKENIZER_BRIDGE_SCHEMA,
    StrictTokenizerBridgeArtifact,
    TokenizerArtifactFile,
)
from neural_forge import lfm2_circuit_balanced_coordinate_recipe as coordinate_recipe


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: canonical_sha256(body)}


def _parent(seed: int):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _blank_target(seed: int):
    parent = _parent(seed)
    tokenizer_fingerprint = _sha(f"tokenizer-{seed}")
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=_sha(f"evidence-{seed}"),
        device="cpu",
        dtype=torch.float32,
    )
    adapters = DynamicTargetAdapterRegistry((adapter,))
    system = AutomaticDynamicFoldingSystem(
        target_adapters=adapters,
        stack_adapters=DynamicFoldStackAdapterRegistry(),
        topology_catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    target_token_interface_mode="untied",
                ),
            )
        ),
    )
    plan = system.plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=8,
        ),
        child_name=f"composite-exact-four-x-{seed}",
        tokenizer_fingerprint=tokenizer_fingerprint,
        adapter_id=adapter.adapter_id,
        blank_seed=seed + 1,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=adapters,
    ).synthesize_planned(
        plan=plan,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer_fingerprint},
    )
    return parent, target


def _sequence_plan(*, donor_rows: int, target_rows: int) -> dict:
    records = []
    for target_id in range(target_rows):
        strict = target_id % donor_rows
        records.append(
            {
                "target_id": target_id,
                "strict_donor_id": strict,
                "donor_sequences": [
                    [strict],
                    [strict, (strict + 1) % donor_rows],
                ],
            }
        )
    body = {
        "schema": LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        "bridge_artifact_sha256": _sha("bridge-artifact"),
        "bridge_profile_sha256": _sha("bridge-profile"),
        "donor_tokenizer_semantics_sha256": _sha("donor-tokenizer"),
        "target_tokenizer_semantics_sha256": _sha("target-tokenizer"),
        "target_row_count": target_rows,
        "records": records,
    }
    return _sealed(body, "plan_sha256")


def _foundation_solve(
    contextual_source,
    *,
    target_hidden: int,
) -> Lfm2SemanticFoundationInterfaceSolve:
    source_hidden = int(contextual_source.shape[1])
    encoder = torch.zeros(source_hidden, target_hidden, dtype=torch.float64)
    for index in range(target_hidden):
        encoder[index % source_hidden, index] = 1.0
    source64 = contextual_source.detach().to(dtype=torch.float64)
    target_weight = source64.matmul(encoder).contiguous()
    active_body = {
        "schema": "neural-forge-tied-causal-interface-fold-v1",
        "source_table_sha256": _sha("active-source-lineage"),
        "target_weight_sha256": _sha("active-target-lineage"),
        "producer_encoder_sha256": _sha("active-encoder-lineage"),
        "minimum_validation_top1_agreement": 0.0,
        "numeric_parent_weight_matching_objective": False,
        "functional_mapped_parent_logit_objective": True,
    }
    active_report = _sealed(active_body, "report_sha256")
    interface = TiedCausalInterfaceFold(
        target_weight=target_weight,
        producer_encoder=encoder,
        report=active_report,
    )
    base_report = _sealed(
        {"schema": "test-authenticated-base-interface"},
        "report_sha256",
    )
    anchor_report = _sealed(
        {
            "schema": "test-authenticated-anchor-interface",
            "structural_safety_is_only_candidate_veto": True,
        },
        "report_sha256",
    )
    contract = _sealed(
        {
            "schema": "neural-forge-lfm2-foundation-anchor-selection-contract-v1",
            "candidate_eligibility_basis": (
                "finite_dense_shape_dtype_device_correct_effectful_residual"
            ),
            "mapped_parent_behavior_retention_review_goal": 0.9,
            "mapped_parent_behavior_retention_review_goal_is_hard_gate": False,
            "relative_table_drift_review_goal": 0.1,
            "relative_table_drift_review_goal_is_hard_gate": False,
            "numeric_metrics_role": "deterministic_diagnostic_ordering_only",
            "numeric_metrics_may_restore_base_control": False,
            "base_control_only_if_all_analytic_residuals_invalid_or_identical": True,
        },
        "contract_sha256",
    )
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
        "compiler_cache_key": _sha("cache-key"),
        "cache_parent_fingerprint": _sha("parent"),
        "cache_manifest_sha256": _sha("manifest"),
        "bridge_artifact_sha256": _sha("bridge-artifact"),
        "trajectory_corpus": "/sealed/local/trajectory.jsonl",
        "trajectory_corpus_sha256": _sha("trajectory"),
        "trace_adapter_sha256": _sha("trace-adapter"),
        "selective_catalog_preparation_receipt_sha256": _sha("preparation"),
        "selective_catalog_binding_sha256": _sha("binding"),
        "selective_catalog_manifest_sha256": _sha("catalog-manifest"),
        "selective_catalog_artifact_sha256": _sha("catalog"),
        "final_norm_authentication_sha256": _sha("final-norm"),
        "target_hidden_width": target_hidden,
        "target_norm_epsilon": 1e-5,
        "base_interface_report": base_report,
        "base_interface_report_sha256": base_report["report_sha256"],
        "causal_anchor_report": anchor_report,
        "causal_anchor_report_sha256": anchor_report["report_sha256"],
        "causal_anchor_selection_contract": contract,
        "active_interface_report": active_report,
        "active_interface_report_sha256": active_report["report_sha256"],
        "source_tied_table_sha256": _foundation_tensor_sha256(
            contextual_source
        ),
        "active_target_weight_sha256": _foundation_tensor_sha256(
            interface.target_weight
        ),
        "producer_encoder_sha256": _foundation_tensor_sha256(
            interface.producer_encoder
        ),
        "only_final_norm_payload_materialized": True,
        "legacy_layer_compiler_calls": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return Lfm2SemanticFoundationInterfaceSolve(
        interface=interface,
        report=_sealed(body, "result_sha256"),
    )


def _anchor_domain_foundation_solve(
    contextual_source,
    *,
    target_hidden: int,
) -> Lfm2SemanticFoundationInterfaceSolve:
    """Give the fixture the report domain emitted by the real anchor solve."""

    base = _foundation_solve(
        contextual_source,
        target_hidden=target_hidden,
    )
    interface = base.interface
    active_body = dict(interface.report)
    active_body.pop("report_sha256")
    active_body.update(
        {
            "schema": "neural-forge-tied-causal-anchor-residual-v1",
            "target_weight_sha256": _anchor_tensor_sha256(
                interface.target_weight
            ),
            "producer_encoder_sha256": _anchor_tensor_sha256(
                interface.producer_encoder
            ),
        }
    )
    active_report = _sealed(active_body, "report_sha256")
    active_interface = TiedCausalInterfaceFold(
        target_weight=interface.target_weight,
        producer_encoder=interface.producer_encoder,
        report=active_report,
    )
    solve_body = dict(base.report)
    solve_body.pop("result_sha256")
    solve_body.update(
        {
            "active_interface_report": active_report,
            "active_interface_report_sha256": active_report["report_sha256"],
            "active_target_weight_sha256": _foundation_tensor_sha256(
                active_interface.target_weight
            ),
            "producer_encoder_sha256": _foundation_tensor_sha256(
                active_interface.producer_encoder
            ),
        }
    )
    return Lfm2SemanticFoundationInterfaceSolve(
        interface=active_interface,
        report=_sealed(solve_body, "result_sha256"),
    )


def _identity_tokenizer_bridge(
    root,
    *,
    vocabulary_size: int,
) -> StrictTokenizerBridgeArtifact:
    root.mkdir()
    payload = b"{}\n"
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_bytes(payload)
    file_record = TokenizerArtifactFile(
        relative_path="tokenizer.json",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class="LifecycleDonor",
        target_tokenizer_class="LifecycleTarget",
        donor_vocab_size=vocabulary_size,
        target_vocab_size=vocabulary_size,
        donor_row_capacity=vocabulary_size,
        target_row_capacity=vocabulary_size,
        alignment_fingerprint=_sha("lifecycle-alignment"),
        row_mappings=tuple(
            StrictTokenRowMapping(
                target_row_index=index,
                donor_row_index=index,
                resolution="lifecycle-identity",
            )
            for index in range(vocabulary_size)
        ),
    )
    body = {
        "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": "sealed-strict-target-tokenizer",
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": _sha("lifecycle-donor"),
        "target_tokenizer_semantics_sha256": _sha("lifecycle-target"),
        "target_files": [file_record.to_dict()],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }
    return StrictTokenizerBridgeArtifact(
        root_dir=root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256=_sha("lifecycle-donor"),
        target_tokenizer_semantics_sha256=_sha("lifecycle-target"),
        target_files=(file_record,),
        artifact_sha256=canonical_sha256(body),
    )


def _attention_parent_inventory(parent) -> dict:
    attention = parent.model.layers[1].self_attn
    inventory = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=attention.q_proj.weight,
        source_key_weight=attention.k_proj.weight,
        source_value_weight=attention.v_proj.weight,
        source_output_weight=attention.out_proj.weight,
    )
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA,
        "attention_qkvo_inventories": {"1": inventory.as_dict()},
    }
    return _sealed(body, "result_sha256")


def _ancestry() -> dict[str, str]:
    return {
        name: _sha(name)
        for name in (
            "parent_terminal_evidence_sha256",
            "cache_key",
            "parent_identity_sha256",
            "selective_session_sha256",
            "manifest_sha256",
            "completion_sha256",
            "semantic_alignment_sha256",
            "tokenizer_live_session_sha256",
            "causal_root_rows_sha256",
            "contextual_transition_plan_sha256",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
            "target_prefix_rows_sha256",
        )
    }


def _child_evidence(model, *, input_sha256: str, body_sha256: str):
    hidden_size = int(model.config.hidden_size)
    hidden_rows = torch.arange(
        3 * hidden_size,
        dtype=torch.float32,
    ).reshape(3, hidden_size)
    state = fingerprint_model_state(model)
    body = {
        "schema": LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
        "status": "complete-target-native-child-residual-replay",
        "evidence_role": "focused-composite-test",
        "trie_evidence_sha256": _sha("trie"),
        **_ancestry(),
        "child_hidden_rows_sha256": _tensor_sha256(hidden_rows),
        "causal_input_table_sha256": input_sha256,
        "causal_body_inventory_sha256": body_sha256,
        "causal_input_table_after_sha256": input_sha256,
        "causal_body_inventory_after_sha256": body_sha256,
        "row_count": 3,
        "hidden_size": hidden_size,
        "child_model_state_before_sha256": state,
        "child_model_state_after_sha256": state,
        "child_state_preserved": True,
        "input_and_body_coordinate_preserved": True,
        "batch_size": 1,
        "full_prefix_use_cache_false": True,
        "position_origin": 0,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 3,
            "child_generate_calls": 0,
            "network_access": False,
        },
    }
    return Lfm2ChildCausalResidualEvidence(
        hidden_rows=hidden_rows,
        receipt=_sealed(body, "evidence_sha256"),
    )


def _causal_bridge(
    model,
    evidence,
    *,
    input_sha256: str,
    body_sha256: str,
    sequence_plan_sha256: str,
):
    input_weight = model.get_input_embeddings().weight
    output_table = input_weight.detach().cpu().clone()
    output_table[0].add_(0.25)
    evidence_receipt = evidence.as_dict()
    body = {
        "schema": LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
        "status": "dense-cache-only-contextual-causal-residual-bridge-compiled",
        **_ancestry(),
        "child_feature_evidence_sha256": evidence_receipt["evidence_sha256"],
        "trie_conditional_evidence_sha256": evidence_receipt[
            "trie_evidence_sha256"
        ],
        "causal_input_table_sha256": input_sha256,
        "causal_body_inventory_sha256": body_sha256,
        "child_model_state_sha256": evidence_receipt[
            "child_model_state_before_sha256"
        ],
        "sequence_plan_sha256": sequence_plan_sha256,
        "context_free_inverse_output_plan_used": False,
        "structural_output_table_sha256": input_sha256,
        "output_table_sha256": _tensor_sha256(output_table),
        "causal_child_residual_features_used": True,
        "compiled_against_parent_hidden_features": False,
        "materialized_runtime_operator": "ordinary-dense-output-table",
        "runtime_parent_or_adapter_required": False,
        "diagnostic_metrics_are_not_acceptance_gates": True,
        # Deliberately terrible diagnostics: integrity, not a numeric cutoff,
        # decides whether this ordinary dense table can be installed.
        "relative_centered_error_after": 999.0,
        "manual_behavior_score": -123.0,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "network_access": False,
        },
    }
    return Lfm2CausalResidualTokenBridge(
        table=output_table,
        receipt=_sealed(body, "result_sha256"),
    )


def _structural_install(seed: int):
    parent, target = _blank_target(seed)
    donor_table = parent.get_input_embeddings().weight
    target_model = target.model
    sequence_plan = _sequence_plan(
        donor_rows=int(donor_table.shape[0]),
        target_rows=int(target_model.config.vocab_size),
    )
    contextual = compile_lfm2_contextual_input_table(
        donor_table,
        sequence_plan,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
    )
    solve = _foundation_solve(
        contextual,
        target_hidden=int(target_model.config.hidden_size),
    )
    installed = install_lfm2_independent_structural_interface(
        target,
        solve,
        donor_tied_table=donor_table,
        sequence_plan=sequence_plan,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
    )
    return parent, target, sequence_plan, installed


def test_dynamic_anchor_interface_survives_the_pre_coordinate_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    """Regress the dynamic production order which exposed mixed hash domains."""

    parent, target = _blank_target(1151)
    child = target.model
    donor_table = parent.get_input_embeddings().weight
    sequence_plan = _sequence_plan(
        donor_rows=int(donor_table.shape[0]),
        target_rows=int(child.config.vocab_size),
    )
    contextual = compile_lfm2_contextual_input_table(
        donor_table,
        sequence_plan,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
    )
    solve = _anchor_domain_foundation_solve(
        contextual,
        target_hidden=int(child.config.hidden_size),
    )
    interface = solve.interface
    target_snapshot = interface.target_weight.clone()
    producer_snapshot = interface.producer_encoder.clone()
    report_snapshot = dict(interface.report)
    interface_storage = {
        _storage_identity(interface.target_weight),
        _storage_identity(interface.producer_encoder),
    }

    def assert_interface_is_immutable_and_independent() -> None:
        assert torch.equal(interface.target_weight, target_snapshot)
        assert torch.equal(interface.producer_encoder, producer_snapshot)
        assert dict(interface.report) == report_snapshot
        assert interface.report["schema"] == (
            "neural-forge-tied-causal-anchor-residual-v1"
        )
        assert interface.report["target_weight_sha256"] == (
            _anchor_tensor_sha256(interface.target_weight)
        )
        assert interface.report["producer_encoder_sha256"] == (
            _anchor_tensor_sha256(interface.producer_encoder)
        )
        assert interface_storage.isdisjoint(
            {_storage_identity(parameter) for parameter in child.parameters()}
        )

    assert_interface_is_immutable_and_independent()
    structural = install_lfm2_independent_structural_interface(
        target,
        solve,
        donor_tied_table=donor_table,
        sequence_plan=sequence_plan,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
    )
    assert structural.model is child
    assert_interface_is_immutable_and_independent()

    neutral_norms = install_lfm2_cpcf_neutral_norms(child)
    assert neutral_norms["accepted"] is True
    assert_interface_is_immutable_and_independent()

    attention_lineage = build_lfm2_semantic_foundation_attention_lineage(
        parent_inventory=_attention_parent_inventory(parent),
        target_model=child,
    )
    assert set(attention_lineage) == {"1"}
    assert attention_lineage["1"]["target_hidden_size"] == int(
        child.config.hidden_size
    )
    assert_interface_is_immutable_and_independent()

    bridge = _identity_tokenizer_bridge(
        tmp_path / "tokenizer",
        vocabulary_size=int(child.config.vocab_size),
    )
    basis_calls = []

    class _ExpensiveBasisBoundaryReached(RuntimeError):
        pass

    def stub_expensive_basis(
        observed_parent,
        observed_cache,
        observed_target_width,
        **_kwargs,
    ):
        basis_calls.append(
            (observed_parent, observed_cache, observed_target_width)
        )
        raise _ExpensiveBasisBoundaryReached

    cache_sentinel = object()
    monkeypatch.setattr(
        coordinate_recipe,
        "build_lfm2_circuit_balanced_basis",
        stub_expensive_basis,
    )
    with pytest.raises(_ExpensiveBasisBoundaryReached):
        coordinate_recipe.build_lfm2_circuit_balanced_coordinate_evidence(
            parent,
            cache_sentinel,
            int(child.config.hidden_size),
            foundation_interface=interface,
            tokenizer_bridge=bridge,
        )
    assert basis_calls == [(parent, cache_sentinel, int(child.config.hidden_size))]
    assert_interface_is_immutable_and_independent()


def test_composite_installs_contextual_interface_then_output_only_causal_head():
    parent, target, sequence_plan, structural = _structural_install(1201)
    model = target.model
    structural_receipt = structural.as_dict()
    assert structural.model is model
    assert (
        structural_receipt["schema"]
        == LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA
    )
    assert structural_receipt["only_interface_state_changed"] is True
    assert (
        structural_receipt["diagnostic_metrics_used_as_installation_gate"]
        is False
    )
    assert structural_receipt["manual_behavior_scores_used_as_installation_gate"] is False
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    assert torch.equal(
        model.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
    )

    # Represent the intended semantic body/final-W2 stages between the two
    # interface transactions.  The causal installer must accept this changed
    # body when it is exactly the coordinate authenticated by child replay.
    input_identity = _storage_identity(model.get_input_embeddings().weight)
    output_identity = _storage_identity(model.get_output_embeddings().weight)
    body_parameter = next(
        parameter
        for parameter in model.parameters()
        if _storage_identity(parameter) not in {input_identity, output_identity}
    )
    with torch.no_grad():
        body_parameter.reshape(-1)[0].add_(0.125)
    input_before = model.get_input_embeddings().weight.detach().clone()
    body_sha256 = fingerprint_lfm2_causal_body_inventory(model)
    input_sha256 = _tensor_sha256(input_before)
    evidence = _child_evidence(
        model,
        input_sha256=input_sha256,
        body_sha256=body_sha256,
    )
    bridge = _causal_bridge(
        model,
        evidence,
        input_sha256=input_sha256,
        body_sha256=body_sha256,
        sequence_plan_sha256=sequence_plan["plan_sha256"],
    )
    body_before = fingerprint_lfm2_causal_body_inventory(model)
    result = install_lfm2_independent_causal_output_head(
        structural,
        bridge,
        evidence,
        expected_input_table_sha256=input_sha256,
        expected_body_inventory_sha256=body_sha256,
    )
    receipt = result.as_dict()
    assert result.model is model
    assert receipt["schema"] == LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA
    assert receipt["only_distinct_output_parameter_may_change"] is True
    assert receipt["input_and_body_coordinate_preserved"] is True
    assert receipt["diagnostic_metrics_used_as_installation_gate"] is False
    assert receipt["manual_behavior_scores_used_as_installation_gate"] is False
    assert torch.equal(model.get_input_embeddings().weight, input_before)
    assert fingerprint_lfm2_causal_body_inventory(model) == body_before
    assert _tensor_sha256(model.get_output_embeddings().weight) == _tensor_sha256(
        bridge.table
    )
    assert not torch.equal(
        model.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
    )
    assert fingerprint_model_state(parent) != fingerprint_model_state(model)


def test_structural_install_rejects_foundation_fit_to_another_context_plan():
    parent, target = _blank_target(1301)
    donor_table = parent.get_input_embeddings().weight
    plan = _sequence_plan(
        donor_rows=int(donor_table.shape[0]),
        target_rows=int(target.model.config.vocab_size),
    )
    contextual = compile_lfm2_contextual_input_table(
        donor_table,
        plan,
        strict_anchor_ppm=0,
        preserve_anchor_norm=False,
    )
    solve = _foundation_solve(
        contextual,
        target_hidden=int(target.model.config.hidden_size),
    )
    with pytest.raises(
        Lfm2Exact4xCompositeInterfaceError,
        match="source coordinate",
    ):
        install_lfm2_independent_structural_interface(
            target,
            solve,
            donor_tied_table=donor_table,
            sequence_plan=plan,
            strict_anchor_ppm=1_000_000,
            preserve_anchor_norm=True,
        )


def test_causal_install_rejects_body_mutation_after_child_feature_replay():
    _parent_model, target, sequence_plan, structural = _structural_install(1401)
    model = target.model
    input_sha256 = _tensor_sha256(model.get_input_embeddings().weight)
    body_sha256 = fingerprint_lfm2_causal_body_inventory(model)
    evidence = _child_evidence(
        model,
        input_sha256=input_sha256,
        body_sha256=body_sha256,
    )
    bridge = _causal_bridge(
        model,
        evidence,
        input_sha256=input_sha256,
        body_sha256=body_sha256,
        sequence_plan_sha256=sequence_plan["plan_sha256"],
    )
    input_identity = _storage_identity(model.get_input_embeddings().weight)
    output_identity = _storage_identity(model.get_output_embeddings().weight)
    body_parameter = next(
        parameter
        for parameter in model.parameters()
        if _storage_identity(parameter) not in {input_identity, output_identity}
    )
    with torch.no_grad():
        body_parameter.reshape(-1)[0].add_(1.0)

    with pytest.raises(
        Lfm2Exact4xCompositeInterfaceError,
        match="Live input/body/state",
    ):
        install_lfm2_independent_causal_output_head(
            structural,
            bridge,
            evidence,
            expected_input_table_sha256=input_sha256,
            expected_body_inventory_sha256=body_sha256,
        )
