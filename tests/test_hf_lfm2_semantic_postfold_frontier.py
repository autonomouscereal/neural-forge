from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold_frontier as frontier  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_cpcf_terminal_action_residual as terminal_action  # noqa: E402

from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionSeed,
    compile_lfm2_cpcf_terminal_action_residual,
)
from neural_forge.hf_lfm2_semantic_preterminal_capsule import (  # noqa: E402
    Lfm2SemanticPreterminalCapsule,
    Lfm2SemanticPreterminalCapsuleReference,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _anchor_selection(candidate_index: int = 9):
    recipe_body = {
        "schema": "fixture-lazy-anchor-candidate-recipe-v1",
        "candidate_index": candidate_index,
        "fixture_label": f"candidate-{candidate_index}",
        "base_target_weight_sha256": _sha("anchor-base-target"),
    }
    recipe = {
        **recipe_body,
        "recipe_sha256": frontier.canonical_sha256(recipe_body),
    }
    body = {
        "schema": postfold.LFM2_SEMANTIC_FOUNDATION_ANCHOR_SELECTION_SCHEMA,
        "interface_frontier_companion_json": "fixture/frontier-companion.json",
        "interface_frontier_companion_file_sha256": _sha("companion-file"),
        "interface_frontier_companion_sha256": _sha("companion-body"),
        "interface_statistics_artifact_sha256": _sha("statistics"),
        "causal_anchor_frontier_sha256": _sha("anchor-frontier"),
        "causal_anchor_materializable_candidate_indices": list(range(10)),
        "causal_anchor_compatibility_default_candidate_index": 7,
        "anchor_base_candidate_index": 0,
        "anchor_base_interface_report_sha256": _sha("anchor-base-interface"),
        "anchor_base_target_weight_sha256": _sha("anchor-base-target"),
        "anchor_base_target_weight_foundation_sha256": _sha(
            "anchor-base-target-foundation"
        ),
        "compatibility_default_interface_report_sha256": _sha(
            "default-interface"
        ),
        "compatibility_default_target_weight_sha256": _sha("default-target"),
        "saved_target_weight_sha256": _sha("saved-target"),
        "saved_child_exactly_matches_compatibility_default": True,
        "selected_anchor_candidate_index": candidate_index,
        "selected_anchor_candidate_recipe": recipe,
        "selected_anchor_candidate_recipe_sha256": recipe["recipe_sha256"],
        "selected_interface_report_sha256": _sha(
            f"selected-interface:{candidate_index}"
        ),
        "selected_target_weight_sha256": _sha(f"selected-target:{candidate_index}"),
        "selected_target_weight_foundation_sha256": _sha(
            f"selected-target-foundation:{candidate_index}"
        ),
        "selected_residual_weight_sha256": _sha(
            f"selected-residual:{candidate_index}"
        ),
        "selected_residual_weight_foundation_sha256": _sha(
            f"selected-residual-foundation:{candidate_index}"
        ),
        "selected_anchor_exactly_equals_base_plus_residual": True,
        "selected_producer_encoder_sha256": _sha("producer-encoder"),
        "selected_candidate_lazily_materialized": True,
        "target_model_installation_deferred_until_authenticated_pipeline_session": (
            True
        ),
        "dense_anchor_candidate_checkpoints_persisted": 0,
        "numeric_metrics_have_final_intelligence_authority": False,
    }
    return {**body, "selection_sha256": frontier.canonical_sha256(body)}


def test_layer_stage_rejects_mismatched_anchor_source_before_compiler(
    monkeypatch,
) -> None:
    selection = _anchor_selection(7)
    target = torch.ones((3, 2), dtype=torch.float64)
    encoder = torch.eye(2, dtype=torch.float64)
    report_body = {
        "schema": postfold.TIED_RESIDUAL_CONGRUENCE_SCHEMA,
        "mode": postfold.TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        "source_interface_report_sha256": _sha("wrong-anchor-interface"),
        "target_weight_sha256": postfold._interface_tensor_sha256(target),
        "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
        "anchor_base_target_weight_anchor_sha256": selection[
            "anchor_base_target_weight_sha256"
        ],
        "anchor_source_target_residual_anchor_sha256": selection[
            "selected_residual_weight_sha256"
        ],
    }
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report={
            **report_body,
            "report_sha256": frontier.canonical_sha256(report_body),
        },
    )
    compiler_called = False

    def resolve_compiler(_value):
        nonlocal compiler_called
        compiler_called = True
        return lambda **_kwargs: None

    monkeypatch.setattr(postfold, "_resolve_layer_stack_installer", resolve_compiler)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="does not bind the residual-aware body interface",
    ):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=object(),
            interface=interface,
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            interface_anchor_selection=selection,
        )

    assert compiler_called is False


def _synthetic_terminal_composite_lineage(
    *,
    cache_key: str,
    cache_parent_fingerprint: str,
    bridge_artifact_sha256: str,
    interface_report_sha256: str,
    compatibility_sha256: str,
    payload_binding_sha256: str,
    session_sha256: str,
):
    source_body = {
        "schema": "neural-forge-lfm2-long-horizon-terminal-source-lineage-v1",
        "cache_key": cache_key,
        "cache_manifest_sha256": _sha("cache-manifest"),
        "cache_parent_fingerprint": cache_parent_fingerprint,
        "cache_layer_topology_fingerprint": _sha("cache-topology"),
        "bridge_artifact_sha256": bridge_artifact_sha256,
        "bridge_transport_profile_fingerprint": _sha("bridge-profile"),
        "target_to_donor_row_indices_sha256": _sha("row-map"),
        "parent_final_norm_authentication_sha256": _sha("parent-norm-auth"),
        "mapped_table_bfloat16_sha256": _sha("mapped-bfloat16"),
        "mapped_table_interface_source_sha256": _sha("mapped-interface"),
        "interface_report_sha256": interface_report_sha256,
        "compiler_source_binding_sha256": _sha("source-binding"),
        "direct_local_checkpoint_source_authenticated": False,
        "behavioral_tolerance_used_for_source_admission": False,
        "exact_lineage_is_source_admission_authority": True,
    }
    source = postfold.Lfm2LongHorizonTerminalSourceLineage(
        frontier._sealed(source_body, "authentication_sha256")
    )
    compiler_inputs = {
        "source_lineage": source.as_dict(),
        "source_lineage_sha256": source.authentication_sha256,
        "cache_key": cache_key,
        "cache_compatibility_sha256": compatibility_sha256,
        "cache_payload_binding_sha256": payload_binding_sha256,
        "cache_session_sha256": session_sha256,
        "parent_action_projection_sha256": [],
        "parent_action_projections": [],
    }
    composite_body = {
        "schema": "neural-forge-lfm2-long-horizon-terminal-composite-lineage-v1",
        "compiler_inputs": compiler_inputs,
        "compiler_binding_sha256": frontier.canonical_sha256(compiler_inputs),
        "numeric_projection_diagnostics_are_admission_authority": False,
        "exact_lineage_is_compiler_admission_authority": True,
    }
    return postfold.Lfm2LongHorizonTerminalCompositeLineage(
        frontier._sealed(composite_body, "authentication_sha256")
    )


def _terminal_result():
    generator = torch.Generator().manual_seed(1)
    hidden = 4
    intermediate = 6
    gate = (torch.randn(intermediate, hidden, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    up = (torch.randn(intermediate, hidden, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    incumbent = torch.zeros(hidden, intermediate, dtype=torch.bfloat16)
    target = (torch.randn(hidden, intermediate, generator=generator) * 0.2).to(
        torch.bfloat16
    )
    tied_table = torch.eye(hidden, dtype=torch.bfloat16)
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=incumbent,
        final_norm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        producer_encoder=torch.eye(hidden, dtype=torch.float64),
        target_tied_weight=tied_table,
        source_tied_table=tied_table,
        norm_epsilon=1e-5,
    )

    def evidence(split: str, rows: int, random_seed: int):
        split_generator = torch.Generator().manual_seed(random_seed)
        mlp_input = (torch.randn(rows, hidden, generator=split_generator) * 0.8).to(
            torch.bfloat16
        )
        pre_mlp = (torch.randn(rows, hidden, generator=split_generator) * 0.2).to(
            torch.float64
        )
        features = torch.nn.functional.silu(mlp_input.float() @ gate.float().T) * (
            mlp_input.float() @ up.float().T
        )
        post = pre_mlp + features.double() @ target.double().T
        teacher = post / (post.square().mean(dim=1, keepdim=True) + 1e-5).sqrt()
        return Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(f"{split}:{index}" for index in range(rows)),
            mlp_input=mlp_input,
            pre_mlp_residual=pre_mlp.to(torch.bfloat16),
            parent_final_norm_output=teacher.to(torch.bfloat16),
            sample_weights=torch.ones(rows, dtype=torch.float64),
        )

    banks = {
        split: evidence(split, 24 if split == "fit" else 12, seed_value)
        for split, seed_value in zip(
            ("fit", "fit_check", "selection", "validation"),
            range(2, 6),
            strict=True,
        )
    }
    return compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=banks["fit"],
        fit_check_evidence=banks["fit_check"],
        selection_evidence=banks["selection"],
        validation_evidence=banks["validation"],
    )


def _terminal_result_with_invalid_first_ridge(monkeypatch, *, mode: str):
    original = terminal_action._ridge_delta
    calls = 0

    def adversarial_delta(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        if calls != 1:
            return value
        if mode == "no-op":
            return torch.zeros_like(value)
        if mode == "incomplete-coverage":
            sparse = torch.zeros_like(value)
            sparse[0, 0] = 1.0
            return sparse
        if mode == "nonfinite":
            nonfinite = value.clone()
            nonfinite[0, 0] = torch.nan
            return nonfinite
        raise AssertionError(f"unsupported adversarial mode {mode!r}")

    monkeypatch.setattr(terminal_action, "_ridge_delta", adversarial_delta)
    return _terminal_result()


def _capsule_reference(
    tmp_path,
    result,
    *,
    recipe_id="fit_guided",
    anchor_selection=None,
):
    recipe = postfold.resolve_lfm2_semantic_body_recipe(recipe_id)
    reference = Lfm2SemanticPreterminalCapsuleReference(
        path=tmp_path / "capsule",
        receipt_sha256=_sha("capsule-receipt"),
        receipt_file_sha256=_sha("capsule-receipt-file"),
        manifest_sha256=_sha("capsule-manifest"),
        bindings_sha256=_sha("capsule-bindings"),
        context_binding_sha256=_sha("capsule-context"),
    )
    reference.path.mkdir(exist_ok=True)
    zero = torch.zeros_like(result.incumbent_down_weight)
    model = {
        "final_w2_semantic_sha256": frontier._tensor_sha256(zero),
        "final_w2_terminal_action_sha256": frontier._terminal_tensor_sha256(
            zero,
            "fixture.final_w2",
        ),
        "final_w2_shape": list(result.incumbent_down_weight.shape),
        "final_w2_dtype": "bfloat16",
        "final_w2_nonzero_element_count": 0,
        "final_w2_exact_zero": True,
    }
    bindings = {
        "semantic_cache_key": _sha("cache"),
        "cache_parent_fingerprint": _sha("cache-parent"),
        "bridge_artifact_sha256": _sha("bridge"),
        "parent_final_norm_receipt_sha256": _sha("parent-norm"),
        "mapped_parent_tied_table_receipt_sha256": _sha("mapped-table"),
    }
    layer_receipt = {
        "result_sha256": _sha("layer-result"),
        "cache_selective_session_sha256": _sha("layer-session"),
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "interface_anchor_selection_sha256": (
            None
            if anchor_selection is None
            else anchor_selection["selection_sha256"]
        ),
        **(
            {"interface_anchor_selection": copy.deepcopy(anchor_selection)}
            if anchor_selection is not None
            else {}
        ),
    }
    attention_authentication = {
        "result_sha256": _sha("attention-authentication"),
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
    }
    interface_authentication = {
        "result_sha256": _sha("interface-authentication"),
        "interface_report_sha256": _sha("interface-report"),
    }
    opened = {
        "manifest": {
            "model": model,
            "bindings": bindings,
            "semantic_layer_stack_receipt": layer_receipt,
            "attention_receipt_authentication": attention_authentication,
            "active_interface_authentication": interface_authentication,
        }
    }
    compatibility = frontier._sealed(
        {
            "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
            "cache_key": bindings["semantic_cache_key"],
        },
        "authentication_sha256",
    )
    payload_binding = frontier._sealed(
        {
            "compatibility_authentication_sha256": compatibility[
                "authentication_sha256"
            ],
            "payload_session_sha256": layer_receipt["cache_selective_session_sha256"],
        },
        "authentication_sha256",
    )
    terminal_interface_authentication = frontier._sealed(
        {
            "schema": "neural-forge-lfm2-fused-terminal-tied-interface-v1",
            "interface_report_sha256": interface_authentication[
                "interface_report_sha256"
            ],
        },
        "result_sha256",
    )
    composite_lineage = _synthetic_terminal_composite_lineage(
        cache_key=bindings["semantic_cache_key"],
        cache_parent_fingerprint=bindings["cache_parent_fingerprint"],
        bridge_artifact_sha256=bindings["bridge_artifact_sha256"],
        interface_report_sha256=interface_authentication["interface_report_sha256"],
        compatibility_sha256=compatibility["authentication_sha256"],
        payload_binding_sha256=payload_binding["authentication_sha256"],
        session_sha256=layer_receipt["cache_selective_session_sha256"],
    )
    context = postfold._build_terminal_frontier_export_context(
        terminal=result,
        cache=SimpleNamespace(
            cache_key=bindings["semantic_cache_key"],
            parent_fingerprint=bindings["cache_parent_fingerprint"],
        ),
        bridge=SimpleNamespace(artifact_sha256=bindings["bridge_artifact_sha256"]),
        parent_final_norm=SimpleNamespace(
            result_sha256=bindings["parent_final_norm_receipt_sha256"]
        ),
        mapped_parent_tied_table=torch.eye(4, dtype=torch.bfloat16),
        layer_receipt=layer_receipt,
        terminal_interface_authentication=terminal_interface_authentication,
        terminal_cache_compatibility=compatibility,
        terminal_cache_payload_binding=payload_binding,
        terminal_composite_lineage=composite_lineage,
        terminal_cache_session_sha256=layer_receipt["cache_selective_session_sha256"],
        same_live_cache_session_as_layer_stack=False,
        cache_content_identity_matches_layer_stack=True,
        preterminal_resume_mode=True,
    )
    return reference, opened, context


def _source_roots(tmp_path, capsule_path=None):
    values = tuple(
        tmp_path / name
        for name in (
            "cache",
            "bridge",
            "parent",
            "donor-tokenizer",
            "target-tokenizer",
        )
    )
    for value in values:
        value.mkdir(exist_ok=True)
    return frontier.Lfm2SemanticTerminalFrontierSourceRoots(
        preterminal_capsule_directory=capsule_path or (tmp_path / "capsule"),
        semantic_cache_directory=tmp_path / "cache",
        tokenizer_bridge_directory=tmp_path / "bridge",
        parent_model_directory=tmp_path / "parent",
        donor_tokenizer_directories=(tmp_path / "donor-tokenizer",),
        target_tokenizer_directories=(tmp_path / "target-tokenizer",),
    )


@pytest.fixture(scope="module")
def compiled_result():
    return _terminal_result()


def _publish_bank(
    tmp_path,
    monkeypatch,
    result,
    *,
    recipe_id="fit_guided",
    anchor_selection=None,
):
    reference, opened, context = _capsule_reference(
        tmp_path,
        result,
        recipe_id=recipe_id,
        anchor_selection=anchor_selection,
    )
    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)
    bank_reference = frontier.publish_lfm2_semantic_terminal_frontier_bank(
        result,
        preterminal_capsule_reference=reference,
        export_context=context,
        source_roots=_source_roots(tmp_path, reference.path),
        output_directory=tmp_path / "frontier-bank",
    )
    bank = frontier.load_lfm2_semantic_terminal_frontier_bank(
        bank_reference.path,
        expected_receipt_file_sha256=bank_reference.receipt_file_sha256,
        expected_receipt_sha256=bank_reference.receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=reference.receipt_sha256,
        expected_terminal_result_sha256=result.result_sha256,
    )
    return reference, bank_reference, bank


def test_bank_persists_every_candidate_without_metric_elimination(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    _capsule, reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )

    assert len(reference.candidate_ids) == 36
    assert reference.candidate_ids == tuple(compiled_result.frontier_down_weights)
    assert set(bank.candidate_weights) == set(compiled_result.frontier_down_weights)
    assert bank.manifest["complete_terminal_frontier_candidate_count"] == 36
    assert bank.manifest["structurally_eligible_candidate_count"] == 36
    assert bank.manifest["structurally_ineligible_candidate_ids"] == ()
    assert bank.manifest["structurally_ineligible_candidate_count"] == 0
    assert bank.manifest["all_structurally_eligible_candidates_exported"] is True
    assert bank.manifest["structurally_ineligible_candidates_exported"] is False
    assert bank.manifest["structural_safety_is_only_candidate_veto"] is True
    assert bank.manifest["all_retained_candidates_exported"] is True
    assert bank.manifest["candidate_metric_threshold_used_for_export"] is False
    assert bank.manifest["candidate_metric_threshold_used_for_eligibility"] is False
    assert bank.manifest["numeric_metrics_have_final_intelligence_authority"] is False
    assert bank.manifest["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert bank.manifest["final_selected_candidate_id"] is None
    assert bank.manifest["diagnostic_order_is_provisional"] is True
    assert bank.manifest["diagnostic_order_has_final_selection_authority"] is False
    assert bank.receipt["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert bank.receipt["final_selected_candidate_id"] is None
    assert bank.manifest["body_recipe_id"] == "fit_guided"
    assert bank.receipt["body_recipe_id"] == "fit_guided"
    assert tuple(bank.manifest["complete_body_recipe_candidate_ids"]) == (
        frontier.LFM2_SEMANTIC_BODY_RECIPE_IDS
    )
    assert bank.manifest["one_terminal_frontier_bank_contains_one_body_recipe"] is True
    assert bank.manifest["other_body_recipes_require_sealed_foundation_replay"] is True
    assert bank.manifest["holistic_whole_response_review_required"] is True
    assert bank.manifest["exact_zero_cold_start"] is True
    assert bank.manifest["incumbent_matches_preterminal_capsule_final_w2"] is True
    assert (
        bank.manifest["terminal_frontier_export_context"][
            "same_live_cache_session_as_layer_stack"
        ]
        is False
    )
    for candidate_id, tensor in bank.candidate_weights.items():
        record = next(
            value
            for value in compiled_result.report["frontier"]
            if value["candidate_id"] == candidate_id
        )
        assert frontier._terminal_tensor_sha256(tensor) == record["state_sha256"]


@pytest.mark.parametrize("mode", ("no-op", "incomplete-coverage"))
def test_bank_exports_exact_structural_subset_without_numeric_elimination(
    tmp_path,
    monkeypatch,
    mode,
) -> None:
    result = _terminal_result_with_invalid_first_ridge(monkeypatch, mode=mode)
    report_records = {
        value["candidate_id"]: value for value in result.report["frontier"][1:]
    }
    structurally_eligible = tuple(
        candidate_id
        for candidate_id in result.frontier_down_weights
        if report_records[candidate_id]["structurally_eligible"] is True
    )
    structurally_ineligible = tuple(
        candidate_id
        for candidate_id in result.frontier_down_weights
        if report_records[candidate_id]["structurally_eligible"] is False
    )

    assert len(result.frontier_down_weights) == 36
    assert len(structurally_ineligible) == len(result.policy.alpha_frontier) - 1
    assert any(
        report_records[candidate_id]["numeric_diagnostic_gate_miss"] is True
        for candidate_id in structurally_eligible
    )
    if mode == "no-op":
        assert all(
            report_records[candidate_id]["effectful_nonincumbent"] is False
            for candidate_id in structurally_ineligible
        )
    else:
        assert all(
            report_records[candidate_id]["effectful_nonincumbent"] is True
            and report_records[candidate_id]["structural_safety_passed"] is False
            for candidate_id in structurally_ineligible
        )

    _capsule, reference, bank = _publish_bank(tmp_path, monkeypatch, result)

    assert reference.candidate_ids == structurally_eligible
    assert set(bank.candidate_weights) == set(structurally_eligible)
    assert bank.manifest["complete_terminal_frontier_candidate_count"] == 36
    assert tuple(bank.manifest["complete_terminal_frontier_candidate_ids"]) == tuple(
        result.frontier_down_weights
    )
    assert tuple(bank.manifest["structurally_eligible_candidate_ids"]) == (
        structurally_eligible
    )
    assert tuple(bank.manifest["structurally_ineligible_candidate_ids"]) == (
        structurally_ineligible
    )
    assert bank.manifest["all_structurally_eligible_candidates_exported"] is True
    assert bank.manifest["structurally_ineligible_candidates_exported"] is False
    assert bank.manifest["structural_safety_is_only_candidate_veto"] is True
    assert bank.manifest["candidate_metric_threshold_used_for_export"] is False
    assert bank.manifest["candidate_metric_threshold_used_for_eligibility"] is False


def test_nonfinite_frontier_writer_fails_before_bank_export(
    monkeypatch,
) -> None:
    with pytest.raises(
        terminal_action.Lfm2CpcfTerminalActionResidualError,
        match="nonfinite",
    ):
        _terminal_result_with_invalid_first_ridge(monkeypatch, mode="nonfinite")


def test_bank_rejects_capsule_relabeling(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    reference, opened, context = _capsule_reference(tmp_path, compiled_result)
    opened["manifest"]["model"]["final_w2_semantic_sha256"] = _sha("another-zero-w2")
    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="exact-zero",
    ):
        frontier.publish_lfm2_semantic_terminal_frontier_bank(
            compiled_result,
            preterminal_capsule_reference=reference,
            export_context=context,
            source_roots=_source_roots(tmp_path, reference.path),
            output_directory=tmp_path / "wrong-bank",
        )


def test_bank_never_substitutes_semantic_hash_for_terminal_action_hash(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    reference, opened, context = _capsule_reference(tmp_path, compiled_result)
    model = opened["manifest"]["model"]
    assert model["final_w2_semantic_sha256"] != model["final_w2_terminal_action_sha256"]
    model["final_w2_terminal_action_sha256"] = model["final_w2_semantic_sha256"]
    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="exact-zero",
    ):
        frontier.publish_lfm2_semantic_terminal_frontier_bank(
            compiled_result,
            context,
            preterminal_capsule_reference=reference,
            source_roots=_source_roots(tmp_path, reference.path),
            output_directory=tmp_path / "wrong-hash-domain-bank",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "mapped_parent_tied_table_sha256",
        "tied_interface_authentication_sha256",
        "terminal_cache_compatibility_sha256",
        "terminal_cache_payload_binding_sha256",
        "body_recipe_sha256",
    ),
)
def test_bank_rejects_resealed_compiler_context_live_fact_relabel(
    tmp_path,
    monkeypatch,
    compiled_result,
    field_name,
) -> None:
    reference, opened, issued = _capsule_reference(tmp_path, compiled_result)
    relabeled = dict(issued)
    relabeled[field_name] = _sha(f"relabel:{field_name}")
    relabeled.pop("context_sha256")
    relabeled["context_sha256"] = frontier.canonical_sha256(relabeled)
    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="failed authentication",
    ):
        frontier.publish_lfm2_semantic_terminal_frontier_bank(
            compiled_result,
            relabeled,
            preterminal_capsule_reference=reference,
            source_roots=_source_roots(tmp_path, reference.path),
            output_directory=tmp_path / f"relabel-{field_name}",
        )


def test_bank_rejects_changed_payload_bytes(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule, reference, _bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    payload = reference.path / "terminal-frontier.safetensors"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 1
    payload.write_bytes(raw)

    with pytest.raises(Exception):
        frontier.load_lfm2_semantic_terminal_frontier_bank(
            reference.path,
            expected_receipt_file_sha256=reference.receipt_file_sha256,
            expected_receipt_sha256=reference.receipt_sha256,
            expected_preterminal_capsule_receipt_sha256=capsule.receipt_sha256,
            expected_terminal_result_sha256=compiled_result.result_sha256,
        )


def test_bank_rejects_fully_resealed_structural_inventory_omission(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule, reference, _bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    manifest_path = reference.path / "frontier-manifest.json"
    receipt_path = reference.path / "frontier-receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    omitted = manifest["candidate_ids"].pop()
    manifest["candidate_count"] -= 1
    manifest["structurally_eligible_candidate_ids"].remove(omitted)
    manifest["structurally_eligible_candidate_count"] -= 1
    manifest["candidate_state_sha256"].pop(omitted)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = frontier.canonical_sha256(manifest)
    manifest_raw = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    manifest_path.write_bytes(manifest_raw)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_file"] = {
        "path": "frontier-manifest.json",
        "size_bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    receipt["manifest_sha256"] = manifest["manifest_sha256"]
    receipt["candidate_ids"].remove(omitted)
    receipt["candidate_count"] -= 1
    receipt["structurally_eligible_candidate_ids"].remove(omitted)
    receipt["structurally_eligible_candidate_count"] -= 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = frontier.canonical_sha256(receipt)
    receipt_raw = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    receipt_path.write_bytes(receipt_raw)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="manifest contract",
    ):
        frontier.load_lfm2_semantic_terminal_frontier_bank(
            reference.path,
            expected_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_preterminal_capsule_receipt_sha256=capsule.receipt_sha256,
            expected_terminal_result_sha256=compiled_result.result_sha256,
        )


class _TinyFeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.w2 = torch.nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
        self.w3 = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = _TinyFeedForward()


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyLayer() for _ in range(14)])
        with torch.no_grad():
            self.model.layers[-1].feed_forward.w2.weight.zero_()


def _loaded_capsule(reference, bank, *, anchor_selection=None):
    model = _TinyModel()
    recipe_fields = {
        "body_recipe": frontier._deep_thaw(bank.manifest["body_recipe"]),
        "body_recipe_sha256": bank.manifest["body_recipe_sha256"],
        "body_recipe_id": bank.manifest["body_recipe_id"],
    }
    selected_anchor = (
        frontier._deep_thaw(bank.manifest["interface_anchor_selection"])
        if anchor_selection is None
        and "interface_anchor_selection" in bank.manifest
        else anchor_selection
    )
    if selected_anchor is not None:
        recipe_fields.update(
            {
                "interface_anchor_selection": copy.deepcopy(selected_anchor),
                "interface_anchor_selection_sha256": selected_anchor[
                    "selection_sha256"
                ],
            }
        )
    manifest = {
        "model": {
            "model_state_sha256": fingerprint_model_state(model),
        }
    }
    return Lfm2SemanticPreterminalCapsule(
        path=reference.path,
        target_model=model,
        target_tokenizer=SimpleNamespace(name="tiny-tokenizer"),
        layer_stack=SimpleNamespace(receipt=MappingProxyType(recipe_fields)),
        interface=None,
        attention_receipt_authentication=MappingProxyType(dict(recipe_fields)),
        bindings=None,
        manifest=MappingProxyType(manifest),
        receipt=MappingProxyType({}),
        reference=reference,
    )


def _fake_checkpoint_publisher(*, target_model, output_directory, **kwargs):
    output = output_directory
    output.mkdir()
    (output / "standalone-marker").write_text("dense", encoding="utf-8")
    reloaded = copy.deepcopy(target_model)
    model_state = fingerprint_model_state(reloaded)
    inventory = independent_parameter_inventory(reloaded)
    checkpoint_body = {
        "schema": "neural-forge-lfm2-semantic-postfold-checkpoint-v1",
        "output_directory": str(output),
        "context_binding_sha256": kwargs["context_binding_sha256"],
        "interface_anchor_selection_sha256": kwargs.get(
            "interface_anchor_selection_sha256"
        ),
        "artifact_sha256": _sha(f"artifact:{model_state}"),
        "model_state_sha256": model_state,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "tokenizer_semantics_sha256": _sha("tokenizer"),
        "parameter_element_count": inventory["unique_parameters"],
        "materialized_ffn_width": 6,
        "staged_reload_verified": True,
        "published_reload_verified": True,
        "standalone_dense_safetensors": True,
        "parent_model_required_for_reload": False,
        "cache_required_for_reload": False,
        "network_access_performed": False,
        "operation_audit": {},
    }
    checkpoint = {
        **checkpoint_body,
        "result_sha256": frontier.canonical_sha256(checkpoint_body),
    }
    return reloaded, MappingProxyType(checkpoint)


@pytest.mark.parametrize(
    "mode",
    ("nonfinite", "incomplete-coverage", "no-op"),
)
def test_materializer_reauthenticates_structural_eligibility_before_copy(
    tmp_path,
    monkeypatch,
    compiled_result,
    mode,
) -> None:
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    capsule = _loaded_capsule(capsule_reference, bank)
    candidate_id = bank.reference.candidate_ids[0]
    candidate = bank.candidate_weights[candidate_id].clone()
    if mode == "nonfinite":
        candidate[0, 0] = torch.nan
    elif mode == "incomplete-coverage":
        candidate.zero_()
        candidate[0, 0] = 1.0
    else:
        candidate.zero_()
    report = frontier._deep_thaw(bank.terminal_result_report)
    record = next(
        value
        for value in report["frontier"]
        if value["candidate_id"] == candidate_id
    )
    coverage = terminal_action._terminal_writer_structural_coverage(candidate)
    record.update(
        {
            "state_sha256": frontier._terminal_tensor_sha256(candidate),
            "eligible": True,
            "structurally_eligible": True,
            "effectful_nonincumbent": True,
            "structural_safety_passed": True,
            "weight_norm_gate_passed": True,
            "structural_coverage": dict(coverage),
            "structural_coverage_sha256": coverage["coverage_sha256"],
        }
    )
    corrupted_weights = dict(bank.candidate_weights)
    corrupted_weights[candidate_id] = candidate
    corrupted_bank = frontier.Lfm2SemanticTerminalFrontierBank(
        path=bank.path,
        candidate_weights=MappingProxyType(corrupted_weights),
        terminal_result_report=MappingProxyType(report),
        manifest=bank.manifest,
        receipt=bank.receipt,
        reference=bank.reference,
    )
    monkeypatch.setattr(
        frontier,
        "publish_lfm2_semantic_postfold_checkpoint",
        lambda **_kwargs: pytest.fail(
            "structurally invalid candidate reached checkpoint publication"
        ),
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="structural eligibility veto",
    ):
        frontier._materialize_loaded_candidate(
            capsule=capsule,
            bank=corrupted_bank,
            candidate_id=candidate_id,
            output_directory=tmp_path / "invalid-candidate",
            device="cpu",
            dtype="bfloat16",
        )

    assert not (tmp_path / "invalid-candidate").exists()
    assert int(
        torch.count_nonzero(
            capsule.target_model.model.layers[-1].feed_forward.w2.weight
        ).item()
    ) == 0


@pytest.mark.parametrize(
    "recipe_id",
    frontier.LFM2_SEMANTIC_BODY_RECIPE_IDS,
)
def test_candidate_materialization_changes_only_final_w2_and_stays_unjudged(
    tmp_path,
    monkeypatch,
    compiled_result,
    recipe_id,
) -> None:
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
        recipe_id=recipe_id,
    )
    capsule = _loaded_capsule(capsule_reference, bank)
    monkeypatch.setattr(
        frontier,
        "publish_lfm2_semantic_postfold_checkpoint",
        _fake_checkpoint_publisher,
    )
    candidate_id = bank.reference.candidate_ids[-1]

    reference = frontier._materialize_loaded_candidate(
        capsule=capsule,
        bank=bank,
        candidate_id=candidate_id,
        output_directory=tmp_path / "candidate",
        device="cpu",
        dtype="bfloat16",
    )
    receipt = json.loads(
        (reference.path / "materialization-receipt.json").read_text(encoding="utf-8")
    )

    assert reference.checkpoint_path.is_dir()
    assert receipt["candidate_id"] == candidate_id
    candidate = bank.candidate_weights[candidate_id]
    terminal_digest = frontier._terminal_tensor_sha256(candidate)
    semantic_digest = frontier._tensor_sha256(candidate)
    assert terminal_digest != semantic_digest
    assert receipt["candidate_state_sha256"] == terminal_digest
    assert receipt["materialized_final_w2_terminal_action_sha256"] == terminal_digest
    assert receipt["materialized_final_w2_semantic_sha256"] == semantic_digest
    assert receipt["state_change_proof"]["changed_state_keys"] == [
        "model.layers.13.feed_forward.w2.weight"
    ]
    assert receipt["fresh_reload_state_change_proof"]["changed_state_keys"] == [
        "model.layers.13.feed_forward.w2.weight"
    ]
    assert receipt["exactly_one_state_tensor_changed"] is True
    assert receipt["only_final_layer_w2_changed"] is True
    assert receipt["parameter_identity_and_storage_preserved_during_install"] is True
    assert receipt["candidate_materialized_without_metric_threshold"] is True
    assert receipt["candidate_metric_fields_read_for_materialization"] is False
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["holistic_all_64_whole_response_review_required"] is True
    assert receipt["holistic_review_complete"] is False
    assert receipt["materialization_is_not_intelligence_acceptance"] is True
    assert receipt["body_recipe_id"] == recipe_id
    assert receipt["complete_body_recipe_candidate_count"] == len(
        frontier.LFM2_SEMANTIC_BODY_RECIPE_IDS
    )
    assert receipt["one_terminal_frontier_bank_contains_one_body_recipe"] is True
    assert receipt["other_body_recipes_require_sealed_foundation_replay"] is True
    assert receipt["standalone_child"] is True
    assert receipt["sidecar_required_for_inference"] is False
    assert receipt["training_performed"] is False
    assert receipt["parameters_pruned"] == 0
    assert receipt["parameters_quantized"] == 0
    assert receipt["parameters_sparsified"] == 0

    checkpoint = receipt["checkpoint"]
    metadata = {
        "model_state_sha256": receipt["materialized_model_state_sha256"],
        "parameter_inventory_sha256": checkpoint["parameter_inventory_sha256"],
        "tokenizer_semantics_sha256": checkpoint["tokenizer_semantics_sha256"],
        "context_binding_sha256": receipt["context_binding"]["context_sha256"],
    }

    def reloader(*_args, **_kwargs):
        return copy.deepcopy(capsule.target_model), capsule.target_tokenizer, metadata

    monkeypatch.setattr(
        frontier,
        "load_lfm2_cpcf_v5_standalone_child",
        reloader,
    )
    loaded = frontier.load_lfm2_semantic_terminal_frontier_materialization(
        reference.path,
        expected_receipt_file_sha256=reference.receipt_file_sha256,
        expected_receipt_sha256=reference.receipt_sha256,
        expected_candidate_id=reference.candidate_id,
        expected_candidate_state_sha256=reference.candidate_state_sha256,
        device="cpu",
    )
    assert loaded.receipt["body_recipe_id"] == recipe_id

    wrong_metadata = {**metadata, "context_binding_sha256": _sha("wrong-context")}
    monkeypatch.setattr(
        frontier,
        "load_lfm2_cpcf_v5_standalone_child",
        lambda *_args, **_kwargs: (
            copy.deepcopy(capsule.target_model),
            capsule.target_tokenizer,
            wrong_metadata,
        ),
    )
    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="differs from its provenance",
    ):
        frontier.load_lfm2_semantic_terminal_frontier_materialization(
            reference.path,
            expected_receipt_file_sha256=reference.receipt_file_sha256,
            expected_receipt_sha256=reference.receipt_sha256,
            expected_candidate_id=reference.candidate_id,
            expected_candidate_state_sha256=reference.candidate_state_sha256,
            device="cpu",
        )


def test_nondefault_anchor_selection_propagates_through_frontier_materialization(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    selection = _anchor_selection(9)
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
        anchor_selection=selection,
    )

    assert frontier._deep_thaw(bank.manifest["interface_anchor_selection"]) == selection
    assert frontier._deep_thaw(bank.receipt["interface_anchor_selection"]) == selection
    assert bank.manifest["interface_anchor_selection_sha256"] == selection[
        "selection_sha256"
    ]
    assert bank.receipt["interface_anchor_selection_sha256"] == selection[
        "selection_sha256"
    ]
    assert bank.manifest["terminal_frontier_export_context"][
        "interface_anchor_selection_sha256"
    ] == selection["selection_sha256"]

    capsule = _loaded_capsule(capsule_reference, bank)
    monkeypatch.setattr(
        frontier,
        "publish_lfm2_semantic_postfold_checkpoint",
        _fake_checkpoint_publisher,
    )
    reference = frontier._materialize_loaded_candidate(
        capsule=capsule,
        bank=bank,
        candidate_id=bank.reference.candidate_ids[-1],
        output_directory=tmp_path / "candidate-9-anchor",
        device="cpu",
        dtype="bfloat16",
    )
    receipt = json.loads(
        (reference.path / "materialization-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["interface_anchor_selection"] == selection
    assert receipt["interface_anchor_selection_sha256"] == selection[
        "selection_sha256"
    ]
    assert receipt["context_binding"]["interface_anchor_selection"] == selection
    assert receipt["checkpoint"]["interface_anchor_selection_sha256"] == selection[
        "selection_sha256"
    ]
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["holistic_review_complete"] is False


def test_materialization_rejects_cross_candidate_anchor_substitution_before_mutation(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    selected = _anchor_selection(9)
    substituted = _anchor_selection(8)
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
        anchor_selection=selected,
    )
    capsule = _loaded_capsule(
        capsule_reference,
        bank,
        anchor_selection=substituted,
    )
    before = (
        capsule.target_model.model.layers[-1].feed_forward.w2.weight.detach().clone()
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="disagree on anchor selection",
    ):
        frontier._materialize_loaded_candidate(
            capsule=capsule,
            bank=bank,
            candidate_id=bank.reference.candidate_ids[-1],
            output_directory=tmp_path / "substituted-anchor",
            device="cpu",
            dtype="bfloat16",
        )

    assert torch.equal(
        before,
        capsule.target_model.model.layers[-1].feed_forward.w2.weight,
    )
    assert not (tmp_path / "substituted-anchor").exists()


def test_materialization_refuses_unknown_candidate(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    capsule = _loaded_capsule(capsule_reference, bank)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="not in the authenticated retained bank",
    ):
        frontier._materialize_loaded_candidate(
            capsule=capsule,
            bank=bank,
            candidate_id="metric-winner-only",
            output_directory=tmp_path / "unknown",
            device="cpu",
            dtype="bfloat16",
        )


def test_bank_publication_loses_destination_race_without_replacing_it(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    reference, opened, context = _capsule_reference(tmp_path, compiled_result)
    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)
    real_rename = frontier._atomic_rename_noreplace
    destination = tmp_path / "raced-bank"

    def race(source, target):
        target.mkdir()
        (target / "owner").write_text("other", encoding="utf-8")
        return real_rename(source, target)

    monkeypatch.setattr(frontier, "_atomic_rename_noreplace", race)
    with pytest.raises(Exception):
        frontier.publish_lfm2_semantic_terminal_frontier_bank(
            compiled_result,
            preterminal_capsule_reference=reference,
            export_context=context,
            source_roots=_source_roots(tmp_path, reference.path),
            output_directory=destination,
        )

    assert (destination / "owner").read_text(encoding="utf-8") == "other"


def test_capsule_resume_orchestrator_uses_two_argument_export_seam(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, opened, export_context = _capsule_reference(
        tmp_path,
        compiled_result,
    )
    _source_roots(tmp_path, capsule_reference.path)
    cache_manifest_sha = _sha("cache-manifest")
    opened["manifest"]["bindings"]["semantic_cache_manifest_sha256"] = (
        cache_manifest_sha
    )
    bindings = SimpleNamespace(
        semantic_cache_key=opened["manifest"]["bindings"]["semantic_cache_key"],
        semantic_cache_manifest_sha256=cache_manifest_sha,
        cache_parent_fingerprint=opened["manifest"]["bindings"][
            "cache_parent_fingerprint"
        ],
        cache_selective_session_sha256=opened["manifest"][
            "semantic_layer_stack_receipt"
        ]["cache_selective_session_sha256"],
        bridge_artifact_sha256=opened["manifest"]["bindings"]["bridge_artifact_sha256"],
        parent_final_norm_receipt_sha256=opened["manifest"]["bindings"][
            "parent_final_norm_receipt_sha256"
        ],
        mapped_parent_tied_table_receipt_sha256=opened["manifest"]["bindings"][
            "mapped_parent_tied_table_receipt_sha256"
        ],
    )
    cache = SimpleNamespace(
        cache_key=bindings.semantic_cache_key,
        parent_fingerprint=bindings.cache_parent_fingerprint,
        manifest={"manifest_sha256": cache_manifest_sha},
    )
    bridge = SimpleNamespace(artifact_sha256=bindings.bridge_artifact_sha256)
    parent_norm = SimpleNamespace(
        result_sha256=bindings.parent_final_norm_receipt_sha256
    )
    tied_interface_authentication = {
        "result_sha256": export_context["tied_interface_authentication_sha256"],
        "interface_report_sha256": export_context[
            "mapped_parent_tied_table_interface_report_sha256"
        ],
    }
    capsule = SimpleNamespace(
        bindings=bindings,
        target_model=object(),
        target_tokenizer=SimpleNamespace(name="target"),
        layer_stack=SimpleNamespace(
            receipt=opened["manifest"]["semantic_layer_stack_receipt"]
        ),
        attention_receipt_authentication=opened["manifest"][
            "attention_receipt_authentication"
        ],
        interface=object(),
        reference=capsule_reference,
    )
    calls = {}

    monkeypatch.setattr(frontier, "_open_outer_capsule", lambda *_a, **_k: opened)
    monkeypatch.setattr(
        frontier,
        "open_lfm2_circuit_trace_cache",
        lambda path, *, verify: cache,
    )
    monkeypatch.setattr(
        frontier,
        "preflight_lfm2_long_horizon_terminal_cache",
        lambda value: {
            "authentication_sha256": export_context[
                "terminal_cache_compatibility_sha256"
            ]
        },
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_preterminal_capsule",
        lambda *_a, **_k: capsule,
    )
    monkeypatch.setattr(
        frontier,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_a, **_k: bridge,
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_a, **_k: parent_norm,
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_a, **_k: (
            torch.eye(4, dtype=torch.bfloat16),
            {"result_sha256": bindings.mapped_parent_tied_table_receipt_sha256},
        ),
    )
    monkeypatch.setattr(
        frontier,
        "build_lfm2_long_horizon_terminal_source_lineage",
        lambda **_kwargs: postfold.Lfm2LongHorizonTerminalSourceLineage(
            export_context["terminal_composite_lineage"]["compiler_inputs"][
                "source_lineage"
            ]
        ),
    )
    monkeypatch.setattr(
        frontier,
        "_authenticate_fused_terminal_tied_interface",
        lambda **_kwargs: tied_interface_authentication,
    )
    monkeypatch.setattr(
        frontier,
        "_authenticate_fused_terminal_cache_evidence",
        lambda _receipt: {
            "terminal_cache_compatibility_sha256": export_context[
                "terminal_cache_compatibility_sha256"
            ],
            "terminal_cache_payload_binding_sha256": export_context[
                "terminal_cache_payload_binding_sha256"
            ],
            "terminal_cache_session_sha256": export_context[
                "terminal_cache_session_sha256"
            ],
        },
    )

    def fused_terminal(**kwargs):
        calls.update(kwargs)
        assert kwargs["authenticated_cache_session"] is None
        assert kwargs["preterminal_resume_mode"] is True
        assert kwargs["body_recipe"] == postfold.resolve_lfm2_semantic_body_recipe(
            "fit_guided"
        )
        reference = kwargs["terminal_frontier_exporter"](
            compiled_result,
            export_context,
        )
        body = {
            "terminal_result": compiled_result.as_dict(),
            "terminal_review_pending_schema": (
                compiled_result.report["review_pending_schema"]
            ),
            "final_selection_status": "deferred-to-manual-whole-response-review",
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "manual_whole_response_review_required": True,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "accepted": False,
            "terminal_publication_decision_accepted": False,
            "terminal_frontier_export_context": dict(export_context),
            "terminal_frontier_export_context_sha256": export_context["context_sha256"],
            "terminal_frontier_export": reference.as_dict(),
            "terminal_frontier_export_sha256": frontier.canonical_sha256(
                reference.as_dict()
            ),
            "terminal_frontier_exported_before_w2_installation": True,
            "tied_interface_authentication": tied_interface_authentication,
            "tied_interface_authentication_sha256": (
                tied_interface_authentication["result_sha256"]
            ),
        }
        return {**body, "result_sha256": frontier.canonical_sha256(body)}

    result = frontier.compile_lfm2_semantic_terminal_frontier_bank_from_capsule(
        preterminal_capsule_directory=tmp_path / "capsule",
        expected_preterminal_receipt_file_sha256=(
            capsule_reference.receipt_file_sha256
        ),
        expected_preterminal_receipt_sha256=capsule_reference.receipt_sha256,
        expected_preterminal_bindings_sha256=capsule_reference.bindings_sha256,
        cache_directory=tmp_path / "cache",
        expected_cache_key=bindings.semantic_cache_key,
        bridge_directory=tmp_path / "bridge",
        expected_bridge_artifact_sha256=bindings.bridge_artifact_sha256,
        donor_tokenizer=SimpleNamespace(name="donor"),
        parent_model_directory=tmp_path / "parent",
        output_directory=tmp_path / "bank",
        device="cpu",
        fused_terminal_compiler=fused_terminal,
    )

    assert calls["terminal_frontier_exporter"].__code__.co_argcount == 2
    assert result.bank_reference.path == tmp_path / "bank"
    assert result.bank_reference.candidate_ids == tuple(
        compiled_result.frontier_down_weights
    )
    assert result.as_dict()["preterminal_resume_mode"] is True
    assert result.as_dict()["same_live_cache_session_as_layer_stack"] is False
    assert result.as_dict()["candidate_metric_threshold_used_for_export"] is False


def test_direct_bank_publisher_rejects_nested_capsule_output_before_load(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    reference, _opened, context = _capsule_reference(tmp_path, compiled_result)
    monkeypatch.setattr(
        frontier,
        "_open_outer_capsule",
        lambda *_a, **_k: pytest.fail("overlap reached capsule loading"),
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="disjoint",
    ):
        frontier.publish_lfm2_semantic_terminal_frontier_bank(
            compiled_result,
            context,
            preterminal_capsule_reference=reference,
            source_roots=_source_roots(tmp_path, reference.path),
            output_directory=reference.path / "nested-bank",
        )


@pytest.mark.parametrize(
    "source_name",
    ("capsule", "cache", "bridge", "parent", "tokenizer"),
)
def test_resume_rejects_every_source_overlap_before_cache_load(
    tmp_path,
    monkeypatch,
    compiled_result,
    source_name,
) -> None:
    reference, _opened, _context = _capsule_reference(tmp_path, compiled_result)
    _source_roots(tmp_path, reference.path)
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    sources = {
        "capsule": reference.path,
        "cache": tmp_path / "cache",
        "bridge": tmp_path / "bridge",
        "parent": tmp_path / "parent",
        "tokenizer": tokenizer_root,
    }
    monkeypatch.setattr(
        frontier,
        "open_lfm2_circuit_trace_cache",
        lambda *_a, **_k: pytest.fail("overlap reached cache loading"),
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="disjoint",
    ):
        frontier.compile_lfm2_semantic_terminal_frontier_bank_from_capsule(
            preterminal_capsule_directory=reference.path,
            expected_preterminal_receipt_file_sha256=reference.receipt_file_sha256,
            expected_preterminal_receipt_sha256=reference.receipt_sha256,
            expected_preterminal_bindings_sha256=reference.bindings_sha256,
            cache_directory=tmp_path / "cache",
            expected_cache_key=_sha("cache"),
            bridge_directory=tmp_path / "bridge",
            expected_bridge_artifact_sha256=_sha("bridge"),
            donor_tokenizer=SimpleNamespace(name_or_path=str(tokenizer_root)),
            parent_model_directory=tmp_path / "parent",
            output_directory=sources[source_name] / "nested-output",
            device="cpu",
        )


def test_private_materializer_rejects_nested_bank_output_before_w2_mutation(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, _bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    capsule = _loaded_capsule(capsule_reference, bank)
    before = (
        capsule.target_model.model.layers[-1].feed_forward.w2.weight.detach().clone()
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="disjoint",
    ):
        frontier._materialize_loaded_candidate(
            capsule=capsule,
            bank=bank,
            candidate_id=bank.reference.candidate_ids[0],
            output_directory=bank.path / "nested-child",
            device="cpu",
            dtype="bfloat16",
        )

    assert torch.equal(
        before,
        capsule.target_model.model.layers[-1].feed_forward.w2.weight,
    )


def test_public_materializer_rejects_nested_bank_output_before_any_load(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, bank_reference, _bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_preterminal_capsule",
        lambda *_a, **_k: pytest.fail("overlap reached capsule loading"),
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_terminal_frontier_bank",
        lambda *_a, **_k: pytest.fail("overlap reached bank loading"),
    )

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="disjoint",
    ):
        frontier.materialize_lfm2_semantic_terminal_frontier_candidate(
            preterminal_capsule_directory=capsule_reference.path,
            expected_preterminal_receipt_file_sha256=(
                capsule_reference.receipt_file_sha256
            ),
            expected_preterminal_receipt_sha256=capsule_reference.receipt_sha256,
            expected_preterminal_bindings_sha256=capsule_reference.bindings_sha256,
            terminal_frontier_bank_directory=bank_reference.path,
            expected_terminal_frontier_bank_receipt_file_sha256=(
                bank_reference.receipt_file_sha256
            ),
            expected_terminal_frontier_bank_receipt_sha256=(
                bank_reference.receipt_sha256
            ),
            expected_terminal_result_sha256=bank_reference.terminal_result_sha256,
            expected_source_roots=_source_roots(
                tmp_path,
                capsule_reference.path,
            ),
            candidate_id=bank_reference.candidate_ids[0],
            output_directory=bank_reference.path / "nested-public-child",
            device="cpu",
        )


def _public_materializer_arguments(
    capsule_reference,
    bank_reference,
    source_roots,
    output_directory,
):
    return {
        "preterminal_capsule_directory": capsule_reference.path,
        "expected_preterminal_receipt_file_sha256": (
            capsule_reference.receipt_file_sha256
        ),
        "expected_preterminal_receipt_sha256": capsule_reference.receipt_sha256,
        "expected_preterminal_bindings_sha256": capsule_reference.bindings_sha256,
        "terminal_frontier_bank_directory": bank_reference.path,
        "expected_terminal_frontier_bank_receipt_file_sha256": (
            bank_reference.receipt_file_sha256
        ),
        "expected_terminal_frontier_bank_receipt_sha256": (
            bank_reference.receipt_sha256
        ),
        "expected_terminal_result_sha256": bank_reference.terminal_result_sha256,
        "expected_source_roots": source_roots,
        "candidate_id": bank_reference.candidate_ids[0],
        "output_directory": output_directory,
        "device": "cpu",
    }


def test_public_materializer_rejects_role_root_equal_ancestor_and_descendant(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, bank_reference, bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    source_roots = _source_roots(tmp_path, capsule_reference.path)
    sealed = bank.manifest["authenticated_source_roots"]
    assert frontier._deep_thaw(sealed) == source_roots.as_dict()
    assert frontier._deep_thaw(bank.receipt["authenticated_source_roots"]) == (
        source_roots.as_dict()
    )
    assert bank.reference.source_roots_sha256 == source_roots.source_roots_sha256
    role_paths = {
        **{
            role: tuple(Path(value) for value in values)
            for role, values in sealed["role_paths"].items()
        },
        "terminal_frontier_bank": (bank_reference.path,),
    }
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_preterminal_capsule",
        lambda *_a, **_k: pytest.fail("overlap reached capsule loading"),
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_terminal_frontier_bank",
        lambda *_a, **_k: pytest.fail("overlap reached payload loading"),
    )

    for role, roots in role_paths.items():
        for index, root in enumerate(roots):
            cases = {
                "equal": root,
                "ancestor": root.parent,
                "descendant": root / f"nested-{role}-{index}",
            }
            for relation, output in cases.items():
                with pytest.raises(frontier.Lfm2SemanticTerminalFrontierError):
                    frontier.materialize_lfm2_semantic_terminal_frontier_candidate(
                        **_public_materializer_arguments(
                            capsule_reference,
                            bank_reference,
                            source_roots,
                            output,
                        )
                    )
                if relation == "descendant":
                    assert not output.exists()


def test_public_materializer_rejects_omitted_or_mismatched_source_root_pins(
    tmp_path,
    monkeypatch,
    compiled_result,
) -> None:
    capsule_reference, bank_reference, _bank = _publish_bank(
        tmp_path,
        monkeypatch,
        compiled_result,
    )
    source_roots = _source_roots(tmp_path, capsule_reference.path)
    monkeypatch.setattr(
        frontier,
        "load_lfm2_semantic_preterminal_capsule",
        lambda *_a, **_k: pytest.fail("source mismatch reached capsule loading"),
    )
    alternate = tmp_path / "alternate-donor-tokenizer"
    alternate.mkdir()
    mismatched = frontier.Lfm2SemanticTerminalFrontierSourceRoots(
        preterminal_capsule_directory=source_roots.preterminal_capsule_directory,
        semantic_cache_directory=source_roots.semantic_cache_directory,
        tokenizer_bridge_directory=source_roots.tokenizer_bridge_directory,
        parent_model_directory=source_roots.parent_model_directory,
        donor_tokenizer_directories=(alternate,),
        target_tokenizer_directories=source_roots.target_tokenizer_directories,
    )

    omitted = _public_materializer_arguments(
        capsule_reference,
        bank_reference,
        source_roots,
        tmp_path / "omitted-output",
    )
    omitted["expected_source_roots"] = None
    with pytest.raises(TypeError, match="explicitly pin"):
        frontier.materialize_lfm2_semantic_terminal_frontier_candidate(**omitted)

    with pytest.raises(
        frontier.Lfm2SemanticTerminalFrontierError,
        match="differ from sealed bank provenance",
    ):
        frontier.materialize_lfm2_semantic_terminal_frontier_candidate(
            **_public_materializer_arguments(
                capsule_reference,
                bank_reference,
                mismatched,
                tmp_path / "mismatched-output",
            )
        )
