from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold_frontier as frontier  # noqa: E402
import neural_forge.hf_lfm2_semantic_preterminal_capsule as capsule  # noqa: E402
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionEvidence,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (  # noqa: E402
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    Lfm2SemanticLayerStackInstallation,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)
from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    TiedResidualAnchorTransportWitness,
    authenticate_tied_residual_congruence_interface,
    compile_tied_residual_congruence_candidate,
)


def _sealed(body: dict[str, object], seal: str) -> dict[str, object]:
    return {**body, seal: canonical_sha256(body)}


def _problem(
    tmp_path: Path,
    monkeypatch,
    *,
    mutate_during_rename: bool = False,
    hybrid: bool = False,
):
    evidence = {}
    for split_index, split in enumerate(
        ("fit", "fit_check", "selection", "validation")
    ):
        evidence[split] = Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(f"{split}:{index}" for index in range(3)),
            mlp_input=torch.full(
                (3, 2),
                float(split_index + 1),
                dtype=torch.bfloat16,
            ),
            pre_mlp_residual=torch.full(
                (3, 2),
                float(split_index + 2),
                dtype=torch.bfloat16,
            ),
            parent_final_norm_output=torch.full(
                (3, 2),
                float(split_index + 3),
                dtype=torch.bfloat16,
            ),
            sample_weights=torch.ones(3, dtype=torch.float64),
        )
    layer_body = {
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "cache_selective_session_sha256": "c" * 64,
        "final_layer_mlp_evidence_sha256": {
            split: evidence[split].authentication_sha256 for split in sorted(evidence)
        },
    }
    layer_stack = Lfm2SemanticLayerStackInstallation(
        receipt=_sealed(layer_body, "result_sha256"),
        final_layer_mlp_evidence=evidence,
    )
    source_table = None
    anchor_transport_witness = None
    if hybrid:
        source_table = torch.tensor(
            [[1.0, 0.25], [-0.5, 1.5]],
            dtype=torch.float64,
        )
        base = torch.tensor(
            [[0.5, -0.25], [0.75, 1.0]],
            dtype=torch.float64,
        )
        residual = torch.tensor(
            [[0.125, -0.0625], [-0.25, 0.1875]],
            dtype=torch.float64,
        )
        encoder = torch.tensor(
            [[1.5, 0.25], [0.125, 0.75]],
            dtype=torch.float64,
        )
        selected_target = base + residual
        source_body = {
            "schema": TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
            "target_weight_sha256": postfold._foundation_lineage_tensor_sha256(
                selected_target
            ),
            "producer_encoder_sha256": (
                postfold._foundation_lineage_tensor_sha256(encoder)
            ),
        }
        source_interface = TiedCausalInterfaceFold(
            target_weight=selected_target,
            producer_encoder=encoder,
            report=_sealed(source_body, "report_sha256"),
        )
        candidate = compile_tied_residual_congruence_candidate(
            source_interface,
            source_table,
            mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
            anchor_base_target_weight=base,
            anchor_source_residual_weight=residual,
        )
        interface = candidate.interface
        anchor_transport_witness = candidate.anchor_transport_witness
        interface_authentication = authenticate_tied_residual_congruence_interface(
            interface,
            mapped_parent_tied_table=source_table,
            anchor_transport_witness=anchor_transport_witness,
        )
        layer_body.update(
            {
                "anchor_residual_transport_authentication": dict(
                    interface_authentication
                ),
                "anchor_residual_transport_authentication_sha256": (
                    interface_authentication["result_sha256"]
                ),
            }
        )
        layer_stack = Lfm2SemanticLayerStackInstallation(
            receipt=_sealed(layer_body, "result_sha256"),
            final_layer_mlp_evidence=evidence,
        )
    else:
        interface_body = {"schema": "synthetic-interface-v1"}
        interface = TiedCausalInterfaceFold(
            target_weight=torch.eye(2, dtype=torch.float64),
            producer_encoder=torch.eye(2, dtype=torch.float64),
            report=_sealed(interface_body, "report_sha256"),
        )
        interface_authentication = _sealed(
            {"schema": "synthetic-interface-authentication-v1"},
            "result_sha256",
        )
    attention = _sealed(
        {"schema": "synthetic-attention-authentication-v1"},
        "result_sha256",
    )
    bindings = capsule.Lfm2SemanticPreterminalCapsuleBindings(
        input_artifact_sha256="1" * 64,
        compiler_diagnostic_sha256="2" * 64,
        compiler_cache_key="3" * 64,
        semantic_cache_key="4" * 64,
        semantic_cache_manifest_sha256="5" * 64,
        cache_parent_fingerprint="6" * 64,
        bridge_artifact_sha256="7" * 64,
        mapped_parent_tied_table_receipt_sha256="8" * 64,
        parent_final_norm_receipt_sha256="9" * 64,
        interface_rehydration_sha256="a" * 64,
        tied_residual_interface_gauge_sha256="b" * 64,
        cache_selective_session_sha256="c" * 64,
    )
    inventory = {
        "inventory_sha256": "d" * 64,
        "parameter_element_count": 12,
    }
    coverage = _sealed(
        {
            "deferred_roles": ["feed_forward.w2.weight"],
            "all_required_roles_structurally_nonzero": True,
        },
        "result_sha256",
    )
    seam = {
        "model_identity_sha256": "e" * 64,
        "model_state_sha256": "f" * 64,
        "parameter_inventory": inventory,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "parameter_element_count": 12,
        "tokenizer_semantics_sha256": "0" * 64,
        "materialized_ffn_width": 4,
        "final_w2_parameter_name": "model.layers.13.feed_forward.w2.weight",
        "final_w2_semantic_sha256": "1" * 64,
        "final_w2_terminal_action_sha256": "2" * 64,
        "final_w2_shape": [2, 4],
        "final_w2_dtype": "bfloat16",
        "final_w2_nonzero_element_count": 0,
        "final_w2_exact_zero": True,
        "final_mlp_producer_coverage": coverage,
        "interface_authentication": interface_authentication,
        "attention_receipt_authentication": attention,
    }
    model = SimpleNamespace(
        final_w2=torch.zeros((2, 4), dtype=torch.bfloat16),
    )
    tokenizer = object()

    def validate_live_seam(**kwargs):
        assert int(torch.count_nonzero(kwargs["target_model"].final_w2).item()) == 0
        return seam

    def publish_checkpoint(**kwargs):
        checkpoint = kwargs["checkpoint"]
        checkpoint.mkdir()
        tensor_path = checkpoint / "model.safetensors"
        save_file({"weight": torch.zeros(1, dtype=torch.bfloat16)}, str(tensor_path))
        tensor_bytes = tensor_path.read_bytes()
        tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
        metadata_body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
            "dense_safetensor_files": [
                {
                    "path": tensor_path.name,
                    "size_bytes": len(tensor_bytes),
                    "sha256": tensor_sha256,
                }
            ],
            "tokenizer_package_files": [],
        }
        metadata = {
            **metadata_body,
            "artifact_sha256": canonical_sha256(metadata_body),
        }
        (checkpoint / LFM2_CPCF_V5_ARTIFACT_METADATA).write_bytes(
            json.dumps(
                metadata,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
        inventory_rows = list(
            capsule.canonical_standalone_fold_artifact_inventory(checkpoint)
        )
        tree = list(
            capsule.scan_standalone_fold_artifact_tree(
                checkpoint,
                expected_file_roles={
                    row["path"]: row["artifact_role"] for row in inventory_rows
                },
            )
        )
        return {
            "artifact_metadata": metadata,
            "artifact_inventory": inventory_rows,
            "artifact_inventory_sha256": canonical_sha256(inventory_rows),
            "artifact_tree": tree,
            "artifact_tree_sha256": canonical_sha256(tree),
        }

    def load_checkpoint(**kwargs):
        assert (kwargs["checkpoint"] / "model.safetensors").is_file()
        return model, tokenizer

    monkeypatch.setattr(capsule, "_validate_live_seam", validate_live_seam)
    monkeypatch.setattr(
        capsule,
        "_publish_quarantine_checkpoint",
        publish_checkpoint,
    )
    monkeypatch.setattr(capsule, "_load_quarantine_checkpoint", load_checkpoint)
    if mutate_during_rename:
        atomic_rename = capsule._atomic_rename_noreplace

        def mutate_then_rename(source, destination):
            checkpoint_tensor = source / "checkpoint" / "model.safetensors"
            checkpoint_tensor.write_bytes(checkpoint_tensor.read_bytes() + b"tamper")
            return atomic_rename(source, destination)

        monkeypatch.setattr(
            capsule,
            "_atomic_rename_noreplace",
            mutate_then_rename,
        )
    output = tmp_path / "preterminal"
    reference = capsule.publish_lfm2_semantic_preterminal_capsule(
        target_model=model,
        target_tokenizer=tokenizer,
        layer_stack=layer_stack,
        interface=interface,
        attention_receipt_authentication=attention,
        bindings=bindings,
        output_directory=output,
        mapped_parent_tied_table=source_table,
        anchor_transport_witness=anchor_transport_witness,
    )
    return SimpleNamespace(
        reference=reference,
        output=output,
        model=model,
        tokenizer=tokenizer,
        evidence=evidence,
        layer_stack=layer_stack,
        interface=interface,
        attention=attention,
        bindings=bindings,
        source_table=source_table,
        anchor_transport_witness=anchor_transport_witness,
    )


def _load(problem):
    reference = problem.reference
    return capsule.load_lfm2_semantic_preterminal_capsule(
        problem.output,
        expected_receipt_file_sha256=reference.receipt_file_sha256,
        expected_receipt_sha256=reference.receipt_sha256,
        expected_bindings_sha256=reference.bindings_sha256,
        device="cpu",
    )


def test_read_only_parameter_inventory_is_thawed_before_canonical_sealing(
    monkeypatch,
) -> None:
    inventory = {
        "method": "unique-materialized-parameters",
        "unique_parameters": 12,
        "independent_methods_agree": True,
        "alias_groups": [],
        "inventory_sha256": "d" * 64,
    }
    readonly_inventory = MappingProxyType(inventory)
    monkeypatch.setattr(
        capsule,
        "independent_parameter_inventory",
        lambda _model: readonly_inventory,
    )

    canonical = capsule._canonical_parameter_inventory(object())

    assert canonical == inventory
    assert isinstance(canonical, dict)


def test_preterminal_capsule_round_trip_exact_banks_and_zero_w2(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    loaded = _load(problem)

    assert loaded.target_model is problem.model
    assert loaded.target_tokenizer is problem.tokenizer
    assert int(torch.count_nonzero(loaded.target_model.final_w2).item()) == 0
    assert loaded.manifest["model"]["final_w2_exact_zero"] is True
    assert loaded.manifest["model"]["final_w2_nonzero_element_count"] == 0
    assert "final_w2_sha256" not in loaded.manifest["model"]
    assert loaded.manifest["model"]["final_w2_semantic_sha256"] == "1" * 64
    assert loaded.manifest["model"]["final_w2_terminal_action_sha256"] == "2" * 64
    assert (
        loaded.manifest["context_binding"]["deferred_final_w2_terminal_action_sha256"]
        == "2" * 64
    )
    assert loaded.manifest["payload_tensor_count"] == 18
    assert len(loaded.manifest["payload_tensor_names"]) == 18
    assert set(loaded.layer_stack.final_layer_mlp_evidence) == set(
        ("fit", "fit_check", "selection", "validation")
    )
    for split, expected in problem.evidence.items():
        actual = loaded.layer_stack.final_layer_mlp_evidence[split]
        assert actual.authentication_sha256 == expected.authentication_sha256
        assert actual.verify_authentication() is True
        torch.testing.assert_close(actual.mlp_input, expected.mlp_input)
    assert loaded.receipt["terminal_complete"] is False
    assert loaded.receipt["promotion_authorized"] is False
    assert loaded.receipt["behavior_capability_proven"] is False


def test_hybrid_capsule_round_trip_persists_typed_witness_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch, hybrid=True)
    loaded = _load(problem)

    assert loaded.manifest["payload_tensor_count"] == 22
    assert loaded.manifest["anchor_residual_transport_inputs_persisted"] is True
    assert isinstance(
        loaded.anchor_transport_witness,
        TiedResidualAnchorTransportWitness,
    )
    assert (
        loaded.anchor_transport_witness.witness_sha256
        == problem.anchor_transport_witness.witness_sha256
    )
    authentication = authenticate_tied_residual_congruence_interface(
        loaded.interface,
        mapped_parent_tied_table=problem.source_table,
        anchor_transport_witness=loaded.anchor_transport_witness,
    )
    assert authentication["anchor_residual_formula_recomputed_exact"] is True

    payload_path = problem.output / "preterminal-seam.safetensors"
    payload = load_file(str(payload_path), device="cpu")
    payload["anchor_source.residual_weight"][0, 0] += 0.5
    save_file(payload, str(payload_path))
    with pytest.raises(Exception):
        _load(problem)


def test_hybrid_capsule_publish_load_frontier_resume_preserves_exact_witness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch, hybrid=True)
    recipe = postfold.resolve_lfm2_semantic_body_recipe("fit_guided")
    cache_directory = tmp_path / "cache"
    bridge_directory = tmp_path / "bridge"
    parent_directory = tmp_path / "parent"
    for directory in (cache_directory, bridge_directory, parent_directory):
        directory.mkdir()

    cache = SimpleNamespace(
        cache_key=problem.bindings.semantic_cache_key,
        parent_fingerprint=problem.bindings.cache_parent_fingerprint,
        manifest={
            "manifest_sha256": problem.bindings.semantic_cache_manifest_sha256
        },
    )
    bridge = SimpleNamespace(
        artifact_sha256=problem.bindings.bridge_artifact_sha256
    )
    parent_norm = SimpleNamespace(
        result_sha256=problem.bindings.parent_final_norm_receipt_sha256
    )
    compatibility_sha256 = "a" * 64
    payload_binding_sha256 = "b" * 64
    terminal_result_sha256 = "d" * 64
    seen: dict[str, object] = {}
    capsule_manifest = json.loads(
        (problem.output / "capsule-manifest.json").read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        frontier,
        "_authenticated_body_recipe",
        lambda *_args, **_kwargs: recipe,
    )
    monkeypatch.setattr(
        frontier,
        "open_lfm2_circuit_trace_cache",
        lambda *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(
        frontier,
        "preflight_lfm2_long_horizon_terminal_cache",
        lambda _cache: {"authentication_sha256": compatibility_sha256},
    )
    monkeypatch.setattr(
        frontier,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_args, **_kwargs: bridge,
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_args, **_kwargs: parent_norm,
    )
    monkeypatch.setattr(
        frontier,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_args, **_kwargs: (
            problem.source_table,
            {
                "result_sha256": (
                    problem.bindings.mapped_parent_tied_table_receipt_sha256
                )
            },
        ),
    )

    def authenticate_interface(**kwargs):
        witness = kwargs["anchor_transport_witness"]
        seen["authentication_witness"] = witness
        exact = authenticate_tied_residual_congruence_interface(
            kwargs["interface"],
            mapped_parent_tied_table=kwargs["mapped_parent_tied_table"],
            anchor_transport_witness=witness,
        )
        assert exact["anchor_residual_formula_recomputed_exact"] is True
        publication_authentication = kwargs["layer_receipt"][
            "anchor_residual_transport_authentication"
        ]
        assert dict(exact) == dict(publication_authentication)
        assert dict(exact) == capsule_manifest["active_interface_authentication"]
        assert kwargs["layer_receipt"][
            "anchor_residual_transport_authentication_sha256"
        ] == exact["result_sha256"]
        seen["fresh_interface_authentication"] = dict(exact)
        return _sealed(
            {
                "schema": "synthetic-terminal-interface-authentication-v1",
                "interface_authentication_sha256": exact["result_sha256"],
                "interface_report_sha256": exact["interface_report_sha256"],
            },
            "result_sha256",
        )

    monkeypatch.setattr(
        frontier,
        "_authenticate_fused_terminal_tied_interface",
        authenticate_interface,
    )

    def build_source_lineage(**kwargs):
        seen["source_lineage_witness"] = kwargs["anchor_transport_witness"]
        return SimpleNamespace(result_sha256="c" * 64)

    monkeypatch.setattr(
        frontier,
        "build_lfm2_long_horizon_terminal_source_lineage",
        build_source_lineage,
    )
    monkeypatch.setattr(
        frontier,
        "_authenticate_fused_terminal_cache_evidence",
        lambda _receipt: {
            "terminal_cache_compatibility_sha256": compatibility_sha256,
            "terminal_cache_payload_binding_sha256": payload_binding_sha256,
            "terminal_cache_session_sha256": (
                problem.bindings.cache_selective_session_sha256
            ),
        },
    )

    def publish_frontier(
        _result,
        _export_context,
        *,
        preterminal_capsule_reference,
        source_roots,
        output_directory,
    ):
        seen["published_capsule_reference"] = preterminal_capsule_reference
        seen["published_source_roots"] = source_roots
        return frontier.Lfm2SemanticTerminalFrontierBankReference(
            path=Path(output_directory),
            receipt_sha256="1" * 64,
            receipt_file_sha256="2" * 64,
            manifest_sha256="3" * 64,
            preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            terminal_result_sha256=terminal_result_sha256,
            source_roots_sha256="4" * 64,
            candidate_ids=("hybrid-frontier",),
        )

    monkeypatch.setattr(
        frontier,
        "publish_lfm2_semantic_terminal_frontier_bank",
        publish_frontier,
    )

    def fused_terminal(**kwargs):
        witness = kwargs["anchor_transport_witness"]
        seen["compiler_witness"] = witness
        interface_authentication = authenticate_interface(
            target_model=kwargs["target_model"],
            interface=kwargs["interface"],
            mapped_parent_tied_table=kwargs["source_selected_tied_table"],
            layer_receipt=kwargs["layer_stack"].receipt,
            anchor_transport_witness=witness,
        )
        context_body = {
            "terminal_result_sha256": terminal_result_sha256,
            "mapped_parent_tied_table_sha256": frontier._tensor_sha256(
                problem.source_table
            ),
            "tied_interface_authentication_sha256": (
                interface_authentication["result_sha256"]
            ),
            "terminal_cache_compatibility_sha256": compatibility_sha256,
            "terminal_cache_payload_binding_sha256": payload_binding_sha256,
            "terminal_cache_session_sha256": (
                problem.bindings.cache_selective_session_sha256
            ),
            "body_recipe_sha256": recipe.recipe_sha256,
            "preterminal_resume_mode": True,
            "same_live_cache_session_as_layer_stack": False,
            "cache_content_identity_matches_layer_stack": True,
            "frontier_export_occurs_before_final_w2_installation": True,
        }
        context = {
            **context_body,
            "context_sha256": canonical_sha256(context_body),
        }
        terminal_result = {
            "schema": "synthetic-terminal-result-v1",
            "result_sha256": terminal_result_sha256,
        }
        reference = kwargs["terminal_frontier_exporter"](
            SimpleNamespace(result_sha256=terminal_result_sha256),
            context,
        )
        body = {
            "terminal_result": terminal_result,
            "terminal_review_pending_schema": (
                postfold.LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
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
            "terminal_frontier_export_context": context,
            "terminal_frontier_export_context_sha256": context["context_sha256"],
            "terminal_frontier_export": reference.as_dict(),
            "terminal_frontier_export_sha256": canonical_sha256(
                reference.as_dict()
            ),
            "terminal_frontier_exported_before_w2_installation": True,
            "tied_interface_authentication": interface_authentication,
            "tied_interface_authentication_sha256": interface_authentication[
                "result_sha256"
            ],
        }
        return _sealed(body, "result_sha256")

    result = frontier.compile_lfm2_semantic_terminal_frontier_bank_from_capsule(
        preterminal_capsule_directory=problem.output,
        expected_preterminal_receipt_file_sha256=(
            problem.reference.receipt_file_sha256
        ),
        expected_preterminal_receipt_sha256=problem.reference.receipt_sha256,
        expected_preterminal_bindings_sha256=problem.reference.bindings_sha256,
        cache_directory=cache_directory,
        expected_cache_key=problem.bindings.semantic_cache_key,
        bridge_directory=bridge_directory,
        expected_bridge_artifact_sha256=(
            problem.bindings.bridge_artifact_sha256
        ),
        donor_tokenizer=object(),
        parent_model_directory=parent_directory,
        output_directory=tmp_path / "frontier",
        device="cpu",
        fused_terminal_compiler=fused_terminal,
    )

    loaded_witness = seen["compiler_witness"]
    assert isinstance(loaded_witness, TiedResidualAnchorTransportWitness)
    assert loaded_witness is not problem.anchor_transport_witness
    assert (
        loaded_witness.witness_sha256
        == problem.anchor_transport_witness.witness_sha256
    )
    assert seen["authentication_witness"] is loaded_witness
    assert seen["source_lineage_witness"] is loaded_witness
    assert seen["fresh_interface_authentication"] == capsule_manifest[
        "active_interface_authentication"
    ]
    assert (
        seen["published_capsule_reference"].receipt_sha256
        == problem.reference.receipt_sha256
    )
    assert result.bank_reference.candidate_ids == ("hybrid-frontier",)


@pytest.mark.parametrize("mutation", ("missing", "extra", "payload", "receipt"))
def test_preterminal_capsule_rejects_inventory_and_byte_tamper(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    if mutation == "missing":
        (problem.output / "capsule-manifest.json").unlink()
    elif mutation == "extra":
        (problem.output / "extra.txt").write_text("unexpected", encoding="ascii")
    elif mutation == "payload":
        payload = problem.output / "preterminal-seam.safetensors"
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)
    else:
        receipt = problem.output / "capsule-receipt.json"
        data = bytearray(receipt.read_bytes())
        data[-2] = ord(" ")
        receipt.write_bytes(data)

    with pytest.raises(Exception):
        _load(problem)


def test_preterminal_capsule_rejects_external_pin_and_stale_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        capsule.load_lfm2_semantic_preterminal_capsule(
            problem.output,
            expected_receipt_file_sha256="f" * 64,
            expected_receipt_sha256=problem.reference.receipt_sha256,
            expected_bindings_sha256=problem.reference.bindings_sha256,
            device="cpu",
        )
    with pytest.raises(Exception):
        capsule.load_lfm2_semantic_preterminal_capsule(
            problem.output,
            expected_receipt_file_sha256=problem.reference.receipt_file_sha256,
            expected_receipt_sha256=problem.reference.receipt_sha256,
            expected_bindings_sha256="e" * 64,
            device="cpu",
        )


def test_preterminal_capsule_outer_auth_rescans_checkpoint_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    checkpoint_tensor = problem.output / "checkpoint" / "model.safetensors"
    checkpoint_tensor.write_bytes(checkpoint_tensor.read_bytes() + b"tamper")

    with pytest.raises(Exception):
        _load(problem)


def test_preterminal_capsule_rechecks_checkpoint_after_atomic_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(Exception):
        _problem(tmp_path, monkeypatch, mutate_during_rename=True)


def test_preterminal_capsule_rejects_symlinked_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    payload = problem.output / "preterminal-seam.safetensors"
    outside = tmp_path / "outside.safetensors"
    shutil.move(payload, outside)
    try:
        os.symlink(outside, payload)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks.")

    with pytest.raises(Exception):
        _load(problem)


def test_preterminal_capsule_rejects_nonfinite_evidence_before_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    problem.evidence["fit"].mlp_input[0, 0] = float("nan")

    with pytest.raises(Exception):
        capsule.publish_lfm2_semantic_preterminal_capsule(
            target_model=problem.model,
            target_tokenizer=problem.tokenizer,
            layer_stack=problem.layer_stack,
            interface=problem.interface,
            attention_receipt_authentication=problem.attention,
            bindings=problem.bindings,
            output_directory=tmp_path / "nonfinite",
        )


def test_capsule_remains_loadable_when_subsequent_terminal_compiler_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    second_output = tmp_path / "preterminal-before-failure"
    captured = []

    def publish():
        reference = capsule.publish_lfm2_semantic_preterminal_capsule(
            target_model=problem.model,
            target_tokenizer=problem.tokenizer,
            layer_stack=problem.layer_stack,
            interface=problem.interface,
            attention_receipt_authentication=problem.attention,
            bindings=problem.bindings,
            output_directory=second_output,
        )
        captured.append(reference)
        return reference

    def fail_terminal(**_kwargs):
        raise RuntimeError("synthetic terminal failure")

    with pytest.raises(RuntimeError, match="synthetic terminal failure"):
        postfold.publish_preterminal_capsule_then_compile_fused_terminal(
            capsule_publisher=publish,
            fused_terminal_compiler=fail_terminal,
            terminal_arguments={},
        )

    assert len(captured) == 1
    reference = captured[0]
    problem.output = second_output
    problem.reference = reference
    loaded = _load(problem)
    assert loaded.reference.receipt_sha256 == reference.receipt_sha256


def test_capsule_aware_frontier_exporter_factory_receives_published_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    problem = _problem(tmp_path, monkeypatch)
    output = tmp_path / "preterminal-before-direct-frontier"
    events: list[str] = []
    exported: list[tuple[object, object]] = []

    def publish():
        reference = capsule.publish_lfm2_semantic_preterminal_capsule(
            target_model=problem.model,
            target_tokenizer=problem.tokenizer,
            layer_stack=problem.layer_stack,
            interface=problem.interface,
            attention_receipt_authentication=problem.attention,
            bindings=problem.bindings,
            output_directory=output,
        )
        events.append("capsule-published")
        return reference

    def exporter_factory(reference):
        events.append("exporter-factory")
        assert isinstance(reference, capsule.Lfm2SemanticPreterminalCapsuleReference)
        assert reference.path == output
        assert (output / "capsule-receipt.json").is_file()

        def exporter(result, context):
            events.append("frontier-exported")
            exported.append((result, context))
            return {"bank_receipt_sha256": "9" * 64}

        return exporter

    def compile_terminal(**kwargs):
        events.append("terminal-compiled")
        reference = kwargs["terminal_frontier_exporter"](
            "terminal-result",
            {"context_sha256": "8" * 64},
        )
        return {"terminal_frontier_export": reference}

    capsule_reference, terminal = (
        postfold.publish_preterminal_capsule_then_compile_fused_terminal(
            capsule_publisher=publish,
            fused_terminal_compiler=compile_terminal,
            terminal_arguments={"unrelated": True},
            terminal_frontier_exporter_factory=exporter_factory,
        )
    )

    assert isinstance(
        capsule_reference,
        capsule.Lfm2SemanticPreterminalCapsuleReference,
    )
    assert events == [
        "capsule-published",
        "exporter-factory",
        "terminal-compiled",
        "frontier-exported",
    ]
    assert exported == [
        ("terminal-result", {"context_sha256": "8" * 64})
    ]
    assert terminal == {
        "terminal_frontier_export": {"bank_receipt_sha256": "9" * 64}
    }


def test_capsule_aware_exporter_factory_rejects_direct_exporter_before_publish() -> (
    None
):
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="mutually exclusive",
    ):
        postfold.publish_preterminal_capsule_then_compile_fused_terminal(
            capsule_publisher=lambda: pytest.fail("conflict published a capsule"),
            fused_terminal_compiler=lambda **_kwargs: {},
            terminal_arguments={"terminal_frontier_exporter": lambda *_args: {}},
            terminal_frontier_exporter_factory=lambda _reference: lambda *_args: {},
        )
