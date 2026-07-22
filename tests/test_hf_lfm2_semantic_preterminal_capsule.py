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
from safetensors.torch import save_file  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
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


def _sealed(body: dict[str, object], seal: str) -> dict[str, object]:
    return {**body, seal: canonical_sha256(body)}


def _problem(tmp_path: Path, monkeypatch, *, mutate_during_rename: bool = False):
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
        "final_layer_mlp_evidence_sha256": {
            split: evidence[split].authentication_sha256 for split in sorted(evidence)
        },
    }
    layer_stack = Lfm2SemanticLayerStackInstallation(
        receipt=_sealed(layer_body, "result_sha256"),
        final_layer_mlp_evidence=evidence,
    )
    interface_body = {"schema": "synthetic-interface-v1"}
    interface = TiedCausalInterfaceFold(
        target_weight=torch.eye(2, dtype=torch.float64),
        producer_encoder=torch.eye(2, dtype=torch.float64),
        report=_sealed(interface_body, "report_sha256"),
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
        "interface_authentication": _sealed(
            {"schema": "synthetic-interface-authentication-v1"},
            "result_sha256",
        ),
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
