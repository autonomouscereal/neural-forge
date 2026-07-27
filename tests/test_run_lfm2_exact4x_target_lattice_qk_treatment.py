from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments import run_lfm2_exact4x_target_lattice_qk_treatment as treatment


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_lfm2_exact4x_target_lattice_qk_treatment.py"


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    ]


def test_treatment_runner_has_one_closure_save_reload_and_six_head_handoff() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert len(
        _calls(
            tree,
            "compile_install_lfm2_saved_body_target_lattice_qk_closure",
        )
    ) == 1
    assert len(_calls(tree, "authenticate_lfm2_saved_attention_lineage")) == 1
    assert len(_calls(tree, "extract_lfm2_qdeim_head_selections")) == 1
    assert len(_calls(tree, "save")) == 1
    assert len(_calls(tree, "frontier.run")) == 0
    assert "downstream = frontier.run(" in source
    assert "treated_body_directory=treated_root" in source
    assert "expected_treated_body_provenance_sha256=" in source
    assert '"all_raw_outputs_retained": True' in source
    assert '"numeric_metrics_have_final_intelligence_authority": False' in source
    assert '"operation_audit_scope"' in source
    assert '"downstream_evaluation_operation_audit"' in source
    assert '"source_body_authentication_basis"' in source
    assert '"source_body_authentication_basis_sha256"' in source
    assert "_build_source_body_authentication_basis(" in source

    forbidden = (
        "load_lfm2_direct_local_model",
        "parent_model_directory",
        "torch.optim",
        "autograd.grad",
        ".backward(",
        "parameters_pruned\": 1",
        "parameters_quantized\": 1",
    )
    assert all(value not in source for value in forbidden)


def test_treatment_runner_preserves_pinned_source_dtype_through_fresh_reload() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    checkpoint_io = _calls(tree, "Lfm2DirectLocalCheckpointIO")

    assert len(_calls(tree, "_resolve_source_checkpoint_dtype_name")) == 1
    assert len(checkpoint_io) == 1
    dtype_keywords = [
        value.value
        for value in checkpoint_io[0].keywords
        if value.arg == "dtype"
    ]
    assert len(dtype_keywords) == 1
    assert isinstance(dtype_keywords[0], ast.Name)
    assert dtype_keywords[0].id == "source_checkpoint_dtype"
    assert "dtype=torch.bfloat16" not in source
    assert '"source_checkpoint_dtype": source_checkpoint_dtype_name' in source
    assert '"treated_checkpoint_dtype": reloaded_parameter_dtype_name' in source
    assert '"checkpoint_dtype_preserved": True' in source


def test_uniform_parameter_dtype_helper_accepts_fp32_and_rejects_mixed() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, dtype=torch.float32),
        torch.nn.Linear(2, 2, dtype=torch.float32),
    )
    assert treatment._uniform_parameter_dtype_name(model) == "float32"
    model[1].to(dtype=torch.bfloat16)
    with pytest.raises(
        treatment.Lfm2Exact4xTargetLatticeQkTreatmentError,
        match="uniform BF16 or FP32",
    ):
        treatment._uniform_parameter_dtype_name(model)


def test_treatment_runner_requires_lineage_and_checkpoint_pins_but_allows_legacy_body() -> None:
    actions = {value.dest: value for value in treatment._parser()._actions}
    for name in (
        "expected_attention_lineage_file_sha256",
        "expected_source_checkpoint_inventory_sha256",
        "expected_semantic_cache_key",
        "expected_bridge_artifact_sha256",
    ):
        assert actions[name].required is True
    assert actions["expected_source_structural_body_receipt_sha256"].required is False


def test_treatment_baseline_uses_tensor_domain_deferred_w2_authentication() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "frontier._authenticate_loaded_deferred_final_w2(" in source
    assert 'deferred_receipt.get("model_state_after_sha256") != baseline_state' not in source


def test_attention_lineage_pin_authenticates_the_exact_parsed_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "attention-lineage.json"
    payload = json.dumps({"raw": {}, "typed": {}}, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(candidate: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    value = treatment._load_pinned_json(
        path,
        expected_file_sha256=hashlib.sha256(payload).hexdigest(),
        label="source attention lineage",
    )

    assert value == {"raw": {}, "typed": {}}
    assert reads == 1


def test_attention_lineage_pin_rejects_unpinned_bytes(tmp_path: Path) -> None:
    path = tmp_path / "attention-lineage.json"
    path.write_text('{"raw": {}, "typed": {}}\n', encoding="utf-8")

    with pytest.raises(
        treatment.Lfm2Exact4xTargetLatticeQkTreatmentError,
        match="external pin",
    ):
        treatment._load_pinned_json(
            path,
            expected_file_sha256="0" * 64,
            label="source attention lineage",
        )
