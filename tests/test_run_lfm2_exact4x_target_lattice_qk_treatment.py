from __future__ import annotations

import ast
from pathlib import Path

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


def test_treatment_runner_requires_lineage_body_and_checkpoint_pins() -> None:
    actions = {value.dest: value for value in treatment._parser()._actions}
    for name in (
        "expected_attention_lineage_file_sha256",
        "expected_source_structural_body_receipt_sha256",
        "expected_source_checkpoint_inventory_sha256",
        "expected_semantic_cache_key",
        "expected_bridge_artifact_sha256",
    ):
        assert actions[name].required is True
