from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments import audit_lfm2_quantizer_geometry as audit


def test_geometry_audit_is_local_diagnostic_only(tmp_path: Path) -> None:
    row = torch.tensor(
        [
            10.0,
            -0.12,
            -0.08,
            -0.04,
            0.01,
            0.04,
            0.08,
            0.12,
            -8.0,
            -0.11,
            -0.07,
            -0.03,
            0.02,
            0.05,
            0.09,
            0.13,
        ],
        dtype=torch.bfloat16,
    )
    model_file = tmp_path / "model.safetensors"
    save_file(
        {
            "aligned.weight": row.repeat(4, 1),
            "ragged.weight": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        },
        model_file,
    )

    report = audit.audit_quantizer_geometry(
        model_file=model_file,
        device="cpu",
        group_sizes=(8, 16),
        bit_widths=(4, 8),
        exact_outlier_counts=(0, 1),
        clip_ratios=(1.0, 0.8),
        sample_stride=1,
        max_groups_per_tensor=100,
    )

    assert report["schema"] == audit.REPORT_SCHEMA
    assert report["method"] == {
        "kind": "local_weight_geometry_diagnostic_only",
        "training": False,
        "generation": False,
        "parent_runtime_calls": 0,
        "child_checkpoint_written": False,
        "row_alignment_required": True,
        "scale_dtype": "torch.float16",
        "quantizer": "symmetric_scalar_with_exact_outlier_peeling",
        "clip_selection": "per_group_minimum_sampled_weight_mse",
    }
    tensors = {entry["name"]: entry for entry in report["tensors"]}
    assert tensors["aligned.weight"]["group_sizes"]["8"]["eligible"] is True
    assert tensors["aligned.weight"]["group_sizes"]["16"]["eligible"] is True
    assert tensors["ragged.weight"]["group_sizes"]["8"] == {
        "eligible": False,
        "reason": "last_dimension_not_divisible",
    }
    assert tensors["ragged.weight"]["group_sizes"]["16"] == {
        "eligible": False,
        "reason": "last_dimension_not_divisible",
    }
    for group_size in ("8", "16"):
        for bit_width in (4, 8):
            control = report["frontiers"][group_size][f"b{bit_width}-exact0"]
            peeled = report["frontiers"][group_size][f"b{bit_width}-exact1"]
            assert peeled["absmax_relative_rmse"] < control["absmax_relative_rmse"]
            assert peeled["best_clip_relative_rmse"] <= peeled["absmax_relative_rmse"]


def test_geometry_audit_sampling_is_deterministic_and_bounded() -> None:
    first = audit._sample_group_indices(
        1000,
        sample_stride=10,
        max_groups_per_tensor=20,
    )
    second = audit._sample_group_indices(
        1000,
        sample_stride=10,
        max_groups_per_tensor=20,
    )
    assert torch.equal(first, second)
    assert first.shape == (20,)
    assert int(first.min()) >= 0
    assert int(first.max()) < 1000
    assert bool(torch.all(first[1:] > first[:-1]))


def test_geometry_audit_rejects_unsafe_clip_grid() -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="include 1.0",
    ):
        audit._parse_float_list("0.9,0.8")


def test_geometry_audit_source_has_no_model_loader_or_training_calls() -> None:
    source = inspect.getsource(audit)
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"from_pretrained", "generate", "backward", "step"}.intersection(
        called_attributes
    )
