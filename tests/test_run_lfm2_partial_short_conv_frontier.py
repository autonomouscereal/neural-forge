from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

from experiments import run_lfm2_partial_short_conv_frontier as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _write_plan(path: Path, **extra: Any) -> dict[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-partial-short-conv-frontier-plan-v1",
        "status": "sealed-test-plan",
        **extra,
    }
    document = {**body, "plan_sha256": canonical_sha256(body)}
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return document


class _TinyShortConv(torch.nn.Module):
    def __init__(self, hidden: int, kernel: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            hidden,
            hidden,
            kernel,
            groups=hidden,
            bias=False,
        )
        self.in_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)


class _TinyLayer(torch.nn.Module):
    def __init__(self, hidden: int, kernel: int) -> None:
        super().__init__()
        self.conv = _TinyShortConv(hidden, kernel)


class _TinyParent(torch.nn.Module):
    def __init__(self, hidden: int = 4, kernel: int = 3) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyLayer(hidden, kernel)])


def _state_candidate(
    *,
    kind: str,
    state: dict[str, torch.Tensor],
    external: int,
    internal: int,
    kernel: int,
    expected_elements: int,
    rank: int | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    archive = {
        f"candidate/{kind}/{name}": value.detach().cpu().contiguous().clone()
        for name, value in state.items()
    }
    candidate: dict[str, Any] = {
        "candidate_id": kind,
        "runtime_kind": kind,
        "layer_index": 0,
        "module_path": "model.layers.0.conv",
        "external_hidden_size": external,
        "internal_channels": internal,
        "kernel_size": kernel,
        "candidate_parameter_elements": expected_elements,
        "state_tensor_keys": {
            name: f"candidate/{kind}/{name}" for name in state
        },
        "state_tensor_sha256": {
            name: lfm2_authenticated_tensor_sha256(value)
            for name, value in state.items()
        },
    }
    if rank is not None:
        candidate["output_action_rank"] = rank
    return candidate, archive


def _installed_state_names(model: _TinyParent) -> set[str]:
    return set(model.get_submodule("model.layers.0.conv").state_dict())


def test_evidence_writer_serializes_nested_sealed_mappings(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    sealed = MappingProxyType(
        {
            "candidate": MappingProxyType(
                {
                    "lineage": MappingProxyType({"rank": 97}),
                    "ids": ("one", "two"),
                }
            )
        }
    )

    runner._write_json(output, sealed)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "candidate": {"ids": ["one", "two"], "lineage": {"rank": 97}}
    }


def test_plan_reader_authenticates_file_and_semantic_seals(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    document = _write_plan(plan_path, candidates=[{"candidate_id": "one"}])
    file_sha256 = _file_sha256(plan_path)

    loaded = runner._read_plan(
        plan_path,
        expected_plan_sha256=document["plan_sha256"],
        expected_file_sha256=file_sha256,
    )
    assert loaded == document

    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Plan file hash changed",
    ):
        runner._read_plan(
            plan_path,
            expected_plan_sha256=document["plan_sha256"],
            expected_file_sha256="0" * 64,
        )

    resealed_file_only = copy.deepcopy(document)
    resealed_file_only["status"] = "tampered"
    plan_path.write_text(
        json.dumps(resealed_file_only, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Plan seal is invalid",
    ):
        runner._read_plan(
            plan_path,
            expected_plan_sha256=document["plan_sha256"],
            expected_file_sha256=_file_sha256(plan_path),
        )


def test_candidate_frontier_rejects_missing_reordered_and_duplicate_rows() -> None:
    candidates = [
        {"candidate_id": f"candidate-{index}", "runtime_kind": kind}
        for index, kind in enumerate(runner.EXPECTED_RUNTIME_KINDS)
    ]
    assert runner._validated_candidates({"candidates": candidates}) == tuple(candidates)

    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="complete five-candidate frontier",
    ):
        runner._validated_candidates({"candidates": candidates[:-1]})

    reordered = copy.deepcopy(candidates)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="order or mechanism set",
    ):
        runner._validated_candidates({"candidates": reordered})

    duplicate = copy.deepcopy(candidates)
    duplicate[-1]["candidate_id"] = duplicate[0]["candidate_id"]
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="nonempty and unique",
    ):
        runner._validated_candidates({"candidates": duplicate})


def test_candidate_selection_runs_only_requested_rows_in_canonical_plan_order() -> None:
    candidates = tuple(
        {"candidate_id": f"candidate-{index}", "runtime_kind": kind}
        for index, kind in enumerate(runner.EXPECTED_RUNTIME_KINDS)
    )
    assert runner._selected_candidates(candidates, None) == candidates
    selected = runner._selected_candidates(
        candidates,
        ("candidate-4", "candidate-1"),
    )
    assert tuple(row["candidate_id"] for row in selected) == (
        "candidate-1",
        "candidate-4",
    )

    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="nonempty and unique",
    ):
        runner._selected_candidates(candidates, ())
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="nonempty and unique",
    ):
        runner._selected_candidates(candidates, ("candidate-1", "candidate-1"))
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="absent from the sealed plan",
    ):
        runner._selected_candidates(candidates, ("not-present",))


def test_archive_reader_authenticates_namespace_shape_dtype_and_content(
    tmp_path: Path,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    archive_path = tmp_path / "candidate.safetensors"
    tensors = {
        "candidate/conv.weight": torch.arange(6, dtype=torch.float32).reshape(2, 1, 3),
        "candidate/in_proj.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
    }
    safetensors.save_file(tensors, str(archive_path))
    records = {name: _tensor_record(value) for name, value in tensors.items()}

    loaded = runner._load_archive(
        archive_path,
        expected_file_sha256=_file_sha256(archive_path),
        records=records,
    )
    assert set(loaded) == set(tensors)
    for name, expected in tensors.items():
        torch.testing.assert_close(loaded[name], expected)

    tampered = {name: value.clone() for name, value in tensors.items()}
    tampered["candidate/conv.weight"][0, 0, 0] += 1.0
    safetensors.save_file(tampered, str(archive_path))
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Archived tensor .* differs from its seal",
    ):
        runner._load_archive(
            archive_path,
            expected_file_sha256=_file_sha256(archive_path),
            records=records,
        )

    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Tensor archive hash changed",
    ):
        runner._load_archive(
            archive_path,
            expected_file_sha256="f" * 64,
            records=records,
        )


def test_installs_projected_parent_output_without_changing_component_geometry() -> None:
    hidden = 4
    kernel = 3
    model = _TinyParent(hidden, kernel)
    source = model.get_submodule("model.layers.0.conv")
    source_elements = sum(parameter.numel() for parameter in source.parameters())
    projected = torch.arange(hidden * hidden, dtype=torch.float32).reshape(
        hidden,
        hidden,
    )
    candidate, archive = _state_candidate(
        kind="projected-parent-output",
        state={"out_proj.weight": projected},
        external=hidden,
        internal=hidden,
        kernel=kernel,
        expected_elements=source_elements,
    )

    installation = runner._install_candidate(
        model,
        candidate=candidate,
        tensors=archive,
        device=torch.device("cpu"),
    )
    installed = model.get_submodule("model.layers.0.conv")

    assert installed is source
    torch.testing.assert_close(installed.out_proj.weight, projected)
    assert installation["installed_parameter_elements"] == source_elements == 76
    assert set(installation["state_tensor_sha256"]) == {
        "conv.weight",
        "in_proj.weight",
        "out_proj.weight",
    }
    assert installation["frozen"] is True
    assert installation["eval_mode"] is True
    assert all(not parameter.requires_grad for parameter in installed.parameters())
    assert installed.training is False


def test_installs_factored_projected_parent_output_with_real_parameter_reduction() -> (
    None
):
    hidden = 4
    kernel = 3
    rank = 1
    model = _TinyParent(hidden, kernel)
    source = model.get_submodule("model.layers.0.conv")
    source_elements = sum(parameter.numel() for parameter in source.parameters())
    state = {
        "out_encoder.weight": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "out_decoder.weight": torch.arange(4, dtype=torch.float32).reshape(
            hidden,
            rank,
        ),
    }
    expected_elements = source_elements - hidden * hidden + 2 * hidden * rank
    candidate, archive = _state_candidate(
        kind="projected-parent-output-factor",
        state=state,
        external=hidden,
        internal=hidden,
        kernel=kernel,
        rank=rank,
        expected_elements=expected_elements,
    )

    installation = runner._install_candidate(
        model,
        candidate=candidate,
        tensors=archive,
        device=torch.device("cpu"),
    )
    installed = model.get_submodule("model.layers.0.conv")

    assert installation["installed_parameter_elements"] == expected_elements == 68
    assert source_elements == 76
    assert _installed_state_names(model) == {
        "conv.weight",
        "in_proj.weight",
        "out_proj.0.weight",
        "out_proj.1.weight",
    }
    torch.testing.assert_close(installed.out_proj[0].weight, state["out_encoder.weight"])
    torch.testing.assert_close(installed.out_proj[1].weight, state["out_decoder.weight"])
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, hidden)
    dense_projected = state["out_decoder.weight"].matmul(
        state["out_encoder.weight"]
    )
    torch.testing.assert_close(
        installed.out_proj(inputs),
        inputs.matmul(dense_projected.T),
    )
    assert all(not parameter.requires_grad for parameter in installed.parameters())
    assert installed.training is False


@pytest.mark.parametrize("kind", ["qdeim-structural", "qdeim-local-writer"])
def test_installs_each_qdeim_runtime_with_exact_physical_count(kind: str) -> None:
    external = 4
    internal = 2
    kernel = 3
    state = {
        "conv.weight": torch.arange(6, dtype=torch.float32).reshape(internal, 1, kernel),
        "in_proj.weight": torch.arange(24, dtype=torch.float32).reshape(
            3 * internal,
            external,
        ),
        "out_proj.weight": torch.arange(8, dtype=torch.float32).reshape(
            external,
            internal,
        ),
    }
    expected_elements = internal * kernel + 3 * internal * external + external * internal
    candidate, archive = _state_candidate(
        kind=kind,
        state=state,
        external=external,
        internal=internal,
        kernel=kernel,
        expected_elements=expected_elements,
    )
    model = _TinyParent(external, kernel)

    installation = runner._install_candidate(
        model,
        candidate=candidate,
        tensors=archive,
        device=torch.device("cpu"),
    )
    installed = model.get_submodule("model.layers.0.conv")

    assert installation["installed_parameter_elements"] == expected_elements == 38
    assert sum(parameter.numel() for parameter in installed.parameters()) == 38
    assert _installed_state_names(model) == set(state)
    assert set(installation["state_tensor_sha256"]) == set(state)
    assert installation["frozen"] is True
    assert installation["eval_mode"] is True
    assert all(not parameter.requires_grad for parameter in installed.parameters())
    assert installed.training is False


def test_installs_factored_output_writer_with_runtime_state_names() -> None:
    external = 4
    internal = 2
    kernel = 3
    rank = 1
    state = {
        "conv.weight": torch.arange(6, dtype=torch.float32).reshape(internal, 1, kernel),
        "in_proj.weight": torch.arange(24, dtype=torch.float32).reshape(
            3 * internal,
            external,
        ),
        "out_encoder.weight": torch.tensor([[1.0, 2.0]]),
        "out_decoder.weight": torch.arange(4, dtype=torch.float32).reshape(
            external,
            rank,
        ),
    }
    expected_elements = (
        internal * kernel
        + 3 * internal * external
        + rank * internal
        + external * rank
    )
    candidate, archive = _state_candidate(
        kind="qdeim-local-writer-output-factor",
        state=state,
        external=external,
        internal=internal,
        kernel=kernel,
        rank=rank,
        expected_elements=expected_elements,
    )
    model = _TinyParent(external, kernel)

    installation = runner._install_candidate(
        model,
        candidate=candidate,
        tensors=archive,
        device=torch.device("cpu"),
    )
    installed = model.get_submodule("model.layers.0.conv")

    assert installation["installed_parameter_elements"] == expected_elements == 36
    assert sum(parameter.numel() for parameter in installed.parameters()) == 36
    assert _installed_state_names(model) == {
        "conv.weight",
        "in_proj.weight",
        "out_proj.0.weight",
        "out_proj.1.weight",
    }
    assert set(installation["state_tensor_sha256"]) == _installed_state_names(model)
    torch.testing.assert_close(
        installed.out_proj[0].weight,
        state["out_encoder.weight"],
    )
    torch.testing.assert_close(
        installed.out_proj[1].weight,
        state["out_decoder.weight"],
    )
    assert installation["frozen"] is True
    assert installation["eval_mode"] is True
    assert all(not parameter.requires_grad for parameter in installed.parameters())
    assert installed.training is False


def test_candidate_installation_rejects_tensor_and_parameter_count_tampering() -> None:
    external = 4
    internal = 2
    kernel = 3
    state = {
        "conv.weight": torch.ones(internal, 1, kernel),
        "in_proj.weight": torch.ones(3 * internal, external),
        "out_proj.weight": torch.ones(external, internal),
    }
    candidate, archive = _state_candidate(
        kind="qdeim-local-writer",
        state=state,
        external=external,
        internal=internal,
        kernel=kernel,
        expected_elements=38,
    )

    tampered_archive = {name: value.clone() for name, value in archive.items()}
    tampered_archive[candidate["state_tensor_keys"]["out_proj.weight"]][0, 0] += 1.0
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Candidate state tensor 'out_proj.weight' changed",
    ):
        runner._install_candidate(
            _TinyParent(external, kernel),
            candidate=candidate,
            tensors=tampered_archive,
            device=torch.device("cpu"),
        )

    wrong_count = copy.deepcopy(candidate)
    wrong_count["candidate_parameter_elements"] = 39
    with pytest.raises(
        runner.Lfm2PartialShortConvFrontierRunError,
        match="Installed component parameter count differs from the plan",
    ):
        runner._install_candidate(
            _TinyParent(external, kernel),
            candidate=wrong_count,
            tensors=archive,
            device=torch.device("cpu"),
        )


def test_manual_review_is_human_decided_and_cached_parent_is_reference_only() -> None:
    candidate = {
        "candidate_id": "qdeim-local-writer",
        "runtime_kind": "qdeim-local-writer",
    }
    responses = [{"id": "prompt-0", "text": "A coherent answer."}]
    executed = runner._review_document(
        candidate=candidate,
        installation={"installation_sha256": "a" * 64},
        responses=responses,
    )

    assert executed["status"] == "awaiting-manual-intelligence-review"
    assert executed["numeric_metrics_have_decision_authority"] is False
    assert executed["automatic_acceptance_or_rejection_performed"] is False
    assert executed["exact_token_match_required"] is False
    assert executed["manual_review"] == {
        "coherent_response_count": None,
        "useful_or_correct_response_count": None,
        "intelligence_retention_relative_to_parent": None,
        "notes": None,
    }
    assert "accepted" not in executed
    assert "rejected" not in executed
    assert "threshold" not in executed

    reference_candidate = {
        "candidate_id": "cached-parent-reference",
        "runtime_kind": "cached-parent-reference",
    }
    parent_rows = [{"id": "prompt-0", "text": "Cached parent answer."}]
    reference = runner._review_document(
        candidate=reference_candidate,
        installation=None,
        responses=parent_rows,
    )

    assert reference["schema"] == runner.REVIEW_SCHEMA
    assert reference["status"] == "reference"
    assert reference["candidate"] == reference_candidate
    assert reference["installation"] is None
    assert reference["responses"] == parent_rows
    assert reference["response_count"] == 1
    assert reference["manual_review"] == {"reference_integrity_notes": None}
    assert reference["fresh_fp32_candidate_decode"] is False
    assert reference["cached_parent_reference_only"] is True
    assert reference["numeric_metrics_have_decision_authority"] is False
    assert reference["automatic_acceptance_or_rejection_performed"] is False
    assert reference["training_steps"] == 0
    assert reference["optimizer_steps"] == 0
    assert reference["backward_calls"] == 0
