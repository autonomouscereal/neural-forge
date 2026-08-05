from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5_runner
import neural_forge.lfm2_exact4x_moving_gauge_v5_candidate_materializer as materializer
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    build_counted_child_state_plan,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), _file_sha256(path)))
    return canonical_sha256(rows)


def _candidate_pack(
    root: Path,
    *,
    plan_sha256: str,
    child_state_sha256: str,
    request_sha256: str | None = None,
    chart_mode: str = "E0R",
    target_arm: str = "joint",
    replacements: dict[str, torch.Tensor] | None = None,
    metadata_override: dict[str, str] | None = None,
    descriptor_override: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    request = request_sha256 or _digest("request")
    tensors = replacements or {
        materializer.REPLACEMENT_TENSOR_NAMES[0]: torch.tensor(
            [[11.0, 12.0], [13.0, 14.0]],
            dtype=torch.float32,
        ),
        materializer.REPLACEMENT_TENSOR_NAMES[1]: torch.tensor(
            [[21.0, 22.0], [23.0, 24.0]],
            dtype=torch.float32,
        ),
    }
    temporary = root / "candidate-building"
    temporary.mkdir()
    tensor_path = temporary / materializer.REPLACEMENT_TENSOR_FILE
    metadata = {
        "neural_forge": canonical_json_bytes(
            {
                "schema": materializer.CANDIDATE_PACK_SCHEMA,
                "request_sha256": request,
                "chart_mode": chart_mode,
                "target_arm": target_arm,
                "installed": False,
            }
        ).decode("ascii")
    }
    if metadata_override is not None:
        metadata = metadata_override
    save_file(tensors, str(tensor_path), metadata=metadata)
    descriptors = {
        name: {
            "shape": list(value.shape),
            "dtype": "float32",
            "numel": value.numel(),
            "tensor_sha256": fingerprint_lfm2_moving_gauge_tensor(value),
        }
        for name, value in tensors.items()
    }
    if descriptor_override is not None:
        descriptors = descriptor_override
    body = {
        "schema": materializer.CANDIDATE_PACK_SCHEMA,
        "request_sha256": request,
        "counted_child_state_plan_fingerprint": plan_sha256,
        "chart_mode": chart_mode,
        "target_arm": target_arm,
        "replacement_tensor_names": list(materializer.REPLACEMENT_TENSOR_NAMES),
        "replacement_tensor_count": 2,
        "tensor_descriptors": descriptors,
        "tensor_file": {
            "filename": materializer.REPLACEMENT_TENSOR_FILE,
            "sha256": _file_sha256(tensor_path),
            "size_bytes": tensor_path.stat().st_size,
        },
        "upstream_receipt_sha256s": {
            name: _digest(name)
            for name in sorted(materializer.UPSTREAM_RECEIPT_NAMES)
        },
        "child_state_before_sha256": child_state_sha256,
        "child_state_after_sha256": child_state_sha256,
        "source_child_state_preserved": True,
        "installed": False,
        "ready_for_atomic_apply": True,
        "contains_complete_child_checkpoint": False,
        "contains_only_two_replacement_tensors_and_receipt": True,
        "parent_runtime_sidecar": False,
        "quantization_credit": False,
        "pruning_credit": False,
        "parameter_sharing_credit": False,
        "training_steps": 0,
        "numeric_metrics_are_diagnostic_only": True,
        "manual_semantic_review_status": "pending",
        "manual_semantic_review_required_downstream": True,
    }
    receipt = {**body, "candidate_pack_sha256": canonical_sha256(body)}
    (temporary / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    destination = root / receipt["candidate_pack_sha256"]
    temporary.rename(destination)
    return destination, receipt


def _load_candidate(
    directory: Path,
    receipt: dict[str, Any],
) -> materializer.AuthenticatedLfm2Exact4xV5CandidatePack:
    return materializer.load_authenticated_lfm2_exact4x_v5_candidate_pack(
        directory,
        expected_candidate_pack_sha256=receipt["candidate_pack_sha256"],
        expected_request_sha256=receipt["request_sha256"],
        expected_chart_mode=receipt["chart_mode"],
        expected_target_arm=receipt["target_arm"],
        expected_counted_plan_fingerprint=receipt[
            "counted_child_state_plan_fingerprint"
        ],
    )


def test_authenticates_exact_canonical_two_tensor_pack(tmp_path: Path) -> None:
    directory, receipt = _candidate_pack(
        tmp_path,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )

    authenticated = _load_candidate(directory, receipt)

    assert authenticated.root == directory.resolve()
    assert authenticated.candidate_pack_sha256 == receipt["candidate_pack_sha256"]
    assert tuple(authenticated.tensors) == materializer.REPLACEMENT_TENSOR_NAMES
    assert all(
        value.dtype == torch.float32 and value.device.type == "cpu"
        for value in authenticated.tensors.values()
    )


def test_candidate_safetensor_single_metadata_key_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, first_receipt = _candidate_pack(
        first_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )
    second, second_receipt = _candidate_pack(
        second_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )

    assert first_receipt == second_receipt
    assert (first / materializer.REPLACEMENT_TENSOR_FILE).read_bytes() == (
        second / materializer.REPLACEMENT_TENSOR_FILE
    ).read_bytes()


def test_loader_accepts_pack_emitted_by_production_v5_stager(
    tmp_path: Path,
) -> None:
    allocations = tuple(
        ContinuousChildStateAllocation(
            state_id=name,
            parameter_elements=materializer.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
            tensor_sha256=_digest(f"{name}-tensor"),
            artifact_sha256=_digest(f"{name}-artifact"),
        )
        for name in ("semantic-carrier-route.c_in", "semantic-carrier-route.c_out")
    )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=materializer.CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=(
            materializer.CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        ),
        continuous_allocations=allocations,
    )
    paths = {}
    for name in sorted(v5_runner.REQUIRED_LOCAL_PATH_KEYS):
        path = tmp_path / name
        path.mkdir()
        paths[name] = str(path)
    lineage = {
        name: _digest(f"lineage-{name}")
        for name in v5_runner.REQUIRED_LINEAGE_KEYS
    }
    lineage["counted_child_state_plan"] = plan.fingerprint
    request = v5_runner.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=paths,
        lineage_sha256s=lineage,
        counted_child_state_plan=plan,
    )
    child_state = _digest("production-stager-child")
    candidate_path, receipt = v5_runner.stage_atomic_candidate_pack(
        request,
        chart_mode="E0A",
        target_arm="sequential_conditioned",
        replacements={
            materializer.REPLACEMENT_TENSOR_NAMES[0]: torch.eye(
                2, dtype=torch.float32
            ),
            materializer.REPLACEMENT_TENSOR_NAMES[1]: torch.ones(
                (2, 3), dtype=torch.float32
            ),
        },
        upstream_receipt_sha256s={
            name: _digest(f"upstream-{name}")
            for name in materializer.UPSTREAM_RECEIPT_NAMES
        },
        child_state_before_sha256=child_state,
        child_state_after_sha256=child_state,
    )

    authenticated = materializer.load_authenticated_lfm2_exact4x_v5_candidate_pack(
        candidate_path,
        expected_candidate_pack_sha256=receipt["candidate_pack_sha256"],
        expected_request_sha256=request.request_sha256,
        expected_chart_mode="E0A",
        expected_target_arm="sequential_conditioned",
        expected_counted_plan_fingerprint=plan.fingerprint,
    )

    assert authenticated.candidate_pack_sha256 == receipt["candidate_pack_sha256"]


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("expected_candidate_pack_sha256", _digest("wrong-pack")),
        ("expected_request_sha256", _digest("wrong-request")),
        ("expected_chart_mode", "E0A"),
        ("expected_target_arm", "sequential_conditioned"),
        ("expected_counted_plan_fingerprint", _digest("wrong-plan")),
    ],
)
def test_expected_cross_binding_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    wrong: str,
) -> None:
    directory, receipt = _candidate_pack(
        tmp_path,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )
    kwargs = {
        "expected_candidate_pack_sha256": receipt["candidate_pack_sha256"],
        "expected_request_sha256": receipt["request_sha256"],
        "expected_chart_mode": receipt["chart_mode"],
        "expected_target_arm": receipt["target_arm"],
        "expected_counted_plan_fingerprint": receipt[
            "counted_child_state_plan_fingerprint"
        ],
    }
    kwargs[field] = wrong

    with pytest.raises(materializer.Lfm2Exact4xV5CandidateMaterializationError):
        materializer.load_authenticated_lfm2_exact4x_v5_candidate_pack(
            directory,
            **kwargs,
        )


def test_candidate_addition_or_tensor_byte_change_fails_closed(
    tmp_path: Path,
) -> None:
    addition_root = tmp_path / "addition"
    addition_root.mkdir()
    added, added_receipt = _candidate_pack(
        addition_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )
    (added / "unsealed.bin").write_bytes(b"unsealed")
    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="exactly two ordinary files",
    ):
        _load_candidate(added, added_receipt)

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed, changed_receipt = _candidate_pack(
        changed_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
    )
    with (changed / materializer.REPLACEMENT_TENSOR_FILE).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="differs from its receipt",
    ):
        _load_candidate(changed, changed_receipt)


def test_nonfinite_tensor_and_wrong_metadata_fail_closed(tmp_path: Path) -> None:
    nonfinite_root = tmp_path / "nonfinite"
    nonfinite_root.mkdir()
    replacements = {
        materializer.REPLACEMENT_TENSOR_NAMES[0]: torch.tensor(
            [[float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32
        ),
        materializer.REPLACEMENT_TENSOR_NAMES[1]: torch.zeros(
            2, 2, dtype=torch.float32
        ),
    }
    nonfinite, receipt = _candidate_pack(
        nonfinite_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
        replacements=replacements,
    )
    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="finite CPU FP32 matrix",
    ):
        _load_candidate(nonfinite, receipt)

    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    wrong_metadata, wrong_receipt = _candidate_pack(
        metadata_root,
        plan_sha256=_digest("plan"),
        child_state_sha256=_digest("child"),
        metadata_override={"schema": "wrong"},
    )
    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="metadata or key inventory",
    ):
        _load_candidate(wrong_metadata, wrong_receipt)


@pytest.mark.parametrize(
    "location",
    (
        "source-child",
        "source-parent",
        "candidate-child",
        "candidate-parent",
        "candidate-equal",
    ),
)
def test_output_root_overlap_is_rejected_without_mutation(
    tmp_path: Path,
    location: str,
) -> None:
    source = tmp_path / "source-bank" / "source"
    candidate = tmp_path / "candidate-bank" / "candidate"
    source.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (source / "source.bin").write_bytes(b"source")
    (candidate / "candidate.bin").write_bytes(b"candidate")
    locations = {
        "source-child": source / "materialized",
        "source-parent": source.parent,
        "candidate-child": candidate / "materialized",
        "candidate-parent": candidate.parent,
        "candidate-equal": candidate,
    }
    output = locations[location]
    before = _tree_digest(tmp_path)

    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="overlaps an immutable input",
    ):
        materializer._validated_disjoint_output_root(
            output,
            immutable_roots=(source, candidate),
        )

    assert _tree_digest(tmp_path) == before
    assert not tuple(tmp_path.rglob("*.publish.lock"))
    assert not tuple(tmp_path.rglob(".v5-materialization-*.tmp"))
    if location in {"source-child", "candidate-child"}:
        assert not output.exists()


def test_phase8_route_state_uses_distinct_hash_domain() -> None:
    route_runtime_sha256 = _digest("route-runtime-publication")
    route_state_sha256 = _digest("route-owned-tensor-state")
    assert route_runtime_sha256 != route_state_sha256
    source = SimpleNamespace(
        identity_sha256=_digest("source-identity"),
        state_dict_sha256=_digest("source-state"),
    )
    evidence = {
        "schema": materializer.V5_LIVE_SCHEMA,
        "kind": "six_atomic_fp32_candidate_packs",
        "packs": {f"pack-{index}": {} for index in range(6)},
        "candidate_pack_count": 6,
        "replacement_tensor_names": list(materializer.REPLACEMENT_TENSOR_NAMES),
        "replacement_dtype": "float32",
        "installed": False,
        "ready_for_atomic_apply": True,
        "manual_semantic_review_status": "pending",
        "source_child_state_preserved": True,
        "source_child_identity_sha256": source.identity_sha256,
        "source_child_state_dict_sha256": source.state_dict_sha256,
        "source_route_state_sha256": route_state_sha256,
    }
    phase8 = {
        "evidence": evidence,
        "child_forward_count": 0,
        "parent_forward_count": 0,
        "heldout_rows_read": False,
        "temporary_child_mutation_performed": False,
        "child_state_restored": False,
        "child_state_before_sha256": None,
        "child_state_after_sha256": None,
        "installed": False,
        "ready_for_atomic_apply": True,
        "authenticated_checkpoint_reuse": False,
    }

    authenticated = materializer._authenticated_phase8_candidate_bank_evidence(
        phase8,
        source_fingerprint=source,
    )

    assert authenticated["source_route_state_sha256"] == route_state_sha256
    assert authenticated["source_route_state_sha256"] != route_runtime_sha256


def test_producer_authenticated_bridge_reference_is_retained_and_cross_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_artifact = _digest("bridge-artifact")
    bridge_profile = _digest("bridge-profile")
    tokenizer_semantics = _digest("target-tokenizer-semantics")
    producer_reference = MappingProxyType(
        {
            "bridge_artifact_sha256": bridge_artifact,
            "bridge_profile_sha256": bridge_profile,
            "target_tokenizer_semantics_sha256": tokenizer_semantics,
            "target_tokenizer_package_fingerprint": _digest(
                "target-tokenizer-package"
            ),
        }
    )
    bridge = object()
    monkeypatch.setattr(
        materializer,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_args, **_kwargs: bridge,
    )

    class Producer:
        @staticmethod
        def authenticate_packaged_tokenizer_bridge_reference(
            package: Any,
            observed_bridge: Any,
        ) -> Mapping[str, Any]:
            assert package == "sealed-package"
            assert observed_bridge is bridge
            return producer_reference

    manifest = {
        "tokenizer_bridge_root": str((tmp_path / "bridge").absolute()),
        "provenance_sha256s": {
            "bridge_artifact": bridge_artifact,
            "bridge_profile": bridge_profile,
        },
    }

    observed = (
        materializer._authenticate_strict_bridge_reference_for_materialization(
            launch=Producer,
            manifest=manifest,
            source_package="sealed-package",
            source_tokenizer_semantics_sha256=tokenizer_semantics,
        )
    )

    assert observed is producer_reference
    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="differs from the child",
    ):
        materializer._authenticate_strict_bridge_reference_for_materialization(
            launch=Producer,
            manifest=manifest,
            source_package="sealed-package",
            source_tokenizer_semantics_sha256=_digest("different-tokenizer"),
        )


def test_candidate_upstreams_derive_from_authenticated_phase_evidence() -> None:
    chart = _digest("chart")
    lineage = _digest("lineage")
    heldout = _digest("heldout")
    joint_operator = _digest("joint-operator")
    joint_mlp = _digest("joint-mlp")
    joint_target = _digest("joint-target")
    conditioned_operator = _digest("conditioned-operator")
    conditioned_mlp = _digest("conditioned-mlp")
    conditioned_target = _digest("conditioned-target")
    outcomes = {
        "authenticate_lineage": SimpleNamespace(artifact_sha256=lineage),
        "freeze_fit_charts": SimpleNamespace(
            evidence={"chart_bundle_sha256s": {"E0A": chart}}
        ),
        "build_baseline_target_arms": SimpleNamespace(
            evidence={
                "charts": {
                    "E0A": {
                        "joint_split_receipt_sha256s": {
                            "operator": joint_operator,
                            "mlp": joint_mlp,
                        },
                        "joint_target_sha256": joint_target,
                    }
                }
            }
        ),
        "freeze_conditioned_targets_and_solutions": SimpleNamespace(
            evidence={
                "charts": {
                    "E0A": {
                        "operator_solution_sha256": conditioned_operator,
                        "mlp_solution_sha256": conditioned_mlp,
                        "target_sha256": conditioned_target,
                    }
                }
            }
        ),
        "compare_frozen_candidates_on_heldout": SimpleNamespace(
            artifact_sha256=heldout
        ),
    }

    joint = materializer._expected_candidate_upstream_receipts(
        outcomes,
        chart_mode="E0A",
        target_arm="joint",
    )
    conditioned = materializer._expected_candidate_upstream_receipts(
        outcomes,
        chart_mode="E0A",
        target_arm="sequential_conditioned",
    )

    assert dict(joint) == {
        "chart": chart,
        "heldout_comparison": heldout,
        "lineage": lineage,
        "mlp_solution": joint_mlp,
        "operator_solution": joint_operator,
        "target_plan": joint_target,
    }
    assert dict(conditioned) == {
        "chart": chart,
        "heldout_comparison": heldout,
        "lineage": lineage,
        "mlp_solution": conditioned_mlp,
        "operator_solution": conditioned_operator,
        "target_plan": conditioned_target,
    }


def test_public_materializer_has_no_caller_path_or_hash_authority() -> None:
    parameters = inspect.signature(
        materializer.materialize_lfm2_exact4x_v5_candidate
    ).parameters
    assert tuple(parameters) == (
        "launch_manifest_json",
        "authority_key_path",
        "chart_mode",
        "target_arm",
    )


class _TinyConfig:
    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "lfm2", "hidden_size": 2, "test_fixture": True}


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Module()
        self.conv.out_proj = nn.Linear(2, 2, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.w2 = nn.Linear(2, 2, bias=False)


class _TinyChild(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.other = nn.Parameter(torch.tensor([7.0], dtype=torch.float32))
        self.config = _TinyConfig()


def test_fresh_state_check_rejects_noncandidate_signed_zero_change() -> None:
    source = _TinyChild()
    result = _TinyChild()
    result.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        source.other.fill_(0.0)
        result.other.copy_(torch.tensor([-0.0], dtype=torch.float32))
    candidate = materializer.AuthenticatedLfm2Exact4xV5CandidatePack(
        root=Path.cwd(),
        receipt=MappingProxyType(
            {"candidate_pack_sha256": _digest("signed-zero-candidate")}
        ),
        tensors=MappingProxyType(
            {
                name: result.state_dict()[name].detach().clone()
                for name in materializer.REPLACEMENT_TENSOR_NAMES
            }
        ),
    )

    with pytest.raises(
        materializer.Lfm2Exact4xV5CandidateMaterializationError,
        match="Non-candidate state tensor 'other' changed",
    ):
        materializer._compare_fresh_child_state(source, result, candidate)


def _load_tiny_child(source: Path) -> _TinyChild:
    state = load_file(str(Path(source) / "model.safetensors"), device="cpu")
    model = _TinyChild()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def _source_tree(root: Path) -> tuple[Path, _TinyChild, dict[str, Any]]:
    child = root / "source-child"
    child.mkdir()
    model = _TinyChild()
    with torch.no_grad():
        model.model.layers[0].conv.out_proj.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        model.model.layers[0].feed_forward.w2.weight.copy_(
            torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        )
    save_file(model.state_dict(), str(child / "model.safetensors"), metadata={"x": "y"})
    (child / "tokenizer.json").write_text("{}\n", encoding="ascii")
    (child / materializer.CARRIER_CHILD_PACKAGE_RECEIPT).write_text(
        "source outer receipt\n", encoding="ascii"
    )
    records = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in (child / "model.safetensors", child / "tokenizer.json")
    ]
    return child, model, {"records": records}


def _install_fake_carrier_environment(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    source_model: _TinyChild,
    source_records: dict[str, Any],
) -> tuple[Any, Any]:
    monkeypatch.setattr(materializer, "CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS", 30)
    monkeypatch.setattr(
        materializer, "CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS", 17
    )
    monkeypatch.setattr(
        materializer, "CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS", 9
    )
    monkeypatch.setattr(
        materializer, "CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS", 4
    )
    monkeypatch.setattr(materializer, "CURRENT_LFM2_CARRIER_TARGET_ROWS", 2)
    monkeypatch.setattr(materializer, "CURRENT_LFM2_CARRIER_HIDDEN_SIZE", 2)
    source_bridge = _digest("source-bridge")
    source_semantics = _digest("semantic-source")
    allocations = tuple(
        ContinuousChildStateAllocation(
            state_id=name,
            parameter_elements=4,
            tensor_sha256=_digest(f"{name}-tensor"),
            artifact_sha256=source_bridge,
        )
        for name in ("carrier-input", "carrier-output")
    )
    route_receipt = MappingProxyType(
        {
            "package_sha256": _digest("routes"),
            "input_table_binding": {"shape": [2, 2], "dtype": "float32"},
            "output_table_binding": {"shape": [2, 2], "dtype": "float32"},
        }
    )
    routes = SimpleNamespace(
        package_sha256=route_receipt["package_sha256"],
        allocations=allocations,
        receipt=route_receipt,
    )
    lexicon = SimpleNamespace(package_sha256=_digest("lexicon"))
    source_package = SimpleNamespace(
        root=source.resolve(),
        package_sha256=_digest("source-package"),
        receipt=MappingProxyType(
            {
                "child_file_records": source_records["records"],
                "child_directories": [],
                "target_rows": 2,
                "hidden_size": 2,
                "source_bridge_artifact_sha256": source_bridge,
                "semantic_source_tokenizer_sha256": source_semantics,
            }
        ),
        routes=routes,
        semantic_source_lexicon=lexicon,
    )
    published: dict[str, Any] = {}

    def load_package(path: Path, **_kwargs: Any) -> Any:
        if Path(path).resolve() == source.resolve():
            return source_package
        if "package" not in published:
            raise AssertionError("output package requested before fake seal")
        return published["package"]

    def seal(path: Path, **_kwargs: Any) -> Any:
        receipt_path = Path(path) / materializer.MATERIALIZATION_RECEIPT
        package_sha = canonical_sha256(
            {
                "model": _file_sha256(Path(path) / "model.safetensors"),
                "materialization": _file_sha256(receipt_path),
                "routes": routes.package_sha256,
            }
        )
        (Path(path) / materializer.CARRIER_CHILD_PACKAGE_RECEIPT).write_text(
            json.dumps({"fake_package_sha256": package_sha}, sort_keys=True) + "\n",
            encoding="ascii",
        )
        package = SimpleNamespace(
            root=Path(path),
            package_sha256=package_sha,
            receipt=source_package.receipt,
            routes=routes,
            semantic_source_lexicon=lexicon,
        )
        published["package"] = package
        return package

    class FakeCheckpointIO:
        @classmethod
        def from_child_directory(cls, **_kwargs: Any) -> "FakeCheckpointIO":
            return cls()

        def load(self, path: Path) -> _TinyChild:
            return _load_tiny_child(path)

    monkeypatch.setattr(
        materializer,
        "load_authenticated_lfm2_carrier_child_package",
        load_package,
    )
    monkeypatch.setattr(materializer, "seal_lfm2_carrier_child_package", seal)
    monkeypatch.setattr(materializer, "Lfm2CarrierChildCheckpointIO", FakeCheckpointIO)
    monkeypatch.setattr(
        materializer._DirectLocalReadOnlyIO,
        "load",
        lambda _self, path: _load_tiny_child(path),
    )
    monkeypatch.setattr(
        materializer,
        "count_lfm2_direct_local_unique_parameter_elements",
        lambda model: sum(value.numel() for value in model.parameters()),
    )
    monkeypatch.setattr(
        materializer,
        "load_lfm2_direct_local_tokenizer",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        materializer,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: _digest("target-tokenizer"),
    )
    return source_package, published


def _install_fake_materialization_provenance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_model: _TinyChild,
    source_package: Any,
    plan: Any,
    candidate_directory: Path,
    candidate_receipt: dict[str, Any],
    output_root: Path,
) -> materializer.AuthenticatedLfm2Exact4xV5MaterializationProvenance:
    candidate = _load_candidate(candidate_directory, candidate_receipt)
    source_fingerprint = fingerprint_lfm2_parent(source_model)
    launch_manifest_sha256 = _digest("launch-manifest")
    receipt_sha256 = _digest("launch-receipt")
    receipt_hmac_sha256 = _digest("launch-receipt-hmac")
    phase8_sha256 = _digest("phase8")
    provenance = materializer.AuthenticatedLfm2Exact4xV5MaterializationProvenance(
        launch_manifest=MappingProxyType(
            {"launch_manifest_sha256": launch_manifest_sha256}
        ),
        final_receipt=MappingProxyType(
            {
                "receipt_sha256": receipt_sha256,
                "receipt_hmac_sha256": receipt_hmac_sha256,
                "receipt_hmac_key_id": "launch-receipt-key-id",
            }
        ),
        phase8_outcome=MappingProxyType(
            {
                "artifact_sha256": phase8_sha256,
                "evidence": {
                    "source_route_state_sha256": _digest(
                        "producer-route-owned-state"
                    )
                },
            }
        ),
        request=SimpleNamespace(request_sha256=candidate_receipt["request_sha256"]),
        source_model=source_model,
        source_package=source_package,
        counted_plan=plan,
        source_fingerprint=source_fingerprint,
        source_tokenizer_semantics_sha256=_digest("target-tokenizer"),
        source_topology_sha256=materializer._state_topology_sha256(source_model),
        candidates=MappingProxyType(
            {
                f"{candidate_receipt['chart_mode']}:"
                f"{candidate_receipt['target_arm']}": candidate
            }
        ),
        selected_candidate=candidate,
        tokenizer_bridge_reference=MappingProxyType(
            {
                "bridge_artifact_sha256": _digest("strict-bridge"),
                "bridge_profile_sha256": _digest("strict-bridge-profile"),
                "target_tokenizer_package_fingerprint": _digest(
                    "strict-target-package"
                ),
            }
        ),
        output_root=output_root.resolve(strict=False),
    )

    def load_provenance(**kwargs: Any) -> Any:
        assert set(kwargs) == {
            "launch_manifest_json",
            "authority_key_path",
            "chart_mode",
            "target_arm",
        }
        assert kwargs["chart_mode"] == candidate_receipt["chart_mode"]
        assert kwargs["target_arm"] == candidate_receipt["target_arm"]
        return provenance

    monkeypatch.setattr(
        materializer,
        "load_authenticated_lfm2_exact4x_v5_materialization_provenance",
        load_provenance,
    )
    return provenance


def test_materializes_two_tensors_into_fresh_content_addressed_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_model, source_records = _source_tree(tmp_path)
    source_package, _published = _install_fake_carrier_environment(
        monkeypatch,
        source,
        source_model,
        source_records,
    )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=30,
        target_total_parameter_elements=17,
        continuous_allocations=source_package.routes.allocations,
    )
    source_state = fingerprint_lfm2_parent(source_model).state_dict_sha256
    candidate_root = tmp_path / "packs"
    candidate_root.mkdir()
    candidate, receipt = _candidate_pack(
        candidate_root,
        plan_sha256=plan.fingerprint,
        child_state_sha256=source_state,
    )
    source_before = _tree_digest(source)
    candidate_before = _tree_digest(candidate)
    output_root = tmp_path / "materialized"
    provenance = _install_fake_materialization_provenance(
        monkeypatch,
        source_model=source_model,
        source_package=source_package,
        plan=plan,
        candidate_directory=candidate,
        candidate_receipt=receipt,
        output_root=output_root,
    )

    result = materializer.materialize_lfm2_exact4x_v5_candidate(
        launch_manifest_json=tmp_path / "launch.json",
        authority_key_path=tmp_path / "authority.key",
        chart_mode=receipt["chart_mode"],
        target_arm=receipt["target_arm"],
    )

    child = Path(result["child_directory"])
    assert child == output_root / result["child_package_sha256"]
    assert result["published_child_artifact_sha256"] == result[
        "child_package_sha256"
    ]
    assert result["candidate_pack_sha256"] == receipt["candidate_pack_sha256"]
    assert source_before == _tree_digest(source)
    assert candidate_before == _tree_digest(candidate)
    assert (child / materializer.MATERIALIZATION_RECEIPT).is_file()
    assert (child / materializer.CARRIER_CHILD_PACKAGE_RECEIPT).is_file()
    installed = load_file(str(child / "model.safetensors"), device="cpu")
    candidate_tensors = load_file(
        str(candidate / materializer.REPLACEMENT_TENSOR_FILE),
        device="cpu",
    )
    for name in materializer.REPLACEMENT_TENSOR_NAMES:
        assert torch.equal(installed[name], candidate_tensors[name])
    assert torch.equal(installed["other"], torch.tensor([7.0]))
    materialization_receipt = json.loads(
        (child / materializer.MATERIALIZATION_RECEIPT).read_text(encoding="ascii")
    )
    assert materialization_receipt["installed"] is True
    assert materialization_receipt["training_steps"] == 0
    assert materialization_receipt["parent_forward_count"] == 0
    assert materialization_receipt["publication_authorized"] is False
    assert materialization_receipt["v5_launch_receipt_hmac_sha256"] == (
        provenance.final_receipt["receipt_hmac_sha256"]
    )
    assert materialization_receipt["v5_phase8_artifact_sha256"] == (
        provenance.phase8_outcome["artifact_sha256"]
    )
    assert materialization_receipt["v5_source_route_state_authentication"] == (
        "producer-hmac-attested"
    )
    assert materialization_receipt["v5_source_route_state_freshly_recomputed"] is (
        False
    )

    repeated = materializer.materialize_lfm2_exact4x_v5_candidate(
        launch_manifest_json=tmp_path / "launch.json",
        authority_key_path=tmp_path / "authority.key",
        chart_mode=receipt["chart_mode"],
        target_arm=receipt["target_arm"],
    )
    assert dict(repeated) == dict(result)
    assert not tuple(output_root.glob(".v5-materialization-*.tmp"))
    assert not tuple(output_root.glob(".*.publish.lock"))


def test_seal_failure_removes_owned_staging_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_model, source_records = _source_tree(tmp_path)
    source_package, _published = _install_fake_carrier_environment(
        monkeypatch,
        source,
        source_model,
        source_records,
    )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=30,
        target_total_parameter_elements=17,
        continuous_allocations=source_package.routes.allocations,
    )
    candidate_root = tmp_path / "packs"
    candidate_root.mkdir()
    candidate, receipt = _candidate_pack(
        candidate_root,
        plan_sha256=plan.fingerprint,
        child_state_sha256=fingerprint_lfm2_parent(source_model).state_dict_sha256,
    )
    output = tmp_path / "output"
    sentinel = output / "keep.txt"
    output.mkdir()
    sentinel.write_text("keep", encoding="ascii")
    _install_fake_materialization_provenance(
        monkeypatch,
        source_model=source_model,
        source_package=source_package,
        plan=plan,
        candidate_directory=candidate,
        candidate_receipt=receipt,
        output_root=output,
    )
    monkeypatch.setattr(
        materializer,
        "seal_lfm2_carrier_child_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("seal failed")),
    )

    with pytest.raises(RuntimeError, match="seal failed"):
        materializer.materialize_lfm2_exact4x_v5_candidate(
            launch_manifest_json=tmp_path / "launch.json",
            authority_key_path=tmp_path / "authority.key",
            chart_mode=receipt["chart_mode"],
            target_arm=receipt["target_arm"],
        )

    assert sentinel.read_text(encoding="ascii") == "keep"
    assert {path.name for path in output.iterdir()} == {"keep.txt"}


def test_foreign_publication_lock_is_never_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_model, source_records = _source_tree(tmp_path)
    source_package, _published = _install_fake_carrier_environment(
        monkeypatch,
        source,
        source_model,
        source_records,
    )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=30,
        target_total_parameter_elements=17,
        continuous_allocations=source_package.routes.allocations,
    )
    candidate_root = tmp_path / "packs"
    candidate_root.mkdir()
    candidate, receipt = _candidate_pack(
        candidate_root,
        plan_sha256=plan.fingerprint,
        child_state_sha256=fingerprint_lfm2_parent(source_model).state_dict_sha256,
    )
    output = tmp_path / "output"
    _install_fake_materialization_provenance(
        monkeypatch,
        source_model=source_model,
        source_package=source_package,
        plan=plan,
        candidate_directory=candidate,
        candidate_receipt=receipt,
        output_root=output,
    )
    first = materializer.materialize_lfm2_exact4x_v5_candidate(
        launch_manifest_json=tmp_path / "launch.json",
        authority_key_path=tmp_path / "authority.key",
        chart_mode=receipt["chart_mode"],
        target_arm=receipt["target_arm"],
    )
    shutil.rmtree(first["child_directory"])
    foreign_lock = output / f".{first['child_package_sha256']}.publish.lock"
    foreign_lock.write_text("owned by another publisher", encoding="ascii")

    with pytest.raises(FileExistsError):
        materializer.materialize_lfm2_exact4x_v5_candidate(
            launch_manifest_json=tmp_path / "launch.json",
            authority_key_path=tmp_path / "authority.key",
            chart_mode=receipt["chart_mode"],
            target_arm=receipt["target_arm"],
        )

    assert foreign_lock.read_text(encoding="ascii") == "owned by another publisher"
    assert not tuple(output.glob(".v5-materialization-*.tmp"))


def test_module_contains_no_parent_execution_or_optimization_entrypoints() -> None:
    source = Path(materializer.__file__).read_text(encoding="utf-8")
    forbidden = (
        ".backward(",
        "torch.optim",
        "AutoModel",
        "AutoTokenizer",
        "from_pretrained(",
        "model.generate(",
        "package_lfm2_carrier_routes_with_child",
        "load_lfm2_carrier_bridge_publication",
        "bind_lfm2_carrier_bridge_publication_to_child",
    )
    assert not [value for value in forbidden if value in source]
