from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from experiments import replay_lfm2_normalized_input_decoder as replay_experiment  # noqa: E402
from experiments import run_lfm2_exact4x_composite_fold as exact4x_runner  # noqa: E402
import neural_forge.hf_lfm2_normalized_input_replay_capsule as capsule_module  # noqa: E402
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_normalized_input_decoder import (  # noqa: E402
    compile_lfm2_normalized_input_decoder,
)
from neural_forge.hf_lfm2_normalized_input_replay_capsule import (  # noqa: E402
    Lfm2NormalizedInputReplayCapsuleError,
    load_lfm2_normalized_input_replay_capsule,
    prepare_lfm2_normalized_input_replay_capture_root,
    write_lfm2_normalized_input_replay_capsule,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    _install_mlp,
    compile_install_lfm2_semantic_layer_stack_postfold,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (  # noqa: E402
    _full_candidate_state,
    prepare_lfm2_semantic_mlp_postfold,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    compile_install_lfm2_semantic_layer_stack_stage,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedNormalizedInputMlpFoldCase,
    lfm2_authenticated_tensor_sha256,
)


def _completion_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _request() -> dict[str, object]:
    residual_chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_gamma = torch.ones(3, dtype=torch.float64)
    target_gamma = torch.ones(2, dtype=torch.float64)
    raw_decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.25]],
        dtype=torch.float64,
    )
    target_input = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    parent_input = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    parent_gate = torch.tensor(
        [[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]],
        dtype=torch.float64,
    )
    parent_up = parent_gate + 0.125
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    parent_endpoint = torch.tensor(
        [[0.4, -0.2], [0.3, 0.9]],
        dtype=torch.float64,
    )
    child_pre = torch.tensor(
        [[0.1, -0.1], [0.2, 0.4]],
        dtype=torch.float64,
    )
    first = Lfm2CachedNormalizedInputMlpFoldCase(
        target_input=target_input,
        parent_normalized_input=parent_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=parent_endpoint,
        target_pre_mlp_residual=child_pre,
        parent_contribution=parent_post.abs() + 0.25,
    )
    second = Lfm2CachedNormalizedInputMlpFoldCase(
        target_input=target_input * 0.75,
        parent_normalized_input=parent_input * 0.75,
        parent_gate=parent_gate * 0.75,
        parent_up=parent_up * 0.75,
        parent_post_swiglu=parent_post * 0.75,
        parent_post_mlp_residual_target_space=parent_endpoint * 0.75,
        target_pre_mlp_residual=child_pre * 0.75,
    )
    completion_body = {
        "schema": "neural-forge-analytic-coordinate-completion-candidate-v1",
        "recipe": "test-request-capsule",
        "evidence_sha256": "1" * 64,
        "coordinate_map_shape": [2, 3],
        "coordinate_map_sha256": _completion_tensor_sha256(raw_decoder),
        "materialized_rank": 2,
        "full_rank_materialized": True,
    }
    completion_sha256 = canonical_sha256(completion_body)
    return {
        "layer_index": 3,
        "residual_chart": residual_chart,
        "residual_chart_sha256": lfm2_authenticated_tensor_sha256(residual_chart),
        "parent_ffn_norm_weight": parent_gamma,
        "parent_ffn_norm_weight_sha256": lfm2_authenticated_tensor_sha256(parent_gamma),
        "target_ffn_norm_weight": target_gamma,
        "target_ffn_norm_weight_sha256": lfm2_authenticated_tensor_sha256(target_gamma),
        "cases": (first, second),
        "case_sample_weights": (2.5, 1.25),
        "normalized_decoder_ridge": 1e-8,
        "normalized_input_correction_rho": 0.25,
        "raw_structural_decoder": raw_decoder,
        "raw_structural_decoder_sha256": lfm2_authenticated_tensor_sha256(raw_decoder),
        "coordinate_completion_receipt": {
            **completion_body,
            "result_sha256": completion_sha256,
        },
        "coordinate_completion_receipt_sha256": completion_sha256,
    }


def _compiler_request(request: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in request.items() if key != "layer_index"}


def test_capsule_round_trip_replays_exact_production_request(tmp_path: Path) -> None:
    request = _request()
    capture_root = tmp_path / "captures"
    capsule = write_lfm2_normalized_input_replay_capsule(
        capture_root,
        **request,
    )
    loaded = load_lfm2_normalized_input_replay_capsule(capsule.root)
    direct = compile_lfm2_normalized_input_decoder(**_compiler_request(request))
    replayed = compile_lfm2_normalized_input_decoder(**loaded.compiler_kwargs())

    assert capsule.root == capture_root / "layer-0003"
    assert capsule.capsule_sha256 == loaded.capsule_sha256
    assert loaded.layer_index == 3
    assert loaded.case_sample_weights == (2.5, 1.25)
    assert (
        loaded.manifest["fit_evidence_sha256"]
        == (replayed.as_dict()["fit_evidence_sha256"])
    )
    assert replayed.receipt_sha256 == direct.receipt_sha256
    assert torch.equal(
        replayed.normalized_input_decoder,
        direct.normalized_input_decoder,
    )
    assert set(path.name for path in capsule.root.iterdir()) == {
        "request.json",
        "request.safetensors",
    }


def test_capsule_refuses_existing_layer_destination(tmp_path: Path) -> None:
    request = _request()
    capture_root = tmp_path / "captures"
    write_lfm2_normalized_input_replay_capsule(capture_root, **request)

    with pytest.raises(
        Lfm2NormalizedInputReplayCapsuleError,
        match="already exists",
    ):
        write_lfm2_normalized_input_replay_capsule(capture_root, **request)


def test_capsule_publishes_atomically_and_cleans_owned_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = prepare_lfm2_normalized_input_replay_capture_root(
        tmp_path / "captures"
    )

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(capsule_module, "_atomic_rename_noreplace", fail_rename)
    with pytest.raises(
        Lfm2NormalizedInputReplayCapsuleError,
        match="atomically published",
    ):
        write_lfm2_normalized_input_replay_capsule(capture_root, **_request())

    assert list(capture_root.iterdir()) == []


def test_capture_root_preflight_rejects_stale_content(tmp_path: Path) -> None:
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    (capture_root / "stale").write_text("stale", encoding="utf-8")

    with pytest.raises(
        Lfm2NormalizedInputReplayCapsuleError,
        match="must be empty",
    ):
        prepare_lfm2_normalized_input_replay_capture_root(capture_root)


def test_capsule_rejects_changed_safetensors_bytes(tmp_path: Path) -> None:
    capsule = write_lfm2_normalized_input_replay_capsule(
        tmp_path / "captures",
        **_request(),
    )
    tensor_path = capsule.root / "request.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(
        Lfm2NormalizedInputReplayCapsuleError,
        match="bytes changed",
    ):
        load_lfm2_normalized_input_replay_capsule(capsule.root)


def test_capsule_rejects_resealed_case_reordering(tmp_path: Path) -> None:
    capsule = write_lfm2_normalized_input_replay_capsule(
        tmp_path / "captures",
        **_request(),
    )
    manifest_path = capsule.root / "request.json"
    manifest = json.loads(manifest_path.read_bytes())
    body = dict(manifest)
    body.pop("capsule_sha256")
    body["case_order"] = list(reversed(body["case_order"]))
    resealed = {**body, "capsule_sha256": canonical_sha256(body)}
    manifest_path.write_bytes(canonical_json_bytes(resealed) + b"\n")

    with pytest.raises(
        Lfm2NormalizedInputReplayCapsuleError,
        match="case indices",
    ):
        load_lfm2_normalized_input_replay_capsule(capsule.root)


def test_standalone_replay_writes_sealed_no_overwrite_result(tmp_path: Path) -> None:
    capsule = write_lfm2_normalized_input_replay_capsule(
        tmp_path / "captures",
        **_request(),
    )
    result_path = tmp_path / "replay-result.json"
    result = replay_experiment.replay(
        capsule_directory=capsule.root,
        result_json=result_path,
    )

    body = dict(result)
    assert body.pop("result_sha256") == canonical_sha256(body)
    assert result["status"] == "production-compiler-succeeded"
    assert result["capsule_authenticated_before_compiler_invocation"] is True
    assert result["parent_checkpoint_loads"] == 0
    assert result["trace_cache_loads"] == 0
    assert result["target_model_loads"] == 0
    assert result["gpu_operations"] == 0
    assert result["training_steps"] == 0
    assert result_path.read_bytes() == canonical_json_bytes(result) + b"\n"

    with pytest.raises(
        replay_experiment.Lfm2NormalizedInputReplayError,
        match="must be new",
    ):
        replay_experiment.replay(
            capsule_directory=capsule.root,
            result_json=result_path,
        )


def test_optional_capture_path_is_propagated_to_the_exact_compiler_boundary() -> None:
    name = "normalized_input_request_capture_directory"
    callables = (
        exact4x_runner.run,
        compile_install_lfm2_semantic_layer_stack_stage,
        compile_install_lfm2_semantic_layer_stack_postfold,
        _install_mlp,
        prepare_lfm2_semantic_mlp_postfold,
        _full_candidate_state,
    )

    assert all(name in inspect.signature(value).parameters for value in callables)
    assert {action.dest: action for action in exact4x_runner._parser()._actions}[
        name
    ].default is None
    source = inspect.getsource(_full_candidate_state)
    assert source.index("write_lfm2_normalized_input_replay_capsule(") < source.index(
        "normalized_control = compile_lfm2_normalized_input_decoder("
    )
    launcher = exact4x_runner.__file__
    assert launcher is not None
    launch_source = (
        Path(launcher).with_name(
            "remote_launch_lfm2_exact4x_fp32_directional_20260724.sh"
        )
    ).read_text(encoding="utf-8")
    assert (
        '--normalized-input-request-capture-directory "${RUN_DIR}/'
        'normalized-input-request-capsules"'
    ) in launch_source
