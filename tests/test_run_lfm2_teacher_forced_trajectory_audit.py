from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from neural_forge.hf_lfm2_teacher_forced_trajectory_audit import (
    LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA,
    SAME_RUN_HIDDEN_CONTRAST_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "run_lfm2_teacher_forced_trajectory_audit.py"
SPEC = importlib.util.spec_from_file_location("trajectory_audit_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


_D = "a" * 64


class _Payload:
    def __init__(self, tensors):
        self.tensors = tensors
        self.selection_sha256 = _D
        self.session_sha256 = _D
        self.released = False


class _Session:
    cache_key = _D
    evidence_sha256 = _D

    def __init__(self, tensors):
        self.tensors = tensors
        self.selected_names = None
        self.payload = None

    def load_selected_tensors(self, index, names):
        assert index == 0
        self.selected_names = names
        self.payload = _Payload({name: self.tensors[name] for name in names})
        return self.payload

    def release_selected_tensors(self, payload):
        assert payload is self.payload
        payload.released = True


def _trace_case():
    prompt = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    forced = torch.tensor([[3, 1]], dtype=torch.int64)
    tensors = {
        "input.input_ids": prompt,
        "input.attention_mask": torch.ones_like(prompt),
        "input.forced_decode_ids": forced,
        "prefill.output.lm_head_input": torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]
        ),
        "prefill.logits": torch.tensor(
            [[[0.0] * 4, [0.0] * 4, [0.0, 0.0, 1.0, 2.0]]]
        ),
        "decode.000.output.lm_head_input": torch.tensor([[[2.0, 1.0]]]),
        "decode.000.logits": torch.tensor([[[1.0, 2.0, 0.0, 0.0]]]),
        "decode.000.forced_token_ids": forced[:, :1],
        "decode.001.output.lm_head_input": torch.tensor([[[2.0, 2.0]]]),
        "decode.001.logits": torch.tensor([[[1.0, 0.0, 3.0, 0.0]]]),
        "decode.001.forced_token_ids": forced[:, 1:2],
    }
    reference = SimpleNamespace(
        index=0,
        input_sha256=_D,
        record_sha256=_D,
        ordered_tensor_names=tuple(sorted(tensors)),
        record={"capture": {"decode_steps": 2}},
    )
    return tensors, reference


def test_trace_extractor_selects_only_trajectory_endpoints_and_releases() -> None:
    tensors, reference = _trace_case()
    session = _Session(tensors)
    batch = runner.load_parent_trajectory_batch(
        session=session,
        reference=reference,
        prompt_id="p0",
        trace_parent_fingerprint=_D,
        donor_row_count=4,
        hidden_size=2,
    )
    assert torch.equal(batch.prompt_token_ids, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.forced_token_ids, torch.tensor([3, 1]))
    assert tuple(batch.parent_hidden_states.shape) == (3, 2)
    assert tuple(batch.parent_logits.shape) == (3, 4)
    assert session.payload.released is True
    assert all("layer." not in name for name in session.selected_names)
    assert "prefill.output.lm_head_input" in session.selected_names
    assert "decode.001.output.lm_head_input" in session.selected_names


def test_trace_extractor_releases_payload_when_forced_identity_fails() -> None:
    tensors, reference = _trace_case()
    tensors["decode.000.forced_token_ids"] = torch.tensor([[0]])
    session = _Session(tensors)
    with pytest.raises(runner.Lfm2TeacherForcedTrajectoryRunnerError, match="identity"):
        runner.load_parent_trajectory_batch(
            session=session,
            reference=reference,
            prompt_id="p0",
            trace_parent_fingerprint=_D,
            donor_row_count=4,
            hidden_size=2,
        )
    assert session.payload.released is True


def test_parser_requires_authentication_and_explicit_dtype_roles() -> None:
    parser = runner.build_parser()
    required = {
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
    }
    assert {
        "expected_parent_source_file_sha256",
        "expected_parent_table_sha256",
        "expected_checkpoint_inventory_sha256",
        "expected_checkpoint_staged_state_sha256",
        "expected_bridge_artifact_sha256",
        "expected_span_evidence_sha256",
        "expected_codebook_artifact_sha256",
        "expected_output_refit_cache_key",
        "expected_output_refit_result_sha256",
        "expected_trace_cache_key",
        "expected_trace_parent_fingerprint",
        "provenance_dtype",
        "runtime_dtype",
        "metric_dtype",
        "score_dtype",
    }.issubset(required)


def test_runner_source_has_no_parent_loader_training_or_network_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert "from_pretrained" not in source
    assert ".backward(" not in source
    assert "optimizer" not in source.lower()
    assert "requests." not in source
    assert "http://" not in source and "https://" not in source
    assert "load_lfm2_direct_local_model" not in source
    assert "Lfm2DirectLocalCheckpointIO" in source
    assert "selective_authenticated_session" in source
    assert source.index(
        "runtime_abi = build_direct_local_runtime_abi_receipt"
    ) < source.index("cached = _cached_audit")
    assert source.index("cached = _cached_audit") < source.index(
        "checkpoint_io.load_authenticated_portable"
    )


def test_direct_local_runtime_abi_is_preconstruction_and_cache_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Lfm2Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    receipt = runner.build_direct_local_runtime_abi_receipt(
        tmp_path,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        provenance_device="cpu",
        provenance_dtype=torch.bfloat16,
    )
    body = dict(receipt)
    supplied = body.pop("abi_sha256")
    assert supplied == runner.canonical_sha256(body)
    assert receipt["child_model_objects_constructed"] == 0
    assert receipt["network_access_performed"] is False
    assert receipt["packages"]["torch"] == str(torch.__version__)
    assert set(receipt["packages"]) == {
        "torch",
        "transformers",
        "safetensors",
        "causal_conv1d",
    }
    source_records = {
        row["identifier"]: row for row in receipt["local_sources"]
    }
    assert {
        "neural_forge.hf_lfm2_teacher_forced_trajectory_audit",
        "neural_forge.hf_lfm2_direct_local",
        "neural_forge.hf_lfm2_dynamic_blank_materializer",
        "neural_forge.hf_lfm2_dynamic_physical_layout",
        "neural_forge.hf_lfm2_semantic_step_token_fold",
        "transformers.configuration_lfm2",
        "transformers.modeling_lfm2",
    }.issubset(source_records)
    assert all(row["file_sha256"] for row in source_records.values())
    execution = receipt["topology_execution"]
    assert execution["attention_layer_count"] == 1
    assert execution["short_conv_layer_count"] == 1
    assert execution["short_conv_backend_selected"] == (
        "torch-depthwise-conv1d-slow"
    )
    assert execution["body_use_cache"] is False
    assert execution["attention_backend_source"]["file_sha256"]

    changed = runner.build_direct_local_runtime_abi_receipt(
        tmp_path,
        runtime_device="cpu",
        runtime_dtype=torch.float64,
        provenance_device="cpu",
        provenance_dtype=torch.bfloat16,
    )
    assert changed["abi_sha256"] != receipt["abi_sha256"]

    request = runner.build_teacher_forced_trajectory_audit_request(
        artifact_bindings={"runtime_abi": receipt, "trace": "a" * 64},
        runtime_dtype_policy=runner.resolve_runtime_dtype_policy("float32", "cpu"),
        metric_dtype="float32",
        score_dtype="float32",
        donor_row_count=8,
        target_row_count=8,
        hidden_size=8,
    )
    changed_request = runner.build_teacher_forced_trajectory_audit_request(
        artifact_bindings={"runtime_abi": changed, "trace": "a" * 64},
        runtime_dtype_policy=runner.resolve_runtime_dtype_policy("float64", "cpu"),
        metric_dtype="float32",
        score_dtype="float32",
        donor_row_count=8,
        target_row_count=8,
        hidden_size=8,
    )
    assert changed_request["cache_key"] != request["cache_key"]

    original_file_sha256 = runner._file_sha256

    def changed_source_sha256(path: Path) -> str:
        if Path(path).name == "modeling_lfm2.py":
            return "f" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runner, "_file_sha256", changed_source_sha256)
    source_changed = runner.build_direct_local_runtime_abi_receipt(
        tmp_path,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        provenance_device="cpu",
        provenance_dtype=torch.bfloat16,
    )
    assert source_changed["abi_sha256"] != receipt["abi_sha256"]
    source_changed_request = runner.build_teacher_forced_trajectory_audit_request(
        artifact_bindings={"runtime_abi": source_changed, "trace": "a" * 64},
        runtime_dtype_policy=runner.resolve_runtime_dtype_policy("float32", "cpu"),
        metric_dtype="float32",
        score_dtype="float32",
        donor_row_count=8,
        target_row_count=8,
        hidden_size=8,
    )
    assert source_changed_request["cache_key"] != request["cache_key"]


def test_checkpoint_inventory_reuses_producer_compatible_recursive_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "tokenizer"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")
    inventory = runner._checkpoint_inventory(tmp_path)
    expected = [
        {
            "relative_path": path.relative_to(tmp_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": runner._file_sha256(path),
        }
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    ]
    assert inventory["files"] == expected
    assert inventory["producer_compatible_inventory"] is True
    assert inventory["inventory_sha256"] == runner.canonical_sha256(expected)


def test_exact_parent_table_reuses_sparse_codebook_identity(
    tmp_path: Path,
) -> None:
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    path = tmp_path / "model.safetensors"
    save_file({runner.PARENT_TIED_KEY: table}, path)
    selected = table[:5].contiguous()
    loaded, receipt = runner.load_exact_parent_table(
        tmp_path,
        tensor_key=runner.PARENT_TIED_KEY,
        donor_row_count=5,
        hidden_size=4,
        expected_source_file_sha256=runner._file_sha256(path),
        expected_table_sha256=runner._sparse_codebook_table_sha256(selected),
    )
    assert torch.equal(loaded, selected)
    assert receipt["selected_sparse_codebook_table_sha256"] == (
        runner._sparse_codebook_table_sha256(selected)
    )
    assert receipt["selected_audit_table_sha256"] == runner.tensor_sha256(selected)


def test_checkpoint_authentication_binds_request_and_trace_identity() -> None:
    tensors, reference = _trace_case()
    result_body = {
        "schema": LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA,
        "batch_index": 0,
        "prompt_id": "p0",
        "lanes": {
            name: {"summary": {}, "tokens": []}
            for name in (
                "exact_exact",
                "folded_exact",
                "exact_folded",
                "folded_folded",
            )
        },
        "target_token_spans_used_as_body_steps": False,
        "same_run_hidden_contrast_contract": dict(
            SAME_RUN_HIDDEN_CONTRAST_CONTRACT
        ),
        "operation_audit": dict(runner._OPERATION_AUDIT) if hasattr(runner, "_OPERATION_AUDIT") else {
            "parent_forward_calls": 0,
            "parent_trace_recapture_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "global_residual_maps_fit": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "teacher_forced_donor_trajectory": True,
            "argmax_branching_used_as_body_input": False,
            "target_token_spans_used_as_body_steps": False,
        },
    }
    result = {**result_body, "result_sha256": runner.canonical_sha256(result_body)}
    checkpoint = runner._batch_checkpoint(
        request_cache_key=_D, reference=reference, prompt_result=result
    )
    loaded = runner._authenticate_batch_checkpoint(
        checkpoint, request_cache_key=_D, reference=reference
    )
    assert loaded["request_cache_key"] == _D
    with pytest.raises(runner.Lfm2TeacherForcedTrajectoryRunnerError):
        runner._authenticate_batch_checkpoint(
            checkpoint, request_cache_key="b" * 64, reference=reference
        )
