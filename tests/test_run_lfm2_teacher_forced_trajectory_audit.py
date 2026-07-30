from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
