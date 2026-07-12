from __future__ import annotations

import sys
from pathlib import Path

import torch

import neural_forge.hf_target_native_behavior_distillation as trainer


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_stage as runner  # noqa: E402


class _ActualFactorPayloadResult:
    def __init__(self, state: trainer.TargetNativeFactorState) -> None:
        self.state = state

    def factor_pack_payload(self):
        return self.state.as_payload()


def test_production_runner_accepts_actual_trainer_factor_payload_abi() -> None:
    names = ("gud.gate_left", "qkvo.q_left")
    tensors = (
        torch.arange(6, dtype=torch.float16).reshape(2, 3),
        torch.arange(8, dtype=torch.float16).reshape(2, 2, 2),
    )
    tensor_sha256s = tuple(trainer._tensor_sha256(value) for value in tensors)
    manifest_rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": digest,
        }
        for name, value, digest in zip(names, tensors, tensor_sha256s, strict=True)
    ]
    state = trainer.TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=tensor_sha256s,
        factor_state_sha256=trainer._canonical_sha256(manifest_rows),
    )

    payload = runner.ProductionStageRuntime().factor_pack_payload(
        _ActualFactorPayloadResult(state)
    )

    assert isinstance(payload["factors"], list)
    assert {value["name"] for value in payload["factors"]} == set(names)
    assert set(payload["factor_tensors"]) == set(names)
    assert all(
        torch.equal(payload["factor_tensors"][name], value)
        for name, value in zip(names, tensors, strict=True)
    )
    assert payload["factor_state_sha256"] == state.factor_state_sha256
    assert payload["parent_state_included"] is False
    assert payload["native_child_state_included"] is False
    assert payload["optimizer_state_included"] is False


def test_production_runner_preserves_v2_lexical_factor_payload_abi() -> None:
    names = (
        "gud.gate_left",
        "lexical.embedding_left",
        "lexical.lm_head_right",
        "qkvo.q_left",
    )
    tensors = tuple(
        torch.full((2, 2), float(index), dtype=torch.float32)
        for index in range(len(names))
    )
    tensor_sha256s = tuple(trainer._tensor_sha256(value) for value in tensors)
    manifest_rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": digest,
        }
        for name, value, digest in zip(names, tensors, tensor_sha256s, strict=True)
    ]
    state = trainer.TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=tensor_sha256s,
        factor_state_sha256=trainer._factor_state_sha256(
            manifest_rows,
            schema=trainer.BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            lexical_rank=32,
        ),
        schema=trainer.BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
        lexical_rank=32,
    )

    payload = runner.ProductionStageRuntime().factor_pack_payload(
        _ActualFactorPayloadResult(state)
    )

    assert payload["schema"] == runner.TRAINER_FACTOR_PACK_V2_SCHEMA
    assert payload["lexical_rank"] == 32
    assert payload["factor_count"] == len(names)
    assert set(payload["factor_tensors"]) == set(names)
