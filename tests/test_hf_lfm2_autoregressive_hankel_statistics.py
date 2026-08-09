from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    Lfm2AutoregressiveHankelStatisticsError,
    accumulate_lfm2_ahb_trajectory,
    extract_lfm2_autoregressive_hankel_batch,
)


_CACHE = "a" * 64
_SESSION = "b" * 64
_INPUT = "c" * 64
_RECORD = "d" * 64
_SELECTION = "e" * 64


class _Payload:
    def __init__(self, *, index, names, tensors):
        self.batch_index = index
        self.input_sha256 = _INPUT
        self.ordered_tensor_names = names
        self.session_sha256 = _SESSION
        self.selection_sha256 = _SELECTION
        self.tensors = tensors
        self.released = False


class _Session:
    cache_key = _CACHE
    evidence_sha256 = _SESSION

    def __init__(self, tensors):
        self.tensors = tensors
        self.selected_names = None
        self.payload = None
        self.load_calls = 0
        self.release_calls = 0

    def load_selected_tensors(self, index, names):
        self.load_calls += 1
        self.selected_names = names
        self.payload = _Payload(
            index=index,
            names=names,
            tensors={name: self.tensors[name] for name in names},
        )
        return self.payload

    def release_selected_tensors(self, payload):
        assert payload is self.payload
        self.release_calls += 1
        payload.released = True


def _descriptor(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _trajectory_case():
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.int64)
    forced = torch.tensor([[13, 14]], dtype=torch.int64)
    tensors = {
        "input.input_ids": prompt,
        "input.attention_mask": torch.ones_like(prompt),
        "input.forced_decode_ids": forced,
    }
    for phase_index, (phase, state_indices) in enumerate(
        (
            ("prefill", (0, 1, 2)),
            ("decode.000", (3,)),
            ("decode.001", (4,)),
        )
    ):
        base = torch.tensor(
            [[[float(index), 0.0] for index in state_indices]],
            dtype=torch.float32,
            requires_grad=phase_index == 0,
        )
        post_zero = base.detach().clone()
        post_zero[:, :, 1] = 1.0
        pre_one = post_zero.clone()
        post_one = base.detach().clone()
        post_one[:, :, 1] = 2.0
        behavior = base.detach().clone()
        behavior[:, :, 1] = 3.0
        tensors[f"{phase}.layer.000.residual.pre"] = base
        tensors[f"{phase}.layer.000.residual.post_layer"] = post_zero
        tensors[f"{phase}.layer.001.residual.pre"] = pre_one
        tensors[f"{phase}.layer.001.residual.post_layer"] = post_one
        tensors[f"{phase}.output.lm_head_input"] = behavior
        tensors[f"{phase}.logits"] = torch.zeros(
            (1, len(state_indices), 101), dtype=torch.float32
        )
    inventory = {name: _descriptor(value) for name, value in tensors.items()}
    reference = SimpleNamespace(
        index=0,
        input_sha256=_INPUT,
        record_sha256=_RECORD,
        ordered_tensor_names=tuple(sorted(tensors)),
        record={
            "capture": {
                "decode_steps": 2,
                "prefill_sample_start": 0,
                "prefill_sample_count": 3,
            },
            "tensors": inventory,
        },
    )
    return tensors, reference


def _assert_zero_execution(receipt):
    audit = receipt["operation_audit"]
    assert audit["gradients_used"] is False
    assert audit["logit_tensors_loaded"] == 0
    for name, value in audit.items():
        if name not in {"gradients_used"}:
            assert value == 0, name


def test_pure_accumulator_forms_exact_future_lag_moments_and_censors() -> None:
    states = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2)
    states.requires_grad_(True)
    behavior = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 2.0], [8.0, 3.0]],
        requires_grad=True,
    )
    with torch.enable_grad():
        result = accumulate_lfm2_ahb_trajectory(
            states,
            behavior,
            max_horizon=2,
            source_receipt_sha256=_SELECTION,
        )

    assert result.lag_pair_counts.tolist() == [5, 4, 3]
    assert result.right_censor_counts.tolist() == [0, 1, 2]
    assert tuple(result.state_sum.shape) == (3, 3, 2)
    assert tuple(result.future_sum.shape) == (3, 2)
    assert tuple(result.state_second_moment.shape) == (3, 3, 2, 2)
    assert tuple(result.future_second_moment.shape) == (3, 2, 2)
    assert tuple(result.state_future_cross.shape) == (3, 3, 2, 2)

    expected_states = states.detach().to(torch.float64)
    expected_behavior = behavior.detach().to(torch.float64)
    for lag, pair_count in enumerate((5, 4, 3)):
        left = expected_states[:, :pair_count, :]
        right = expected_behavior[lag:, :]
        assert torch.equal(result.state_sum[:, lag, :], left.sum(dim=1))
        assert torch.equal(result.future_sum[lag, :], right.sum(dim=0))
        assert torch.equal(
            result.state_second_moment[:, lag],
            torch.einsum("jnh,jnk->jhk", left, left),
        )
        assert torch.equal(result.future_second_moment[lag], right.T @ right)
        assert torch.equal(
            result.state_future_cross[:, lag],
            torch.einsum("jnh,nk->jhk", left, right),
        )
    for tensor in (
        result.state_sum,
        result.future_sum,
        result.state_second_moment,
        result.future_second_moment,
        result.state_future_cross,
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
    assert result.receipt["contains_nonzero_future_lags"] is True
    assert result.receipt["static_gram_only_method"] is False
    assert result.receipt["short_conv_only_fhvr_method"] is False
    assert result.receipt["terminal_rows_zero_padded"] == 0
    assert result.receipt["tensor_payload"]["tensor_count"] == 7
    assert len(result.receipt["tensor_payload"]["payload_sha256"]) == 64
    assert set(result.receipt["tensor_payload"]["tensors"]) == {
        "lag_pair_counts",
        "right_censor_counts",
        "state_sum",
        "future_sum",
        "state_second_moment",
        "future_second_moment",
        "state_future_cross",
    }
    _assert_zero_execution(result.receipt)


def test_accumulator_rejects_horizon_zero_and_uncensored_overrun() -> None:
    states = torch.zeros((2, 3, 2), dtype=torch.float32)
    behavior = torch.zeros((3, 2), dtype=torch.float32)
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="positive integer"
    ):
        accumulate_lfm2_ahb_trajectory(states, behavior, max_horizon=0)
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="smaller"
    ):
        accumulate_lfm2_ahb_trajectory(states, behavior, max_horizon=3)


def test_sparse_future_lags_preserve_exact_censoring_without_dense_storage() -> None:
    states = torch.arange(20, dtype=torch.float64).reshape(2, 5, 2)
    behavior = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    result = accumulate_lfm2_ahb_trajectory(
        states,
        behavior,
        future_lags=(0, 2, 4),
    )
    assert result.lag_pair_counts.tolist() == [5, 3, 1]
    assert result.right_censor_counts.tolist() == [0, 2, 4]
    assert tuple(result.state_second_moment.shape) == (2, 3, 2, 2)
    assert tuple(result.state_future_cross.shape) == (2, 3, 2, 2)
    assert result.receipt["future_lags"] == [0, 2, 4]
    assert result.receipt["dense_future_lag_schedule"] is False
    for lag_index, lag in enumerate((0, 2, 4)):
        left = states[:, : 5 - lag, :]
        right = behavior[lag:, :]
        assert torch.equal(
            result.state_second_moment[:, lag_index],
            torch.einsum("jnh,jnk->jhk", left, left),
        )
        assert torch.equal(
            result.state_future_cross[:, lag_index],
            torch.einsum("jnh,nk->jhk", left, right),
        )
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="sorted, unique"
    ):
        accumulate_lfm2_ahb_trajectory(
            states,
            behavior,
            future_lags=(0, 4, 2),
        )
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="exactly one"
    ):
        accumulate_lfm2_ahb_trajectory(
            states,
            behavior,
            max_horizon=2,
            future_lags=(0, 2),
        )


def test_authenticated_extractor_reconstructs_every_boundary_without_logits() -> None:
    tensors, reference = _trajectory_case()
    session = _Session(tensors)
    with torch.enable_grad():
        batch = extract_lfm2_autoregressive_hankel_batch(
            session=session,
            reference=reference,
        )

    assert tuple(batch.boundary_rows.shape) == (3, 5, 2)
    assert tuple(batch.behavior_rows.shape) == (5, 2)
    assert batch.boundary_names == (
        "layer.000.residual.pre",
        "layer.000.residual.post_layer",
        "layer.001.residual.post_layer",
    )
    assert torch.equal(
        batch.boundary_rows[:, :, 1],
        torch.tensor(
            [[0.0] * 5, [1.0] * 5, [2.0] * 5], dtype=torch.float64
        ),
    )
    assert torch.equal(
        batch.boundary_rows[:, :, 0],
        torch.tensor([[0, 1, 2, 3, 4]] * 3, dtype=torch.float64),
    )
    assert torch.equal(
        batch.behavior_rows,
        torch.tensor(
            [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3]],
            dtype=torch.float64,
        ),
    )
    assert batch.state_token_ids.tolist() == [10, 11, 12, 13, 14]
    assert batch.observed_action_token_ids.tolist() == [11, 12, 13, 14]
    assert batch.prefix_positions.tolist() == [0, 1, 2, 3, 4]
    assert session.load_calls == 1
    assert session.release_calls == 1
    assert session.payload.released is True
    assert all("logits" not in name for name in session.selected_names)
    assert "prefill.layer.001.residual.pre" not in session.selected_names
    assert batch.receipt["selected_payload_sha256"] == _SELECTION
    assert batch.receipt["model_wide_residual_boundaries_present"] is True
    assert batch.receipt["future_lag_statistics_required"] is True
    assert batch.receipt["static_gram_only_method"] is False
    assert batch.receipt["short_conv_only_fhvr_method"] is False
    assert batch.receipt["tensor_payload"]["tensor_count"] == 5
    assert len(batch.receipt["tensor_payload"]["payload_sha256"]) == 64
    _assert_zero_execution(batch.receipt)
    for value in (batch.boundary_rows, batch.behavior_rows):
        assert value.dtype == torch.float64
        assert value.requires_grad is False
        assert value.grad_fn is None

    statistics = accumulate_lfm2_ahb_trajectory(
        batch.boundary_rows,
        batch.behavior_rows,
        max_horizon=2,
        source_receipt_sha256=batch.receipt["receipt_sha256"],
    )
    assert statistics.lag_pair_counts.tolist() == [5, 4, 3]
    assert statistics.receipt["source_receipt_sha256"] == (
        batch.receipt["receipt_sha256"]
    )

    changed = accumulate_lfm2_ahb_trajectory(
        batch.boundary_rows + 1.0,
        batch.behavior_rows,
        max_horizon=2,
    )
    assert changed.receipt["tensor_payload"]["payload_sha256"] != (
        statistics.receipt["tensor_payload"]["payload_sha256"]
    )


def test_authenticated_extractor_releases_payload_on_post_load_failure() -> None:
    tensors, reference = _trajectory_case()
    tensors["decode.001.output.lm_head_input"][0, 0, 0] = float("nan")
    session = _Session(tensors)
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="non-finite"
    ):
        extract_lfm2_autoregressive_hankel_batch(
            session=session,
            reference=reference,
        )
    assert session.load_calls == 1
    assert session.release_calls == 1
    assert session.payload.released is True


def test_authenticated_extractor_rejects_a_discontinuous_residual_chain() -> None:
    tensors, reference = _trajectory_case()
    reference.record["tensors"][
        "decode.000.layer.001.residual.pre"
    ] = _descriptor(torch.full((1, 1, 2), 99.0))
    session = _Session(tensors)
    with pytest.raises(
        Lfm2AutoregressiveHankelStatisticsError, match="discontinuous"
    ):
        extract_lfm2_autoregressive_hankel_batch(
            session=session,
            reference=reference,
        )
    assert session.load_calls == 0
    assert session.release_calls == 0
