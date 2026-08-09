from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_autoregressive_hankel_phase_cache import (  # noqa: E402
    Lfm2AutoregressiveHankelPhaseAuthenticationError,
    describe_lfm2_autoregressive_hankel_phase,
    load_or_build_lfm2_autoregressive_hankel_statistics,
)
from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    Lfm2AutoregressiveHankelStatisticsError,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
)
import neural_forge.hf_lfm2_autoregressive_hankel_phase_cache as phase_cache  # noqa: E402


_PARENT = "1" * 64
_CACHE = "2" * 64
_CORPUS = "3" * 64
_TOPOLOGY = "4" * 64
_TRACE_IMPL = "5" * 64
_SESSION = "6" * 64
_INPUT = "7" * 64
_RECORD = "8" * 64


class _Payload:
    def __init__(
        self,
        *,
        index: int,
        input_sha256: str,
        names: tuple[str, ...],
        session_sha256: str,
        selection_sha256: str,
        tensors,
    ) -> None:
        self.batch_index = index
        self.input_sha256 = input_sha256
        self.ordered_tensor_names = names
        self.session_sha256 = session_sha256
        self.selection_sha256 = selection_sha256
        self.tensors = tensors
        self.released = False


class _Session:
    closed = False

    def __init__(
        self,
        reference,
        tensors,
        *,
        cache_key: str = _CACHE,
        session_sha256: str = _SESSION,
        parent_sha256: str = _PARENT,
        topology_sha256: str = _TOPOLOGY,
    ) -> None:
        self.cache_key = cache_key
        self.evidence_sha256 = session_sha256
        self.batch_references = (reference,)
        self.tensors = tensors
        self.manifest = {
            "parent": {"identity_sha256": parent_sha256},
            "layer_topology": {
                "layer_topology_sha256": topology_sha256,
            },
            "corpus": {"corpus_sha256": _CORPUS},
            "implementation": {
                "implementation_sha256": _TRACE_IMPL,
            },
        }
        self.load_calls = 0
        self.release_calls = 0
        self.payload = None

    def load_selected_tensors(self, index, names):
        self.load_calls += 1
        reference = self.batch_references[index]
        inventory = reference.record["tensors"]
        logical_bytes = sum(int(inventory[name]["size_bytes"]) for name in names)
        selection_body = {
            "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
            "session_sha256": self.evidence_sha256,
            "cache_key": self.cache_key,
            "batch_index": index,
            "input_sha256": reference.input_sha256,
            "record_sha256": reference.record_sha256,
            "ordered_tensor_names": list(names),
            "ordered_tensor_descriptors": [dict(inventory[name]) for name in names],
            "logical_bytes": logical_bytes,
        }
        self.payload = _Payload(
            index=index,
            input_sha256=reference.input_sha256,
            names=names,
            session_sha256=self.evidence_sha256,
            selection_sha256=canonical_sha256(selection_body),
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


def _case():
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
    return _Session(reference, tensors), reference


def _assert_same(left, right):
    for name in (
        "lag_pair_counts",
        "right_censor_counts",
        "state_sum",
        "future_sum",
        "state_second_moment",
        "future_second_moment",
        "state_future_cross",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def test_cold_batch_checkpoints_then_hit_loads_no_source_tensor(tmp_path, monkeypatch):
    session, reference = _case()
    cold = load_or_build_lfm2_autoregressive_hankel_statistics(
        tmp_path,
        session=session,
        reference=reference,
        future_lags=(0, 2, 4),
    )
    assert cold.cache_hit is False
    assert session.load_calls == 1
    assert session.release_calls == 1
    assert session.payload.released is True
    assert cold.receipt["checkpoint_granularity_batches"] == 1
    assert cold.receipt["raw_uncentered_moments"] is True
    assert cold.receipt["globally_center_only_after_aggregate"] is True
    assert cold.receipt["boundary_names"] == [
        "layer.000.residual.pre",
        "layer.000.residual.post_layer",
        "layer.001.residual.post_layer",
    ]
    assert cold.receipt["phase_key"]["future_lags"] == [0, 2, 4]
    assert cold.receipt["phase_key"]["parent_identity_sha256"] == _PARENT
    assert cold.receipt["phase_key"]["member_cache_key"] == _CACHE
    assert cold.receipt["phase_key"]["member_session_sha256"] == _SESSION
    assert cold.receipt["phase_key"]["trace_input_sha256"] == _INPUT
    assert cold.receipt["phase_key"]["trace_record_sha256"] == _RECORD
    assert cold.receipt["phase_key"]["selected_payload_sha256"] == (
        session.payload.selection_sha256
    )
    assert {path.name for path in cold.path.iterdir()} == {
        "receipt.json",
        "statistics.safetensors",
    }

    monkeypatch.setattr(
        phase_cache._statistics,
        "extract_lfm2_autoregressive_hankel_batch",
        lambda **_kwargs: pytest.fail("cache hit re-entered extractor"),
    )
    monkeypatch.setattr(
        phase_cache._statistics,
        "accumulate_lfm2_ahb_trajectory",
        lambda *_args, **_kwargs: pytest.fail("cache hit re-entered accumulator"),
    )
    hit = load_or_build_lfm2_autoregressive_hankel_statistics(
        tmp_path,
        session=session,
        reference=reference,
        future_lags=(0, 2, 4),
    )
    assert hit.cache_hit is True
    assert session.load_calls == 1
    assert session.release_calls == 1
    assert hit.path == cold.path
    assert hit.phase_key_sha256 == cold.phase_key_sha256
    _assert_same(cold.statistics, hit.statistics)


def test_hit_fails_closed_when_safetensors_bytes_change(tmp_path, monkeypatch):
    session, reference = _case()
    cold = load_or_build_lfm2_autoregressive_hankel_statistics(
        tmp_path,
        session=session,
        reference=reference,
        future_lags=(0, 1),
    )
    tensor_path = cold.path / "statistics.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 0x01
    tensor_path.write_bytes(payload)
    monkeypatch.setattr(
        phase_cache._statistics,
        "extract_lfm2_autoregressive_hankel_batch",
        lambda **_kwargs: pytest.fail("corrupt hit silently rebuilt source"),
    )

    with pytest.raises(
        Lfm2AutoregressiveHankelPhaseAuthenticationError,
        match="bytes changed",
    ):
        load_or_build_lfm2_autoregressive_hankel_statistics(
            tmp_path,
            session=session,
            reference=reference,
            future_lags=(0, 1),
        )
    assert session.load_calls == 1
    assert session.release_calls == 1


def test_accumulator_failure_releases_source_and_publishes_no_shard(tmp_path):
    session, reference = _case()
    identity = describe_lfm2_autoregressive_hankel_phase(
        session=session,
        reference=reference,
        future_lags=(0, 5),
    )
    with pytest.raises(Lfm2AutoregressiveHankelStatisticsError, match="smaller"):
        load_or_build_lfm2_autoregressive_hankel_statistics(
            tmp_path,
            session=session,
            reference=reference,
            future_lags=(0, 5),
        )
    assert session.load_calls == 1
    assert session.release_calls == 1
    assert session.payload.released is True
    expected = (
        tmp_path / "shards" / identity.phase_key_sha256[:2] / identity.phase_key_sha256
    )
    assert not expected.exists()


def test_identity_binds_sparse_lags_parent_topology_and_numeric_source_abis():
    session, reference = _case()
    baseline = describe_lfm2_autoregressive_hankel_phase(
        session=session,
        reference=reference,
        future_lags=(0, 2, 4),
    )
    other_lags = describe_lfm2_autoregressive_hankel_phase(
        session=session,
        reference=reference,
        future_lags=(0, 1, 4),
    )
    other_parent_session = _Session(
        reference,
        session.tensors,
        parent_sha256="9" * 64,
    )
    other_parent = describe_lfm2_autoregressive_hankel_phase(
        session=other_parent_session,
        reference=reference,
        future_lags=(0, 2, 4),
    )
    other_topology_session = _Session(
        reference,
        session.tensors,
        topology_sha256="a" * 64,
    )
    other_topology = describe_lfm2_autoregressive_hankel_phase(
        session=other_topology_session,
        reference=reference,
        future_lags=(0, 2, 4),
    )

    assert (
        len(
            {
                baseline.phase_key_sha256,
                other_lags.phase_key_sha256,
                other_parent.phase_key_sha256,
                other_topology.phase_key_sha256,
            }
        )
        == 4
    )
    body = baseline.body
    assert body["future_lags"] == [0, 2, 4]
    assert body["boundary_topology"]["boundary_count"] == 3
    assert len(body["boundary_topology_sha256"]) == 64
    assert len(body["numeric_abi_sha256"]) == 64
    assert body["numeric_abi"]["accumulation_dtype"] == "float64"
    assert body["numeric_abi"]["count_dtype"] == "int64"
    assert len(body["statistics_source_sha256"]) == 64
    assert len(body["phase_cache_source_sha256"]) == 64
    assert body["checkpoint_granularity_batches"] == 1
    assert body["member_shards_independently_reusable"] is True


class _EvidenceSet:
    def __init__(self, member_session):
        self.members = (SimpleNamespace(session=member_session),)
        self._input_sha256 = member_session.batch_references[0].input_sha256

    def batch(self, index):
        assert index == 17
        return SimpleNamespace(
            member_ordinal=0,
            member_batch_index=0,
            input_sha256=self._input_sha256,
        )


def test_merged_coordinates_reuse_the_same_independent_member_shard(tmp_path):
    member_session, member_reference = _case()
    evidence_set = _EvidenceSet(member_session)
    merged_reference = SimpleNamespace(
        index=17,
        input_sha256=member_reference.input_sha256,
        record_sha256=member_reference.record_sha256,
        ordered_tensor_names=member_reference.ordered_tensor_names,
        record=member_reference.record,
    )
    merged_session = SimpleNamespace(
        evidence_set=evidence_set,
        cache_key="b" * 64,
        evidence_sha256="c" * 64,
    )
    direct_identity = describe_lfm2_autoregressive_hankel_phase(
        session=member_session,
        reference=member_reference,
        future_lags=(0, 2),
    )
    merged_identity = describe_lfm2_autoregressive_hankel_phase(
        session=merged_session,
        reference=merged_reference,
        future_lags=(0, 2),
    )
    assert merged_identity.phase_key_sha256 == direct_identity.phase_key_sha256
    assert merged_identity.body["member_cache_key"] == _CACHE
    assert merged_identity.body["member_session_sha256"] == _SESSION

    cold = load_or_build_lfm2_autoregressive_hankel_statistics(
        tmp_path,
        session=merged_session,
        reference=merged_reference,
        future_lags=(0, 2),
    )
    hit = load_or_build_lfm2_autoregressive_hankel_statistics(
        tmp_path,
        session=member_session,
        reference=member_reference,
        future_lags=(0, 2),
    )
    assert cold.cache_hit is False
    assert hit.cache_hit is True
    assert cold.path == hit.path
    assert member_session.load_calls == 1
    assert member_session.release_calls == 1
