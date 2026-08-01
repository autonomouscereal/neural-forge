from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.dynamic_audio_fold import (
    BEHAVIOR_SCHEMA,
    GATE_REPORT_SCHEMA,
    TRACE_CACHE_SCHEMA,
    AudioBehaviorGateConfig,
    AudioParentBehaviorCache,
    AudioTraceStatisticsCache,
    DynamicAudioFoldError,
    NativeAudioBehaviorGate,
    audio_lane_status,
    build_native_audio_fold_trace_cache,
    load_audio_parent_behavior_cache,
    normalize_audio_answer,
    write_audio_parent_output_cache,
)
from neural_forge.dynamic_behavior_gate import (
    AUDIO_HASH_BASIS,
    MediaReference,
    MultimodalWholeOutputCase,
    multimodal_cases_from_manifest,
    resolve_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import (
    CircuitEdge,
    CircuitKind,
    CircuitNode,
    CircuitPort,
    EdgeKind,
    ModalityInterface,
    ModelCircuitGraph,
    ParameterSpec,
    PortKind,
    canonical_sha256,
)
from neural_forge.native_audio_probe import (
    CATEGORIES,
    CLOSED_ANSWERS,
    PROBE_SCHEMA,
    SPLITS,
    WAVETABLE_SHA256,
    NativeAudioProbeError,
    authenticate_native_audio_probe,
    build_native_audio_probe,
    canonical_audio_sha256,
    decoded_audio_sha256,
    read_pcm16_wav,
    write_pcm16_wav,
)


torch = pytest.importorskip("torch")

EXAMPLES_PER_CATEGORY = 4
HIDDEN = 4
SEQUENCE = 8
INNER = 6
DEPTH = 2


# --------------------------------------------------------------------------
# Deterministic probe generation and manifest sealing
# --------------------------------------------------------------------------


def test_same_seed_reproduces_every_decoded_audio_hash(tmp_path: Path) -> None:
    first = build_native_audio_probe(
        tmp_path / "first",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=7,
    )
    second = build_native_audio_probe(
        tmp_path / "second",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=7,
    )
    third = build_native_audio_probe(
        tmp_path / "third",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=8,
    )

    def digests(payload: dict[str, Any]) -> list[str]:
        return [record["audio_sha256"] for record in payload["records"]]

    assert digests(first) == digests(second)
    assert digests(first) != digests(third)
    assert first["manifest_seal"] == second["manifest_seal"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["audio_hash_basis"] == AUDIO_HASH_BASIS
    assert first["wavetable_sha256"] == WAVETABLE_SHA256
    assert first["schema"] == PROBE_SCHEMA
    assert first["total"] == EXAMPLES_PER_CATEGORY * len(CATEGORIES)
    assert first["calibration_count"] + first["heldout_count"] == first["total"]


def test_probe_answers_stay_inside_their_closed_sets(tmp_path: Path) -> None:
    payload = build_native_audio_probe(
        tmp_path / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=1,
    )
    for record in payload["records"]:
        assert record["answer"] in CLOSED_ANSWERS[record["category"]]
        assert record["split"] in SPLITS
        assert record["sample_rate"] == 16_000
        assert record["channels"] == 1
    observed = {record["category"] for record in payload["records"]}
    assert observed == set(CATEGORIES)


def test_synthesized_content_matches_its_declared_class(tmp_path: Path) -> None:
    payload = build_native_audio_probe(
        tmp_path / "probe",
        examples_per_category=3,
        seed=2,
    )
    root = Path(payload["manifest"]).parent
    energies: dict[str, float] = {}
    for record in payload["records"]:
        if record["category"] != "sound_type":
            continue
        samples, _rate, _channels = read_pcm16_wav(root / record["audio"])
        energies[record["answer"]] = float(np.abs(samples.astype(np.int64)).mean())
    assert energies["silence"] == 0.0
    assert energies["tone"] > 0.0
    assert energies["noise"] > 0.0


def test_manifest_authentication_fails_closed_on_any_tamper(tmp_path: Path) -> None:
    payload = build_native_audio_probe(
        tmp_path / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=3,
    )
    manifest_path = Path(payload["manifest"])
    _sealed, probes = authenticate_native_audio_probe(manifest_path)
    assert len(probes) == payload["total"]
    assert probes[0].record_sha256 == canonical_sha256(payload["records"][0])
    assert decoded_audio_sha256(probes[0].path) == probes[0].audio_sha256

    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. A relabeled answer breaks the manifest seal.
    relabeled = copy.deepcopy(original)
    relabeled["records"][0]["answer"] = "5"
    manifest_path.write_text(json.dumps(relabeled), encoding="utf-8")
    with pytest.raises(NativeAudioProbeError):
        authenticate_native_audio_probe(manifest_path)

    # 2. A reseal with a mutated body still fails: the waveform no longer
    #    decodes to the sealed content digest.
    resealed = copy.deepcopy(original)
    resealed["records"][0]["audio_sha256"] = "0" * 64
    body = {key: value for key, value in resealed.items() if key != "manifest_seal"}
    resealed["manifest_seal"] = canonical_sha256(body)
    manifest_path.write_text(json.dumps(resealed), encoding="utf-8")
    with pytest.raises(NativeAudioProbeError):
        authenticate_native_audio_probe(manifest_path)

    # 3. Rewriting the stored waveform fails even with an intact manifest.
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    target = manifest_path.parent / original["records"][0]["audio"]
    samples, rate, channels = read_pcm16_wav(target)
    mutated = np.array(samples, dtype=np.int16)
    mutated[0] = np.int16(int(mutated[0]) ^ 1)
    target.unlink()
    write_pcm16_wav(target, mutated, sample_rate=rate, channels=channels)
    with pytest.raises(NativeAudioProbeError):
        authenticate_native_audio_probe(manifest_path)


def test_manifest_rejects_a_foreign_wavetable(tmp_path: Path) -> None:
    payload = build_native_audio_probe(
        tmp_path / "probe",
        examples_per_category=1,
        seed=0,
    )
    manifest_path = Path(payload["manifest"])
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body.pop("manifest_seal")
    body["wavetable_sha256"] = "f" * 64
    body["manifest_seal"] = canonical_sha256(
        {key: value for key, value in body.items() if key != "manifest_seal"}
    )
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(NativeAudioProbeError, match="wavetable"):
        authenticate_native_audio_probe(manifest_path)


def test_canonical_audio_hash_requires_int16_and_whole_frames() -> None:
    samples = np.array([0, 1, -1, 2], dtype=np.int16)
    expected = hashlib.sha256(
        b"PCM16:16000:1:4\n" + samples.astype("<i2").tobytes()
    ).hexdigest()
    assert canonical_audio_sha256(samples) == expected
    with pytest.raises(NativeAudioProbeError):
        canonical_audio_sha256(samples.astype(np.float32))
    with pytest.raises(NativeAudioProbeError):
        canonical_audio_sha256(np.array([1, 2, 3], dtype=np.int16), channels=2)


# --------------------------------------------------------------------------
# Parent behavior cache authentication
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("audio-lane")
    payload = build_native_audio_probe(
        root / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=11,
    )
    outputs = write_audio_parent_output_cache(
        payload["manifest"],
        root / "parent-outputs.jsonl",
        parent_runner=lambda probe: probe.answer,
    )
    return {"payload": payload, "outputs": str(outputs), "root": str(root)}


@pytest.fixture()
def behavior_cache(probe_suite: dict[str, Any]) -> AudioParentBehaviorCache:
    return load_audio_parent_behavior_cache(
        probe_suite["payload"]["manifest"],
        probe_suite["outputs"],
    )


def test_behavior_cache_authenticates_manifest_outputs_and_waveforms(
    probe_suite: dict[str, Any],
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    assert len(behavior_cache.records) == probe_suite["payload"]["total"]
    assert len(behavior_cache.calibration) == probe_suite["payload"][
        "calibration_count"
    ]
    assert len(behavior_cache.heldout) == probe_suite["payload"]["heldout_count"]
    assert all(record.parent_correct for record in behavior_cache.records)
    behavior_cache.verify()

    pinned = load_audio_parent_behavior_cache(
        probe_suite["payload"]["manifest"],
        probe_suite["outputs"],
        expected_manifest_sha256=behavior_cache.manifest_sha256,
        expected_outputs_sha256=behavior_cache.outputs_sha256,
    )
    assert pinned.fingerprint == behavior_cache.fingerprint

    with pytest.raises(DynamicAudioFoldError):
        load_audio_parent_behavior_cache(
            probe_suite["payload"]["manifest"],
            probe_suite["outputs"],
            expected_outputs_sha256="0" * 64,
        )


def test_behavior_records_expose_content_bound_media(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    record = behavior_cache.records[0]
    media = record.media
    assert isinstance(media, MediaReference)
    assert media.modality == "audio"
    assert media.hash_basis == AUDIO_HASH_BASIS
    assert media.content_sha256 == decoded_audio_sha256(record.audio_path)
    assert media.verify_container() == Path(record.audio_path)

    case = record.as_case()
    assert isinstance(case, MultimodalWholeOutputCase)
    assert case.inputs == (media,)
    assert case.modalities == ("audio", "text")
    assert MultimodalWholeOutputCase.from_dict(case.as_dict()) == case


def test_tampered_parent_outputs_fail_cache_authentication(
    probe_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    lines = Path(probe_suite["outputs"]).read_bytes().splitlines()

    swapped = json.loads(lines[0].decode("utf-8"))
    swapped["raw_output"] = "5"
    forged = tmp_path / "forged.jsonl"
    forged.write_bytes(
        b"\n".join(
            [json.dumps(swapped, sort_keys=True).encode("utf-8"), *lines[1:]]
        )
        + b"\n"
    )
    with pytest.raises(DynamicAudioFoldError, match="authentication"):
        load_audio_parent_behavior_cache(probe_suite["payload"]["manifest"], forged)

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_bytes(b"\n".join(lines[:-1]) + b"\n")
    with pytest.raises(DynamicAudioFoldError, match="count differs"):
        load_audio_parent_behavior_cache(
            probe_suite["payload"]["manifest"],
            truncated,
        )

    unterminated = tmp_path / "unterminated.jsonl"
    unterminated.write_bytes(b"\n".join(lines))
    with pytest.raises(DynamicAudioFoldError, match="truncated"):
        load_audio_parent_behavior_cache(
            probe_suite["payload"]["manifest"],
            unterminated,
        )

    relabeled = json.loads(lines[0].decode("utf-8"))
    relabeled["correct"] = not relabeled["correct"]
    lied = tmp_path / "lied.jsonl"
    lied.write_bytes(
        b"\n".join(
            [json.dumps(relabeled, sort_keys=True).encode("utf-8"), *lines[1:]]
        )
        + b"\n"
    )
    with pytest.raises(DynamicAudioFoldError, match="authentication"):
        load_audio_parent_behavior_cache(probe_suite["payload"]["manifest"], lied)


def test_answer_normalization_is_a_closed_set(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    assert normalize_audio_answer("tone_count", "3") == "3"
    assert normalize_audio_answer("tone_count", " 3 ") == "3"
    assert normalize_audio_answer("tone_count", "6") is None
    assert normalize_audio_answer("tone_count", "three") is None
    assert normalize_audio_answer("pitch_direction", "Rising") == "rising"
    assert normalize_audio_answer("pitch_direction", "it is rising") is None
    assert normalize_audio_answer("sound_type", "noise.") is None
    assert normalize_audio_answer("sound_type", None) is None
    with pytest.raises(DynamicAudioFoldError):
        normalize_audio_answer("melody", "rising")
    assert all(
        record.parent_normalized_prediction == record.expected
        for record in behavior_cache.records
    )


# --------------------------------------------------------------------------
# Parent-relative behavior gate
# --------------------------------------------------------------------------


class _Probe(torch.nn.Module):
    """Smallest model that can prove the declared entrypoint executed."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, values: Any) -> Any:
        return self.linear(values)


def _child_runner(answers: dict[str, str]):
    def runner(model: Any, record: Any) -> str:
        model(torch.zeros(1, HIDDEN))
        return answers.get(record.probe_id, record.expected)

    return runner


def test_gate_accepts_a_child_that_retains_every_parent_answer(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    gate = NativeAudioBehaviorGate(
        behavior_cache,
        child_runner=_child_runner({}),
    )
    decision = gate.evaluate_model(_Probe().eval())

    assert decision.accepted
    metrics = dict(decision.metrics)
    assert metrics["parent_forward_count"] == 0
    assert metrics["child_outputs_scored"] == len(behavior_cache.records)
    assert metrics["heldout_parent_correct_retained"] == metrics[
        "heldout_parent_correct"
    ]
    assert metrics["observed_child_root_forward_calls"] == len(
        behavior_cache.records
    )
    report = gate.last_report
    assert report["schema"] == GATE_REPORT_SCHEMA
    assert report["modality"] == "audio"
    assert report["parent_forward_count"] == 0
    assert report["failures"] == []
    assert report["exact_parent_output_is_diagnostic_only"] is True


def test_gate_rejects_a_heldout_regression(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    victim = behavior_cache.heldout[0]
    wrong = next(
        answer
        for answer in CLOSED_ANSWERS[victim.category]
        if answer != victim.expected
    )
    gate = NativeAudioBehaviorGate(
        behavior_cache,
        child_runner=_child_runner({victim.probe_id: wrong}),
    )
    decision = gate.evaluate_model(_Probe().eval())

    assert not decision.accepted
    assert "heldout:parent_correct_retention" in decision.reason
    failures = gate.last_report["failures"]
    assert "heldout:accuracy" in failures
    assert f"heldout:{victim.category}:accuracy" in failures


def test_gate_rejects_a_direct_answer_regression(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    def prose_runner(model: Any, record: Any) -> str:
        model(torch.zeros(1, HIDDEN))
        return f"I believe the answer is {record.expected}."

    gate = NativeAudioBehaviorGate(behavior_cache, child_runner=prose_runner)
    decision = gate.evaluate_model(_Probe().eval())

    assert not decision.accepted
    failures = gate.last_report["failures"]
    assert "calibration:direct_answers" in failures
    assert "heldout:direct_answers" in failures
    assert all(row["child_normalized_prediction"] is None for row in
               gate.last_report["rows"])


def test_gate_is_parent_relative_rather_than_absolute(
    probe_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = probe_suite["payload"]["manifest"]
    _sealed, probes = authenticate_native_audio_probe(manifest)
    blind = probes[0]

    def stubborn_parent(probe: Any) -> str:
        if probe.probe_id == blind.probe_id:
            return "the clip is unclear"
        return probe.answer

    outputs = write_audio_parent_output_cache(
        manifest,
        tmp_path / "partial-parent.jsonl",
        parent_runner=stubborn_parent,
    )
    cache = load_audio_parent_behavior_cache(manifest, outputs)
    assert not cache.records[0].parent_correct
    assert not cache.records[0].parent_direct_answer

    # A child that mirrors the parent's failure is still promotable.
    gate = NativeAudioBehaviorGate(
        cache,
        child_runner=_child_runner({blind.probe_id: "the clip is unclear"}),
    )
    assert gate.evaluate_model(_Probe().eval()).accepted

    # A child that improves on the parent is also promotable.
    improving = NativeAudioBehaviorGate(cache, child_runner=_child_runner({}))
    decision = improving.evaluate_model(_Probe().eval())
    assert decision.accepted
    metrics = dict(decision.metrics)
    assert metrics["calibration_child_correct"] > metrics[
        "calibration_parent_correct"
    ]


def test_gate_rejects_incomplete_parent_correct_coverage(
    probe_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = probe_suite["payload"]["manifest"]
    _sealed, probes = authenticate_native_audio_probe(manifest)
    blind = next(
        probe
        for probe in probes
        if probe.split == "heldout" and probe.category == "sound_type"
    )
    outputs = write_audio_parent_output_cache(
        manifest,
        tmp_path / "uncovered-parent.jsonl",
        parent_runner=lambda probe: (
            "no idea" if probe.probe_id == blind.probe_id else probe.answer
        ),
    )
    cache = load_audio_parent_behavior_cache(manifest, outputs)
    gate = NativeAudioBehaviorGate(cache, child_runner=_child_runner({}))
    decision = gate.evaluate_model(_Probe().eval())

    assert not decision.accepted
    assert "heldout:sound_type:parent_correct_coverage" in decision.reason
    assert gate.last_report["rows"] == []
    assert dict(decision.metrics)["child_outputs_scored"] == 0

    relaxed = NativeAudioBehaviorGate(
        cache,
        child_runner=_child_runner({}),
        config=AudioBehaviorGateConfig(
            require_parent_correct_task_coverage=False,
            require_split_category_parent_correct_coverage=False,
        ),
    )
    assert relaxed.evaluate_model(_Probe().eval()).accepted


def test_gate_requires_the_declared_inference_entrypoint_to_execute(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    gate = NativeAudioBehaviorGate(
        behavior_cache,
        child_runner=lambda model, record: record.expected,
    )
    with pytest.raises(DynamicAudioFoldError, match="declared inference"):
        gate.evaluate_model(_Probe().eval())


def test_gate_seals_media_capable_cases_into_the_canonical_manifest(
    behavior_cache: AudioParentBehaviorCache,
) -> None:
    gate = NativeAudioBehaviorGate(behavior_cache, child_runner=_child_runner({}))
    binding = resolve_behavior_gate_binding(gate)

    assert binding.baseline_fingerprint == behavior_cache.fingerprint
    assert multimodal_cases_from_manifest(binding.manifest) == behavior_cache.cases()
    coverage = [
        component
        for component in binding.manifest["components"]
        if component["schema"] == "neural-forge-multimodal-case-coverage-v1"
    ]
    assert coverage[0]["input_modalities"] == ["audio"]
    assert coverage[0]["media_input_count"] == len(behavior_cache.records)
    assert binding.manifest["inference_execution"]["entrypoint"] == "__call__"
    assert binding.manifest["baseline"]["parent_outputs_sha256"] == (
        behavior_cache.outputs_sha256
    )


# --------------------------------------------------------------------------
# Audio-conditioned trace evidence
# --------------------------------------------------------------------------


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(HIDDEN, INNER, bias=False)
        self.down_proj = torch.nn.Linear(INNER, HIDDEN, bias=False)

    def forward(self, hidden_states: Any) -> Any:
        return self.down_proj(torch.tanh(self.up_proj(hidden_states)))


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()

    def forward(self, hidden_states: Any) -> Any:
        return hidden_states + self.mlp(hidden_states)


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(DEPTH))


class _AudioParent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Decoder()

    def forward(self, features: Any) -> Any:
        hidden = features
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def _audio_graph() -> ModelCircuitGraph:
    audio_node = CircuitNode(
        node_id="input.audio",
        kind=CircuitKind.MODALITY_INPUT,
        order=0,
        path="audio_front_end",
        layer_index=None,
        inputs=(
            CircuitPort(
                "waveform",
                PortKind.MODALITY,
                ("batch", "frames"),
                modality="audio",
            ),
        ),
        outputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", HIDDEN),
                modality="audio",
            ),
        ),
    )
    layer_nodes = tuple(
        CircuitNode(
            node_id=f"layer.{index:03d}.mlp",
            kind=CircuitKind.GATED_MLP,
            order=index + 1,
            path=f"model.layers.{index}.mlp",
            layer_index=index,
            inputs=(
                CircuitPort(
                    "residual",
                    PortKind.RESIDUAL,
                    ("batch", "sequence", HIDDEN),
                ),
            ),
            outputs=(
                CircuitPort(
                    "residual",
                    PortKind.RESIDUAL,
                    ("batch", "sequence", HIDDEN),
                ),
            ),
            parameters=(
                ParameterSpec(
                    path=f"model.layers.{index}.mlp.up_proj.weight",
                    role="mlp_up",
                    shape=(INNER, HIDDEN),
                    dtype="torch.float32",
                    storage_group=f"storage-{index}-up",
                    trainable=False,
                ),
                ParameterSpec(
                    path=f"model.layers.{index}.mlp.down_proj.weight",
                    role="mlp_down",
                    shape=(HIDDEN, INNER),
                    dtype="torch.float32",
                    storage_group=f"storage-{index}-down",
                    trainable=False,
                ),
            ),
        )
        for index in range(DEPTH)
    )
    edges = [
        CircuitEdge(
            "input.audio",
            "residual",
            "layer.000.mlp",
            "residual",
            EdgeKind.MODALITY,
        )
    ]
    edges.extend(
        CircuitEdge(
            f"layer.{index:03d}.mlp",
            "residual",
            f"layer.{index + 1:03d}.mlp",
            "residual",
            EdgeKind.RESIDUAL,
        )
        for index in range(DEPTH - 1)
    )
    parameter_count = DEPTH * 2 * INNER * HIDDEN
    return ModelCircuitGraph(
        model_name="tiny-audio-parent",
        family="test_audio",
        model_class="tests.TinyAudioParent",
        nodes=(audio_node, *layer_nodes),
        edges=tuple(edges),
        interfaces=(
            ModalityInterface(
                interface_id="audio.input",
                modality="audio",
                direction="input",
                node_id="input.audio",
                port_name="waveform",
                representation="pcm16",
            ),
        ),
        logical_parameter_count=parameter_count,
        unique_parameter_count=parameter_count,
        unique_parameter_bytes=parameter_count * 4,
        config_fingerprint=canonical_sha256({"model": "tiny-audio-parent"}),
    )


def _trace_runner(model: Any, record: Any) -> Any:
    samples, _rate, _channels = read_pcm16_wav(record.audio_path)
    window = SEQUENCE * HIDDEN * 25
    values = np.asarray(samples[:window], dtype=np.float32) / 32_768.0
    features = torch.from_numpy(
        values.reshape(SEQUENCE, HIDDEN, 25).mean(axis=2).reshape(1, SEQUENCE, HIDDEN)
    )
    return model(features)


def test_trace_cache_captures_calibration_only_and_seals_statistics(
    behavior_cache: AudioParentBehaviorCache,
    tmp_path: Path,
) -> None:
    parent = _AudioParent().eval()
    graph = _audio_graph()
    state_before = {
        name: value.detach().clone() for name, value in parent.state_dict().items()
    }

    cache = build_native_audio_fold_trace_cache(
        parent,
        graph,
        behavior_cache,
        trace_runner=_trace_runner,
        destination=tmp_path / "trace-cache",
        sketch_dim=8,
    )

    assert cache.layer_count == DEPTH
    assert cache.sample_runner_calls == len(behavior_cache.calibration)
    assert cache.trace_rows == (
        len(behavior_cache.calibration) * SEQUENCE,
    ) * DEPTH
    assert set(cache.layer_statistics(0)) == {
        "mean_abs",
        "mean_square",
        "sketch",
        "down_norm",
        "residual_endpoint_mean_square_l2",
    }
    assert tuple(cache.layer_statistics(0)["sketch"].shape) == (8, INNER)
    assert float(cache.layer_statistics(0)["mean_abs"].sum()) > 0.0

    # The parent is untouched by evidence capture.
    for name, value in parent.state_dict().items():
        assert torch.equal(value, state_before[name])

    cache.verify(parent_model=parent, parent_graph=graph)
    reopened = AudioTraceStatisticsCache.open(
        tmp_path / "trace-cache",
        behavior_cache=behavior_cache,
    )
    assert reopened.cache_key == cache.cache_key
    for index in range(DEPTH):
        for name, tensor in cache.layer_statistics(index).items():
            assert torch.equal(reopened.layer_statistics(index)[name], tensor)

    manifest = json.loads(
        (tmp_path / "trace-cache" / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["schema"] == TRACE_CACHE_SCHEMA
    assert manifest["modality"] == "audio"
    assert manifest["fold_selection_included"] is False
    assert manifest["evidence_excluded_splits"] == ["heldout"]
    assert manifest["behavior_cache_fingerprint"] == behavior_cache.fingerprint
    assert manifest["statistics_byte_order"] == "little"


def test_trace_cache_reload_fails_closed_on_tamper(
    behavior_cache: AudioParentBehaviorCache,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "trace-cache"
    build_native_audio_fold_trace_cache(
        _AudioParent().eval(),
        _audio_graph(),
        behavior_cache,
        trace_runner=_trace_runner,
        destination=destination,
        sketch_dim=8,
    )

    blob = destination / "statistics.bin"
    payload = bytearray(blob.read_bytes())
    payload[0] ^= 0xFF
    blob.write_bytes(bytes(payload))
    with pytest.raises(DynamicAudioFoldError, match="statistics file"):
        AudioTraceStatisticsCache.open(destination, behavior_cache=behavior_cache)

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["sketch_dim"] = 16
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(DynamicAudioFoldError, match="seal is invalid"):
        AudioTraceStatisticsCache.open(destination, behavior_cache=behavior_cache)


def test_trace_capture_rejects_a_depth_mismatch_and_a_dirty_parent(
    behavior_cache: AudioParentBehaviorCache,
    tmp_path: Path,
) -> None:
    parent = _AudioParent().eval()
    graph = _audio_graph()

    with pytest.raises(DynamicAudioFoldError, match="decoder depth"):
        build_native_audio_fold_trace_cache(
            parent,
            graph,
            behavior_cache,
            trace_runner=_trace_runner,
            destination=tmp_path / "depth-mismatch",
            layer_resolver=lambda model: tuple(model.model.layers)[:1],
            sketch_dim=8,
        )

    dirty = _AudioParent()
    output = dirty(torch.zeros(1, SEQUENCE, HIDDEN) + 0.5).sum()
    output.backward()
    with pytest.raises(DynamicAudioFoldError, match="zero-gradient"):
        build_native_audio_fold_trace_cache(
            dirty,
            graph,
            behavior_cache,
            trace_runner=_trace_runner,
            destination=tmp_path / "dirty-parent",
            sketch_dim=8,
        )


def test_audio_lane_status_declares_its_open_seams() -> None:
    status = audio_lane_status()
    assert status["modality"] == "audio"
    assert status["parent_forwards_during_gating"] == 0
    assert "trace_runner" in status["model_agnostic_seams"]
    # The inspector, selection rule, and fold method landed in the executable
    # wave; what remains open is a production parent and output-side review.
    assert any("inspector" in item for item in status["provided"])
    assert any("parent model choice" in item for item in status["not_provided"])
    assert any("MODALITY_OUTPUT" in item for item in status["not_provided"])
    assert BEHAVIOR_SCHEMA.endswith("-v1")
