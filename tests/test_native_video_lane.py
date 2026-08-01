from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.dynamic_behavior_gate import (
    VIDEO_HASH_BASIS,
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
from neural_forge.dynamic_video_fold import (
    BEHAVIOR_SCHEMA,
    GATE_REPORT_SCHEMA,
    TRACE_CACHE_SCHEMA,
    DynamicVideoFoldError,
    NativeVideoBehaviorGate,
    VideoBehaviorGateConfig,
    VideoParentBehaviorCache,
    VideoTraceStatisticsCache,
    build_native_video_fold_trace_cache,
    load_video_parent_behavior_cache,
    normalize_video_answer,
    video_lane_status,
    write_video_parent_output_cache,
)
from neural_forge.native_video_probe import (
    CATEGORIES,
    CLOSED_ANSWERS,
    CONTAINER_HEADER_BYTES,
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PALETTE_SHA256,
    PROBE_FRAMES,
    PROBE_SCHEMA,
    SPLITS,
    NativeVideoProbeError,
    authenticate_native_video_probe,
    build_native_video_probe,
    canonical_video_sha256,
    decoded_video_sha256,
    read_rgb24_video,
    write_rgb24_video,
)


torch = pytest.importorskip("torch")

EXAMPLES_PER_CATEGORY = 4
HIDDEN = 4
SEQUENCE = PROBE_FRAMES
INNER = 6
DEPTH = 2
_BACKGROUND = np.array([245, 245, 245], dtype=np.uint8)


# --------------------------------------------------------------------------
# Deterministic probe generation and manifest sealing
# --------------------------------------------------------------------------


def test_same_seed_reproduces_every_decoded_video_hash(tmp_path: Path) -> None:
    first = build_native_video_probe(
        tmp_path / "first",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=7,
    )
    second = build_native_video_probe(
        tmp_path / "second",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=7,
    )
    third = build_native_video_probe(
        tmp_path / "third",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=8,
    )

    def digests(payload: dict[str, Any]) -> list[str]:
        return [record["video_sha256"] for record in payload["records"]]

    assert digests(first) == digests(second)
    assert digests(first) != digests(third)
    assert first["manifest_seal"] == second["manifest_seal"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["video_hash_basis"] == VIDEO_HASH_BASIS
    assert first["palette_sha256"] == PALETTE_SHA256
    assert first["schema"] == PROBE_SCHEMA
    assert first["total"] == EXAMPLES_PER_CATEGORY * len(CATEGORIES)
    assert first["calibration_count"] + first["heldout_count"] == first["total"]
    assert first["synthesis"]["floating_point_state"] is False
    assert first["synthesis"]["generation_order_dependent"] is False


def test_probe_answers_stay_inside_their_closed_sets(tmp_path: Path) -> None:
    payload = build_native_video_probe(
        tmp_path / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=1,
    )
    for record in payload["records"]:
        assert record["answer"] in CLOSED_ANSWERS[record["category"]]
        assert record["split"] in SPLITS
        assert record["frames"] == PROBE_FRAMES
        assert record["width"] == FRAME_WIDTH
        assert record["height"] == FRAME_HEIGHT
        assert record["fps"] == FPS
    observed = {record["category"] for record in payload["records"]}
    assert observed == set(CATEGORIES)


def _foreground(frame: Any) -> Any:
    return ~np.all(frame == _BACKGROUND, axis=-1)


def test_synthesized_content_matches_its_declared_class(tmp_path: Path) -> None:
    payload = build_native_video_probe(
        tmp_path / "probe",
        examples_per_category=6,
        seed=2,
    )
    root = Path(payload["manifest"]).parent
    observed_motion: set[str] = set()
    observed_events: set[str] = set()
    for record in payload["records"]:
        frames, _fps = read_rgb24_video(root / record["video"])
        first = _foreground(frames[0])
        last = _foreground(frames[-1])
        if record["category"] == "motion_direction":
            start = np.argwhere(first).mean(axis=0)
            end = np.argwhere(last).mean(axis=0)
            row_shift = float(end[0] - start[0])
            column_shift = float(end[1] - start[1])
            direction = "none"
            if column_shift > 1.0:
                direction = "right"
            elif column_shift < -1.0:
                direction = "left"
            elif row_shift > 1.0:
                direction = "down"
            elif row_shift < -1.0:
                direction = "up"
            assert direction == record["answer"], record["probe_id"]
            observed_motion.add(direction)
        elif record["category"] == "shape_event":
            if first.any() and last.any():
                event = "persists"
            elif last.any():
                event = "appears"
            else:
                event = "disappears"
            assert event == record["answer"], record["probe_id"]
            observed_events.add(event)
        elif record["category"] == "shape_count":
            remaining = set(map(tuple, np.argwhere(first)))
            squares = 0
            while remaining:
                top, left = min(remaining)
                block = {
                    (top + row, left + column)
                    for row in range(8)
                    for column in range(8)
                }
                assert block <= remaining, record["probe_id"]
                remaining -= block
                squares += 1
            assert str(squares) == record["answer"], record["probe_id"]
    assert observed_motion == set(CLOSED_ANSWERS["motion_direction"])
    assert observed_events == set(CLOSED_ANSWERS["shape_event"])


def test_manifest_authentication_fails_closed_on_any_tamper(tmp_path: Path) -> None:
    payload = build_native_video_probe(
        tmp_path / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=3,
    )
    manifest_path = Path(payload["manifest"])
    _sealed, probes = authenticate_native_video_probe(manifest_path)
    assert len(probes) == payload["total"]
    assert probes[0].record_sha256 == canonical_sha256(payload["records"][0])
    assert decoded_video_sha256(probes[0].path) == probes[0].video_sha256

    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. A relabeled answer breaks the manifest seal.
    relabeled = copy.deepcopy(original)
    relabeled["records"][0]["answer"] = "none"
    manifest_path.write_text(json.dumps(relabeled), encoding="utf-8")
    with pytest.raises(NativeVideoProbeError):
        authenticate_native_video_probe(manifest_path)

    # 2. A reseal with a mutated body still fails: the clip no longer decodes
    #    to the sealed content digest.
    resealed = copy.deepcopy(original)
    resealed["records"][0]["video_sha256"] = "0" * 64
    body = {key: value for key, value in resealed.items() if key != "manifest_seal"}
    resealed["manifest_seal"] = canonical_sha256(body)
    manifest_path.write_text(json.dumps(resealed), encoding="utf-8")
    with pytest.raises(NativeVideoProbeError):
        authenticate_native_video_probe(manifest_path)

    # 3. Rewriting one pixel of the stored clip fails even with an intact
    #    manifest.
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    target = manifest_path.parent / original["records"][0]["video"]
    frames, fps = read_rgb24_video(target)
    mutated = np.array(frames, dtype=np.uint8)
    mutated[0, 0, 0, 0] = np.uint8(int(mutated[0, 0, 0, 0]) ^ 1)
    target.unlink()
    write_rgb24_video(target, mutated, fps=fps)
    with pytest.raises(NativeVideoProbeError):
        authenticate_native_video_probe(manifest_path)


def test_manifest_rejects_a_foreign_palette(tmp_path: Path) -> None:
    payload = build_native_video_probe(
        tmp_path / "probe",
        examples_per_category=1,
        seed=0,
    )
    manifest_path = Path(payload["manifest"])
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body.pop("manifest_seal")
    body["palette_sha256"] = "f" * 64
    body["manifest_seal"] = canonical_sha256(
        {key: value for key, value in body.items() if key != "manifest_seal"}
    )
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(NativeVideoProbeError, match="palette"):
        authenticate_native_video_probe(manifest_path)


def test_canonical_video_hash_binds_geometry_rate_and_frame_count() -> None:
    frames = np.zeros((2, 1, 2, 3), dtype=np.uint8)
    frames[1, 0, 1] = (10, 20, 30)
    expected = hashlib.sha256(b"VIDEO:2x1:8:2\n" + frames.tobytes()).hexdigest()
    assert canonical_video_sha256(frames, fps=8) == expected
    assert canonical_video_sha256(frames, fps=16) != expected
    assert canonical_video_sha256(frames.reshape(1, 2, 2, 3), fps=8) != expected
    with pytest.raises(NativeVideoProbeError):
        canonical_video_sha256(frames.astype(np.float32))
    with pytest.raises(NativeVideoProbeError):
        canonical_video_sha256(frames.reshape(2, 2, 3))


def test_container_is_self_describing_and_fails_closed(tmp_path: Path) -> None:
    frames = np.zeros((PROBE_FRAMES, 4, 4, 3), dtype=np.uint8)
    frames[2, 1, 1] = (7, 8, 9)
    path = write_rgb24_video(tmp_path / "clip.nfv", frames, fps=FPS)
    assert path.stat().st_size == CONTAINER_HEADER_BYTES + frames.size

    decoded, fps = read_rgb24_video(path)
    assert fps == FPS
    assert np.array_equal(decoded, frames)
    assert decoded_video_sha256(path) == canonical_video_sha256(frames, fps=FPS)

    truncated = tmp_path / "truncated.nfv"
    truncated.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(NativeVideoProbeError, match="truncated"):
        read_rgb24_video(truncated)

    foreign = tmp_path / "foreign.nfv"
    foreign.write_bytes(b"RIFF" + path.read_bytes()[4:])
    with pytest.raises(NativeVideoProbeError, match="NFVID1"):
        read_rgb24_video(foreign)


# --------------------------------------------------------------------------
# Parent behavior cache authentication
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("video-lane")
    payload = build_native_video_probe(
        root / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=11,
    )
    outputs = write_video_parent_output_cache(
        payload["manifest"],
        root / "parent-outputs.jsonl",
        parent_runner=lambda probe: probe.answer,
    )
    return {"payload": payload, "outputs": str(outputs), "root": str(root)}


@pytest.fixture()
def behavior_cache(probe_suite: dict[str, Any]) -> VideoParentBehaviorCache:
    return load_video_parent_behavior_cache(
        probe_suite["payload"]["manifest"],
        probe_suite["outputs"],
    )


def test_behavior_cache_authenticates_manifest_outputs_and_clips(
    probe_suite: dict[str, Any],
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    assert len(behavior_cache.records) == probe_suite["payload"]["total"]
    assert len(behavior_cache.calibration) == probe_suite["payload"][
        "calibration_count"
    ]
    assert len(behavior_cache.heldout) == probe_suite["payload"]["heldout_count"]
    assert all(record.parent_correct for record in behavior_cache.records)
    behavior_cache.verify()

    pinned = load_video_parent_behavior_cache(
        probe_suite["payload"]["manifest"],
        probe_suite["outputs"],
        expected_manifest_sha256=behavior_cache.manifest_sha256,
        expected_outputs_sha256=behavior_cache.outputs_sha256,
    )
    assert pinned.fingerprint == behavior_cache.fingerprint

    with pytest.raises(DynamicVideoFoldError):
        load_video_parent_behavior_cache(
            probe_suite["payload"]["manifest"],
            probe_suite["outputs"],
            expected_outputs_sha256="0" * 64,
        )


def test_behavior_records_expose_content_bound_media(
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    record = behavior_cache.records[0]
    media = record.media
    assert isinstance(media, MediaReference)
    assert media.modality == "video"
    assert media.hash_basis == VIDEO_HASH_BASIS
    assert media.content_sha256 == decoded_video_sha256(record.video_path)
    assert media.verify_container() == Path(record.video_path)
    assert dict(media.attributes) == {
        "fps": FPS,
        "frames": PROBE_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
    }

    case = record.as_case()
    assert isinstance(case, MultimodalWholeOutputCase)
    assert case.inputs == (media,)
    assert case.modalities == ("text", "video")
    assert MultimodalWholeOutputCase.from_dict(case.as_dict()) == case


def test_tampered_parent_outputs_fail_cache_authentication(
    probe_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    lines = Path(probe_suite["outputs"]).read_bytes().splitlines()

    swapped = json.loads(lines[0].decode("utf-8"))
    swapped["raw_output"] = "none"
    forged = tmp_path / "forged.jsonl"
    forged.write_bytes(
        b"\n".join(
            [json.dumps(swapped, sort_keys=True).encode("utf-8"), *lines[1:]]
        )
        + b"\n"
    )
    with pytest.raises(DynamicVideoFoldError, match="authentication"):
        load_video_parent_behavior_cache(probe_suite["payload"]["manifest"], forged)

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_bytes(b"\n".join(lines[:-1]) + b"\n")
    with pytest.raises(DynamicVideoFoldError, match="count differs"):
        load_video_parent_behavior_cache(
            probe_suite["payload"]["manifest"],
            truncated,
        )

    unterminated = tmp_path / "unterminated.jsonl"
    unterminated.write_bytes(b"\n".join(lines))
    with pytest.raises(DynamicVideoFoldError, match="truncated"):
        load_video_parent_behavior_cache(
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
    with pytest.raises(DynamicVideoFoldError, match="authentication"):
        load_video_parent_behavior_cache(probe_suite["payload"]["manifest"], lied)


def test_answer_normalization_is_a_closed_set(
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    assert normalize_video_answer("shape_count", "3") == "3"
    assert normalize_video_answer("shape_count", " 3 ") == "3"
    assert normalize_video_answer("shape_count", "4") is None
    assert normalize_video_answer("shape_count", "three") is None
    assert normalize_video_answer("motion_direction", "Left") == "left"
    assert normalize_video_answer("motion_direction", "it moves left") is None
    assert normalize_video_answer("shape_event", "appears.") is None
    assert normalize_video_answer("shape_color", None) is None
    with pytest.raises(DynamicVideoFoldError):
        normalize_video_answer("frame_rate", "left")
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
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    gate = NativeVideoBehaviorGate(
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
    assert report["modality"] == "video"
    assert report["parent_forward_count"] == 0
    assert report["failures"] == []
    assert report["exact_parent_output_is_diagnostic_only"] is True


def test_gate_rejects_a_heldout_regression(
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    victim = behavior_cache.heldout[0]
    wrong = next(
        answer
        for answer in CLOSED_ANSWERS[victim.category]
        if answer != victim.expected
    )
    gate = NativeVideoBehaviorGate(
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
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    def prose_runner(model: Any, record: Any) -> str:
        model(torch.zeros(1, HIDDEN))
        return f"I believe the answer is {record.expected}."

    gate = NativeVideoBehaviorGate(behavior_cache, child_runner=prose_runner)
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
    _sealed, probes = authenticate_native_video_probe(manifest)
    blind = probes[0]

    def stubborn_parent(probe: Any) -> str:
        if probe.probe_id == blind.probe_id:
            return "the clip is unclear"
        return probe.answer

    outputs = write_video_parent_output_cache(
        manifest,
        tmp_path / "partial-parent.jsonl",
        parent_runner=stubborn_parent,
    )
    cache = load_video_parent_behavior_cache(manifest, outputs)
    assert not cache.records[0].parent_correct
    assert not cache.records[0].parent_direct_answer

    # A child that mirrors the parent's failure is still promotable.
    gate = NativeVideoBehaviorGate(
        cache,
        child_runner=_child_runner({blind.probe_id: "the clip is unclear"}),
    )
    assert gate.evaluate_model(_Probe().eval()).accepted

    # A child that improves on the parent is also promotable.
    improving = NativeVideoBehaviorGate(cache, child_runner=_child_runner({}))
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
    _sealed, probes = authenticate_native_video_probe(manifest)
    blind = next(
        probe
        for probe in probes
        if probe.split == "heldout" and probe.category == "shape_event"
    )
    outputs = write_video_parent_output_cache(
        manifest,
        tmp_path / "uncovered-parent.jsonl",
        parent_runner=lambda probe: (
            "no idea" if probe.probe_id == blind.probe_id else probe.answer
        ),
    )
    cache = load_video_parent_behavior_cache(manifest, outputs)
    gate = NativeVideoBehaviorGate(cache, child_runner=_child_runner({}))
    decision = gate.evaluate_model(_Probe().eval())

    assert not decision.accepted
    assert "heldout:shape_event:parent_correct_coverage" in decision.reason
    assert gate.last_report["rows"] == []
    assert dict(decision.metrics)["child_outputs_scored"] == 0

    relaxed = NativeVideoBehaviorGate(
        cache,
        child_runner=_child_runner({}),
        config=VideoBehaviorGateConfig(
            require_parent_correct_task_coverage=False,
            require_split_category_parent_correct_coverage=False,
        ),
    )
    assert relaxed.evaluate_model(_Probe().eval()).accepted


def test_gate_requires_the_declared_inference_entrypoint_to_execute(
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    gate = NativeVideoBehaviorGate(
        behavior_cache,
        child_runner=lambda model, record: record.expected,
    )
    with pytest.raises(DynamicVideoFoldError, match="declared inference"):
        gate.evaluate_model(_Probe().eval())


def test_gate_seals_media_capable_cases_into_the_canonical_manifest(
    behavior_cache: VideoParentBehaviorCache,
) -> None:
    gate = NativeVideoBehaviorGate(behavior_cache, child_runner=_child_runner({}))
    binding = resolve_behavior_gate_binding(gate)

    assert binding.baseline_fingerprint == behavior_cache.fingerprint
    assert multimodal_cases_from_manifest(binding.manifest) == behavior_cache.cases()
    coverage = [
        component
        for component in binding.manifest["components"]
        if component["schema"] == "neural-forge-multimodal-case-coverage-v1"
    ]
    assert coverage[0]["input_modalities"] == ["video"]
    assert coverage[0]["media_input_count"] == len(behavior_cache.records)
    assert binding.manifest["inference_execution"]["entrypoint"] == "__call__"
    assert binding.manifest["baseline"]["parent_outputs_sha256"] == (
        behavior_cache.outputs_sha256
    )


# --------------------------------------------------------------------------
# Video-conditioned trace evidence
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


class _VideoParent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Decoder()

    def forward(self, features: Any) -> Any:
        hidden = features
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def _video_graph() -> ModelCircuitGraph:
    video_node = CircuitNode(
        node_id="input.video",
        kind=CircuitKind.MODALITY_INPUT,
        order=0,
        path="video_front_end",
        layer_index=None,
        inputs=(
            CircuitPort(
                "clip",
                PortKind.MODALITY,
                ("batch", "channels", "frames", "height", "width"),
                modality="video",
            ),
        ),
        outputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", HIDDEN),
                modality="video",
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
            "input.video",
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
        model_name="tiny-video-parent",
        family="test_video",
        model_class="tests.TinyVideoParent",
        nodes=(video_node, *layer_nodes),
        edges=tuple(edges),
        interfaces=(
            ModalityInterface(
                interface_id="video.input",
                modality="video",
                direction="input",
                node_id="input.video",
                port_name="clip",
                representation="rgb24_frames",
            ),
        ),
        logical_parameter_count=parameter_count,
        unique_parameter_count=parameter_count,
        unique_parameter_bytes=parameter_count * 4,
        config_fingerprint=canonical_sha256({"model": "tiny-video-parent"}),
    )


def _trace_runner(model: Any, record: Any) -> Any:
    frames, _fps = read_rgb24_video(record.video_path)
    values = np.asarray(frames, dtype=np.float32).reshape(SEQUENCE, HIDDEN, -1)
    features = torch.from_numpy(
        (values.mean(axis=2) / 255.0).reshape(1, SEQUENCE, HIDDEN).copy()
    )
    return model(features)


def test_trace_cache_captures_calibration_only_and_seals_statistics(
    behavior_cache: VideoParentBehaviorCache,
    tmp_path: Path,
) -> None:
    parent = _VideoParent().eval()
    graph = _video_graph()
    state_before = {
        name: value.detach().clone() for name, value in parent.state_dict().items()
    }

    cache = build_native_video_fold_trace_cache(
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
    reopened = VideoTraceStatisticsCache.open(
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
    assert manifest["modality"] == "video"
    assert manifest["fold_selection_included"] is False
    assert manifest["evidence_excluded_splits"] == ["heldout"]
    assert manifest["behavior_cache_fingerprint"] == behavior_cache.fingerprint
    assert manifest["statistics_byte_order"] == "little"


def test_trace_cache_reload_fails_closed_on_tamper(
    behavior_cache: VideoParentBehaviorCache,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "trace-cache"
    build_native_video_fold_trace_cache(
        _VideoParent().eval(),
        _video_graph(),
        behavior_cache,
        trace_runner=_trace_runner,
        destination=destination,
        sketch_dim=8,
    )

    blob = destination / "statistics.bin"
    payload = bytearray(blob.read_bytes())
    payload[0] ^= 0xFF
    blob.write_bytes(bytes(payload))
    with pytest.raises(DynamicVideoFoldError, match="statistics file"):
        VideoTraceStatisticsCache.open(destination, behavior_cache=behavior_cache)

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["sketch_dim"] = 16
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(DynamicVideoFoldError, match="seal is invalid"):
        VideoTraceStatisticsCache.open(destination, behavior_cache=behavior_cache)


def test_trace_capture_rejects_a_depth_mismatch_and_a_dirty_parent(
    behavior_cache: VideoParentBehaviorCache,
    tmp_path: Path,
) -> None:
    parent = _VideoParent().eval()
    graph = _video_graph()

    with pytest.raises(DynamicVideoFoldError, match="decoder depth"):
        build_native_video_fold_trace_cache(
            parent,
            graph,
            behavior_cache,
            trace_runner=_trace_runner,
            destination=tmp_path / "depth-mismatch",
            layer_resolver=lambda model: tuple(model.model.layers)[:1],
            sketch_dim=8,
        )

    dirty = _VideoParent()
    output = dirty(torch.zeros(1, SEQUENCE, HIDDEN) + 0.5).sum()
    output.backward()
    with pytest.raises(DynamicVideoFoldError, match="zero-gradient"):
        build_native_video_fold_trace_cache(
            dirty,
            graph,
            behavior_cache,
            trace_runner=_trace_runner,
            destination=tmp_path / "dirty-parent",
            sketch_dim=8,
        )


def test_video_lane_status_declares_its_open_seams() -> None:
    status = video_lane_status()
    assert status["modality"] == "video"
    assert status["parent_forwards_during_gating"] == 0
    assert "trace_runner" in status["model_agnostic_seams"]
    # The inspector, selection rule, and fold method land with this wave; what
    # remains open is a production parent and output-side review.
    assert any("inspector" in item for item in status["provided"])
    assert any("parent model choice" in item for item in status["not_provided"])
    assert any("MODALITY_OUTPUT" in item for item in status["not_provided"])
    assert BEHAVIOR_SCHEMA.endswith("-v1")
