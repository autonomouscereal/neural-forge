from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_closed_loop_fold import (
    CapturePoint,
    ChildClosedLoopCompiler,
    ChildTrajectoryCapture,
    ClosedLoopCompilerConfig,
    ClosedLoopFoldError,
    CompiledFoldStage,
    CompileStage,
    InstallReceipt,
    ParentEndpointBatch,
    TrajectoryRecord,
    TransactionReceipt,
    TERMINAL_ONLY_RESTART_RESUME_POLICY_EXACT_ACCEPTED_LAYER_V1,
    ValidationDecision,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.hf_lfm2_closed_loop_restart_capsule import (  # noqa: E402
    CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256,
    Lfm2ClosedLoopRestartCapsuleBindings,
    Lfm2ClosedLoopRestartCapsuleError,
    Lfm2ClosedLoopRestartCapsuleWriter,
    Lfm2ClosedLoopTerminalOnlyRestartWriter,
    Lfm2TerminalOnlyRestartResumePolicy,
    authenticate_lfm2_closed_loop_restart_capsule,
    discover_lfm2_closed_loop_restart_capsules,
    latest_lfm2_closed_loop_restart_capsule,
    load_lfm2_closed_loop_restart_capsule,
    load_lfm2_terminal_only_restart_resume_policy,
    portable_child_state_sha256,
    portable_child_topology_sha256,
)


# ---------------------------------------------------------------------------
# Tiny tied child + fake adapter
# ---------------------------------------------------------------------------


class _TinyChild(torch.nn.Module):
    """A tiny child with one tied table so alias groups are exercised."""

    def __init__(self, *, seed: int = 4113) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embed = torch.nn.Embedding(6, 4)
            self.block = torch.nn.Linear(4, 4, bias=True)
            self.head = torch.nn.Linear(4, 6, bias=False)
        self.head.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(4, dtype=torch.float32))
        self.requires_grad_(False)
        self.eval()


class _FakeLayer:
    def __init__(self, index: int) -> None:
        self.index = index


class _FakeAdapter:
    """Minimal adapter surface the capsule writer authenticates."""

    def __init__(
        self,
        child_model: torch.nn.Module,
        *,
        layer_count: int = 3,
        body_reclosure_mode: str = "disabled",
    ) -> None:
        self.child_model = child_model
        self.child_layers = tuple(_FakeLayer(index) for index in range(layer_count))
        self._selected_layers = tuple(range(layer_count))
        self.body_reclosure_mode = body_reclosure_mode

    def child_state_fingerprint(self) -> str:
        # Mirrors the real adapter: process-local Parameter object identity is
        # mixed in, so this digest is deliberately not portable.
        return canonical_sha256(
            {
                "process_local": [
                    [name, id(value)]
                    for name, value in sorted(
                        self.child_model.named_parameters(
                            remove_duplicate=False
                        ),
                        key=lambda item: item[0],
                    )
                ],
                "portable": portable_child_state_sha256(self.child_model),
            }
        )

    def child_topology_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "process_local_topology": portable_child_topology_sha256(
                    self.child_model
                )
            }
        )

    def restart_capsule_identity_bindings(self):
        body = {
            "schema": (
                "neural-forge-lfm2-closed-loop-restart-capsule-bindings-v1"
            ),
            "adapter_schema": "fake-adapter-v1",
            "coordinate_map_sha256": canonical_sha256({"map": "fake"}),
            "child_basis_sha256": canonical_sha256({"map": "fake"}),
            "child_input_vocabulary_sha256": canonical_sha256({"vocab": 6}),
            "parent_cache_key": canonical_sha256({"cache": "fake"}),
            "parent_identity_sha256": canonical_sha256({"parent": "fake"}),
            "source_graph_fingerprint": None,
            "tokenizer_bridge_artifact_sha256": None,
            "identity_tokenizer_evidence_sha256": None,
            "parent_residual_operator_manifest_sha256": None,
            "body_reclosure_mode": self.body_reclosure_mode,
            "body_coordinate_completion_recipe": "bridge-procrustes-isometric",
            "replay_capture_mode": "prefix_by_prefix",
            "child_layer_count": len(self.child_layers),
            "selected_layer_indices": list(self._selected_layers),
        }
        return {**body, "adapter_identity_sha256": canonical_sha256(body)}

    def assert_mid_body_resume_supported(
        self,
        *,
        completed_layer_indices,
        remaining_layer_indices,
        body_reclosure_enabled,
    ):
        if body_reclosure_enabled and self.body_reclosure_mode != "disabled":
            raise Lfm2ClosedLoopRestartCapsuleError(
                "moving_gauge_v3 body reclosure requires a full installed stack."
            )
        body = {
            "schema": "fake-mid-body-resume-admission-v1",
            "completed_layer_indices": list(completed_layer_indices),
            "remaining_layer_indices": list(remaining_layer_indices),
            "body_reclosure_enabled": bool(body_reclosure_enabled),
        }
        return {**body, "admission_sha256": canonical_sha256(body)}


class _FakeProgressSink:
    """Duck-type of the dense fold hash-chained progress sink."""

    def __init__(self, *, binding: str = "dense-fold-progress") -> None:
        self.binding_sha256 = canonical_sha256({"binding": binding})
        self._sequence = 0
        self._previous_sha256 = CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256

    def emit(self) -> None:
        self._previous_sha256 = canonical_sha256(
            {
                "binding": self.binding_sha256,
                "sequence": self._sequence,
                "previous": self._previous_sha256,
            }
        )
        self._sequence += 1


def _write_canonical(path: Path, body: dict) -> None:
    """Rewrite one capsule document as canonical JSON, bypassing every seal."""

    path.write_bytes(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _split_digest(label: str = "restart-capsule-split") -> str:
    return canonical_sha256({"split": label})


def _decision_digest(layer_index: int) -> str:
    return canonical_sha256({"validation": layer_index})


def _writer(
    tmp_path: Path,
    adapter: _FakeAdapter,
    *,
    resume=None,
    rotation_keep: int = 2,
    directory: str = "capsules",
    split_label: str = "restart-capsule-split",
    fold_progress_sink=None,
) -> Lfm2ClosedLoopRestartCapsuleWriter:
    root = tmp_path / directory
    root.mkdir(exist_ok=True)
    return Lfm2ClosedLoopRestartCapsuleWriter(
        adapter=adapter,
        directory=root,
        run_id="restart-capsule-test-run",
        child_name="tiny-restart-child",
        trajectory_split_sha256=_split_digest(split_label),
        run_schema="neural-forge-dynamic-child-closed-loop-fold-v3",
        resume=resume,
        rotation_keep=rotation_keep,
        fold_progress_sink=fold_progress_sink,
    )


def _accept(
    writer: Lfm2ClosedLoopRestartCapsuleWriter,
    layer_index: int,
    *,
    layer_count: int = 3,
    split_label: str = "restart-capsule-split",
):
    completed = tuple(range(layer_index + 1))
    remaining = tuple(range(layer_index + 1, layer_count))
    return writer.record_accepted_layer(
        layer_index=layer_index,
        layer_ordinal=layer_index,
        completed_layer_indices=completed,
        remaining_layer_indices=remaining,
        selected_layer_indices=tuple(range(layer_count)),
        validation_decision_sha256=_decision_digest(layer_index),
        child_state_sha256=writer.adapter.child_state_fingerprint(),
        child_topology_sha256=writer.adapter.child_topology_fingerprint(),
        trajectory_split_sha256=_split_digest(split_label),
    )


def _terminal_policy(reference) -> Lfm2TerminalOnlyRestartResumePolicy:
    authentication = authenticate_lfm2_closed_loop_restart_capsule(reference)
    return Lfm2TerminalOnlyRestartResumePolicy(
        capsule_reference=reference,
        expected_child_topology_sha256=authentication.manifest[
            "child_topology_sha256"
        ],
        expected_adapter_identity_sha256=(
            authentication.bindings.adapter_identity_sha256
        ),
        expected_run_bindings_sha256=(
            authentication.bindings.bindings_sha256
        ),
        expected_run_id=authentication.bindings.run_id,
    )


def _mutate(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        model.block.weight.add_(value)
        model.embed.weight.add_(value / 2.0)


# ---------------------------------------------------------------------------
# Capsule publication, authentication, resume
# ---------------------------------------------------------------------------


def test_capsule_publishes_and_resumes_the_exact_accepted_child(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    adapter = _FakeAdapter(child)
    writer = _writer(tmp_path, adapter)

    _mutate(child, 0.25)
    _accept(writer, 0)
    _mutate(child, 0.5)
    record = _accept(writer, 1)
    accepted_state = portable_child_state_sha256(child)
    accepted_topology = portable_child_topology_sha256(child)
    accepted_block = child.block.weight.detach().clone()

    # The live child keeps moving after the capsule was sealed.
    _mutate(child, 1.5)
    assert portable_child_state_sha256(child) != accepted_state

    assert record["record_sha256"] == canonical_sha256(
        {name: value for name, value in record.items() if name != "record_sha256"}
    )
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    assert reference is not None
    assert reference.layer_index == 1
    assert reference.capsule_sequence == 1

    fresh = _TinyChild()
    assert portable_child_state_sha256(fresh) != accepted_state
    resume = load_lfm2_closed_loop_restart_capsule(
        reference,
        child_model=fresh,
        expected_bindings=writer.bindings,
        expected_adapter_identity=adapter.restart_capsule_identity_bindings(),
    )

    assert resume.completed_layer_indices == (0, 1)
    assert resume.resume_layer_index == 2
    assert resume.child_state_sha256 == accepted_state
    assert resume.child_topology_sha256 == accepted_topology
    assert portable_child_state_sha256(fresh) == accepted_state
    assert resume.validation_decision_sha256 == _decision_digest(1)
    # The tied table survived as one shared storage, not two copies.
    assert fresh.head.weight is fresh.embed.weight
    assert torch.equal(fresh.block.weight, accepted_block)
    payload = resume.as_dict()
    assert payload["resume_sha256"] == canonical_sha256(
        {name: value for name, value in payload.items() if name != "resume_sha256"}
    )
    assert payload["skipped_layers_are_not_replayed"] is True


def test_capsule_manifest_is_non_promotable_and_content_addressed(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    authentication = authenticate_lfm2_closed_loop_restart_capsule(reference)
    manifest = authentication.manifest

    assert manifest["status"] == "quarantined-mid-body-restart-only"
    assert manifest["promotion_authorized"] is False
    assert manifest["accepted_artifact"] is False
    assert manifest["behavior_capability_proven"] is False
    assert manifest["training_performed"] is False
    assert manifest["optimizer_steps"] == 0
    assert manifest["directory_name"] == reference.path.name
    assert reference.capsule_key in reference.path.name
    assert manifest["previous_capsule_chain_sha256"] == (
        CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256
    )


def test_capsule_resume_fails_closed_on_a_flipped_child_state_byte(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)

    state_path = reference.path / "child-state.safetensors"
    raw = bytearray(state_path.read_bytes())
    # Flip one payload byte well past the safetensors header.
    raw[-1] ^= 0x01
    state_path.write_bytes(bytes(raw))

    with pytest.raises(Exception) as failure:
        load_lfm2_closed_loop_restart_capsule(
            reference,
            child_model=_TinyChild(),
        )
    assert "child-state.safetensors" in str(failure.value)

    with pytest.raises(Exception):
        authenticate_lfm2_closed_loop_restart_capsule(reference)
    with pytest.raises(Exception):
        discover_lfm2_closed_loop_restart_capsules(writer.directory)


def test_capsule_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)

    manifest_path = reference.path / "capsule-manifest.json"
    body = json.loads(manifest_path.read_bytes().decode("ascii"))
    body["completed_layer_indices"] = [0, 1]
    _write_canonical(manifest_path, body)

    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="manifest seal is invalid",
    ):
        authenticate_lfm2_closed_loop_restart_capsule(reference)


def test_capsule_rotation_keeps_only_the_newest_capsules(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child, layer_count=4), rotation_keep=2)

    keys = []
    for layer_index in range(4):
        _mutate(child, 0.1)
        record = _accept(writer, layer_index, layer_count=4)
        keys.append(record["capsule"]["capsule_key"])
        retained = tuple(record["retained_capsule_keys"])
        assert len(retained) <= 2

    references = discover_lfm2_closed_loop_restart_capsules(writer.directory)
    assert [value.capsule_key for value in references] == keys[-2:]
    assert [value.capsule_sequence for value in references] == [2, 3]
    assert [value.layer_index for value in references] == [2, 3]
    assert len(list(Path(writer.directory).iterdir())) == 2


def test_capsule_chain_gap_fails_closed(tmp_path: Path) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child), rotation_keep=3)
    for layer_index in range(3):
        _mutate(child, 0.1)
        _accept(writer, layer_index)

    references = discover_lfm2_closed_loop_restart_capsules(writer.directory)
    assert len(references) == 3

    middle = references[1].path
    for entry in middle.iterdir():
        entry.unlink()
    middle.rmdir()

    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="discontiguous",
    ):
        discover_lfm2_closed_loop_restart_capsules(writer.directory)


def test_capsule_chain_records_and_advances_the_fold_progress_chain(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    sink = _FakeProgressSink()
    writer = _writer(
        tmp_path,
        _FakeAdapter(child),
        rotation_keep=3,
        fold_progress_sink=sink,
    )
    for layer_index in range(3):
        sink.emit()
        _mutate(child, 0.1)
        _accept(writer, layer_index)

    references = discover_lfm2_closed_loop_restart_capsules(writer.directory)
    progress = [
        authenticate_lfm2_closed_loop_restart_capsule(reference).manifest[
            "fold_progress"
        ]
        for reference in references
    ]
    assert [value["available"] for value in progress] == [True, True, True]
    assert [value["sequence"] for value in progress] == [1, 2, 3]
    assert len({value["binding_sha256"] for value in progress}) == 1
    assert progress[0]["previous_event_sha256"] != (
        progress[1]["previous_event_sha256"]
    )
    assert [value["process_boundary"] for value in progress] == [
        False,
        False,
        False,
    ]


def test_capsule_chain_requires_a_declared_process_boundary(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    first_sink = _FakeProgressSink(binding="first-process")
    writer = _writer(
        tmp_path,
        _FakeAdapter(child),
        rotation_keep=4,
        fold_progress_sink=first_sink,
    )
    first_sink.emit()
    _accept(writer, 0)

    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    fresh = _TinyChild()
    resume = load_lfm2_closed_loop_restart_capsule(
        reference,
        child_model=fresh,
    )
    second_sink = _FakeProgressSink(binding="second-process")
    second_sink.emit()
    resumed_writer = _writer(
        tmp_path,
        _FakeAdapter(fresh),
        resume=resume,
        rotation_keep=4,
        fold_progress_sink=second_sink,
    )
    _accept(resumed_writer, 1)

    references = discover_lfm2_closed_loop_restart_capsules(
        resumed_writer.directory
    )
    manifests = [
        authenticate_lfm2_closed_loop_restart_capsule(value).manifest
        for value in references
    ]
    assert manifests[1]["fold_progress"]["process_boundary"] is True
    assert manifests[1]["previous_capsule_manifest_sha256"] == (
        manifests[0]["manifest_sha256"]
    )
    assert manifests[1]["resumed_from_capsule_manifest_sha256"] == (
        manifests[0]["manifest_sha256"]
    )

    # Forging away the process boundary breaks the manifest seal, and a
    # correctly resealed forgery still breaks the capsule content key.
    manifest_path = references[1].path / "capsule-manifest.json"
    body = json.loads(manifest_path.read_bytes().decode("ascii"))
    body["fold_progress"]["process_boundary"] = False
    body.pop("manifest_sha256")
    body["manifest_sha256"] = canonical_sha256(body)
    _write_canonical(manifest_path, body)
    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="content key is invalid",
    ):
        discover_lfm2_closed_loop_restart_capsules(resumed_writer.directory)


def test_capsule_load_refuses_a_foreign_run_identity(tmp_path: Path) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)

    foreign = Lfm2ClosedLoopRestartCapsuleBindings(
        run_id="another-run",
        child_name="tiny-restart-child",
        child_layer_count=3,
        selected_layer_indices=(0, 1, 2),
        adapter_identity_sha256=writer.bindings.adapter_identity_sha256,
        trajectory_split_sha256=writer.bindings.trajectory_split_sha256,
        run_schema=writer.bindings.run_schema,
    )
    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="different run identity",
    ):
        load_lfm2_closed_loop_restart_capsule(
            reference,
            child_model=_TinyChild(),
            expected_bindings=foreign,
        )


def test_capsule_load_refuses_a_structurally_different_child(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)

    untied = _TinyChild()
    untied.head.weight = torch.nn.Parameter(
        untied.embed.weight.detach().clone()
    )
    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="alias grouping",
    ):
        load_lfm2_closed_loop_restart_capsule(
            reference,
            child_model=untied,
        )


def test_writer_refuses_a_resume_that_body_reclosure_cannot_support(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    adapter = _FakeAdapter(child, body_reclosure_mode="moving_gauge_v3")
    writer = _writer(tmp_path, adapter)
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)

    fresh = _TinyChild()
    resume = load_lfm2_closed_loop_restart_capsule(
        reference,
        child_model=fresh,
    )
    resumed_adapter = _FakeAdapter(fresh, body_reclosure_mode="moving_gauge_v3")
    resumed_writer = _writer(tmp_path, resumed_adapter, resume=resume)

    admission_kwargs = {
        "completed_layer_indices": (0,),
        "remaining_layer_indices": (1, 2),
        "selected_layer_indices": (0, 1, 2),
        "trajectory_split_sha256": _split_digest(),
        "child_state_sha256": resumed_adapter.child_state_fingerprint(),
        "child_topology_sha256": resumed_adapter.child_topology_fingerprint(),
    }
    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="full installed stack",
    ):
        resumed_writer.assert_resume_supported(
            body_reclosure_enabled=True,
            **admission_kwargs,
        )

    admission = resumed_writer.assert_resume_supported(
        body_reclosure_enabled=False,
        **admission_kwargs,
    )
    assert admission["resume_layer_index"] == 1
    assert admission["skipped_layers_are_not_replayed"] is True
    assert admission["admission_sha256"] == canonical_sha256(
        {
            name: value
            for name, value in admission.items()
            if name != "admission_sha256"
        }
    )


def test_exact_terminal_only_policy_authenticates_body_complete_capsule(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    adapter = _FakeAdapter(child, body_reclosure_mode="moving_gauge_v3")
    writer = _writer(tmp_path, adapter, rotation_keep=3)
    for layer_index in range(3):
        _mutate(child, 0.1 + layer_index)
        _accept(writer, layer_index)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    assert reference is not None
    policy = _terminal_policy(reference)
    policy_path = tmp_path / "terminal-only-policy.json"
    _write_canonical(policy_path, policy.as_dict())

    loaded_policy = load_lfm2_terminal_only_restart_resume_policy(policy_path)
    fresh = _TinyChild()
    resume = load_lfm2_closed_loop_restart_capsule(
        loaded_policy.capsule_reference,
        child_model=fresh,
    )
    resumed_adapter = _FakeAdapter(
        fresh,
        body_reclosure_mode="moving_gauge_v3",
    )
    terminal_writer = Lfm2ClosedLoopTerminalOnlyRestartWriter(
        adapter=resumed_adapter,
        resume=resume,
        policy=loaded_policy,
        run_id="restart-capsule-test-run",
        child_name="tiny-restart-child",
        trajectory_split_sha256=_split_digest(),
        run_schema="neural-forge-dynamic-child-closed-loop-fold-v3",
    )

    admission = terminal_writer.assert_terminal_only_resume_supported(
        completed_layer_indices=(0, 1, 2),
        remaining_layer_indices=(),
        selected_layer_indices=(0, 1, 2),
        available_layer_indices=(0, 1, 2),
        body_reclosure_enabled=False,
        final_normalization_enabled=True,
        output_interface_enabled=True,
        trajectory_split_sha256=_split_digest(),
        child_state_sha256=resumed_adapter.child_state_fingerprint(),
        child_topology_sha256=resumed_adapter.child_topology_fingerprint(),
    )

    assert admission["phase"] == (
        "accepted_layer_post_validation_pre_body_reclosure"
    )
    assert admission["remaining_layer_indices"] == []
    assert admission["configured_adapter_body_reclosure_mode"] == (
        "moving_gauge_v3"
    )
    assert admission["effective_compiler_body_reclosure_mode"] == "disabled"
    assert admission["latest_capsule_discovery_used"] is False
    assert admission["legacy_capsule_writes_enabled"] is False
    writer_report = terminal_writer.as_dict()
    assert writer_report["published_capsule_count"] == 0
    assert writer_report["legacy_capsule_directory"] is None
    assert writer_report["terminal_or_post_output_capsule_writes_enabled"] is False


def test_writer_refuses_a_resume_whose_split_changed(tmp_path: Path) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    fresh = _TinyChild()
    resume = load_lfm2_closed_loop_restart_capsule(
        reference,
        child_model=fresh,
    )
    resumed_adapter = _FakeAdapter(fresh)
    resumed_writer = _writer(tmp_path, resumed_adapter, resume=resume)

    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="different deterministic split",
    ):
        resumed_writer.assert_resume_supported(
            completed_layer_indices=(0,),
            remaining_layer_indices=(1, 2),
            selected_layer_indices=(0, 1, 2),
            body_reclosure_enabled=False,
            trajectory_split_sha256=_split_digest("another-split"),
            child_state_sha256=resumed_adapter.child_state_fingerprint(),
            child_topology_sha256=(
                resumed_adapter.child_topology_fingerprint()
            ),
        )


def test_writer_refuses_a_resume_whose_live_child_moved(
    tmp_path: Path,
) -> None:
    child = _TinyChild()
    writer = _writer(tmp_path, _FakeAdapter(child))
    _accept(writer, 0)
    reference = latest_lfm2_closed_loop_restart_capsule(writer.directory)
    fresh = _TinyChild()
    resume = load_lfm2_closed_loop_restart_capsule(
        reference,
        child_model=fresh,
    )
    resumed_adapter = _FakeAdapter(fresh)
    resumed_writer = _writer(tmp_path, resumed_adapter, resume=resume)
    _mutate(fresh, 0.75)

    with pytest.raises(
        Lfm2ClosedLoopRestartCapsuleError,
        match="exact child the capsule restored",
    ):
        resumed_writer.assert_resume_supported(
            completed_layer_indices=(0,),
            remaining_layer_indices=(1, 2),
            selected_layer_indices=(0, 1, 2),
            body_reclosure_enabled=False,
            trajectory_split_sha256=_split_digest(),
            child_state_sha256=resumed_adapter.child_state_fingerprint(),
            child_topology_sha256=(
                resumed_adapter.child_topology_fingerprint()
            ),
        )


# ---------------------------------------------------------------------------
# Orchestrator integration on a scripted adapter
# ---------------------------------------------------------------------------


def _records() -> tuple[TrajectoryRecord, ...]:
    return tuple(
        TrajectoryRecord(
            trajectory_id=f"trajectory-{trajectory}",
            path_id=path,
            record_sha256=canonical_sha256(
                {"trajectory": trajectory, "path": path}
            ),
            payload={"trajectory": trajectory, "path": path},
        )
        for trajectory in range(5)
        for path in ("audio", "text")
    )


class _ScriptedTransaction:
    def __init__(self, adapter: "_ScriptedAdapter") -> None:
        self.adapter = adapter
        self.snapshot = copy.deepcopy(adapter.state)
        self.state_sha256_before = adapter.child_state_fingerprint()
        self.topology_sha256_before = adapter.child_topology_fingerprint()
        self.active = True

    def commit(self) -> TransactionReceipt:
        after = self.adapter.child_state_fingerprint()
        self.active = False
        return TransactionReceipt(
            action="commit",
            accepted=True,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=after,
        )

    def rollback(self) -> TransactionReceipt:
        self.adapter.state = copy.deepcopy(self.snapshot)
        self.active = False
        return TransactionReceipt(
            action="rollback",
            accepted=False,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=self.adapter.child_state_fingerprint(),
        )


class _ScriptedAdapter:
    """Family-neutral scripted adapter for orchestrator-level assertions."""

    def __init__(self, *, merged: bool = False) -> None:
        self.state = {
            "layers": {},
            "body_reclosure": None,
            "final_normalization": None,
            "output_interface": None,
        }
        self.topology = {"layers": [0, 1, 2]}
        self.log: list[str] = []
        self.replays = 0
        self.merged = bool(merged)
        self.observed = {
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }

    # -- adapter protocol ---------------------------------------------

    def child_state_fingerprint(self) -> str:
        return canonical_sha256(self.state)

    def child_topology_fingerprint(self) -> str:
        return canonical_sha256(self.topology)

    def audit_counters(self):
        return dict(self.observed)

    def begin_child_transaction(self, _audit):
        return _ScriptedTransaction(self)

    def layer_indices(self, _audit):
        return (0, 1, 2)

    def parent_trajectory_records(self, _audit):
        return _records()

    def bridge_parent_endpoints(self, scope, records, _audit):
        keys = tuple(sorted(record.key for record in records))
        return ParentEndpointBatch(
            scope=scope,
            record_keys=keys,
            batch_sha256=canonical_sha256(
                {"scope": scope.scope_id, "keys": [list(k) for k in keys]}
            ),
            payload={"keys": keys},
        )

    def _capture(self, scope, point, records, state):
        keys = tuple(sorted(record.key for record in records))
        self.log.append(f"{scope.scope_id}:capture:{point.value}")
        return ChildTrajectoryCapture(
            scope=scope,
            point=point,
            record_keys=keys,
            capture_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "point": point.value,
                    "keys": [list(k) for k in keys],
                    "child_state": state,
                }
            ),
            payload={"child_state": state},
        )

    def replay_and_capture_child(self, scope, point, records, _audit):
        self.replays += 1
        return self._capture(scope, point, records, copy.deepcopy(self.state))

    def supports_merged_child_captures(self) -> bool:
        return self.merged

    def replay_and_capture_child_merged(self, requests, _audit):
        self.replays += 1
        state = copy.deepcopy(self.state)
        self.log.append(
            "merged:"
            + "|".join(
                f"{scope.scope_id}:{point.value}"
                for scope, point, _records in requests
            )
        )
        return tuple(
            self._capture(scope, point, records, state)
            for scope, point, records in requests
        )

    def _compiled(self, scope, stage, records, capture):
        keys = tuple(sorted(record.key for record in records))
        return CompiledFoldStage(
            scope=scope,
            stage=stage,
            fit_record_keys=keys,
            artifact_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "stage": stage.value,
                    "keys": [list(k) for k in keys],
                    "capture": capture.capture_sha256,
                }
            ),
            payload={"capture": capture.capture_sha256},
        )

    def compile_producer(self, scope, records, _endpoints, capture, _audit):
        assert capture.point is CapturePoint.PRE_PRODUCER
        assert scope.layer_index not in capture.payload["child_state"]["layers"]
        self.log.append(f"{scope.scope_id}:compile:producer")
        return self._compiled(scope, CompileStage.PRODUCER, records, capture)

    def fit_consumer(self, scope, records, _endpoints, capture, _audit):
        assert capture.point is CapturePoint.POST_PRODUCER
        self.log.append(f"{scope.scope_id}:compile:consumer")
        return self._compiled(scope, CompileStage.CONSUMER, records, capture)

    def compile_downstream(self, scope, records, _endpoints, capture, _audit):
        assert capture.point is CapturePoint.POST_CONSUMER
        self.log.append(f"{scope.scope_id}:compile:downstream")
        return self._compiled(scope, CompileStage.DOWNSTREAM, records, capture)

    def compile_final_normalization(
        self, scope, records, _endpoints, capture, _audit
    ):
        return self._compiled(
            scope, CompileStage.FINAL_NORMALIZATION, records, capture
        )

    def compile_output_interface(
        self, scope, records, _endpoints, capture, _audit
    ):
        return self._compiled(
            scope, CompileStage.OUTPUT_INTERFACE, records, capture
        )

    def _install(self, scope, compiled):
        before = self.child_state_fingerprint()
        if scope.kind == "layer":
            layer = self.state["layers"].setdefault(scope.layer_index, {})
            layer[compiled.stage.value] = compiled.artifact_sha256
        else:
            self.state[scope.kind] = compiled.artifact_sha256
        self.log.append(f"{scope.scope_id}:install:{compiled.stage.value}")
        return InstallReceipt(
            scope=scope,
            stage=compiled.stage,
            artifact_sha256=compiled.artifact_sha256,
            accepted=True,
            state_sha256_before=before,
            state_sha256_after=self.child_state_fingerprint(),
            topology_preserved=True,
        )

    def install_producer(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def install_consumer(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def install_downstream(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def install_final_normalization(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def install_output_interface(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def _decision(self, scope, records, reason):
        return ValidationDecision(
            scope=scope,
            accepted=True,
            validation_record_keys=tuple(
                sorted(record.key for record in records)
            ),
            reason=reason,
            metrics={"trajectory_score": 0.95},
        )

    def validate_layer(self, scope, records, _endpoints, capture, _audit):
        assert capture.point is CapturePoint.POST_DOWNSTREAM
        self.log.append(f"{scope.scope_id}:validate")
        return self._decision(scope, records, "layer accepted")

    def validate_final_normalization(
        self, scope, records, _endpoints, capture, _audit
    ):
        return self._decision(scope, records, "final normalization accepted")

    def validate_output_interface(
        self, scope, records, _endpoints, baseline, candidate, _audit
    ):
        return self._decision(scope, records, "terminal interface accepted")


class _ScriptedRestartWriter:
    """Restart-capsule writer double that records the orchestrator contract."""

    def __init__(self, *, completed: tuple[int, ...] = ()) -> None:
        self.completed = tuple(completed)
        self.records: list[dict] = []
        self.admissions: list[dict] = []

    def completed_layer_indices(self):
        return self.completed

    def assert_resume_supported(self, **kwargs):
        admission = {
            "schema": "scripted-resume-admission-v1",
            "completed_layer_indices": list(
                kwargs["completed_layer_indices"]
            ),
            "remaining_layer_indices": list(
                kwargs["remaining_layer_indices"]
            ),
            "body_reclosure_enabled": bool(kwargs["body_reclosure_enabled"]),
        }
        self.admissions.append(admission)
        return admission

    def assert_terminal_only_resume_supported(self, **kwargs):
        admission = {
            "schema": "scripted-terminal-only-admission-v1",
            "completed_layer_indices": list(
                kwargs["completed_layer_indices"]
            ),
            "remaining_layer_indices": list(
                kwargs["remaining_layer_indices"]
            ),
            "selected_layer_indices": list(kwargs["selected_layer_indices"]),
            "available_layer_indices": list(
                kwargs["available_layer_indices"]
            ),
            "body_reclosure_enabled": kwargs["body_reclosure_enabled"],
            "final_normalization_enabled": kwargs[
                "final_normalization_enabled"
            ],
            "output_interface_enabled": kwargs["output_interface_enabled"],
        }
        self.admissions.append(admission)
        return admission

    def record_accepted_layer(self, **kwargs):
        record = {
            "schema": "scripted-capsule-record-v1",
            "layer_index": kwargs["layer_index"],
            "completed_layer_indices": list(
                kwargs["completed_layer_indices"]
            ),
            "remaining_layer_indices": list(
                kwargs["remaining_layer_indices"]
            ),
            "validation_decision_sha256": kwargs["validation_decision_sha256"],
            "child_state_sha256": kwargs["child_state_sha256"],
        }
        self.completed = tuple(kwargs["completed_layer_indices"])
        self.records.append(record)
        return record

    def as_dict(self):
        return {
            "schema": "scripted-capsule-writer-v1",
            "completed_layer_indices": list(self.completed),
            "published": len(self.records),
        }


def _config(**overrides) -> ClosedLoopCompilerConfig:
    values = {
        "validation_fraction": 0.4,
        "split_seed": "restart-capsule-orchestrator",
        "fold_output_interface": False,
    }
    values.update(overrides)
    return ClosedLoopCompilerConfig(**values)


def test_compiler_is_unaffected_when_no_restart_capsule_is_configured() -> None:
    adapter = _ScriptedAdapter()
    report = ChildClosedLoopCompiler(adapter, _config()).run()

    assert report["restart_capsule_enabled"] is False
    assert report["restart_capsule"]["mode"] == "explicit_no_restart_persistence"
    assert report["restart_capsule"]["capsules_written"] == 0
    assert report["restart_capsule_resumed_layers"] == []
    assert report["layers_executed_in_this_process"] == [0, 1, 2]
    assert report["merge_adjacent_layer_captures_enabled"] is False
    assert adapter.replays == 14
    disabled = dict(report["restart_capsule"])
    assert disabled.pop("report_sha256") == canonical_sha256(disabled)


def test_compiler_writes_one_capsule_after_every_accepted_layer() -> None:
    adapter = _ScriptedAdapter()
    writer = _ScriptedRestartWriter()

    report = ChildClosedLoopCompiler(
        adapter,
        _config(),
        restart_capsule_config=writer,
    ).run()

    assert report["restart_capsule_enabled"] is True
    assert report["restart_capsule"]["capsules_written"] == 3
    assert [value["layer_index"] for value in writer.records] == [0, 1, 2]
    assert [
        value["completed_layer_indices"] for value in writer.records
    ] == [[0], [0, 1], [0, 1, 2]]
    assert [
        value["remaining_layer_indices"] for value in writer.records
    ] == [[1, 2], [2], []]
    assert writer.admissions == []
    for record, layer_report in zip(
        writer.records, report["layer_reports"], strict=True
    ):
        assert record["validation_decision_sha256"] == (
            layer_report["validation"]["decision_sha256"]
        )
    # Every capsule was sealed against the state after that layer installed.
    assert len({value["child_state_sha256"] for value in writer.records}) == 3


def test_compiler_skips_completed_layers_on_a_restart_resume() -> None:
    adapter = _ScriptedAdapter()
    # A prior process accepted and installed layers 0 and 1.
    adapter.state["layers"][0] = {
        "producer": "a" * 64,
        "consumer": "b" * 64,
        "downstream": "c" * 64,
    }
    adapter.state["layers"][1] = {
        "producer": "d" * 64,
        "consumer": "e" * 64,
        "downstream": "f" * 64,
    }
    writer = _ScriptedRestartWriter(completed=(0, 1))

    report = ChildClosedLoopCompiler(
        adapter,
        _config(),
        restart_capsule_config=writer,
    ).run()

    assert report["restart_capsule_resumed_layers"] == [0, 1]
    assert report["layers_executed_in_this_process"] == [2]
    assert len(writer.admissions) == 1
    assert writer.admissions[0]["remaining_layer_indices"] == [2]
    assert [value["layer_index"] for value in writer.records] == [2]

    for scope_id in ("layer:000000", "layer:000001"):
        assert not any(entry.startswith(scope_id) for entry in adapter.log)
    assert any(entry.startswith("layer:000002") for entry in adapter.log)

    skipped = report["layer_reports"][:2]
    for layer_index, layer_report in enumerate(skipped):
        assert layer_report["mode"] == "accepted_layer_restart_capsule_resume"
        assert layer_report["executed_in_this_process"] is False
        assert layer_report["validation"] is None
        assert layer_report["scope"]["layer_index"] == layer_index
        body = dict(layer_report)
        assert body.pop("report_sha256") == canonical_sha256(body)
    assert report["layer_reports"][2]["executed_in_this_process"] is True
    # Two skipped layers remove eight full child replays.
    assert adapter.replays == 6


def test_compiler_rejects_completed_layers_that_are_not_the_prefix() -> None:
    adapter = _ScriptedAdapter()
    writer = _ScriptedRestartWriter(completed=(1,))

    with pytest.raises(ClosedLoopFoldError, match="ascending prefix"):
        ChildClosedLoopCompiler(
            adapter,
            _config(),
            restart_capsule_config=writer,
        ).run()


def test_compiler_rejects_a_resume_that_leaves_no_layer() -> None:
    adapter = _ScriptedAdapter()
    writer = _ScriptedRestartWriter(completed=(0, 1, 2))

    with pytest.raises(ClosedLoopFoldError, match="no layer to execute"):
        ChildClosedLoopCompiler(
            adapter,
            _config(),
            restart_capsule_config=writer,
        ).run()


def test_compiler_explicit_terminal_only_resume_runs_only_terminal_stages() -> None:
    adapter = _ScriptedAdapter()
    adapter.state["layers"] = {
        layer_index: {
            "producer": f"sealed-producer-{layer_index}",
            "consumer": f"sealed-consumer-{layer_index}",
            "downstream": f"sealed-downstream-{layer_index}",
        }
        for layer_index in range(3)
    }
    writer = _ScriptedRestartWriter(completed=(0, 1, 2))
    events: list[tuple[str, dict[str, object]]] = []

    report = ChildClosedLoopCompiler(
        adapter,
        _config(
            fold_body_reclosure=False,
            fold_output_interface=True,
            terminal_only_restart_resume_policy=(
                TERMINAL_ONLY_RESTART_RESUME_POLICY_EXACT_ACCEPTED_LAYER_V1
            ),
        ),
        progress_sink=(
            lambda stage, metadata: events.append((stage, dict(metadata)))
        ),
        restart_capsule_config=writer,
    ).run()

    assert report["layers_executed_in_this_process"] == []
    assert writer.records == []
    assert writer.admissions[0]["remaining_layer_indices"] == []
    assert report["body_reclosure_fold_enabled"] is False
    assert report["output_interface_fold_enabled"] is True
    receipt = report["terminal_only_restart_resume"]
    assert receipt["completed_equals_selected_equals_available"] is True
    assert receipt["body_layer_execution_count"] == 0
    assert receipt["body_reclosure_executed"] is False
    assert receipt["final_normalization_executed"] is True
    assert receipt["output_interface_executed"] is True
    assert receipt["legacy_restart_capsule_writes"] == 0
    body = dict(receipt)
    assert body.pop("receipt_sha256") == canonical_sha256(body)
    stages = [stage for stage, _metadata in events]
    assert "child_closed_loop_terminal_only_resume_admitted" in stages
    assert "child_closed_loop_layer_started" not in stages
    assert "child_closed_loop_body_reclosure_started" not in stages
    assert "child_closed_loop_final_normalization_completed" in stages
    assert "child_closed_loop_output_interface_completed" in stages


@pytest.mark.parametrize(
    ("fold_body_reclosure", "fold_output_interface", "match"),
    (
        (True, True, "fold_body_reclosure=False"),
        (False, False, "fold_output_interface=True"),
    ),
)
def test_compiler_terminal_only_resume_rejects_incomplete_terminal_policy(
    fold_body_reclosure: bool,
    fold_output_interface: bool,
    match: str,
) -> None:
    writer = _ScriptedRestartWriter(completed=(0, 1, 2))
    with pytest.raises(ClosedLoopFoldError, match=match):
        ChildClosedLoopCompiler(
            _ScriptedAdapter(),
            _config(
                fold_body_reclosure=fold_body_reclosure,
                fold_output_interface=fold_output_interface,
                terminal_only_restart_resume_policy=(
                    TERMINAL_ONLY_RESTART_RESUME_POLICY_EXACT_ACCEPTED_LAYER_V1
                ),
            ),
            restart_capsule_config=writer,
        ).run()

    assert writer.admissions == []


def test_compiler_terminal_only_resume_rejects_selected_layer_subset() -> None:
    writer = _ScriptedRestartWriter(completed=(0, 1))
    with pytest.raises(
        ClosedLoopFoldError,
        match="completed == selected == available",
    ):
        ChildClosedLoopCompiler(
            _ScriptedAdapter(),
            _config(
                selected_layers=(0, 1),
                fold_output_interface=True,
                terminal_only_restart_resume_policy=(
                    TERMINAL_ONLY_RESTART_RESUME_POLICY_EXACT_ACCEPTED_LAYER_V1
                ),
            ),
            restart_capsule_config=writer,
        ).run()

    assert writer.admissions == []


def test_compiler_rejects_a_restart_writer_that_breaks_the_protocol() -> None:
    with pytest.raises(TypeError, match="restart capsule writer protocol"):
        ChildClosedLoopCompiler(
            _ScriptedAdapter(),
            _config(),
            restart_capsule_config=object(),
        )


def test_merged_captures_remove_one_full_replay_per_layer_boundary() -> None:
    baseline_adapter = _ScriptedAdapter()
    baseline = ChildClosedLoopCompiler(baseline_adapter, _config()).run()

    merged_adapter = _ScriptedAdapter(merged=True)
    merged = ChildClosedLoopCompiler(
        merged_adapter,
        _config(merge_adjacent_layer_captures=True),
    ).run()

    assert baseline["merge_adjacent_layer_captures_enabled"] is False
    assert merged["merge_adjacent_layer_captures_enabled"] is True
    assert baseline_adapter.replays == 14
    # Three layers share two boundaries, so two full replays disappear.
    assert merged_adapter.replays == 12
    assert merged_adapter.log.count(
        "merged:layer:000000:post_downstream|layer:000001:pre_producer"
    ) == 1
    assert merged_adapter.log.count(
        "merged:layer:000001:post_downstream|layer:000002:pre_producer"
    ) == 1

    merge_reports = [
        value["merged_captures"] for value in merged["layer_reports"]
    ]
    assert [
        value["pre_producer_served_by_merged_replay"] for value in merge_reports
    ] == [False, True, True]
    assert [
        value["post_downstream_served_next_layer_pre_producer"]
        for value in merge_reports
    ] == [True, True, False]
    assert [value["merged_next_layer_index"] for value in merge_reports] == [
        1,
        2,
        None,
    ]

    # Sealed per-point capture envelopes survive the merge: every merged layer
    # still records its own distinct capture seal per capture point.
    for baseline_layer, merged_layer in zip(
        baseline["layer_reports"], merged["layer_reports"], strict=True
    ):
        assert set(baseline_layer["captures"]) == set(merged_layer["captures"])
        for name, record in merged_layer["captures"].items():
            assert record["point"] == baseline_layer["captures"][name]["point"]
            assert record["scope"] == baseline_layer["captures"][name]["scope"]
            assert record["record_keys"] == (
                baseline_layer["captures"][name]["record_keys"]
            )
        seals = {
            record["capture_sha256"]
            for record in merged_layer["captures"].values()
        }
        assert len(seals) == len(merged_layer["captures"])


def test_merged_captures_and_restart_capsules_compose() -> None:
    adapter = _ScriptedAdapter(merged=True)
    writer = _ScriptedRestartWriter()

    report = ChildClosedLoopCompiler(
        adapter,
        _config(merge_adjacent_layer_captures=True),
        restart_capsule_config=writer,
    ).run()

    assert report["restart_capsule"]["capsules_written"] == 3
    assert report["merge_adjacent_layer_captures_enabled"] is True
    assert adapter.replays == 12


def test_merged_captures_are_ignored_when_the_adapter_declines() -> None:
    adapter = _ScriptedAdapter(merged=False)
    report = ChildClosedLoopCompiler(
        adapter,
        _config(merge_adjacent_layer_captures=True),
    ).run()

    assert report["merge_adjacent_layer_captures_requested"] is True
    assert report["merge_adjacent_layer_captures_enabled"] is False
    assert adapter.replays == 14


def test_merged_capture_that_changes_a_point_fails_closed() -> None:
    class _WrongPointAdapter(_ScriptedAdapter):
        def replay_and_capture_child_merged(self, requests, audit):
            scope, point, records = requests[0]
            wrong = (
                (scope, CapturePoint.POST_CONSUMER, records),
                requests[1],
            )
            return super().replay_and_capture_child_merged(wrong, audit)

    adapter = _WrongPointAdapter(merged=True)
    with pytest.raises(ClosedLoopFoldError, match="Merged child replay"):
        ChildClosedLoopCompiler(
            adapter,
            _config(merge_adjacent_layer_captures=True),
        ).run()


def test_deferred_merged_capture_rejects_a_moved_child_state() -> None:
    class _MutatingAdapter(_ScriptedAdapter):
        def validate_layer(self, scope, records, endpoints, capture, audit):
            decision = super().validate_layer(
                scope, records, endpoints, capture, audit
            )
            if scope.layer_index == 0:
                self.state["body_reclosure"] = "9" * 64
            return decision

    adapter = _MutatingAdapter(merged=True)
    with pytest.raises(ClosedLoopFoldError, match="exact child state"):
        ChildClosedLoopCompiler(
            adapter,
            _config(merge_adjacent_layer_captures=True),
        ).run()
