from __future__ import annotations

import copy
import json
import weakref

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_closed_loop_fold import (
    CapturePoint,
    ChildClosedLoopCompiler,
    ChildTrajectoryCapture,
    ClosedLoopCompilerConfig,
    ClosedLoopFoldError,
    CompileStage,
    CompiledFoldStage,
    InstallReceipt,
    ParentEndpointBatch,
    RESTART_CONTINUATION_EXACT_CONFIGURED,
    RESTART_CONTINUATION_TERMINAL_ONLY_NO_RECLOSURE,
    TrajectoryRecord,
    TransactionReceipt,
    ValidationDecision,
    deterministic_whole_trajectory_split,
)


torch = pytest.importorskip("torch")


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


class _ToyTransaction:
    def __init__(self, adapter: "_ToyAdapter") -> None:
        self.adapter = adapter
        self.snapshot = copy.deepcopy(adapter.state)
        self.state_sha256_before = adapter.child_state_fingerprint()
        self.topology_sha256_before = adapter.child_topology_fingerprint()
        self.active = True

    def commit(self) -> TransactionReceipt:
        assert self.active
        self.adapter.log.append("commit")
        after = self.adapter.child_state_fingerprint()
        self.active = False
        self.adapter.commit_calls += 1
        return TransactionReceipt(
            action="commit",
            accepted=True,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=after,
        )

    def rollback(self) -> TransactionReceipt:
        assert self.active
        self.adapter.state = copy.deepcopy(self.snapshot)
        self.adapter.log.append("rollback")
        self.active = False
        self.adapter.rollback_calls += 1
        return TransactionReceipt(
            action="rollback",
            accepted=False,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=self.adapter.child_state_fingerprint(),
        )


class _ToyRestartCapsuleWriter:
    def __init__(self, completed_layer_indices=(0, 1, 2)) -> None:
        self.completed = tuple(completed_layer_indices)
        self.admissions: list[dict[str, object]] = []
        self.records: list[dict[str, object]] = []

    def completed_layer_indices(self):
        return self.completed

    def assert_resume_supported(self, **kwargs):
        body = {
            "completed_layer_indices": list(
                kwargs["completed_layer_indices"]
            ),
            "remaining_layer_indices": list(
                kwargs["remaining_layer_indices"]
            ),
            "selected_layer_indices": list(kwargs["selected_layer_indices"]),
            "body_reclosure_enabled": kwargs["body_reclosure_enabled"],
            "restart_continuation_mode": kwargs[
                "restart_continuation_mode"
            ],
            "trajectory_split_sha256": kwargs["trajectory_split_sha256"],
            "child_state_sha256": kwargs["child_state_sha256"],
            "child_topology_sha256": kwargs["child_topology_sha256"],
            "authenticated": True,
        }
        admission = {
            **body,
            "admission_sha256": canonical_sha256(body),
        }
        self.admissions.append(admission)
        return admission

    def record_accepted_layer(self, **kwargs):
        record = dict(kwargs)
        self.records.append(record)
        return record

    def as_dict(self):
        return {
            "completed_layer_indices": list(self.completed),
            "admission_count": len(self.admissions),
            "record_count": len(self.records),
        }


class _ToyAdapter:
    def __init__(
        self,
        *,
        reject_layer: int | None = None,
        reject_body_reclosure: bool = False,
        reject_output_interface: bool = False,
        parent_forward_in_bridge: bool = False,
        backward_in_producer: bool = False,
    ) -> None:
        self.state = {
            "layers": {},
            "body_reclosure": None,
            "final_normalization": None,
            "output_interface": None,
            "blank_target": True,
        }
        self.topology = {
            "layers": [0, 1, 2],
            "stages": ["producer", "consumer", "downstream"],
            "body_reclosure": True,
            "final_normalization": True,
            "output_interface": True,
        }
        self.log: list[str] = []
        self.reject_layer = reject_layer
        self.reject_body_reclosure = reject_body_reclosure
        self.reject_output_interface = reject_output_interface
        self.parent_forward_in_bridge = parent_forward_in_bridge
        self.backward_in_producer = backward_in_producer
        self.commit_calls = 0
        self.rollback_calls = 0
        self.observed = {
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }

    def child_state_fingerprint(self) -> str:
        return canonical_sha256(self.state)

    def child_topology_fingerprint(self) -> str:
        return canonical_sha256(self.topology)

    def audit_counters(self):
        return dict(self.observed)

    def begin_child_transaction(self, _audit):
        self.log.append("begin")
        return _ToyTransaction(self)

    def layer_indices(self, _audit):
        self.log.append("discover_layers")
        return (2, 0, 1)

    def parent_trajectory_records(self, _audit):
        self.log.append("load_sealed_parent_trajectories")
        return tuple(reversed(_records()))

    def bridge_parent_endpoints(self, scope, records, _audit):
        self.log.append(f"{scope.scope_id}:bridge")
        if self.parent_forward_in_bridge:
            self.observed["parent_forward_calls"] += 1
        keys = tuple(sorted(record.key for record in records))
        return ParentEndpointBatch(
            scope=scope,
            record_keys=keys,
            batch_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "kind": "parent_endpoints",
                    "keys": [list(value) for value in keys],
                }
            ),
            payload={"keys": keys},
        )

    def replay_and_capture_child(self, scope, point, records, _audit):
        self.log.append(f"{scope.scope_id}:capture:{point.value}")
        keys = tuple(sorted(record.key for record in records))
        state = copy.deepcopy(self.state)
        return ChildTrajectoryCapture(
            scope=scope,
            point=point,
            record_keys=keys,
            capture_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "point": point.value,
                    "keys": [list(value) for value in keys],
                    "child_state": state,
                }
            ),
            payload={"child_state": state},
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
                    "fit_keys": [list(value) for value in keys],
                    "capture": capture.capture_sha256,
                }
            ),
            payload={"capture": capture.capture_sha256},
        )

    def compile_producer(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append(f"{scope.scope_id}:compile:producer")
        assert capture.point is CapturePoint.PRE_PRODUCER
        if self.backward_in_producer:
            torch.tensor(1.0, requires_grad=True).backward()
        return self._compiled(
            scope, CompileStage.PRODUCER, records, capture
        )

    def fit_consumer(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append(f"{scope.scope_id}:compile:consumer")
        assert capture.point is CapturePoint.POST_PRODUCER
        installed = capture.payload["child_state"]["layers"][
            scope.layer_index
        ]
        assert "producer" in installed
        return self._compiled(
            scope, CompileStage.CONSUMER, records, capture
        )

    def compile_downstream(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append(f"{scope.scope_id}:compile:downstream")
        assert capture.point is CapturePoint.POST_CONSUMER
        installed = capture.payload["child_state"]["layers"][
            scope.layer_index
        ]
        assert set(installed) == {"producer", "consumer"}
        return self._compiled(
            scope, CompileStage.DOWNSTREAM, records, capture
        )

    def compile_final_normalization(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append("final_normalization:compile")
        assert capture.point is CapturePoint.PRE_FINAL_NORMALIZATION
        assert sorted(capture.payload["child_state"]["layers"]) == [0, 1, 2]
        return self._compiled(
            scope, CompileStage.FINAL_NORMALIZATION, records, capture
        )

    def compile_body_reclosure(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append("body_reclosure:compile")
        assert capture.point is CapturePoint.PRE_BODY_RECLOSURE
        assert sorted(capture.payload["child_state"]["layers"]) == [0, 1, 2]
        assert capture.payload["child_state"]["final_normalization"] is None
        return self._compiled(
            scope, CompileStage.BODY_RECLOSURE, records, capture
        )

    def compile_output_interface(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append("output_interface:compile")
        assert capture.point is CapturePoint.PRE_OUTPUT_INTERFACE
        assert capture.payload["child_state"]["final_normalization"] is not None
        return self._compiled(
            scope, CompileStage.OUTPUT_INTERFACE, records, capture
        )

    def _install(self, scope, compiled):
        before = self.child_state_fingerprint()
        if scope.kind == "layer":
            layer = self.state["layers"].setdefault(scope.layer_index, {})
            layer[compiled.stage.value] = compiled.artifact_sha256
        elif scope.kind == "body_reclosure":
            self.state["body_reclosure"] = compiled.artifact_sha256
        elif scope.kind == "final_normalization":
            self.state["final_normalization"] = compiled.artifact_sha256
        else:
            self.state["output_interface"] = compiled.artifact_sha256
        after = self.child_state_fingerprint()
        self.log.append(f"{scope.scope_id}:install:{compiled.stage.value}")
        return InstallReceipt(
            scope=scope,
            stage=compiled.stage,
            artifact_sha256=compiled.artifact_sha256,
            accepted=True,
            state_sha256_before=before,
            state_sha256_after=after,
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

    def install_body_reclosure(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def install_output_interface(self, scope, compiled, _audit):
        return self._install(scope, compiled)

    def validate_layer(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append(f"{scope.scope_id}:validate")
        assert capture.point is CapturePoint.POST_DOWNSTREAM
        assert set(self.state["layers"][scope.layer_index]) == {
            "producer",
            "consumer",
            "downstream",
        }
        return ValidationDecision(
            scope=scope,
            accepted=scope.layer_index != self.reject_layer,
            validation_record_keys=tuple(
                sorted(record.key for record in records)
            ),
            reason=(
                "held-out whole trajectories accepted"
                if scope.layer_index != self.reject_layer
                else "intentional held-out rejection"
            ),
            metrics={"trajectory_score": 0.95},
        )

    def validate_final_normalization(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append("final_normalization:validate")
        assert capture.point is CapturePoint.POST_FINAL_NORMALIZATION
        assert self.state["final_normalization"] is not None
        return ValidationDecision(
            scope=scope,
            accepted=True,
            validation_record_keys=tuple(
                sorted(record.key for record in records)
            ),
            reason="final normalization held-out trajectories accepted",
            metrics={"trajectory_score": 0.96},
        )

    def validate_body_reclosure(
        self, scope, records, _endpoints, capture, _audit
    ):
        self.log.append("body_reclosure:validate")
        assert capture.point is CapturePoint.POST_BODY_RECLOSURE
        assert self.state["body_reclosure"] is not None
        accepted = not self.reject_body_reclosure
        return ValidationDecision(
            scope=scope,
            accepted=accepted,
            validation_record_keys=tuple(
                sorted(record.key for record in records)
            ),
            reason=(
                "body reclosure held-out trajectories accepted"
                if accepted
                else "intentional body-reclosure rejection"
            ),
            metrics={"trajectory_score": 0.955},
        )

    def validate_output_interface(
        self,
        scope,
        records,
        _endpoints,
        baseline_capture,
        candidate_capture,
        _audit,
    ):
        self.log.append("output_interface:validate")
        assert baseline_capture.point is CapturePoint.PRE_OUTPUT_INTERFACE
        assert candidate_capture.point is CapturePoint.POST_OUTPUT_INTERFACE
        assert (
            baseline_capture.payload["child_state"]["output_interface"]
            is None
        )
        assert self.state["output_interface"] is not None
        accepted = not self.reject_output_interface
        return ValidationDecision(
            scope=scope,
            accepted=accepted,
            validation_record_keys=tuple(
                sorted(record.key for record in records)
            ),
            reason=(
                "terminal interface held-out trajectories accepted"
                if accepted
                else "intentional terminal interface rejection"
            ),
            metrics={
                "teacher_forced_terminal_acceptance_is_necessary_not_sufficient": (
                    True
                )
            },
        )


def _restored_terminal_resume_adapter() -> _ToyAdapter:
    adapter = _ToyAdapter()
    adapter.state["layers"] = {
        layer_index: {
            "producer": f"sealed-producer-{layer_index}",
            "consumer": f"sealed-consumer-{layer_index}",
            "downstream": f"sealed-downstream-{layer_index}",
        }
        for layer_index in range(3)
    }
    return adapter


class _WeakCapturePayload:
    __slots__ = ("child_state", "__weakref__")

    def __init__(self, child_state: dict[str, object]) -> None:
        self.child_state = child_state

    def __getitem__(self, name: str) -> object:
        if name != "child_state":
            raise KeyError(name)
        return self.child_state


class _CaptureLifetimeToyAdapter(_ToyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.layer_capture_refs: list[
            weakref.ReferenceType[_WeakCapturePayload]
        ] = []
        self.prior_layer_capture_released: list[bool] = []
        self._prior_layer_capture_ref: (
            weakref.ReferenceType[_WeakCapturePayload] | None
        ) = None

    def replay_and_capture_child(self, scope, point, records, _audit):
        if self._prior_layer_capture_ref is not None:
            self.prior_layer_capture_released.append(
                self._prior_layer_capture_ref() is None
            )
            assert self._prior_layer_capture_ref() is None
            self._prior_layer_capture_ref = None
        if scope.kind != "layer":
            return super().replay_and_capture_child(scope, point, records, _audit)

        self.log.append(f"{scope.scope_id}:capture:{point.value}")
        keys = tuple(sorted(record.key for record in records))
        state = copy.deepcopy(self.state)
        payload = _WeakCapturePayload(state)
        payload_ref = weakref.ref(payload)
        self.layer_capture_refs.append(payload_ref)
        self._prior_layer_capture_ref = payload_ref
        return ChildTrajectoryCapture(
            scope=scope,
            point=point,
            record_keys=keys,
            capture_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "point": point.value,
                    "keys": [list(value) for value in keys],
                    "child_state": state,
                }
            ),
            payload=payload,
        )


class _BodyCaptureLifetimeToyAdapter(_ToyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.body_capture_refs: list[
            weakref.ReferenceType[_WeakCapturePayload]
        ] = []
        self.pre_body_released_before_install = False
        self.pre_body_released_before_post_capture = False

    def replay_and_capture_child(self, scope, point, records, audit):
        if scope.kind != "body_reclosure":
            return super().replay_and_capture_child(
                scope, point, records, audit
            )
        if point is CapturePoint.POST_BODY_RECLOSURE:
            self.pre_body_released_before_post_capture = bool(
                self.body_capture_refs
                and self.body_capture_refs[0]() is None
            )
            assert self.pre_body_released_before_post_capture
        keys = tuple(sorted(record.key for record in records))
        state = copy.deepcopy(self.state)
        payload = _WeakCapturePayload(state)
        self.body_capture_refs.append(weakref.ref(payload))
        return ChildTrajectoryCapture(
            scope=scope,
            point=point,
            record_keys=keys,
            capture_sha256=canonical_sha256(
                {
                    "scope": scope.scope_id,
                    "point": point.value,
                    "keys": [list(value) for value in keys],
                    "child_state": state,
                }
            ),
            payload=payload,
        )

    def install_body_reclosure(self, scope, compiled, audit):
        self.pre_body_released_before_install = bool(
            self.body_capture_refs
            and self.body_capture_refs[0]() is None
        )
        assert self.pre_body_released_before_install
        return super().install_body_reclosure(scope, compiled, audit)


def test_whole_trajectory_split_is_deterministic_and_defaults_to_all_paths() -> None:
    records = _records()

    first = deterministic_whole_trajectory_split(
        records,
        validation_fraction=0.4,
        split_seed="repeatable",
    )
    second = deterministic_whole_trajectory_split(
        tuple(reversed(records)),
        validation_fraction=0.4,
        split_seed="repeatable",
    )

    assert first.split_sha256 == second.split_sha256
    assert first.selected_paths == ("audio", "text")
    assert first.as_dict()["default_all_paths"] is True
    assert not (
        set(first.fit_trajectory_ids)
        & set(first.validation_trajectory_ids)
    )
    for trajectory_id in first.fit_trajectory_ids:
        assert {
            value.path_id
            for value in first.fit
            if value.trajectory_id == trajectory_id
        } == {"audio", "text"}
    text_only = deterministic_whole_trajectory_split(
        records,
        validation_fraction=0.4,
        split_seed="repeatable",
        selected_paths=("text",),
    )
    assert text_only.selected_paths == ("text",)
    assert {value.path_id for value in text_only.fit} == {"text"}
    assert {value.path_id for value in text_only.validation} == {"text"}


def test_compiler_runs_ascending_child_closed_loop_and_final_normalization() -> None:
    adapter = _ToyAdapter()
    before = adapter.child_state_fingerprint()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="compiler-test",
        ),
    ).run()

    assert report["selected_layers"] == [0, 1, 2]
    assert report["default_all_layers"] is True
    assert report["layers_executed_in_ascending_order"] is True
    assert report["trajectory_split"]["default_all_paths"] is True
    assert report["child_state_sha256_before"] == before
    assert report["child_state_sha256_after"] != before
    assert report["transaction_committed"] is True
    assert report["body_reclosure_fold_enabled"] is False
    assert report["body_reclosure"]["mode"] == (
        "explicit_identity_preservation"
    )
    disabled_body = dict(report["body_reclosure"])
    disabled_body_sha256 = disabled_body.pop("report_sha256")
    assert disabled_body_sha256 == canonical_sha256(disabled_body)
    assert adapter.commit_calls == 1
    assert adapter.rollback_calls == 0
    assert sorted(adapter.state["layers"]) == [0, 1, 2]
    assert adapter.state["final_normalization"] is not None
    assert adapter.state["output_interface"] is not None
    assert adapter.log.index("layer:000000:compile:producer") < (
        adapter.log.index("layer:000001:compile:producer")
    )
    assert adapter.log.index("layer:000001:compile:producer") < (
        adapter.log.index("layer:000002:compile:producer")
    )
    assert adapter.log.index("layer:000002:validate") < (
        adapter.log.index("final_normalization:compile")
    )
    assert adapter.log.index("final_normalization:validate") < (
        adapter.log.index("output_interface:compile")
    )
    for layer in report["layer_reports"]:
        assert layer["consumer"][
            "fitted_after_installed_producer_recapture"
        ] is True
        assert layer["downstream"][
            "compiled_after_installed_consumer_recapture"
        ] is True
        assert layer["fit_validation_trajectory_leakage"] is False
        fit_keys = layer["producer"]["compiled"]["fit_record_keys"]
        validation_keys = layer["validation"]["validation_record_keys"]
        assert not ({tuple(value) for value in fit_keys} & {
            tuple(value) for value in validation_keys
        })
    assert report["output_interface"][
        "teacher_forced_terminal_acceptance_is_necessary_not_sufficient"
    ]
    assert report["output_interface"][
        "external_whole_output_behavior_gate_required_for_promotion"
    ]
    assert report["audit"]["child_replay_capture_calls"] == 17
    assert report["audit"]["parent_endpoint_bridge_calls"] == 10
    assert report["audit"]["install_calls"] == 11
    assert report["audit"]["validation_calls"] == 5
    assert report["audit"]["output_interface_compile_calls"] == 1
    assert report["audit"]["parent_forward_calls"] == 0
    assert report["audit"]["training_steps"] == 0
    assert report["audit"]["backward_calls"] == 0
    assert report["audit"]["optimizer_steps"] == 0
    assert report["audit"]["zero_parent_forward_tripwire_passed"] is True
    assert report["audit"]["zero_training_tripwire_passed"] is True
    assert report["audit"]["torch_fold_only_guard_passed"] is True


def test_completed_restart_resume_keeps_configured_terminal_policy_by_default() -> (
    None
):
    adapter = _restored_terminal_resume_adapter()
    writer = _ToyRestartCapsuleWriter()
    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="terminal-resume-configured-policy",
        ),
        restart_capsule_config=writer,
    ).run()

    assert len(writer.admissions) == 1
    assert writer.admissions[0]["remaining_layer_indices"] == []
    assert writer.admissions[0]["body_reclosure_enabled"] is False
    assert writer.admissions[0]["restart_continuation_mode"] == (
        RESTART_CONTINUATION_EXACT_CONFIGURED
    )
    assert writer.records == []
    assert "terminal_only_restart_resume" not in report
    assert report["body_reclosure_fold_enabled"] is False
    assert report["body_reclosure_executed"] is False
    assert "body_reclosure:compile" not in adapter.log
    assert "final_normalization:compile" in adapter.log
    assert "output_interface:compile" in adapter.log
    assert adapter.commit_calls == 1
    assert adapter.rollback_calls == 0


def test_explicit_terminal_only_restart_resume_authenticates_and_runs_terminals() -> (
    None
):
    adapter = _restored_terminal_resume_adapter()
    writer = _ToyRestartCapsuleWriter()
    events: list[tuple[str, dict[str, object]]] = []

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="terminal-resume-success",
            fold_body_reclosure=True,
            fold_output_interface=True,
            allow_terminal_only_restart_resume=True,
        ),
        progress_sink=(
            lambda stage, metadata: events.append((stage, dict(metadata)))
        ),
        restart_capsule_config=writer,
    ).run()

    assert len(writer.admissions) == 1
    assert writer.admissions[0]["remaining_layer_indices"] == []
    assert writer.admissions[0]["body_reclosure_enabled"] is True
    assert writer.admissions[0]["restart_continuation_mode"] == (
        RESTART_CONTINUATION_TERMINAL_ONLY_NO_RECLOSURE
    )
    assert writer.records == []
    assert report["layers_executed_in_this_process"] == []
    assert report["restart_capsule"]["layer_indices_executed"] == []
    assert all(
        layer["executed_in_this_process"] is False
        for layer in report["layer_reports"]
    )
    provenance = report["terminal_only_restart_resume"]
    assert provenance["admitted"] is True
    assert provenance["restart_continuation_mode"] == (
        RESTART_CONTINUATION_TERMINAL_ONLY_NO_RECLOSURE
    )
    assert provenance["resume_admission_sha256"] == (
        writer.admissions[0]["admission_sha256"]
    )
    assert provenance["body_layer_execution_count"] == 0
    assert provenance["body_reclosure_configured"] is True
    assert provenance["body_reclosure_executed"] is False
    assert provenance["final_normalization_executed"] is True
    assert provenance["output_interface_executed"] is True
    assert provenance["capsule_identity_special_case_used"] is False
    provenance_body = dict(provenance)
    provenance_sha256 = provenance_body.pop("report_sha256")
    assert provenance_sha256 == canonical_sha256(provenance_body)
    stages = [stage for stage, _metadata in events]
    assert (
        "child_closed_loop_terminal_only_restart_resume_admitted" in stages
    )
    assert "child_closed_loop_layer_started" not in stages
    assert "child_closed_loop_body_reclosure_started" not in stages
    assert "child_closed_loop_body_reclosure_skipped" in stages
    assert "child_closed_loop_final_normalization_completed" in stages
    assert "child_closed_loop_output_interface_completed" in stages
    assert "body_reclosure:compile" not in adapter.log
    assert "final_normalization:compile" in adapter.log
    assert "output_interface:compile" in adapter.log
    assert adapter.commit_calls == 1
    assert adapter.rollback_calls == 0


def test_terminal_only_restart_resume_requires_output_interface() -> None:
    adapter = _restored_terminal_resume_adapter()
    writer = _ToyRestartCapsuleWriter()

    with pytest.raises(
        ClosedLoopFoldError,
        match="requires fold_output_interface=True",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="terminal-resume-policy-reject",
                fold_body_reclosure=True,
                fold_output_interface=False,
                allow_terminal_only_restart_resume=True,
            ),
            restart_capsule_config=writer,
        ).run()

    assert writer.admissions == []
    assert writer.records == []
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0


def test_terminal_only_restart_resume_rejects_partial_prefix() -> None:
    adapter = _ToyAdapter()
    adapter.state["layers"] = {
        0: {
            "producer": "sealed-producer-0",
            "consumer": "sealed-consumer-0",
            "downstream": "sealed-downstream-0",
        }
    }
    writer = _ToyRestartCapsuleWriter((0,))

    with pytest.raises(
        ClosedLoopFoldError,
        match="completed prefix covering every selected layer",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="terminal-resume-partial-prefix-reject",
                fold_body_reclosure=True,
                fold_output_interface=True,
                allow_terminal_only_restart_resume=True,
            ),
            restart_capsule_config=writer,
        ).run()

    assert writer.admissions == []
    assert writer.records == []
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
    assert not any(value.startswith("layer:") for value in adapter.log)


def test_terminal_only_restart_resume_rejects_subset_selection() -> None:
    adapter = _restored_terminal_resume_adapter()
    writer = _ToyRestartCapsuleWriter((0, 1))

    with pytest.raises(
        ClosedLoopFoldError,
        match="selected_layers to cover every available layer",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="terminal-resume-subset-reject",
                selected_layers=(0, 1),
                allow_terminal_only_restart_resume=True,
            ),
            restart_capsule_config=writer,
        ).run()

    assert writer.admissions == []
    assert writer.records == []
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0


def test_terminal_only_restart_resume_opt_in_must_be_boolean() -> None:
    with pytest.raises(
        ClosedLoopFoldError,
        match="allow_terminal_only_restart_resume must be boolean",
    ):
        ClosedLoopCompilerConfig(
            allow_terminal_only_restart_resume=1,  # type: ignore[arg-type]
        )


def test_layer_capture_payload_is_released_before_the_next_replay() -> None:
    adapter = _CaptureLifetimeToyAdapter()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="capture-lifetime-test",
        ),
    ).run()

    assert report["transaction_committed"] is True
    assert len(adapter.layer_capture_refs) == 12
    assert adapter.prior_layer_capture_released == [True] * 12
    assert all(reference() is None for reference in adapter.layer_capture_refs)
    for layer_report in report["layer_reports"]:
        captures = layer_report["captures"]
        assert tuple(captures) == (
            "pre_producer_fit",
            "post_producer_fit",
            "post_consumer_fit",
            "post_downstream_validation",
        )
        assert all("payload" not in value for value in captures.values())


def test_body_reclosure_runs_after_layers_and_before_final_normalization() -> None:
    adapter = _ToyAdapter()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="body-reclosure-order-test",
            fold_body_reclosure=True,
        ),
    ).run()

    assert report["body_reclosure_fold_enabled"] is True
    assert report["body_reclosure"]["fold_enabled"] is True
    assert adapter.state["body_reclosure"] is not None
    assert adapter.log.index("layer:000002:validate") < adapter.log.index(
        "body_reclosure:compile"
    )
    assert adapter.log.index("body_reclosure:validate") < adapter.log.index(
        "final_normalization:compile"
    )
    assert report["body_reclosure"]["runs_after_all_ascending_layers"] is True
    assert report["body_reclosure"]["runs_before_final_normalization"] is True
    assert report["audit"]["body_reclosure_compile_calls"] == 1
    assert report["audit"]["child_replay_capture_calls"] == 19
    assert report["audit"]["parent_endpoint_bridge_calls"] == 12
    assert report["audit"]["install_calls"] == 12
    assert report["audit"]["validation_calls"] == 6


def test_body_capture_payloads_are_released_at_compile_and_validation_boundaries() -> None:
    adapter = _BodyCaptureLifetimeToyAdapter()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="body-capture-lifetime-test",
            fold_body_reclosure=True,
        ),
    ).run()

    assert report["transaction_committed"] is True
    assert adapter.pre_body_released_before_install is True
    assert adapter.pre_body_released_before_post_capture is True
    assert len(adapter.body_capture_refs) == 2
    assert all(reference() is None for reference in adapter.body_capture_refs)
    assert all(
        "payload" not in value
        for value in report["body_reclosure"]["captures"].values()
    )


def test_progress_sink_observes_exact_closed_loop_and_reclosure_boundaries() -> None:
    adapter = _ToyAdapter()
    events: list[tuple[str, dict[str, object]]] = []

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="body-reclosure-progress-test",
            fold_body_reclosure=True,
        ),
        progress_sink=(
            lambda stage, metadata: events.append((stage, dict(metadata)))
        ),
    ).run()

    stages = [stage for stage, _metadata in events]
    assert stages[0] == "child_closed_loop_compiler_started"
    assert stages[-1] == "child_closed_loop_compiler_candidate_ready"
    assert [
        metadata["layer_index"]
        for stage, metadata in events
        if stage == "child_closed_loop_layer_started"
    ] == [0, 1, 2]
    assert [
        metadata["layer_index"]
        for stage, metadata in events
        if stage == "child_closed_loop_layer_completed"
    ] == [0, 1, 2]
    layer_zero_stages = [
        stage
        for stage, metadata in events
        if metadata.get("layer_index") == 0
    ]
    assert layer_zero_stages == [
        "child_closed_loop_layer_started",
        "child_closed_loop_layer_parent_endpoints_ready",
        "child_closed_loop_layer_capture_started",
        "child_closed_loop_layer_capture_completed",
        "child_closed_loop_layer_producer_compiled",
        "child_closed_loop_layer_producer_installed",
        "child_closed_loop_layer_capture_started",
        "child_closed_loop_layer_capture_completed",
        "child_closed_loop_layer_consumer_compiled",
        "child_closed_loop_layer_consumer_installed",
        "child_closed_loop_layer_capture_started",
        "child_closed_loop_layer_capture_completed",
        "child_closed_loop_layer_downstream_compiled",
        "child_closed_loop_layer_downstream_installed",
        "child_closed_loop_layer_capture_started",
        "child_closed_loop_layer_capture_completed",
        "child_closed_loop_layer_validation_completed",
        "child_closed_loop_layer_completed",
    ]
    assert [
        metadata["capture_point"]
        for stage, metadata in events
        if stage == "child_closed_loop_layer_capture_started"
        and metadata["layer_index"] == 0
    ] == [
        CapturePoint.PRE_PRODUCER.value,
        CapturePoint.POST_PRODUCER.value,
        CapturePoint.POST_CONSUMER.value,
        CapturePoint.POST_DOWNSTREAM.value,
    ]
    assert all(
        "schema" not in metadata and "stage" not in metadata
        for _stage, metadata in events
    )
    last_layer_completed = max(
        index
        for index, stage in enumerate(stages)
        if stage == "child_closed_loop_layer_completed"
    )
    assert last_layer_completed < stages.index(
        "child_closed_loop_body_reclosure_started"
    )
    assert stages.index("child_closed_loop_body_reclosure_started") < stages.index(
        "child_closed_loop_body_reclosure_completed"
    )
    assert stages.index("child_closed_loop_body_reclosure_completed") < stages.index(
        "child_closed_loop_final_normalization_started"
    )
    assert stages.index("child_closed_loop_final_normalization_completed") < (
        stages.index("child_closed_loop_output_interface_started")
    )
    assert report["body_reclosure_fold_enabled"] is True
    assert events[-1][1]["candidate_state_sha256"] == (
        report["child_state_sha256_after"]
    )


def test_candidate_ready_progress_failure_rolls_back_before_commit() -> None:
    adapter = _ToyAdapter()
    before = copy.deepcopy(adapter.state)
    before_sha = adapter.child_state_fingerprint()

    def reject_candidate(stage: str, _metadata: object) -> None:
        if stage == "child_closed_loop_compiler_candidate_ready":
            raise RuntimeError("candidate progress sink rejected evidence")

    with pytest.raises(RuntimeError, match="candidate progress sink rejected"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="candidate-progress-rollback-test",
                fold_body_reclosure=True,
            ),
            progress_sink=reject_candidate,
        ).run()

    assert adapter.state == before
    assert adapter.child_state_fingerprint() == before_sha
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
    assert adapter.log[-1] == "rollback"


def test_body_reclosure_rejection_rolls_back_every_layer_exactly() -> None:
    adapter = _ToyAdapter(reject_body_reclosure=True)
    before = copy.deepcopy(adapter.state)
    before_sha = adapter.child_state_fingerprint()

    with pytest.raises(ClosedLoopFoldError, match="validation rejected"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="body-reclosure-rollback-test",
                fold_body_reclosure=True,
            ),
        ).run()

    assert adapter.state == before
    assert adapter.child_state_fingerprint() == before_sha
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
    assert adapter.log[-2:] == ["body_reclosure:validate", "rollback"]
    assert "final_normalization:compile" not in adapter.log


def test_validation_rejection_rolls_back_the_entire_ascending_run() -> None:
    adapter = _ToyAdapter(reject_layer=1)
    before = copy.deepcopy(adapter.state)
    before_sha = adapter.child_state_fingerprint()

    with pytest.raises(ClosedLoopFoldError, match="validation rejected"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="rollback-test",
            ),
        ).run()

    assert adapter.state == before
    assert adapter.child_state_fingerprint() == before_sha
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
    assert adapter.log[-1] == "rollback"
    assert "final_normalization:compile" not in adapter.log


def test_validation_rejection_emits_full_authenticated_decision_before_raise() -> None:
    adapter = _ToyAdapter(reject_layer=1)
    events: list[tuple[str, dict[str, object]]] = []

    with pytest.raises(ClosedLoopFoldError, match="validation rejected"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="rejected-validation-telemetry-test",
            ),
            progress_sink=(
                lambda stage, metadata: events.append((stage, dict(metadata)))
            ),
        ).run()

    rejected = [
        metadata
        for stage, metadata in events
        if stage == "child_closed_loop_validation_decision_rejected"
    ]
    assert len(rejected) == 1
    event = rejected[0]
    assert event["scope_id"] == "layer:000001"
    assert event["layer_index"] == 1
    assert event["validation_accepted"] is False
    decision = json.loads(str(event["validation_decision_json"]))
    metrics = json.loads(str(event["validation_metrics_json"]))
    assert decision["metrics"] == metrics == {"trajectory_score": 0.95}
    assert decision["reason"] == "intentional held-out rejection"
    assert canonical_sha256(decision) == event["validation_decision_sha256"]
    assert canonical_sha256(metrics) == event["validation_metrics_sha256"]
    # The progress callback observed the rejection before the outer
    # transaction performed its exact rollback.
    assert adapter.log.index("layer:000001:validate") < adapter.log.index(
        "rollback"
    )
    assert adapter.log[-1] == "rollback"
    assert "final_normalization:compile" not in adapter.log


def test_output_interface_rejection_rolls_back_every_prior_stage() -> None:
    adapter = _ToyAdapter(reject_output_interface=True)
    before = copy.deepcopy(adapter.state)
    before_sha = adapter.child_state_fingerprint()

    with pytest.raises(ClosedLoopFoldError, match="validation rejected"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="output-interface-rollback-test",
            ),
        ).run()

    assert adapter.state == before
    assert adapter.child_state_fingerprint() == before_sha
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
    assert adapter.log[-2:] == ["output_interface:validate", "rollback"]


def test_explicit_identity_output_interface_skips_terminal_mutation() -> None:
    adapter = _ToyAdapter()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="identity-output-interface-test",
            fold_output_interface=False,
        ),
    ).run()

    assert report["transaction_committed"] is True
    assert report["output_interface_fold_enabled"] is False
    assert report["output_interface"]["mode"] == (
        "explicit_identity_preservation"
    )
    assert report["output_interface"]["fold_enabled"] is False
    assert report["output_interface"][
        "identity_is_not_a_behavioral_acceptance"
    ] is True
    assert adapter.state["output_interface"] is None
    assert "output_interface:compile" not in adapter.log
    assert "output_interface:validate" not in adapter.log
    assert report["audit"]["child_replay_capture_calls"] == 14
    assert report["audit"]["parent_endpoint_bridge_calls"] == 8
    assert report["audit"]["install_calls"] == 10
    assert report["audit"]["validation_calls"] == 4
    assert report["audit"]["output_interface_compile_calls"] == 0


def test_parent_forward_counter_tripwire_rolls_back() -> None:
    adapter = _ToyAdapter(parent_forward_in_bridge=True)
    before = adapter.child_state_fingerprint()

    with pytest.raises(Exception, match="parent_forward_calls"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="parent-tripwire",
            ),
        ).run()

    assert adapter.child_state_fingerprint() == before
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0


def test_runtime_backward_tripwire_rolls_back_without_training() -> None:
    adapter = _ToyAdapter(backward_in_producer=True)
    before = adapter.child_state_fingerprint()

    with pytest.raises(Exception, match="Backward calls are forbidden"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.4,
                split_seed="backward-tripwire",
            ),
        ).run()

    assert adapter.child_state_fingerprint() == before
    assert adapter.rollback_calls == 1
    assert adapter.commit_calls == 0
