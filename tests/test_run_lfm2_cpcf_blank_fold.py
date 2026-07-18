from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_lfm2_cpcf_blank_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_cpcf_blank_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-model-directory",
        str(tmp_path / "parent"),
        "--donor-tokenizer-directory",
        str(tmp_path / "donor-tokenizer"),
        "--target-tokenizer-directory",
        str(tmp_path / "target-tokenizer"),
        "--bridge-directory",
        str(tmp_path / "bridge"),
        "--trace-cache-directory",
        str(tmp_path / "cache"),
        "--trajectory-corpus",
        str(tmp_path / "trajectories.json"),
        "--output-directory",
        str(tmp_path / "child"),
    ]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_parser_requires_one_severe_parameter_target(tmp_path: Path) -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args(tmp_path))
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "100",
                "--target-unique-parameters",
                "1000",
            ]
        )
    args = parser.parse_args([*_required_args(tmp_path), "--compression-ratio", "100"])
    config = MODULE.config_from_args(args)
    assert config.compression.compression_ratio == "100"
    assert config.compression.target_unique_parameters is None
    assert config.short_conv_fold_version == "v3"
    assert config.short_conv_v4_ridge_alphas == (1e-8, 1e-4)


@pytest.mark.parametrize(
    ("kind", "validator"),
    (
        ("file", MODULE._local_file),
        ("directory", MODULE._local_directory),
    ),
)
def test_local_inputs_reject_unresolved_link_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    validator,
) -> None:
    candidate = tmp_path / kind
    if kind == "file":
        candidate.write_text("local", encoding="utf-8")
    else:
        candidate.mkdir()
    unresolved = candidate.expanduser().absolute()
    observed = []
    original_resolve = Path.resolve

    def report_link(path):
        observed.append(path)
        return path == unresolved

    def reject_resolve(path, *args, **kwargs):
        if path == unresolved:
            raise AssertionError("link path was resolved before rejection")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(MODULE, "_is_link_or_junction", report_link)
    monkeypatch.setattr(Path, "resolve", reject_resolve)

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="explicit nonsymlink",
    ):
        validator(candidate, f"local_{kind}")

    assert observed == [unresolved]


@pytest.mark.parametrize(
    ("kind", "validator"),
    (
        ("file", MODULE._local_file),
        ("directory", MODULE._local_directory),
    ),
)
def test_local_inputs_reject_real_symlink(
    tmp_path: Path,
    kind: str,
    validator,
) -> None:
    target = tmp_path / f"{kind}-target"
    alias = tmp_path / f"{kind}-alias"
    if kind == "file":
        target.write_text("local", encoding="utf-8")
    else:
        target.mkdir()
    try:
        alias.symlink_to(target, target_is_directory=kind == "directory")
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="explicit nonsymlink",
    ):
        validator(alias, f"local_{kind}")


def test_parser_authenticates_explicit_v4_solver_controls(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--target-unique-parameters",
            "2291840",
            "--short-conv-fold-version",
            "v4",
            "--short-conv-v4-ridge-alpha",
            "1e-8",
            "--short-conv-v4-ridge-alpha",
            "1e-4",
            "--short-conv-v4-variable-projection-passes",
            "2",
            "--short-conv-v4-context-objective-weight",
            "0.075",
        ]
    )

    config = MODULE.config_from_args(args)

    assert config.short_conv_fold_version == "v4"
    assert config.short_conv_v4_ridge_alphas == (1e-8, 1e-4)
    assert config.short_conv_v4_variable_projection_passes == 2
    assert config.short_conv_v4_context_objective_weight == 0.075


def test_parser_accepts_pinned_resume_store_and_checkpoint_manifest(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    reference = tmp_path / "checkpoint" / "manifest.json"
    args = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--target-unique-parameters",
            "2291840",
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--resume-checkpoint-reference",
            str(reference),
        ]
    )

    config = MODULE.config_from_args(args)

    assert config.resume_store_directory == tmp_path / "resume"
    assert config.resume_checkpoint_reference == reference


def test_resume_checkpoint_requires_store_and_real_fold(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    checkpoint_only = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--compression-ratio",
            "100",
            "--resume-checkpoint-reference",
            str(tmp_path / "manifest.json"),
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="requires resume_store_directory",
    ):
        MODULE.config_from_args(checkpoint_only)

    dry_store = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--compression-ratio",
            "100",
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--structural-dry-plan",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="only valid for a real fold",
    ):
        MODULE.config_from_args(dry_store)


def test_checkpoint_reference_reader_accepts_pinned_reference_and_manifest(
    tmp_path: Path,
) -> None:
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        RESUME_ARTIFACT_SCHEMA,
    )

    reference_body = {
        "artifact_kind": "checkpoint",
        "artifact_key": _digest("checkpoint"),
        "manifest_sha256": _digest("manifest"),
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(reference_body, sort_keys=True),
        encoding="utf-8",
    )
    reference = MODULE._read_resume_checkpoint_reference(reference_path)
    assert reference.as_dict() == reference_body

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": RESUME_ARTIFACT_SCHEMA,
                **reference_body,
                "untrusted_extra_is_authenticated_by_store": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    from_manifest = MODULE._read_resume_checkpoint_reference(manifest_path)
    assert from_manifest.as_dict() == reference_body


def test_checkpoint_reference_reader_rejects_unpinned_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "checkpoint",
                "artifact_key": _digest("checkpoint"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="exact pinned",
    ):
        MODULE._read_resume_checkpoint_reference(path)


def test_v4_ridge_candidates_must_be_unique_and_increasing(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--compression-ratio",
            "100",
            "--short-conv-fold-version",
            "v4",
            "--short-conv-v4-ridge-alpha",
            "1e-4",
            "--short-conv-v4-ridge-alpha",
            "1e-8",
        ]
    )

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="unique and increasing",
    ):
        MODULE.config_from_args(args)


@pytest.mark.parametrize("ratio", ["0.5", "1", "NaN"])
def test_noncompacting_ratio_is_refused(ratio: str) -> None:
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="greater than one",
    ):
        MODULE.CompressionRequest(
            compression_ratio=ratio,
            target_unique_parameters=None,
        )


def test_exact_parameter_budget_remains_generic_until_promotion_audit() -> None:
    request = MODULE.CompressionRequest(
        compression_ratio=None,
        target_unique_parameters=1_000_000,
    )
    assert request.as_kwargs() == {
        "compression_ratio": None,
        "target_unique_parameters": 1_000_000,
    }
    control = MODULE.CompressionRequest(
        compression_ratio="10",
        target_unique_parameters=None,
    )
    assert control.compression_ratio == "10"
    assert MODULE.MINIMUM_CLAIMABLE_COMPRESSION_RATIO == 100


def test_trajectory_reader_preserves_whole_record_splits(
    tmp_path: Path,
) -> None:
    assert MODULE.TRAJECTORY_SCHEMA == ("neural-forge-lfm2-bridge-trajectory-text-v1")
    source = tmp_path / "trajectories.json"
    rows = [
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "selection-1",
            "split": "selection",
            "input_text": "Pick one.",
            "forced_decode_text": "One.",
        },
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "fit-1",
            "split": "fit",
            "input_text": "Add two numbers.",
            "forced_decode_text": "Four.",
        },
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "validation-1",
            "split": "validation",
            "input_text": "Explain rain.",
            "forced_decode_text": "Water condenses.",
        },
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    loaded = MODULE._read_trajectory_rows(source)
    assert tuple(row["split"] for row in loaded) == (
        "selection",
        "fit",
        "validation",
    )
    assert loaded[2]["forced_decode_text"] == "Water condenses."


def test_compiler_implementation_fingerprint_binds_every_local_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("VALUE = 1\n", encoding="utf-8")
    second.write_text("VALUE = 2\n", encoding="utf-8")
    modules = (
        SimpleNamespace(__name__="compiler.first", __file__=str(first)),
        SimpleNamespace(__name__="compiler.second", __file__=str(second)),
    )

    before = MODULE._compiler_implementation_fingerprint(modules)
    assert before == MODULE._compiler_implementation_fingerprint(
        tuple(reversed(modules))
    )

    second.write_text("VALUE = 3\n", encoding="utf-8")
    after = MODULE._compiler_implementation_fingerprint(modules)
    assert after != before


def test_resume_boundary_bindings_authenticate_live_inputs_without_parent_copy() -> (
    None
):
    torch = pytest.importorskip("torch")
    phase = SimpleNamespace(
        batch_index=0,
        record_id="fit-0",
        split="fit",
        phase="prefill",
        phase_sha256=_digest("phase"),
        parent_evidence=SimpleNamespace(evidence_sha256=_digest("parent-evidence")),
    )
    policy = SimpleNamespace(
        as_dict=lambda: {
            "ridge": 1e-8,
            "minimum_predictive_energy_fraction": 0.9,
        }
    )
    source_table = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    kwargs = {
        "parent_fingerprint": _digest("parent"),
        "cache_key": _digest("cache"),
        "phases": (phase,),
        "trace_sha256": _digest("trace"),
        "boundary_bridge_sha256": _digest("boundary-bridge"),
        "tokenizer_sha256": _digest("tokenizer"),
        "source_tied_table": source_table,
        "blank_report": {
            "blank_state_fingerprint": _digest("blank"),
            "target_topology_fingerprint": _digest("topology"),
        },
        "policy": policy,
        "compiler_fingerprint_sha256": _digest("compiler"),
    }

    first = MODULE._build_resume_boundary_bindings(**kwargs)
    second = MODULE._build_resume_boundary_bindings(**kwargs)

    assert first == second
    assert first.parent_sha256 == _digest("parent")
    assert first.bridge_sha256 == _digest("boundary-bridge")
    assert first.source_table_sha256 == MODULE._tensor_payload_sha256(
        source_table,
        "source_tied_table",
    )
    phase.parent_evidence.evidence_sha256 = _digest("changed-evidence")
    changed = MODULE._build_resume_boundary_bindings(**kwargs)
    assert changed.evidence_sha256 != first.evidence_sha256


def test_failure_publication_wires_exact_proofs_and_excludes_parent_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import neural_forge.hf_lfm2_cpcf_resume_bridge as resume_bridge

    phase = SimpleNamespace(
        batch_index=0,
        record_id="fit-0",
        split="fit",
        phase="prefill",
        phase_sha256=_digest("phase"),
        parent_evidence=SimpleNamespace(evidence_sha256=_digest("parent-evidence")),
    )
    failure = SimpleNamespace(
        boundary_bundle=SimpleNamespace(layers=(object(),)),
        completed_layer_receipts=(),
        report={
            "failed_prefix_target_cpu_tensor_snapshot": {
                "snapshot_sha256": _digest("guard")
            }
        },
        rollback_proof={
            "source_state_fingerprint_before": _digest("parent-proof"),
            "proof_sha256": _digest("rollback"),
        },
    )

    class Target:
        def named_parameters(self, *, remove_duplicate: bool):
            assert remove_duplicate is False
            parameter = torch.nn.Parameter(torch.zeros(1))
            return iter(
                (
                    ("model.embed_tokens.weight", parameter),
                    ("lm_head.weight", parameter),
                    ("model.embedding_norm.weight", parameter),
                    ("model.layers.0.operator_norm.weight", parameter),
                )
            )

    captured = {}

    def publish(store, observed_failure, **kwargs):
        captured["store"] = store
        captured["failure"] = observed_failure
        captured.update(kwargs)
        return "published"

    monkeypatch.setattr(
        resume_bridge,
        "flatten_lfm2_cpcf_boundary_bundle",
        lambda _bundle: SimpleNamespace(bridge_sha256=_digest("boundary-bridge")),
    )
    monkeypatch.setattr(
        resume_bridge,
        "publish_lfm2_cpcf_model_fold_failure_context",
        publish,
    )
    result = MODULE._publish_failure_resume_context(
        store="store",
        failure=failure,
        target_model=Target(),
        phases=(phase,),
        parent_fingerprint=_digest("parent"),
        cache_key=_digest("cache"),
        trace_sha256=_digest("trace"),
        tokenizer_sha256=_digest("tokenizer"),
        source_tied_table=torch.zeros((4, 3)),
        blank_report={
            "blank_state_fingerprint": _digest("blank"),
            "target_topology_fingerprint": _digest("topology"),
        },
        policy=SimpleNamespace(as_dict=lambda: {"ridge": 1e-8}),
        bridge_artifact_sha256=_digest("tokenizer-bridge"),
        full_policy_sha256=_digest("policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        checkpoint_reference=None,
    )

    assert result == "published"
    assert captured["failure"] is failure
    assert captured["guard_proof_sha256"] == _digest("guard")
    assert captured["parent_proof_sha256"] == _digest("parent-proof")
    assert captured["rollback_proof_sha256"] == _digest("rollback")
    assert captured["global_installed_parameter_paths"] == (
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.embedding_norm.weight",
    )
    assert "parent_evidence" not in captured


def _progress_publisher_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    torch = pytest.importorskip("torch")
    import neural_forge.hf_lfm2_cpcf_resume_bridge as resume_bridge
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
    )

    policy_sha = _digest("policy")
    compiler_sha = _digest("compiler")
    bundle = SimpleNamespace(layers=(object(), object()))
    phase = SimpleNamespace(
        batch_index=0,
        record_id="fit-0",
        split="fit",
        phase="prefill",
        phase_sha256=_digest("phase"),
        parent_evidence=SimpleNamespace(evidence_sha256=_digest("parent-evidence")),
    )
    boundary_reference = Lfm2CpcfArtifactReference(
        artifact_kind="boundary",
        artifact_key=_digest("boundary-key"),
        manifest_sha256=_digest("boundary-manifest"),
    )
    boundary = SimpleNamespace(
        reference=boundary_reference,
        path=tmp_path / "resume" / "boundaries" / _digest("boundary-key"),
    )
    resume_root = tmp_path / "resume"
    resume_root.mkdir()

    class Store:
        root = resume_root

        def final_reauthenticate_boundary(self, value):
            return value

        def final_reauthenticate_checkpoint(self, value):
            assert (value.path / "manifest.json").is_file()
            return value

    class Event:
        def __init__(self, next_layer_index: int):
            self.next_layer_index = next_layer_index
            self.full_policy_sha256 = policy_sha
            self.boundary_bundle = bundle
            self.report_sha256 = _digest(f"event-{next_layer_index}")
            self.verify_count = 0

        def verify(self):
            self.verify_count += 1

    publications = []

    def publish(
        _store,
        event,
        *,
        boundary_bindings,
        full_policy_sha256,
        compiler_fingerprint_sha256,
        global_installed_parameter_paths,
        pinned_boundary,
        previous_checkpoint,
    ):
        assert full_policy_sha256 == policy_sha
        assert compiler_fingerprint_sha256 == compiler_sha
        assert global_installed_parameter_paths == (
            "lm_head.weight",
            "model.embed_tokens.weight",
        )
        assert boundary_bindings.bridge_sha256 == _digest("progress-boundary")
        if publications:
            assert pinned_boundary is boundary
            assert previous_checkpoint is publications[-1].checkpoint
        else:
            assert pinned_boundary is None
            assert previous_checkpoint is None
        checkpoint_reference = Lfm2CpcfArtifactReference(
            artifact_kind="checkpoint",
            artifact_key=_digest(f"checkpoint-{event.next_layer_index}"),
            manifest_sha256=_digest(f"manifest-{event.next_layer_index}"),
        )
        checkpoint_path = (
            tmp_path / "resume" / "checkpoints" / checkpoint_reference.artifact_key
        )
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "manifest.json").write_text(
            json.dumps(checkpoint_reference.as_dict()),
            encoding="utf-8",
        )
        checkpoint = SimpleNamespace(
            reference=checkpoint_reference,
            path=checkpoint_path,
            completed_layers={"next_layer_index": event.next_layer_index},
        )
        result = SimpleNamespace(
            boundary=boundary,
            checkpoint=checkpoint,
        )
        publications.append(result)
        return result

    monkeypatch.setattr(
        resume_bridge,
        "flatten_lfm2_cpcf_boundary_bundle",
        lambda _bundle: SimpleNamespace(bridge_sha256=_digest("progress-boundary")),
    )
    monkeypatch.setattr(
        resume_bridge,
        "publish_lfm2_cpcf_model_fold_progress_checkpoint",
        publish,
        raising=False,
    )
    publisher = MODULE._Lfm2CpcfResumeProgressPublisher(
        store=Store(),
        phases=(phase,),
        parent_fingerprint=_digest("parent"),
        cache_key=_digest("cache"),
        trace_sha256=_digest("trace"),
        tokenizer_sha256=_digest("tokenizer"),
        source_tied_table=torch.zeros((4, 3)),
        blank_report={
            "blank_state_fingerprint": _digest("blank"),
            "target_topology_fingerprint": _digest("topology"),
        },
        policy=SimpleNamespace(as_dict=lambda: {"ridge": 1e-8}),
        full_policy_sha256=policy_sha,
        compiler_fingerprint_sha256=compiler_sha,
        bridge_artifact_sha256=_digest("tokenizer-bridge"),
        global_installed_parameter_paths=(
            "lm_head.weight",
            "model.embed_tokens.weight",
        ),
    )
    return publisher, Event, publications


def test_progress_publisher_keeps_latest_checkpoint_across_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, Event, publications = _progress_publisher_fixture(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "child"
    seed = Event(0)
    accepted = Event(1)
    final_nonresumable = Event(2)

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        publisher(seed)
        publisher(accepted)
        publisher(final_nonresumable)
        raise KeyboardInterrupt("cancelled")

    assert seed.verify_count == 1
    assert accepted.verify_count == 1
    assert final_nonresumable.verify_count == 1
    assert len(publications) == 2
    assert publisher.latest_checkpoint is publications[-1].checkpoint
    assert publisher.latest_checkpoint_manifest.is_file()
    assert publisher.latest_pointer_path.is_file()
    selection = MODULE._read_resume_checkpoint_reference(publisher.latest_pointer_path)
    assert selection.reference == publisher.latest_checkpoint.reference
    assert selection.latest_pointer["next_layer_index"] == 1
    assert publisher.as_dict()["latest_resumable_next_layer_index"] == 1
    assert publisher.skipped_nonresumable_event_count == 1
    assert publisher.as_dict()["event_objects_retained"] == 0
    assert not output.exists()


def test_progress_publication_failure_never_advances_verified_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, Event, publications = _progress_publisher_fixture(
        tmp_path,
        monkeypatch,
    )
    import neural_forge.hf_lfm2_cpcf_resume_bridge as resume_bridge

    publisher(Event(0))
    first = publisher.latest_checkpoint

    def fail_publish(*_args, **_kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(
        resume_bridge,
        "publish_lfm2_cpcf_model_fold_progress_checkpoint",
        fail_publish,
    )
    with pytest.raises(OSError, match="synthetic publication failure"):
        publisher(Event(1))

    assert len(publications) == 1
    assert publisher.latest_checkpoint is first
    assert publisher.latest_checkpoint_manifest.is_file()
    assert publisher.published_checkpoint_count == 1
    assert not (tmp_path / "child").exists()


def test_latest_pointer_rejects_tamper_and_wrong_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, Event, _publications = _progress_publisher_fixture(
        tmp_path,
        monkeypatch,
    )
    publisher(Event(0))
    pointer = json.loads(publisher.latest_pointer_path.read_text(encoding="utf-8"))

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="differs from this invocation",
    ):
        MODULE._validate_latest_resume_pointer(
            pointer,
            expected_invocation_sha256=_digest("wrong-invocation"),
        )

    pointer["next_layer_index"] = 1
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="self-seal",
    ):
        MODULE._validate_latest_resume_pointer(pointer)


def test_latest_pointer_same_checkpoint_is_idempotent_across_event_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, Event, _publications = _progress_publisher_fixture(
        tmp_path,
        monkeypatch,
    )
    publisher(Event(0))
    pointer_path = publisher.latest_pointer_path
    original_bytes = pointer_path.read_bytes()
    original = json.loads(original_bytes)
    replacement_event_sha256 = _digest("same-checkpoint-new-process")
    assert replacement_event_sha256 != original["event_report_sha256"]

    observed_path, observed = MODULE._publish_latest_resume_pointer(
        store=publisher.store,
        boundary_bindings=publisher.boundary_bindings,
        boundary=publisher.pinned_boundary,
        checkpoint=publisher.latest_checkpoint,
        full_policy_sha256=publisher.full_policy_sha256,
        compiler_fingerprint_sha256=(publisher.compiler_fingerprint_sha256),
        bridge_artifact_sha256=publisher.bridge_artifact_sha256,
        base_checkpoint_reference=publisher.base_checkpoint_reference,
        previous_checkpoint=publisher.latest_checkpoint,
        event_report_sha256=replacement_event_sha256,
    )

    assert observed_path == pointer_path
    assert observed == original
    assert pointer_path.read_bytes() == original_bytes
    assert observed["event_report_sha256"] != replacement_event_sha256


def test_interrupted_latest_pointer_update_preserves_prior_valid_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher, Event, _publications = _progress_publisher_fixture(
        tmp_path,
        monkeypatch,
    )
    publisher(Event(0))
    pointer_path = publisher.latest_pointer_path
    prior = pointer_path.read_bytes()
    prior_checkpoint = publisher.latest_checkpoint

    def interrupt_replace(*_args, **_kwargs):
        raise KeyboardInterrupt("replace interrupted")

    monkeypatch.setattr(MODULE.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt, match="replace interrupted"):
        publisher(Event(1))

    assert pointer_path.read_bytes() == prior
    verified = MODULE._read_resume_checkpoint_reference(pointer_path)
    assert verified.reference == prior_checkpoint.reference
    assert publisher.latest_checkpoint is prior_checkpoint
    assert not tuple(pointer_path.parent.glob("*.stage-*"))
    assert not (tmp_path / "child").exists()


def test_source_has_no_remote_loader_training_or_fallback_path() -> None:
    source = inspect.getsource(MODULE)
    assert ".from_pretrained(" not in source
    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "build_lfm2_circuit_trace_cache" not in source
    assert "capture_lfm2" not in source
    assert "parent.generate(" not in source
    assert "structural_dry_plan_only" in source
    assert 'folded_child_created": False' in source
    assert "marginal_compaction_reported_as_success" in source
    assert "--diagnostic-control" in source
    assert "compile_lfm2_cpcf_model_fold" in source
    assert "except BaseException as error:" in source
    assert 'output / "cpcf_run_receipt.json"' in source


def test_standalone_save_cancellation_removes_private_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()

    def cancel_save(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt("cancelled")

    monkeypatch.setattr(model, "save_pretrained", cancel_save)
    output = tmp_path / "child"

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        MODULE._save_reload_evaluate(
            model,
            object(),
            output=output,
            parent_model=model,
            planned_parameters=sum(
                parameter.numel()
                for parameter in {
                    id(value): value for value in model.parameters()
                }.values()
            ),
            rows=(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            evaluation_limit=1,
            evaluation_max_new_tokens=1,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".child.stage-*"))


def test_direct_script_entry_resolves_sibling_experiment_imports(
    tmp_path: Path,
) -> None:
    probe = (
        "import importlib.util,sys;"
        f"path={str(SCRIPT)!r};"
        "spec=importlib.util.spec_from_file_location('cpcf_direct',path);"
        "module=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module);"
        "assert importlib.util.find_spec("
        "'experiments.run_lfm2_residual_attention_fold') is not None"
    )
    command = [
        sys.executable,
        "-c",
        probe,
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_repetition_proxy_is_diagnostic_not_intelligence_claim() -> None:
    healthy = MODULE._repetition_proxy([1, 2, 3, 2, 4, 5])
    collapsed = MODULE._repetition_proxy([7] * 12)
    assert healthy["passed"] is True
    assert collapsed["passed"] is False
    source = inspect.getsource(MODULE._standalone_whole_response_evaluation)
    assert "manual_intelligence_retention_proven" in source
    assert "False" in source


def test_standalone_receipt_seals_generation_limits_and_token_ids() -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(
            self,
            _text: str,
            *,
            add_special_tokens: bool,
            return_tensors: str,
        ) -> dict[str, object]:
            assert add_special_tokens is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[11, 12]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
        ) -> str:
            assert skip_special_tokens is True
            return " ".join(str(token_id) for token_id in token_ids)

    class Model:
        def __init__(self) -> None:
            self.max_new_tokens_calls: list[int] = []

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> object:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            self.max_new_tokens_calls.append(int(kwargs["max_new_tokens"]))
            continuation = torch.tensor(
                [[21, 22, 23]],
                device=input_ids.device,
            )
            return torch.cat((input_ids, continuation), dim=1)

    rows = [
        {
            "id": f"validation-{index}",
            "split": "validation",
            "input_text": f"Prompt {index}",
            "forced_decode_text": f"Reference {index}",
        }
        for index in range(3)
    ]
    model = Model()
    report = MODULE._standalone_whole_response_evaluation(
        model,
        Tokenizer(),
        rows=rows,
        device=torch.device("cpu"),
        limit=2,
        max_new_tokens=128,
    )

    assert report["schema"] == MODULE.STANDALONE_EVALUATION_SCHEMA
    assert report["requested_case_limit"] == 2
    assert report["requested_max_new_tokens"] == 128
    assert report["generation_limit"] == 128
    assert report["generation_limit_semantics"] == ("maximum_new_tokens_per_case")
    assert report["requested_max_new_tokens_matches_generation_limit"] is True
    assert report["case_count"] == 2
    assert model.max_new_tokens_calls == [128, 128]
    assert report["all_observed_token_counts_match_continuation_token_ids"] is True
    assert report["all_observed_token_counts_within_generation_limit"] is True
    for case in report["cases"]:
        assert case["continuation_token_ids"] == [21, 22, 23]
        assert case["observed_continuation_token_count"] == 3
        assert case["observed_token_count_matches_continuation_token_ids"] is True
        assert case["observed_token_count_within_generation_limit"] is True
        assert case["repetition_proxy"]["token_count"] == 3

    sealed_body = dict(report)
    report_sha256 = sealed_body.pop("report_sha256")
    assert report_sha256 == MODULE._sha(sealed_body)
    mutated = json.loads(json.dumps(sealed_body))
    mutated["cases"][0]["continuation_token_ids"][0] = 999
    assert MODULE._sha(mutated) != report_sha256


def test_local_lfm2_save_reload_preserves_and_restores_ffn_alias(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
    )

    config = transformers.Lfm2Config(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=47,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    before = MODULE._restore_lfm2_ffn_config_alias(
        model,
        "test_target_before_save",
    )
    checkpoint = tmp_path / "child"
    model.save_pretrained(checkpoint, safe_serialization=True)
    saved = MODULE._verify_saved_lfm2_ffn_config(
        checkpoint,
        expected_width=47,
    )
    reloaded = load_local_lfm2_model(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
    )
    consumed_alias = getattr(reloaded.config, "block_ff_dim", None)
    after = MODULE._restore_lfm2_ffn_config_alias(
        reloaded,
        "test_target_after_reload",
    )

    assert before["materialized_ffn_width"] == 47
    assert saved["serialized_aliases"]["block_ff_dim"] == 47
    assert consumed_alias is None
    assert after["previous_block_ff_dim"] is None
    assert reloaded.config.block_ff_dim == 47
    assert reloaded.config.to_dict()["block_ff_dim"] == 47
    assert {
        int(layer.feed_forward.w1.weight.shape[0]) for layer in reloaded.model.layers
    } == {47}


def test_ffn_alias_audit_rejects_conflicting_serialized_width() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Lfm2Config(
        vocab_size=12,
        hidden_size=8,
        intermediate_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv"],
        block_auto_adjust_ff_dim=False,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.config.block_ff_dim = 12

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="conflicts",
    ):
        MODULE._restore_lfm2_ffn_config_alias(
            model,
            "conflicting_target",
        )
