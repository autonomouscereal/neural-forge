from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_cpcf_v5_selective_blank_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_cpcf_v5_selective_blank_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _catalog_preparation() -> dict[str, object]:
    digests = {
        name: MODULE._sha({"fixture": name})
        for name in (
            "preparation_receipt_sha256",
            "binding_sha256",
            "artifact_manifest_sha256",
            "artifact_catalog_sha256",
            "selective_catalog_sha256",
            "selective_source_sha256",
            "selective_provider_sha256",
            "codec_sha256",
            "provider_implementation_sha256",
            "artifact_store_source_sha256",
            "selective_provider_source_sha256",
        )
    }
    return {
        "schema": MODULE.SELECTIVE_CATALOG_PREPARATION_SCHEMA,
        "status": "prepared_authenticated_tensor_free_catalog",
        **digests,
        "warm_hit_required_for_production_fold": True,
        "cold_rebuild_permitted_in_production_fold": False,
    }


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
        "--diagnostic-json",
        str(tmp_path / "diagnostic.json"),
        "--target-unique-parameters",
        "2291840",
        "--maximum-process-rss-gib",
        "24",
    ]


def test_module_import_is_cpu_only_and_keeps_heavy_imports_lazy() -> None:
    command = (
        "import runpy,sys;"
        f"runpy.run_path({str(SCRIPT)!r});"
        "forbidden=[name for name in sys.modules "
        "if name=='torch' or name.startswith('transformers') "
        "or name.startswith('neural_forge')];"
        "print(forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


def test_parser_reuses_shared_compression_request_and_bounds_memory(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(_required_args(tmp_path))
    config = MODULE.config_from_args(args)

    assert config.compression.target_unique_parameters == 2_291_840
    assert config.compression.compression_ratio is None
    assert config.maximum_process_rss_bytes == 24 * 1024**3
    assert config.maximum_cuda_gib_per_device == 20.0
    assert config.maximum_cuda_bytes_per_device == 20 * 1024**3
    assert config.maximum_logical_resident_gib == 0.0
    assert config.minimum_physical_rss_reduction_fraction == 0.10
    assert config.model_device == "cuda:0"
    assert config.dtype == "bfloat16"
    assert config.short_conv_v4_ridge_alphas == (1e-8, 1e-4)
    assert config.resume_store_directory is None
    assert config.resume_admission_sha256 is None
    assert config.forbidden_requests == ()


def test_catalog_store_overlap_is_rejected_at_configuration(
    tmp_path: Path,
) -> None:
    args = _required_args(tmp_path)
    args.extend(
        (
            "--selective-catalog-store-directory",
            str(tmp_path / "cache" / "catalogs"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-preparation.json"),
        )
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="cannot equal, contain, or be contained",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(args))


def test_catalog_receipt_accepts_exact_safe_sibling_layout(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text('{"fixture":"trajectory"}\n', "ascii")
    args = _required_args(tmp_path)
    args[args.index("--trajectory-corpus") + 1] = str(trajectory)
    args[args.index("--diagnostic-json") + 1] = str(
        tmp_path / "work" / "diagnostic.json"
    )
    args.extend(
        (
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-control" / "preparation.json"),
        )
    )

    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))

    assert config.selective_catalog_receipt_json == (
        tmp_path / "catalog-control" / "preparation.json"
    )


def test_catalog_receipt_rejects_nested_ancestor_and_equal_paths(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text('{"fixture":"trajectory"}\n', "ascii")
    args = _required_args(tmp_path)
    args[args.index("--trajectory-corpus") + 1] = str(trajectory)
    args[args.index("--diagnostic-json") + 1] = str(
        tmp_path / "work" / "diagnostic.json"
    )
    args.extend(
        (
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-control" / "preparation.json"),
        )
    )
    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))
    invalid_receipts = (
        config.parent_model_directory / "nested-receipt.json",
        tmp_path,
        config.trajectory_corpus,
        config.resume_store_directory / "nested-receipt.json",
        config.diagnostic_json,
        config.selective_catalog_store_directory,
    )

    for receipt in invalid_receipts:
        with pytest.raises(
            MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
            match="cannot equal, contain, or be contained",
        ):
            replace(
                config,
                selective_catalog_receipt_json=receipt,
            )


def test_production_fold_authenticates_catalog_before_torch_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "parent",
        "donor-tokenizer",
        "target-tokenizer",
        "bridge",
        "cache",
    ):
        (tmp_path / name).mkdir()
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text(
        '{"fixture":"trajectory"}\n',
        "ascii",
    )
    args = _required_args(tmp_path)
    trajectory_index = args.index("--trajectory-corpus") + 1
    args[trajectory_index] = str(trajectory)
    diagnostic_index = args.index("--diagnostic-json") + 1
    args[diagnostic_index] = str(tmp_path / "work" / "diagnostic.json")
    args.extend(
        (
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-preparation.json"),
            "--expected-selective-catalog-receipt-sha256",
            "0" * 64,
        )
    )
    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))

    def forbidden_torch_configuration():
        raise AssertionError("torch configured before catalog authentication")

    monkeypatch.setattr(
        MODULE,
        "_configure_torch",
        forbidden_torch_configuration,
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="receipt is missing",
    ):
        MODULE._run(config)


def test_catalog_prepare_and_fold_source_order_is_fail_closed() -> None:
    run_source = inspect.getsource(MODULE._run)
    prepare_source = inspect.getsource(MODULE.prepare_selective_catalog)

    assert run_source.index(
        "_read_selective_catalog_preparation(config)"
    ) < run_source.index("_configure_torch()")
    assert "open_lfm2_cpcf_selective_evidence_provider_from_catalog" in run_source
    assert "open_or_build_lfm2_cpcf_selective_evidence_provider" not in run_source
    assert "load_local_lfm2_model" not in prepare_source
    assert "plan_and_synthesize_lfm2_cpcf_blank" not in prepare_source
    assert "torch.cuda.is_initialized()" in prepare_source


def test_catalog_post_run_reverification_record_is_exact() -> None:
    preparation = _catalog_preparation()
    record = dict(MODULE.selective_catalog_post_run_reverification_record(preparation))

    assert (
        dict(
            MODULE.authenticate_selective_catalog_post_run_reverification(
                record,
                catalog_preparation=preparation,
            )
        )
        == record
    )
    tampered = dict(record)
    tampered["artifact_catalog_sha256"] = "0" * 64
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact authenticated runner record",
    ):
        MODULE.authenticate_selective_catalog_post_run_reverification(
            tampered,
            catalog_preparation=preparation,
        )


def test_cuda_allocator_cap_is_enforced_on_both_logical_devices() -> None:
    class Properties:
        total_memory = 24 * 1024**3
        name = MODULE.EXPECTED_GPU_NAME

    class Cuda:
        def __init__(self) -> None:
            self.calls = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(_index):
            return Properties()

        def set_per_process_memory_fraction(self, fraction, *, device):
            self.calls.append((device, fraction))

    cuda = Cuda()
    receipt = MODULE._enforce_cuda_allocator_caps(
        SimpleNamespace(cuda=cuda),
        maximum_cuda_bytes_per_device=20 * 1024**3,
    )

    assert [device for device, _fraction in cuda.calls] == [0, 1]
    assert all(0.0 < fraction < 1.0 for _device, fraction in cuda.calls)
    assert receipt["all_devices_enforced"] is True
    assert receipt["requested_maximum_cuda_bytes_per_device"] == 20 * 1024**3
    assert all(
        row["effective_allocator_maximum_bytes"] <= 20 * 1024**3
        and row["allowed_bytes"] == 20 * 1024**3
        and row["observed_total_memory_bytes"] == 24 * 1024**3
        and row["enforced_before_substantive_cuda_allocations"] is True
        for row in receipt["device_receipts"]
    )
    body = dict(receipt)
    claimed = body.pop("receipt_sha256")
    assert claimed == MODULE._sha(body)


def test_cuda_allocator_cap_fails_when_runtime_cannot_enforce_it() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=24 * 1024**3,
            name=MODULE.EXPECTED_GPU_NAME,
        ),
        set_per_process_memory_fraction=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("allocator refused")
        ),
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Could not enforce",
    ):
        MODULE._enforce_cuda_allocator_caps(
            SimpleNamespace(cuda=cuda),
            maximum_cuda_bytes_per_device=20 * 1024**3,
        )


def test_parser_enables_cold_checkpointing_or_exact_admitted_resume(
    tmp_path: Path,
) -> None:
    store = tmp_path / "resume"
    digest = hashlib.sha256(b"admitted-v5-prefix").hexdigest()

    cold_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
        ]
    )
    cold = MODULE.config_from_args(cold_args)
    assert cold.resume_store_directory == store
    assert cold.resume_admission_sha256 is None
    assert cold.as_dict()["resume_checkpointing_requested"] is True
    assert cold.as_dict()["resume_requested"] is False

    resume_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
            "--resume-admission-sha256",
            digest,
        ]
    )
    resumed = MODULE.config_from_args(resume_args)
    assert resumed.resume_store_directory == store
    assert resumed.resume_admission_sha256 == digest
    assert resumed.as_dict()["resume_requested"] is True


def test_catalog_plan_pin_and_request_derived_ratio_are_preserved(
    tmp_path: Path,
) -> None:
    plan_sha256 = hashlib.sha256(b"catalog-plan").hexdigest()
    argv = _required_args(tmp_path)
    target_flag = argv.index("--target-unique-parameters")
    argv[target_flag : target_flag + 2] = ["--compression-ratio", "4"]
    args = MODULE.build_parser().parse_args(
        [
            *argv,
            "--expected-target-plan-fingerprint",
            plan_sha256,
        ]
    )
    config = MODULE.config_from_args(args)

    assert config.expected_target_plan_fingerprint == plan_sha256
    assert config.as_dict()["expected_target_plan_fingerprint"] == plan_sha256
    assert (
        MODULE._request_derived_structural_compression_floor(config.compression) == 4.0
    )

    exact = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_required_args(tmp_path))
    )
    assert (
        MODULE._request_derived_structural_compression_floor(exact.compression) == 1.0
    )


def test_catalog_plan_pin_requires_an_exact_digest(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--expected-target-plan-fingerprint",
            "not-a-plan",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact lowercase SHA-256",
    ):
        MODULE.config_from_args(args)


def test_resume_admission_requires_exact_digest_and_local_store(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()

    missing_store = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--resume-admission-sha256",
            hashlib.sha256(b"head").hexdigest(),
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="requires an explicit local",
    ):
        MODULE.config_from_args(missing_store)

    invalid_digest = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--resume-admission-sha256",
            "not-a-v5-admission",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact lowercase SHA-256",
    ):
        MODULE.config_from_args(invalid_digest)


def test_real_v5_adapter_uses_explicit_preregistered_measurement_gates() -> None:
    pytest.importorskip("torch")
    adapter = MODULE._build_preregistered_real_v5_numerical_adapter()

    assert (
        adapter.basis_policy.maximum_projected_fir_offdiagonal_relative_frobenius
        == MODULE.PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
        == 0.0
    )
    assert (
        adapter.basis_policy.maximum_per_lag_commutator_relative_frobenius
        == MODULE.PREREGISTERED_PER_LAG_COMMUTATOR_GATE
        == 1.0
    )
    assert adapter.basis_policy.verify_authentication()
    assert adapter.ridge_policy.verify_authentication()


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--resume", None),
        ("--resume-state", "resume.json"),
        ("--resume-checkpoint-reference", "checkpoint.json"),
        ("--publish", None),
        ("--output-directory", "output"),
        ("--publication-directory", "published"),
        ("--save-child", None),
        ("--regenerate-traces", None),
        ("--train", None),
        ("--prune", None),
        ("--quantize", None),
        ("--download", None),
    ),
)
def test_forbidden_actions_fail_closed_with_an_explicit_reason(
    tmp_path: Path,
    flag: str,
    value: str | None,
) -> None:
    argv = [*_required_args(tmp_path), flag]
    if value is not None:
        argv.append(str(tmp_path / value))
    args = MODULE.build_parser().parse_args(argv)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="forbids requested",
    ):
        MODULE.config_from_args(args)


def test_cold_resume_store_constructs_and_passes_typed_coordinator(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
    )

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))
    observed = {}

    def compile_stub(source, target, **kwargs):
        observed.update(kwargs)
        return (source, target)

    result = MODULE._compile_with_authenticated_resume(
        compile_stub,
        "source",
        "target",
        resume_runtime=runtime,
        compile_kwargs={"sealed_input": True},
    )

    assert result == ("source", "target")
    assert type(runtime.coordinator) is Lfm2CpcfV5ResumeCoordinator
    assert runtime.state is None
    assert observed["resume_coordinator"] is runtime.coordinator
    assert observed["resume_state"] is None
    report = runtime.progress.as_dict()
    assert report["checkpointing_enabled"] is True
    assert report["resume_requested"] is False
    assert report["child_publication_attempted"] is False
    assert report["parent_or_cache_tensor_sidecars"] == 0


def test_exact_coordinator_accepts_real_publish_pending_contract(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    with pytest.raises(TypeError, match="event must be Lfm2CpcfV5ProgressEvent"):
        runtime.coordinator.publish_pending(
            object(),
            rss_checkpoint_expectation={},
        )
    with pytest.raises(TypeError, match="rss_checkpoint_expectation"):
        runtime.coordinator.publish_pending(object())

    report = runtime.progress.as_dict()
    assert report["published_pending_prefix_count"] == 0
    assert report["outstanding_pending_prefix_count"] == 0


def test_real_runtime_exact_coordinator_completes_prefix_zero_admission_proof(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_admission_proof as admission_proof
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge import hf_lfm2_cpcf_v5_resume as v5_resume
    from tests.test_hf_lfm2_cpcf_resume_bridge import (
        _v5_progress,
        _v5_rss_expectation,
        _v5_rss_receipt,
    )

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))
    coordinator = runtime.coordinator

    assert type(coordinator) is v5_resume.Lfm2CpcfV5ResumeCoordinator
    assert (
        v5_compiler.Lfm2CpcfV5ResumeCoordinator is v5_resume.Lfm2CpcfV5ResumeCoordinator
    )
    assert (
        admission_proof.resume.Lfm2CpcfV5ResumeCoordinator
        is v5_resume.Lfm2CpcfV5ResumeCoordinator
    )

    prefix_zero = _v5_progress(0)
    expectation = _v5_rss_expectation(0)

    def compile_prefix_zero(_source, _target, **kwargs):
        received = kwargs["resume_coordinator"]
        assert received is coordinator
        pending = received.publish_pending(
            prefix_zero["event"],
            rss_checkpoint_expectation=expectation,
        )
        admission = received.admit(
            pending,
            rss_checkpoint_receipt=_v5_rss_receipt(
                expectation,
                passed=True,
            ),
        )
        proof = admission_proof.build_lfm2_cpcf_v5_admission_chain_proof(
            received,
            admission["admission_sha256"],
        )
        record = {
            "next_layer_index": 0,
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": pending.frontier_sha256,
            "admission_sha256": admission["admission_sha256"],
            "admission_chain_proof": dict(proof),
            "admission_chain_proof_sha256": proof["proof_sha256"],
        }
        return SimpleNamespace(report={"resume_admitted_prefixes": [record]})

    compiled = MODULE._compile_with_authenticated_resume(
        compile_prefix_zero,
        object(),
        object(),
        resume_runtime=runtime,
        compile_kwargs={},
    )

    assert compiled.report["resume_admitted_prefixes"][0]["next_layer_index"] == 0
    report = runtime.progress.as_dict()
    assert report["published_pending_prefix_count"] == 1
    assert report["admitted_prefix_count"] == 1
    assert report["outstanding_pending_prefix_count"] == 0
    assert report["rejected_prefix_count"] == 0
    assert report["admitted_prefixes"] == (compiled.report["resume_admitted_prefixes"])


def test_exact_admission_opens_only_typed_v5_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
    )
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
        Lfm2CpcfV5ResumeState,
    )

    admission_sha = hashlib.sha256(b"admitted-v5-prefix").hexdigest()
    frontier_sha = hashlib.sha256(b"frontier").hexdigest()
    frontier_reference = Lfm2CpcfArtifactReference(
        artifact_kind="boundary",
        artifact_key=hashlib.sha256(b"boundary-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"boundary-manifest").hexdigest(),
    )
    checkpoint_reference = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=hashlib.sha256(b"checkpoint-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"checkpoint-manifest").hexdigest(),
    )

    def open_admitted(self, exact_digest):
        assert exact_digest == admission_sha
        return Lfm2CpcfV5ResumeState(
            coordinator=self,
            frontier=SimpleNamespace(
                next_layer_index=3,
                frontier_sha256=frontier_sha,
            ),
            frontier_reference=frontier_reference,
            checkpoint_reference=checkpoint_reference,
            admission={"admission_sha256": admission_sha},
        )

    monkeypatch.setattr(
        Lfm2CpcfV5ResumeCoordinator,
        "open_admitted",
        open_admitted,
    )
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--resume-admission-sha256",
            admission_sha,
        ]
    )

    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    assert type(runtime.state) is Lfm2CpcfV5ResumeState
    assert runtime.state.coordinator is runtime.coordinator
    head = runtime.progress.as_dict()["invocation_head"]
    assert head["admission_sha256"] == admission_sha
    assert head["next_layer_index"] == 3
    assert head["typed_v5_state_opened"] is True
    assert head["checkpoint_reference"] == checkpoint_reference.as_dict()


@pytest.mark.parametrize("kind", ("pending", "rejected"))
def test_pending_or_rejected_digest_cannot_open_as_admitted_resume(
    tmp_path: Path,
    kind: str,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeError,
    )

    digest = hashlib.sha256(kind.encode("ascii")).hexdigest()
    store = tmp_path / f"{kind}-store"
    cold_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
        ]
    )
    MODULE._build_resume_runtime(MODULE.config_from_args(cold_args))
    if kind == "rejected":
        (store / "v5-rejections").mkdir(exist_ok=True)
    resume_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
            "--resume-admission-sha256",
            digest,
        ]
    )

    with pytest.raises(Lfm2CpcfV5ResumeError) as caught:
        MODULE._build_resume_runtime(MODULE.config_from_args(resume_args))

    assert "admission" in str(caught.value).lower()
    progress = caught.value.resume_progress_report
    assert progress["requested_admission_sha256"] == digest
    assert progress["invocation_head"] is None


def test_wrong_live_resume_context_refusal_is_preserved(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    def reject_context(_source, _target, **kwargs):
        assert kwargs["resume_coordinator"] is runtime.coordinator
        raise RuntimeError("Live resume context differs from the admitted frontier.")

    with pytest.raises(RuntimeError, match="context differs") as caught:
        MODULE._compile_with_authenticated_resume(
            reject_context,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={"runtime_policy": "different"},
        )

    progress = caught.value.resume_progress_report
    assert progress["checkpointing_enabled"] is True
    assert progress["admitted_prefix_count"] == 0
    assert progress["child_publication_attempted"] is False


def test_failed_rss_preserves_pending_and_rejected_prefix_references(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    class Reference:
        def __init__(self, kind: str, label: str) -> None:
            self.value = {
                "artifact_kind": kind,
                "artifact_key": hashlib.sha256(
                    f"{label}-key".encode("ascii")
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    f"{label}-manifest".encode("ascii")
                ).hexdigest(),
            }

        def as_dict(self):
            return dict(self.value)

    pending = SimpleNamespace(
        next_layer_index=1,
        frontier_reference=Reference("boundary", "frontier"),
        checkpoint=SimpleNamespace(reference=Reference("checkpoint", "checkpoint")),
        frontier_sha256=hashlib.sha256(b"frontier").hexdigest(),
        event_report_sha256=hashlib.sha256(b"event").hexdigest(),
    )
    rejection = {"rejection_sha256": hashlib.sha256(b"rejection").hexdigest()}
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    def fail_rss(_source, _target, **_kwargs):
        runtime.progress.record_pending(pending)
        runtime.progress.record_rejected(pending, rejection)
        raise RuntimeError("physical RSS gate failed")

    with pytest.raises(RuntimeError, match="RSS gate failed") as caught:
        MODULE._compile_with_authenticated_resume(
            fail_rss,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={},
        )

    diagnostic = MODULE.build_failure_diagnostic(caught.value)
    progress = diagnostic["resume_progress_reports"][0]
    assert progress["published_pending_prefix_count"] == 1
    assert progress["rejected_prefix_count"] == 1
    assert progress["outstanding_pending_prefix_count"] == 0
    assert progress["rejected_prefixes"][0]["promotable"] is False
    assert diagnostic["resume_artifacts_preserved_for_diagnosis"] is True
    assert diagnostic["publication_attempted"] is False
    assert diagnostic["resume_parent_or_cache_tensor_sidecars"] == 0


def test_untyped_or_v4_resume_cannot_enter_compiler_kwargs(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(_required_args(tmp_path))
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="authenticated runner seam",
    ):
        MODULE._compile_with_authenticated_resume(
            lambda *_args, **_kwargs: None,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={"resume_state": {"schema": "v4-resume"}},
        )


def _consumer_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _consumer_candidate_saver(model, destination: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.save(model.state_dict(), destination / "weights.pt")
    (destination / "config.json").write_text(
        json.dumps({"input_width": 3, "output_width": 2}, sort_keys=True),
        encoding="utf-8",
    )


def _consumer_candidate_loader(source: Path):
    torch = pytest.importorskip("torch")
    if not source.is_dir():
        raise FileNotFoundError(source)
    model = torch.nn.Linear(3, 2).requires_grad_(False).eval()
    state = torch.load(
        source / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model


def _consumer_candidate_evaluator(model):
    torch = pytest.importorskip("torch")
    from neural_forge.dynamic_candidate_capture import (
        QuarantineEvaluation,
    )

    value = model(torch.tensor([[1.0, -2.0, 0.5]]))
    return QuarantineEvaluation(
        output={"rows": value.detach().cpu().tolist()},
        evaluation={
            "finite": bool(torch.isfinite(value).all()),
            "shape": list(value.shape),
        },
    )


def _capture_consumer_candidate(
    context,
    destination: Path,
    *,
    parent_sha256: str | None = None,
):
    from neural_forge.dynamic_candidate_capture import (
        QuarantineCaptureRequest,
        QuarantineZeroOperationAudit,
        capture_quarantined_candidate,
        fingerprint_quarantine_callable,
    )

    parent_sha = context.parent_sha256 if parent_sha256 is None else parent_sha256
    counters = context.zero_operation_audit
    request = QuarantineCaptureRequest(
        plan_sha256=context.target_plan_fingerprint,
        parent_sha256=parent_sha,
        cache_sha256=context.cache_sha256,
        bridge_sha256=context.bridge_sha256,
        corpus_sha256=context.corpus_sha256,
        compiler_sha256=context.compiler_receipt_sha256,
        source_sha256=context.source_binding_sha256,
        saver_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_saver,
            role="saver",
        ),
        loader_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_loader,
            role="loader",
        ),
        evaluator_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_evaluator,
            role="evaluator",
        ),
        expected_model_state_sha256=context.target_state_sha256,
        prior_audit=QuarantineZeroOperationAudit(
            optimizer_constructions=counters["optimizer_constructions"],
            optimizer_steps=counters["optimizer_steps"],
            backward_calls=counters["backward_calls"],
            autograd_grad_calls=counters["autograd_grad_calls"],
            training_steps=counters["training_steps"],
            pruning_operations=counters["pruning_operations"],
            quantization_operations=counters["quantization_operations"],
            parent_forward_calls=counters["parent_forward_calls"],
            parent_inference_calls=counters["parent_inference_calls"],
        ),
    )
    return capture_quarantined_candidate(
        destination,
        candidate=context.target_model,
        request=request,
        saver=_consumer_candidate_saver,
        loader=_consumer_candidate_loader,
        evaluator=_consumer_candidate_evaluator,
        forbidden_parent_models=(context.forbidden_parent_model,),
    )


def _compiled_consumer_context():
    torch = pytest.importorskip("torch")
    parent = torch.nn.Linear(3, 3).requires_grad_(False).eval()
    target = torch.nn.Linear(3, 2).requires_grad_(False).eval()
    parent_auth = MODULE._live_model_authentication(parent)
    target_auth = MODULE._live_model_authentication(target)
    plan_sha = _consumer_digest("plan")
    parent_sha = _consumer_digest("parent")
    cache_sha = _consumer_digest("cache")
    bridge_sha = _consumer_digest("bridge")
    corpus_sha = _consumer_digest("corpus")
    source_sha = _consumer_digest("selective-source")
    compiler_sha = _consumer_digest("compiler-implementation")
    runner_sha = MODULE._file_sha256(SCRIPT)
    compiler_source_files = [
        {
            "module": "tests.fake_v5_compiler",
            "filename": "fake_v5_compiler.py",
            "sha256": _consumer_digest("fake-v5-compiler-source"),
        }
    ]
    parent_tokenizer_sha = _consumer_digest("parent-tokenizer")
    tokenizer_sha = _consumer_digest("target-tokenizer")
    blank_body = {
        "schema": "test-blank-v1",
        "target_plan_fingerprint": plan_sha,
        "target_unique_parameters": target_auth["parameter_inventory"][
            "unique_parameters"
        ],
    }
    blank = {**blank_body, "report_sha256": MODULE._sha(blank_body)}
    compiled_body = {
        "schema": "test-compiled-v1",
        "blank_target_report_sha256": blank["report_sha256"],
        "selective_source_sha256": source_sha,
        "compiler_implementation": {
            "implementation_sha256": compiler_sha,
            "ordered_source_files": compiler_source_files,
        },
    }
    compiled = {
        **compiled_body,
        "result_sha256": MODULE._sha(compiled_body),
    }
    zero_audit = {
        "schema": MODULE.COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA,
        **{name: 0 for name in MODULE._CONSUMER_ZERO_COUNTER_FIELDS},
        "clean": True,
    }
    from neural_forge import dynamic_fold_executor

    guard_identity = MODULE._fold_only_guard_implementation_identity(
        dynamic_fold_executor
    )
    static_audit_body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-static-fold-mutation-source-audit-v1"),
        "compiler_implementation_sha256": compiler_sha,
        "authenticated_compiler_source_files": compiler_source_files,
        "runner_source_sha256": runner_sha,
        "audited_source_count": 2,
        "source_audits": [],
        "forbidden_findings": [],
        "forbidden_finding_count": 0,
        "audit_scope": "test authenticated sources",
        "clean": True,
    }
    static_audit = {
        **static_audit_body,
        "report_sha256": MODULE._sha(static_audit_body),
    }
    guard_body = {
        "schema": "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1",
        "guard_implementation": dict(guard_identity),
        "static_pruning_quantization_source_audit": static_audit,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parent_module_forward_calls": 0,
        "parent_root_forward_calls": 0,
        "parent_inference_entrypoint_calls": 0,
        "target_module_forward_calls": 2,
        "target_root_forward_calls": 0,
        "target_inference_entrypoint_calls": 0,
        "training_steps": 0,
        "training_steps_runtime_observed_directly": False,
        "training_steps_zero_derivation": (
            "zero optimizer/backward/autograd API attempts under no-grad"
        ),
        "pruning_observation_scope": (
            "common-public-pytorch-pruning-mutation-entrypoints"
        ),
        "quantization_observation_scope": (
            "common-public-pytorch-quantization-mutation-entrypoints"
        ),
        "ordinary_bfloat16_and_float32_casts_blocked": False,
        "clean": True,
    }
    guard_report = {
        **guard_body,
        "report_sha256": MODULE._sha(guard_body),
    }
    zero_audit["compiler_runtime_guard"] = guard_report
    catalog_preparation = _catalog_preparation()
    source_binding_sha = MODULE._source_binding_sha256(
        selective_source_sha256=source_sha,
        compiler_implementation_sha256=compiler_sha,
        runner_source_sha256=runner_sha,
        fold_only_runtime_guard_identity_sha256=guard_identity["identity_sha256"],
        catalog_preparation=catalog_preparation,
    )
    seam = {
        "schema": MODULE.COMPILED_CANDIDATE_CONTEXT_SCHEMA,
        "target_plan_fingerprint": plan_sha,
        "blank_target_report_sha256": blank["report_sha256"],
        "compiler_receipt_sha256": compiled["result_sha256"],
        "parent_sha256": parent_sha,
        "cache_sha256": cache_sha,
        "bridge_sha256": bridge_sha,
        "corpus_sha256": corpus_sha,
        "source_sha256": source_sha,
        "compiler_sha256": compiler_sha,
        "runner_source_sha256": runner_sha,
        "source_binding_sha256": source_binding_sha,
        "selective_catalog_preparation_receipt_sha256": (
            catalog_preparation["preparation_receipt_sha256"]
        ),
        "selective_catalog_binding_sha256": catalog_preparation["binding_sha256"],
        "selective_catalog_artifact_manifest_sha256": (
            catalog_preparation["artifact_manifest_sha256"]
        ),
        "selective_catalog_artifact_sha256": catalog_preparation[
            "artifact_catalog_sha256"
        ],
        "selective_catalog_codec_sha256": catalog_preparation["codec_sha256"],
        "selective_catalog_provider_implementation_sha256": (
            catalog_preparation["provider_implementation_sha256"]
        ),
        "fold_only_runtime_guard_identity_sha256": guard_identity["identity_sha256"],
        "parent_tokenizer_semantics_sha256": parent_tokenizer_sha,
        "target_tokenizer_semantics_sha256": tokenizer_sha,
        "target_tokenizer_package_sha256": None,
        "target_state_sha256": target_auth["state_sha256"],
        "target_runtime_sha256": target_auth["runtime_sha256"],
        "parent_state_sha256": parent_auth["state_sha256"],
        "parent_runtime_sha256": parent_auth["runtime_sha256"],
        "zero_operation_audit": zero_audit,
    }
    diagnostic_body = {
        "schema": MODULE.RUN_SCHEMA,
        "inputs": {
            "parent_fingerprint": parent_sha,
            "cache_key": cache_sha,
            "bridge_artifact_sha256": bridge_sha,
            "trajectory_corpus_sha256": corpus_sha,
            "donor_tokenizer_semantics_sha256": parent_tokenizer_sha,
            "target_tokenizer_semantics_sha256": tokenizer_sha,
        },
        "blank_target": blank,
        "compiled_model_fold": compiled,
        "compiler_runtime_fold_only_guard": guard_report,
        "parent_parameter_inventory": parent_auth["parameter_inventory"],
        "blank_parameter_inventory": target_auth["parameter_inventory"],
        "compiled_parameter_inventory": target_auth["parameter_inventory"],
        "compiled_candidate_consumer_seam": seam,
        "selective_catalog_artifact": catalog_preparation,
        "selective_catalog_post_run_reverification": dict(
            MODULE.selective_catalog_post_run_reverification_record(catalog_preparation)
        ),
        "claims": {
            "child_checkpoint_saved": False,
            "publication_attempted": False,
        },
    }
    diagnostic = {
        **diagnostic_body,
        "receipt_sha256": MODULE._sha(diagnostic_body),
    }
    context = MODULE.V5CompiledCandidateConsumerContext(
        target_model=target,
        target_tokenizer=SimpleNamespace(name="target-tokenizer"),
        forbidden_parent_model=parent,
        target_plan_fingerprint=plan_sha,
        blank_target_report=blank,
        compiled_model_fold_report=compiled,
        compiler_receipt_sha256=compiled["result_sha256"],
        compile_diagnostic=diagnostic,
        parent_sha256=parent_sha,
        cache_sha256=cache_sha,
        bridge_sha256=bridge_sha,
        corpus_sha256=corpus_sha,
        source_sha256=source_sha,
        compiler_sha256=compiler_sha,
        runner_source_sha256=runner_sha,
        source_binding_sha256=source_binding_sha,
        parent_tokenizer_semantics_sha256=parent_tokenizer_sha,
        target_tokenizer_semantics_sha256=tokenizer_sha,
        target_tokenizer_package_sha256=None,
        parent_parameter_inventory=parent_auth["parameter_inventory"],
        blank_parameter_inventory=target_auth["parameter_inventory"],
        compiled_parameter_inventory=target_auth["parameter_inventory"],
        zero_operation_audit=zero_audit,
        target_state_sha256=target_auth["state_sha256"],
        target_runtime_sha256=target_auth["runtime_sha256"],
        parent_state_sha256=parent_auth["state_sha256"],
        parent_runtime_sha256=parent_auth["runtime_sha256"],
    )
    return context, target, parent


def _compiled_consumer_result(
    context,
    *,
    candidate,
    parent_sha256: str | None = None,
    verification_receipt=None,
):
    parent_sha = context.parent_sha256 if parent_sha256 is None else parent_sha256
    capture = candidate.receipt.as_dict()
    return MODULE.V5CompiledCandidateConsumerResult(
        candidate=candidate,
        capture_receipt=capture,
        capture_receipt_sha256=capture["receipt_sha256"],
        compiler_receipt_sha256=context.compiler_receipt_sha256,
        target_plan_fingerprint=context.target_plan_fingerprint,
        target_state_sha256=context.target_state_sha256,
        parent_sha256=parent_sha,
        cache_sha256=context.cache_sha256,
        bridge_sha256=context.bridge_sha256,
        corpus_sha256=context.corpus_sha256,
        source_sha256=context.source_sha256,
        compiler_sha256=context.compiler_sha256,
        runner_source_sha256=context.runner_source_sha256,
        source_binding_sha256=context.source_binding_sha256,
        parent_tokenizer_semantics_sha256=(context.parent_tokenizer_semantics_sha256),
        target_tokenizer_semantics_sha256=(context.target_tokenizer_semantics_sha256),
        compile_diagnostic=context.compile_diagnostic,
        verification_receipt=(
            candidate.verification_receipt
            if verification_receipt is None
            else verification_receipt
        ),
    )


def test_static_fold_audit_finds_prebound_prune_and_quantization_aliases() -> None:
    findings = MODULE._fold_only_static_source_findings(
        "\n".join(
            (
                "import torch",
                "from torch.nn.utils.prune import L1Unstructured as Plant",
                "plant = Plant.apply",
                "quant = torch.quantize_per_tensor",
                "plant(model, 'weight', amount=0.5)",
                "quant(value, 0.1, 0, torch.qint8)",
            )
        ),
        source_label="prebound_aliases.py",
    )

    kinds = {finding["kind"] for finding in findings}
    symbols = {finding["symbol"] for finding in findings}
    assert "forbidden-import-alias" in kinds
    assert "forbidden-prebound-alias" in kinds
    assert "forbidden-call" in kinds
    assert "torch.nn.utils.prune.L1Unstructured.apply" in symbols
    assert "torch.quantize_per_tensor" in symbols


def test_authenticated_compiler_source_audit_rejects_prebound_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "prebound_compiler.py"
    source_path.write_text(
        "from torch import quantize_per_tensor as hidden_quantize\n",
        encoding="utf-8",
    )
    module_name = "tests.prebound_compiler"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(__file__=str(source_path)),
    )
    implementation_body = {
        "schema": "test-compiler-implementation-v1",
        "ordered_source_files": [
            {
                "module": module_name,
                "filename": source_path.name,
                "sha256": MODULE._file_sha256(source_path),
            }
        ],
    }
    implementation = {
        **implementation_body,
        "implementation_sha256": MODULE._sha(implementation_body),
    }

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="alias that could evade",
    ):
        MODULE._audit_authenticated_compiler_sources(implementation)


def test_compiled_candidate_context_exposes_exact_authenticated_bindings() -> None:
    context, target, parent = _compiled_consumer_context()

    assert context.target_model is target
    assert context.forbidden_parent_model is parent
    assert (
        context.blank_target_report["target_plan_fingerprint"]
        == context.target_plan_fingerprint
    )
    assert (
        context.compiled_model_fold_report["result_sha256"]
        == context.compiler_receipt_sha256
    )
    assert context.zero_operation_audit["clean"] is True
    assert all(
        context.zero_operation_audit[name] == 0
        for name in MODULE._CONSUMER_ZERO_COUNTER_FIELDS
    )


def test_compiled_candidate_consumer_accepts_ordinary_quarantine_capture(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "ordinary-quarantine",
    )

    result = MODULE._invoke_compiled_candidate_consumer(
        context,
        lambda inner: _compiled_consumer_result(
            inner,
            candidate=candidate,
        ),
    )

    assert type(result) is MODULE.V5CompiledCandidateConsumerResult
    assert result.compiler_receipt_sha256 == context.compiler_receipt_sha256
    assert result.capture_receipt_sha256 != result.compiler_receipt_sha256
    assert result.compiler_child_checkpoint_saved is False
    assert result.quarantine_candidate_saved is True
    assert result.quarantine_publication_nonpromotable is True
    assert dict(result.compile_diagnostic) == dict(context.compile_diagnostic)
    assert dict(result.verification_receipt) == dict(candidate.verification_receipt)
    assert result.consumer_runtime_guard_receipt["clean"] is True
    assert (
        result.consumer_runtime_guard_receipt["capture_verification_sha256"]
        == result.verification_receipt["verification_sha256"]
    )


def test_compiled_candidate_result_rejects_duck_typed_candidate(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "exact-type-quarantine",
    )
    fake = SimpleNamespace(
        path=candidate.path,
        artifact_path=candidate.artifact_path,
        model=candidate.model,
        receipt=candidate.receipt,
        verification_receipt=candidate.verification_receipt,
        promotable=False,
        publication_authorized=False,
        requires_reviewed_rerun=True,
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact generic quarantined candidate type",
    ):
        _compiled_consumer_result(context, candidate=fake)


def test_compiled_candidate_result_rejects_resealed_verification_tamper(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "verification-tamper-quarantine",
    )
    tampered = copy.deepcopy(dict(candidate.verification_receipt))
    tampered["output_sha256"] = _consumer_digest("unrelated-output")
    tampered_body = dict(tampered)
    tampered_body.pop("verification_sha256")
    tampered["verification_sha256"] = MODULE._sha(tampered_body)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Final retained-destination reload verification",
    ):
        _compiled_consumer_result(
            context,
            candidate=candidate,
            verification_receipt=tampered,
        )


def test_outer_consumer_guard_rejects_swallowed_pruning_attempt(
    tmp_path: Path,
) -> None:
    from torch.nn.utils import prune

    context, target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "consumer-pruning-quarantine",
    )

    def attempt_pruning(inner):
        try:
            prune.l1_unstructured(target, "weight", amount=0.5)
        except Exception:
            pass
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Consumer runtime fold-only guard receipt",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            attempt_pruning,
        )


def test_compiled_candidate_consumer_rejects_target_mutation(
    tmp_path: Path,
) -> None:
    context, target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "mutation-quarantine",
    )

    def mutate(inner):
        with pytest.importorskip("torch").no_grad():
            next(target.parameters()).add_(1.0)
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="mutated",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, mutate)


def test_compiled_candidate_consumer_rejects_target_gradients(
    tmp_path: Path,
) -> None:
    context, target, _parent = _compiled_consumer_context()
    torch = pytest.importorskip("torch")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "gradient-quarantine",
    )

    def attach_gradient(inner):
        parameter = next(target.parameters())
        parameter.grad = torch.ones_like(parameter)
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="mutated",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, attach_gradient)


@pytest.mark.parametrize("direct_forward", (False, True))
def test_compiled_candidate_consumer_rejects_parent_inference(
    direct_forward: bool,
    tmp_path: Path,
) -> None:
    context, _target, parent = _compiled_consumer_context()
    torch = pytest.importorskip("torch")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / f"parent-call-{direct_forward}",
    )

    def call_parent(inner):
        if direct_forward:
            parent.forward(torch.zeros(1, 3))
        else:
            parent(torch.zeros(1, 3))
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="forbidden live parent inference",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, call_parent)


def test_compiled_candidate_consumer_rejects_wrong_result_type() -> None:
    context, _target, _parent = _compiled_consumer_context()

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="wrong exact result type",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            lambda _inner: SimpleNamespace(capture="untyped"),
        )


def test_compiled_candidate_consumer_rejects_unrelated_typed_receipt(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    wrong_parent = _consumer_digest("wrong-parent")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "wrong-parent-quarantine",
        parent_sha256=wrong_parent,
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from the exact",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            lambda inner: _compiled_consumer_result(
                inner,
                candidate=candidate,
                parent_sha256=wrong_parent,
            ),
        )


def test_compiled_candidate_result_rejects_unrelated_candidate_model(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "unrelated-model-quarantine",
    )
    with pytest.importorskip("torch").no_grad():
        next(candidate.model.parameters()).add_(1.0)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="candidate model state differs",
    ):
        _compiled_consumer_result(
            context,
            candidate=candidate,
        )


def test_current_plus_next_budget_includes_final_norm() -> None:
    references = (
        SimpleNamespace(logical_bytes=10),
        SimpleNamespace(logical_bytes=20),
        SimpleNamespace(logical_bytes=15),
    )

    resolved, required = MODULE._resolved_logical_resident_bytes(
        references=references,
        final_norm_logical_bytes=44,
        requested_gib=0.0,
    )

    assert MODULE._maximum_current_plus_next_bytes(references) == 35
    assert required == 44
    assert resolved == 44


def test_absolute_rss_cap_tightens_the_relative_compiler_policy() -> None:
    fraction, allowed = MODULE._derive_effective_rss_reduction_fraction(
        full_phase_tree_logical_bytes=1_000,
        compiler_baseline_rss_bytes=9_700,
        maximum_process_rss_bytes=10_000,
        requested_minimum_reduction_fraction=0.10,
    )

    assert allowed == 300
    assert fraction >= 0.70
    assert int(1_000 * (1.0 - fraction)) <= allowed


def test_absolute_rss_policy_rejects_an_over_limit_baseline() -> None:
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="baseline RSS already exceeds",
    ):
        MODULE._derive_effective_rss_reduction_fraction(
            full_phase_tree_logical_bytes=1_000,
            compiler_baseline_rss_bytes=10_001,
            maximum_process_rss_bytes=10_000,
            requested_minimum_reduction_fraction=0.10,
        )


def test_current_two_gpu_policy_receipt_matches_live_replay_sources() -> None:
    receipt = MODULE._load_proven_gpu_parity_receipt()

    assert receipt["receipt_sha256"] == (
        "2a80198832f265a5f1621d57c03081c1f612fc13848de99e3786727b8e2b26a3"
    )
    assert receipt["file_sha256"] == (
        "1431076192c4596244f92f4b46b52a0447e328cd3752ba6ff97a24ec5298f55b"
    )
    assert receipt["real_two_gpu_parity_passed"] is True


def test_historical_two_gpu_policy_receipt_rejects_changed_sources() -> None:
    historical = MODULE._REPOSITORY_ROOT / "reports" / "12061b4-two-rtx3090-bf16.json"

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="GPU replay source changed",
    ):
        MODULE._load_proven_gpu_parity_receipt(historical)


def test_failure_diagnostic_preserves_structured_reports_from_causes() -> None:
    nested_body = {
        "schema": "test-fold-failure-v1",
        "stage": "short_conv",
        "promotion_claimed": False,
    }
    nested = RuntimeError("inner failure")
    nested.report = {
        **nested_body,
        "report_sha256": MODULE._sha(nested_body),
    }
    outer = MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError("outer failure")
    outer.diagnostic_report = {
        "schema": MODULE.FAILURE_SCHEMA,
        "failure_evidence_only": True,
    }
    outer.__cause__ = nested

    diagnostic = MODULE.build_failure_diagnostic(outer)

    assert diagnostic["status"] == "failed"
    assert diagnostic["structured_failure_report_count"] == 2
    assert (
        diagnostic["preserved_structured_failure_reports"][1]["report"] == nested.report
    )
    body = dict(diagnostic)
    claimed = body.pop("report_sha256")
    assert claimed == MODULE._sha(body)
    assert diagnostic["publication_claimed"] is False
    assert diagnostic["child_checkpoint_saved"] is False


def test_failure_diagnostic_preserves_real_adapter_diagnostics() -> None:
    inner = RuntimeError("real adapter gate failed")
    inner.diagnostics = {
        "schema": "test-real-v5-adapter-diagnostics-v1",
        "kind": "fir_commutator_gate_failure_evidence",
        "requested_gate": 1.0,
        "observed": 1.25,
    }
    outer = MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError("outer failure")
    outer.__cause__ = inner

    diagnostic = MODULE.build_failure_diagnostic(outer)

    preserved = diagnostic["preserved_structured_failure_reports"]
    assert len(preserved) == 1
    assert preserved[0]["name"] == "diagnostics"
    assert preserved[0]["report"] == inner.diagnostics


def test_main_persists_failure_diagnostic_without_loading_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_path = tmp_path / "failure.json"
    argv = _required_args(tmp_path)
    argv[argv.index("--diagnostic-json") + 1] = str(diagnostic_path)
    nested_body = {
        "schema": "test-runtime-failure-v1",
        "stage": "catalog",
        "promotion_claimed": False,
    }

    def fail(_config):
        error = RuntimeError("synthetic pre-CUDA failure")
        error.report = {
            **nested_body,
            "report_sha256": MODULE._sha(nested_body),
        }
        raise error

    monkeypatch.setattr(MODULE, "run", fail)

    assert MODULE.main(argv) == 2
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["structured_failure_report_count"] == 1
    assert (
        diagnostic["preserved_structured_failure_reports"][0]["report"]["stage"]
        == "catalog"
    )
    assert diagnostic["publication_claimed"] is False


def test_runner_source_has_no_disallowed_execution_or_loader_lane() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        ".back" + "ward(",
        "torch." + "optim",
        "from_" + "pretrained(",
        "build_lfm2_" + "circuit_trace_cache(",
        "save_" + "pretrained(",
    )

    assert all(token not in source for token in forbidden)
    assert "v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold," in source
    assert "Lfm2CpcfV5EndpointFirstCandidateSelector()" in source
    assert "_build_preregistered_real_v5_numerical_adapter()" in source
    assert '"numerical_frontier_adapter": numerical_adapter' in source
    assert "require_short_conv_v5_real_numerical_frontier_adapter=True" in source
    assert "require_short_conv_v5_authenticated_whole_macrocell_selector=True" in source
    assert (
        '"short_conv_v5_authenticated_whole_macrocell_selection_integrated"' in source
    )
    assert '"short_conv_v5_macrocell_selector_final_authority"' in source
    assert '"short_conv_v5_endpoint_first_selector_final_authority"' in source
    assert '"short_conv_v5_endpoint_prefilter_control_integrated"' in source
    assert '"macrocell_candidate_selector_layer_indices"' in source
    assert "open_lfm2_cpcf_selective_evidence_provider_from_catalog(" in source
    assert "prepare_selective_catalog(" in source
    assert "expected_selective_catalog_receipt_sha256" in source
    assert "selective_catalog_post_run_reverification" in source
    assert "plan_and_synthesize_lfm2_cpcf_blank(" in source
    assert "Lfm2CpcfV5ResumeCoordinator" in source
    assert "coordinator.open_admitted(" in source
    assert "resume_state=resume_runtime.state" in source
    assert "resume_coordinator=resume_runtime.coordinator" in source
    assert '"resume_supported") is not True' in source
    assert '"resume_fail_closed") is not True' in source
    assert '"resume_parent_or_cache_tensor_sidecars": 0' in source
    assert '"child_checkpoint_saved": False' in source
    assert '"publication_attempted": False' in source
    assert "GPU_LOGICAL_DEVICES = (0, 1)" in source
    assert hashlib.sha256(source.encode("utf-8")).hexdigest()
