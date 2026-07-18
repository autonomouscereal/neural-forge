from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
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
    assert config.maximum_logical_resident_gib == 0.0
    assert config.model_device == "cuda:0"
    assert config.dtype == "bfloat16"
    assert config.short_conv_v4_ridge_alphas == (1e-8, 1e-4)
    assert config.forbidden_requests == ()


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--resume", None),
        ("--resume-state", "resume.json"),
        ("--resume-store-directory", "resume"),
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


def test_proven_two_gpu_policy_receipt_is_source_bound() -> None:
    receipt = MODULE._load_proven_gpu_parity_receipt()

    assert receipt["real_two_gpu_parity_passed"] is True
    assert receipt["target_dtype"] == "bfloat16"
    assert receipt["scheduler_policy"] == (MODULE._expected_gpu_scheduler_policy())
    assert len(receipt["receipt_sha256"]) == 64
    assert len(receipt["file_sha256"]) == 64


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
    assert "compile_lfm2_cpcf_v5_selective_model_fold(" in source
    assert "Lfm2CpcfV5EndpointFirstCandidateSelector()" in source
    assert "build_lfm2_cpcf_selective_evidence_provider(cache)" in source
    assert "plan_and_synthesize_lfm2_cpcf_blank(" in source
    assert "GPU_LOGICAL_DEVICES = (0, 1)" in source
    assert hashlib.sha256(source.encode("utf-8")).hexdigest()
