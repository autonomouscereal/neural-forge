from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from experiments import (  # noqa: E402
    migrate_lfm2_semantic_foundation_statistics_frontier as cli,
)
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_semantic_foundation_statistics import (  # noqa: E402
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA,
    Lfm2SemanticFoundationInterfaceFrontierCompanion,
)


ARTIFACT = "1" * 64
CACHE = "2" * 64
CACHE_MANIFEST = "3" * 64
PARENT = "4" * 64
BRIDGE = "5" * 64
BRIDGE_PROFILE = "6" * 64
FINAL_NORM = "7" * 64
MAPPED = "8" * 64
MAPPED_SOURCE = "9" * 64
FRONTIER = "a" * 64


def _seal(body, field):
    return {**body, field: canonical_sha256(body)}


def _canonical_file(path: Path, value) -> str:
    raw = canonical_json_bytes(dict(value)) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _companion(receipt_sha256: str, receipt_file_sha256: str):
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA,
        "status": "frontier-aware-companion-derived-from-legacy-compact-statistics",
        "statistics_artifact_sha256": ARTIFACT,
        "legacy_foundation_receipt_sha256": receipt_sha256,
        "legacy_foundation_receipt_file_sha256": receipt_file_sha256,
        "frontier_sha256": FRONTIER,
        "frontier_materializable_candidate_indices": list(range(10)),
        "frontier_whole_response_review_order_candidate_indices": list(range(10)),
        "numeric_metrics_have_final_intelligence_authority": False,
        "whole_response_review_selects_final_candidate": True,
        "operation_audit": dict(cli._EXPECTED_COMPANION_OPERATION_AUDIT),
    }
    receipt = _seal(body, "companion_sha256")
    return Lfm2SemanticFoundationInterfaceFrontierCompanion(
        interface=None,
        anchor_fold=None,
        legacy_solve_report={},
        companion_receipt=receipt,
    )


def _args(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, object]]:
    interface_solve = _seal(
        {
            "cache_parent_fingerprint": PARENT,
            "bridge_artifact_sha256": BRIDGE,
        },
        "result_sha256",
    )
    legacy_body = {
        "schema": "fixture-legacy-foundation-v1",
        "parent_identity_sha256": PARENT,
        "cache_parent_fingerprint": PARENT,
        "bridge_artifact_sha256": BRIDGE,
        "interface_solve": interface_solve,
    }
    legacy = _seal(legacy_body, "receipt_sha256")
    legacy_path = tmp_path / "legacy-foundation.json"
    legacy_file_sha256 = _canonical_file(legacy_path, legacy)
    for name in (
        "statistics",
        "cache",
        "bridge",
        "donor-tokenizer",
        "target-tokenizer",
        "parent",
    ):
        (tmp_path / name).mkdir()
    args = argparse.Namespace(
        legacy_statistics_artifact_directory=tmp_path / "statistics",
        expected_legacy_statistics_artifact_sha256=ARTIFACT,
        semantic_cache_directory=tmp_path / "cache",
        expected_semantic_cache_key=CACHE,
        bridge_directory=tmp_path / "bridge",
        expected_bridge_artifact_sha256=BRIDGE,
        donor_tokenizer_directory=tmp_path / "donor-tokenizer",
        target_tokenizer_directory=tmp_path / "target-tokenizer",
        parent_model_directory=tmp_path / "parent",
        legacy_foundation_receipt_json=legacy_path,
        expected_legacy_foundation_receipt_sha256=legacy["receipt_sha256"],
        expected_legacy_foundation_receipt_file_sha256=legacy_file_sha256,
        output_companion_json=tmp_path / "frontier-companion.json",
        parent_final_norm_tensor_key="model.embedding_norm.weight",
        parent_norm_epsilon_config_key="norm_eps",
        parent_tied_tensor_key="model.embed_tokens.weight",
    )
    return args, {
        "legacy": legacy,
        "legacy_file_sha256": legacy_file_sha256,
    }


def _install_fakes(monkeypatch, args, fixture):
    calls = []
    cache = SimpleNamespace(
        cache_key=CACHE,
        parent_fingerprint=PARENT,
        layer_topology_fingerprint="b" * 64,
        manifest={"manifest_sha256": CACHE_MANIFEST},
    )
    bridge = SimpleNamespace(
        artifact_sha256=BRIDGE,
        profile_fingerprint=BRIDGE_PROFILE,
    )
    parent_norm = SimpleNamespace(result_sha256=FINAL_NORM)
    mapped_receipt = _seal(
        {
            "cache_key": CACHE,
            "cache_parent_fingerprint": PARENT,
            "bridge_artifact_sha256": BRIDGE,
            "parent_final_norm_load_sha256": FINAL_NORM,
        },
        "result_sha256",
    )
    source_report = _seal(
        {
            "cache_key": CACHE,
            "cache_manifest_sha256": CACHE_MANIFEST,
            "cache_parent_fingerprint": PARENT,
            "bridge_artifact_sha256": BRIDGE,
            "bridge_transport_profile_fingerprint": BRIDGE_PROFILE,
            "parent_final_norm_load_sha256": FINAL_NORM,
            "source_checkpoint_exactly_matches_authenticated_parent": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "network_access_performed": False,
        },
        "result_sha256",
    )
    source_table = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    companion = _companion(
        fixture["legacy"]["receipt_sha256"],
        fixture["legacy_file_sha256"],
    )

    def open_cache(path, *, verify):
        calls.append(("cache", path, verify))
        return cache

    def load_tokenizer(path):
        calls.append(("tokenizer", path))
        return SimpleNamespace(path=path)

    def load_bridge(
        path,
        *,
        donor_tokenizer,
        target_tokenizer,
        expected_artifact_sha256,
    ):
        calls.append(
            (
                "bridge",
                path,
                donor_tokenizer.path,
                target_tokenizer.path,
                expected_artifact_sha256,
            )
        )
        return bridge

    def load_norm(cache_value, path, *, tensor_key, epsilon_config_key):
        calls.append(
            ("final_norm", cache_value, path, tensor_key, epsilon_config_key)
        )
        return parent_norm

    def load_table(
        path,
        *,
        cache,
        bridge,
        authenticated_parent_final_norm,
        tensor_key,
        include_source_lineage_report,
    ):
        calls.append(
            (
                "mapped_table",
                path,
                cache,
                bridge,
                authenticated_parent_final_norm,
                tensor_key,
                include_source_lineage_report,
            )
        )
        return source_table, mapped_receipt, source_report

    def migrate(
        artifact_directory,
        table,
        *,
        expected_artifact_sha256,
        verify_runtime,
        legacy_foundation_receipt,
        legacy_foundation_receipt_file_sha256,
    ):
        calls.append(
            (
                "migrate",
                artifact_directory,
                table,
                expected_artifact_sha256,
                verify_runtime,
                legacy_foundation_receipt,
                legacy_foundation_receipt_file_sha256,
            )
        )
        return companion

    monkeypatch.setattr(cli, "open_lfm2_circuit_trace_cache", open_cache)
    monkeypatch.setattr(cli, "load_local_tokenizer", load_tokenizer)
    monkeypatch.setattr(cli, "load_strict_tokenizer_bridge_artifact", load_bridge)
    monkeypatch.setattr(
        cli,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        load_norm,
    )
    monkeypatch.setattr(
        cli,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        load_table,
    )
    monkeypatch.setattr(
        cli,
        "migrate_lfm2_semantic_foundation_interface_statistics_frontier",
        migrate,
    )
    return calls, companion


def _argv(args):
    return [
        "--legacy-statistics-artifact-directory",
        str(args.legacy_statistics_artifact_directory),
        "--expected-legacy-statistics-artifact-sha256",
        args.expected_legacy_statistics_artifact_sha256,
        "--semantic-cache-directory",
        str(args.semantic_cache_directory),
        "--expected-semantic-cache-key",
        args.expected_semantic_cache_key,
        "--bridge-directory",
        str(args.bridge_directory),
        "--expected-bridge-artifact-sha256",
        args.expected_bridge_artifact_sha256,
        "--donor-tokenizer-directory",
        str(args.donor_tokenizer_directory),
        "--target-tokenizer-directory",
        str(args.target_tokenizer_directory),
        "--parent-model-directory",
        str(args.parent_model_directory),
        "--legacy-foundation-receipt-json",
        str(args.legacy_foundation_receipt_json),
        "--expected-legacy-foundation-receipt-sha256",
        args.expected_legacy_foundation_receipt_sha256,
        "--expected-legacy-foundation-receipt-file-sha256",
        args.expected_legacy_foundation_receipt_file_sha256,
        "--output-companion-json",
        str(args.output_companion_json),
    ]


def test_successfully_writes_canonical_companion_and_prints_sealed_zero_ops(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    args, fixture = _args(tmp_path)
    calls, companion = _install_fakes(monkeypatch, args, fixture)

    assert cli.main(_argv(args)) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    report_body = dict(report)
    report_sha256 = report_body.pop("report_sha256")

    expected = canonical_json_bytes(dict(companion.companion_receipt)) + b"\n"
    assert args.output_companion_json.read_bytes() == expected
    assert report_sha256 == canonical_sha256(report_body)
    assert report["companion_canonical_sha256"] == companion.companion_receipt[
        "companion_sha256"
    ]
    assert report["companion_document_sha256"] == canonical_sha256(
        dict(companion.companion_receipt)
    )
    assert report["companion_file_sha256"] == hashlib.sha256(expected).hexdigest()
    assert set(report["operation_audit"].values()) == {0}
    assert set(companion.companion_receipt["operation_audit"].values()) == {0}
    assert report["authentication_audit"]["legacy_foundation_receipt_read_calls"] == 1
    assert report["numeric_metrics_have_final_intelligence_authority"] is False
    assert report["whole_response_review_selects_final_candidate"] is True
    assert captured.err == ""
    assert ": " not in captured.out
    assert sum(call[0] == "cache" for call in calls) == 1
    mapped = next(call for call in calls if call[0] == "mapped_table")
    assert mapped[-1] is True
    migration = next(call for call in calls if call[0] == "migrate")
    assert migration[3] == ARTIFACT
    assert migration[4] is True
    assert migration[-1] == fixture["legacy_file_sha256"]


@pytest.mark.parametrize(
    ("field", "message"),
    (
        (
            "expected_legacy_foundation_receipt_sha256",
            "canonical SHA-256 pin",
        ),
        (
            "expected_legacy_foundation_receipt_file_sha256",
            "file SHA-256 pin",
        ),
    ),
)
def test_rejects_wrong_legacy_receipt_pins_before_opening_dependencies(
    monkeypatch,
    tmp_path,
    field,
    message,
) -> None:
    args, fixture = _args(tmp_path)
    calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    setattr(args, field, "0" * 64)

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match=message,
    ):
        cli.run(args)

    assert calls == []
    assert not args.output_companion_json.exists()


def test_rejects_wrong_semantic_cache_pin(monkeypatch, tmp_path) -> None:
    args, fixture = _args(tmp_path)
    calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    args.expected_semantic_cache_key = "0" * 64

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="Semantic cache differs",
    ):
        cli.run(args)

    assert [call[0] for call in calls] == ["cache"]
    assert not args.output_companion_json.exists()


def test_rejects_wrong_bridge_pin_even_if_loader_self_reports_success(
    monkeypatch,
    tmp_path,
) -> None:
    args, fixture = _args(tmp_path)
    _calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    args.expected_bridge_artifact_sha256 = "0" * 64

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="Tokenizer bridge differs",
    ):
        cli.run(args)

    assert not args.output_companion_json.exists()


def test_rejects_wrong_statistics_artifact_pin_from_returned_companion(
    monkeypatch,
    tmp_path,
) -> None:
    args, fixture = _args(tmp_path)
    _calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    args.expected_legacy_statistics_artifact_sha256 = "0" * 64

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="pinned legacy bindings",
    ):
        cli.run(args)

    assert not args.output_companion_json.exists()


def test_existing_output_is_never_replaced(monkeypatch, tmp_path) -> None:
    args, fixture = _args(tmp_path)
    calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    args.output_companion_json.write_bytes(b"user-owned\n")

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="Refusing to replace",
    ):
        cli.run(args)

    assert args.output_companion_json.read_bytes() == b"user-owned\n"
    assert calls == []


def test_output_inside_authenticated_input_tree_is_rejected_before_asset_load(
    monkeypatch,
    tmp_path,
) -> None:
    args, fixture = _args(tmp_path)
    calls, _companion_value = _install_fakes(monkeypatch, args, fixture)
    args.output_companion_json = args.semantic_cache_directory / "companion.json"

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="disjoint from every authenticated input",
    ):
        cli.run(args)

    assert calls == []
    assert not args.output_companion_json.exists()


def test_legacy_foundation_must_match_live_cache_and_bridge_lineage(
    monkeypatch,
    tmp_path,
) -> None:
    args, fixture = _args(tmp_path)
    bad_solve = _seal(
        {
            "cache_parent_fingerprint": PARENT,
            "bridge_artifact_sha256": "0" * 64,
        },
        "result_sha256",
    )
    bad_body = {
        "schema": "fixture-legacy-foundation-v1",
        "parent_identity_sha256": PARENT,
        "cache_parent_fingerprint": PARENT,
        "bridge_artifact_sha256": "0" * 64,
        "interface_solve": bad_solve,
    }
    bad_receipt = _seal(bad_body, "receipt_sha256")
    bad_file_sha256 = _canonical_file(
        args.legacy_foundation_receipt_json,
        bad_receipt,
    )
    args.expected_legacy_foundation_receipt_sha256 = bad_receipt["receipt_sha256"]
    args.expected_legacy_foundation_receipt_file_sha256 = bad_file_sha256
    fixture["legacy"] = bad_receipt
    fixture["legacy_file_sha256"] = bad_file_sha256
    _calls, _companion_value = _install_fakes(monkeypatch, args, fixture)

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="differs from the legacy foundation",
    ):
        cli.run(args)

    assert not args.output_companion_json.exists()


@pytest.mark.parametrize("mode", ("noncanonical", "tampered"))
def test_persisted_companion_reader_rejects_noncanonical_or_tampered_bytes(
    tmp_path,
    mode,
) -> None:
    receipt_sha = "d" * 64
    receipt_file = "e" * 64
    companion = dict(_companion(receipt_sha, receipt_file).companion_receipt)
    path = tmp_path / "companion.json"
    if mode == "noncanonical":
        raw = (json.dumps(companion, indent=2, sort_keys=True) + "\n").encode("ascii")
    else:
        companion["frontier_sha256"] = "f" * 64
        raw = canonical_json_bytes(companion) + b"\n"
    path.write_bytes(raw)

    with pytest.raises(cli.Lfm2SemanticFoundationFrontierMigrationCliError):
        cli._read_pinned_canonical_json(
            path,
            expected_canonical_sha256=_companion(
                receipt_sha,
                receipt_file,
            ).companion_receipt["companion_sha256"],
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            seal_field="companion_sha256",
            name="persisted frontier companion",
        )


def test_tampered_migration_companion_is_rejected_before_publication(
    monkeypatch,
    tmp_path,
) -> None:
    args, fixture = _args(tmp_path)
    _calls, companion = _install_fakes(monkeypatch, args, fixture)
    tampered = dict(companion.companion_receipt)
    tampered["frontier_sha256"] = "f" * 64
    bad = Lfm2SemanticFoundationInterfaceFrontierCompanion(
        interface=None,
        anchor_fold=None,
        legacy_solve_report={},
        companion_receipt=tampered,
    )
    monkeypatch.setattr(
        cli,
        "migrate_lfm2_semantic_foundation_interface_statistics_frontier",
        lambda *args, **kwargs: bad,
    )

    with pytest.raises(
        cli.Lfm2SemanticFoundationFrontierMigrationCliError,
        match="seal is invalid",
    ):
        cli.run(args)

    assert not args.output_companion_json.exists()
