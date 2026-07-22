from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from experiments import run_lfm2_semantic_terminal_frontier as runner


class _Reference:
    def __init__(self, path: Path, candidate_ids: tuple[str, ...]) -> None:
        self.path = path
        self.receipt_sha256 = "1" * 64
        self.receipt_file_sha256 = "2" * 64
        self.manifest_sha256 = "3" * 64
        self.preterminal_capsule_receipt_sha256 = "4" * 64
        self.terminal_result_sha256 = "5" * 64
        self.source_roots_sha256 = "6" * 64
        self.candidate_ids = candidate_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "preterminal_capsule_receipt_sha256": (
                self.preterminal_capsule_receipt_sha256
            ),
            "terminal_result_sha256": self.terminal_result_sha256,
            "source_roots_sha256": self.source_roots_sha256,
            "candidate_ids": list(self.candidate_ids),
        }


def _source_roots(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    paths = {}
    for name in ("capsule", "cache", "bridge", "parent", "target"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    # The real local artifact commonly sources the donor tokenizer from the
    # parent model directory.  Shared read roots are valid and must not be
    # confused with a write/read collision.
    paths["donor"] = paths["parent"]
    role_paths = {
        "preterminal_capsule": [str(paths["capsule"])],
        "semantic_cache": [str(paths["cache"])],
        "tokenizer_bridge": [str(paths["bridge"])],
        "parent_model": [str(paths["parent"])],
        "donor_tokenizer": [str(paths["donor"])],
        "target_tokenizer": [str(paths["target"])],
    }
    body = {
        "schema": "neural-forge-lfm2-semantic-terminal-frontier-source-roots-v1",
        "role_paths": role_paths,
        "required_roles": list(role_paths),
        "role_count": len(role_paths),
        "canonical_root_count": len(role_paths),
        "all_roles_explicit_and_nonempty": True,
    }
    body["source_roots_sha256"] = runner.canonical_sha256(body)
    return body, paths


def test_complete_frontier_allows_shared_parent_and_donor_read_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_roots, paths = _source_roots(tmp_path)
    candidate_ids = ("writer/alpha=1", "writer:alpha=2", "writer alpha=3")
    bank_output = tmp_path / "bank"
    materializations = tmp_path / "children"
    receipt = tmp_path / "run.json"
    reference = _Reference(bank_output, candidate_ids)
    calls: list[str] = []

    monkeypatch.setattr(runner, "load_local_tokenizer", lambda _path: object())

    def compile_frontier(**_kwargs: object) -> SimpleNamespace:
        bank_output.mkdir()
        return SimpleNamespace(
            preterminal_capsule_reference=SimpleNamespace(
                as_dict=lambda: {"path": str(paths["capsule"])}
            ),
            bank_reference=reference,
            as_dict=lambda: {"candidate_ids": list(candidate_ids)},
        )

    monkeypatch.setattr(
        runner,
        "compile_lfm2_semantic_terminal_frontier_bank_from_capsule",
        compile_frontier,
    )
    monkeypatch.setattr(
        runner,
        "load_lfm2_semantic_terminal_frontier_bank",
        lambda *_args, **_kwargs: SimpleNamespace(
            reference=reference,
            manifest=MappingProxyType(
                {
                    "authenticated_source_roots": MappingProxyType(
                        {
                            **source_roots,
                            "role_paths": MappingProxyType(
                                {
                                    role: tuple(values)
                                    for role, values in source_roots[
                                        "role_paths"
                                    ].items()
                                }
                            ),
                        }
                    )
                }
            ),
        ),
    )

    def materialize(**kwargs: object) -> SimpleNamespace:
        candidate_id = str(kwargs["candidate_id"])
        destination = Path(kwargs["output_directory"])
        destination.mkdir()
        calls.append(candidate_id)
        return SimpleNamespace(
            as_dict=lambda: {
                "candidate_id": candidate_id,
                "path": str(destination),
                "checkpoint_path": str(destination / "checkpoint"),
            }
        )

    monkeypatch.setattr(
        runner,
        "materialize_lfm2_semantic_terminal_frontier_candidate",
        materialize,
    )

    document = runner.run_complete_frontier(
        preterminal_capsule_directory=paths["capsule"],
        expected_preterminal_receipt_file_sha256="a" * 64,
        expected_preterminal_receipt_sha256="b" * 64,
        expected_preterminal_bindings_sha256="c" * 64,
        cache_directory=paths["cache"],
        expected_cache_key="d" * 64,
        bridge_directory=paths["bridge"],
        expected_bridge_artifact_sha256="e" * 64,
        donor_tokenizer_directory=paths["donor"],
        parent_model_directory=paths["parent"],
        frontier_bank_directory=bank_output,
        materializations_directory=materializations,
        receipt_json=receipt,
        device="cpu",
    )

    assert calls == list(candidate_ids)
    assert paths["donor"] == paths["parent"]
    assert document["candidate_ids"] == list(candidate_ids)
    assert document["materialized_count"] == len(candidate_ids)
    assert document["all_retained_candidates_materialized"] is True
    assert document["candidate_metric_threshold_used_for_materialization"] is False
    assert document["whole_response_intelligence_review_is_authoritative"] is True
    assert document["operation_audit"] == runner._ZERO_OPERATION_AUDIT
    assert json.loads(receipt.read_text(encoding="utf-8")) == document
    progress = json.loads(
        (materializations / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["materialized_count"] == len(candidate_ids)
    child_names = sorted(
        path.name for path in materializations.iterdir() if path.is_dir()
    )
    assert len(child_names) == len(candidate_ids)
    assert len(set(child_names)) == len(candidate_ids)


def test_frontier_transaction_rejects_overlapping_outputs(tmp_path: Path) -> None:
    source_roots, paths = _source_roots(tmp_path)
    del source_roots
    with pytest.raises(runner.Lfm2SemanticTerminalFrontierRunnerError, match="overlap"):
        runner.run_complete_frontier(
            preterminal_capsule_directory=paths["capsule"],
            expected_preterminal_receipt_file_sha256="a" * 64,
            expected_preterminal_receipt_sha256="b" * 64,
            expected_preterminal_bindings_sha256="c" * 64,
            cache_directory=paths["cache"],
            expected_cache_key="d" * 64,
            bridge_directory=paths["bridge"],
            expected_bridge_artifact_sha256="e" * 64,
            donor_tokenizer_directory=paths["donor"],
            parent_model_directory=paths["parent"],
            frontier_bank_directory=tmp_path / "bank",
            materializations_directory=paths["cache"] / "children",
            receipt_json=tmp_path / "run.json",
            device="cpu",
        )
