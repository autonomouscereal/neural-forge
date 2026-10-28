"""CLI contract tests for the canonical component-method cell runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "run_lfm2_component_method_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_lfm2_component_method_matrix_for_test",
        _MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module()

from neural_forge.hf_lfm2_component_method_matrix import (  # noqa: E402
    Lfm2ComponentMethodMatrixError,
    discover_matrix_kernel_providers,
    full_component_matrix_cells,
    load_matrix_kernel_provider,
    matrix_campaign_pair_coverage,
)


def test_print_wave_one_is_canonical_and_unique(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--print-wave-one"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["cell_ids"]) == len(set(payload["cell_ids"])) == 32
    assert "C13.T5.M16.L05" in payload["cell_ids"]
    assert "C06+C07.T7.M16.L02" in payload["cell_ids"]


def test_run_arguments_fail_closed_when_incomplete(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--request, --executor, and --out"):
        runner.main(["--request", str(tmp_path / "request.json")])


def test_kernel_provider_inventory_is_authenticated_and_non_authoritative() -> None:
    providers = discover_matrix_kernel_providers()
    assert set(providers) == {"T4", "T5", "T7", "T8"}
    assert all(len(provider.module_sha256) == 64 for provider in providers.values())
    assert all(
        provider.as_dict()["kernel_only_no_compaction_claim_authority"] is True
        for provider in providers.values()
    )
    assert providers["T7"].physical_component_ids(("C06",)) == ("C06", "C07")
    assert providers["T7"].physical_component_ids(("C06", "C07")) == (
        "C06",
        "C07",
    )


def test_provider_applicability_fails_closed_before_kernel_math() -> None:
    provider = load_matrix_kernel_provider("T4")
    with pytest.raises(Lfm2ComponentMethodMatrixError, match="does not support"):
        provider.invoke(
            measurement_component_ids=("C11",),
            level_id="M16",
            payload={},
        )
    with pytest.raises(Lfm2ComponentMethodMatrixError, match="No kernel provider"):
        load_matrix_kernel_provider("T6")


def test_full_campaign_accounts_for_every_technique_component_pair() -> None:
    pairs = matrix_campaign_pair_coverage()
    identities = {
        (row["technique_id"], row["measurement_component_ids"][0]) for row in pairs
    }
    assert len(pairs) == len(identities) == 9 * 15
    assert all(row["individual_measurement_scope"] is True for row in pairs)
    assert all(len(row["measurement_component_ids"]) == 1 for row in pairs)
    assert any(row["implementation_state"] == "kernel_missing" for row in pairs)
    assert any(row["implementation_state"] == "not_applicable" for row in pairs)

    q_relation = next(
        row
        for row in pairs
        if row["technique_id"] == "T7"
        and row["measurement_component_ids"] == ["C06"]
    )
    assert q_relation["physical_component_ids"] == ["C06", "C07"]
    assert q_relation["implementation_state"] == "canonical_kernel_available"

    norm_route = next(
        row
        for row in pairs
        if row["technique_id"] == "T5"
        and row["measurement_component_ids"] == ["C14"]
    )
    assert norm_route["designed_applicable"] is False
    assert norm_route["levels"] == ["NA"]


def test_full_campaign_cells_are_individual_unique_and_staged() -> None:
    cells = full_component_matrix_cells()
    ids = [row["cell_id"] for row in cells]
    assert len(ids) == len(set(ids))
    assert all("+" not in cell_id for cell_id in ids)
    assert all(row["evidence_class"] == "isolation_only" for row in cells)
    assert all(row["numeric_metrics_are_diagnostic_only"] is True for row in cells)
    assert all(row["manual_intelligence_review_required"] is True for row in cells)
    assert "C13.T5.M16.L00" in ids
    assert "C13.T5.X4.L13" in ids
    assert "C06.T7.M16.L02" in ids
    assert "C07.T7.M16.L02" in ids
    assert "C14.T8.NA.FINAL" in ids
