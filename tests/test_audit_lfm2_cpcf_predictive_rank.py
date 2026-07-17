from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.audit_lfm2_cpcf_predictive_rank import (  # noqa: E402
    ARTIFACT_SCHEMA,
    Lfm2CpcfPredictiveRankAuditCliError,
    _candidate_widths,
    _publish_artifact,
    _sealed_artifact,
    _verify_artifact,
)


def test_sealed_artifact_is_canonical_and_detects_mutation() -> None:
    artifact = _sealed_artifact(
        {
            "schema": ARTIFACT_SCHEMA,
            "audit": {"source_width": 512, "layer_count": 14},
        }
    )

    _verify_artifact(artifact)
    assert len(artifact["artifact_sha256"]) == 64
    artifact["audit"]["source_width"] = 304
    with pytest.raises(
        Lfm2CpcfPredictiveRankAuditCliError,
        match="does not match",
    ):
        _verify_artifact(artifact)


def test_publish_artifact_is_atomic_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "report.json"
    artifact = _sealed_artifact(
        {
            "schema": ARTIFACT_SCHEMA,
            "audit": {"report_sha256": "a" * 64},
        }
    )

    first_sha = _publish_artifact(destination, artifact)
    second_sha = _publish_artifact(destination, artifact)

    assert first_sha == second_sha
    assert json.loads(destination.read_text(encoding="ascii")) == artifact
    assert not tuple(destination.parent.glob(".report.json.stage-*"))

    different = _sealed_artifact(
        {
            "schema": ARTIFACT_SCHEMA,
            "audit": {"report_sha256": "b" * 64},
        }
    )
    with pytest.raises(
        Lfm2CpcfPredictiveRankAuditCliError,
        match="Refusing to replace",
    ):
        _publish_artifact(destination, different)
    assert json.loads(destination.read_text(encoding="ascii")) == artifact


def test_candidate_widths_are_positive_unique_and_ordered() -> None:
    assert _candidate_widths((304, 128, 64, 32, 8)) == (
        304,
        128,
        64,
        32,
        8,
    )
    for invalid in ((), (0,), (32, 32), (True,)):
        with pytest.raises(
            Lfm2CpcfPredictiveRankAuditCliError,
            match="unique positive",
        ):
            _candidate_widths(invalid)


def test_predictive_rank_measurement_can_record_negative_heldout_transfer() -> (
    None
):
    torch = pytest.importorskip("torch")
    from neural_forge.hf_closed_form_causal_circuit_fold import (
        CausalFeatureBlock,
        fit_predictive_boundary_map,
    )

    generator = torch.Generator().manual_seed(11)
    fit_states = torch.randn(24, 4, generator=generator, dtype=torch.float64)
    validation_states = torch.randn(
        12,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    fit_target = fit_states[:, :2].clone()
    # The deliberately inverted held-out relation has negative retained
    # energy. A measurement-only audit must serialize it, while compilation
    # still uses the function's strict default and fails closed.
    validation_target = -validation_states[:, :2].clone()
    measured = fit_predictive_boundary_map(
        fit_states,
        (
            CausalFeatureBlock(
                name="action",
                role="test_action",
                values=fit_target,
            ),
        ),
        target_width=4,
        validation_states=validation_states,
        validation_features=(
            CausalFeatureBlock(
                name="action",
                role="test_action",
                values=validation_target,
            ),
        ),
        minimum_validation_block_retention=0.0,
        enforce_validation_block_floor=False,
    )

    assert measured.report["validation_block_floor_enforced"] is False
    assert (
        measured.report["minimum_observed_validation_block_retention"] < 0.0
    )
