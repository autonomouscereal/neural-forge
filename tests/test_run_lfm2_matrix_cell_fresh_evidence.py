"""The fresh-evidence split authority and cell 0b's second calibration reading.

`NOTES-MX-A.md` §5.3 left the campaign with no readable evidence base: the
sealed parent trace cache the design names carries an implementation identity
the current collector's compatibility policy does not admit, and the policy is
the audit gate, so it is not extended here.  The coordinator's decision is the
w12e precedent -- recapture under the current identity -- which produces a
cache with no balanced-basis phase bound to it.  These tests pin the two
surfaces that decision adds to the cell runner:

* `_authenticated_split_manifest`, the *second* split authority, which has to
  enforce exactly the invariants `authoritative_split_by_batch` enforces or the
  campaign's held-out numbers stop being comparable with the sealed ledger; and
* `_a9_calibration`'s `behavioral-agreement` mode, which is the addendum's
  reading of cell 0b's gate and must keep the artifact comparison visible
  rather than dropping it.

Every test here is a refusal test except the two that prove the happy path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# The runner registers the A9 route into the process-wide technique table at
# import time, so importing it twice in one pytest session is a duplicate
# registration.  Reuse whatever another test module already loaded.
if "run_lfm2_matrix_cell" in sys.modules:
    runner = sys.modules["run_lfm2_matrix_cell"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "run_lfm2_matrix_cell", _ROOT / "experiments" / "run_lfm2_matrix_cell.py"
    )
    runner = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = runner
    _SPEC.loader.exec_module(runner)

from neural_forge.matrix_technique_registry import CellTechniqueResult  # noqa: E402

CACHE_KEY = "a" * 64
PARENT = "b" * 64
RECORDS = [hashlib.sha256(f"batch-{index}".encode("ascii")).hexdigest() for index in range(4)]


class _Reference:
    def __init__(self, input_sha256: str) -> None:
        self.input_sha256 = input_sha256


class _Cache:
    cache_key = CACHE_KEY
    parent_fingerprint = PARENT


def _seal(body: dict) -> dict:
    body = {key: value for key, value in body.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return body


def _manifest(**overrides) -> dict:
    rows = [
        {
            "batch_index": index,
            "record_id": RECORDS[index],
            "split": "heldout" if index == 3 else "fit",
        }
        for index in range(4)
    ]
    body = {
        "schema": runner.SPLIT_MANIFEST_SCHEMA,
        "trace_cache_key": CACHE_KEY,
        "parent_identity_sha256": PARENT,
        "split": {
            "method": "whole-record-seeded-sha256-ranking",
            "fit_batches": 3,
            "heldout_batches": 1,
            "rows": rows,
        },
    }
    body.update(overrides)
    return _seal(body)


def _write(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "evidence-manifest.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _references() -> list[_Reference]:
    return [_Reference(value) for value in RECORDS]


def test_a_well_formed_split_manifest_binds(tmp_path):
    path = _write(tmp_path, _manifest())
    body = runner._authenticated_split_manifest(
        path, trace_cache=_Cache(), references=_references()
    )
    assert body["split"]["fit_batches"] == 3
    assert [row["split"] for row in body["split"]["rows"]] == [
        "fit",
        "fit",
        "fit",
        "heldout",
    ]


def test_a_broken_seal_is_refused(tmp_path):
    body = _manifest()
    # One field edited after sealing: the digest no longer covers the content,
    # which is the whole reason the digest is checked before anything else.
    body["split"]["rows"][3]["split"] = "fit"
    path = _write(tmp_path, body)
    with pytest.raises(runner.MatrixCellRunError, match="seal is broken"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_a_manifest_for_another_cache_is_refused(tmp_path):
    path = _write(tmp_path, _manifest(trace_cache_key="c" * 64))
    with pytest.raises(runner.MatrixCellRunError, match="names trace cache"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_a_manifest_for_another_parent_is_refused(tmp_path):
    path = _write(tmp_path, _manifest(parent_identity_sha256="d" * 64))
    with pytest.raises(runner.MatrixCellRunError, match="different parent"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_a_record_id_that_is_not_the_batch_reference_is_refused(tmp_path):
    body = _manifest()
    body["split"]["rows"][1]["record_id"] = "e" * 64
    path = _write(tmp_path, _seal(body))
    with pytest.raises(runner.MatrixCellRunError, match="differ from the trace"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_a_split_missing_one_half_is_refused(tmp_path):
    body = _manifest()
    for row in body["split"]["rows"]:
        row["split"] = "fit"
    path = _write(tmp_path, _seal(body))
    with pytest.raises(runner.MatrixCellRunError, match="both evidence halves"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_a_short_split_is_refused(tmp_path):
    """Fewer rows than batches means some batch has no declared split."""

    body = _manifest()
    body["split"]["rows"] = body["split"]["rows"][:3]
    path = _write(tmp_path, _seal(body))
    with pytest.raises(runner.MatrixCellRunError, match="complete per-batch split"):
        runner._authenticated_split_manifest(
            path, trace_cache=_Cache(), references=_references()
        )


def test_two_split_authorities_are_refused(tmp_path):
    """One authority or the other; never both, because they can disagree."""

    with pytest.raises(runner.MatrixCellRunError, match="exactly one split authority"):
        runner.load_site_evidence(
            trace_cache_path=tmp_path / "cache",
            layer_index=0,
            endpoints=("mlp.input",),
            forced_decode_count=8,
            fit_batch_limit=None,
            heldout_batch_limit=None,
            cache_directory=None,
            basis_phase_cache_root=tmp_path / "phase",
            basis_phase_key_sha256="f" * 64,
            basis_phase_hmac_key_file=tmp_path / "key",
            basis_phase_hmac_key_id="id",
            evidence_split_manifest=tmp_path / "evidence-manifest.json",
            target_hidden_size=816,
        )


# --------------------------------------------------------------------------
# cell 0b's second reading
# --------------------------------------------------------------------------


SEALED_A9 = {
    "sparse_semantic_codebook_artifact_sha256": "4439a2b7" + "0" * 56,
    "base_codebook_artifact_sha256": "4439a2b7" + "0" * 56,
    "codebook_phase_key_sha256": "5b5557b8" + "0" * 56,
    "codebook_phase_completion_sha256": "7dab1207" + "0" * 56,
    "candidate_support_math_abi_sha256": "8cb1feff" + "0" * 56,
    "coefficient_solve_math_abi_sha256": "48bbbf46" + "0" * 56,
    "cached_hidden_top1_match_count": 551,
    "cached_hidden_top5_match_count": 589,
    "cached_hidden_top10_match_count": 605,
    "cached_hidden_row_count": 648,
}


def _route_result(directory: Path, *, digest: str = "82c497f4" + "0" * 56):
    return CellTechniqueResult(
        installed_weights={},
        stored_parameter_count=1,
        solve={
            "solver_path": "hf_lfm2_sparse_semantic_codebook.residual_omp",
            "ridge_relative": 0.0,
            "condition_number": 1.0,
            "condition_cap_saturated": False,
            "prior_norm": 0.0,
            "correction_norm": 0.0,
            "requested_clamp_ratio": 0.0,
            "allowed_clamp_ratio": float("inf"),
            "clamp_engaged": False,
            "omp": {"k": 32, "support_reuse_prefix_steps": 0, "double_mgs": True},
            "cached_hidden_top1_match_count": 540,
            "cached_hidden_top5_match_count": 580,
            "cached_hidden_top10_match_count": 600,
            "cached_hidden_row_count": 648,
        },
        predict=None,
        artifact_identities={
            "sparse_semantic_codebook_artifact_sha256": digest,
            "base_codebook_artifact_sha256": digest,
            "codebook_phase_key_sha256": "8c814244" + "0" * 56,
            "codebook_phase_completion_sha256": "11111111" + "0" * 56,
            "candidate_support_math_abi_sha256": "22222222" + "0" * 56,
            "coefficient_solve_math_abi_sha256": "33333333" + "0" * 56,
            "result_sha256": "44444444" + "0" * 56,
            "whole_response_dossier_sha256": "55555555" + "0" * 56,
        },
        child_directory=directory,
        notes="",
    )


def test_the_original_reading_still_fails_on_a_different_artifact(tmp_path):
    """§5.2's registration is untouched and still the default."""

    (tmp_path / "whole-response-dossier.json").write_text("{}", encoding="utf-8")
    calibration = runner._a9_calibration(_route_result(tmp_path), SEALED_A9)
    assert calibration["mode"] == "artifact-bit-identity"
    assert calibration["passed"] is False
    assert calibration["artifact_bit_identical"] is False


def test_the_behavioral_reading_passes_a_complete_route_and_keeps_the_comparison(
    tmp_path,
):
    """The addendum's gate is completeness; the digests stay on the record."""

    (tmp_path / "whole-response-dossier.json").write_text("{}", encoding="utf-8")
    calibration = runner._a9_calibration(
        _route_result(tmp_path), SEALED_A9, mode="behavioral-agreement"
    )
    assert calibration["passed"] is True
    # The artifact comparison is not dropped -- it is the evidence of what
    # moved, and a reader has to be able to see that it did move.
    assert calibration["artifact_bit_identical"] is False
    assert calibration["artifact_comparison_is_diagnostic"] is True
    assert any(
        entry["field"] == "sparse_semantic_codebook_artifact_sha256"
        and entry["matched"] is False
        for entry in calibration["checks"]
    )


def test_the_behavioral_reading_refuses_a_route_with_no_dossier(tmp_path):
    """No gradeable text means the cell is void, not evidence."""

    calibration = runner._a9_calibration(
        _route_result(tmp_path), SEALED_A9, mode="behavioral-agreement"
    )
    assert calibration["route_dossier_present"] is False
    assert calibration["passed"] is False


def test_an_unknown_calibration_mode_is_refused(tmp_path):
    with pytest.raises(runner.MatrixCellRunError, match="route_calibration_mode"):
        runner._a9_calibration(_route_result(tmp_path), SEALED_A9, mode="whatever")


def test_a_completed_route_solve_block_is_sealable(tmp_path):
    """The regression that cost a full route compile.

    `_a9_token_route` used to report `allowed_clamp_ratio: float("inf")`.  The
    receipt sealer canonicalizes with `allow_nan=False`, so the *first* route
    that ever ran to completion raised "IR payload is not canonical JSON" at the
    seal -- after the entire 4,096-row OMP compile and all sixteen whole-response
    generations.  Every field a route puts in the receipt has to survive the
    sealer, so this walks the real blank receipt through the real seal.
    """

    from neural_forge.matrix_cell_receipt import (
        blank_receipt,
        seal_receipt,
        verify_seal,
    )

    run_receipt = tmp_path / "run-receipt.json"
    run_receipt.write_text(
        json.dumps(
            {
                "active_k": 32,
                "cached_hidden_diagnostic": {
                    "cached_hidden_top1_match_count": 5286,
                    "cached_hidden_row_count": 6335,
                    "cached_hidden_mean_reciprocal_rank": 0.87,
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = blank_receipt(
        cell_id="CAL.C01.T5.X4",
        cell_instance_id="0" * 32,
        campaign_wave=1,
        repo_head="a31747d8",
        src_tree_sha="c" * 64,
        probe_set_sha="d" * 64,
        smoke_prompt_sha="e" * 64,
        compute_architecture="sm86 NVIDIA GeForce RTX 3090",
    )
    solve = dict(receipt["solve"])
    # The production block, not a fixture that could drift away from it.
    solve.update(runner.route_solve_block(json.loads(run_receipt.read_text()), run_receipt))
    receipt["solve"] = solve
    receipt["calibration"] = {
        "copy_exact_reproduction_relL2": 0.0,
        "passed": True,
        "applicable": True,
    }
    receipt["behavior"] = {
        "free_run": {
            "probe_prompts": 16,
            "tokens_per_prompt": 0,
            "token_match_vs_control": 1.0,
            "mean_lcp": 1.0,
        },
        "decode_smoke": [
            {"prompt_sha": "f" * 64, "output_verbatim": "text", "terminated": "eos"}
        ],
    }
    sealed = seal_receipt(receipt)
    verify_seal(sealed)
    assert sealed["outcome"]["status"] == "pending-review"
    assert sealed["outcome"]["first_divergent_metric"] is None
