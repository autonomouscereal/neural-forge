"""Pin the real-scale cross-family runner's seal schema.

The run itself is hours of CPU and is published as
``reports/cross-family-real-20260804``; nothing here executes it.  What is
pinned here is the part a later reader has to trust without re-running
anything: the receipt seal is a canonical digest of its own body, a tampered
receipt is detected, the stage lineage is the declared one, the configuration
identity every later stage chains off is reproducible, and the report-level
verification refuses a report whose bytes moved underneath it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("transformers")

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "run_cross_family_real_scale_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_cross_family_real_scale_fold", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


def _write(path: Path, body: dict) -> dict:
    sealed = MODULE.seal_receipt(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(sealed, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    return sealed


def test_the_runner_declares_its_schema_and_stage_lineage() -> None:
    assert MODULE.RUNNER_SCHEMA == "neural-forge-cross-family-real-scale-fold-runner-v1"
    assert MODULE.INTERFACE_SCHEMA == (
        "neural-forge-chart-projected-pair-code-interface-v1"
    )
    assert MODULE.STAGES == (
        "configuration",
        "tokenizers",
        "evidence",
        "charts",
        "fold",
        "behavior",
    )
    # The interface operator claims the method the planner selects for an
    # independent vocabulary, and nothing else.
    assert MODULE.ChartProjectedPairCodeTokenHeadInterfaceOperator.method_id == (
        "token_interface_bridge"
    )
    assert MODULE.MASKED_ROW_POLICY == (
        "leave-uncovered-target-rows-at-incumbent-blank"
    )
    # The published geometry is the one stage 2 measured, named so a
    # synthesized-parent run can never be read as the pretrained one.
    smollm2 = dict(MODULE.GEOMETRIES["smollm2-135m"])
    assert smollm2["hidden_size"] == 576
    assert smollm2["num_hidden_layers"] == 30
    assert smollm2["num_attention_heads"] == 9
    assert smollm2["num_key_value_heads"] == 3
    assert smollm2["head_dim"] == 64
    assert smollm2["intermediate_size"] == 1536


def test_a_receipt_seal_is_the_canonical_digest_of_its_own_body() -> None:
    body = {"schema": MODULE.RUNNER_SCHEMA, "stage": "fold", "value": [3, 1, 2]}

    sealed = MODULE.seal_receipt(body)

    assert sealed["receipt_sha256"] == canonical_sha256(body)
    assert MODULE.verify_receipt(sealed)
    # Sealing is idempotent: re-sealing an already sealed receipt reproduces it.
    assert MODULE.seal_receipt(sealed) == sealed


@pytest.mark.parametrize(
    "mutation",
    (
        {"value": [3, 1, 999]},
        {"stage": "behavior"},
        {"added": True},
    ),
)
def test_a_tampered_receipt_body_fails_its_seal(mutation: dict) -> None:
    sealed = MODULE.seal_receipt(
        {"schema": MODULE.RUNNER_SCHEMA, "stage": "fold", "value": [3, 1, 2]}
    )

    assert not MODULE.verify_receipt({**sealed, **mutation})


def test_reading_a_broken_receipt_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    sealed = _write(path, {"schema": MODULE.RUNNER_SCHEMA, "stage": "fold"})
    assert MODULE._read_receipt(path) == sealed

    path.write_text(
        json.dumps({**sealed, "stage": "behavior"}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.RealScaleFoldError, match="Receipt seal is broken"):
        MODULE._read_receipt(path)


def test_report_verification_covers_every_manifest_file(tmp_path: Path) -> None:
    root = tmp_path / "report"
    _write(root / "configuration.json", {"schema": MODULE.RUNNER_SCHEMA, "stage": "c"})
    _write(root / "fold.json", {"schema": MODULE.RUNNER_SCHEMA, "stage": "fold"})
    (root / "child").mkdir(parents=True, exist_ok=True)
    (root / "child" / "model.safetensors").write_bytes(b"weights")
    _write(
        root / "manifest.json",
        {
            "schema": MODULE.RUNNER_SCHEMA,
            "stage": "manifest",
            "files": MODULE._tree_manifest(root, skip=("manifest.json",)),
        },
    )

    result = MODULE.verify_report(root)

    assert result["verified"] is True
    assert result["problems"] == []
    assert result["files"] == 3

    # A silently rewritten model shard is a digest change, not a missing file.
    (root / "child" / "model.safetensors").write_bytes(b"other!!")
    changed = MODULE.verify_report(root)
    assert changed["verified"] is False
    assert changed["problems"] == ["digest changed: child/model.safetensors"]

    (root / "child" / "model.safetensors").unlink()
    missing = MODULE.verify_report(root)
    assert missing["problems"] == ["missing: child/model.safetensors"]


def test_verification_refuses_a_report_without_a_sealed_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(MODULE.RealScaleFoldError, match="No sealed manifest"):
        MODULE.verify_report(tmp_path)


def _configuration(tmp_path: Path, **overrides: object) -> dict:
    """Seal one configuration stage without touching a model or a tokenizer."""

    argv = [
        "--report-root",
        str(tmp_path),
        "--parent-geometry",
        "smollm2-135m",
        "--parent-residual-rank",
        "432",
        "--child-hidden-size",
        "432",
        "--child-intermediate-size",
        "1152",
        "--child-query-heads",
        "6",
        "--child-key-value-heads",
        "3",
        "--child-vocab-size",
        "16384",
        "--calibration-prompts-count",
        "32",
        "--holdout-prompts-count",
        "16",
        "--evidence-window",
        "88",
        "--holdout-window",
        "64",
        "--decode-new-tokens",
        "48",
    ]
    for name, value in overrides.items():
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    arguments = MODULE.build_parser().parse_args(argv)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return MODULE._configuration_stage(arguments, tmp_path)["receipt"]


def test_the_configuration_identity_excludes_the_clock_and_the_thread_count(
    tmp_path: Path,
) -> None:
    """The identity every later stage chains off has to be reproducible.

    ``generated_utc``, the library versions and the thread count are recorded
    in the configuration receipt but must stay out of ``input_sha256``.  If any
    of them leaked in, two runs of the same configuration would produce
    different stage identities, and the driver would recompute the whole fold
    on every restart while still reporting itself as resumable.
    """

    first = _configuration(tmp_path / "a")
    second = _configuration(tmp_path / "b")

    assert first["input_sha256"] == second["input_sha256"]
    volatile = {
        "generated_utc",
        "torch_version",
        "transformers_version",
        "torch_threads",
        "device",
        "input_sha256",
        "receipt_sha256",
    }
    identity = {key: value for key, value in first.items() if key not in volatile}
    assert set(identity) == {
        "schema",
        "stage",
        "parent",
        "child",
        "tokenizers",
        "corpus",
        "fold",
    }
    assert first["input_sha256"] == canonical_sha256(identity)
    # A different crossing is a different identity, so nothing resumes across
    # one.
    assert _configuration(tmp_path / "c", child_hidden_size=480)["input_sha256"] != (
        first["input_sha256"]
    )


def test_the_configuration_seal_states_the_crossing_and_the_provenance(
    tmp_path: Path,
) -> None:
    receipt = _configuration(tmp_path)
    parent = receipt["parent"]
    child = receipt["child"]

    # A synthesized parent can never be read as a pretrained one.
    assert parent["weight_source"] == "seeded-random-init"
    assert parent["pretrained"] is False
    assert parent["geometry_name"] == "smollm2-135m"
    assert parent["residual_confinement_rank"] == 432
    assert parent["folded_layer_prefix"] == 30
    assert parent["bounded_layer_prefix"] is False
    # Every crossing is declared rather than inferred by a later reader.
    assert child["family"] == "qwen3"
    assert child["query_key_norm"] == "per_head_rms_norm"
    assert child["tie_word_embeddings"] is False
    assert child["hidden_size"] < parent["geometry"]["hidden_size"]
    assert child["num_attention_heads"] < parent["geometry"]["num_attention_heads"]
    assert child["intermediate_size"] < parent["geometry"]["intermediate_size"]
    assert receipt["tokenizers"]["bridge_kind"] == (
        "sealed-independent-tokenizer-pair-bridge"
    )
    assert receipt["fold"]["enabled_experimental_method_ids"] == [
        "causal_gqa_rope_residual_attention_layer",
        "causal_swiglu_residual_mlp_layer",
    ]
    # The corpus is named by content, and the held-out split is disjoint.
    assert len(receipt["corpus"]["file_sha256"]) == 64
    assert not set(receipt["corpus"]["calibration_prompt_indices"]) & set(
        receipt["corpus"]["holdout_prompt_indices"]
    )


def test_a_bounded_layer_prefix_says_so_in_the_seal(tmp_path: Path) -> None:
    receipt = _configuration(tmp_path, layers=12)

    assert receipt["parent"]["folded_layer_prefix"] == 12
    assert receipt["parent"]["full_depth"] == 30
    assert receipt["parent"]["bounded_layer_prefix"] is True


def test_the_configuration_stage_refuses_a_holdout_it_also_calibrated_on(
    tmp_path: Path,
) -> None:
    with pytest.raises(MODULE.RealScaleFoldError, match="overlap"):
        _configuration(tmp_path, calibration_prompts_count=64)


def test_the_configuration_stage_refuses_a_depth_the_parent_does_not_have(
    tmp_path: Path,
) -> None:
    with pytest.raises(MODULE.RealScaleFoldError, match=r"must fall in 1\.\.30"):
        _configuration(tmp_path, layers=31)


def test_the_named_geometry_carries_the_charted_rotary_base(tmp_path: Path) -> None:
    """RoPE base is geometry: SmolLM2-135M is 100000, not the 10000 default."""

    del tmp_path
    assert MODULE.GEOMETRIES["smollm2-135m"]["rope_theta_kilo"] == 100
    parent = MODULE._build_parent(
        geometry=dict(MODULE.GEOMETRIES["smollm2-135m"]),
        vocabulary_size=64,
        layers=1,
        seed=1,
    )
    assert float(dict(parent.config.rope_parameters)["rope_theta"]) == 100000.0
    assert int(parent.config.hidden_size) == 576
    assert int(parent.config.num_key_value_heads) == 3


def test_the_summary_helper_reports_an_empty_series_as_absent() -> None:
    assert MODULE._summarize([]) == {
        "count": 0,
        "min": None,
        "median": None,
        "max": None,
    }
    assert MODULE._summarize([1.0, 3.0]) == {
        "count": 2,
        "min": 1.0,
        "median": 2.0,
        "max": 3.0,
    }
    assert MODULE._summarize([5.0, 1.0, 3.0])["median"] == 3.0
