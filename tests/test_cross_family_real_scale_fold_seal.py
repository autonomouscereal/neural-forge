"""Pin the real-scale cross-family runner's seal schema.

The run itself is hours of CPU and is published as
``reports/cross-family-real-20260804``; nothing here executes it.  What is
pinned here is the part a later reader has to trust without re-running
anything: the receipt seal is a canonical digest of its own body, a tampered
receipt is detected, the stage lineage is the declared one, the configuration
identity every later stage chains off is reproducible, and the report-level
verification refuses a report whose bytes moved underneath it.

The second half pins the ``local-pretrained`` parent mode added for
``reports/cross-family-real-20260804-pretrained``: a checkpoint's declared
geometry is verified field by field before any compute, its weight digest is
part of the run identity, every argument combination whose receipt would claim
something false is refused, and -- the invariant that keeps the committed
synthesized runs reproducible -- none of it changes the identity a
``seeded-random-init`` run seals.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
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


# ---------------------------------------------------------------------------
# The local-pretrained parent mode
# ---------------------------------------------------------------------------


#: SmolLM2-135M's real ``config.json``, verbatim in the fields that matter.
#: ``head_dim`` is deliberately absent, exactly as it is in the checkpoint, so
#: the derivation from ``hidden_size // num_attention_heads`` stays covered.
CHECKPOINT_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000,
    "max_position_embeddings": 8192,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}


def _checkpoint(tmp_path: Path, **overrides: object) -> Path:
    """A checkpoint directory with real metadata and a stand-in weight file.

    Nothing here loads the weights: the configuration stage only reads the
    config and digests the file, which is the whole point of doing the geometry
    check before an hour of compute rather than after it.
    """

    directory = tmp_path / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    body = {**CHECKPOINT_CONFIG, **overrides}
    (directory / "config.json").write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / "model.safetensors").write_bytes(b"stand-in weights")
    return directory


def _pretrained_configuration(
    tmp_path: Path,
    directory: Path,
    **overrides: object,
) -> dict:
    argv = [
        "--report-root",
        str(tmp_path),
        "--parent-geometry",
        "smollm2-135m",
        "--parent-weights",
        "local-pretrained",
        "--parent-model-directory",
        str(directory),
        "--layers",
        "6",
        "--calibration-prompts-count",
        "48",
        "--holdout-prompts-count",
        "16",
    ]
    for name, value in overrides.items():
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    arguments = MODULE.build_parser().parse_args(argv)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return MODULE._configuration_stage(arguments, tmp_path)["receipt"]


def test_the_runner_declares_both_parent_weight_sources() -> None:
    assert MODULE.PARENT_WEIGHT_SOURCES == ("seeded-random-init", "local-pretrained")
    assert MODULE.DEFAULT_SOURCE_TOKENIZER == "sshleifer/tiny-gpt2"
    # The geometry name is only a claim until these fields are checked.
    assert MODULE.BOUND_PARENT_CONFIG_FIELDS == (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
    )


def test_a_pretrained_run_seals_the_checkpoint_digest_and_config_identity(
    tmp_path: Path,
) -> None:
    directory = _checkpoint(tmp_path)

    receipt = _pretrained_configuration(tmp_path / "run", directory)

    parent = receipt["parent"]
    assert parent["weight_source"] == "local-pretrained"
    assert parent["pretrained"] is True
    checkpoint = parent["pretrained_checkpoint"]
    assert checkpoint["weight_file"] == "model.safetensors"
    assert len(checkpoint["weight_file_sha256"]) == 64
    assert len(checkpoint["config_file_sha256"]) == 64
    assert checkpoint["config_vocab_size"] == 49152
    assert checkpoint["config_tie_word_embeddings"] is True
    # ``head_dim`` is not in the checkpoint config and is resolved, not skipped.
    assert checkpoint["verified_config_fields"] == {
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "head_dim": 64,
        "rms_norm_eps": 1e-05,
        "rope_theta": 100000.0,
    }
    # The digest is part of the identity, so two checkpoints are two runs.
    assert checkpoint["weight_file_sha256"] in json.dumps(receipt)
    (directory / "model.safetensors").write_bytes(b"a different checkpoint")
    moved = _pretrained_configuration(tmp_path / "run2", directory)
    assert moved["input_sha256"] != receipt["input_sha256"]


@pytest.mark.parametrize(
    ("override", "expected"),
    (
        ({"hidden_size": 512}, "hidden_size: declared 576, checkpoint 512"),
        (
            {"num_key_value_heads": 9},
            "num_key_value_heads: declared 3, checkpoint 9",
        ),
        ({"num_hidden_layers": 24}, "num_hidden_layers: declared 30, checkpoint 24"),
        ({"rope_theta": 10000}, "rope_theta: declared 100000.0, checkpoint 10000.0"),
        ({"intermediate_size": 2048}, "intermediate_size: declared 1536"),
        ({"num_attention_heads": 12}, "head_dim: declared 64, checkpoint 48"),
    ),
)
def test_a_checkpoint_that_disagrees_with_the_declared_geometry_is_refused(
    tmp_path: Path,
    override: dict,
    expected: str,
) -> None:
    directory = _checkpoint(tmp_path, **override)

    with pytest.raises(MODULE.RealScaleFoldError) as failure:
        _pretrained_configuration(tmp_path / "run", directory)

    assert "does not match the declared parent geometry" in str(failure.value)
    assert expected in str(failure.value)


def test_a_checkpoint_from_another_family_is_refused(tmp_path: Path) -> None:
    directory = _checkpoint(
        tmp_path, model_type="mistral", architectures=["MistralForCausalLM"]
    )

    with pytest.raises(MODULE.RealScaleFoldError, match="llama-family causal LM"):
        _pretrained_configuration(tmp_path / "run", directory)


def test_a_sharded_or_incomplete_checkpoint_is_refused(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path)
    (directory / "model.safetensors").unlink()

    with pytest.raises(MODULE.RealScaleFoldError, match="sharded checkpoint"):
        _pretrained_configuration(tmp_path / "run", directory)


def test_a_pretrained_parent_needs_a_directory_and_a_synthesized_one_refuses_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MODULE.RealScaleFoldError, match="requires --parent-model-directory"
    ):
        arguments = MODULE.build_parser().parse_args(
            ["--report-root", str(tmp_path), "--parent-weights", "local-pretrained"]
        )
        MODULE._parent_weight_arguments(arguments)

    directory = _checkpoint(tmp_path)
    with pytest.raises(MODULE.RealScaleFoldError, match="only applies to"):
        arguments = MODULE.build_parser().parse_args(
            [
                "--report-root",
                str(tmp_path),
                "--parent-model-directory",
                str(directory),
            ]
        )
        MODULE._parent_weight_arguments(arguments)


def test_a_pretrained_parent_refuses_the_residual_confinement(tmp_path: Path) -> None:
    """A confined checkpoint is not the checkpoint, and the seal would lie."""

    directory = _checkpoint(tmp_path)

    with pytest.raises(MODULE.RealScaleFoldError, match="residual-facing parent"):
        _pretrained_configuration(tmp_path / "run", directory, parent_residual_rank=432)


def test_the_source_tokenizer_defaults_to_the_parents_own(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path)
    pretrained = MODULE.build_parser().parse_args(
        [
            "--report-root",
            str(tmp_path),
            "--parent-weights",
            "local-pretrained",
            "--parent-model-directory",
            str(directory),
        ]
    )
    synthesized = MODULE.build_parser().parse_args(["--report-root", str(tmp_path)])
    overridden = MODULE.build_parser().parse_args(
        [
            "--report-root",
            str(tmp_path),
            "--parent-weights",
            "local-pretrained",
            "--parent-model-directory",
            str(directory),
            "--source-tokenizer",
            "sshleifer/tiny-gpt2",
        ]
    )

    assert MODULE.resolved_source_tokenizer(pretrained) == str(directory.resolve())
    assert MODULE.resolved_source_tokenizer(synthesized) == "sshleifer/tiny-gpt2"
    assert MODULE.resolved_source_tokenizer(overridden) == "sshleifer/tiny-gpt2"


def test_the_pretrained_mode_does_not_move_a_synthesized_run_identity(
    tmp_path: Path,
) -> None:
    """The committed 20260803 receipts chain off this identity.

    Every field the pretrained mode adds is conditional on the mode being
    active.  If one of them leaked into the synthesized path, re-running the
    published commands would produce a different ``input_sha256`` and none of
    the committed receipts would chain any more.
    """

    receipt = _configuration(tmp_path)

    assert set(receipt["parent"]) == {
        "geometry_name",
        "geometry",
        "folded_layer_prefix",
        "full_depth",
        "bounded_layer_prefix",
        "weight_source",
        "pretrained",
        "seed",
        "residual_confinement_rank",
    }
    assert set(receipt["tokenizers"]) == {
        "source_repository",
        "source_local_files_only",
        "target_construction",
        "target_vocab_size",
        "bridge_kind",
    }
    assert receipt["tokenizers"]["source_repository"] == "sshleifer/tiny-gpt2"
    assert receipt["parent"]["pretrained"] is False


def test_byte_level_atom_coverage_names_what_a_real_donor_is_missing() -> None:
    """The promoted subset compiler's precondition, measured rather than assumed."""

    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = tuple(sorted(set(ByteLevel.alphabet())))
    complete = MODULE.byte_level_atom_coverage({atom: index for index, atom in
                                                enumerate(alphabet)})
    assert complete["complete"] is True
    assert complete["present_byte_level_atoms"] == 256
    assert complete["missing_byte_level_atom_characters"] == []

    incomplete = MODULE.byte_level_atom_coverage(
        {atom: index for index, atom in enumerate(alphabet[:-2])}
    )
    assert incomplete["complete"] is False
    assert incomplete["missing_byte_level_atoms"] == 2
    assert incomplete["missing_byte_level_atom_characters"] == list(alphabet[-2:])


class _StubTransport:
    """The only surface the strict row source reads off a transport profile."""

    donor_row_capacity = 8
    donor_vocab_size = 8
    target_row_capacity = 3
    target_vocab_size = 3
    alignment_fingerprint = "a" * 64
    alignment_resolution = "decoded-token-surface"

    def __init__(self, rows: tuple[int, ...]) -> None:
        self.target_to_donor_row_indices = tuple(rows)

    def fingerprint(self) -> str:
        return "f" * 64


def test_the_strict_row_source_selects_the_donor_rows_it_was_sealed_with() -> None:
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)

    rows = MODULE.SealedStrictDonorRowInterfaceRows(
        _StubTransport((5, 1, 7)),
        source_tokenizer_semantics_sha256="s" * 64,
        target_tokenizer_semantics_sha256="t" * 64,
    )
    evidence, mask, reports = rows.compile(weight)

    assert torch.equal(evidence, weight[[5, 1, 7]].to(torch.float64))
    # Every target row has evidence, so the masked-row policy has nothing to
    # leave alone -- which is exactly what the committed pair-bridge runs also
    # measured against a subset target.
    assert bool(mask.all().item()) and int(mask.numel()) == 3
    assert reports == ["f" * 64]
    assert rows.capacities == {
        "source_row_capacity": 8,
        "source_vocabulary_size": 8,
        "target_row_capacity": 3,
        "target_vocabulary_size": 3,
    }
    assert rows.descriptor["bridge_kind"] == "sealed-strict-donor-row-transport"


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        ((5, 1), "does not cover every target row"),
        ((5, 1, 5), "maps two target rows to one donor row"),
    ),
)
def test_the_strict_row_source_refuses_a_row_map_that_is_not_a_bijection(
    rows: tuple[int, ...],
    expected: str,
) -> None:
    with pytest.raises(MODULE.DynamicCausalOperatorPromotionError, match=expected):
        MODULE.SealedStrictDonorRowInterfaceRows(
            _StubTransport(rows),
            source_tokenizer_semantics_sha256="s" * 64,
            target_tokenizer_semantics_sha256="t" * 64,
        )


def test_the_interface_operator_refuses_an_unsealed_row_source() -> None:
    with pytest.raises(
        MODULE.DynamicCausalOperatorPromotionError,
        match="sealed interface row source",
    ):
        MODULE.ChartProjectedPairCodeTokenHeadInterfaceOperator(
            rows=object(),
            chart_set=object(),
        )


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
