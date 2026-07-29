from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from neural_forge.dynamic_lfm2_behavior import (
    seal_lfm2_cached_whole_output_envelope,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_lfm2_dynamic_catalog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_catalog",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-model",
        str(tmp_path / "parent"),
        "--calibration",
        str(tmp_path / "calibration.json"),
        "--cached-parent-outputs",
        str(tmp_path / "parent-outputs.json"),
        "--cached-parent-outputs-sha256",
        "d" * 64,
        "--cache-root",
        str(tmp_path / "cache"),
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--publish-destination",
        str(tmp_path / "published"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    ]


def _without_calibration(arguments: list[str]) -> list[str]:
    result = list(arguments)
    index = result.index("--calibration")
    del result[index : index + 2]
    return result


class _FixedVocabularyTokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _sealed_test_parent(*, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def sealed_parent_cache_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealed-parent-cache")
    parent = _sealed_test_parent(seed=701)
    batches = tuple(
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4 + index]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor(
                [[8 + index]],
                dtype=torch.long,
            ),
        }
        for index in range(3)
    )
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=1,
        logit_rtol=1e-5,
        logit_atol=1e-6,
        require_top1_agreement=True,
    )
    cache = MODULE.build_lfm2_circuit_trace_cache(
        root / "cache-root",
        parent,
        batches,
        config=capture_config,
        input_device="cpu",
    )
    return SimpleNamespace(
        root=root,
        parent=parent,
        cache=cache,
        capture_config=capture_config,
        tokenizer=_FixedVocabularyTokenizer(31),
    )


def _sealed_resolution_config(
    root: Path,
    cache_directory: Path,
    *,
    calibration: Path | None = None,
) -> object:
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=root / "parent",
        calibration=calibration,
        sealed_parent_cache_directory=cache_directory,
        cached_parent_outputs=root / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=root / "runtime-cache",
        snapshot_root=root / "snapshots",
        publish_destination=root / "published",
        ledger_path=root / "ledger.jsonl",
        receipt_path=root / "receipt.json",
        device="cpu",
        dtype="float32",
    )


def test_parser_defaults_to_accepted_ffn_2552_exact_budget(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(parser.parse_args(_required_args(tmp_path)))

    assert config.target_query_heads == 16
    assert config.target_key_value_heads == 8
    assert config.target_intermediate_size == 2552
    assert config.exact_target_unique_parameters == 229_349_120
    assert config.target_topology == MODULE.Lfm2Topology(16, 8, 2552)
    assert config.snapshot_chunk_bytes == 64 * 1024 * 1024
    assert config.require_capture_top1_agreement is False
    assert (
        config.child_decode_trace_mode
        == MODULE.FULL_PREFIX_CANONICAL_TRACE_MODE
    )
    assert config.identity()["capture"]["parent_decode_trace_mode"] == (
        MODULE.CACHED_VERIFIED_TRACE_MODE
    )
    assert config.whole_output_max_new_tokens == 64
    assert config.whole_output_minimum_task_score_retention == 1.0
    assert config.whole_output_maximum_parent_task_score_regressions == 0
    assert config.dense_minimum_hidden_size == 32
    assert config.identity()["target"]["dense_minimum_hidden_size"] == 32
    assert config.dense_macro_cell_attractor_gate_enabled is False
    assert config.dense_macro_cell_attractor_maximum_records == 4
    assert config.dense_macro_cell_attractor_maximum_steps == 8
    assert config.dense_attention_consumer_mode == "output-only"
    assert config.dense_attention_score_candidate_gate_enabled is False
    assert config.dense_attention_score_top_k == 8
    assert config.identity()["target"][
        "dense_macro_cell_attractor_gate"
    ]["exact_token_match_is_diagnostic_only"]
    assert config.dense_vocabulary_margin_candidate_gate_enabled is False
    assert config.dense_vocabulary_margin_candidate_replay_mode == (
        MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.9)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.9)
    )
    help_text = parser.format_help().lower()
    assert "zero-gradient" in help_text
    assert "229,349,120" in help_text
    assert "--dense-vocabulary-margin-candidate-gate" in help_text
    assert "--dense-macro-cell-attractor-gate" in help_text
    assert "--dense-attention-score-candidate-gate" in help_text
    assert "--dense-minimum-hidden-size" in help_text
    assert "--sealed-parent-cache-directory" in help_text


@pytest.mark.parametrize("minimum_hidden_size", (0, 7))
def test_dense_minimum_hidden_size_must_be_a_positive_multiple_of_eight(
    tmp_path,
    minimum_hidden_size,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match=(
            "dense_minimum_hidden_size must be a positive integer"
            if minimum_hidden_size == 0
            else "dense_minimum_hidden_size must be a multiple of eight"
        ),
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-minimum-hidden-size",
                    str(minimum_hidden_size),
                ]
            )
        )


def test_parser_accepts_sealed_cache_without_calibration(tmp_path) -> None:
    arguments = [
        *_without_calibration(_required_args(tmp_path)),
        "--sealed-parent-cache-directory",
        str(tmp_path / "sealed-cache"),
    ]

    config = MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))

    assert config.calibration is None
    assert config.sealed_parent_cache_directory == tmp_path / "sealed-cache"
    assert (
        config.parent_cache_source_mode
        == MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    )
    assert config.identity()["parent_cache_source_mode"] == (
        MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    )


def test_configuration_requires_calibration_or_sealed_cache(tmp_path) -> None:
    arguments = _without_calibration(_required_args(tmp_path))

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="calibration is required unless",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))


def test_parser_can_explicitly_restore_cached_verified_child_traces(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--child-decode-trace-mode",
                MODULE.CACHED_VERIFIED_TRACE_MODE,
            ]
        )
    )

    assert config.child_decode_trace_mode == MODULE.CACHED_VERIFIED_TRACE_MODE


def test_parser_accepts_explicit_ninety_percent_behavior_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--whole-output-maximum-new-empty-responses",
                "2",
                "--whole-output-maximum-new-fragment-responses",
                "2",
                "--whole-output-minimum-parent-nonempty-retention",
                "0.9",
                "--whole-output-minimum-task-score-retention",
                "0.9",
                "--whole-output-maximum-parent-task-score-regressions",
                "2",
                "--whole-output-maximum-new-generation-limit-saturations",
                "2",
                "--whole-output-maximum-repeated-ngram-share-regressions",
                "2",
            ]
        )
    )

    assert config.whole_output_maximum_new_empty_responses == 2
    assert config.whole_output_maximum_new_fragment_responses == 2
    assert config.whole_output_minimum_parent_nonempty_retention == 0.9
    assert config.whole_output_minimum_task_score_retention == 0.9
    assert config.whole_output_maximum_parent_task_score_regressions == 2
    assert config.whole_output_maximum_new_generation_limit_saturations == 2
    assert config.whole_output_maximum_repeated_ngram_share_regressions == 2


def test_parser_accepts_automatic_dense_ratio_and_cached_margin_basis(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-maximum-candidates",
                "17",
                "--dense-residual-basis-mode",
                "circuit_balanced",
                "--dense-vocabulary-margin-observability",
                "--dense-vocabulary-margin-top-k",
                "12",
                "--dense-vocabulary-margin-floor-fraction",
                "0.125",
                "--dense-vocabulary-margin-candidate-gate",
                "--dense-vocabulary-margin-candidate-replay-mode",
                MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                "--dense-vocabulary-margin-candidate-minimum-sign-agreement",
                "0.81",
                (
                    "--dense-vocabulary-margin-candidate-minimum-"
                    "top-competitor-nonnegative"
                ),
                "0.82",
                (
                    "--dense-vocabulary-margin-candidate-minimum-"
                    "all-top-k-nonnegative"
                ),
                "0.83",
                "--dense-vocabulary-margin-candidate-minimum-exact-top1",
                "0.84",
            ]
        )
    )

    assert config.compression_ratio == 10.0
    assert config.dense_maximum_candidates == 17
    assert config.dense_residual_basis_mode == "circuit_balanced"
    assert config.dense_vocabulary_margin_observability is True
    assert config.dense_vocabulary_margin_top_k == 12
    assert config.dense_vocabulary_margin_floor_fraction == 0.125
    assert config.dense_vocabulary_margin_candidate_gate_enabled is True
    assert config.dense_vocabulary_margin_candidate_replay_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.81)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.82)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.83)
    )
    assert (
        config
        .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.84)
    )
    target_identity = config.identity()["target"]
    assert target_identity["fold_method_id"] == MODULE.DENSE_BUDGET_MODEL_METHOD
    assert target_identity["compression_ratio"] == 10.0
    assert target_identity[
        "dense_vocabulary_margin_candidate_gate_enabled"
    ] is True
    assert target_identity[
        "dense_vocabulary_margin_candidate_replay_mode"
    ] == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    assert target_identity[
        "dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction"
    ] == pytest.approx(0.81)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_"
            "top_competitor_nonnegative_fraction"
        )
    ] == pytest.approx(0.82)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_"
            "all_top_k_nonnegative_fraction"
        )
    ] == pytest.approx(0.83)
    assert target_identity[
        (
            "dense_vocabulary_margin_candidate_minimum_"
            "exact_top1_agreement_fraction"
        )
    ] == pytest.approx(0.84)


def test_parser_accepts_cached_heterogeneous_exact_budget(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "4",
                "--dense-heterogeneous-layout",
                "--dense-heterogeneous-hidden-size",
                "96",
                "--dense-heterogeneous-target-parameters",
                "123456",
            ]
        )
    )

    assert config.compression_ratio == 4.0
    assert config.dense_heterogeneous_layout is True
    assert config.dtype == "float32"
    assert config.dense_heterogeneous_hidden_size == 96
    assert config.dense_heterogeneous_target_parameters == 123_456
    target_identity = config.identity()["target"]
    assert target_identity["dense_heterogeneous_layout"] is True
    assert target_identity["dense_heterogeneous_hidden_size"] == 96
    assert target_identity["dense_heterogeneous_target_parameters"] == 123_456


def test_cache_bound_parent_dtype_is_independent_from_fp32_child(tmp_path) -> None:
    parent_directory = tmp_path / "parent"
    parent = _sealed_test_parent(seed=704).to(dtype=torch.bfloat16)
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=tmp_path,
    )
    expected = MODULE.fingerprint_lfm2_parent(parent)
    sealed_cache = tmp_path / "sealed-cache"
    sealed_cache.mkdir()
    (sealed_cache / "manifest.json").write_text(
        json.dumps({"parent": expected.as_dict()}),
        encoding="ascii",
    )
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=parent_directory,
        calibration=None,
        sealed_parent_cache_directory=sealed_cache,
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        device="cpu",
        dtype="bfloat16",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
    )

    loaded, parent_dtype, parent_dtype_name = MODULE._load_cache_bound_parent(
        config,
        parent_directory,
        device=torch.device("cpu"),
        child_dtype=MODULE.residual._torch_dtype(config.dtype),
    )

    assert config.dtype == "float32"
    assert parent_dtype is torch.bfloat16
    assert parent_dtype_name == "bfloat16"
    assert MODULE.fingerprint_lfm2_parent(loaded) == expected


def test_parser_accepts_macro_cell_attractor_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-run-closed-loop",
                "--dense-macro-cell-attractor-gate",
                "--dense-macro-cell-attractor-maximum-records",
                "6",
                "--dense-macro-cell-attractor-maximum-steps",
                "12",
                "--dense-macro-cell-attractor-maximum-entropy-regression",
                "0.11",
                "--dense-macro-cell-attractor-maximum-repetition-regression",
                "0.12",
                "--dense-macro-cell-attractor-maximum-dominance-regression",
                "0.13",
                "--dense-macro-cell-attractor-maximum-eos-regression",
                "0.14",
                "--dense-macro-cell-attractor-collapse-entropy-threshold",
                "0.15",
                "--dense-macro-cell-attractor-collapse-repetition-threshold",
                "0.75",
                "--dense-macro-cell-attractor-collapse-dominance-threshold",
                "0.76",
                "--dense-macro-cell-attractor-maximum-new-collapsed-records",
                "2",
            ]
        )
    )

    assert config.dense_run_closed_loop
    assert config.dense_macro_cell_attractor_gate_enabled
    assert config.dense_macro_cell_attractor_maximum_records == 6
    assert config.dense_macro_cell_attractor_maximum_steps == 12
    identity = config.identity()["target"][
        "dense_macro_cell_attractor_gate"
    ]
    assert identity["enabled"]
    assert identity["maximum_entropy_regression"] == pytest.approx(0.11)
    assert identity["maximum_repetition_regression"] == pytest.approx(0.12)
    assert identity["maximum_dominance_regression"] == pytest.approx(0.13)
    assert identity["maximum_eos_regression"] == pytest.approx(0.14)
    assert identity["collapse_entropy_threshold"] == pytest.approx(0.15)
    assert identity["collapse_repetition_threshold"] == pytest.approx(0.75)
    assert identity["collapse_dominance_threshold"] == pytest.approx(0.76)
    assert identity["maximum_new_collapsed_records"] == 2


def test_macro_cell_attractor_gate_requires_dense_closed_loop(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an automatic compression_ratio request with "
        "dense_run_closed_loop=True",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "10",
                    "--dense-macro-cell-attractor-gate",
                ]
            )
        )


def test_parser_accepts_heldout_attention_score_policy(tmp_path) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "10",
                "--dense-run-closed-loop",
                "--dense-attention-consumer-mode",
                "coupled-value-output",
                "--dense-attention-score-candidate-gate",
                "--dense-attention-score-top-k",
                "6",
                "--dense-attention-score-margin-floor-fraction",
                "0.4",
                "--dense-attention-score-minimum-top1-key-agreement",
                "0.81",
                (
                    "--dense-attention-score-minimum-top-k-pair-"
                    "order-agreement"
                ),
                "0.82",
                (
                    "--dense-attention-score-minimum-winner-margin-"
                    "sign-agreement"
                ),
                "0.83",
                (
                    "--dense-attention-score-minimum-relative-margin-"
                    "floor-retention"
                ),
                "0.84",
                "--dense-attention-score-maximum-softmax-fisher-error",
                "0.12",
                "--dense-attention-score-maximum-js-divergence",
                "0.13",
            ]
        )
    )

    assert config.dense_attention_consumer_mode == "coupled-value-output"
    assert config.dense_attention_score_candidate_gate_enabled
    assert config.dense_attention_score_top_k == 6
    policy = config.identity()["target"][
        "dense_attention_score_candidate_gate"
    ]
    assert policy["enabled"]
    assert policy["top_k"] == 6
    assert policy["margin_floor_fraction"] == pytest.approx(0.4)
    assert policy["minimum_top1_key_agreement_fraction"] == pytest.approx(
        0.81
    )
    assert policy[
        "minimum_top_k_pair_order_agreement_fraction"
    ] == pytest.approx(0.82)
    assert policy[
        "minimum_winner_margin_sign_agreement_fraction"
    ] == pytest.approx(0.83)
    assert policy[
        "minimum_relative_margin_floor_retention_fraction"
    ] == pytest.approx(0.84)
    assert policy["maximum_softmax_fisher_error"] == pytest.approx(0.12)
    assert policy["maximum_js_divergence"] == pytest.approx(0.13)
    assert policy["numeric_weight_matching_used"] is False
    assert policy["parent_forward_calls"] == 0


def test_attention_score_gate_requires_dense_closed_loop(tmp_path) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="attention-score candidate gating requires",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "10",
                    "--dense-attention-score-candidate-gate",
                ]
            )
        )


def test_dense_margin_basis_fails_closed_without_automatic_ratio(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires an automatic compression_ratio",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--dense-vocabulary-margin-observability",
                ]
            )
        )


def test_dense_candidate_gate_fails_closed_without_margin_observability(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires dense_vocabulary_margin_observability",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "2",
                    "--dense-vocabulary-margin-candidate-gate",
                ]
            )
        )


def test_candidate_gate_thresholds_fail_closed_outside_unit_interval(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="must be at most one",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--compression-ratio",
                    "2",
                    "--dense-vocabulary-margin-observability",
                    "--dense-vocabulary-margin-candidate-gate",
                    "--dense-vocabulary-margin-candidate-minimum-exact-top1",
                    "1.01",
                ]
            )
        )


def test_parser_accepts_opt_in_hybrid_model_topology_without_changing_defaults(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--hybrid-model-fold",
                "--target-layer-types",
                "full_attention,conv",
                "--ordered-source-spans",
                "0,1;2",
                "--exact-target-parameters",
                "12345",
            ]
        )
    )

    assert config.hybrid_model_fold is True
    assert config.target_layer_types == ("full_attention", "conv")
    assert config.ordered_source_spans == ((0, 1), (2,))
    assert config.target_depth == 2
    assert config.exact_target_unique_parameters == 12345
    assert config.whole_output_minimum_task_score_retention == 1.0


def test_hybrid_model_flag_fails_closed_without_topology(tmp_path) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires target_layer_types",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--hybrid-model-fold",
                ]
            )
        )


def test_parser_accepts_full_depth_producer_first_and_rejects_hybrid_mix(
    tmp_path,
) -> None:
    parser = MODULE.build_parser()
    config = MODULE.config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--producer-first-model-fold",
            ]
        )
    )
    assert config.producer_first_model_fold is True
    assert config.hybrid_model_fold is False
    assert config.target_layer_types is None

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="mutually exclusive",
    ):
        MODULE.config_from_args(
            parser.parse_args(
                [
                    *_required_args(tmp_path),
                    "--producer-first-model-fold",
                    "--hybrid-model-fold",
                    "--target-layer-types",
                    "full_attention,conv",
                    "--ordered-source-spans",
                    "0,1;2",
                ]
            )
        )


class _Cache:
    cache_key = "a" * 64
    corpus_fingerprint = "b" * 64

    def __len__(self):
        return 0

    def verify(self):
        return None


class _Rows:
    cache_key = _Cache.cache_key
    corpus_fingerprint = _Cache.corpus_fingerprint


def _trajectory_rows(
    *,
    layer_kinds: tuple[str, ...],
    layer_values: tuple[float, ...],
    cache_key: str,
    trace_rows_sha256: str,
    row_width: int = 4,
) -> object:
    assert len(layer_kinds) == len(layer_values)
    return MODULE.residual.TraceRowIndex(
        paths=("path-0", "path-1", "path-2"),
        path_splits=("selection", "fit", "validation"),
        layer_kinds=layer_kinds,
        layer_fields=MappingProxyType(
            {
                index: MappingProxyType(
                    {
                        "residual.post_layer": torch.full(
                            (3, row_width),
                            value,
                            dtype=torch.float64,
                        )
                    }
                )
                for index, value in enumerate(layer_values)
            }
        ),
        cache_key=cache_key,
        corpus_fingerprint="b" * 64,
        trace_rows_sha256=trace_rows_sha256,
    )


def test_span_mapped_parent_trajectory_view_authenticates_hybrid_endpoints() -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
        row_width=2,
    )

    projected, mapping = MODULE.build_span_mapped_parent_trajectory_rows(
        parent_rows,
        child_rows,
        ordered_source_spans=((0, 1), (2,)),
        target_layer_types=("full_attention", "conv"),
    )

    projected.assert_aligned(child_rows)
    assert projected.layer_kinds == child_rows.layer_kinds
    assert torch.equal(
        projected.field(0, "residual.post_layer"),
        parent_rows.field(1, "residual.post_layer"),
    )
    assert torch.equal(
        projected.field(1, "residual.post_layer"),
        parent_rows.field(2, "residual.post_layer"),
    )
    assert mapping["complete_parent_layer_partition"] is True
    assert mapping["complete_child_layer_coverage"] is True
    assert mapping["entries"][0]["source_span_endpoint_layer_index"] == 1
    assert mapping["entries"][1]["source_span_endpoint_layer_index"] == 2
    assert mapping["entries"][0]["parent_row_shape"] == [3, 4]
    assert mapping["entries"][0]["child_row_shape"] == [3, 2]
    assert mapping["entries"][0]["cross_width"] is True
    assert (
        mapping["entries"][0]["representation_comparison"]
        == "cross_width_linear_cka"
    )


def test_cached_trajectory_evaluator_reaches_gate_with_different_hybrid_topology(
    tmp_path,
    monkeypatch,
) -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )
    parent_cache = _Cache()
    payload = MODULE.ProductionParentCachePayload(
        cache=parent_cache,
        rows=parent_rows,
        calibration_batches=(
            {
                "forced_decode_ids": torch.tensor([[1]], dtype=torch.long),
            },
        ),
        calibration_splits=("fit",),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    child_cache = SimpleNamespace(cache_key=child_rows.cache_key)
    monkeypatch.setattr(
        MODULE.residual,
        "build_child_trace_cache",
        lambda *_args, **_kwargs: (child_cache, child_rows),
    )
    gate_calls = 0

    def fake_gate(
        _parent_cache,
        _child_cache,
        _model,
        projected_parent_rows,
        observed_child_rows,
        *,
        config,
    ):
        nonlocal gate_calls
        del config
        gate_calls += 1
        projected_parent_rows.assert_aligned(observed_child_rows)
        assert projected_parent_rows.layer_kinds == (
            "full_attention",
            "short_conv",
        )
        assert torch.equal(
            projected_parent_rows.field(0, "residual.post_layer"),
            parent_rows.field(1, "residual.post_layer"),
        )
        return {
            "passed": False,
            "reasons": ["synthetic final rejection"],
            "score": 0.5,
            "teacher_forced": {"top1_agreement": 0.5},
            "free_run": {
                "token_match_fraction": 0.5,
                "lcp_fraction": 0.5,
            },
            "mean_residual_cosine": 1.0,
            "mean_residual_relative_rmse": 0.0,
            "cached_full_self_consistent": True,
            "cached_full_top1_diagnostic_fraction": 1.0,
        }

    monkeypatch.setattr(
        MODULE.residual,
        "evaluate_trajectory_gate",
        fake_gate,
    )
    evaluator = MODULE.CachedTrajectoryEvaluator(
        payload=payload,
        capture_config=MODULE.Lfm2CircuitTraceCaptureConfig(),
        gate_config=MODULE.residual.TrajectoryGateConfig(),
        final_cache_root=tmp_path / "final",
        reload_cache_root=tmp_path / "reload",
        ordered_source_spans=((0, 1), (2,)),
        target_layer_types=("full_attention", "conv"),
    )

    decision = evaluator._evaluate_model(
        object(),
        phase="accepted_in_memory_child",
        cache_root=tmp_path / "final",
    )

    assert gate_calls == 1
    assert decision.accepted is False
    assert evaluator.events[0]["residual_trajectory_mapping"][
        "mapping_mode"
    ] == "parent_source_span_endpoint_to_child_layer"
    assert evaluator.events[0]["report"]["residual_trajectory_mapping"][
        "complete_child_layer_coverage"
    ] is True


def test_cached_trajectory_evaluator_uses_dense_child_mapping_markers(
    tmp_path,
    monkeypatch,
) -> None:
    parent_rows = _trajectory_rows(
        layer_kinds=("short_conv", "full_attention", "short_conv"),
        layer_values=(1.0, 2.0, 3.0),
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
        row_width=4,
    )
    child_rows = _trajectory_rows(
        layer_kinds=("full_attention", "short_conv"),
        layer_values=(2.0, 3.0),
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
        row_width=2,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=parent_rows,
        calibration_batches=(
            {
                "forced_decode_ids": torch.tensor([[1]], dtype=torch.long),
            },
        ),
        calibration_splits=("fit",),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    child_cache = SimpleNamespace(cache_key=child_rows.cache_key)
    monkeypatch.setattr(
        MODULE.residual,
        "build_child_trace_cache",
        lambda *_args, **_kwargs: (child_cache, child_rows),
    )

    def fake_gate(
        _parent_cache,
        _child_cache,
        _model,
        projected_parent_rows,
        observed_child_rows,
        *,
        config,
    ):
        del config
        projected_parent_rows.assert_aligned(observed_child_rows)
        assert projected_parent_rows.field(
            0,
            "residual.post_layer",
        ).shape == (3, 4)
        assert observed_child_rows.field(
            0,
            "residual.post_layer",
        ).shape == (3, 2)
        return {
            "passed": False,
            "reasons": ["synthetic final rejection"],
            "score": 0.5,
            "teacher_forced": {"top1_agreement": 0.5},
            "free_run": {
                "token_match_fraction": 0.5,
                "lcp_fraction": 0.5,
            },
            "mean_residual_cosine": 0.75,
            "mean_residual_relative_rmse": 0.5,
            "cached_full_self_consistent": True,
            "cached_full_top1_diagnostic_fraction": 1.0,
        }

    monkeypatch.setattr(
        MODULE.residual,
        "evaluate_trajectory_gate",
        fake_gate,
    )
    evaluator = MODULE.CachedTrajectoryEvaluator(
        payload=payload,
        capture_config=MODULE.Lfm2CircuitTraceCaptureConfig(),
        gate_config=MODULE.residual.TrajectoryGateConfig(),
        final_cache_root=tmp_path / "final",
        reload_cache_root=tmp_path / "reload",
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            _neural_forge_dense_ordered_source_spans="0,1;2",
            _neural_forge_dense_target_layer_types="full_attention,conv",
        )
    )

    decision = evaluator._evaluate_model(
        model,
        phase="accepted_in_memory_child",
        cache_root=tmp_path / "final",
    )

    assert decision.accepted is False
    mapping = evaluator.events[0]["residual_trajectory_mapping"]
    assert mapping["mapping_binding"] == "authenticated_dense_child_config"
    assert mapping["entries"][0]["cross_width"] is True


class _Accept:
    def evaluate(self, _context):
        return MODULE.FoldDecision.accept("test gate")


class _HybridBuilder:
    def __call__(self, _request):
        return None

    def build_model(self, _request):
        return None

    def build_producer_first(self, _request):
        return None


def test_wiring_uses_automatic_lfm_catalog_real_bridge_and_disk_snapshots(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    ledger = MODULE.ExperimentLedger()
    final = _Accept()
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=final,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=ledger,
        device="cpu",
        dtype=torch.float32,
    )

    assert isinstance(
        wiring.system,
        MODULE.AutomaticDynamicFoldingSystem,
    )
    assert wiring.target_request.exact_unique_parameters == 229_349_120
    assert dict(wiring.target_request.configuration) == {"source_parent": "parent"}
    assert wiring.target_adapter.terminal_topology == MODULE.Lfm2Topology(
        16,
        8,
        2552,
    )
    resources = wiring.stack_adapter.resources
    assert resources.parent_cache_payload is payload
    assert resources.parent_cache_fingerprint == payload.cache_key
    assert resources.evidence_builder is not None
    assert resources.final_evaluator is final
    assert resources.snapshot_config.backend == "disk"
    assert Path(resources.snapshot_config.root_directory) == (tmp_path / "snapshots")
    assert wiring.system.ledger is ledger
    assert wiring.target_adapter.ledger is ledger
    assert resources.ledger is ledger
    assert wiring.system.topology_catalogs is not None
    assert wiring.system.topology_catalogs.catalog_ids == (
        MODULE.Lfm2DenseTargetTopologyCatalog().catalog_id,
    )


def test_wiring_automatically_plans_dense_blank_from_parent_parameter_budget(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=2.0,
        dense_maximum_candidates=8,
        dense_run_closed_loop=True,
        dense_macro_cell_attractor_gate_enabled=True,
        dense_macro_cell_attractor_maximum_records=5,
        dense_macro_cell_attractor_maximum_steps=7,
        dense_macro_cell_attractor_maximum_entropy_regression=0.11,
        dense_macro_cell_attractor_maximum_repetition_regression=0.12,
        dense_macro_cell_attractor_maximum_dominance_regression=0.13,
        dense_macro_cell_attractor_maximum_eos_regression=0.14,
        dense_macro_cell_attractor_collapse_entropy_threshold=0.15,
        dense_macro_cell_attractor_collapse_repetition_threshold=0.75,
        dense_macro_cell_attractor_collapse_dominance_threshold=0.76,
        dense_macro_cell_attractor_maximum_new_collapsed_records=2,
        dense_attention_consumer_mode="coupled-value-output",
        dense_attention_score_candidate_gate_enabled=True,
        dense_attention_score_top_k=6,
        dense_attention_score_margin_floor_fraction=0.4,
        dense_attention_score_minimum_top1_key_agreement_fraction=0.81,
        dense_attention_score_minimum_top_k_pair_order_agreement_fraction=0.82,
        dense_attention_score_minimum_winner_margin_sign_agreement_fraction=0.83,
        dense_attention_score_minimum_relative_margin_floor_retention_fraction=0.84,
        dense_attention_score_maximum_softmax_fisher_error=0.12,
        dense_attention_score_maximum_js_divergence=0.13,
        dense_vocabulary_margin_observability=True,
        dense_vocabulary_margin_top_k=11,
        dense_vocabulary_margin_floor_fraction=0.2,
        dense_vocabulary_margin_candidate_gate_enabled=True,
        dense_vocabulary_margin_candidate_replay_mode=(
            MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        ),
        dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction=0.71,
        dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=0.72,
        dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=0.73,
        dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=0.74,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=[
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
            ],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )
    parent_graph = MODULE.inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint="c" * 64,
    )
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.budget_plan is not None
    # The authenticated dense selected-target payload owns runtime method
    # selection.  A caller-forced method marker is deliberately rejected.
    assert wiring.target_request.fold_method_id is None
    assert MODULE.DENSE_BUDGET_MODEL_METHOD in wiring.stack_adapter.method_ids
    assert wiring.target_request.exact_unique_parameters <= (
        parent_graph.unique_parameter_count // 2
    )
    assert wiring.target_adapter.terminal_topology == MODULE.Lfm2Topology(
        1,
        1,
        32,
    )
    selected_rank = wiring.budget_plan.target_certificate.selected_candidate_rank
    selected = wiring.budget_plan.frontier.candidates[selected_rank]
    assert wiring.target_request.exact_unique_parameters == (
        selected.unique_parameters
    )
    resources = wiring.stack_adapter.resources
    assert resources.dense_fold_config.residual_basis_mode == "circuit_balanced"
    assert resources.dense_fold_config.run_closed_loop is True
    attractor = (
        resources.dense_fold_config.macro_cell_attractor_gate_config
    )
    assert isinstance(
        attractor,
        MODULE.Lfm2MacroCellAttractorGateConfig,
    )
    assert attractor.maximum_records == 5
    assert attractor.maximum_steps == 7
    assert (
        attractor.maximum_entropy_parent_distance_regression
        == pytest.approx(0.11)
    )
    assert (
        attractor.maximum_repetition_excess_regression
        == pytest.approx(0.12)
    )
    assert (
        attractor.maximum_dominant_token_excess_regression
        == pytest.approx(0.13)
    )
    assert (
        attractor.maximum_eos_position_error_regression
        == pytest.approx(0.14)
    )
    assert attractor.collapse_entropy_threshold == pytest.approx(0.15)
    assert (
        attractor.collapse_repetition_share_threshold
        == pytest.approx(0.75)
    )
    assert (
        attractor.collapse_dominant_token_share_threshold
        == pytest.approx(0.76)
    )
    assert attractor.maximum_new_collapsed_records == 2
    score_config = resources.dense_fold_config.closed_loop_adapter_config
    assert score_config.replay_capture_mode == (
        MODULE.CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert score_config.attention_consumer_mode == "coupled-value-output"
    assert score_config.attention_score_candidate_gate_enabled
    assert score_config.attention_score_top_k == 6
    assert score_config.attention_score_margin_floor_fraction == pytest.approx(
        0.4
    )
    assert (
        score_config.attention_score_minimum_top1_key_agreement_fraction
        == pytest.approx(0.81)
    )
    assert (
        score_config
        .attention_score_minimum_top_k_pair_order_agreement_fraction
        == pytest.approx(0.82)
    )
    assert (
        score_config
        .attention_score_minimum_winner_margin_sign_agreement_fraction
        == pytest.approx(0.83)
    )
    assert (
        score_config
        .attention_score_minimum_relative_margin_floor_retention_fraction
        == pytest.approx(0.84)
    )
    assert (
        score_config.attention_score_maximum_softmax_fisher_error
        == pytest.approx(0.12)
    )
    assert score_config.attention_score_maximum_js_divergence == pytest.approx(
        0.13
    )
    assert resources.dense_fold_config.vocabulary_margin_observability is True
    assert resources.dense_fold_config.vocabulary_margin_top_k == 11
    assert (
        resources.dense_fold_config.vocabulary_margin_floor_fraction
        == 0.2
    )
    assert (
        resources.dense_fold_config.vocabulary_margin_candidate_gate_enabled
        is True
    )
    assert (
        resources.dense_fold_config.vocabulary_margin_candidate_replay_mode
        == MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_sign_agreement_fraction
        == pytest.approx(0.71)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        == pytest.approx(0.72)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        == pytest.approx(0.73)
    )
    assert (
        resources.dense_fold_config
        .vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.74)
    )
    topology_catalog = wiring.system.topology_catalogs.select(
        parent_graph,
        target_family="lfm2",
        target_modalities=("text",),
    )
    assert topology_catalog.minimum_hidden_size == 32
    assert topology_catalog.compiler_capabilities == (
        MODULE.Lfm2DenseTargetTopologyCatalog().compiler_capabilities
    )
    assert wiring.budget_plan.frontier.diagnostics[
        "compiler_capability_selection_fail_closed"
    ]
    assert selected.compiler_executable
    assert selected.missing_compiler_capabilities == ()


def test_wiring_uses_cached_heterogeneous_options_and_exact_budget(
    tmp_path,
    monkeypatch,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
        dense_heterogeneous_hidden_size=96,
        dense_heterogeneous_target_parameters=123_456,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent_model = object()
    parent_graph = SimpleNamespace(fingerprint="1" * 64)
    selection_input = object()

    class _OptionBank:
        evidence_sha256 = "2" * 64

        def as_dict(self):
            return MappingProxyType(
                {
                    "schema": "test-cached-option-bank-v1",
                    "evidence_sha256": self.evidence_sha256,
                }
            )

    option_bank = _OptionBank()
    observed: dict[str, object] = {}

    def fake_selection_builder(
        received_config,
        *,
        parent_model,
        parent_graph,
        parent_cache,
        tokenizer_fingerprint,
        tokenizer_bridge,
    ):
        observed["selection_builder"] = {
            "config": received_config,
            "parent_model": parent_model,
            "parent_graph": parent_graph,
            "parent_cache": parent_cache,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "tokenizer_bridge": tokenizer_bridge,
        }
        return selection_input, option_bank

    class _Catalog:
        catalog_id = "test-heterogeneous-catalog"
        version = 1
        priority = 100

        def supports(self, *_args, **_kwargs):
            return True

        def enumerate_frontier(self, *_args, **_kwargs):
            raise AssertionError("Planning is intercepted in this wiring test.")

    def fake_catalog(**kwargs):
        observed["catalog_kwargs"] = dict(kwargs)
        return _Catalog()

    def fake_plan_budgeted_target(
        _system,
        *,
        parent_graph,
        budget_request,
        child_name,
        tokenizer_fingerprint,
        adapter_id,
        blank_seed,
        prefer_component_risk,
        **_kwargs,
    ):
        observed["budget_request"] = budget_request
        observed["planned_parent_graph"] = parent_graph
        observed["prefer_component_risk"] = prefer_component_risk
        candidate = SimpleNamespace(
            configuration=(),
            fingerprint="3" * 64,
            source_unique_parameters=500_000,
            requested_unique_parameters=123_456,
            unique_parameters=123_456,
            logical_parameters=123_456,
            native_tied_parameter_count=0,
            compression_ratio=500_000 / 123_456,
            budget_delta=0,
            at_or_below_budget=True,
            family="lfm2",
            modalities=("text",),
            ordered_source_spans=((0,),),
            target_vocabulary_size=64,
            tokenizer_bridge_required=False,
            required_compiler_capabilities=(),
            declared_compiler_capabilities=(),
            missing_compiler_capabilities=(),
            compiler_executable=True,
        )
        request = SimpleNamespace(
            child_name=child_name,
            exact_unique_parameters=budget_request.target_unique_parameters,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=adapter_id,
            blank_seed=blank_seed,
        )
        return SimpleNamespace(
            target_request=request,
            fingerprint="4" * 64,
            frontier=SimpleNamespace(
                candidates=(candidate,),
                fingerprint="5" * 64,
                request=budget_request,
            ),
            target_certificate=SimpleNamespace(
                selected_candidate_rank=0,
                fingerprint="6" * 64,
                selection_policy="test-cached-exact",
            ),
        )

    monkeypatch.setattr(
        MODULE,
        "_build_cached_heterogeneous_selection_input",
        fake_selection_builder,
    )
    monkeypatch.setattr(MODULE, "Lfm2DenseTargetTopologyCatalog", fake_catalog)
    monkeypatch.setattr(
        MODULE.AutomaticDynamicFoldingSystem,
        "plan_budgeted_target",
        fake_plan_budgeted_target,
    )

    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_model=parent_model,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    builder_inputs = observed["selection_builder"]
    assert builder_inputs["parent_model"] is parent_model
    assert builder_inputs["parent_graph"] is parent_graph
    assert builder_inputs["parent_cache"] is payload.cache
    catalog_kwargs = observed["catalog_kwargs"]
    assert catalog_kwargs["target_token_interface_mode"] == "untied"
    assert (
        catalog_kwargs["heterogeneous_layout_selection_input"]
        is selection_input
    )
    budget_request = observed["budget_request"]
    assert budget_request.target_unique_parameters == 123_456
    assert budget_request.compression_ratio is None
    assert budget_request.match_policy == "exact"
    assert observed["prefer_component_risk"] is True
    assert wiring.target_request.exact_unique_parameters == 123_456
    assert wiring.heterogeneous_capacity_option_bank is option_bank
    planned = MODULE._planned_dense_target_evidence(wiring)
    assert planned["requested_unique_parameters"] == 123_456
    assert planned["requested_compression_ratio"] is None
    assert planned["heterogeneous_capacity_option_bank"] == {
        "schema": "test-cached-option-bank-v1",
        "evidence_sha256": "2" * 64,
    }


def test_wiring_plans_h8_at_roughly_250x_without_weakening_capability_gate(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=250.0,
        dense_maximum_candidates=16,
        dense_minimum_hidden_size=8,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            layer_types=[
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
            ],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )
    parent_graph = MODULE.inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint="c" * 64,
    )

    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        parent_graph=parent_graph,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=lambda _request: None,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.budget_plan is not None
    selected_rank = wiring.budget_plan.target_certificate.selected_candidate_rank
    selected = wiring.budget_plan.frontier.candidates[selected_rank]
    selected_configuration = dict(selected.configuration)
    topology_catalog = wiring.system.topology_catalogs.select(
        parent_graph,
        target_family="lfm2",
        target_modalities=("text",),
    )

    assert config.identity()["target"]["dense_minimum_hidden_size"] == 8
    assert topology_catalog.minimum_hidden_size == 8
    assert selected_configuration["hidden_size"] == 8
    assert wiring.budget_plan.target_certificate.selection_policy == (
        "supported_pareto_balanced_executable_v1"
    )
    assert dict(wiring.target_request.configuration)["hidden_size"] == 8
    assert wiring.target_request.exact_unique_parameters <= (
        parent_graph.unique_parameter_count // 250
    )
    assert selected.compiler_executable
    assert selected.missing_compiler_capabilities == ()
    assert topology_catalog.compiler_capabilities == (
        MODULE.Lfm2DenseTargetTopologyCatalog().compiler_capabilities
    )
    assert selected.required_compiler_capabilities == (
        topology_catalog.compiler_capabilities
    )
    assert wiring.budget_plan.frontier.diagnostics[
        "minimum_hidden_size"
    ] == 8
    assert wiring.budget_plan.frontier.diagnostics[
        "compiler_capability_selection_fail_closed"
    ]


def test_wiring_emits_explicit_hybrid_method_and_ordered_span_request(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        exact_target_unique_parameters=12345,
        hybrid_model_fold=True,
        target_layer_types=("full_attention", "conv"),
        ordered_source_spans=((0, 1), (2,)),
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = _HybridBuilder()
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=builder,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    request = wiring.target_request
    assert request.fold_method_id == MODULE.HYBRID_MODEL_METHOD
    assert dict(request.configuration) == {
        "source_parent": "parent",
        "target_depth": 2,
        "target_layer_types": "full_attention,conv",
        "ordered_source_spans": "0,1;2",
    }
    assert wiring.stack_adapter.resources.model_evidence_builder == (
        builder.build_model
    )
    assert wiring.stack_adapter.resources.model_fold_config is not None
    assert MODULE.HYBRID_MODEL_METHOD in wiring.stack_adapter.method_ids


def test_wiring_emits_explicit_producer_first_full_depth_method(tmp_path) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        exact_target_unique_parameters=12345,
        producer_first_model_fold=True,
    )
    payload = MODULE.ProductionParentCachePayload(
        cache=_Cache(),
        rows=_Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = _HybridBuilder()
    wiring = MODULE.build_production_catalog_wiring(
        config,
        parent_cache_payload=payload,
        tokenizer_fingerprint="c" * 64,
        evidence_builder=builder,
        final_evaluator=_Accept(),
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=lambda _model: True,
        ledger=MODULE.ExperimentLedger(),
        device="cpu",
        dtype=torch.float32,
    )

    assert wiring.target_request.fold_method_id == (
        MODULE.PRODUCER_FIRST_MODEL_METHOD
    )
    assert dict(wiring.target_request.configuration) == {
        "source_parent": "parent"
    }
    resources = wiring.stack_adapter.resources
    assert resources.producer_first_evidence_builder == (
        builder.build_producer_first
    )
    assert resources.producer_first_fold_config is not None
    assert resources.model_evidence_builder is None
    assert (
        MODULE.PRODUCER_FIRST_MODEL_METHOD
        in wiring.stack_adapter.method_ids
    )


def test_cached_prefix_mismatch_is_exposed_without_weakening_tolerance() -> None:
    diagnostic = MODULE.diagnose_failure(
        RuntimeError(
            "decode.000 cached logits differ from use_cache=False "
            "full-prefix logits (maximum error 1.375)."
        )
    )

    assert diagnostic["category"] == ("cached_vs_full_prefix_logit_mismatch")
    assert diagnostic["phase"] == "decode.000"
    assert diagnostic["maximum_logit_error"] == 1.375
    assert diagnostic["tolerance_weakened"] is False
    assert diagnostic["fail_closed"] is True
    assert diagnostic["next_function_to_fix"].endswith("._capture_batch")
    assert diagnostic["detector"].endswith("._compare_logits")


def test_new_ledger_attempt_evidence_exposes_only_this_invocation() -> None:
    old = SimpleNamespace(
        record_sha256="1" * 64,
        status="accepted",
        reason_class="old",
        reason="old record",
        signature=SimpleNamespace(
            method_id="old-method",
            candidate_sha256="2" * 64,
        ),
        metrics=(("score", 0.1),),
    )
    current = SimpleNamespace(
        record_sha256="3" * 64,
        status="rejected",
        reason_class="behavior_gate_reject",
        reason="candidate margin gate rejected",
        signature=SimpleNamespace(
            method_id=MODULE.DENSE_BUDGET_MODEL_METHOD,
            candidate_sha256="4" * 64,
        ),
        metrics=(
            ("vocabulary_margin_candidate_gate_enabled", True),
            ("vocabulary_margin_candidate_gate_passed", False),
            (
                "vocabulary_margin_candidate_"
                "candidate_exact_parent_top1_agreement_fraction",
                0.42,
            ),
        ),
    )
    ledger = SimpleNamespace(records=(old, current))

    assert MODULE._new_ledger_attempt_evidence(ledger, 1) == [
        {
            "record_sha256": "3" * 64,
            "status": "rejected",
            "reason_class": "behavior_gate_reject",
            "reason": "candidate margin gate rejected",
            "method_id": MODULE.DENSE_BUDGET_MODEL_METHOD,
            "candidate_sha256": "4" * 64,
            "metrics": {
                "vocabulary_margin_candidate_gate_enabled": True,
                "vocabulary_margin_candidate_gate_passed": False,
                (
                    "vocabulary_margin_candidate_"
                    "candidate_exact_parent_top1_agreement_fraction"
                ): 0.42,
            },
        }
    ]


def test_sealed_parent_cache_never_calls_builder_or_parent_forward(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not execute this function")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )
    monkeypatch.setattr(fixture.parent, "forward", forbidden)
    monkeypatch.setattr(MODULE.Lfm2CircuitTraceCache, "iter_batches", forbidden)
    monkeypatch.setattr(MODULE.Lfm2CircuitTraceCache, "load_batch", forbidden)

    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
    )

    assert resolved.cache.cache_key == fixture.cache.cache_key
    assert resolved.source_mode == MODULE.SEALED_PARENT_CACHE_SOURCE_MODE
    assert resolved.evidence_split_source == "sealed_cache_deterministic"
    assert resolved.parent_cache_build_called is False
    assert resolved.parent_cache_opened_with_full_verification is True
    assert resolved.parent_trace_regeneration_forward_count == 0
    assert resolved.supplied_calibration_matches_cache is None
    assert resolved.calibration_source_sha256 is None
    assert len(resolved.evidence_split_sha256) == 64
    assert sorted(resolved.calibration_splits) == [
        "fit",
        "selection",
        "validation",
    ]
    receipt_fields = resolved.runtime_receipt_fields()
    assert receipt_fields["parent_cache_build_called"] is False
    assert receipt_fields["parent_cache_build_forward_count"] == 0
    assert receipt_fields["parent_trace_regeneration_forward_count"] == 0
    assert receipt_fields["parent_trace_regeneration_after_cache"] == 0
    assert receipt_fields["parent_cache_selective_session_reused"] is True
    assert len(receipt_fields["parent_cache_selective_session_sha256"]) == 64
    assert resolved.selective_session is not None
    resolved.selective_session.close()


def test_tokenizer_facing_parent_accepts_padded_model_vocabulary() -> None:
    parent = _sealed_test_parent(seed=703)
    tokenizer = _FixedVocabularyTokenizer(29)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 28]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_tokenizer_facing_parent_rejects_tokenizer_beyond_model_rows() -> None:
    parent = _sealed_test_parent(seed=704)
    tokenizer = _FixedVocabularyTokenizer(32)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="Tokenizer vocabulary exceeds",
    ):
        MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_tokenizer_facing_parent_rejects_cache_id_beyond_model_rows() -> None:
    parent = _sealed_test_parent(seed=705)
    tokenizer = _FixedVocabularyTokenizer(31)
    batches = (
        {
            "input_ids": torch.tensor([[1, 2, 31]], dtype=torch.long),
            "forced_decode_ids": torch.tensor([[7]], dtype=torch.long),
        },
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="row capacity",
    ):
        MODULE._validate_tokenizer_facing_parent(parent, tokenizer, batches)


def test_sealed_parent_cache_ignores_mismatched_calibration_for_splits(
    tmp_path,
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    calibration = tmp_path / "different-calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [1, 20, 21],
                    "forced_decode_ids": [22],
                },
                {
                    "split": "fit",
                    "input_ids": [1, 23, 24],
                    "forced_decode_ids": [25],
                },
            ]
        ),
        encoding="utf-8",
    )
    config = _sealed_resolution_config(
        tmp_path,
        fixture.cache.path,
        calibration=calibration,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    resolved = MODULE.resolve_production_parent_cache(
        config,
        fixture.parent,
        fixture.tokenizer,
        fixture.capture_config,
        input_device="cpu",
    )

    assert resolved.supplied_calibration_matches_cache is False
    assert resolved.supplied_calibration_sha256 == MODULE._file_sha256(
        calibration
    )
    assert resolved.evidence_split_source == "sealed_cache_deterministic"
    assert resolved.calibration_source_sha256 is None
    assert len(resolved.calibration_splits) == len(fixture.cache)


def test_sealed_parent_cache_rejects_different_live_parent(
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    different_parent = _sealed_test_parent(seed=702)
    config = _sealed_resolution_config(
        fixture.root,
        fixture.cache.path,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="fingerprint does not match",
    ):
        MODULE.resolve_production_parent_cache(
            config,
            different_parent,
            fixture.tokenizer,
            fixture.capture_config,
            input_device="cpu",
        )


def test_sealed_parent_cache_rejects_corrupt_trace_file(
    tmp_path,
    monkeypatch,
    sealed_parent_cache_fixture,
) -> None:
    fixture = sealed_parent_cache_fixture
    corrupt_cache = tmp_path / "corrupt-cache"
    shutil.copytree(fixture.cache.path, corrupt_cache)
    trace_path = next(corrupt_cache.rglob("trace.safetensors"))
    contents = bytearray(trace_path.read_bytes())
    contents[-1] ^= 1
    trace_path.write_bytes(contents)
    config = _sealed_resolution_config(tmp_path, corrupt_cache)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed cache mode must not call the builder")

    monkeypatch.setattr(
        MODULE,
        "build_lfm2_circuit_trace_cache",
        forbidden,
    )

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="failed full verification",
    ):
        MODULE.resolve_production_parent_cache(
            config,
            fixture.parent,
            fixture.tokenizer,
            fixture.capture_config,
            input_device="cpu",
        )


def test_missing_local_parent_writes_compact_failed_receipt_and_no_artifact(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    stored = json.loads(config.receipt_path.read_text(encoding="ascii"))

    assert receipt == stored
    assert receipt["status"] == "failed_closed"
    assert receipt["publication_present"] is False
    assert receipt["training_steps"] == 0
    assert receipt["tolerance_weakened"] is False
    assert receipt["ledger"]["new_attempts"] == []
    assert len(receipt["receipt_sha256"]) == 64
    assert config.ledger_path.read_bytes() == b""
    assert config.diagnostics_path.is_file()
    assert not config.publish_destination.exists()


def test_failed_receipt_explicitly_records_candidate_gate_policy(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_vocabulary_margin_observability=True,
        dense_vocabulary_margin_candidate_gate_enabled=True,
        dense_vocabulary_margin_candidate_replay_mode=(
            MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        ),
        dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction=0.81,
        dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=0.82,
        dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=0.83,
        dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=0.84,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)

    assert receipt["status"] == "failed_closed"
    assert receipt["configuration_sha256"] == MODULE.canonical_sha256(
        config.identity()
    )
    assert receipt["vocabulary_margin_candidate_gate_policy"] == {
        "enabled": True,
        "replay_mode": MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "minimum_sign_agreement_fraction": pytest.approx(0.81),
        "minimum_top_competitor_nonnegative_fraction": pytest.approx(0.82),
        "minimum_all_top_k_nonnegative_fraction": pytest.approx(0.83),
        "minimum_exact_top1_agreement_fraction": pytest.approx(0.84),
    }
    assert json.loads(
        config.receipt_path.read_text(encoding="ascii")
    ) == receipt


def test_failed_receipt_records_macro_cell_attractor_policy(
    tmp_path,
) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_run_closed_loop=True,
        dense_macro_cell_attractor_gate_enabled=True,
        dense_macro_cell_attractor_maximum_records=5,
        dense_macro_cell_attractor_maximum_steps=7,
        dense_macro_cell_attractor_maximum_entropy_regression=0.11,
        dense_macro_cell_attractor_maximum_repetition_regression=0.12,
        dense_macro_cell_attractor_maximum_dominance_regression=0.13,
        dense_macro_cell_attractor_maximum_eos_regression=0.14,
        dense_macro_cell_attractor_collapse_entropy_threshold=0.15,
        dense_macro_cell_attractor_collapse_repetition_threshold=0.75,
        dense_macro_cell_attractor_collapse_dominance_threshold=0.76,
        dense_macro_cell_attractor_maximum_new_collapsed_records=2,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    policy = receipt["macro_cell_attractor_gate_policy"]

    assert receipt["status"] == "failed_closed"
    assert policy["enabled"]
    assert policy["maximum_records"] == 5
    assert policy["maximum_steps"] == 7
    assert (
        policy["maximum_entropy_parent_distance_regression"]
        == pytest.approx(0.11)
    )
    assert policy["maximum_repetition_excess_regression"] == pytest.approx(
        0.12
    )
    assert (
        policy["maximum_dominant_token_excess_regression"]
        == pytest.approx(0.13)
    )
    assert (
        policy["maximum_eos_position_error_regression"]
        == pytest.approx(0.14)
    )
    assert policy["maximum_new_collapsed_records"] == 2
    assert policy["parent_outputs_cached"]
    assert policy["child_only_free_rollout"]
    assert policy["exact_token_match_is_diagnostic_only"]
    assert policy["training_steps"] == 0


def test_failed_receipt_records_attention_score_policy(tmp_path) -> None:
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "missing-parent",
        calibration=tmp_path / "missing-calibration.json",
        cached_parent_outputs=tmp_path / "missing-parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=10.0,
        dense_run_closed_loop=True,
        dense_attention_consumer_mode="coupled-value-output",
        dense_attention_score_candidate_gate_enabled=True,
        dense_attention_score_top_k=6,
        dense_attention_score_margin_floor_fraction=0.4,
        dense_attention_score_minimum_top1_key_agreement_fraction=0.81,
        dense_attention_score_minimum_top_k_pair_order_agreement_fraction=0.82,
        dense_attention_score_minimum_winner_margin_sign_agreement_fraction=0.83,
        dense_attention_score_minimum_relative_margin_floor_retention_fraction=0.84,
        dense_attention_score_maximum_softmax_fisher_error=0.12,
        dense_attention_score_maximum_js_divergence=0.13,
        device="cpu",
        dtype="float32",
    )

    receipt = MODULE.run_production_catalog(config)
    policy = receipt["attention_score_candidate_gate_policy"]

    assert receipt["status"] == "failed_closed"
    assert policy["enabled"]
    assert policy["consumer_mode"] == "coupled-value-output"
    assert policy["top_k"] == 6
    assert policy["relative_margin_floor"] == pytest.approx(0.4)
    assert policy["minimum_top1_key_agreement_fraction"] == pytest.approx(
        0.81
    )
    assert policy[
        "minimum_top_k_pair_order_agreement_fraction"
    ] == pytest.approx(0.82)
    assert policy[
        "minimum_winner_margin_sign_agreement_fraction"
    ] == pytest.approx(0.83)
    assert policy[
        "minimum_relative_margin_floor_retention_fraction"
    ] == pytest.approx(0.84)
    assert policy["maximum_softmax_fisher_error"] == pytest.approx(0.12)
    assert policy["maximum_js_divergence"] == pytest.approx(0.13)
    assert policy["masked_keys_excluded"]
    assert policy["numeric_weight_matching_used"] is False
    assert policy["parent_forward_calls"] == 0
    assert policy["training_steps"] == 0


def test_tiny_real_local_catalog_folds_publishes_and_reloads(
    tmp_path,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    parent_directory = tmp_path / "parent"
    parent_config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(123)
    parent = transformers.Lfm2ForCausalLM(parent_config).eval()
    MODULE.residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=tmp_path,
    )
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                "<bos>": 2,
                "<eos>": 3,
                    "4": 4,
                "b": 5,
                "c": 6,
                "d": 7,
                "e": 8,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(parent_directory / "tokenizer.json"))
    (parent_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    target_config = transformers.Lfm2Config(**parent_config.to_dict())
    target_config.intermediate_size = 5
    target_config.block_ff_dim = 5
    target_config.block_auto_adjust_ff_dim = False
    target_count = MODULE.residual.unique_parameter_count(
        transformers.Lfm2ForCausalLM(target_config)
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "split": "selection",
                    "input_ids": [2, 4, 5],
                    "forced_decode_ids": [6, 7],
                },
                {
                    "split": "fit",
                    "input_ids": [2, 5, 6],
                    "forced_decode_ids": [7, 8],
                },
                {
                    "split": "validation",
                    "input_ids": [2, 6, 7],
                    "forced_decode_ids": [8, 4],
                },
            ]
        ),
        encoding="utf-8",
    )
    cached_parent_output_rows = tmp_path / "parent-output-rows.json"
    cached_parent_output_rows.write_text(
        json.dumps(
            [
                {
                    "id": "tiny",
                        "prompt": "4",
                        "category": "math",
                        "expected_response": "4",
                        "parent_output_text": "4 4 4 4",
                        "parent_output_ids": [4, 4, 4, 4],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded_calibration = MODULE.residual.load_calibration_batches(calibration)
    capture_config = MODULE.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=8,
        max_decode_steps=8,
        max_capture_bytes=128 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )
    parent_cache = MODULE.build_lfm2_circuit_trace_cache(
        tmp_path / "cache" / "parent",
        parent,
        loaded_calibration.batches,
        config=capture_config,
        input_device="cpu",
    )
    cached_parent_outputs = tmp_path / "parent-output-envelope.json"
    envelope_receipt = seal_lfm2_cached_whole_output_envelope(
        cached_parent_output_rows,
        cached_parent_outputs,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=MODULE.fingerprint_local_tokenizer(parent_directory),
        generation_limit=4,
    )
    config = MODULE.ProductionLfm2CatalogConfig(
        parent_model=parent_directory,
        calibration=calibration,
        cached_parent_outputs=cached_parent_outputs,
        cached_parent_outputs_sha256=envelope_receipt["file_sha256"],
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        device="cpu",
        dtype="float32",
        target_query_heads=4,
        target_key_value_heads=2,
        target_intermediate_size=5,
        exact_target_unique_parameters=target_count,
        max_selection_rows=16,
        max_fit_rows=16,
        max_validation_rows=16,
        endpoint_ridge=0.0,
        max_endpoint_relative_rmse=1e6,
        min_endpoint_cosine_similarity=-1.0,
        min_residual_cosine=-1.0,
        whole_output_max_new_tokens=4,
        snapshot_chunk_bytes=1024 * 1024,
    )

    receipt = MODULE.run_production_catalog(config)
    diagnostics = json.loads(config.diagnostics_path.read_text(encoding="ascii"))

    assert receipt["status"] == "complete"
    assert receipt["target"]["exact_unique_parameters"] == target_count
    assert receipt["target"]["dense_minimum_hidden_size"] == 32
    assert receipt["target"]["parameters_removed"] > 0
    assert receipt["target"]["hub_loader_used"] is False
    assert receipt["fold"]["snapshot_backend"] == "disk"
    assert receipt["fold"]["optimizer_steps"] == 0
    assert receipt["fold"]["backward_calls"] == 0
    assert receipt["parent"]["parent_forward_count_after_cache"] == 0
    assert set(receipt["behavior"]) == {
        "accepted_in_memory_child",
        "freshly_reloaded_child",
    }
    assert receipt["artifact"]["fresh_reload_accepted"] is True
    assert (config.publish_destination / "certificate.json").is_file()
    assert (config.publish_destination / "child" / "tokenizer.json").is_file()
    assert diagnostics["parent_traces_regenerated_after_cache"] is False
    assert diagnostics["evidence_builds"]


def test_production_runner_source_has_no_training_or_hub_loader_path() -> None:
    source = inspect.getsource(MODULE)

    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert ".from_pretrained(" not in source
    assert "hf_hub_download" not in source
    assert "snapshot_download" not in source
