from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from neural_forge.cli import build_parser
from neural_forge.hf_qwen_target import (
    Qwen3PlanInfeasibleError,
    TARGET_REPORT_FILENAME,
    count_qwen3_parameters,
    materialize_qwen3_target,
    plan_qwen3_target,
    qwen3_parameter_breakdown,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_qwen_target_cli_exposes_budget_and_exact_shape_constraints() -> None:
    args = build_parser().parse_args(
        [
            "hf-qwen-build-target",
            "Qwen/source",
            "blank-output",
            "--target-params",
            "600000000",
            "--hidden-sizes",
            "1024",
            "--intermediate-sizes",
            "3072",
            "--layer-counts",
            "24",
            "--query-head-counts",
            "8",
            "--key-value-head-counts",
            "2",
            "--head-dims",
            "128",
            "--seed",
            "7",
        ]
    )

    assert args.target_params == 600_000_000
    assert args.hidden_sizes == "1024"
    assert args.intermediate_sizes == "3072"
    assert args.layer_counts == "24"
    assert args.seed == 7


def test_qwen_target_cli_uses_materialized_default_constraint_values(
    monkeypatch, tmp_path: Path
) -> None:
    import neural_forge.hf_qwen_target as target_module
    import neural_forge.hf_tools as tools_module

    captured = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    fake_transformers = type(
        "FakeTransformers",
        (),
        {"AutoTokenizer": FakeAutoTokenizer, "AutoConfig": FakeAutoConfig},
    )
    monkeypatch.setattr(
        tools_module,
        "_torch_and_transformers",
        lambda: (object(), fake_transformers),
    )

    def fake_plan(_target_params, **kwargs):
        captured.update(kwargs)
        return {"report": {"selected": True}}

    monkeypatch.setattr(target_module, "plan_qwen3_target", fake_plan)
    monkeypatch.setattr(
        target_module,
        "materialize_qwen3_target",
        lambda *_args, **_kwargs: {"materialized": True},
    )
    args = build_parser().parse_args(
        [
            "hf-qwen-build-target",
            "Qwen/source",
            str(tmp_path / "blank"),
            "--target-params",
            "600000000",
        ]
    )

    result = args.func(args)

    constraints = captured["constraints"]
    assert isinstance(constraints.hidden_sizes, tuple)
    assert isinstance(constraints.layer_counts, tuple)
    assert isinstance(constraints.query_head_counts, tuple)
    assert isinstance(constraints.key_value_head_counts, tuple)
    assert isinstance(constraints.head_dims, tuple)
    assert result["artifact"] == {"materialized": True}


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("attention_bias", [False, True])
def test_formula_matches_actual_tiny_qwen3_model_numel(
    tie_word_embeddings: bool, attention_bias: bool
) -> None:
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=4,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=attention_bias,
    )
    model = transformers.Qwen3ForCausalLM(config)

    expected = count_qwen3_parameters(config)
    actual = sum(parameter.numel() for parameter in model.parameters())

    assert expected == actual
    assert (model.lm_head.weight is model.model.embed_tokens.weight) is (
        tie_word_embeddings
    )


def test_breakdown_covers_gqa_swiglu_norms_biases_and_tied_head() -> None:
    breakdown = qwen3_parameter_breakdown(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=4,
        tie_word_embeddings=True,
        attention_bias=True,
    )

    assert breakdown["derived_dimensions"] == {
        "query_projection_width": 24,
        "key_projection_width": 8,
        "value_projection_width": 8,
        "query_heads_per_key_value_head": 3,
    }
    layer = breakdown["per_decoder_layer"]
    assert layer["attention_weights"] == {
        "q_proj": 576,
        "k_proj": 192,
        "v_proj": 192,
        "o_proj": 576,
    }
    assert layer["attention_biases"] == {
        "q_proj": 24,
        "k_proj": 8,
        "v_proj": 8,
        "o_proj": 24,
    }
    assert layer["attention_rmsnorms"] == {"q_norm": 4, "k_norm": 4}
    assert layer["swiglu_weights"] == {
        "gate_proj": 960,
        "up_proj": 960,
        "down_proj": 960,
    }
    assert layer["decoder_rmsnorms"] == {
        "input_layernorm": 24,
        "post_attention_layernorm": 24,
    }
    assert breakdown["embedding_and_head"]["output_head_logical"] == 768
    assert breakdown["embedding_and_head"]["output_head_unique"] == 0
    assert breakdown["components"]["final_rmsnorm"] == 24
    assert sum(breakdown["components"].values()) == breakdown["total"]


def _two_intermediate_constraints(intermediate_sizes: list[int]) -> dict:
    return {
        "hidden_sizes": [16],
        "intermediate_sizes": intermediate_sizes,
        "layer_counts": [1],
        "query_head_counts": [4],
        "key_value_head_counts": [2],
        "head_dims": [4],
    }


def test_planning_is_deterministic_and_prefers_any_under_budget_shape() -> None:
    lower_count = count_qwen3_parameters(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    upper_count = count_qwen3_parameters(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    budget = upper_count - 1
    assert budget - lower_count > upper_count - budget

    first = plan_qwen3_target(
        budget,
        vocab_size=40,
        constraints=_two_intermediate_constraints([64, 32, 64]),
    )
    second = plan_qwen3_target(
        budget,
        vocab_size=40,
        constraints=_two_intermediate_constraints([32, 64]),
    )

    assert first == second
    assert first["config"]["intermediate_size"] == 32
    assert first["parameter_count"] == lower_count
    assert first["under_budget"] is True
    closest_absolute = first["report"]["closest_absolute_candidate"]
    assert closest_absolute["shape"]["intermediate_size"] == 64
    assert closest_absolute["absolute_budget_distance"] == 1
    assert first["report"]["ranking"]["under_budget_precedes_over_budget"] is True
    assert len(first["plan_sha256"]) == 64


def test_no_under_budget_candidate_falls_back_to_closest_absolute() -> None:
    plan = plan_qwen3_target(
        10,
        vocab_size=40,
        constraints=_two_intermediate_constraints([32, 64]),
    )

    assert plan["under_budget"] is False
    assert plan["config"]["intermediate_size"] == 32
    assert plan["report"]["selected"] == plan["report"]["closest_absolute_candidate"]


def test_infeasible_gqa_constraints_have_structured_report() -> None:
    constraints = {
        "hidden_sizes": [12],
        "intermediate_sizes": [24],
        "layer_counts": [1],
        "query_head_counts": [3],
        "key_value_head_counts": [2],
        "head_dims": [4],
    }

    with pytest.raises(Qwen3PlanInfeasibleError) as raised:
        plan_qwen3_target(10_000, vocab_size=32, constraints=constraints)

    report = raised.value.report
    assert report["status"] == "infeasible"
    assert report["infeasibility"]["reason"] == "no_valid_shape"
    assert report["search"]["valid_candidate_count"] == 0
    assert report["search"]["rejected_gqa_divisibility_count"] == 1


def test_require_under_budget_reports_nearest_infeasible_candidate() -> None:
    with pytest.raises(Qwen3PlanInfeasibleError) as raised:
        plan_qwen3_target(
            10,
            vocab_size=40,
            constraints=_two_intermediate_constraints([32]),
            require_under_budget=True,
        )

    infeasibility = raised.value.report["infeasibility"]
    assert infeasibility["reason"] == "no_candidate_at_or_under_budget"
    assert infeasibility["nearest_candidate"]["parameter_count"] > 10


class _TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, size: int = 32) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def save_pretrained(self, output_dir: str | Path) -> tuple[str, str]:
        target = Path(output_dir)
        tokenizer_config = target / "tokenizer_config.json"
        special_tokens = target / "special_tokens_map.json"
        tokenizer_config.write_text(
            json.dumps({"tokenizer_class": type(self).__name__}),
            encoding="utf-8",
        )
        special_tokens.write_text(
            json.dumps({"pad_token": "<pad>", "eos_token": "</s>"}),
            encoding="utf-8",
        )
        return str(tokenizer_config), str(special_tokens)


def _tiny_plan(tokenizer: _TinyTokenizer) -> dict:
    return plan_qwen3_target(
        100_000,
        tokenizer=tokenizer,
        constraints={
            "hidden_sizes": [16],
            "intermediate_sizes": [32],
            "layer_counts": [1],
            "query_head_counts": [4],
            "key_value_head_counts": [2],
            "head_dims": [4],
        },
        config_overrides={
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
        },
    )


def test_materializer_is_seed_reproducible_and_preserves_caller_rng(
    tmp_path: Path,
) -> None:
    tokenizer = _TinyTokenizer()
    plan = _tiny_plan(tokenizer)

    torch.manual_seed(991)
    expected_next = torch.rand(5)
    torch.manual_seed(991)
    first = materialize_qwen3_target(plan, tmp_path / "first", tokenizer, seed=17)
    observed_next = torch.rand(5)
    second = materialize_qwen3_target(plan, tmp_path / "second", tokenizer, seed=17)
    third = materialize_qwen3_target(plan, tmp_path / "third", tokenizer, seed=18)

    assert torch.equal(observed_next, expected_next)
    first_hash = first["random_initialization"]["state_dict_sha256"]
    second_hash = second["random_initialization"]["state_dict_sha256"]
    third_hash = third["random_initialization"]["state_dict_sha256"]
    assert first_hash == second_hash
    assert first_hash != third_hash
    assert first["random_initialization"]["pretrained_child_weights_loaded"] is False
    assert first["training"] == {
        "performed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_with_grad": 0,
    }


def test_materializer_saves_model_config_tokenizer_and_metadata(
    tmp_path: Path,
) -> None:
    tokenizer = _TinyTokenizer()
    plan = _tiny_plan(tokenizer)
    output_dir = tmp_path / "artifact"

    report = materialize_qwen3_target(plan, output_dir, tokenizer, seed=23)
    saved_report = json.loads(
        (output_dir / TARGET_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    artifact_names = {item["path"] for item in report["artifacts"]["files"]}

    assert saved_report == report
    assert "config.json" in artifact_names
    assert "model.safetensors" in artifact_names
    assert "tokenizer_config.json" in artifact_names
    assert "special_tokens_map.json" in artifact_names
    assert TARGET_REPORT_FILENAME not in artifact_names
    assert report["parameter_count"]["matches"] is True
    assert report["parameter_count"]["formula"] == plan["parameter_count"]
    assert report["shape_plan"]["provenance"] == "deterministic-shape-plan"
    assert report["saved_config"]["model_type"] == "qwen3"
    assert report["saved_config"]["pad_token_id"] == tokenizer.pad_token_id
    assert report["tokenizer"]["supplied_object_saved_directly"] is True
    assert all(len(item["sha256"]) == 64 for item in report["artifacts"]["files"])


def test_module_contains_no_pretrained_model_load_or_training_calls() -> None:
    source_path = (
        Path(__file__).parents[1] / "src" / "neural_forge" / "hf_qwen_target.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported_modules = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "from_pretrained" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert not {"pydantic", "sqlalchemy"} & imported_modules
