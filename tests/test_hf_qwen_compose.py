from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neural_forge.cli import build_parser
import neural_forge.hf_qwen_compose as qwen_compose
from neural_forge.hf_qwen_compose import (
    _fit_linear_prompt_split,
    _packed_donor_attention_output,
    _packed_donor_mlp_output,
    _path_materialization_gate,
    _validation_gate,
    build_contiguous_pair_plan,
    build_coupled_swiglu_packing,
    build_gqa_group_packing,
    composed_pair_endpoint,
    initialize_qwen_pair_composition_models,
    keyword_domain_balanced_texts,
    prior_preserving_trace_scaled_ridge,
    prompt_level_split,
)


def _path_gate_report(
    *,
    baseline_rmse: float,
    fitted_rmse: float,
    baseline_cosine: float = 0.7,
    fitted_cosine: float = 0.72,
) -> dict:
    return {
        "diagnostics": {
            "held_out": {
                "approximate": True,
                "baseline": {
                    "absolute_rmse": baseline_rmse,
                    "cosine_similarity_mean": baseline_cosine,
                },
                "fitted": {
                    "absolute_rmse": fitted_rmse,
                    "cosine_similarity_mean": fitted_cosine,
                },
            }
        }
    }


def test_path_materialization_gate_accepts_real_held_out_gain() -> None:
    gate = _path_materialization_gate(
        _path_gate_report(baseline_rmse=1.0, fitted_rmse=0.8),
        min_relative_improvement=0.1,
        max_cosine_drop=0.0,
    )

    assert gate["accepted"] is True
    assert gate["reverted"] is False
    assert gate["materialization_mode"] == "all-path-aggregate"
    assert gate["relative_rmse_improvement"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("report", "reason"),
    [
        (
            _path_gate_report(baseline_rmse=1.0, fitted_rmse=1.01),
            "held_out_absolute_rmse_not_improved",
        ),
        (
            _path_gate_report(baseline_rmse=1.0, fitted_rmse=0.99),
            "held_out_relative_improvement_below_threshold",
        ),
        (
            _path_gate_report(
                baseline_rmse=1.0,
                fitted_rmse=0.8,
                baseline_cosine=0.8,
                fitted_cosine=0.79,
            ),
            "held_out_cosine_drop_above_threshold",
        ),
    ],
)
def test_path_materialization_gate_reverts_harmful_or_weak_fit(
    report: dict, reason: str
) -> None:
    gate = _path_materialization_gate(
        report,
        min_relative_improvement=0.02,
        max_cosine_drop=0.0,
    )

    assert gate["accepted"] is False
    assert gate["reverted"] is True
    assert gate["reason"] == reason
    assert gate["materialization_mode"] == "selected-native-fallback"


def test_qwen_pair_compose_cli_matches_detached_launcher_surface(monkeypatch) -> None:
    captured = {}

    def fake_initialize(**kwargs):
        captured.update(kwargs)
        return {"zero_gradient": True}

    monkeypatch.setattr(
        qwen_compose, "initialize_qwen_pair_composition", fake_initialize
    )
    args = build_parser().parse_args(
        [
            "hf-qwen-pair-compose",
            "donor",
            "blank",
            "out",
            "--ridge",
            "0.01",
            "--max-tokens-per-layer",
            "512",
            "--max-anchors",
            "2048",
            "--fit-gate-min-validation-rows",
            "32",
            "--no-observable-coordinates",
            "--observable-coordinate-ridge",
            "0.002",
            "--no-path-decomposition",
            "--path-decomposition-interpolation-mode",
            "dense",
            "--rollout-refold-rounds",
            "2",
            "--rollout-refold-max-prompts",
            "12",
            "--rollout-refold-max-new-tokens",
            "7",
            "--rollout-refold-candidate-alphas",
            "1,0.25",
            "--rollout-gate-min-action-count-ratio",
            "0.8",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "64",
            "--device",
            "cpu",
        ]
    )
    monkeypatch.setattr(
        "neural_forge.cli._load_recovery_calibration_texts",
        lambda _path, _limit: ["alpha", "beta", "gamma"],
    )

    report = args.func(args)

    assert report == {"zero_gradient": True}
    assert captured["donor_model"] == "donor"
    assert captured["target_model"] == "blank"
    assert captured["max_fit_rows"] == 512
    assert captured["basis_max_rows"] == 2048
    assert captured["min_validation_rows"] == 32
    assert captured["observable_coordinates"] is False
    assert captured["observable_coordinate_ridge"] == pytest.approx(0.002)
    assert captured["path_decomposition"] is False
    assert captured["path_decomposition_interpolation_mode"] == "dense"
    assert captured["rollout_refold_rounds"] == 2
    assert captured["rollout_refold_max_prompts"] == 12
    assert captured["rollout_refold_max_new_tokens"] == 7
    assert captured["rollout_refold_candidate_alphas"] == [1.0, 0.25]
    assert captured["rollout_gate_min_action_count_ratio"] == pytest.approx(0.8)
    assert captured["calibration_texts"] == ["alpha", "beta", "gamma"]


def test_pair_plan_is_exact_contiguous_36_to_18() -> None:
    plan = build_contiguous_pair_plan(36, 18)

    assert len(plan) == 18
    assert plan[0].donor_layers == (0, 1)
    assert plan[9].donor_layers == (18, 19)
    assert plan[-1].donor_layers == (34, 35)
    assert [index for item in plan for index in item.donor_layers] == list(range(36))

    with pytest.raises(ValueError, match="exactly twice"):
        build_contiguous_pair_plan(35, 18)


def test_keyword_domain_balanced_texts_prevents_prefix_domain_collapse() -> None:
    texts = [f"Question: What is {index} + 1?" for index in range(20)]
    texts.extend(
        [
            "Write a Python function that returns one.",
            "Explain tokenizer vocabulary mismatch.",
            "Explain neuron merging before pruning.",
            "Lightning and thunder require safety.",
            "Summarize this ordinary paragraph.",
        ]
    )

    selected = keyword_domain_balanced_texts(texts, 6)

    assert len(selected) == 6
    assert any("Python" in text for text in selected)
    assert any("tokenizer" in text for text in selected)
    assert any("neuron" in text for text in selected)
    assert any("Lightning" in text for text in selected)
    assert any("ordinary" in text for text in selected)


def test_gqa_plan_packs_complete_128_coordinate_heads_by_group() -> None:
    plan = build_gqa_group_packing(
        donor_query_heads=32,
        donor_key_value_heads=8,
        target_query_heads=8,
        target_key_value_heads=2,
        head_dim=128,
        selected_kv_heads=[3, 5],
    )

    assert plan["query_heads_per_kv"] == 4
    assert plan["available_source_group_count"] == 16
    assert plan["selected_source_group_count"] == 2
    assert plan["discarded_source_group_count"] == 14
    assert plan["covered_query_coordinates"] == 8 * 128
    assert plan["selected_full_head_coordinates_preserved"] is True
    assert plan["nonlinear_head_averaging"] is False
    assert plan["selected_kv_heads"] == [3, 5]
    seen_query_heads = []
    for group in plan["groups"]:
        source = group["source"]
        assert source["donor_kv_slice"][1] - source["donor_kv_slice"][0] == 128
        assert source["target_kv_slice"][1] - source["target_kv_slice"][0] == 128
        for query in source["query_heads"]:
            assert query["donor_slice"][1] - query["donor_slice"][0] == 128
            assert query["target_slice"][1] - query["target_slice"][0] == 128
            seen_query_heads.append((source["pair_offset"], query["donor_query_head"]))
    assert seen_query_heads == [
        (0, 12),
        (0, 13),
        (0, 14),
        (0, 15),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
    ]


def test_gqa_plan_rejects_query_per_kv_ratio_change() -> None:
    with pytest.raises(ValueError, match="query-per-KV ratios"):
        build_gqa_group_packing(32, 8, 8, 1, head_dim=128)


def test_swiglu_gate_up_and_down_use_identical_channel_indices() -> None:
    plan = build_coupled_swiglu_packing(11, 5, pair_width=2)

    assert plan["gate_up_down_indices_coupled"] is True
    assert plan["all_source_channels_packed"] is False
    assert plan["nonlinear_channel_averaging"] is False
    assert plan["selected_per_donor_layer"] == [3, 2]
    covered = []
    for target in plan["target_channels"]:
        assert target["gate_indices"] == target["up_indices"] == target["down_indices"]
        assert len(target["sources"]) == 1
        covered.extend(tuple(item) for item in target["sources"])
    assert covered == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]


def test_swiglu_plan_accepts_explicit_post_nonlinearity_support() -> None:
    selected = [(0, 4), (1, 2), (0, 1), (1, 5)]
    plan = build_coupled_swiglu_packing(
        7,
        4,
        pair_width=2,
        selected_sources=selected,
    )

    assert plan["selection_source"] == "explicit-post-nonlinearity-support"
    assert plan["selected_per_donor_layer"] == [2, 2]
    assert [tuple(item["sources"][0]) for item in plan["target_channels"]] == selected


def test_donor_attention_outputs_pack_the_selected_complete_groups() -> None:
    torch = pytest.importorskip("torch")
    plan = build_gqa_group_packing(
        donor_query_heads=4,
        donor_key_value_heads=2,
        target_query_heads=4,
        target_key_value_heads=2,
        head_dim=2,
        selected_kv_heads=[1, 0],
    )
    first = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    second = first + 100.0

    packed_q = _packed_donor_attention_output(torch, first, plan, "q", 0)
    packed_q.add_(_packed_donor_attention_output(torch, second, plan, "q", 1))
    packed_k = _packed_donor_attention_output(torch, first[..., :4], plan, "k", 0)
    packed_k.add_(_packed_donor_attention_output(torch, second[..., :4], plan, "k", 1))

    torch.testing.assert_close(
        packed_q, torch.tensor([[[4.0, 5.0, 6.0, 7.0, 100.0, 101.0, 102.0, 103.0]]])
    )
    torch.testing.assert_close(packed_k, torch.tensor([[[2.0, 3.0, 100.0, 101.0]]]))


def test_donor_mlp_outputs_pack_selected_channels_from_both_layers() -> None:
    torch = pytest.importorskip("torch")
    plan = build_coupled_swiglu_packing(
        donor_intermediate_size=4,
        target_intermediate_size=4,
        pair_width=2,
        channel_scores=[[1.0, 4.0, 3.0, 2.0], [8.0, 6.0, 7.0, 5.0]],
    )
    first = torch.tensor([[[10.0, 11.0, 12.0, 13.0]]])
    second = torch.tensor([[[20.0, 21.0, 22.0, 23.0]]])

    packed = _packed_donor_mlp_output(torch, first, plan, 0)
    packed.add_(_packed_donor_mlp_output(torch, second, plan, 1))

    torch.testing.assert_close(packed, torch.tensor([[[11.0, 12.0, 20.0, 22.0]]]))


def test_prompt_split_holds_out_whole_prompts_deterministically() -> None:
    prompts = [f"prompt-{index}" for index in range(12)]
    first = prompt_level_split(
        prompts, validation_fraction=0.25, min_validation_prompts=2, seed=17
    )
    second = prompt_level_split(
        prompts, validation_fraction=0.25, min_validation_prompts=2, seed=17
    )

    assert first == second
    assert first["unit"] == "prompt"
    assert len(first["validation_indices"]) == 3
    assert set(first["train_indices"]).isdisjoint(first["validation_indices"])
    assert sorted(first["train_indices"] + first["validation_indices"]) == list(
        range(len(prompts))
    )


def test_trace_scaled_fp64_ridge_is_stable_and_prior_preserving() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        dtype=torch.float32,
    )
    target = torch.stack([3.0 * x[:, 0], -2.0 * x[:, 0]], dim=1)
    prior = torch.tensor([[0.5, 0.5], [-0.25, -0.25]], dtype=torch.float32)

    fitted, bias, report = prior_preserving_trace_scaled_ridge(
        x,
        target,
        prior,
        ridge=1e-6,
        fit_bias=False,
        torch_module=torch,
    )

    assert bias is None
    assert report["dtype"] == "float64"
    assert report["ridge_mode"] == "trace-scaled"
    assert report["effective_ridge"] > 0.0
    assert torch.isfinite(fitted).all()
    prior_error = (
        (x.double().matmul(prior.double().t()) - target.double()).pow(2).mean()
    )
    fitted_error = (x.double().matmul(fitted.t()) - target.double()).pow(2).mean()
    assert fitted_error < prior_error * 1e-6


def test_validation_gate_reverts_when_ridge_cannot_beat_prior() -> None:
    torch = pytest.importorskip("torch")
    inputs = [torch.tensor([[1.0], [2.0]]) for _ in range(4)]
    prior = torch.tensor([[3.0]])
    targets = [rows.matmul(prior.t()) for rows in inputs]

    result = _fit_linear_prompt_split(
        torch,
        inputs,
        targets,
        train_indices=[0, 1, 2],
        validation_indices=[3],
        prior_weight=prior,
        prior_bias=None,
        ridge=1e-3,
        min_relative_improvement=0.0,
        max_cosine_drop=0.01,
    )

    assert result["report"]["accepted"] is False
    assert result["report"]["reverted"] is True
    assert result["report"]["reason"] == "validation_mse_not_improved"
    torch.testing.assert_close(result["weight"], prior.double())


def test_validation_gate_rejects_better_but_absolutely_bad_fit() -> None:
    accepted, reason, diagnostics = _validation_gate(
        {"mse": 100.0, "mean_row_cosine": 0.1, "relative_rmse": 1.5},
        {"mse": 25.0, "mean_row_cosine": 0.9, "relative_rmse": 0.9},
        min_relative_improvement=0.02,
        max_cosine_drop=0.0,
        max_fitted_relative_rmse=0.75,
    )

    assert accepted is False
    assert reason == "validation_relative_rmse_above_ceiling"
    assert diagnostics["validation_relative_improvement"] == pytest.approx(0.75)


def test_composed_endpoint_calls_second_layer_on_first_layer_output() -> None:
    torch = pytest.importorskip("torch")

    class First(torch.nn.Module):
        def forward(self, values, **_kwargs):
            return values + 2.0

    class Second(torch.nn.Module):
        def forward(self, values, **_kwargs):
            return values.square() - 1.0

    values = torch.tensor([[1.0, 3.0]])
    endpoint = composed_pair_endpoint(First(), Second(), values)

    torch.testing.assert_close(endpoint, (values + 2.0).square() - 1.0)
    assert not torch.equal(endpoint, (First()(values) + Second()(values)) / 2.0)


def test_tiny_qwen_pair_composition_runs_end_to_end_on_current_child_prefix() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    class TinyTokenizer:
        def __init__(self, vocab_size: int = 17) -> None:
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"

        def __len__(self) -> int:
            return self.vocab_size

        def get_vocab(self):
            return {f"token-{index}": index for index in range(self.vocab_size)}

        def encode(self, text, add_special_tokens=True):
            values = [
                2 + (ord(character) % (self.vocab_size - 2)) for character in str(text)
            ]
            if add_special_tokens:
                values.append(self.eos_token_id)
            return values or [self.eos_token_id]

        def __call__(
            self,
            texts,
            return_tensors=None,
            padding=True,
            truncation=True,
            max_length=32,
        ):
            if isinstance(texts, str):
                texts = [texts]
            rows = [self.encode(text)[: int(max_length)] for text in texts]
            width = max(len(row) for row in rows) if padding else None
            padded = []
            masks = []
            for row in rows:
                row_width = width if width is not None else len(row)
                padded.append(row + [self.pad_token_id] * (row_width - len(row)))
                masks.append([1] * len(row) + [0] * (row_width - len(row)))
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": masks}

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = 1e-6

        def forward(self, values):
            variance = values.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = values.float() * torch.rsqrt(variance + self.variance_epsilon)
            return normalized.to(values.dtype) * self.weight

    class TinyAttention(torch.nn.Module):
        def __init__(
            self, hidden_size: int, query_heads: int, kv_heads: int, head_dim: int
        ) -> None:
            super().__init__()
            self.num_heads = query_heads
            self.num_key_value_heads = kv_heads
            self.head_dim = head_dim
            self.q_proj = torch.nn.Linear(
                hidden_size, query_heads * head_dim, bias=False
            )
            self.k_proj = torch.nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
            self.v_proj = torch.nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
            self.o_proj = torch.nn.Linear(
                query_heads * head_dim, hidden_size, bias=False
            )
            self.q_norm = TinyRMSNorm(head_dim)
            self.k_norm = TinyRMSNorm(head_dim)

        def forward(self, values, **_kwargs):
            batch, length, _hidden = values.shape
            ratio = self.num_heads // self.num_key_value_heads
            query = self.q_proj(values).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            key = self.k_proj(values).reshape(
                batch, length, self.num_key_value_heads, self.head_dim
            )
            value = self.v_proj(values).reshape(
                batch, length, self.num_key_value_heads, self.head_dim
            )
            query = self.q_norm(query)
            key = self.k_norm(key).repeat_interleave(ratio, dim=2)
            value = value.repeat_interleave(ratio, dim=2)
            mixed = torch.tanh(query + key) + value
            return self.o_proj(mixed.reshape(batch, length, -1))

    class TinyMLP(torch.nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, values):
            return self.down_proj(
                torch.nn.functional.silu(self.gate_proj(values)) * self.up_proj(values)
            )

    class TinyLayer(torch.nn.Module):
        def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            query_heads: int,
            kv_heads: int,
            head_dim: int,
        ):
            super().__init__()
            self.input_layernorm = TinyRMSNorm(hidden_size)
            self.self_attn = TinyAttention(hidden_size, query_heads, kv_heads, head_dim)
            self.post_attention_layernorm = TinyRMSNorm(hidden_size)
            self.mlp = TinyMLP(hidden_size, intermediate_size)

        def forward(self, hidden_states, **kwargs):
            hidden_states = hidden_states + self.self_attn(
                self.input_layernorm(hidden_states), **kwargs
            )
            return hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states)
            )

    class TinyBackbone(torch.nn.Module):
        def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            layers,
            query_heads,
            kv_heads,
            head_dim,
        ):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.layers = torch.nn.ModuleList(
                [
                    TinyLayer(
                        hidden_size, intermediate_size, query_heads, kv_heads, head_dim
                    )
                    for _ in range(layers)
                ]
            )
            self.norm = TinyRMSNorm(hidden_size)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden, attention_mask=attention_mask, **kwargs)
            return SimpleNamespace(last_hidden_state=self.norm(hidden))

    class TinyCausalLM(torch.nn.Module):
        def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            layers,
            query_heads,
            kv_heads,
            head_dim,
        ):
            super().__init__()
            self.model = TinyBackbone(
                vocab_size,
                hidden_size,
                intermediate_size,
                layers,
                query_heads,
                kv_heads,
                head_dim,
            )
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            self.config = SimpleNamespace(use_cache=False, tie_word_embeddings=False)

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, module):
            self.lm_head = module

        def resize_token_embeddings(self, _size):
            raise AssertionError("Tiny test vocabulary should not be resized.")

    tokenizer = TinyTokenizer()
    donor = TinyCausalLM(
        17,
        hidden_size=8,
        intermediate_size=6,
        layers=4,
        query_heads=4,
        kv_heads=2,
        head_dim=2,
    )
    target = TinyCausalLM(
        17,
        hidden_size=4,
        intermediate_size=3,
        layers=2,
        query_heads=4,
        kv_heads=2,
        head_dim=2,
    )
    prompts = ["alpha", "bravo", "charlie", "delta", "echo"]

    report = initialize_qwen_pair_composition_models(
        donor,
        target,
        tokenizer,
        tokenizer,
        calibration_texts=prompts,
        ridge=1e-3,
        validation_fraction=0.2,
        min_validation_prompts=1,
        min_train_prompts=2,
        max_length=12,
        batch_size=2,
        max_fit_rows=64,
        min_validation_rows=1,
        basis_max_rows=17,
        basis_mode="embedding-grouped",
        head_validation_vocab_rows=17,
        expected_donor_layers=4,
        expected_target_layers=2,
        required_head_dim=2,
        device="cpu",
    )

    assert report["pair_plan"] == [
        {"target_layer": 0, "donor_layers": [0, 1], "contiguous": True},
        {"target_layer": 1, "donor_layers": [2, 3], "contiguous": True},
    ]
    assert report["zero_gradient"] is True
    assert report["optimizer_steps"] == 0
    assert report["optimizer_constructed"] is False
    assert report["parameters_with_grad"] == 0
    assert report["unwritten_target_parameters"] == 0
    assert report["residual_basis_scope"] == "layer-varying-logit-observable"
    assert report["observable_coordinates"] is True
    assert report["observable_coordinate_report"]["enabled"] is True
    assert report["post_nonlinearity_path_decomposition"] is True
    assert report["path_decomposition_interpolation_mode"] == "sparse-signature"
    assert all(
        item["mlp"]["post_nonlinearity_path_decomposition"]["interpolation_mode"]
        == "sparse-signature"
        for item in report["structural_initialization"]["pairs"]
    )
    assert report["sequential_on_policy_child_prefix_tracing"] is False
    assert report["sequential_child_depth_prefix_retrace"] is True
    assert report["trace_distribution"] == "fixed-calibration-teacher-forced"
    assert report["student_generated_token_prefixes"] is False
    assert report["composed_endpoint_targets"] is True
    assert report["nonlinear_producer_path_fitting"] is True
    assert report["attention_producer_fit_transaction"] == "all-qkv-or-none-per-pair"
    assert report["mlp_producer_fit_transaction"] == "gate-and-up-or-none-per-pair"
    assert all(item["sequential_prefix_trace"] for item in report["layers"])
    assert all(
        item["trace_distribution"] == "fixed-calibration-teacher-forced"
        and item["student_generated_token_prefixes"] is False
        for item in report["layers"]
    )
    assert all("attention_producer_fits" in item for item in report["layers"])
    assert all("mlp_producer_fits" in item for item in report["layers"])
    assert all(
        item["attention_target"]
        == "sum-of-both-donor-attention-residual-deltas"
        for item in report["layers"]
    )
    assert all(
        set(item["attention_producer_transaction"])
        >= {"accepted", "reverted", "producer_applied", "output_applied"}
        for item in report["layers"]
    )
    assert all(
        set(item["mlp_producer_transaction"])
        >= {"accepted", "reverted", "producer_applied", "output_applied"}
        for item in report["layers"]
    )
    assert all("reverted" in item["pair_validation"] for item in report["layers"])
    assert report["accepted_pair_fits"] + report["reverted_pair_fits"] == 2
    assert 0 <= report["accepted_attention_producer_transactions"] <= 2
    assert 0 <= report["accepted_mlp_producer_transactions"] <= 2
    assert "reverted" in report["final_norm_fit"]
    assert "reverted" in report["projected_final_head_fit"]
    assert report["final_head_fit"]["zero_gradient"] is True
    assert 0.0 <= report["final_head_fit"]["chosen_alpha"] <= 1.0
    assert all(torch.isfinite(parameter).all() for parameter in target.parameters())


def test_tiny_transformers_qwen_uses_real_layer_contract(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __len__(self):
            return 19

        def get_vocab(self):
            return {f"token-{index}": index for index in range(19)}

        def encode(self, text, add_special_tokens=True):
            rows = [2 + (ord(character) % 17) for character in str(text)]
            return rows + ([self.eos_token_id] if add_special_tokens else [])

        def __call__(
            self,
            texts,
            return_tensors=None,
            padding=True,
            truncation=True,
            max_length=16,
        ):
            if isinstance(texts, str):
                texts = [texts]
            rows = [self.encode(text)[: int(max_length)] for text in texts]
            width = max(len(row) for row in rows)
            padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
            masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    def qwen(hidden_size, intermediate_size, layers, query_heads, kv_heads):
        config = transformers.Qwen3Config(
            vocab_size=19,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=query_heads,
            num_key_value_heads=kv_heads,
            head_dim=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        return transformers.Qwen3ForCausalLM(config)

    donor = qwen(8, 12, 4, 4, 2)
    target = qwen(4, 6, 2, 4, 2)
    tokenizer = Tokenizer()
    report = initialize_qwen_pair_composition_models(
        donor,
        target,
        tokenizer,
        tokenizer,
        calibration_texts=["alpha", "bravo", "charlie", "delta", "echo"],
        validation_fraction=0.2,
        min_train_prompts=1,
        max_length=12,
        batch_size=2,
        max_fit_rows=64,
        min_validation_rows=1,
        basis_max_rows=19,
        head_validation_vocab_rows=19,
        rollout_refold_rounds=1,
        rollout_refold_max_prompts=2,
        rollout_refold_max_new_tokens=2,
        rollout_refold_batch_size=2,
        rollout_refold_candidate_alphas=(1.0, 0.5),
        rollout_gate_min_action_count_ratio=0.5,
        expected_donor_layers=4,
        expected_target_layers=2,
        required_head_dim=2,
        device="cpu",
        output_dir=tmp_path,
    )

    assert report["zero_gradient"] is True
    assert report["unwritten_target_parameters"] == 0
    assert report["explicitly_preserved_parameter_count"] == 0
    assert report["gqa_selected_full_head_coordinate_packing"] is True
    assert report["nonlinear_producer_parameter_averaging"] is False
    assert report["swiglu_gate_up_channel_coupling"] is True
    assert report["rollout_refold"]["enabled"] is True
    assert report["rollout_refold"]["token_contract"]["exact_token_ids"] is True
    assert report["rollout_refold"]["trace_distribution"] == (
        "current-child-greedy-rollout"
    )
    saved_report = json.loads(
        (tmp_path / "neural_forge_qwen_pair_compose_report.json").read_text()
    )
    assert saved_report["optimizer_steps"] == 0
    assert saved_report["unwritten_target_parameters"] == 0
