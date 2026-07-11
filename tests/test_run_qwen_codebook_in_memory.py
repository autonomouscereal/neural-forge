from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_qwen_codebook_in_memory.py"
)
SPEC = importlib.util.spec_from_file_location("run_qwen_codebook_in_memory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_checkpoint_io_lease_has_a_stable_cross_process_lock_file(tmp_path) -> None:
    lock = tmp_path / "shared" / ".compact_checkpoint_io.lock"
    with MODULE._checkpoint_io_lease(lock) as lease:
        assert lock.is_file()
        assert lease["path"] == str(lock)
        assert lease["wait_seconds"] >= 0.0
        assert lease["serialized"] is (MODULE.os.name == "posix")
    with MODULE._checkpoint_io_lease(lock) as second:
        assert second["wait_seconds"] >= 0.0


def test_locally_staged_checkpoint_is_hash_verified_and_atomically_published(
    tmp_path,
) -> None:
    source = tmp_path / "local" / "compact_child"
    source.mkdir(parents=True)
    files = {
        "model.safetensors": b"bounded-test-tensors",
        "config.json": b"{}",
        "tokenizer_contract.json": b'{"vocab_size":7}',
    }
    for name, payload in files.items():
        (source / name).write_bytes(payload)
    manifest = {
        "files": {
            name: MODULE._sha256_file(source / name)
            for name in sorted(files)
        }
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    destination = tmp_path / "share" / "compact_child"

    report = MODULE._publish_checkpoint_directory(source, destination)

    assert report["atomic_directory_publish"] is True
    assert report["destination"] == str(destination)
    assert destination.is_dir()
    assert MODULE._verify_checkpoint_file_hashes(destination) == manifest["files"]
    assert not list(destination.parent.glob(".compact_child.publish.*"))
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="already exists"):
        MODULE._publish_checkpoint_directory(source, destination)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        pieces = [3 + (sum(piece.encode("utf-8")) % 23) for piece in text.split()]
        return (
            ([1] if add_special_tokens else [])
            + pieces
            + ([2] if add_special_tokens else [])
        )

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert truncation is True
        values = self.encode(text, add_special_tokens=add_special_tokens)[:max_length]
        input_ids = torch.tensor([values], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class _ExactTokenizer:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = aliases or {}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        normalized = self.aliases.get(text, text)
        values = list(normalized.encode("utf-8"))
        return ([1] if add_special_tokens else []) + values + (
            [2] if add_special_tokens else []
        )


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.logits)


class _MLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        self.double_down = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        output = self.down_proj(hidden)
        if self.double_down:
            self.down_proj(hidden)
        return output


class _Attention(torch.nn.Module):
    def __init__(
        self,
        hidden: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        query_width = query_heads * head_dim
        key_value_width = key_value_heads * head_dim
        self.q_proj = torch.nn.Linear(hidden, query_width, bias=False)
        self.k_proj = torch.nn.Linear(hidden, key_value_width, bias=False)
        self.v_proj = torch.nn.Linear(hidden, key_value_width, bias=False)
        self.o_proj = torch.nn.Linear(query_width, hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(inputs)
        self.k_proj(inputs)
        self.v_proj(inputs)
        return self.o_proj(query)


class _Layer(torch.nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(
            hidden,
            query_heads,
            key_value_heads,
            head_dim,
        )
        self.mlp = _MLP(hidden, intermediate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.self_attn(inputs)
        return hidden + self.mlp(hidden)


class _Backbone(torch.nn.Module):
    def __init__(
        self,
        layer_count: int,
        hidden: int,
        intermediate: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, hidden)
        self.layers = torch.nn.ModuleList(
            [
                _Layer(
                    hidden,
                    intermediate,
                    query_heads,
                    key_value_heads,
                    head_dim,
                )
                for _ in range(layer_count)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert input_ids.shape == attention_mask.shape
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeQwen(torch.nn.Module):
    def __init__(self, layer_count: int = 3) -> None:
        super().__init__()
        hidden = 4
        intermediate = 5
        query_heads = 4
        key_value_heads = 2
        head_dim = 1
        self.config = SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=layer_count,
            num_attention_heads=query_heads,
            num_key_value_heads=key_value_heads,
            head_dim=head_dim,
        )
        self.model = _Backbone(
            layer_count,
            hidden,
            intermediate,
            query_heads,
            key_value_heads,
            head_dim,
        )


def _fake_qwen(layer_count: int = 3) -> _FakeQwen:
    torch.manual_seed(701)
    return _FakeQwen(layer_count)


def test_token_disjoint_selection_fills_duplicates_and_eval_overlap() -> None:
    prompts = [
        "alpha",
        "duplicate-a",
        "held-out",
        "beta",
        "duplicate-b",
        "gamma",
        "delta",
        "epsilon",
    ]
    source = _ExactTokenizer(
        {"duplicate-a": "shared", "duplicate-b": "shared"}
    )
    target = _ExactTokenizer()
    selected, report = MODULE.select_token_disjoint_calibration_prompts(
        prompts,
        5,
        source_tokenizer=source,
        target_tokenizer=target,
        eval_suite={"text_evals": ["held-out"]},
    )

    assert len(selected) == 5
    assert "held-out" not in selected
    assert not {"duplicate-a", "duplicate-b"}.issubset(selected)
    assert len({tuple(source.encode(text)) for text in selected}) == 5
    assert len({tuple(target.encode(text)) for text in selected}) == 5
    assert report["selected_prompt_count"] == 5
    assert sum(report["rejected"].values()) >= 1
    MODULE.verify_token_split_disjoint(
        source,
        selected,
        {"text_evals": ["held-out"]},
    )


def test_token_disjoint_selection_rejects_insufficient_unique_paths() -> None:
    tokenizer = _ExactTokenizer({"one": "same", "two": "same", "three": "same"})
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="usable unique"):
        MODULE.select_token_disjoint_calibration_prompts(
            ["one", "two", "three"],
            2,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            eval_suite={"text_evals": []},
        )


def test_token_disjoint_selection_uses_truncated_token_paths() -> None:
    tokenizer = _ExactTokenizer({"one": "shared suffix-one", "two": "shared suffix-two"})
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="usable unique"):
        MODULE.select_token_disjoint_calibration_prompts(
            ["one", "two"],
            2,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            eval_suite={"text_evals": []},
            max_length=3,
        )


def test_noop_logit_gate_reports_fp16_rounding_without_claiming_exactness() -> None:
    expected = torch.tensor([[[4.0, 1.0, -2.0]]], dtype=torch.float16)
    actual = expected.clone()
    actual[..., 1] += torch.tensor(0.015625, dtype=torch.float16)
    report = MODULE._verify_noop_logits(
        _FixedLogitModel(actual),
        MODULE.NoopReference({}, expected),
        device="cpu",
    )

    assert report["logits_exact"] is False
    assert report["logits_bitwise_exact"] is False
    assert report["logits_numerically_equivalent"] is True
    assert report["top1_exact"] is True
    assert report["top_k_exact"] is True
    assert report["probabilities_numerically_equivalent"] is True
    assert report["maximum_absolute_logit_error"] == pytest.approx(0.015625)


def test_noop_logit_gate_rejects_changed_top1() -> None:
    expected = torch.tensor([[[2.0, 1.999]]], dtype=torch.float16)
    actual = torch.tensor([[[1.999, 2.0]]], dtype=torch.float16)
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="top1_matches=0/1"):
        MODULE._verify_noop_logits(
            _FixedLogitModel(actual),
            MODULE.NoopReference({}, expected),
            device="cpu",
        )


@torch.no_grad()
def _couple_permuted_layers(
    model: _FakeQwen,
    neuron_permutations: torch.Tensor,
    group_permutations: torch.Tensor,
) -> None:
    source = model.model.layers[0]
    head_dim = int(model.config.head_dim)
    query_rows = (
        int(model.config.num_attention_heads)
        // int(model.config.num_key_value_heads)
        * head_dim
    )
    for layer_index, target in enumerate(model.model.layers):
        if layer_index == 0:
            continue
        for canonical, physical in enumerate(neuron_permutations[layer_index].tolist()):
            target.mlp.gate_proj.weight[physical].copy_(
                source.mlp.gate_proj.weight[canonical]
            )
            target.mlp.up_proj.weight[physical].copy_(
                source.mlp.up_proj.weight[canonical]
            )
            target.mlp.down_proj.weight[:, physical].copy_(
                source.mlp.down_proj.weight[:, canonical]
            )
        for canonical, physical in enumerate(group_permutations[layer_index].tolist()):
            canonical_q = slice(canonical * query_rows, (canonical + 1) * query_rows)
            physical_q = slice(physical * query_rows, (physical + 1) * query_rows)
            canonical_kv = slice(canonical * head_dim, (canonical + 1) * head_dim)
            physical_kv = slice(physical * head_dim, (physical + 1) * head_dim)
            target.self_attn.q_proj.weight[physical_q].copy_(
                source.self_attn.q_proj.weight[canonical_q]
            )
            target.self_attn.k_proj.weight[physical_kv].copy_(
                source.self_attn.k_proj.weight[canonical_kv]
            )
            target.self_attn.v_proj.weight[physical_kv].copy_(
                source.self_attn.v_proj.weight[canonical_kv]
            )
            target.self_attn.o_proj.weight[:, physical_q].copy_(
                source.self_attn.o_proj.weight[:, canonical_q]
            )


def test_bounded_activation_collection_is_deterministic_and_streamed() -> None:
    model = _fake_qwen()
    model.train()
    tokenizer = _Tokenizer()
    prompts = ["alpha beta", "gamma delta epsilon", "zeta eta theta iota"]

    first = MODULE.collect_bounded_qwen_activations(
        model,
        tokenizer,
        prompts,
        max_length=8,
        activation_rows=7,
        device="cpu",
    )
    second = MODULE.collect_bounded_qwen_activations(
        model,
        tokenizer,
        prompts,
        max_length=8,
        activation_rows=7,
        device="cpu",
    )

    assert model.training is True
    assert first.report == second.report
    assert first.report["bounded"] is True
    assert first.report["streaming_batch_size"] == 1
    assert first.report["materialized_full_depth_activation_tensor"] is False
    assert first.report["selected_rows_per_layer"] == 7
    assert first.report["hook_calls_per_neuron_layer"] == [3, 3, 3]
    assert first.report["hook_calls_per_attention_layer"] == [3, 3, 3]
    assert first.report["cached_elements"] == first.report["cached_element_limit"]
    assert (
        first.report["peak_host_activation_bytes"]
        >= first.report["retained_host_activation_bytes"]
    )
    for left, right in zip(
        first.neuron_activations, second.neuron_activations, strict=True
    ):
        assert left.shape == (7, 5)
        assert left.device.type == "cpu"
        assert left.dtype == torch.float32
        assert torch.equal(left, right)
    for left, right in zip(
        first.attention_activations, second.attention_activations, strict=True
    ):
        assert left.shape == (7, 4)
        assert torch.equal(left, right)
    for layer in model.model.layers:
        assert not layer.mlp.down_proj._forward_pre_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_activation_collection_enforces_peak_host_cap() -> None:
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="Peak host activation"):
        MODULE.collect_bounded_qwen_activations(
            _fake_qwen(),
            _Tokenizer(),
            ["one two", "three four"],
            max_length=8,
            activation_rows=4,
            device="cpu",
            max_host_activation_bytes=1,
        )


def test_activation_collection_fails_closed_on_rows_hooks_and_nonfinite() -> None:
    tokenizer = _Tokenizer()
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="Insufficient"):
        MODULE.collect_bounded_qwen_activations(
            _fake_qwen(),
            tokenizer,
            ["one"],
            max_length=8,
            activation_rows=8,
            device="cpu",
        )

    mismatched = _fake_qwen()
    mismatched.model.layers[1].mlp.double_down = True
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="hook mismatch"):
        MODULE.collect_bounded_qwen_activations(
            mismatched,
            tokenizer,
            ["one two"],
            max_length=8,
            activation_rows=2,
            device="cpu",
        )
    assert all(
        not layer.mlp.down_proj._forward_pre_hooks for layer in mismatched.model.layers
    )

    nonfinite = _fake_qwen()
    with torch.no_grad():
        nonfinite.model.layers[0].mlp.gate_proj.weight[0, 0] = torch.nan
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="NaN or infinity"):
        MODULE.collect_bounded_qwen_activations(
            nonfinite,
            tokenizer,
            ["one two"],
            max_length=8,
            activation_rows=2,
            device="cpu",
        )


def test_bundle_alignment_recovers_coupled_neuron_and_gqa_permutations() -> None:
    model = _fake_qwen()
    neuron_permutations = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [2, 4, 0, 3, 1],
            [4, 1, 3, 0, 2],
        ],
        dtype=torch.int64,
    )
    group_permutations = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64)
    _couple_permuted_layers(model, neuron_permutations, group_permutations)
    bundles = MODULE.build_qwen3_bundles(model)

    first = MODULE.infer_codebook_permutations(
        bundles,
        neuron_activations=None,
        attention_activations=None,
        neuron_weight_sketch_size=64,
        neuron_activation_sketch_size=0,
        attention_weight_sketch_size=64,
        attention_activation_sketch_size=0,
        local_block_size=8,
    )
    second = MODULE.infer_codebook_permutations(
        bundles,
        neuron_activations=None,
        attention_activations=None,
        neuron_weight_sketch_size=64,
        neuron_activation_sketch_size=0,
        attention_weight_sketch_size=64,
        attention_activation_sketch_size=0,
        local_block_size=8,
    )

    assert torch.equal(first.neurons.permutations, neuron_permutations)
    assert torch.equal(first.attention.permutations, group_permutations)
    assert torch.equal(first.neurons.permutations, second.neurons.permutations)
    assert torch.equal(first.attention.permutations, second.attention.permutations)
    assert first.neurons.diagnostics.permutation_proof is True
    assert first.attention.diagnostics.permutation_proof is True
    assert first.neurons.diagnostics.materialized_full_pairwise_matrix is False
    assert first.attention.diagnostics.exact_small_group_assignment is True
    assert first.attention.diagnostics.maximum_pairwise_block_elements <= 2**2


def test_token_split_overlap_fails_closed(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "text_evals": ["alpha beta"],
                "choice_probes": [],
                "generation_prompts": [],
                "behavior_probes": [],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = _Tokenizer()

    with pytest.raises(MODULE.QwenCodebookRunnerError, match="token splits overlap"):
        MODULE.verify_token_split_disjoint(tokenizer, ["alpha beta"], suite)

    report = MODULE.verify_token_split_disjoint(tokenizer, ["gamma delta"], suite)
    assert report["disjoint"] is True
    assert report["overlap_count"] == 0
    assert len(report["calibration_token_split_sha256"]) == 64


def test_parser_exposes_production_controls_and_full_depth_alias(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "Qwen/Qwen3-4B-Base",
            "--source-tokenizer",
            "source-tokenizer",
            "--target-tokenizer",
            "target-tokenizer",
            "--output-dir",
            str(tmp_path / "run"),
            "--calibration-jsonl",
            str(tmp_path / "calibration.jsonl"),
            "--eval-suite",
            str(tmp_path / "eval.json"),
            "--cached-parent-eval",
            str(tmp_path / "parent.json"),
            "--calibration-count",
            "48",
            "--calibration-max-length",
            "192",
            "--activation-rows",
            "96",
            "--p-neuron",
            "34",
            "--p-attention",
            "logical_layers",
            "--neuron-weight-sketch-size",
            "80",
            "--attention-weight-sketch-size",
            "72",
            "--dtype",
            "bfloat16",
            "--device",
            "cuda:3",
        ]
    )

    assert args.source_tokenizer == "source-tokenizer"
    assert args.target_tokenizer == "target-tokenizer"
    assert args.calibration_count == 48
    assert args.calibration_max_length == 192
    assert args.activation_rows == 96
    assert args.p_neuron == 34
    assert args.p_attention == "logical_layers"
    assert MODULE._resolve_prototype_depth(args.p_neuron, 36, "p_neuron") == 34
    assert MODULE._resolve_prototype_depth(args.p_attention, 36, "p_attention") == 36
    assert args.neuron_weight_sketch_size == 80
    assert args.attention_weight_sketch_size == 72
    assert args.dtype == "bfloat16"
    assert args.device == "cuda:3"
    assert args.max_text_evals == 1000
    assert args.max_behavior_probes == 1000
    assert args.max_generation_prompts == 1000
    assert args.max_choice_probes == 250
    assert args.manual_sample_count == 128
    assert args.max_host_activation_bytes > 0
    assert args.max_cuda_allocated_bytes > 0
    args.max_generation_prompts = 0
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="positive integer"):
        MODULE._validate_args(args)


def test_run_report_shape_and_invariants(tmp_path: Path) -> None:
    alignment = {
        "neuron": {
            "canonical_to_physical_sha256": "a" * 64,
            "diagnostics": {"permutation_proof": True},
        },
        "attention": {
            "canonical_to_physical_sha256": "b" * 64,
            "diagnostics": {"permutation_proof": True},
        },
        "invariants": {"all_passed": True},
    }
    comparison = {
        "parent": {"behavior_correct": 1000},
        "child": {"behavior_correct": 900},
        "param_compression_ratio": 1.1,
        "behavior": {"overall": {"retained_parent_pass": 900}},
        "generation": {"overall": {"retained_parent_pass": 875}},
        "choice": {"overall": {"retained_parent_pass": 225}},
        "manual_samples": {
            "behavior": [{"id": "code-retained"}],
            "generation": [],
            "selected_count": 1,
            "all_available_strata_represented": True,
            "strata": {"behavior:code:retained_pass": {"selected": 1}},
        },
    }
    report = MODULE.build_run_report(
        source_model="parent",
        source_tokenizer="source",
        target_tokenizer="target",
        p_neuron=34,
        p_attention=32,
        calibration={"bounded": True, "selected_rows_per_layer": 64},
        alignment=alignment,
        alignment_path=tmp_path / "alignment.json",
        compile_report={"logical_layers": 36},
        accounting={"invariants": {"all_passed": True}},
        verification={"all_passed": True},
        candidate_eval={"param_count": 123},
        comparison=comparison,
        comparison_path=tmp_path / "comparison.json",
        manual_sample_count=1,
        certification_contract={"contract": {"counts": {"text_evals": 1}}},
    )

    assert report["schema"] == MODULE.RUN_SCHEMA
    assert report["configuration"]["p_neuron"] == 34
    assert report["configuration"]["p_attention"] == 32
    assert report["alignment"]["report"] == str(tmp_path / "alignment.json")
    assert report["aggregate"]["choice"]["retained_parent_pass"] == 225
    assert report["manual_samples"] == comparison["manual_samples"]
    assert report["zero_optimizer_steps"] == 0
    assert report["backward_called"] is False
    assert report["status"] == "complete"
    MODULE.validate_run_report(report)

    report["calibration"]["bounded"] = False
    with pytest.raises(MODULE.QwenCodebookRunnerError, match="bounded calibration"):
        MODULE.validate_run_report(report)


def test_manual_samples_are_category_and_outcome_stratified() -> None:
    def item(item_id: str, category: str, passed: bool) -> dict[str, object]:
        return {
            "id": item_id,
            "category": category,
            "prompt": item_id,
            "passed": passed,
        }

    parent = {
        "behavior_suite": {
            "items": [
                item("code-loss", "code", True),
                item("code-retained", "code", True),
                item("model-gain", "model_surgery", False),
                item("model-shared", "model_surgery", False),
            ]
        },
        "generations": [
            item("generation-loss", "generation", True),
            item("generation-retained", "generation", True),
        ],
        "generation_diagnostics": [
            item("generation-loss", "generation", True),
            item("generation-retained", "generation", True),
        ],
    }
    child = {
        "behavior_suite": {
            "items": [
                item("code-loss", "code", False),
                item("code-retained", "code", True),
                item("model-gain", "model_surgery", True),
                item("model-shared", "model_surgery", False),
            ]
        },
        "generations": [
            item("generation-loss", "generation", False),
            item("generation-retained", "generation", True),
        ],
        "generation_diagnostics": [
            item("generation-loss", "generation", False),
            item("generation-retained", "generation", True),
        ],
    }

    report = MODULE._stratified_manual_samples(parent, child, 6)

    assert report["selected_count"] == 6
    assert report["all_available_strata_represented"] is True
    assert {
        "behavior:code:loss",
        "behavior:code:retained_pass",
        "behavior:model_surgery:gain",
        "behavior:model_surgery:shared_failure",
        "generation:generation:loss",
        "generation:generation:retained_pass",
    } == set(report["strata"])


def test_atomic_json_write_replaces_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text('{"stale":true}\n', encoding="utf-8")
    MODULE._write_json(destination, {"fresh": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"fresh": 1}
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_real_qwen3_hooks_capture_post_swiglu_and_gqa_o_proj_inputs() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=16,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    tokenizer = _Tokenizer()
    prompt = "alpha beta"
    encoded = MODULE._encode_calibration_prompts(tokenizer, [prompt], 8)[0]
    layer = model.model.layers[0]
    captures: dict[str, torch.Tensor] = {}

    def capture_mlp(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["mlp_input"] = inputs[0].detach()

    def capture_o_proj(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captures["o_input"] = inputs[0].detach()

    handles = [
        layer.mlp.register_forward_pre_hook(capture_mlp),
        layer.self_attn.o_proj.register_forward_pre_hook(capture_o_proj),
    ]
    try:
        with torch.no_grad():
            model.model(
                **{name: value for name, value in encoded.model_inputs.items()},
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    cache = MODULE.collect_bounded_qwen_activations(
        model,
        tokenizer,
        [prompt],
        max_length=8,
        activation_rows=encoded.token_count,
        device="cpu",
    )
    expected_swiglu = layer.mlp.act_fn(layer.mlp.gate_proj(captures["mlp_input"]))
    expected_swiglu = expected_swiglu * layer.mlp.up_proj(captures["mlp_input"])
    torch.testing.assert_close(cache.neuron_activations[0], expected_swiglu.cpu().reshape(-1, 6).float())
    torch.testing.assert_close(cache.attention_activations[0], captures["o_input"].cpu().reshape(-1, 8).float())
