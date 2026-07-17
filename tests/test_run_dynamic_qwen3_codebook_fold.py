from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_qwen_adapter import (
    build_qwen3_codebook_tokenizer_contract,
    build_qwen3_global_codebook_fold_cache,
    build_qwen3_global_codebook_fold_cache_from_checkpoint,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_qwen_codebook_model import (
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")
tokenizers = pytest.importorskip("tokenizers")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_dynamic_qwen3_codebook_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_dynamic_qwen3_codebook_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_to_mixed_p35_attention_p36_neuron_budget() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--parent-model",
            "parent",
            "--parent-tokenizer",
            "parent-tokenizer",
            "--target-tokenizer",
            "target-tokenizer",
            "--fold-checkpoint",
            "checkpoint",
            "--checkpoint-manifest-sha256",
            "a" * 64,
            "--cached-parent-eval",
            "parent-eval.json",
            "--cached-candidate-eval",
            "candidate-eval.json",
            "--cached-candidate-eval-sha256",
            "b" * 64,
            "--output",
            "output",
        ]
    )

    assert args.target_parameters == 3_995_570_176
    assert args.attention_prototypes == 35
    assert args.neuron_prototypes == 36
    assert args.parent_device == "cuda:0"
    assert args.child_device == "cuda:1"
    assert args.reload_device == "cuda:0"
    assert args.release_parent_before_reload is True


def _cached_eval(path: Path) -> None:
    generations = []
    diagnostics = []
    for category in ("alpha", "beta"):
        for parent_pass in (False, True):
            for index in range(2):
                case_id = f"{category}-{int(parent_pass)}-{index}"
                prompt = f"Prompt {case_id}:"
                continuation = (
                    f" keyword {category} coherent response {index}"
                    if parent_pass
                    else f" unrelated response {index}"
                )
                generations.append(
                    {
                        "id": case_id,
                        "category": category,
                        "prompt": prompt,
                        "output": prompt + continuation,
                        "relevance_rules": [[rf"\b{category}\b"], [r"\bkeyword\b"]],
                        "diagnostics": {
                            "generated_token_count": 6,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                )
                diagnostics.append(
                    {
                        "id": case_id,
                        "passed": parent_pass,
                        "diagnostics": {
                            "generated_token_count": 6,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                )
    path.write_text(
        json.dumps(
            {
                "eval_suite": {
                    "schema": "tiny-eval-suite-v1",
                    "generation_prompt_count": len(generations),
                },
                "generations": generations,
                "generation_diagnostics": diagnostics,
            }
        ),
        encoding="utf-8",
    )


def test_cached_generation_subset_is_deterministic_and_stratified(
    tmp_path,
) -> None:
    source = tmp_path / "parent.json"
    _cached_eval(source)

    first = MODULE.load_cached_generation_baseline(
        source,
        subset_size=4,
        subset_seed=17,
        generation_limit=8,
        fragment_minimum_characters=4,
    )
    second = MODULE.load_cached_generation_baseline(
        source,
        subset_size=4,
        subset_seed=17,
        generation_limit=8,
        fragment_minimum_characters=4,
    )

    assert [case.case_id for case in first.cases] == [
        case.case_id for case in second.cases
    ]
    assert set(first.selected_strata) == {
        ("alpha", False),
        ("alpha", True),
        ("beta", False),
        ("beta", True),
    }
    assert first.fingerprint == second.fingerprint
    assert first.task_score(first.cases[0], "keyword alpha") in {True, False}
    alpha = next(case for case in first.cases if case.category == "alpha")
    assert first.task_score(alpha, "keyword alpha") is True
    assert first.task_score(alpha, "unrelated") is False
    assert first.as_dict()["parent_forward_passes"] == 0

    gate = MODULE.build_cached_whole_output_gate(
        first,
        generator=lambda _model, cases: {
            case.case_id: case.parent_output for case in cases
        },
        maximum_new_empty_responses=0,
        maximum_new_fragment_responses=0,
        minimum_parent_nonempty_retention=1.0,
        minimum_task_score_retention=1.0,
        maximum_new_generation_limit_saturations=0,
        maximum_repeated_bigram_share_regressions=0,
        maximum_repeated_bigram_share_delta=0.0,
    )
    final_binding = resolve_behavior_gate_binding(gate.evaluate)
    fresh_binding = resolve_behavior_gate_binding(gate.evaluate_model)
    assert final_binding.manifest_fingerprint == fresh_binding.manifest_fingerprint
    assert final_binding.gate_id == (
        "neural_forge.qwen3_codebook.whole_output_behavior"
    )
    assert final_binding.gate_version == 2
    assert final_binding.manifest["coverage_policy"]["required_splits"] == ["heldout"]


def test_cached_generation_subset_requires_every_stratum(tmp_path) -> None:
    source = tmp_path / "parent.json"
    _cached_eval(source)

    with pytest.raises(
        MODULE.DynamicQwen3RunnerError,
        match="cover every",
    ):
        MODULE.load_cached_generation_baseline(
            source,
            subset_size=3,
            subset_seed=1,
            generation_limit=8,
            fragment_minimum_characters=4,
        )


def test_cached_candidate_preflight_requires_exact_supplied_sha256(tmp_path) -> None:
    parent = tmp_path / "parent.json"
    candidate = tmp_path / "candidate.json"
    _cached_eval(parent)
    payload = json.loads(parent.read_text(encoding="utf-8"))
    payload.update(
        {
            "model": "qwen3-codebook:pn2:pa2",
            "param_count": 123,
        }
    )
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    baseline = MODULE.load_cached_generation_baseline(
        parent,
        subset_size=4,
        subset_seed=17,
        generation_limit=8,
        fragment_minimum_characters=4,
    )

    with pytest.raises(
        MODULE.DynamicQwen3RunnerError,
        match="fingerprint differs",
    ):
        MODULE.preflight_cached_candidate(
            baseline,
            candidate_eval=candidate,
            expected_candidate_sha256="0" * 64,
            checkpoint_manifest_sha256="a" * 64,
            target_parameters=123,
            attention_prototypes=2,
            neuron_prototypes=2,
            maximum_new_empty_responses=0,
            maximum_new_fragment_responses=0,
            minimum_parent_nonempty_retention=1.0,
            minimum_task_score_retention=1.0,
            maximum_new_generation_limit_saturations=0,
            maximum_repeated_bigram_share_regressions=0,
            maximum_repeated_bigram_share_delta=0.05,
        )


def test_uniform_pn35_pa35_candidate_preflight_rejects_recorded_collapse() -> None:
    """Lock the older uniform p35/p35 failure, not the pn36/pa35 incumbent."""

    root = Path(__file__).resolve().parents[1]
    parent = (
        root / "runs" / "eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json"
    )
    candidate = (
        root
        / "runs"
        / "evidence"
        / "qwen3_4b_codebook_pn35_pa35_gpu4_r1_20260710"
        / "candidate_eval.json"
    )
    baseline = MODULE.load_cached_generation_baseline(
        parent,
        subset_size=25,
        subset_seed=20_260_716,
        generation_limit=96,
        fragment_minimum_characters=12,
    )

    result = MODULE.preflight_cached_candidate(
        baseline,
        candidate_eval=candidate,
        expected_candidate_sha256=MODULE._file_sha256(candidate),
        checkpoint_manifest_sha256="a" * 64,
        target_parameters=3_920_859_136,
        attention_prototypes=35,
        neuron_prototypes=35,
        maximum_new_empty_responses=0,
        maximum_new_fragment_responses=0,
        minimum_parent_nonempty_retention=1.0,
        minimum_task_score_retention=1.0,
        maximum_new_generation_limit_saturations=0,
        maximum_repeated_bigram_share_regressions=0,
        maximum_repeated_bigram_share_delta=0.05,
    )

    assert result.accepted is False
    assert result.diagnostics["severe_repetition_regression_count"] == 3
    assert result.diagnostics["pruning_task_regression_count"] == 3
    assert result.diagnostics["repetition_regression_count"] == 4
    assert result.diagnostics["task_regression_count"] == 4
    metrics = dict(result.decision.metrics)
    assert metrics["new_empty_count"] == 3
    assert metrics["new_fragment_count"] == 3
    assert result.as_dict()["model_loading_performed"] is False
    assert result.as_dict()["gpu_loading_performed"] is False


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, _text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is True
        values = torch.tensor([[1, 2]], dtype=torch.long)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
        }

    def decode(
        self,
        _token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "x y x y"


class _TinyGeneratorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def generate(self, *, input_ids, **_kwargs):
        suffix = torch.tensor([[7, 8, 7, 8]], device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


def test_greedy_generator_emits_mechanical_gate_metadata() -> None:
    generator = MODULE.GreedyWholeOutputGenerator(
        tokenizer=_TinyTokenizer(),
        max_new_tokens=4,
        maximum_prompt_tokens=8,
        fragment_minimum_characters=3,
    )
    case = MODULE.WholeOutputCase(
        "case",
        "Prompt",
        "parent output",
        parent_metadata=MODULE.WholeOutputMetadata(
            generated_token_count=3,
            generation_limit=4,
            maximum_repeated_ngram_share=0.0,
        ),
    )

    output = generator(_TinyGeneratorModel(), (case,))["case"]

    assert output.output == "x y x y"
    assert output.metadata.generated_token_count == 4
    assert output.metadata.generation_limit_saturated is True
    assert output.metadata.maximum_repeated_ngram_share == pytest.approx(1 / 3)
    assert output.fragment_like is False
    assert generator.receipt()["invocation_count"] == 1
    assert generator.receipt()["parent_forward_passes"] == 0


def _serialized_state(model) -> dict[str, torch.Tensor]:
    values = {}
    for name, value in model.state_dict().items():
        values[name] = value.detach().cpu().contiguous().clone()
    return values


def test_direct_local_qwen_loader_reconstructs_dense_state_and_buffers(
    tmp_path,
) -> None:
    torch.manual_seed(55)
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    source = transformers.Qwen3ForCausalLM(config).eval()
    (tmp_path / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    safetensors.save_file(
        _serialized_state(source),
        str(tmp_path / "model.safetensors"),
    )

    loaded = MODULE.load_local_qwen3_model(
        tmp_path,
        device="cpu",
        dtype=torch.float32,
    )

    assert type(loaded) is transformers.Qwen3ForCausalLM
    assert MODULE.fingerprint_model_state(loaded) == (
        MODULE.fingerprint_model_state(source)
    )
    assert loaded.get_input_embeddings().weight is (
        loaded.get_output_embeddings().weight
    )
    assert torch.equal(
        loaded.model.rotary_emb.inv_freq,
        source.model.rotary_emb.inv_freq,
    )
    source_text = inspect.getsource(MODULE.load_local_qwen3_model)
    assert "AutoModel" not in source_text
    assert "from_pretrained" not in source_text


def test_direct_local_tokenizer_and_semantic_fingerprint(tmp_path) -> None:
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "hello": 3,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )

    tokenizer = MODULE.load_local_tokenizer(tmp_path)
    identity = MODULE.fingerprint_local_tokenizer(tmp_path, tokenizer)

    assert tokenizer.get_vocab()["hello"] == 3
    assert identity.vocab_size == 4
    assert len(identity.fingerprint) == 64
    assert set(identity.file_sha256) == {
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_parent_release_wrapper_preserves_values_and_releases_once(
    tmp_path,
) -> None:
    del tmp_path
    parent = torch.nn.Linear(3, 2, bias=False)
    expected = MODULE.fingerprint_model_state(parent)
    controller = MODULE.ParentReleaseBeforeFreshLoad(
        parent=parent,
        expected_parent_fingerprint=expected,
        enabled=True,
        reload_device="cpu",
    )
    wrapped = controller.wrap(lambda _path: torch.nn.Linear(2, 1))

    loaded = wrapped(Path("child"))

    assert isinstance(loaded, torch.nn.Linear)
    assert MODULE.fingerprint_model_state(parent) == expected
    assert controller.report["parent_values_unchanged"] is True
    assert controller.report["parent_devices_after"] == ["cpu"]
    second = wrapped(Path("child"))
    assert isinstance(second, torch.nn.Linear)
    assert controller.report["valid_load_count"] == 2


class _RowTokenizer:
    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, index):
        return self.tokens[index]


def test_checkpoint_topology_adapter_erases_weights_without_refitting(
    tmp_path,
) -> None:
    torch.manual_seed(81)
    parent = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
    ).eval()
    parent_tokenizer = _RowTokenizer([f"token-{index}" for index in range(12)])
    target_tokenizer = _RowTokenizer(
        ["token-0", "token-1", "token-2", "token-4", "token-7"]
    )
    fresh_cache = build_qwen3_global_codebook_fold_cache(
        source_parent="parent",
        parent_model=parent,
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=1,
    )
    runtime = copy.deepcopy(parent)
    prepare_qwen3_codebook_model(
        runtime,
        parent_tokenizer,
        target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=1,
    )
    contract = build_qwen3_codebook_tokenizer_contract(
        target_tokenizer=target_tokenizer,
        cache=fresh_cache,
        tokenizer_name_or_path="tiny-target",
    )
    checkpoint = tmp_path / "checkpoint"
    save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=contract,
    )
    manifest_sha = MODULE._file_sha256(checkpoint / "manifest.json")
    cache = build_qwen3_global_codebook_fold_cache_from_checkpoint(
        checkpoint=checkpoint,
        source_parent="parent",
        live_parent_model=parent,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=1,
        tokenizer_contract=contract,
        expected_manifest_sha256=manifest_sha,
    )
    inspectors = default_inspector_registry()
    parent_graph = inspectors.inspect(parent, model_name="parent")
    adapter = MODULE.CheckpointTopologyQwen3TargetSynthesisAdapter(
        checkpoint=checkpoint,
        cache=cache,
        device="cpu",
        dtype=torch.float32,
        manifest_sha256=manifest_sha,
    )
    request = DynamicTargetRequest(
        child_name="blank",
        target_family="qwen3_codebook",
        target_modalities=("text",),
        exact_unique_parameters=cache.expected_unique_parameters,
        blank_seed=99,
        configuration=(("source_parent", "parent"),),
    )

    materialized = adapter.materialize(
        request,
        {"parent": parent},
        {"parent": parent_graph},
        inspectors,
    )

    assert materialized.graph.family == "qwen3_codebook"
    assert materialized.report["fresh_codebook_fit_performed"] is False
    assert materialized.report["blank_learned_parameters_parent_derived"] is False
    assert cache.matches_parameters(materialized.model) is False
    cache.assert_topology(materialized.model)

    synthesized = DynamicTargetSynthesizer(
        inspectors=inspectors,
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
    )

    assert synthesized.structural_buffer_policy is not None
    assert synthesized.structural_buffer_policy.compiler_reviewed is True
    assert synthesized.structural_buffer_policy.source.startswith(
        "compiler_concrete_runtime_registry:"
    )
    policy_paths = {
        path for path, _role in synthesized.structural_buffer_policy.entries
    }
    assert {
        "parent_token_rows",
        "attention_codebook.registry.assignments",
        "attention_codebook.registry.prototype_source_indices",
        "attention_codebook.registry.group_permutations",
        "neuron_codebook.registry.assignments",
        "neuron_codebook.registry.prototype_source_indices",
    }.issubset(policy_paths)
    assert synthesized.structural_buffer_audit is not None
    assert synthesized.structural_buffer_audit.nonzero_buffer_count > 0


def _write_local_tokenizer(path: Path, tokens: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {token: index for index, token in enumerate(tokens)},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )


def test_tiny_production_runner_publishes_and_freshly_gates_without_training(
    tmp_path,
) -> None:
    parent_directory = tmp_path / "parent"
    target_tokenizer_directory = tmp_path / "target-tokenizer"
    parent_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "hello",
        "world",
        "x",
        "y",
        "z",
        "answer",
        "fold",
        "model",
    ]
    target_tokens = parent_tokens[:8]
    _write_local_tokenizer(parent_directory, parent_tokens)
    _write_local_tokenizer(target_tokenizer_directory, target_tokens)
    parent_tokenizer = MODULE.load_local_tokenizer(parent_directory)
    target_tokenizer = MODULE.load_local_tokenizer(target_tokenizer_directory)
    torch.manual_seed(120)
    config = transformers.Qwen3Config(
        vocab_size=len(parent_tokens),
        hidden_size=8,
        intermediate_size=16,
        # Four layers makes this a genuine whole-runtime-state compaction.
        # With only two layers and two prototypes the learned saving was
        # 128 bytes, smaller than the codebook registry payload.
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    parent = transformers.Qwen3ForCausalLM(config).eval()
    (parent_directory / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    safetensors.save_file(
        _serialized_state(parent),
        str(parent_directory / "model.safetensors"),
    )
    fresh_cache = build_qwen3_global_codebook_fold_cache(
        source_parent="parent",
        parent_model=parent,
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=2,
        neuron_prototypes_per_neuron=2,
    )
    runtime = copy.deepcopy(parent)
    prepare_qwen3_codebook_model(
        runtime,
        parent_tokenizer,
        target_tokenizer,
        attention_prototypes_per_group=2,
        neuron_prototypes_per_neuron=2,
    )
    contract = build_qwen3_codebook_tokenizer_contract(
        target_tokenizer=target_tokenizer,
        cache=fresh_cache,
        tokenizer_name_or_path="tiny-target",
    )
    checkpoint = tmp_path / "fold-checkpoint"
    save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=contract,
    )
    manifest_sha = MODULE._file_sha256(checkpoint / "manifest.json")
    target_parameters = qwen3_codebook_parameter_report(runtime).unique_parameter_count
    cached_eval = tmp_path / "cached-parent.json"
    cached_candidate = tmp_path / "cached-candidate.json"
    cached_payload = {
        "eval_suite": {
            "schema": "tiny-production-suite-v1",
            "generation_prompt_count": 1,
        },
        "generations": [
            {
                "id": "hello",
                "category": "chat",
                "prompt": "hello",
                "output": "hello parent",
                "relevance_rules": [[r".*"]],
                "diagnostics": {
                    "generated_token_count": 1,
                    "repeated_bigram_share": 0.0,
                },
            }
        ],
        "generation_diagnostics": [
            {
                "id": "hello",
                "passed": True,
                "diagnostics": {
                    "generated_token_count": 1,
                    "repeated_bigram_share": 0.0,
                },
            }
        ],
    }
    cached_eval.write_text(
        json.dumps(cached_payload),
        encoding="utf-8",
    )
    candidate_payload = copy.deepcopy(cached_payload)
    candidate_payload.update(
        {
            "model": "qwen3-codebook:pn2:pa2",
            "param_count": target_parameters,
        }
    )
    cached_candidate.write_text(
        json.dumps(candidate_payload),
        encoding="utf-8",
    )
    output = tmp_path / "published"
    args = MODULE.build_parser().parse_args(
        [
            "--parent-model",
            str(parent_directory),
            "--parent-tokenizer",
            str(parent_directory),
            "--target-tokenizer",
            str(target_tokenizer_directory),
            "--fold-checkpoint",
            str(checkpoint),
            "--checkpoint-manifest-sha256",
            manifest_sha,
            "--cached-parent-eval",
            str(cached_eval),
            "--cached-candidate-eval",
            str(cached_candidate),
            "--cached-candidate-eval-sha256",
            MODULE._file_sha256(cached_candidate),
            "--output",
            str(output),
            "--target-parameters",
            str(target_parameters),
            "--attention-prototypes",
            "2",
            "--neuron-prototypes",
            "2",
            "--parent-device",
            "cpu",
            "--child-device",
            "cpu",
            "--reload-device",
            "cpu",
            "--dtype",
            "float32",
            "--behavior-subset-size",
            "1",
            "--max-new-tokens",
            "1",
            "--maximum-prompt-tokens",
            "16",
            "--fragment-minimum-characters",
            "1",
            "--maximum-new-empty-responses",
            "1",
            "--maximum-new-fragment-responses",
            "1",
            "--minimum-parent-nonempty-retention",
            "0",
            "--maximum-new-generation-limit-saturations",
            "1",
            "--maximum-repeated-bigram-share-regressions",
            "1",
            "--maximum-repeated-bigram-share-delta",
            "1",
        ]
    )

    receipt = MODULE.run(args)

    assert receipt["complete"] is True
    assert receipt["target"]["blank_materialization"] == {
        "topology_loaded_from_immutable_checkpoint": True,
        "fresh_codebook_fit_performed": False,
        "blank_learned_parameters_parent_derived": False,
    }
    assert receipt["execution"]["snapshot_backend"] == "disk"
    assert receipt["execution"]["behavior_generator"]["invocation_count"] == 3
    assert receipt["execution"]["cached_candidate_preflight_accepted"] is True
    preflight = receipt["inputs"]["cached_candidate_preflight"]
    assert preflight["decision"]["accepted"] is True
    assert preflight["candidate_sha256"] == MODULE._file_sha256(cached_candidate)
    assert preflight["checkpoint_manifest_sha256"] == manifest_sha
    assert receipt["execution"]["parent_forward_passes"] == 0
    assert receipt["execution"]["optimizer_steps"] == 0
    assert (output / "child" / "manifest.json").is_file()
    assert (tmp_path / "published.run_receipt.json").is_file()
    assert (
        MODULE.snapshot_directory_inventory(tmp_path / ".published.fold-snapshots")
        == {}
    )


def test_preflight_rejection_writes_receipt_before_any_model_or_gpu_load(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        '{"schema":"test-checkpoint-v1"}',
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_contract.json").write_text(
        '{"tokenizer_name_or_path":"local"}',
        encoding="utf-8",
    )
    manifest_sha = MODULE._file_sha256(checkpoint / "manifest.json")
    parent_eval = tmp_path / "parent-eval.json"
    candidate_eval = tmp_path / "candidate-eval.json"
    suite = {
        "schema": "preflight-rejection-suite-v1",
        "generation_prompt_count": 1,
    }
    generation_contract = {
        "id": "case",
        "category": "task",
        "prompt": "Question:",
        "relevance_rules": [[r"\banswer\b"]],
    }
    parent_eval.write_text(
        json.dumps(
            {
                "eval_suite": suite,
                "generations": [
                    {
                        **generation_contract,
                        "output": "Question: answer",
                        "diagnostics": {
                            "generated_token_count": 1,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                ],
                "generation_diagnostics": [
                    {
                        "id": "case",
                        "passed": True,
                        "diagnostics": {
                            "generated_token_count": 1,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_eval.write_text(
        json.dumps(
            {
                "model": "qwen3-codebook:pn2:pa2",
                "param_count": 123,
                "eval_suite": suite,
                "generations": [
                    {
                        **generation_contract,
                        "output": "Question:",
                        "diagnostics": {
                            "generated_token_count": 0,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                ],
                "generation_diagnostics": [
                    {
                        "id": "case",
                        "passed": False,
                        "diagnostics": {
                            "generated_token_count": 0,
                            "repeated_bigram_share": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "published"
    args = MODULE.build_parser().parse_args(
        [
            "--parent-model",
            str(tmp_path / "must-not-resolve-parent"),
            "--parent-tokenizer",
            str(tmp_path / "must-not-resolve-parent-tokenizer"),
            "--target-tokenizer",
            str(tmp_path / "must-not-resolve-target-tokenizer"),
            "--fold-checkpoint",
            str(checkpoint),
            "--checkpoint-manifest-sha256",
            manifest_sha,
            "--cached-parent-eval",
            str(parent_eval),
            "--cached-candidate-eval",
            str(candidate_eval),
            "--cached-candidate-eval-sha256",
            MODULE._file_sha256(candidate_eval),
            "--output",
            str(output),
            "--target-parameters",
            "123",
            "--attention-prototypes",
            "2",
            "--neuron-prototypes",
            "2",
            "--behavior-subset-size",
            "1",
            "--max-new-tokens",
            "4",
            "--fragment-minimum-characters",
            "1",
        ]
    )

    def fail_loader(*_args, **_kwargs):
        pytest.fail("Preflight rejection must occur before any model/tokenizer load.")

    monkeypatch.setattr(MODULE, "load_local_tokenizer", fail_loader)
    monkeypatch.setattr(MODULE, "load_local_qwen3_model", fail_loader)

    receipt = MODULE.run(args)

    assert receipt["status"] == "cached_candidate_preflight_rejected"
    assert receipt["complete"] is False
    assert receipt["execution"] == {
        "model_loading_performed": False,
        "gpu_loading_performed": False,
        "tokenizer_loading_performed": False,
        "parent_forward_passes": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
    }
    assert receipt["artifact"]["published"] is False
    assert (
        receipt["inputs"]["cached_candidate_preflight"]["decision"]["accepted"] is False
    )
    assert not output.exists()
    assert (tmp_path / "published.run_receipt.json").is_file()
    assert not (tmp_path / ".published.fold-snapshots").exists()


def test_manifest_pin_and_atomic_receipt_helpers(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text('{"schema":"x"}', encoding="utf-8")
    (checkpoint / "tokenizer_contract.json").write_text(
        '{"tokenizer_name_or_path":"local"}',
        encoding="utf-8",
    )
    manifest_sha = MODULE._file_sha256(checkpoint / "manifest.json")

    resolved, contract, observed = MODULE._checkpoint_contract(
        checkpoint,
        manifest_sha,
    )

    assert resolved == checkpoint.resolve()
    assert contract["tokenizer_name_or_path"] == "local"
    assert observed == manifest_sha
    with pytest.raises(MODULE.DynamicQwen3RunnerError, match="manifest differs"):
        MODULE._checkpoint_contract(checkpoint, "0" * 64)

    receipt = tmp_path / "receipt.json"
    MODULE._atomic_write_json(receipt, {"complete": True})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"complete": True}
    with pytest.raises(MODULE.DynamicQwen3RunnerError, match="already exists"):
        MODULE._atomic_write_json(receipt, {"complete": False})
