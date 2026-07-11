from __future__ import annotations

import gc
import importlib.util
import json
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_qwen_compensated_neuron_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_qwen_compensated_neuron_fold", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _tiny_qwen(seed: int = 1101, layers: int = 2):
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=23,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}
    padding_side = "right"
    truncation_side = "right"

    def get_vocab(self):
        return {f"token-{index}": index for index in range(23)}

    def get_added_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        values = [3 + (ord(value) % 17) for value in str(text)] or [3]
        if add_special_tokens:
            return [1, *values, 2]
        return values

    def __call__(
        self,
        text,
        *,
        return_tensors=None,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
        **_kwargs,
    ):
        values = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            values = values[:max_length]
        if return_tensors == "pt":
            input_ids = torch.tensor([values], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return {"input_ids": values, "attention_mask": [1] * len(values)}


class _ActivationProvider:
    def __init__(self, seed: int = 1201) -> None:
        self.seed = seed
        self.calls = 0

    def __call__(self, model):
        self.calls += 1
        calibration = []
        validation = []
        for layer_index, layer in enumerate(model.model.layers):
            width = int(layer.mlp.gate_proj.out_features)
            generator = torch.Generator().manual_seed(
                self.seed + self.calls * 100 + layer_index
            )
            rows = torch.randn(20, width, generator=generator)
            gate = torch.randn(16, width, generator=generator)
            rows[:, -1] = 1.75 * rows[:, 0]
            gate[:, -1] = 1.75 * gate[:, 0]
            calibration.append(rows)
            validation.append(gate)
        return MODULE.ActivationPair(
            tuple(calibration),
            tuple(validation),
            {
                "bounded": True,
                "refresh": self.calls,
                "real_forward_required_in_production": True,
            },
        )


class _Decisions:
    def __init__(self, values: list[bool]) -> None:
        self.values = values
        self.calls = 0

    def __call__(self, model, attempt, proposal):
        del model
        decision = self.values[self.calls]
        self.calls += 1
        return decision, {
            "attempt": attempt,
            "layer": proposal.layer_index,
            "domains": {
                "behavior": {"passed": decision},
                "choice": {"passed": decision},
                "generation": {"passed": decision},
            },
            "perplexity_used_for_acceptance": False,
            "passed": decision,
        }


def _loop_config(**values):
    defaults = {
        "device": "cpu",
        "calibration_max_length": 16,
        "activation_rows": 16,
        "max_host_activation_bytes": 16 * 1024**2,
        "anchor_fraction": 0.0,
        "minimum_anchor_count": 0,
        "projection_count": 16,
        "neighbor_window": 5,
        "max_pair_candidates": 200,
        "calibration_provenance_id": "calibration-provenance-sha",
        "validation_provenance_id": "validation-provenance-sha",
        "merges_per_transaction": 1,
        "max_candidate_attempts": 16,
        "max_transactions": 8,
        "full_milestone_interval": 8,
        "seed": 7,
    }
    defaults.update(values)
    return MODULE.FoldLoopConfig(**defaults)


def test_cached_parent_gate_never_recomputes_parent(tmp_path: Path) -> None:
    parent = {
        "behavior_suite": {"items": [{"id": "b1", "passed": True}]},
        "choice_accuracy": {"items": [{"id": "c1", "correct": True}]},
        "generation_diagnostics": [{"id": "g1", "passed": True}],
        "generation_health": {"contract": dict(MODULE.GENERATION_HEALTH_CONTRACT)},
    }
    candidate = {
        "behavior_suite": {"items": [{"id": "b1", "passed": True}]},
        "choice_accuracy": {"items": [{"id": "c1", "correct": True}]},
        "generation_diagnostics": [{"id": "g1", "passed": True}],
        "generation_health": {"contract": dict(MODULE.GENERATION_HEALTH_CONTRACT)},
    }
    calls = []

    def evaluator(model, tokenizer, output_file, **kwargs):
        del model, tokenizer, output_file, kwargs
        calls.append("candidate")
        return candidate

    suite = {
        "text_evals": ["text"],
        "behavior_probes": [{"id": "b1", "prompt": "behavior"}],
        "choice_probes": [{"id": "c1", "prompt": "choice"}],
        "generation_prompts": [{"id": "g1", "prompt": "generation"}],
    }
    gate = MODULE.CachedParentGateEvaluator(
        parent_report=parent,
        gate_suite=suite,
        gate_ids={"behavior": ("b1",), "choice": ("c1",), "generation": ("g1",)},
        thresholds=MODULE.GateThresholds(),
        tokenizer=object(),
        device="cpu",
        max_new_tokens=4,
        output_directory=tmp_path,
        split_provenance_sha256="a" * 64,
        evaluator=evaluator,
    )
    proposal = SimpleNamespace(layer_index=0)
    for attempt in (1, 2):
        accepted, report = gate(object(), attempt, proposal)
        assert accepted
        assert report["verdict"]["parent_forward_passes"] == 0

    assert calls == ["candidate", "candidate"]
    assert gate.parent_evaluation_calls == 0
    assert gate.candidate_evaluation_calls == 2
    assert report["split_provenance_sha256"] == "a" * 64
    assert report["certification_candidate_evaluation_calls"] == 0
    assert report["selection_data_adaptively_reused"] is True


def test_discovery_selection_certification_splits_are_pairwise_disjoint(
    tmp_path: Path,
) -> None:
    calibration_source = tmp_path / "calibration.jsonl"
    selection_source = tmp_path / "selection.json"
    certification_source = tmp_path / "certification.json"
    parent_selection_source = tmp_path / "parent-selection.json"
    parent_certification_source = tmp_path / "parent-certification.json"
    calibration_source.write_text('{"prompt":"calibration alpha"}\n', encoding="utf-8")
    selection_source.write_text('{"suite":"selection"}\n', encoding="utf-8")
    certification_source.write_text('{"suite":"certification"}\n', encoding="utf-8")
    parent_selection_source.write_text('{"model":"parent"}\n', encoding="utf-8")
    parent_certification_source.write_text('{"model":"parent"}\n', encoding="utf-8")
    selection_suite = {
        "text_evals": ["selection perplexity text"],
        "choice_probes": [{"id": "sc", "prompt": "selection choice"}],
        "generation_prompts": [{"id": "sg", "prompt": "selection generation"}],
        "behavior_probes": [{"id": "sb", "prompt": "selection behavior"}],
    }
    certification_suite = {
        "text_evals": ["certification perplexity text"],
        "choice_probes": [{"id": "cc", "prompt": "certification choice"}],
        "generation_prompts": [{"id": "cg", "prompt": "certification generation"}],
        "behavior_probes": [{"id": "cb", "prompt": "certification behavior"}],
    }

    report = MODULE.build_three_split_provenance(
        _Tokenizer(),
        ("calibration alpha", "calibration beta"),
        ("validation gamma", "validation delta"),
        selection_suite,
        certification_suite,
        calibration_source=calibration_source,
        selection_source=selection_source,
        cached_parent_selection_source=parent_selection_source,
        certification_source=certification_source,
        cached_parent_certification_source=parent_certification_source,
        max_length=32,
    )

    assert report["pairwise_token_path_overlaps"] == {
        "calibration_validation": 0,
        "calibration_selection": 0,
        "validation_selection": 0,
        "calibration_certification": 0,
        "validation_certification": 0,
        "selection_certification": 0,
    }
    assert report["fit_visible_splits"] == ["calibration", "validation"]
    assert (
        report[
            "certification_scores_or_parent_outputs_visible_to_fit_or_selection"
        ]
        is False
    )
    assert report["pair_discovery_uses_validation"] is False
    assert report["all_provenance_bound"] is True
    assert len(report["provenance_sha256"]) == 64


def test_full_selection_suite_must_be_token_disjoint_from_certification() -> None:
    suite = {
        "text_evals": ["same text"],
        "choice_probes": [{"id": "c", "prompt": "same choice"}],
        "generation_prompts": [{"id": "g", "prompt": "same generation"}],
        "behavior_probes": [{"id": "b", "prompt": "same behavior"}],
    }
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="identical token paths"):
        MODULE.verify_selection_certification_disjoint(
            _Tokenizer(), suite, suite, max_length=32
        )
    selection = {
        name: [
            (
                f"adaptive {value}"
                if isinstance(value, str)
                else {**value, "id": f"selection-{value['id']}", "prompt": f"adaptive {value['prompt']}"}
            )
            for value in values
        ]
        for name, values in suite.items()
    }
    report = MODULE.verify_selection_certification_disjoint(
        _Tokenizer(), selection, suite, max_length=32
    )
    assert report["overlap_count"] == 0
    assert report["verified"] is True


def test_rejected_transaction_rolls_back_then_accept_refreshes() -> None:
    model = _tiny_qwen()
    initial = MODULE.model_accounting(model)
    provider = _ActivationProvider()
    gate = _Decisions([False, True])

    report = MODULE.run_fold_transactions(
        model,
        activation_provider=provider,
        small_gate=gate,
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(),
    )

    assert report["attempt_count"] == 2
    assert report["rejected_count"] == 1
    assert report["accepted_count"] == 1
    assert report["rejected_transactions"][0]["rollback_exact"] is True
    assert report["rejected_transactions"][0]["before"] == report[
        "rejected_transactions"
    ][0]["after"]
    assert report["activation_refresh_after_every_accept"] is True
    assert report["activation_collection_count"] == 2
    assert provider.calls == 2
    assert all(parameter.grad is None for parameter in model.parameters())
    assert report["zero_optimizer_steps"] == 0
    assert report["backward_called"] is False


def test_explicit_layer_skip_keeps_one_layer_exact() -> None:
    model = _tiny_qwen(1301)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1302),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(excluded_layers=(0,)),
    )

    assert report["target"]["reached"] is True
    assert report["final"]["layer_widths"] == [6, 5]
    assert report["exact_layers"] == [0]
    assert any(
        item["layer_index"] == 0 and item["reason"] == "explicitly-excluded"
        for item in report["layer_skips"]
    )


def test_calibration_prefilter_bounds_exact_layer_proposal_work() -> None:
    model = _tiny_qwen(1321)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1322),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(proposal_layer_limit=1, prefilter_columns=4),
    )

    proposal = report["accepted_transactions"][0]["proposal"]
    assert proposal["calibration_prefilter_score"] >= 0.0
    assert any(
        item["reason"] == "bounded-calibration-prefilter-not-selected"
        for item in report["layer_skips"]
    )


def test_uniform_commit_updates_config_without_false_accounting_failure() -> None:
    model = _tiny_qwen(1351, layers=1)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1352),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(),
    )

    assert report["target"]["reached"] is True
    assert report["final"]["layer_widths"] == [5]
    assert report["final"]["configured_intermediate_size"] == 5
    assert model.config.intermediate_size == 5


def test_parameter_and_byte_targets_use_unique_accounting() -> None:
    model = _tiny_qwen(1401)
    initial = MODULE.model_accounting(model)
    per_neuron_bytes = 24 * model.model.layers[0].mlp.gate_proj.weight.element_size()
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1402),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=initial.unique_parameter_bytes - per_neuron_bytes,
        config=_loop_config(),
    )

    assert report["target"]["reached"] is True
    assert report["final"]["unique_parameter_count"] == (
        initial.unique_parameter_count - 24
    )
    assert report["final"]["unique_parameter_bytes"] == (
        initial.unique_parameter_bytes - per_neuron_bytes
    )
    proposal = report["accepted_transactions"][0]["proposal"]
    assert proposal["arbitrary_pairing_enabled"] is True
    assert proposal["adjacent_pair_count"] + proposal["nonadjacent_pair_count"] == 1
    assert proposal["strategy_evidence"]["calibration_provenance_id"] == (
        "calibration-provenance-sha"
    )
    assert proposal["strategy_evidence"]["validation_provenance_id"] == (
        "validation-provenance-sha"
    )
    assert proposal["strategy_evidence"]["work_caps"]["all_work_caps_verified"] is True


def test_real_hooks_refresh_ragged_post_swiglu_widths() -> None:
    model = _tiny_qwen(1451)
    initial = MODULE.collect_bounded_ragged_qwen_activations(
        model,
        _Tokenizer(),
        ("alpha", "beta"),
        max_length=12,
        activation_rows=6,
        device="cpu",
        max_host_activation_bytes=2 * 1024**2,
    )
    mlp = model.model.layers[0].mlp
    activations = initial.neuron_activations[0]
    activations[:, 5] = 1.5 * activations[:, 0]
    fold = MODULE.fit_compensated_neuron_pairs(
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=16,
        neighbor_window=5,
        max_pair_candidates=200,
    )
    handle = MODULE.install_qwen3_compensated_fold(model, 0, fold)
    handle.commit()

    refreshed = MODULE.collect_bounded_ragged_qwen_activations(
        model,
        _Tokenizer(),
        ("gamma", "delta"),
        max_length=12,
        activation_rows=6,
        device="cpu",
        max_host_activation_bytes=2 * 1024**2,
    )

    assert [tuple(value.shape) for value in refreshed.neuron_activations] == [
        (6, 5),
        (6, 6),
    ]
    assert refreshed.report["layer_widths"] == [5, 6]
    assert refreshed.report["ragged"] is True
    assert refreshed.report["post_swiglu_down_projection_inputs"] is True
    assert refreshed.report["bounded"] is True


def test_parent_free_ragged_safetensors_reload_is_exact(tmp_path: Path) -> None:
    model = _tiny_qwen(1501)
    mlp = model.model.layers[0].mlp
    activations = torch.randn(20, 6, generator=torch.Generator().manual_seed(1502))
    activations[:, 5] = -2.0 * activations[:, 0]
    fold = MODULE.fit_compensated_neuron_pairs(
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=16,
        neighbor_window=5,
        max_pair_candidates=200,
    )
    handle = MODULE.install_qwen3_compensated_fold(model, 0, fold)
    handle.commit()
    input_ids = torch.tensor([[1, 4, 7, 2]])
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    reload_reference = MODULE.capture_reload_reference(
        model,
        _Tokenizer(),
        ("alpha", "beta"),
        device="cpu",
        max_length=12,
        generation_tokens=2,
    )

    checkpoint = tmp_path / "ragged"
    manifest = MODULE.save_ragged_qwen_checkpoint(
        model,
        None,
        checkpoint,
        transaction_records=[{"layer": 0, "fold_sha256": fold.fold_sha256}],
        source_provenance={"source_model": "tiny-test-only"},
    )
    reference = weakref.ref(model)
    del handle
    del mlp
    del model
    gc.collect()
    assert reference() is None

    reloaded, tokenizer, loaded_manifest = MODULE.load_ragged_qwen_checkpoint(
        checkpoint, device="cpu"
    )
    with torch.inference_mode():
        actual = reloaded(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert tokenizer is None
    assert MODULE.model_accounting(reloaded).layer_widths == (5, 6)
    contract = manifest["payload"]["loader_contract"]
    assert contract["ragged_widths_require_custom_loader"] is True
    assert contract["stock_huggingface_reload_supported"] is False
    assert contract["required_loader"] == MODULE.LOADER_CONTRACT
    assert loaded_manifest["payload_sha256"] == manifest["payload_sha256"]
    assert loaded_manifest["runtime_load"]["construction_device"] == "meta"
    assert loaded_manifest["runtime_load"]["full_random_model_allocated"] is False
    equivalence = MODULE.verify_reload_equivalence(
        reload_reference, reloaded, _Tokenizer(), device="cpu"
    )
    assert equivalence["probe_count"] == 2
    assert equivalence["tokenization_bitwise_exact"] is True
    assert equivalence["deterministic_generation_bitwise_exact"] is True
    assert not list(checkpoint.rglob("*.bin"))
    assert not list(checkpoint.rglob("*.pt"))
    accounting = manifest["payload"]["artifact_accounting"]
    actual_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    assert accounting["artifact_total_bytes"] == actual_bytes
    assert accounting["all_artifact_files_hashed_and_counted"] is True


def test_large_checkpoint_publish_uses_shared_lease_and_exact_hashes(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "local-staging" / "ragged"
    staged.mkdir(parents=True)
    (staged / "model.safetensors").write_bytes(b"tensor-bytes" * 4096)
    (staged / "manifest.json").write_text(
        json.dumps({"schema": MODULE.CHECKPOINT_SCHEMA}), encoding="utf-8"
    )
    published = tmp_path / "cifs-simulated" / "run" / "ragged_child"
    lock = published.parent.parent / ".compact_checkpoint_io.lock"

    report = MODULE.publish_staged_checkpoint(
        staged,
        published,
        lock_path=lock,
    )

    assert lock.is_file()
    assert report["io_lease"]["path"] == str(lock.resolve())
    assert report["lease_wait_seconds"] >= 0.0
    assert report["copy_seconds"] >= 0.0
    assert report["fsync_seconds"] >= 0.0
    assert report["hashes_match"] is True
    assert report["bytes_match"] is True
    assert report["hashes_verified_during_copy"] is True
    assert report["content_readback_after_publish"] is False
    assert report["staging"]["tree_sha256"] == report["published"]["tree_sha256"]
    assert report["staging"]["total_bytes"] == report["published"]["total_bytes"]
    assert report["single_writer_large_io"] is (MODULE.os.name == "posix")
    assert (published / "model.safetensors").read_bytes() == (
        staged / "model.safetensors"
    ).read_bytes()


def test_staging_root_is_configurable_local_and_cleanup_is_scoped(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published" / "run"
    staging_root = tmp_path / "local-disk"
    run_root, checkpoint, report = MODULE.prepare_local_checkpoint_staging(
        output,
        staging_root,
        required_bytes=1024,
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "small").write_bytes(b"ok")

    assert run_root.parent == staging_root.resolve()
    assert report["checkpoint_directory"] == str(checkpoint)
    assert report["local_staging_required"] is True
    MODULE.remove_checkpoint_staging(run_root, staging_root)
    assert not run_root.exists()
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="root itself"):
        MODULE.remove_checkpoint_staging(staging_root, staging_root)


def test_cached_parent_gate_rejects_any_default_domain_regression() -> None:
    parent = {
        "behavior_suite": {"items": [{"id": "b", "passed": True}]},
        "choice_accuracy": {"items": [{"id": "c", "correct": True}]},
        "generation_diagnostics": [{"id": "g", "passed": True}],
    }
    candidate = {
        "behavior_suite": {"items": [{"id": "b", "passed": True}]},
        "choice_accuracy": {"items": [{"id": "c", "correct": False}]},
        "generation_diagnostics": [{"id": "g", "passed": True}],
    }
    verdict = MODULE.score_cached_parent_gate(
        parent,
        candidate,
        behavior_ids=("b",),
        choice_ids=("c",),
        generation_ids=("g",),
        thresholds=MODULE.GateThresholds(),
    )

    assert verdict["passed"] is False
    assert verdict["domains"]["choice"]["loss_count"] == 1
    assert verdict["perplexity_used_for_acceptance"] is False


def test_parser_defaults_to_full_production_evaluation(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "source",
            "--source-revision",
            "revision",
            "--output-dir",
            str(tmp_path / "out"),
            "--calibration-jsonl",
            str(tmp_path / "cal.jsonl"),
            "--eval-suite",
            str(tmp_path / "eval.json"),
            "--cached-parent-eval",
            str(tmp_path / "parent.json"),
            "--selection-eval-suite",
            str(tmp_path / "selection.json"),
            "--cached-parent-selection-eval",
            str(tmp_path / "parent-selection.json"),
            "--target-unique-parameters",
            "100",
        ]
    )

    assert args.max_behavior_probes == 1000
    assert args.max_generation_prompts == 1000
    assert args.max_choice_probes == 250
    assert args.manual_sample_count == 128
    assert args.validation_count == 64
    assert not hasattr(args, "heldout_count")
    assert args.max_gate_behavior_losses == 0
    assert args.max_gate_choice_losses == 0
    assert args.max_gate_generation_losses == 0
    assert args.merges_per_transaction == 1
    assert args.checkpoint_staging_root is None
    assert args.keep_checkpoint_staging is False
    assert args.source_revision == "revision"
    assert args.selection_max_behavior_probes == 256
    assert json.loads(json.dumps(vars(args), default=str))["source_model"] == "source"


@pytest.mark.parametrize("failure_stage", ["small_gate", "milestone", "refresh"])
def test_every_provisional_failure_path_restores_exact_model_state(
    failure_stage: str,
) -> None:
    model = _tiny_qwen(1701)
    initial = MODULE.model_accounting(model)
    initial_config = model.config.intermediate_size
    input_ids = torch.tensor([[1, 4, 7, 2]])
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    provider = _ActivationProvider(1702)

    def gate(current, attempt, proposal):
        del current, attempt, proposal
        if failure_stage == "small_gate":
            raise RuntimeError("small gate exploded")
        return True, {"verdict": {"passed": True}}

    def milestone(current, attempt, proposal):
        del current, attempt, proposal
        if failure_stage == "milestone":
            raise RuntimeError("milestone exploded")
        return True, {"verdict": {"passed": True}}

    if failure_stage == "refresh":
        original_provider = provider

        def provider(current):
            if original_provider.calls:
                raise RuntimeError("refresh exploded")
            return original_provider(current)

    with pytest.raises(RuntimeError, match="exploded"):
        MODULE.run_fold_transactions(
            model,
            activation_provider=provider,
            small_gate=gate,
            milestone_gate=milestone,
            target_parameters=initial.unique_parameter_count - 24,
            target_bytes=None,
            config=_loop_config(full_milestone_interval=1),
        )

    assert MODULE.model_accounting(model) == initial
    assert model.config.intermediate_size == initial_config
    with torch.inference_mode():
        actual = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_commit_exception_rolls_back_model_history_and_streamed_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    model = _tiny_qwen(1721)
    initial = MODULE.model_accounting(model)
    initial_config = model.config.intermediate_size
    ledger = tmp_path / "transactions.jsonl"
    real_install = MODULE.install_qwen3_compensated_fold

    def install_then_fail_commit(*args, **kwargs):
        handle = real_install(*args, **kwargs)

        def fail_commit():
            raise RuntimeError("commit exploded")

        handle.commit = fail_commit
        return handle

    monkeypatch.setattr(
        MODULE, "install_qwen3_compensated_fold", install_then_fail_commit
    )
    with pytest.raises(RuntimeError, match="commit exploded"):
        MODULE.run_fold_transactions(
            model,
            activation_provider=_ActivationProvider(1722),
            small_gate=_Decisions([True]),
            target_parameters=initial.unique_parameter_count - 24,
            target_bytes=None,
            config=_loop_config(),
            transaction_ledger=ledger,
        )

    assert MODULE.model_accounting(model) == initial
    assert model.config.intermediate_size == initial_config
    assert ledger.read_bytes() == b""


def test_ledger_cleanup_exception_cannot_prevent_model_rollback(
    tmp_path: Path, monkeypatch
) -> None:
    model = _tiny_qwen(1726)
    initial = MODULE.model_accounting(model)
    real_install = MODULE.install_qwen3_compensated_fold

    def install_then_fail_commit(*args, **kwargs):
        handle = real_install(*args, **kwargs)

        def fail_commit():
            raise RuntimeError("commit exploded")

        handle.commit = fail_commit
        return handle

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("ledger cleanup exploded")

    monkeypatch.setattr(
        MODULE, "install_qwen3_compensated_fold", install_then_fail_commit
    )
    monkeypatch.setattr(MODULE, "_truncate_transaction_ledger", fail_cleanup)
    with pytest.raises(OSError, match="ledger cleanup exploded"):
        MODULE.run_fold_transactions(
            model,
            activation_provider=_ActivationProvider(1727),
            small_gate=_Decisions([True]),
            target_parameters=initial.unique_parameter_count - 24,
            target_bytes=None,
            config=_loop_config(),
            transaction_ledger=tmp_path / "transactions.jsonl",
        )

    assert MODULE.model_accounting(model) == initial


def test_candidate_attempt_cap_stops_consecutive_rejected_proposals() -> None:
    model = _tiny_qwen(1731)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1732),
        small_gate=_Decisions([False]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(max_candidate_attempts=1, max_transactions=8),
    )

    assert report["attempt_count"] == 1
    assert report["rejected_count"] == 1
    assert report["stop_reason"] == "max-consecutive-candidate-attempts"
    assert report["max_consecutive_candidate_attempts_observed"] == 1
    assert MODULE.model_accounting(model) == initial


def test_persistent_ablation_anchors_are_selected_once_and_keep_lineage() -> None:
    model = _tiny_qwen(1751, layers=1)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1752),
        small_gate=_Decisions([True, True]),
        target_parameters=initial.unique_parameter_count - 48,
        target_bytes=None,
        config=_loop_config(
            anchor_fraction=0.0,
            minimum_anchor_count=1,
            max_transactions=2,
            neighbor_window=4,
        ),
    )

    anchors = report["persistent_anchors"]["0"]
    assert report["accepted_count"] == 2
    assert anchors["selection_epoch"] == 0
    assert anchors["immutable_after_selection"] is True
    assert anchors["measurement"] == "exact-single-neuron-layer-output-ablation-frobenius"
    assert len(anchors["source_neuron_ids"]) == 1
    assert len(anchors["final_current_indices"]) == 1
    assert anchors["lineage_audit"]["all_source_paths_traced"] is True
    assert anchors["lineage_audit"]["maximum_source_path_multiplicity"] == 1


def test_transaction_history_is_streamed_and_inline_bounded(tmp_path: Path) -> None:
    model = _tiny_qwen(1801)
    initial = MODULE.model_accounting(model)
    ledger = tmp_path / "transactions.jsonl"
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1802),
        small_gate=_Decisions([False, False]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(max_transactions=2, max_history_records=1),
        transaction_ledger=ledger,
    )

    assert report["rejected_count"] == 2
    assert len(report["rejected_transactions"]) == 1
    assert report["history"]["rejected_omitted_from_inline"] == 1
    assert len(ledger.read_text(encoding="ascii").splitlines()) == 2
    assert report["history"]["ledger_sha256"] == MODULE._file_sha256(ledger)


def test_bounded_history_failure_does_not_destroy_previous_records() -> None:
    history = [{"attempt": 1, "decision": "kept"}]
    original = list(history)
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="single compact"):
        MODULE._append_bounded_history(
            history,
            {"attempt": 2, "payload": "x" * 10_000},
            max_records=2,
            max_json_bytes=MODULE._history_size(history),
        )
    assert history == original


def _identity_eval_suite(path: Path, suffix: str = "a") -> None:
    path.write_text(
        json.dumps(
            {
                "source": f"identity-{suffix}",
                "text_evals": [f"text {suffix}"],
                "choice_probes": [
                    {
                        "id": f"choice-{suffix}",
                        "prompt": f"choice {suffix}",
                        "choices": [" a", " b"],
                        "answer": 0,
                    }
                ],
                "generation_prompts": [
                    {"id": f"generation-{suffix}", "prompt": f"generate {suffix}"}
                ],
                "behavior_probes": [
                    {
                        "id": f"behavior-{suffix}",
                        "prompt": f"behavior {suffix}",
                        "kind": "first_number",
                        "answer": "1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cached_parent_binding_rejects_state_tokenizer_and_eval_mismatch(
    tmp_path: Path,
) -> None:
    model = _tiny_qwen(1851)
    tokenizer = _Tokenizer()
    suite_a = tmp_path / "suite-a.json"
    suite_b = tmp_path / "suite-b.json"
    _identity_eval_suite(suite_a, "a")
    _identity_eval_suite(suite_b, "b")
    identity = MODULE.build_exact_parent_identity(
        model,
        tokenizer,
        source_model="tiny-source",
        source_revision="tiny-revision",
        source_tokenizer="tiny-tokenizer",
        eval_suite=suite_a,
    )
    report = {
        "identity_binding": identity,
        "generation_health": {"contract": dict(MODULE.GENERATION_HEALTH_CONTRACT)},
    }
    assert MODULE.validate_bound_parent_report(
        report, identity, label="parent"
    )["state_sha256"] == identity["state_sha256"]

    with torch.inference_mode():
        model.model.embed_tokens.weight[0, 0].add_(1.0)
    changed_state = MODULE.build_exact_parent_identity(
        model,
        tokenizer,
        source_model="tiny-source",
        source_revision="tiny-revision",
        source_tokenizer="tiny-tokenizer",
        eval_suite=suite_a,
    )
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="state_sha256"):
        MODULE.validate_bound_parent_report(report, changed_state, label="parent")

    changed_eval = MODULE.derive_parent_identity_for_eval_suite(identity, suite_b)
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="eval_suite"):
        MODULE.validate_bound_parent_report(report, changed_eval, label="parent")


@pytest.mark.parametrize("behavior_count", [999, 1001])
def test_certification_contract_requires_exact_cardinality(
    behavior_count: int,
) -> None:
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="not exact"):
        MODULE._assert_production_certification_contract(
            {
                "counts": {
                    "behavior_probes": behavior_count,
                    "generation_prompts": 1000,
                    "choice_probes": 250,
                }
            },
            manual_sample_count=128,
        )


def test_manual_packet_contains_verbatim_outputs_and_requires_bound_verdict() -> None:
    parent = {
        "behavior_suite": {
            "items": [
                {
                    "id": "b",
                    "output": "parent behavior text",
                    "diagnostics": {"tokens": 3},
                }
            ]
        },
        "generations": [
            {"id": "g", "output": "parent generation text", "diagnostics": {}}
        ],
        "generation_diagnostics": [{"id": "g", "passed": True}],
    }
    child = {
        "behavior_suite": {
            "items": [
                {
                    "id": "b",
                    "output": "child behavior text",
                    "diagnostics": {"tokens": 3},
                }
            ]
        },
        "generations": [
            {"id": "g", "output": "child generation text", "diagnostics": {}}
        ],
        "generation_diagnostics": [{"id": "g", "passed": True}],
    }
    comparison = {
        "manual_samples": {
            "behavior": [{"id": "b", "outcome": "retained_pass"}],
            "generation": [{"id": "g", "outcome": "retained_pass"}],
            "selected_count": 2,
        }
    }
    packet = MODULE.build_manual_review_packet(
        parent, child, comparison, required_count=2
    )
    assert packet["selected_sample_count"] == 2
    assert all(item["parent_output"] and item["child_output"] for item in packet["samples"])
    bad = {
        "schema": "neural-forge-qualitative-review-verdict-v1",
        "verdict": "pass",
        "reviewer": "reviewer",
        "packet_sha256": "wrong",
        "reviewed_sample_count": 2,
        "notes": "Inspected every output.",
    }
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="different packet"):
        MODULE.validate_manual_review_verdict(bad, packet)
    bad["packet_sha256"] = packet["packet_sha256"]
    assert MODULE.validate_manual_review_verdict(bad, packet)["validated"] is True

    duplicate = {
        "manual_samples": {
            "behavior": [
                {"id": "b", "outcome": "retained_pass"},
                {"id": "b", "outcome": "retained_pass"},
            ],
            "generation": [{"id": "g", "outcome": "retained_pass"}],
            "selected_count": 3,
        }
    }
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="must be unique"):
        MODULE.build_manual_review_packet(
            parent, child, duplicate, required_count=3
        )

    child["generations"][0]["output"] = ""
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="cannot be empty"):
        MODULE.build_manual_review_packet(
            parent, child, comparison, required_count=2
        )


def test_resource_guard_enforces_host_cap(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "_current_process_rss_bytes", lambda: (1024, 2048))
    guard = MODULE.WorkflowResourceGuard(
        "cpu", 512, 1024, 1024, 60, {}
    )
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="Host RSS cap"):
        guard.check("adversarial")


def test_resource_guard_resolves_cuda_string_to_visible_integer_index(
    monkeypatch,
) -> None:
    resets: list[int] = []
    monkeypatch.setattr(MODULE, "_current_process_rss_bytes", lambda: (0, 0))
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(MODULE.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(MODULE.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "max_memory_allocated", lambda index: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "max_memory_reserved", lambda index: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "memory_allocated", lambda index: 1)
    monkeypatch.setattr(MODULE.torch.cuda, "memory_reserved", lambda index: 1)
    monkeypatch.setattr(
        MODULE.torch.cuda,
        "reset_peak_memory_stats",
        lambda index: resets.append(index),
    )
    guard = MODULE.WorkflowResourceGuard(
        "cuda:0", 1024, 1024, 1024, 60, {}
    )

    with guard.phase("gpu-phase"):
        pass

    assert resets == [0]


def test_resource_guard_skips_peak_reset_before_cuda_allocator_initialization(
    monkeypatch,
) -> None:
    resets: list[int] = []
    monkeypatch.setattr(MODULE, "_current_process_rss_bytes", lambda: (0, 0))
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(MODULE.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(MODULE.torch.cuda, "max_memory_allocated", lambda index: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "max_memory_reserved", lambda index: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "memory_allocated", lambda index: 0)
    monkeypatch.setattr(MODULE.torch.cuda, "memory_reserved", lambda index: 0)
    monkeypatch.setattr(
        MODULE.torch.cuda,
        "reset_peak_memory_stats",
        lambda index: resets.append(index),
    )
    guard = MODULE.WorkflowResourceGuard(
        "cuda:0", 1024, 1024, 1024, 60, {}
    )

    with guard.phase("pre-load"):
        pass

    assert resets == []


def test_certified_publication_refuses_unsealed_artifact(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.safetensors").write_bytes(b"weights")
    (staged / "manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "published"
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="not sealed"):
        MODULE.publish_certified_checkpoint(
            staged, destination, lock_path=tmp_path / "io.lock"
        )
    assert not destination.exists()


def test_multi_anchor_runner_strategy_uses_bound_split_ids_and_same_gate() -> None:
    model = _tiny_qwen(1901, layers=1)
    initial = MODULE.model_accounting(model)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1902),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(
            fold_strategy="multi-anchor",
            calibration_provenance_id="calibration-provenance-sha",
            validation_provenance_id="validation-provenance-sha",
            multi_anchor_candidate_pool_size=2,
            multi_anchor_max_anchors=2,
            multi_anchor_similarity_chunk_size=2,
            multi_anchor_max_similarity_products=64,
            multi_anchor_max_projection_fits=64,
        ),
    )

    proposal = report["accepted_transactions"][0]["proposal"]
    assert report["target"]["reached"] is True
    assert proposal["fold_strategy"] == "multi-anchor"
    assert proposal["multi_anchor_merge_count"] == 1
    assert proposal["internal_split_proof"]["pair_discovery"] == "calibration-only"
    assert proposal["strategy_evidence"]["independent_aggregate_metrics_trusted"] is True
    assert proposal["strategy_evidence"]["per_merge_metrics_used_for_runner_selection"] is False
    assert proposal["strategy_evidence"]["work_caps"]["all_work_caps_verified"] is True
    assert proposal["strategy_evidence"]["work_caps"]["similarity_products"] <= 64
    assert proposal["strategy_evidence"]["work_caps"]["projection_fit_count"] <= 64
    assert proposal["strategy_evidence"]["activation_evidence"][
        "activation_hashes_distinct"
    ] is True
    assert report["persistent_anchors"]["0"]["lineage_audit"][
        "all_source_paths_traced"
    ] is True
    lineage_audit = report["persistent_anchors"]["0"]["lineage_audit"]
    assert lineage_audit["covered_source_path_count"] == lineage_audit["source_width"]
    assert lineage_audit["missing_source_path_count"] == 0
    merge = proposal["merges"][0]
    assert merge["anchors"]
    assert merge["dropped"] not in merge["anchors"]


def test_multi_anchor_strategy_rejects_tampered_observed_work_caps(monkeypatch) -> None:
    model = _tiny_qwen(1911, layers=1)
    initial = MODULE.model_accounting(model)
    real_fit = MODULE.fit_multi_anchor_neuron_fold

    def tampered_fit(*args, **kwargs):
        fold = real_fit(*args, **kwargs)
        fold.diagnostics = replace(
            fold.diagnostics,
            similarity_products=int(kwargs["max_similarity_products"]) + 1,
        )
        return fold

    monkeypatch.setattr(MODULE, "fit_multi_anchor_neuron_fold", tampered_fit)
    report = MODULE.run_fold_transactions(
        model,
        activation_provider=_ActivationProvider(1912),
        small_gate=_Decisions([True]),
        target_parameters=initial.unique_parameter_count - 24,
        target_bytes=None,
        config=_loop_config(
            fold_strategy="multi-anchor",
            multi_anchor_candidate_pool_size=2,
            multi_anchor_max_anchors=2,
            multi_anchor_similarity_chunk_size=2,
            multi_anchor_max_similarity_products=64,
            multi_anchor_max_projection_fits=64,
        ),
    )

    assert report["target"]["reached"] is False
    assert report["accepted_count"] == 0
    assert report["stop_reason"] == "no-eligible-arbitrary-pair-proposal"
    assert "similarity" in report["layer_skips"][0]["reason"].lower()
    assert MODULE.model_accounting(model) == initial


@pytest.mark.parametrize("strategy", ["pair", "multi-anchor"])
def test_fold_strategy_fails_closed_without_distinct_provenance(
    strategy: str,
) -> None:
    with pytest.raises(MODULE.QwenCompensatedRunnerError, match="distinct exact"):
        _loop_config(
            fold_strategy=strategy,
            calibration_provenance_id="same",
            validation_provenance_id="same",
        )
