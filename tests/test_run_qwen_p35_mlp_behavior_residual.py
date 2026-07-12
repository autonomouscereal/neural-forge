from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.",
        allow_module_level=True,
    )

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_p35_mlp_behavior_residual as runner  # noqa: E402
from neural_forge.hf_behavior_residual_swiglu import (  # noqa: E402
    audit_mlp_codebook_redirects,
)
from neural_forge.hf_functional_decode_trace import (  # noqa: E402
    FunctionalDecodeLimits,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 24

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


def _parent_and_p35():
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    torch.manual_seed(5150)
    parent = transformers.Qwen3ForCausalLM(config).eval()
    child = transformers.Qwen3ForCausalLM(config)
    child.load_state_dict(parent.state_dict(), strict=True)
    tokenizer = _Tokenizer()
    child = prepare_qwen3_codebook_model(
        child.eval(),
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=3,
        neuron_prototypes_per_neuron=2,
    )
    registry = child.neuron_codebook.registry
    candidate_audits = [
        audit_mlp_codebook_redirects(registry, selected_layers=(index,))
        for index in range(3)
    ]
    selected = max(
        candidate_audits,
        key=lambda value: value.selected_redirected_assignments,
    ).selected_layers
    return parent, child, selected


def _limits() -> FunctionalDecodeLimits:
    return FunctionalDecodeLimits(
        max_horizon=1,
        max_prompt_tokens=16,
        max_vocabulary=32,
        max_logit_bytes=100_000,
        max_model_devices=1,
    )


def test_exact_constants_and_accounting_capture_real_p35_budget() -> None:
    assert sum(runner.EXACT_REDIRECT_COUNTS) == 9728
    assert runner.EXACT_REDIRECT_COUNTS[1] == 2122
    assert runner.EXACT_FIT_REFRESH_STEPS == (8, 16, 24)
    assert 2 * 3 * 2 * 4 == 48
    assert len(runner.REJECTED_V2_FIT_BATCH_MANIFEST_SHA256) == 64
    audit = runner.MLPCodebookRedirectAudit(
        logical_layers=36,
        intermediate_size=9728,
        prototypes_per_neuron=35,
        prototype_count=340_480,
        redirected_assignments_by_layer=runner.EXACT_REDIRECT_COUNTS,
        selected_layers=(1,),
        selected_redirected_assignments=2122,
        assignment_sha256=runner.EXACT_ASSIGNMENT_SHA256,
        prototype_source_indices_sha256=runner.EXACT_PROTOTYPE_SOURCE_SHA256,
    )
    runner._validate_exact_p35_audit(audit)
    accounting = runner.build_p35_mlp_residual_accounting(audit, residual_width=64)
    assert accounting.residual_parameters == 491_520
    assert accounting.net_parameters_saved_vs_unfolded_codebook == 74_219_520

    lineage = {
        "p35_manifest_sha256": runner.EXACT_P35_MANIFEST_SHA256,
        "p35_model_safetensors_sha256": runner.EXACT_P35_MODEL_SHA256,
        "p35_state_key_sha256": runner.EXACT_P35_STATE_KEY_SHA256,
        "tokenizer_contract_sha256": runner.EXACT_TOKENIZER_CONTRACT_SHA256,
        "tokenizer_vocab_sha256": runner.EXACT_TOKENIZER_VOCAB_SHA256,
        "parent_token_rows_sha256": runner.EXACT_PARENT_TOKEN_ROWS_SHA256,
        "raw_parent_revision": runner.EXACT_PARENT_REVISION,
        "raw_parent_shard_content_hashes": list(
            runner.EXACT_PARENT_SHARD_CONTENT_HASHES
        ),
        "checkpoint_loader_verified_manifest": True,
        "live_tokenizer_contract_verified": True,
        "raw_parent_content_address_verified": True,
    }
    runner._validate_exact_lineage(lineage)
    with pytest.raises(runner.QwenP35MLPBehaviorResidualError, match="lineage"):
        runner._validate_exact_lineage({**lineage, "raw_parent_revision": "wrong"})
    with pytest.raises(
        runner.QwenP35MLPBehaviorResidualError,
        match="pre-model sealed fresh heldout",
    ):
        runner.run_qwen_p35_mlp_behavior_residual(
            torch.nn.Identity(),
            torch.nn.Identity(),
            (torch.tensor([[1]]),),
            (torch.tensor([[2]]),),
            (torch.tensor([[3]]),),
            split_provenance={
                "verified_exact_fixture_provenance": True,
                "fresh_manifest_verified_before_model_access": True,
                "fresh_ids_and_text_disjoint_from_all_prior_splits": True,
                "fresh_v3_heldout_fixture_sha256": "0" * 64,
                "fresh_v3_heldout_manifest_sha256": "1" * 64,
            },
        )


def test_bounded_runner_locks_before_heldout_and_restores_p35(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child, selected = _parent_and_p35()
    registry = child.neuron_codebook.registry
    state_before = {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }
    layer_views_before = tuple(layer.mlp for layer in child.model.layers)
    locked = False
    original_train = runner.train_behavior_residual_swiglu_disjoint
    original_collect = runner.action_runner._collect_greedy_split
    child_calibration_calls = 0
    child_fit_calls = 0

    def train_then_lock(*args, **kwargs):
        nonlocal locked
        result = original_train(*args, **kwargs)
        locked = True
        return result

    def reject_early_heldout(*args, split, **kwargs):
        nonlocal child_calibration_calls, child_fit_calls
        if split == "heldout-evaluation":
            assert locked
        if args[0] is child and split == "calibration-selection":
            child_calibration_calls += 1
        if args[0] is child and split == "fit":
            child_fit_calls += 1
        return original_collect(*args, split=split, **kwargs)

    monkeypatch.setattr(
        runner, "train_behavior_residual_swiglu_disjoint", train_then_lock
    )
    monkeypatch.setattr(
        runner.action_runner, "_collect_greedy_split", reject_early_heldout
    )
    report = runner.run_qwen_p35_mlp_behavior_residual(
        parent,
        child,
        (torch.tensor([[1, 3, 4]]), torch.tensor([[1, 3, 9]])),
        (torch.tensor([[1, 5, 6]]),),
        (torch.tensor([[1, 7, 8]]),),
        split_provenance={"test": True},
        require_exact_p35=False,
        layer_indices=selected,
        residual_width=1,
        horizon=1,
        steps=4,
        micro_batch_size=1,
        gate_up_learning_rate=1e-3,
        down_learning_rate=2e-3,
        calibration_steps=(0, 2, 4),
        fit_refresh_steps=(2,),
        trace_limits=_limits(),
    )

    assert report["p35_runtime_restored_exactly"] is True
    assert report["checkpoint_saved"] is False
    assert report["factor_pack_saved"] is False
    assert report["broad_behavior_certification_passed"] is False
    assert report["training"]["calibration_rollout_evaluations"] == 4
    assert child_calibration_calls == 5
    assert child_fit_calls == 2
    assert report["calibration_winner_lock"]["selected_by"] == (
        "calibration-free-rollout-then-action-lexicographic"
    )
    assert report["calibration_rollout_callback_audit"]["callback_calls"] == 4
    assert report["on_policy_fit_refresh_audit"]["callback_calls"] == 1
    assert report["on_policy_fit_refresh_audit"]["callback_splits"] == ["fit"]
    assert (
        report["on_policy_fit_refresh_audit"]["calibration_callback_calls_for_training"]
        == 0
    )
    assert (
        report["on_policy_fit_refresh_audit"]["heldout_callback_calls_for_training"]
        == 0
    )
    assert report["on_policy_fit_refresh_audit"]["manual_adam_moment_reset_count"] == 0
    assert report["on_policy_fit_refresh_audit"]["manual_adam_moments_preserved"]
    assert report["on_policy_fit_refresh_audit"]["update_mix"] == {
        "parent_greedy_anchor_updates": 2,
        "student_greedy_updates_total": 2,
        "initial_exact_zero_student_updates": 1,
        "post_refresh_current_student_updates": 1,
        "intentional_mixed_dagger_parent_anchors": True,
    }
    assert report["on_policy_fit_refresh_audit"]["discarded_refreshed_fit_batches"] == 0
    assert (
        report["calibration_rollout_callback_audit"]["heldout_split_callback_calls"]
        == 0
    )
    assert (
        report["calibration_rollout_callback_audit"][
            "prelock_heldout_model_forward_event_count"
        ]
        == 0
    )
    assert (
        report["heldout_access_audit"]["first_heldout_model_forward_event_index"]
        > report["heldout_access_audit"]["calibration_lock_event_index"]
    )
    assert tuple(layer.mlp for layer in child.model.layers) == layer_views_before
    for name, expected in state_before.items():
        assert torch.equal(registry.state_dict()[name], expected)
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_parser_requires_separate_parent_and_p35_sources() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--parent-model",
            "parent",
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "tokenizer",
            "--output",
            "report.json",
        ]
    )
    assert args.parent_device != args.device
    assert args.residual_width == 64
    assert args.steps == 32
    assert tuple(args.calibration_steps) == (0, 8, 16, 32)
    assert runner.EXACT_CALIBRATION_STEPS == (0, 8, 16, 32)
    assert tuple(args.fit_refresh_steps) == (8, 16, 24)
    assert args.heldout_evaluation_prompts_json == runner.DEFAULT_HELDOUT_PROMPTS
    assert args.sealed_heldout_manifest == runner.DEFAULT_SEALED_HELDOUT_MANIFEST


def test_v3_source_lineage_content_addresses_every_runtime_source() -> None:
    project_root = Path(__file__).resolve().parents[1]
    lineage = runner.source_file_lineage(project_root)
    assert tuple(lineage["files_sha256"]) == runner.SOURCE_LINEAGE_FILES
    for relative_path, observed in lineage["files_sha256"].items():
        expected = hashlib.sha256(
            (project_root / relative_path).read_bytes()
        ).hexdigest()
        assert observed == expected
    assert len(lineage["source_lineage_sha256"]) == 64


def test_v2_loader_seals_fresh_heldout_before_model_access(tmp_path: Path) -> None:
    fixtures = runner.load_verified_v2_prompt_splits(
        runner.DEFAULT_FIT_PROMPTS,
        runner.DEFAULT_CALIBRATION_PROMPTS,
        runner.V2_DEVELOPMENT_HELDOUT_PROMPTS,
        sealed_heldout_manifest_path=runner.V2_SEALED_HELDOUT_MANIFEST,
    )
    assert len(fixtures.heldout) == 8
    assert len(fixtures.v1_development_heldout) == 8
    assert set(fixtures.heldout).isdisjoint(
        {
            *fixtures.fit,
            *fixtures.calibration,
            *fixtures.v1_development_heldout,
        }
    )
    assert fixtures.provenance["fresh_manifest_verified_before_model_access"]
    assert fixtures.provenance["fresh_ids_and_text_disjoint_from_all_v1_splits"]
    assert (
        hashlib.sha256(runner.V2_DEVELOPMENT_HELDOUT_PROMPTS.read_bytes()).hexdigest()
        == runner.V2_SEALED_HELDOUT_FIXTURE_SHA256
    )
    assert (
        hashlib.sha256(runner.V2_SEALED_HELDOUT_MANIFEST.read_bytes()).hexdigest()
        == runner.V2_SEALED_HELDOUT_MANIFEST_SHA256
    )

    mutated = tmp_path / "mutated-heldout.json"
    mutated.write_bytes(runner.V2_DEVELOPMENT_HELDOUT_PROMPTS.read_bytes() + b" ")
    with pytest.raises(runner.QwenP35MLPBehaviorResidualError, match="SHA-256"):
        runner.load_verified_v2_prompt_splits(
            runner.DEFAULT_FIT_PROMPTS,
            runner.DEFAULT_CALIBRATION_PROMPTS,
            mutated,
            sealed_heldout_manifest_path=runner.V2_SEALED_HELDOUT_MANIFEST,
        )


def test_v3_loader_excludes_every_v1_v2_evaluation_before_model_access() -> None:
    fixtures = runner.load_verified_v3_prompt_splits(
        runner.DEFAULT_FIT_PROMPTS,
        runner.DEFAULT_CALIBRATION_PROMPTS,
        runner.DEFAULT_HELDOUT_PROMPTS,
    )
    assert len(fixtures.heldout) == 8
    assert set(fixtures.heldout).isdisjoint(
        {
            *fixtures.fit,
            *fixtures.calibration,
            *fixtures.v1_development_heldout,
            *fixtures.v2_development_heldout,
        }
    )
    assert fixtures.provenance["fresh_ids_and_text_disjoint_from_all_prior_splits"]
    main_source = inspect.getsource(runner.main)
    assert main_source.index("load_verified_v3_prompt_splits") < main_source.index(
        "from_pretrained"
    )


def test_on_policy_phase_changes_only_student_derived_fit_batches() -> None:
    parent, child, _selected = _parent_and_p35()
    prompts = (torch.tensor([[1, 3, 4]]), torch.tensor([[1, 5, 6]]))
    limits = FunctionalDecodeLimits(
        max_horizon=2,
        max_prompt_tokens=16,
        max_vocabulary=32,
        max_logit_bytes=100_000,
        max_model_devices=1,
    )
    parent_view = runner._ParentTokenRowView(parent, child.parent_token_rows)
    parent_fit, _ = runner.action_runner._collect_greedy_split(
        parent_view, prompts, split="fit", horizon=2, limits=limits
    )
    child_fit, child_drivers = runner.action_runner._collect_greedy_split(
        child, prompts, split="fit", horizon=2, limits=limits
    )
    parent_on_child = runner.action_runner._collect_forced_split(
        parent_view, prompts, child_drivers, split="fit", horizon=2, limits=limits
    )
    changed_drivers = {
        index: tuple((token + 1) % 24 for token in values)
        for index, values in child_drivers.items()
    }
    changed_child_forced = runner.action_runner._collect_forced_split(
        child, prompts, changed_drivers, split="fit", horizon=2, limits=limits
    )
    changed_child_fit = tuple(
        replace(trace, lane="greedy") for trace in changed_child_forced
    )
    parent_on_changed = runner.action_runner._collect_forced_split(
        parent_view,
        prompts,
        changed_drivers,
        split="fit",
        horizon=2,
        limits=limits,
    )
    with pytest.raises(
        runner.QwenP35MLPBehaviorResidualError,
        match="semantically bound",
    ):
        runner._build_on_policy_fit_phase(
            parent_fit,
            changed_child_forced,
            parent_on_changed,
            prompts,
            phase_index=1,
            refresh_after_step=1,
            micro_batch_size=1,
        )
    baseline_phase = runner._build_on_policy_fit_phase(
        parent_fit,
        child_fit,
        parent_on_child,
        prompts,
        phase_index=1,
        refresh_after_step=1,
        micro_batch_size=1,
    )
    changed_phase = runner._build_on_policy_fit_phase(
        parent_fit,
        changed_child_fit,
        parent_on_changed,
        prompts,
        phase_index=1,
        refresh_after_step=1,
        micro_batch_size=1,
    )
    baseline_parent = [
        batch.as_dict()
        for batch in baseline_phase.batches
        if batch.prefix_source == "parent-greedy"
    ]
    changed_parent = [
        batch.as_dict()
        for batch in changed_phase.batches
        if batch.prefix_source == "parent-greedy"
    ]
    baseline_student = [
        batch.as_dict()
        for batch in baseline_phase.batches
        if batch.prefix_source == "student-greedy"
    ]
    changed_student = [
        batch.as_dict()
        for batch in changed_phase.batches
        if batch.prefix_source == "student-greedy"
    ]
    assert baseline_parent == changed_parent
    assert baseline_student != changed_student
    assert baseline_phase.provenance["fit_split"] == "fit"
    assert changed_phase.provenance["calibration_model_forwards"] == 0
    assert changed_phase.provenance["heldout_model_forwards"] == 0


def test_parent_token_row_view_maps_inputs_and_projects_logits() -> None:
    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(5, 3)
            self.output = torch.nn.Linear(3, 5, bias=False)
            self.config = SimpleNamespace(vocab_size=5)
            self.seen = None

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.output

        def forward(self, *, input_ids, **_kwargs):
            self.seen = input_ids.detach().clone()
            width = int(input_ids.shape[1])
            logits = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
            return SimpleNamespace(logits=logits.expand(1, width, 5).clone())

    parent = Parent()
    view = runner._ParentTokenRowView(parent, torch.tensor([2, 0, 4]))
    output = view(input_ids=torch.tensor([[0, 1, 2]]))
    assert view.config is not parent.config
    assert view.config.vocab_size == 3
    assert torch.equal(parent.seen, torch.tensor([[2, 0, 4]]))
    assert torch.equal(output.logits[0, 0], torch.tensor([2.0, 0.0, 4.0]))


def test_content_hash_lineage_rejects_one_byte_mutation(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first-shard")
    second.write_bytes(b"second-shard")
    expected = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)
    )
    assert runner._verify_file_content_hashes(
        (first, second), expected, name="test shards"
    ) == list(expected)
    second.write_bytes(b"second-shard!")
    with pytest.raises(runner.QwenP35MLPBehaviorResidualError, match="content hashes"):
        runner._verify_file_content_hashes(
            (first, second), expected, name="test shards"
        )


def test_remote_wrapper_pins_two_gpu_order_and_content_lineage() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "remote_run_p35_mlp_behavior_residual_width64_p40_pair_20260711.sh"
    ).read_text(encoding="utf-8")
    assert "--runtime=nvidia" in wrapper
    assert 'NVIDIA_VISIBLE_DEVICES="$PARENT_GPU,$CHILD_GPU"' in wrapper
    assert '--gpus "device=$PARENT_GPU,$CHILD_GPU"' not in wrapper
    assert 'test "$VISIBLE_GPU_ROWS"' in wrapper
    assert "--calibration-steps 0 8 16 32" in wrapper
    assert "--fit-refresh-steps 8 16 24" in wrapper
    assert "dagger_v3" in wrapper
    assert runner.RUN_SCHEMA in wrapper
    assert "calibration_rollout_selection_key" in wrapper
    assert runner.EXACT_SEALED_HELDOUT_FIXTURE_SHA256 in wrapper
    assert runner.EXACT_SEALED_HELDOUT_MANIFEST_SHA256 in wrapper
    assert runner.DEFAULT_HELDOUT_PROMPTS.name in wrapper
    assert runner.DEFAULT_SEALED_HELDOUT_MANIFEST.name in wrapper
    for digest in (
        runner.EXACT_P35_MANIFEST_SHA256,
        runner.EXACT_P35_MODEL_SHA256,
        runner.EXACT_TOKENIZER_CONTRACT_SHA256,
        *runner.EXACT_PARENT_SHARD_CONTENT_HASHES,
        *runner.source_file_lineage()["files_sha256"].values(),
    ):
        assert digest in wrapper
