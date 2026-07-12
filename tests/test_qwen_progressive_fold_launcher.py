from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.launch_qwen_progressive_fold_run as launcher


SOURCE_SHA = "a" * 64
IMPLEMENTATION_SHA = "b" * 64


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_qwen4b_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151_936,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "attention_bias": False,
    }
    (path / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fixture-parent-weights")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = _write_qwen4b_checkpoint(tmp_path / "source")
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text":"calibration"}\n', encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "behavior_probes": [],
                "generation_prompts": [],
                "choice_probes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source, calibration, suite


def _build_plan(tmp_path: Path, *, output_name: str = "run") -> dict:
    source, calibration, suite = _write_inputs(tmp_path)
    return launcher.build_progressive_plan(
        source_model=source,
        output_dir=tmp_path / output_name,
        cache_dir=tmp_path / "cache",
        calibration_file=calibration,
        eval_suite=suite,
        python="python-test",
        stages=launcher.default_stage_specs(),
        compose_settings=launcher.ComposeSettings(),
        source_content_sha256=SOURCE_SHA,
        implementation_sha256=IMPLEMENTATION_SHA,
    )


def test_default_schedule_has_documented_shapes_and_exact_counts() -> None:
    stages = launcher.default_stage_specs()

    assert [stage.num_hidden_layers for stage in stages] == [30, 24, 18]
    assert [stage.hidden_size for stage in stages] == [1536, 1024, 384]
    assert [stage.intermediate_size for stage in stages] == [5120, 3072, 1152]
    assert [stage.num_attention_heads for stage in stages] == [16, 8, 8]
    assert [stage.num_key_value_heads for stage in stages] == [4, 2, 2]
    assert [stage.head_dim for stage in stages] == [128, 128, 128]

    counts = [launcher._stage_parameter_count(stage, 151_936) for stage in stages]
    assert counts == [1_410_567_168, 600_628_224, 158_288_256]


def test_stage_manifest_and_compact_cli_overrides_are_strict(tmp_path: Path) -> None:
    manifest_path = tmp_path / "stages.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": launcher.STAGE_MANIFEST_SCHEMA,
                "stages": [
                    {
                        "name": "middle",
                        "h": 256,
                        "i": 768,
                        "l": 4,
                        "q": 4,
                        "kv": 2,
                        "hd": 64,
                        "seed": 11,
                        "rollout": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = launcher.load_stage_specs(stage_manifest=manifest_path)
    compact = launcher.load_stage_specs(
        stage_overrides=["name=small,h=128,i=256,l=3,q=2,kv=1,hd=64,tie=false"],
        default_seed=7,
    )

    assert loaded[0].seed == 11
    assert loaded[0].rollout_refold_rounds == 2
    assert compact[0].name == "small"
    assert compact[0].seed == 7
    assert compact[0].tie_word_embeddings is False
    with pytest.raises(ValueError, match="either"):
        launcher.load_stage_specs(
            stage_manifest=manifest_path,
            stage_overrides=["name=x,h=1,i=1,l=1,q=1,kv=1,hd=1"],
        )
    with pytest.raises(ValueError, match="unknown stage field"):
        launcher.parse_stage_override("name=x,h=1,i=1,l=1,q=1,kv=1,hd=1,mystery=2")


def test_target_command_locks_seed_shape_and_config_only_builder(
    tmp_path: Path,
) -> None:
    stage = launcher.default_stage_specs(seed=17)[0]
    count = launcher._stage_parameter_count(stage, 151_936)
    command = launcher.build_target_command(
        python="python-test",
        tokenizer_model=tmp_path / "tokenizer",
        source_model=tmp_path / "donor",
        output=tmp_path / "blank",
        source_parameter_count=4_000_000_000,
        parameter_count=count,
        stage=stage,
        dtype="float32",
    )

    assert command[:4] == [
        "python-test",
        "-m",
        "neural_forge.cli",
        "hf-qwen-build-target",
    ]
    assert _value_after(command, "--target-params") == str(count)
    assert _value_after(command, "--hidden-sizes") == "1536"
    assert _value_after(command, "--intermediate-sizes") == "5120"
    assert _value_after(command, "--layer-counts") == "30"
    assert _value_after(command, "--query-head-counts") == "16"
    assert _value_after(command, "--key-value-head-counts") == "4"
    assert _value_after(command, "--head-dims") == "128"
    assert _value_after(command, "--seed") == "17"
    assert "from_pretrained" not in " ".join(command)


def test_compose_command_always_uses_adaptive_spans_and_optional_rollout(
    tmp_path: Path,
) -> None:
    settings = launcher.ComposeSettings(
        adaptive_depth_critical_quantile=0.72,
        adaptive_depth_causal_magnitude_weight=2.0,
        rollout_refold_rounds=0,
    )
    command = launcher.build_compose_command(
        python="python-test",
        source_model=tmp_path / "parent",
        blank_target=tmp_path / "blank",
        output=tmp_path / "folded",
        calibration_file=tmp_path / "calibration.jsonl",
        settings=settings,
        rollout_refold_rounds=2,
    )

    assert command[:4] == [
        "python-test",
        "-m",
        "neural_forge.cli",
        "hf-qwen-pair-compose",
    ]
    assert "--adaptive-depth-spans" in command
    assert "--donor-layer-spans" not in command
    assert _value_after(command, "--adaptive-depth-critical-quantile") == "0.72"
    assert _value_after(command, "--adaptive-depth-causal-magnitude-weight") == "2.0"
    assert _value_after(command, "--rollout-refold-rounds") == "2"
    flags = [token.lower() for token in command if token.startswith("--")]
    assert not any(
        forbidden in flag
        for flag in flags
        for forbidden in ("recover", "train", "optimizer", "epochs", "steps")
    )


def test_every_eval_uses_full_limits_and_manual_generation(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)

    records = [plan["parent"], *plan["stages"]]
    for record in records:
        eval_command = record["commands"]["eval"]["argv"]
        manual_command = record["commands"]["manual"]["argv"]
        assert _value_after(eval_command, "--max-behavior-probes") == "1000"
        assert _value_after(eval_command, "--max-generation-prompts") == "1000"
        assert _value_after(eval_command, "--max-choice-probes") == "250"
        assert "_manual-eval" in manual_command
        assert _value_after(manual_command, "--max-new-tokens") == "96"
    assert plan["final"]["eval_command"] == plan["stages"][-1]["commands"]["eval"]
    assert plan["final"]["manual_command"] == plan["stages"][-1]["commands"]["manual"]


def test_plan_records_random_blank_provenance_counts_and_zero_recovery(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)

    assert plan["zero_recovery_assertion"] == {
        "asserted": True,
        "gradient_updates": 0,
        "recovery_commands": 0,
        "training_commands": 0,
        "blank_children_use_pretrained_weights": False,
        "allowed_model_mutations": (
            "seeded config-only target construction and closed-form pair composition"
        ),
        "command_count_checked": 14,
    }
    assert [stage["parameter_count"] for stage in plan["stages"]] == [
        1_410_567_168,
        600_628_224,
        158_288_256,
    ]
    for stage in plan["stages"]:
        assert stage["blank_target_origin"]["pretrained_child_weights_loaded"] is False
        assert stage["depth_plan"]["adaptive"] is True
        assert stage["depth_plan"]["maximum_donor_span_length"] == 2
        assert stage["seed"] == 0
    assert plan["final"]["parameter_count"] == 158_288_256
    assert plan["final"]["cumulative_compression_ratio"] > 25


def test_eval_cache_keys_are_content_and_settings_stable_across_run_paths(
    tmp_path: Path,
) -> None:
    first = _build_plan(tmp_path / "first", output_name="run-a")
    second = _build_plan(tmp_path / "second", output_name="run-b")

    assert first["parent"]["cache"]["eval_key"] == second["parent"]["cache"]["eval_key"]
    assert (
        first["parent"]["cache"]["manual_key"]
        == second["parent"]["cache"]["manual_key"]
    )
    assert [stage["model_identity"] for stage in first["stages"]] == [
        stage["model_identity"] for stage in second["stages"]
    ]
    assert [stage["cache"]["eval_key"] for stage in first["stages"]] == [
        stage["cache"]["eval_key"] for stage in second["stages"]
    ]

    changed = launcher.build_progressive_plan(
        source_model=Path(first["inputs"]["source_model"]),
        output_dir=tmp_path / "changed",
        cache_dir=tmp_path / "cache",
        calibration_file=Path(first["inputs"]["calibration_file"]),
        eval_suite=Path(first["inputs"]["eval_suite"]),
        python="python-test",
        stages=launcher.default_stage_specs(seed=1),
        compose_settings=launcher.ComposeSettings(),
        source_content_sha256=SOURCE_SHA,
        implementation_sha256=IMPLEMENTATION_SHA,
    )
    assert (
        changed["stages"][0]["cache"]["eval_key"]
        != first["stages"][0]["cache"]["eval_key"]
    )


def test_validated_parent_reports_are_reused_without_rebenchmarking(
    tmp_path: Path,
) -> None:
    source, calibration, suite = _write_inputs(tmp_path)
    cached_eval = tmp_path / "cached_parent_eval.json"
    cached_eval.write_text(
        json.dumps(
            {
                "behavior_suite": {"correct": 400, "total": 1000},
                "generation_health": {"correct": 388, "total": 1000},
                "choice_accuracy": {"correct": 250, "total": 250},
            }
        ),
        encoding="utf-8",
    )
    cached_manual = tmp_path / "cached_parent_manual.json"
    cached_manual.write_text(
        json.dumps({"manual_generations": [{"prompt": "Q", "output": "A"}]}),
        encoding="utf-8",
    )

    plan = launcher.build_progressive_plan(
        source_model=source,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        calibration_file=calibration,
        eval_suite=suite,
        python="python-test",
        stages=launcher.default_stage_specs(),
        compose_settings=launcher.ComposeSettings(),
        source_content_sha256=SOURCE_SHA,
        implementation_sha256=IMPLEMENTATION_SHA,
        cached_parent_eval=cached_eval,
        cached_parent_manual=cached_manual,
    )
    script = launcher.render_run_script(plan, tmp_path / "run_manifest.json")

    assert plan["parent"]["cache"]["eval_report"] == str(cached_eval.resolve())
    assert plan["parent"]["cache"]["manual_report"] == str(cached_manual.resolve())
    assert plan["parent"]["supplied_eval"]["validated"] is True
    assert plan["parent"]["supplied_manual"]["validated"] is True
    assert plan["cache_policy"]["validated_supplied_parent_eval_reused"] is True
    assert str(cached_eval.resolve()) in script
    assert str(cached_manual.resolve()) in script

    cached_eval.write_text(
        json.dumps(
            {
                "behavior_suite": {"total": 3},
                "generation_health": {"total": 3},
                "choice_accuracy": {"total": 3},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 1000 behavior"):
        launcher.build_progressive_plan(
            source_model=source,
            output_dir=tmp_path / "invalid",
            cache_dir=tmp_path / "cache",
            calibration_file=calibration,
            eval_suite=suite,
            python="python-test",
            stages=launcher.default_stage_specs(),
            compose_settings=launcher.ComposeSettings(),
            source_content_sha256=SOURCE_SHA,
            implementation_sha256=IMPLEMENTATION_SHA,
            cached_parent_eval=cached_eval,
        )


def test_rendered_sidecar_is_sequential_cached_and_contains_no_recovery(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    script = launcher.render_run_script(plan, tmp_path / "run_manifest.json")

    assert script.count("hf-qwen-build-target") == 3
    assert script.count("hf-qwen-pair-compose") == 3
    assert script.count("hf-eval") == 4
    assert script.count("_manual-eval") == 4
    assert script.count('echo "cache_hit=') == 8
    assert "--adaptive-depth-spans" in script
    assert "grep -Fqx --" in script
    assert ".neural_forge_progressive_identity" in script
    assert "nohup" not in script
    assert "ssh " not in script
    assert "hf-recover" not in script
    assert "hf-xfamily-block-recover" not in script


def test_stage_bounds_reject_one_shot_long_spans_and_wrong_count_assertion(
    tmp_path: Path,
) -> None:
    source, calibration, suite = _write_inputs(tmp_path)
    too_deep = launcher.StageSpec("too_small", 384, 1152, 10, 8, 2, 128)
    wrong_count = launcher.StageSpec(
        "wrong_count",
        1536,
        5120,
        30,
        16,
        4,
        128,
        target_params=123,
    )

    with pytest.raises(ValueError, match="spans longer than two"):
        launcher.build_progressive_plan(
            source_model=source,
            output_dir=tmp_path / "too-deep",
            cache_dir=tmp_path / "cache",
            calibration_file=calibration,
            eval_suite=suite,
            python="python-test",
            stages=[too_deep],
            compose_settings=launcher.ComposeSettings(),
            source_content_sha256=SOURCE_SHA,
            implementation_sha256=IMPLEMENTATION_SHA,
        )
    with pytest.raises(ValueError, match="does not match exact"):
        launcher.build_progressive_plan(
            source_model=source,
            output_dir=tmp_path / "wrong-count",
            cache_dir=tmp_path / "cache",
            calibration_file=calibration,
            eval_suite=suite,
            python="python-test",
            stages=[wrong_count],
            compose_settings=launcher.ComposeSettings(),
            source_content_sha256=SOURCE_SHA,
            implementation_sha256=IMPLEMENTATION_SHA,
        )


def test_checkpoint_content_identity_is_path_independent(tmp_path: Path) -> None:
    first = _write_qwen4b_checkpoint(tmp_path / "first")
    second = _write_qwen4b_checkpoint(tmp_path / "second")

    assert launcher.checkpoint_content_sha256(
        first
    ) == launcher.checkpoint_content_sha256(second)
    (second / "model.safetensors").write_bytes(b"different")
    assert launcher.checkpoint_content_sha256(
        first
    ) != launcher.checkpoint_content_sha256(second)


def test_main_emits_manifest_without_executing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, calibration, suite = _write_inputs(tmp_path)
    run_root = tmp_path / "runs"
    run_root.mkdir()
    monkeypatch.setattr(
        launcher, "implementation_content_sha256", lambda: IMPLEMENTATION_SHA
    )

    result = launcher.main(
        [
            "--source-model",
            str(source),
            "--run-root",
            str(run_root),
            "--tag",
            "proof",
            "--calibration-file",
            str(calibration),
            "--eval-suite",
            str(suite),
            "--source-content-sha256",
            SOURCE_SHA,
            "--python",
            "python-test",
        ]
    )

    manifest = json.loads(
        (run_root / "proof" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert result == 0
    assert manifest["orchestration"]["status"] == "planned"
    assert manifest["orchestration"]["execute_requested"] is False
    assert manifest["orchestration"]["remote_submission"] is False
    assert not (run_root / "proof" / "SUCCESS").exists()
