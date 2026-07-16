from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_folded_blank_recovery_inputs as preparer  # noqa: E402
from neural_forge import hf_recovery, hf_teacher_token_replay  # noqa: E402


STATE_SHA = "1" * 64


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _IdentityTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        **_kwargs: Any,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is True
        assert truncation is False
        return {"input_ids": [ord(character) + 10 for character in text]}

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(chr(int(token_id) - 10) for token_id in values)


def _token_prompt(base: int) -> dict[str, Any]:
    token_ids = [base + offset for offset in range(8)]
    digest = preparer.split_tools._token_sequence_sha256(token_ids)
    return {
        "token_ids": token_ids,
        "prompt_id": f"prompt-{digest}",
        "token_ids_sha256": digest,
    }


def _fresh_selection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[preparer.FreshRecoverySelection, _IdentityTokenizer, list[str]]:
    tokenizer = _IdentityTokenizer()
    split_bindings: dict[str, Any] = {}
    for role, count in preparer.PINNED_V4_SPLIT_COUNTS.items():
        prompts = [
            _token_prompt(1_000_000 + role.__len__() * 100_000 + index * 32)
            for index in range(count)
        ]
        payload = preparer.split_tools._split_payload(
            role,
            prompts,
            original_hashes={
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in ("fit", "candidate", "sealed")
            },
            corpus_sha256="a" * 64,
            corpus_line_count=800,
            tokenizer_sha256="b" * 64,
            components=[],
            selection_schema=(
                preparer.PINNED_V4_CANDIDATE_SELECTION_SCHEMA
                if role == "candidate"
                else preparer.split_tools.CONTINUATION_SELECTION_SCHEMA
            ),
        )
        path = tmp_path / f"{role}.json"
        _write_json(path, payload)
        split_bindings[role] = SimpleNamespace(
            path=path,
            sha256=preparer._file_sha256(path),
            expected_count=count,
            payload=payload,
        )
    prereg = tmp_path / "preregistration.json"
    _write_json(
        prereg,
        {
            "schema": preparer.PINNED_V4_PREREGISTRATION_SCHEMA,
            "sealed_before_model_access": True,
        },
    )
    prereg_sha = preparer._file_sha256(prereg)
    contract = SimpleNamespace(
        path=prereg,
        file_sha256=prereg_sha,
        splits=split_bindings,
        evaluation_contract_sha256="c" * 64,
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus_rows = [
        f"{index:08x} unique recovery prompt with enough distinct content {index}."
        for index in range(800)
    ]
    corpus.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in corpus_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparer, "PINNED_RECOVERY_CORPUS_SHA256", preparer._file_sha256(corpus)
    )
    monkeypatch.setattr(preparer, "PINNED_RECOVERY_CORPUS_LINE_COUNT", 800)
    selection = preparer._authenticate_and_select_fresh_recovery_candidate(
        v4_preregistration=prereg,
        v4_preregistration_sha256=prereg_sha,
        corpus_path=corpus,
        calibration_texts=corpus_rows[:640],
        tokenizer=tokenizer,
        preregistration_authenticator=lambda _path, _sha: contract,
    )
    return selection, tokenizer, corpus_rows[:640]


def _replay_audit_fixture() -> tuple[list[str], list[list[int]], dict[str, Any]]:
    decoded = [f"Fresh held-out candidate prompt {index:02d}." for index in range(24)]
    token_rows = [
        [1_000 + index * 32 + offset for offset in range(16)] for index in range(24)
    ]
    teacher = {
        "payload_sha256": "7" * 64,
        "calibration_texts": ["Calibration text exists only in recovery training."],
        "precomputed_teacher_generated_texts": [
            "Teacher-generated response is unrelated to every held-out prompt."
        ],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [
            {
                "step": 1,
                "rows": [
                    {
                        "prompt_input_ids": [8_000, 8_001],
                        "rollout_input_ids": [8_000, 8_001, 8_002],
                    }
                ],
            }
        ],
        "tokenizer_fingerprints": {"candidate": {"tokenizer_length": 10_000}},
    }
    return decoded, token_rows, teacher


def test_replay_disjointness_audit_rejects_reverse_text_containment() -> None:
    decoded, token_rows, teacher = _replay_audit_fixture()
    teacher["precomputed_teacher_generated_texts"] = ["held-out fragment"]
    decoded[0] = "Fresh held-out fragment remains evaluation-only."

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="contained in fresh decoded candidate prompt",
    ):
        preparer._fresh_candidate_replay_disjointness_audit(
            decoded_prompts=decoded,
            prompt_token_ids=token_rows,
            teacher_payload=teacher,
        )


def test_replay_disjointness_audit_scans_later_stepwise_prompt_windows() -> None:
    decoded, token_rows, teacher = _replay_audit_fixture()
    collision = token_rows[0][4:12]
    prompt_ids = [42, *collision, 43]
    teacher["stepwise_replay_bank"] = [
        {
            "step": 1,
            "rows": [
                {
                    "prompt_input_ids": prompt_ids,
                    "rollout_input_ids": [*prompt_ids, 44],
                }
            ],
        }
    ]

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="stepwise prompt replay",
    ):
        preparer._fresh_candidate_replay_disjointness_audit(
            decoded_prompts=decoded,
            prompt_token_ids=token_rows,
            teacher_payload=teacher,
        )


def test_replay_disjointness_audit_scans_nonzero_rollout_windows() -> None:
    decoded, token_rows, teacher = _replay_audit_fixture()
    prompt_ids = [42, 43]
    teacher["stepwise_replay_bank"] = [
        {
            "step": 1,
            "rows": [
                {
                    "prompt_input_ids": prompt_ids,
                    "rollout_input_ids": [*prompt_ids, *token_rows[0][:8], 44],
                }
            ],
        }
    ]

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="stepwise rollout replay",
    ):
        preparer._fresh_candidate_replay_disjointness_audit(
            decoded_prompts=decoded,
            prompt_token_ids=token_rows,
            teacher_payload=teacher,
        )


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _behavior_suite() -> dict[str, Any]:
    return {
        "split": "candidate",
        "sealed_data_accessed": False,
        "contains_sealed_data": False,
        "publication_authorized": False,
        "suite": {
            "text_evals": [],
            "choice_probes": [],
            "generation_prompts": [
                {
                    "id": "candidate-generation-1",
                    "category": "reasoning",
                    "prompt": "Continue with one concise, relevant sentence:\n",
                    "relevance_rules": [],
                }
            ],
            "behavior_probes": [
                {
                    "id": "candidate-behavior-1",
                    "category": "reasoning",
                    "prompt": "State the number immediately after four.\nAnswer:",
                    "kind": "contains",
                    "targets": ["five", "5"],
                }
            ],
        },
    }


def _fixture_paths(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"fake"}\n', encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"canonical-parent-weights")
    (source / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
    candidate = tmp_path / "candidate-tokenizer"
    candidate.mkdir()
    (candidate / "tokenizer.json").write_text(
        '{"version":"candidate"}\n', encoding="utf-8"
    )
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "First calibration row stays first and is deliberately long enough.\n"
        "Second calibration row stays second and is deliberately long enough.\n",
        encoding="utf-8",
    )
    behavior = tmp_path / "candidate-behavior.json"
    _write_json(behavior, _behavior_suite())
    return {
        "output_dir": tmp_path / "frozen-inputs",
        "source_model": source,
        "source_tokenizer": source,
        "candidate_tokenizer": candidate,
        "calibration_file": calibration,
        "behavior_eval_suite": behavior,
        "expected_parent_artifact_tree_sha256": preparer.stage._artifact_tree(source)[
            "tree_sha256"
        ],
        "teacher_device": "cuda:1",
        "recovery_device": "cuda:6",
    }


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def _fake_components(captured: dict[str, Any]):
    def loader(**kwargs: Any) -> preparer.TeacherComponents:
        captured["loader"] = dict(kwargs)
        cuda = _FakeCuda()
        captured["cuda"] = cuda
        return preparer.TeacherComponents(
            torch=SimpleNamespace(cuda=cuda),
            source_model=object(),
            source_tokenizer=object(),
            candidate_tokenizer=object(),
        )

    return loader


def _fake_teacher_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    calibration = list(kwargs["calibration_texts"])
    parameters = {
        name: kwargs[name]
        for name in (
            "steps",
            "batch_size",
            "max_length",
            "response_loss_mode",
            "calibration_order",
            "stepwise_donor_max_new_tokens",
            "on_policy_max_new_tokens",
            "teacher_generate_max_new_tokens",
            "on_policy_interval",
            "device",
        )
    }
    return {
        "schema": hf_teacher_token_replay.TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": calibration,
        "parameters": parameters,
        "source_model_fingerprint": {
            "model_state_fingerprint": {"state_dict_sha256": STATE_SHA}
        },
        "provenance_sha256": {
            "calibration_texts": hf_recovery._ordered_recovery_texts_sha256(calibration)
        },
        "payload_sha256": "2" * 64,
    }


def _fake_teacher_builder(captured: dict[str, Any]):
    def builder(**kwargs: Any) -> dict[str, Any]:
        captured["builder"] = dict(kwargs)
        return _fake_teacher_payload(kwargs)

    return builder


@pytest.fixture
def fake_teacher_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(payload: Any) -> dict[str, Any]:
        assert isinstance(payload, dict)
        assert payload["schema"] == hf_teacher_token_replay.TEACHER_TOKEN_REPLAY_SCHEMA
        assert payload["status"] == "frozen"
        assert payload["payload_sha256"] == "2" * 64
        return json.loads(json.dumps(payload))

    monkeypatch.setattr(
        preparer.hf_teacher_token_replay,
        "verify_teacher_token_replay_payload",
        verify,
    )


def test_preparer_freezes_exact_order_recipe_lineage_and_atomic_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, Any] = {}
    original_loader = preparer.hf_recovery.load_calibration_texts

    def calibration_loader(*args: Any, **kwargs: Any) -> list[str]:
        captured["calibration_args"] = (args, kwargs)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        preparer.hf_recovery,
        "load_calibration_texts",
        calibration_loader,
    )
    report = preparer.prepare_recovery_inputs(
        **paths,
        component_loader=_fake_components(captured),
        teacher_replay_builder=_fake_teacher_builder(captured),
    )

    output = Path(paths["output_dir"])
    assert sorted(path.name for path in output.iterdir()) == sorted(
        (
            preparer.REPLAY_FILENAME,
            preparer.PLAN_FILENAME,
            preparer.BUILD_REPORT_FILENAME,
        )
    )
    assert not list(tmp_path.glob(".frozen-inputs.*.staging"))
    replay = _strict_json(output / preparer.REPLAY_FILENAME)
    plan = _strict_json(output / preparer.PLAN_FILENAME)
    disk_report = _strict_json(output / preparer.BUILD_REPORT_FILENAME)
    preparer.verify_frozen_replay_wrapper(replay)
    preparer.verify_recovery_plan(
        plan,
        teacher_payload=replay["teacher_token_replay"],
    )
    preparer.verify_build_report(disk_report)
    assert report == disk_report
    assert report["teacher_generation_device"] == "cuda:1"
    assert report["recovery_device"] == "cuda:6"

    calibration = captured["builder"]["calibration_texts"]
    assert calibration == [
        "First calibration row stays first and is deliberately long enough.",
        "Second calibration row stays second and is deliberately long enough.",
    ]
    assert captured["builder"]["calibration_order"] == "original"
    assert captured["builder"]["steps"] == 180
    assert captured["builder"]["teacher_generate_max_new_tokens"] == 48
    assert captured["builder"]["stepwise_donor_max_new_tokens"] == 16
    assert captured["builder"]["on_policy_max_new_tokens"] == 24
    assert captured["builder"]["on_policy_interval"] == 8
    assert captured["builder"]["device"] == "cuda:1"
    assert captured["calibration_args"][1] == {
        "max_texts": 640,
        "min_chars": 32,
        "max_chars": 4096,
        "preserve_whitespace": True,
    }
    assert captured["loader"]["local_files_only"] is True
    assert captured["loader"]["trust_remote_code"] is False
    assert captured["loader"]["source_torch_dtype"] == "float16"
    assert captured["loader"]["device"] == "cuda:1"
    assert captured["cuda"].empty_cache_calls == 1

    kwargs = plan["recovery_kwargs"]
    assert kwargs == preparer.strongest_recovery_kwargs("cuda:6")
    assert kwargs["device"] == "cuda:6"
    assert kwargs["lr"] == pytest.approx(5e-7)
    assert kwargs["train_scope"] == "all"
    assert kwargs["torch_dtype"] == "float16"
    assert kwargs["candidate_torch_dtype"] == "float32"
    assert kwargs["anchor_kd_weight"] == pytest.approx(0.04)
    assert kwargs["dual_space_kd_weight"] == pytest.approx(0.08)
    assert kwargs["attention_influence_weight"] == pytest.approx(0.02)
    assert kwargs["attention_influence_layer"] == "middle"
    assert kwargs["span_hidden_kd_weight"] == pytest.approx(0.10)
    assert kwargs["token_focus"] == "numbers"
    assert kwargs["answer_first_token_margin"] == pytest.approx(1.5)
    assert kwargs["teacher_rank_temperature"] == pytest.approx(0.7)
    assert kwargs["stepwise_donor_kd_weight"] == pytest.approx(0.025)
    assert kwargs["on_policy_unlikelihood_weight"] == pytest.approx(0.002)
    assert kwargs["adamw_foreach"] is False
    assert kwargs["save_best_loss"] is False
    assert kwargs["save_best_behavior"] is False
    assert plan["checkpoint_steps"] == list(preparer.CHECKPOINT_STEPS)
    assert plan["retained_checkpoint_steps"] == []

    source = replay["teacher_source_artifact"]
    assert (
        source["expected_parent_artifact_tree_sha256"]
        == paths["expected_parent_artifact_tree_sha256"]
    )
    assert (
        source["actual_parent_artifact_tree_sha256"]
        == paths["expected_parent_artifact_tree_sha256"]
    )
    assert source["tree_match_verified_before_model_load"] is True
    assert source["source_model_state_dict_sha256"] == STATE_SHA
    assert replay["behavior_eval_suite"]["split"] == "candidate"
    assert replay["contains_sealed_data"] is False
    assert replay["publication_authorized"] is False
    assert "calibration_texts" not in replay
    assert disk_report["source_torch_dtype"] == "float16"
    assert disk_report["candidate_recovery_torch_dtype"] == "float32"
    assert disk_report["training_launched"] is False
    assert disk_report["optimizer_created"] is False
    assert disk_report["accepted_artifact"] is False
    assert disk_report["sealed_data_accessed"] is False
    assert (
        disk_report["teacher_source_artifact"]["source_model_state_dict_sha256"]
        == STATE_SHA
    )


def test_expected_parent_tree_mismatch_stops_before_model_load_or_publication(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    paths["expected_parent_artifact_tree_sha256"] = "0" * 64
    called = False

    def forbidden_loader(**_kwargs: Any) -> preparer.TeacherComponents:
        nonlocal called
        called = True
        raise AssertionError("model loader must not run")

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="canonical parent lineage",
    ):
        preparer.prepare_recovery_inputs(
            **paths,
            component_loader=forbidden_loader,
            teacher_replay_builder=lambda **_kwargs: {},
        )
    assert called is False
    assert not Path(paths["output_dir"]).exists()
    assert not list(tmp_path.glob(".frozen-inputs.*.staging"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("split", "sealed"),
        ("sealed_data_accessed", True),
        ("contains_sealed_data", True),
        ("publication_authorized", True),
    ),
)
def test_sealed_or_non_candidate_suite_is_rejected_before_model_load(
    tmp_path: Path,
    fake_teacher_verifier: None,
    field: str,
    value: Any,
) -> None:
    paths = _fixture_paths(tmp_path)
    behavior = _behavior_suite()
    behavior[field] = value
    _write_json(Path(paths["behavior_eval_suite"]), behavior)

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="candidate-only, sealed-free",
    ):
        preparer.prepare_recovery_inputs(
            **paths,
            component_loader=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("model loader must not run")
            ),
            teacher_replay_builder=lambda **_kwargs: {},
        )
    assert not Path(paths["output_dir"]).exists()


def test_teacher_generation_failure_leaves_no_output_or_staging(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, Any] = {}

    def fail_generation(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("deliberate teacher generation failure")

    with pytest.raises(RuntimeError, match="deliberate teacher generation failure"):
        preparer.prepare_recovery_inputs(
            **paths,
            component_loader=_fake_components(captured),
            teacher_replay_builder=fail_generation,
        )
    assert captured["cuda"].empty_cache_calls == 1
    assert not Path(paths["output_dir"]).exists()
    assert not list(tmp_path.glob(".frozen-inputs.*.staging"))


def test_write_failure_cleans_owned_staging_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, Any] = {}
    original_write = preparer._write_json_exclusive

    def fail_on_plan(path: Path, payload: Any) -> None:
        if path.name == preparer.PLAN_FILENAME:
            raise OSError("deliberate write failure")
        original_write(path, payload)

    monkeypatch.setattr(preparer, "_write_json_exclusive", fail_on_plan)
    with pytest.raises(OSError, match="deliberate write failure"):
        preparer.prepare_recovery_inputs(
            **paths,
            component_loader=_fake_components(captured),
            teacher_replay_builder=_fake_teacher_builder(captured),
        )
    assert not Path(paths["output_dir"]).exists()
    assert not list(tmp_path.glob(".frozen-inputs.*.staging"))


def test_post_rename_verification_failure_removes_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, Any] = {}
    original_verify = preparer._verify_published_tree
    calls = 0

    def fail_second_verify(path: Path) -> None:
        nonlocal calls
        calls += 1
        original_verify(path)
        if calls == 2:
            raise RuntimeError("deliberate post-rename failure")

    monkeypatch.setattr(preparer, "_verify_published_tree", fail_second_verify)
    with pytest.raises(RuntimeError, match="post-rename failure"):
        preparer.prepare_recovery_inputs(
            **paths,
            component_loader=_fake_components(captured),
            teacher_replay_builder=_fake_teacher_builder(captured),
        )
    assert calls == 2
    assert not Path(paths["output_dir"]).exists()
    assert not list(tmp_path.glob(".frozen-inputs.*.staging"))


def test_resealed_plan_cannot_change_replay_schedule_or_recipe(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, Any] = {}
    preparer.prepare_recovery_inputs(
        **paths,
        component_loader=_fake_components(captured),
        teacher_replay_builder=_fake_teacher_builder(captured),
    )
    output = Path(paths["output_dir"])
    replay = _strict_json(output / preparer.REPLAY_FILENAME)
    plan = _strict_json(output / preparer.PLAN_FILENAME)
    plan["recovery_kwargs"]["on_policy_interval"] = 9
    body = dict(plan)
    body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = preparer.canonical_sha256(body)
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="pinned strongest known recipe",
    ):
        preparer.verify_recovery_plan(
            plan,
            teacher_payload=replay["teacher_token_replay"],
        )


def test_default_loader_pins_local_only_trust_false_and_fp16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_paths(tmp_path)
    captured: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "tokenizers": [],
        "models": [],
    }

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> object:
            captured["tokenizers"].append((path, dict(kwargs)))
            return object()

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.eval_calls = 0

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.eval_calls += 1

    model = FakeModel()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> FakeModel:
            captured["models"].append((path, dict(kwargs)))
            return model

    fake_torch = SimpleNamespace(float16=object())
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizerLoader,
        AutoModelForCausalLM=FakeModelLoader,
    )
    monkeypatch.setattr(
        preparer.hf_recovery,
        "_torch_and_transformers",
        lambda: (fake_torch, fake_transformers),
    )
    components = preparer.load_teacher_components(
        source_model_path=Path(paths["source_model"]),
        source_tokenizer_path=Path(paths["source_tokenizer"]),
        candidate_tokenizer_path=Path(paths["candidate_tokenizer"]),
        device="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
        source_torch_dtype="float16",
    )
    assert components.source_model is model
    assert len(captured["tokenizers"]) == 2
    assert all(
        kwargs == {"local_files_only": True, "trust_remote_code": False}
        for _path, kwargs in captured["tokenizers"]
    )
    assert captured["models"] == [
        (
            str(paths["source_model"]),
            {
                "torch_dtype": fake_torch.float16,
                "device_map": None,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert model.device == "cuda:0"
    assert model.eval_calls == 1
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="local_files_only=True",
    ):
        preparer.load_teacher_components(
            source_model_path=Path(paths["source_model"]),
            source_tokenizer_path=Path(paths["source_tokenizer"]),
            candidate_tokenizer_path=Path(paths["candidate_tokenizer"]),
            device="cuda:0",
            local_files_only=False,
            trust_remote_code=False,
            source_torch_dtype="float16",
        )


def test_fresh_selection_excludes_v4_and_exact_640_recovery_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _tokenizer, calibration = _fresh_selection_fixture(tmp_path, monkeypatch)
    assert len(selection.prompts) == 24
    assert len(set(selection.decoded_prompts)) == 24
    assert all(8 <= len(prompt.token_ids) <= 128 for prompt in selection.prompts)
    assert all(
        not any(prompt in row or row in prompt for row in calibration)
        for prompt in selection.decoded_prompts
    )
    provenance = selection.provenance
    assert provenance["causal_ab_eligible"] is True
    assert provenance["model_accessed_during_selection"] is False
    assert provenance["sealed_exclusion_manifest_used_model_free"] is True
    assert provenance["sealed_prompts_supplied_to_model_or_trainer"] is False
    assert provenance["sealed_outputs_or_selection_scores_computed"] is False
    assert provenance["recovery_calibration"]["text_count"] == 640
    assert provenance["selected_candidate"]["minimum_token_count"] >= 8
    assert provenance["selected_candidate"]["maximum_token_count"] <= 128
    assert provenance["disjointness"] == {
        "exact_token_prompt_overlap_count": 0,
        "meaningful_prefix_overlap_count": 0,
        "decoded_candidate_text_in_recovery_calibration_count": 0,
        "minimum_meaningful_prefix_tokens": 8,
        "recovery_calibration_prefixes_bounded_to_candidate_max_tokens": True,
    }


def test_fresh_parent_adapter_pins_all_decoded_prefix_probes_and_rejects_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, tokenizer, _calibration = _fresh_selection_fixture(tmp_path, monkeypatch)

    class Parent:
        def eval(self) -> None:
            return None

        def generate(self, *, input_ids: Any, **kwargs: Any) -> Any:
            assert kwargs["max_new_tokens"] == 16
            assert kwargs["do_sample"] is False
            suffix = torch.tensor(
                [[ord(character) + 10 for character in "a b c d e f g h "]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix), dim=1)

    tokenizer_fingerprint = {"sha256": "d" * 64, "tokenizer_length": 2_000_000}
    teacher = {
        "payload_sha256": "1" * 64,
        "calibration_texts": _calibration,
        "precomputed_teacher_generated_texts": [
            "Teacher replay output remains separate from every evaluation prompt."
        ],
        "on_policy_replay_bank": [
            ["Frozen on-policy recovery text remains training-only."]
        ],
        "stepwise_replay_bank": [
            {
                "step": 1,
                "rows": [
                    {
                        "prompt_input_ids": [1_500_000, 1_500_001],
                        "rollout_input_ids": [
                            1_500_000,
                            1_500_001,
                            1_500_002,
                        ],
                    }
                ],
            }
        ],
        "tokenizer_fingerprints": {
            "source": tokenizer_fingerprint,
            "candidate": tokenizer_fingerprint,
        },
        "source_model_fingerprint": {
            "sha256": "e" * 64,
            "model_state_fingerprint": {"state_dict_sha256": "f" * 64},
        },
    }
    binding = preparer._build_fresh_behavior_binding(
        torch=torch,
        source_model=Parent(),
        tokenizer=tokenizer,
        selection=selection,
        teacher_payload=teacher,
        device="cpu",
    )
    verified = preparer.verify_behavior_suite_binding(binding)
    assert verified["mode"] == preparer.BEHAVIOR_MODE_FRESH
    assert verified["causal_ab_eligible"] is True
    assert verified["behavior_probe_count"] == 48
    assert verified["first_action_probe_count"] == 24
    assert verified["lcp4_probe_count"] == 24
    assert verified["generation_prompt_count"] == 24
    assert all(
        probe["kind"] == "literal_prefix"
        for probe in verified["suite"]["behavior_probes"]
    )
    assert all(
        prompt["kind"] == "literal_prefix"
        and prompt["require_generation_health"] is True
        and prompt["expected_prefix_generated_token_count"] == 4
        for prompt in verified["suite"]["generation_prompts"]
    )
    assert verified["adapter_provenance"]["parent_self_score_pass_count"] == 48
    assert verified["adapter_provenance"]["parent_generation_health_pass_count"] == 24

    tampered = copy.deepcopy(verified)
    tampered["suite"]["behavior_probes"][0]["expected_prefix"] = "Wrong"
    tampered["suite_sha256"] = preparer.canonical_sha256(tampered["suite"])
    tampered["adapter_provenance"]["suite_sha256"] = tampered["suite_sha256"]
    adapter_body = dict(tampered["adapter_provenance"])
    adapter_body.pop("payload_sha256")
    tampered["adapter_provenance"]["payload_sha256"] = preparer.canonical_sha256(
        adapter_body
    )
    binding_body = dict(tampered)
    binding_body.pop("payload_sha256")
    tampered["payload_sha256"] = preparer.canonical_sha256(binding_body)
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="failed authentication",
    ):
        preparer.verify_behavior_suite_binding(tampered)


@pytest.mark.parametrize("suffix", ["a b ", "a b !!!!"])
def test_fresh_parent_adapter_rejects_prefix_without_coherent_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    selection, tokenizer, calibration = _fresh_selection_fixture(tmp_path, monkeypatch)

    class Parent:
        def eval(self) -> None:
            return None

        def generate(self, *, input_ids: Any, **_kwargs: Any) -> Any:
            suffix_ids = torch.tensor(
                [[ord(character) + 10 for character in suffix]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix_ids), dim=1)

    tokenizer_fingerprint = {"sha256": "d" * 64, "tokenizer_length": 2_000_000}
    teacher = {
        "payload_sha256": "1" * 64,
        "calibration_texts": calibration,
        "precomputed_teacher_generated_texts": ["Independent teacher replay text."],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [
            {
                "step": 1,
                "rows": [
                    {
                        "prompt_input_ids": [1_500_000, 1_500_001],
                        "rollout_input_ids": [
                            1_500_000,
                            1_500_001,
                            1_500_002,
                        ],
                    }
                ],
            }
        ],
        "tokenizer_fingerprints": {
            "source": tokenizer_fingerprint,
            "candidate": tokenizer_fingerprint,
        },
        "source_model_fingerprint": {
            "sha256": "e" * 64,
            "model_state_fingerprint": {"state_dict_sha256": "f" * 64},
        },
    }

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="prefix-plus-coherence generation probe",
    ):
        preparer._build_fresh_behavior_binding(
            torch=torch,
            source_model=Parent(),
            tokenizer=tokenizer,
            selection=selection,
            teacher_payload=teacher,
            device="cpu",
        )


def test_literal_prefix_scoring_is_case_sensitive_and_rejects_blank_target() -> None:
    probe = {
        "id": "literal",
        "category": "decoded-prefix",
        "prompt": "Prompt:",
        "kind": "literal_prefix",
        "expected_prefix": "Answer",
    }
    passed = preparer.hf_tools._score_behavior_probe(
        probe, {"output": "Prompt:Answer follows"}
    )
    failed = preparer.hf_tools._score_behavior_probe(
        probe, {"output": "Prompt:answer follows"}
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    with pytest.raises(ValueError, match="nonblank expected_prefix"):
        preparer.hf_tools._score_behavior_probe(
            {**probe, "expected_prefix": "   "},
            {"output": "Prompt:anything"},
        )


def test_fresh_preregistration_failure_occurs_before_parent_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "text": (
                        f"{index:08x} calibration row remains long enough for recovery."
                    )
                }
            )
            + "\n"
            for index in range(640)
        ),
        encoding="utf-8",
    )
    prereg = tmp_path / "bad-preregistration.json"
    _write_json(
        prereg,
        {"schema": "wrong-schema", "sealed_before_model_access": True},
    )
    tokenizer = _IdentityTokenizer()
    monkeypatch.setattr(
        preparer.hf_teacher_token_replay,
        "tokenizer_functional_fingerprint",
        lambda _tokenizer: {"sha256": "1" * 64},
    )
    model_loaded = False

    def forbidden_model_loader(**_kwargs: Any) -> Any:
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("parent model must not load")

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="schema or pre-model seal",
    ):
        preparer.prepare_recovery_inputs(
            output_dir=tmp_path / "outputs",
            source_model=source,
            source_tokenizer=source,
            candidate_tokenizer=source,
            calibration_file=corpus,
            behavior_eval_suite=None,
            behavior_mode=preparer.BEHAVIOR_MODE_FRESH,
            v4_preregistration=prereg,
            v4_preregistration_sha256=preparer._file_sha256(prereg),
            expected_parent_artifact_tree_sha256=preparer.stage._artifact_tree(source)[
                "tree_sha256"
            ],
            teacher_device="cpu",
            recovery_device="cpu",
            component_loader=None,
            tokenizer_loader=lambda **_kwargs: preparer.TokenizerComponents(
                torch=SimpleNamespace(),
                transformers=SimpleNamespace(),
                source_tokenizer=tokenizer,
                candidate_tokenizer=tokenizer,
            ),
            source_model_loader=forbidden_model_loader,
            teacher_replay_builder=lambda **_kwargs: {},
        )
    assert model_loaded is False
