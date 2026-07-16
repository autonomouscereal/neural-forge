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
from neural_forge import (  # noqa: E402
    hf_recovery,
    hf_teacher_replay_cache,
    hf_teacher_token_replay,
)


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


class _UniquePrefixTokenizer(_IdentityTokenizer):
    def __init__(self) -> None:
        self._text_to_ids: dict[str, tuple[int, ...]] = {}
        self._ids_to_text: dict[tuple[int, ...], str] = {}

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
        token_ids = self._text_to_ids.get(text)
        if token_ids is None:
            base = 100_000 + len(self._text_to_ids) * 32
            token_ids = tuple(base + offset for offset in range(16))
            self._text_to_ids[text] = token_ids
            self._ids_to_text[token_ids] = text
        return {"input_ids": list(token_ids)}

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        values = tuple(
            int(value)
            for value in (
                token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            )
        )
        for prompt_ids, text in self._ids_to_text.items():
            if values[: len(prompt_ids)] == prompt_ids:
                suffix = values[len(prompt_ids) :]
                return text + "".join(chr(token_id - 10) for token_id in suffix)
        raise AssertionError("Tokenizer lost the exact prompt-token mapping.")


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
) -> tuple[
    preparer.FreshRecoveryParentCandidatePool,
    preparer.FreshRecoveryCandidateReservoir,
    _IdentityTokenizer,
    list[str],
]:
    tokenizer = _UniquePrefixTokenizer()
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
    corpus_rows[-1] = corpus_rows[-2]
    corpus.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in corpus_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparer, "PINNED_RECOVERY_CORPUS_SHA256", preparer._file_sha256(corpus)
    )
    monkeypatch.setattr(preparer, "PINNED_RECOVERY_CORPUS_LINE_COUNT", 800)
    monkeypatch.setattr(
        preparer,
        "PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT",
        1,
    )
    reservoir = preparer._authenticate_fresh_recovery_candidate_reservoir(
        v4_preregistration=prereg,
        v4_preregistration_sha256=prereg_sha,
        corpus_path=corpus,
        calibration_texts=corpus_rows[:640],
        tokenizer=tokenizer,
        preregistration_authenticator=lambda _path, _sha: contract,
    )
    teacher = {
        "payload_sha256": "7" * 64,
        "calibration_texts": corpus_rows[:640],
        "precomputed_teacher_generated_texts": [
            "Teacher-generated response is unrelated to every held-out prompt."
        ],
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
        "tokenizer_fingerprints": {"candidate": {"tokenizer_length": 2_000_000}},
    }
    selection = preparer._select_fresh_recovery_candidate(
        reservoir=reservoir,
        teacher_payload=teacher,
    )
    return selection, reservoir, tokenizer, corpus_rows[:640]


def _fresh_parent_teacher(calibration: list[str]) -> dict[str, Any]:
    tokenizer_fingerprint = {"sha256": "d" * 64, "tokenizer_length": 2_000_000}
    return {
        "payload_sha256": "1" * 64,
        "calibration_texts": calibration,
        "precomputed_teacher_generated_texts": [
            "Teacher replay output remains separate from every evaluation prompt."
        ],
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


def _balanced_parent_binding(
    *,
    pool: preparer.FreshRecoveryParentCandidatePool,
    tokenizer: _IdentityTokenizer,
    calibration: list[str],
    rejected_pool_orders: set[int] | None = None,
    rejected_categories: set[str] | None = None,
) -> dict[str, Any]:
    rejected_pool_orders = set(rejected_pool_orders or ())
    rejected_categories = set(rejected_categories or ())
    eos_token_id = 1_999_999
    prompt_order = {
        tuple(prompt.token_ids): index for index, prompt in enumerate(pool.prompts)
    }
    category_by_order = {
        int(row["pool_order"]): str(row["category_id"])
        for row in pool.provenance["parent_candidate_pool"]["prompt_rows"]
    }

    class StructuralTokenizer:
        eos_token_id = 1_999_999
        pad_token_id = 0

        @staticmethod
        def decode(token_ids: Any, **kwargs: Any) -> str:
            values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            return tokenizer.decode(
                [value for value in values if int(value) != eos_token_id],
                **kwargs,
            )

    class Parent:
        calls: list[int] = []

        def eval(self) -> None:
            return None

        def generate(self, *, input_ids: Any, **kwargs: Any) -> Any:
            assert kwargs["max_new_tokens"] == 16
            order = prompt_order[
                tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
            ]
            self.calls.append(order)
            if (
                order in rejected_pool_orders
                or category_by_order[order] in rejected_categories
            ):
                suffix = [eos_token_id]
            else:
                suffix = [ord(character) + 10 for character in "a b c d e f g h "]
            suffix_tensor = torch.tensor(
                [suffix],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix_tensor), dim=1)

    return preparer._build_fresh_behavior_binding(
        torch=torch,
        source_model=Parent(),
        tokenizer=StructuralTokenizer(),
        selection=pool,
        teacher_payload=_fresh_parent_teacher(calibration),
        device="cpu",
    )


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

    @staticmethod
    def is_available() -> bool:
        return False


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
    assert disk_report["teacher_generation_completed"] is True
    assert disk_report["teacher_replay_acquisition"] == {
        "schema": preparer.TEACHER_REPLAY_ACQUISITION_SCHEMA,
        "cache_mode": "off",
        "origin": "generated",
        "cache_result": "disabled",
        "cache_request_sha256": None,
        "cache_envelope_sha256": None,
        "cache_entry_relative_path": None,
        "cache_verified_before_final_selection": False,
        "cache_import": None,
    }
    assert disk_report["parent_model_usage"] == {
        "schema": preparer.PARENT_MODEL_USAGE_SCHEMA,
        "model_load_count": 1,
        "load_purpose": "teacher-replay",
        "teacher_replay_generated_this_run": True,
        "teacher_replay_reused_from_cache": False,
        "fresh_target_generation_performed": False,
    }
    assert (
        disk_report["teacher_source_artifact"]["source_model_state_dict_sha256"]
        == STATE_SHA
    )

    falsely_resealed = copy.deepcopy(disk_report)
    falsely_resealed["teacher_replay_acquisition"]["cache_result"] = "hit"
    falsely_resealed_body = dict(falsely_resealed)
    falsely_resealed_body.pop("build_report_sha256")
    falsely_resealed["build_report_sha256"] = preparer.canonical_sha256(
        falsely_resealed_body
    )
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="replay acquisition or parent usage is inconsistent",
    ):
        preparer.verify_build_report(falsely_resealed)

    cached_report = copy.deepcopy(disk_report)
    cached_report["behavior_mode"] = preparer.BEHAVIOR_MODE_FRESH
    cached_report["causal_ab_eligible"] = True
    cached_report["fresh_candidate_reservoir"] = dict(
        preparer._load_external_fresh_reservoir_seed().receipt
    )
    cached_report["teacher_generation_completed"] = False
    cached_report["teacher_replay_acquisition"] = {
        "schema": preparer.TEACHER_REPLAY_ACQUISITION_SCHEMA,
        "cache_mode": "auto",
        "origin": "cache-hit",
        "cache_result": "hit",
        "cache_request_sha256": "c" * 64,
        "cache_envelope_sha256": "d" * 64,
        "cache_entry_relative_path": (
            f"{'c' * 64}/{hf_teacher_replay_cache.CACHE_FILENAME}"
        ),
        "cache_verified_before_final_selection": True,
        "cache_import": None,
    }
    cached_report["parent_model_usage"] = {
        "schema": preparer.PARENT_MODEL_USAGE_SCHEMA,
        "model_load_count": 1,
        "load_purpose": "fresh-targets",
        "teacher_replay_generated_this_run": False,
        "teacher_replay_reused_from_cache": True,
        "fresh_target_generation_performed": True,
    }
    cached_report_body = dict(cached_report)
    cached_report_body.pop("build_report_sha256")
    cached_report["build_report_sha256"] = preparer.canonical_sha256(cached_report_body)
    assert preparer.verify_build_report(cached_report) == cached_report


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


def test_fresh_parent_pool_excludes_v4_and_exact_640_recovery_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _reservoir, _tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    assert len(selection.prompts) == preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT
    assert len(set(selection.decoded_prompts)) == (
        preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT
    )
    assert all(8 <= len(prompt.token_ids) <= 128 for prompt in selection.prompts)
    assert all(
        not any(prompt in row or row in prompt for row in calibration)
        for prompt in selection.decoded_prompts
    )
    provenance = selection.provenance
    assert provenance["causal_ab_eligible"] is False
    assert provenance["candidate_reservoir_frozen_before_parent_load"] is True
    assert (
        provenance["parent_candidate_pool_derived_without_parent_output_access"] is True
    )
    assert provenance["teacher_replay_used_for_exclusion_only"] is True
    assert provenance["parent_forward_or_score_during_pool_derivation"] is False
    assert provenance["tokenizer_called_during_pool_derivation"] is False
    assert provenance["child_or_arm_outputs_accessed_during_pool_derivation"] is False
    assert provenance["sealed_exclusion_manifest_used_model_free"] is True
    assert provenance["sealed_prompts_supplied_to_model_or_trainer"] is False
    assert provenance["sealed_outputs_or_selection_scores_computed"] is False
    assert provenance["recovery_calibration"]["text_count"] == 640
    assert provenance["parent_candidate_pool"]["minimum_token_count"] >= 8
    assert provenance["parent_candidate_pool"]["maximum_token_count"] <= 128
    reservoir = provenance["candidate_reservoir"]
    assert len(reservoir["rows"]) > 24
    assert reservoir["rows_sha256"] == preparer.canonical_sha256(reservoir["rows"])
    derivation = provenance["replay_exclusion_derivation"]
    assert (
        derivation["selected_reservoir_indices"]
        == [
            row["reservoir_index"]
            for row in derivation["classification_rows"]
            if row["classification"] == "replay_disjoint_survivor"
        ][: preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT]
    )
    assert derivation["survivor_margin"] > 0
    assert derivation["full_reservoir_classified"] is True
    assert (
        reservoir["provenance"]["preselection_statistics"]["duplicate_source_row_count"]
        == 0
    )
    pool = provenance["parent_candidate_pool"]
    assert pool["all_external_categories_meet_final_quota"] is True
    assert set(pool["category_histogram"]) == {
        "science",
        "reasoning",
        "code",
        "mathematics",
        "systems",
        "language",
        "practical",
        "data",
    }
    assert all(
        value >= preparer.FRESH_CATEGORY_QUOTA
        for value in pool["category_histogram"].values()
    )
    assert provenance["disjointness"] == {
        "exact_token_prompt_overlap_count": 0,
        "meaningful_prefix_overlap_count": 0,
        "decoded_candidate_text_in_recovery_calibration_count": 0,
        "minimum_meaningful_prefix_tokens": 8,
        "recovery_calibration_prefixes_bounded_to_candidate_max_tokens": True,
    }


def test_external_reservoir_seed_is_static_round_robin_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = preparer._load_external_fresh_reservoir_seed()
    assert len(seed.prompts) == 384
    assert seed.receipt == {
        **seed.receipt,
        "source_file_sha256": preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_FILE_SHA256,
        "source_file_byte_count": preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_BYTE_COUNT,
        "source_payload_sha256": (
            preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_PAYLOAD_SHA256
        ),
        "expanded_prompt_manifest_sha256": (
            preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_MANIFEST_SHA256
        ),
        "expanded_prompt_text_manifest_sha256": (
            preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_TEXT_MANIFEST_SHA256
        ),
        "expanded_prompt_count": 384,
        "first_24_cover_all_categories": True,
        "contains_cached_prompt_or_target_content": False,
        "contains_parent_or_child_outputs": False,
    }
    categories = [row["category_id"] for row in seed.prompts[:24]]
    assert len(set(categories)) == 8
    assert all(categories.count(category) == 3 for category in set(categories))
    assert len({row["source_candidate_id"] for row in seed.prompts}) == 384
    assert len({row["text"] for row in seed.prompts}) == 384

    tampered = tmp_path / "tampered-reservoir.json"
    payload = json.loads(
        preparer.PINNED_EXTERNAL_FRESH_RESERVOIR_PATH.read_text(encoding="utf-8")
    )
    payload["categories"][0]["topics"][0]["label"] += " changed"
    _write_json(tampered, payload)
    monkeypatch.setattr(preparer, "PINNED_EXTERNAL_FRESH_RESERVOIR_PATH", tampered)
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="seed bytes changed",
    ):
        preparer._load_external_fresh_reservoir_seed()


def test_external_reservoir_survives_massive_original_corpus_replay_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _selection, reservoir, _tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    teacher = {
        "payload_sha256": "9" * 64,
        "calibration_texts": calibration,
        "precomputed_teacher_generated_texts": [
            f"{index:08x} unique recovery prompt with enough distinct content {index}."
            for index in range(640, 800)
        ],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [],
        "tokenizer_fingerprints": {"candidate": {"tokenizer_length": 2_000_000}},
    }
    selection = preparer._select_fresh_recovery_candidate(
        reservoir=reservoir,
        teacher_payload=teacher,
    )
    assert len(selection.prompts) == preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT
    assert (
        selection.provenance["replay_exclusion_derivation"]["survivor_count"]
        >= preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT
    )
    assert (
        selection.provenance["parent_candidate_pool"][
            "all_external_categories_meet_final_quota"
        ]
        is True
    )


def test_selected_survivors_fail_closed_if_one_external_domain_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _selection, reservoir, _tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    seed = preparer._load_external_fresh_reservoir_seed()
    category_by_line = {
        int(row["source_line_number"]): row["category_id"] for row in seed.prompts
    }
    blocked_rows = [
        {
            "prompt_input_ids": list(prompt.token_ids),
            "rollout_input_ids": [*prompt.token_ids, 1_999_999],
        }
        for prompt in reservoir.prompts
        if category_by_line[int(prompt.payload["source_line_number"])] == "science"
    ]
    teacher = {
        "payload_sha256": "6" * 64,
        "calibration_texts": calibration,
        "precomputed_teacher_generated_texts": ["Independent recovery output."],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [
            {"step": index + 1, "rows": [row]} for index, row in enumerate(blocked_rows)
        ],
        "tokenizer_fingerprints": {"candidate": {"tokenizer_length": 2_000_000}},
    }
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="does not contain at least",
    ):
        preparer._select_fresh_recovery_candidate(
            reservoir=reservoir,
            teacher_payload=teacher,
        )


@pytest.mark.parametrize(
    ("collision_location", "expected_reason"),
    (
        ("prompt", "stepwise_prompt_window_collision"),
        ("rollout", "stepwise_rollout_window_collision"),
    ),
)
def test_final_selection_skips_replay_window_collisions_before_freezing_24(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_location: str,
    expected_reason: str,
) -> None:
    _baseline, reservoir, _tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    collision = list(reservoir.prompts[0].token_ids[3:11])
    prompt_ids = [42, 43]
    rollout_ids = [42, 43, 44]
    if collision_location == "prompt":
        prompt_ids = [42, *collision, 43]
        rollout_ids = [*prompt_ids, 44]
    else:
        rollout_ids = [42, 43, *collision, 44]
    teacher = {
        "payload_sha256": "8" * 64,
        "calibration_texts": calibration,
        "precomputed_teacher_generated_texts": [
            "Independent teacher-generated recovery response."
        ],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [
            {
                "step": 8,
                "rows": [
                    {
                        "prompt_input_ids": prompt_ids,
                        "rollout_input_ids": rollout_ids,
                    }
                ],
            }
        ],
        "tokenizer_fingerprints": {"candidate": {"tokenizer_length": 2_000_000}},
    }

    first = preparer._select_fresh_recovery_candidate(
        reservoir=reservoir,
        teacher_payload=teacher,
    )
    second = preparer._select_fresh_recovery_candidate(
        reservoir=reservoir,
        teacher_payload=copy.deepcopy(teacher),
    )

    derivation = first.provenance["replay_exclusion_derivation"]
    assert derivation["classification_rows"][0] == {
        "reservoir_index": 0,
        "classification": expected_reason,
    }
    expected_indices = [
        row["reservoir_index"]
        for row in derivation["classification_rows"]
        if row["classification"] == "replay_disjoint_survivor"
    ][: preparer.FRESH_PARENT_CANDIDATE_POOL_COUNT]
    assert derivation["selected_reservoir_indices"] == expected_indices
    assert [prompt.prompt_id for prompt in first.prompts] == [
        reservoir.prompts[index].prompt_id for index in expected_indices
    ]
    assert first.provenance == second.provenance
    assert first.prompts == second.prompts


def test_rehashed_reservoir_reorder_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _reservoir, _tokenizer, _calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    payload = copy.deepcopy(selection.provenance["candidate_reservoir"])
    rows = payload["rows"]
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["reservoir_index"] = 0
    rows[1]["reservoir_index"] = 1
    payload["rows_sha256"] = preparer.canonical_sha256(rows)
    manifest = [
        {
            key: value
            for key, value in row.items()
            if key not in {"token_ids", "decoded_prompt"}
        }
        for row in rows
    ]
    provenance = payload["provenance"]
    provenance["reservoir_manifest_sha256"] = preparer.canonical_sha256(manifest)
    provenance_body = dict(provenance)
    provenance_body.pop("candidate_reservoir_payload_sha256")
    provenance["candidate_reservoir_payload_sha256"] = preparer.canonical_sha256(
        provenance_body
    )

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="strict external source order/content",
    ):
        preparer._verify_fresh_candidate_reservoir_payload(payload)


def test_rehashed_reservoir_incomplete_preselection_counts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _reservoir, _tokenizer, _calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    payload = copy.deepcopy(selection.provenance["candidate_reservoir"])
    provenance = payload["provenance"]
    provenance["preselection_statistics"]["accepted_into_reservoir"] += 1
    provenance_body = dict(provenance)
    provenance_body.pop("candidate_reservoir_payload_sha256")
    provenance["candidate_reservoir_payload_sha256"] = preparer.canonical_sha256(
        provenance_body
    )

    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="provenance failed authentication",
    ):
        preparer._verify_fresh_candidate_reservoir_payload(payload)


def test_fresh_parent_adapter_pins_all_decoded_prefix_probes_and_rejects_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _reservoir, tokenizer, _calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )

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


def test_parent_structural_selection_replaces_immediate_eos_with_later_same_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    binding = _balanced_parent_binding(
        pool=pool,
        tokenizer=tokenizer,
        calibration=calibration,
        rejected_pool_orders={11},
    )
    verified = preparer.verify_behavior_suite_binding(binding)
    adapter = verified["adapter_provenance"]
    selection = adapter["selection"]
    structural = selection["parent_structural_selection"]
    assert structural["scanned_count"] == 28
    assert structural["completion_pool_order"] == 27
    assert structural["accepted_pool_orders"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        27,
    ]
    rejected = structural["classification_rows"][11]
    replacement = structural["classification_rows"][27]
    assert rejected["category_id"] == replacement["category_id"] == "mathematics"
    assert rejected["classification"] == "rejected_parent_minimum_pre_eos"
    assert rejected["pre_eos_token_count"] == 0
    assert rejected["generated_continuation_token_ids_sha256"]
    assert rejected["generated_continuation_decoded_text_sha256"]
    assert "generated_continuation_token_ids" not in rejected
    assert "generated_continuation_decoded_text" not in rejected
    assert replacement["classification"] == "accepted_balanced_category_quota"
    assert all(
        count == preparer.FRESH_CATEGORY_QUOTA
        for count in structural["accepted_category_histogram"].values()
    )
    assert selection["selected_candidate"]["category_histogram"] == {
        category: preparer.FRESH_CATEGORY_QUOTA
        for category in structural["authenticated_category_ids"]
    }
    assert adapter["behavior_probe_count"] == 48
    assert adapter["generation_prompt_count"] == 24
    assert structural["selection_is_child_blind"] is True
    assert structural["selection_is_not_parent_capability_ranked"] is True
    assert structural["parent_health_used_as_fixed_binary_gate_only"] is True
    assert structural["eligible_rows_sorted_or_ranked_after_gate"] is False
    assert structural["child_or_fold_outputs_accessed"] is False


def test_parent_structural_selection_is_deterministic_first_quota_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    first = _balanced_parent_binding(
        pool=pool, tokenizer=tokenizer, calibration=calibration
    )
    second = _balanced_parent_binding(
        pool=pool, tokenizer=tokenizer, calibration=calibration
    )
    first_structural = first["adapter_provenance"]["selection"][
        "parent_structural_selection"
    ]
    second_structural = second["adapter_provenance"]["selection"][
        "parent_structural_selection"
    ]
    assert first_structural == second_structural
    assert first_structural["scanned_count"] == 24
    assert first_structural["accepted_pool_orders"] == list(range(24))
    assert first_structural["completion_pool_order"] == 23
    assert first_structural["unscanned_candidate_pool_count"] == 40
    assert first_structural["stopped_exactly_at_first_completed_balanced_quota"] is True


def test_parent_structural_selection_fails_if_one_domain_has_fewer_than_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="could not complete the balanced category quota",
    ):
        _balanced_parent_binding(
            pool=pool,
            tokenizer=tokenizer,
            calibration=calibration,
            rejected_categories={"science"},
        )


@pytest.mark.parametrize("tamper_kind", ["order", "eligibility", "category"])
def test_parent_structural_selection_rejects_resealed_classification_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    pool, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    tampered = copy.deepcopy(
        _balanced_parent_binding(
            pool=pool, tokenizer=tokenizer, calibration=calibration
        )
    )
    adapter = tampered["adapter_provenance"]
    selection = adapter["selection"]
    structural = selection["parent_structural_selection"]
    row = structural["classification_rows"][0]
    if tamper_kind == "order":
        row["pool_order"] = 1
    elif tamper_kind == "eligibility":
        row["parent_structurally_eligible"] = False
    else:
        row["category_id"] = "code"
    structural["classification_rows_sha256"] = preparer.canonical_sha256(
        structural["classification_rows"]
    )
    structural_body = dict(structural)
    structural_body.pop("payload_sha256")
    structural["payload_sha256"] = preparer.canonical_sha256(structural_body)
    selection_body = dict(selection)
    selection_body.pop("payload_sha256")
    selection["payload_sha256"] = preparer.canonical_sha256(selection_body)
    adapter["parent_structural_selection_payload_sha256"] = structural["payload_sha256"]
    adapter_body = dict(adapter)
    adapter_body.pop("payload_sha256")
    adapter["payload_sha256"] = preparer.canonical_sha256(adapter_body)
    binding_body = dict(tampered)
    binding_body.pop("payload_sha256")
    tampered["payload_sha256"] = preparer.canonical_sha256(binding_body)
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="failed authentication",
    ):
        preparer.verify_behavior_suite_binding(tampered)


def test_parent_structural_selection_rejects_resealed_missing_health_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )
    tampered = copy.deepcopy(
        _balanced_parent_binding(
            pool=pool,
            tokenizer=tokenizer,
            calibration=calibration,
            rejected_pool_orders={11},
        )
    )
    adapter = tampered["adapter_provenance"]
    selection = adapter["selection"]
    structural = selection["parent_structural_selection"]
    row = structural["classification_rows"][11]
    assert row["classification"] == "rejected_parent_minimum_pre_eos"
    row.update(
        {
            "generated_continuation_token_count": 4,
            "generated_continuation_token_ids_sha256": preparer.canonical_sha256(
                [101, 102, 103, 104]
            ),
            "pre_eos_token_count": 4,
            "pre_eos_token_ids_sha256": preparer.canonical_sha256([101, 102, 103, 104]),
            "minimum_pre_eos_pass": True,
            "decoded_prefix_contract_pass": True,
            "decoded_prefix_self_score_pass": True,
            "generation_health_pass": False,
            "generation_health_receipt": None,
            "generation_health_receipt_sha256": None,
            "parent_structurally_eligible": False,
            "accepted_for_balanced_quota": False,
            "classification": "rejected_parent_generation_health",
        }
    )
    structural["classification_rows_sha256"] = preparer.canonical_sha256(
        structural["classification_rows"]
    )
    structural_body = dict(structural)
    structural_body.pop("payload_sha256")
    structural["payload_sha256"] = preparer.canonical_sha256(structural_body)
    selection_body = dict(selection)
    selection_body.pop("payload_sha256")
    selection["payload_sha256"] = preparer.canonical_sha256(selection_body)
    adapter["parent_structural_selection_payload_sha256"] = structural["payload_sha256"]
    adapter_body = dict(adapter)
    adapter_body.pop("payload_sha256")
    adapter["payload_sha256"] = preparer.canonical_sha256(adapter_body)
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
    selection, _reservoir, tokenizer, calibration = _fresh_selection_fixture(
        tmp_path, monkeypatch
    )

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
        match="could not complete the balanced category quota",
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


def test_teacher_replay_cache_modes_freeze_before_read_reuse_and_survive_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"canonical parent weights")
    calibration = tmp_path / "calibration.jsonl"
    calibration_texts = [
        f"{index:08x} cache orchestration calibration row remains unique and long enough."
        for index in range(640)
    ]
    calibration.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in calibration_texts),
        encoding="utf-8",
    )
    preregistration = tmp_path / "preregistration.json"
    _write_json(preregistration, {"sealed_before_model_access": True})

    tokenizer_fingerprint = {"sha256": "a" * 64}
    source_fingerprint = {
        "class": "tests.FakeParent",
        "model_state_fingerprint": {"state_dict_sha256": STATE_SHA},
        "sha256": "b" * 64,
    }
    parameters = preparer._teacher_parameters("cpu")
    teacher_payload = {
        "schema": hf_teacher_token_replay.TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": calibration_texts,
        "parameters": parameters,
        "tokenizer_fingerprints": {
            "source": tokenizer_fingerprint,
            "candidate": tokenizer_fingerprint,
        },
        "source_model_fingerprint": source_fingerprint,
        "provenance_sha256": {
            "calibration_texts": hf_recovery._ordered_recovery_texts_sha256(
                calibration_texts
            )
        },
        "payload_sha256": "2" * 64,
    }
    cuda = _FakeCuda()
    torch_module = SimpleNamespace(
        __version__="test-torch",
        cuda=cuda,
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(),
    )
    transformers = SimpleNamespace(__version__="test-transformers")
    tokenizer = object()
    events: list[str] = []
    requests: list[dict[str, Any]] = []

    monkeypatch.setattr(
        preparer.hf_teacher_token_replay,
        "tokenizer_functional_fingerprint",
        lambda _tokenizer: copy.deepcopy(tokenizer_fingerprint),
    )

    def authenticate_reservoir(**_kwargs: Any) -> object:
        assert "parent-model-load" not in events
        events.append("reservoir-frozen")
        return object()

    monkeypatch.setattr(
        preparer,
        "_authenticate_fresh_recovery_candidate_reservoir",
        authenticate_reservoir,
    )
    original_request = preparer._teacher_replay_cache_request

    def observed_request(**kwargs: Any) -> dict[str, Any]:
        assert events and events[-1] == "reservoir-frozen"
        events.append("cache-request")
        request = original_request(**kwargs)
        requests.append(request)
        return request

    monkeypatch.setattr(preparer, "_teacher_replay_cache_request", observed_request)
    original_read = hf_teacher_replay_cache.read_entry

    def observed_read(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        assert "reservoir-frozen" in events
        events.append("cache-read")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(hf_teacher_replay_cache, "read_entry", observed_read)
    original_publish = hf_teacher_replay_cache.publish_entry

    def observed_publish(**kwargs: Any) -> tuple[dict[str, Any], str]:
        events.append("cache-publish")
        return original_publish(**kwargs)

    monkeypatch.setattr(hf_teacher_replay_cache, "publish_entry", observed_publish)

    def load_parent(**_kwargs: Any) -> object:
        events.append("parent-model-load")
        return object()

    def fingerprint_parent(_model: Any) -> dict[str, Any]:
        events.append("parent-model-fingerprint")
        return copy.deepcopy(source_fingerprint)

    def build_teacher(**_kwargs: Any) -> dict[str, Any]:
        events.append("teacher-builder")
        return copy.deepcopy(teacher_payload)

    class LateSelectionFailure(RuntimeError):
        pass

    def fail_late_selection(**_kwargs: Any) -> Any:
        events.append("final-selection")
        raise LateSelectionFailure("deliberate failure after replay acquisition")

    monkeypatch.setattr(
        preparer,
        "_select_fresh_recovery_candidate",
        fail_late_selection,
    )

    base_args = {
        "source_model": source,
        "source_tokenizer": source,
        "candidate_tokenizer": source,
        "calibration_file": calibration,
        "behavior_eval_suite": None,
        "behavior_mode": preparer.BEHAVIOR_MODE_FRESH,
        "v4_preregistration": preregistration,
        "v4_preregistration_sha256": preparer._file_sha256(preregistration),
        "expected_parent_artifact_tree_sha256": preparer.stage._artifact_tree(source)[
            "tree_sha256"
        ],
        "teacher_device": "cpu",
        "recovery_device": "cpu",
        "component_loader": None,
        "tokenizer_loader": lambda **_kwargs: preparer.TokenizerComponents(
            torch=torch_module,
            transformers=transformers,
            source_tokenizer=tokenizer,
            candidate_tokenizer=tokenizer,
        ),
        "source_model_loader": load_parent,
        "source_model_fingerprinter": fingerprint_parent,
        "teacher_replay_builder": build_teacher,
        "_test_only_cache_injection_authority": (
            preparer._CACHE_TEST_INJECTION_AUTHORITY
        ),
    }

    def run_late_failure(mode: str, cache_dir: Path, output_name: str) -> list[str]:
        events.clear()
        with pytest.raises(
            LateSelectionFailure,
            match="failure after replay acquisition",
        ):
            preparer.prepare_recovery_inputs(
                **base_args,
                output_dir=tmp_path / output_name,
                teacher_replay_cache_dir=cache_dir,
                teacher_replay_cache_mode=mode,
            )
        assert not (tmp_path / output_name).exists()
        return list(events)

    missing_cache = tmp_path / "missing-cache"
    events.clear()
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="Required teacher replay cache entry is missing",
    ):
        preparer.prepare_recovery_inputs(
            **base_args,
            output_dir=tmp_path / "require-missing-output",
            teacher_replay_cache_dir=missing_cache,
            teacher_replay_cache_mode="require",
        )
    assert events[:3] == ["reservoir-frozen", "cache-request", "cache-read"]
    assert "parent-model-load" not in events
    assert "teacher-builder" not in events

    cache_dir = tmp_path / "durable-cache"
    miss_events = run_late_failure("auto", cache_dir, "auto-miss-output")
    assert miss_events.index("reservoir-frozen") < miss_events.index("cache-read")
    assert miss_events.index("cache-read") < miss_events.index("parent-model-load")
    assert miss_events.index("teacher-builder") < miss_events.index("cache-publish")
    assert miss_events.index("cache-publish") < miss_events.index("final-selection")
    assert miss_events.count("parent-model-fingerprint") == 1
    request = requests[-1]
    persisted = original_read(cache_dir, request)
    assert persisted is not None
    entry_path = hf_teacher_replay_cache.cache_entry_path(cache_dir, request)
    envelope_path = entry_path / hf_teacher_replay_cache.CACHE_FILENAME
    original_bytes = envelope_path.read_bytes()

    hit_events = run_late_failure("auto", cache_dir, "auto-hit-output")
    assert "teacher-builder" not in hit_events
    assert "cache-publish" not in hit_events
    assert hit_events.count("parent-model-fingerprint") == 1
    assert hit_events.index("cache-read") < hit_events.index("parent-model-load")
    assert hit_events.index("parent-model-fingerprint") < hit_events.index(
        "final-selection"
    )

    require_events = run_late_failure("require", cache_dir, "require-hit-output")
    assert "teacher-builder" not in require_events
    assert "cache-publish" not in require_events
    assert require_events.count("parent-model-fingerprint") == 1

    refresh_events = run_late_failure("refresh", cache_dir, "refresh-output")
    assert refresh_events.index("teacher-builder") < refresh_events.index(
        "cache-publish"
    )
    assert refresh_events.index("cache-publish") < refresh_events.index(
        "final-selection"
    )
    assert refresh_events.count("parent-model-fingerprint") == 1
    assert envelope_path.read_bytes() == original_bytes
    assert original_read(cache_dir, requests[-1]) == persisted

    for field in (
        "tokenizer_loader",
        "source_model_loader",
        "teacher_replay_builder",
        "source_model_fingerprinter",
    ):
        production_args = dict(base_args)
        production_args["_test_only_cache_injection_authority"] = None
        production_args["tokenizer_loader"] = preparer._CANONICAL_TOKENIZER_LOADER
        production_args["source_model_loader"] = preparer._CANONICAL_SOURCE_MODEL_LOADER
        production_args["teacher_replay_builder"] = (
            preparer._CANONICAL_TEACHER_REPLAY_BUILDER
        )
        production_args["source_model_fingerprinter"] = (
            preparer._CANONICAL_SOURCE_MODEL_FINGERPRINTER
        )
        production_args[field] = lambda **_kwargs: {}
        events.clear()
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="canonical tokenizer loader, source-model loader, teacher replay",
        ):
            preparer.prepare_recovery_inputs(
                **production_args,
                output_dir=tmp_path / f"noncanonical-{field}",
                teacher_replay_cache_dir=tmp_path / f"noncanonical-cache-{field}",
                teacher_replay_cache_mode="auto",
            )
        assert events == []

    wrong_fingerprint = copy.deepcopy(source_fingerprint)
    wrong_fingerprint["model_state_fingerprint"]["state_dict_sha256"] = "f" * 64

    def fingerprint_wrong_parent(_model: Any) -> dict[str, Any]:
        events.append("wrong-parent-model-fingerprint")
        return copy.deepcopy(wrong_fingerprint)

    wrong_miss_args = dict(base_args)
    wrong_miss_args["source_model_fingerprinter"] = fingerprint_wrong_parent
    wrong_miss_cache = tmp_path / "wrong-live-miss-cache"
    events.clear()
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="cache publication failed closed",
    ):
        preparer.prepare_recovery_inputs(
            **wrong_miss_args,
            output_dir=tmp_path / "wrong-live-miss-output",
            teacher_replay_cache_dir=wrong_miss_cache,
            teacher_replay_cache_mode="auto",
        )
    assert events.count("wrong-parent-model-fingerprint") == 1
    assert events.count("teacher-builder") == 1
    assert "final-selection" not in events
    assert original_read(wrong_miss_cache, requests[-1]) is None

    wrong_refresh_args = dict(base_args)
    wrong_refresh_args["source_model_fingerprinter"] = fingerprint_wrong_parent
    events.clear()
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="cache publication failed closed",
    ):
        preparer.prepare_recovery_inputs(
            **wrong_refresh_args,
            output_dir=tmp_path / "wrong-live-refresh-output",
            teacher_replay_cache_dir=cache_dir,
            teacher_replay_cache_mode="refresh",
        )
    assert events.count("wrong-parent-model-fingerprint") == 1
    assert events.count("teacher-builder") == 1
    assert "final-selection" not in events
    assert envelope_path.read_bytes() == original_bytes
    assert original_read(cache_dir, requests[-1]) == persisted


def _cache_request_with_producer_rows(
    current_request: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    request = copy.deepcopy(current_request)
    producer_body = {
        "schema": request["producer_byte_hash_manifest"]["schema"],
        "files": sorted(rows, key=lambda row: row["role"]),
    }
    request["producer_byte_hash_manifest"] = {
        **producer_body,
        "sha256": hf_teacher_replay_cache._canonical_sha256(producer_body),
    }
    request_body = dict(request)
    request_body.pop("request_sha256")
    request["request_sha256"] = hf_teacher_replay_cache._canonical_sha256(request_body)
    return hf_teacher_replay_cache.verify_request(request)


def _cache_request_with_runtime_identity(
    current_request: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    request = copy.deepcopy(current_request)
    runtime = copy.deepcopy(runtime_identity)
    runtime_body = dict(runtime)
    runtime_body.pop("sha256", None)
    runtime["sha256"] = hf_teacher_replay_cache._canonical_sha256(runtime_body)
    request["runtime_identity"] = runtime
    request_body = dict(request)
    request_body.pop("request_sha256")
    request["request_sha256"] = hf_teacher_replay_cache._canonical_sha256(request_body)
    return hf_teacher_replay_cache.verify_request(request)


def test_explicit_compatible_cache_import_republishes_without_teacher_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    source = tmp_path / "source-import"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"canonical parent weights")
    calibration = tmp_path / "calibration-import.jsonl"
    calibration_texts = [
        f"{index:08x} import calibration row remains unique and long enough."
        for index in range(640)
    ]
    calibration.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in calibration_texts),
        encoding="utf-8",
    )
    preregistration = tmp_path / "preregistration-import.json"
    _write_json(preregistration, {"sealed_before_model_access": True})
    tokenizer_fingerprint = {"sha256": "a" * 64}
    source_fingerprint = {
        "class": "tests.FakeParent",
        "model_state_fingerprint": {"state_dict_sha256": STATE_SHA},
        "sha256": "b" * 64,
    }
    teacher_payload = {
        "schema": hf_teacher_token_replay.TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": calibration_texts,
        "parameters": preparer._teacher_parameters("cpu"),
        "tokenizer_fingerprints": {
            "source": tokenizer_fingerprint,
            "candidate": tokenizer_fingerprint,
        },
        "source_model_fingerprint": source_fingerprint,
        "provenance_sha256": {
            "calibration_texts": hf_recovery._ordered_recovery_texts_sha256(
                calibration_texts
            )
        },
        "payload_sha256": "2" * 64,
    }
    torch_module = SimpleNamespace(
        __version__="test-torch",
        cuda=_FakeCuda(),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(),
    )
    transformers = SimpleNamespace(__version__="test-transformers")
    actual_tree = preparer.stage._artifact_tree(source)
    current_request = preparer._teacher_replay_cache_request(
        actual_parent_tree=actual_tree,
        expected_parent_tree_sha256=actual_tree["tree_sha256"],
        calibration_path=calibration,
        ordered_calibration=calibration_texts,
        source_tokenizer_fingerprint=tokenizer_fingerprint,
        candidate_tokenizer_fingerprint=tokenizer_fingerprint,
        torch=torch_module,
        transformers=transformers,
        teacher_device="cpu",
    )
    current_rows = copy.deepcopy(
        current_request["producer_byte_hash_manifest"]["files"]
    )
    legacy_rows = [
        (
            dict(preparer.LEGACY_TEACHER_CACHE_IMPORT_PREP_PRODUCER)
            if row["role"] == "prepare_recovery_inputs"
            else row
        )
        for row in current_rows
    ]
    legacy_request = _cache_request_with_producer_rows(current_request, legacy_rows)
    cache_dir = tmp_path / "compatible-import-cache"
    legacy_envelope, _result = hf_teacher_replay_cache.publish_entry(
        cache_dir=cache_dir,
        request=legacy_request,
        teacher_payload=teacher_payload,
        live_source_model_fingerprint=source_fingerprint,
    )
    external_prompt = preparer._load_external_fresh_reservoir_seed().prompts[0]["text"]
    assert external_prompt not in json.dumps(current_request, sort_keys=True)
    assert external_prompt not in json.dumps(legacy_envelope, sort_keys=True)

    events: list[str] = []
    captured_receipts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        preparer.hf_teacher_token_replay,
        "tokenizer_functional_fingerprint",
        lambda _tokenizer: copy.deepcopy(tokenizer_fingerprint),
    )

    def freeze_reservoir(**_kwargs: Any) -> object:
        events.append("reservoir-frozen")
        return object()

    monkeypatch.setattr(
        preparer, "_authenticate_fresh_recovery_candidate_reservoir", freeze_reservoir
    )
    original_import = preparer._read_exact_compatible_teacher_cache_import

    def observed_import(**kwargs: Any) -> preparer.CompatibleTeacherCacheImport:
        assert events and events[0] == "reservoir-frozen"
        events.append("import-read")
        return original_import(**kwargs)

    monkeypatch.setattr(
        preparer, "_read_exact_compatible_teacher_cache_import", observed_import
    )
    original_finalize = preparer._finalize_teacher_cache_import_receipt

    def observed_finalize(**kwargs: Any) -> dict[str, Any]:
        receipt = original_finalize(**kwargs)
        captured_receipts.append(receipt)
        return receipt

    monkeypatch.setattr(
        preparer, "_finalize_teacher_cache_import_receipt", observed_finalize
    )

    def load_parent(**_kwargs: Any) -> object:
        events.append("parent-model-load")
        return object()

    def fingerprint_parent(_model: Any) -> dict[str, Any]:
        events.append("parent-model-fingerprint")
        return copy.deepcopy(source_fingerprint)

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("teacher replay generation is forbidden during import")

    class LateSelectionFailure(RuntimeError):
        pass

    def fail_late_selection(**_kwargs: Any) -> Any:
        events.append("final-selection")
        raise LateSelectionFailure("late failure preserves migrated current cache")

    monkeypatch.setattr(
        preparer, "_select_fresh_recovery_candidate", fail_late_selection
    )
    base_args = {
        "source_model": source,
        "source_tokenizer": source,
        "candidate_tokenizer": source,
        "calibration_file": calibration,
        "behavior_eval_suite": None,
        "behavior_mode": preparer.BEHAVIOR_MODE_FRESH,
        "v4_preregistration": preregistration,
        "v4_preregistration_sha256": preparer._file_sha256(preregistration),
        "expected_parent_artifact_tree_sha256": actual_tree["tree_sha256"],
        "teacher_device": "cpu",
        "recovery_device": "cpu",
        "component_loader": None,
        "tokenizer_loader": lambda **_kwargs: preparer.TokenizerComponents(
            torch=torch_module,
            transformers=transformers,
            source_tokenizer=object(),
            candidate_tokenizer=object(),
        ),
        "source_model_loader": load_parent,
        "source_model_fingerprinter": fingerprint_parent,
        "teacher_replay_builder": forbidden_builder,
        "_test_only_cache_injection_authority": (
            preparer._CACHE_TEST_INJECTION_AUTHORITY
        ),
    }
    with pytest.raises(LateSelectionFailure, match="preserves migrated"):
        preparer.prepare_recovery_inputs(
            **base_args,
            output_dir=tmp_path / "migrated-late-output",
            teacher_replay_cache_dir=cache_dir,
            teacher_replay_cache_mode="require",
            teacher_replay_cache_import_request_sha256=legacy_request["request_sha256"],
            teacher_replay_cache_import_envelope_sha256=legacy_envelope[
                "envelope_sha256"
            ],
        )
    assert events.index("reservoir-frozen") < events.index("import-read")
    assert events.index("import-read") < events.index("parent-model-load")
    assert "final-selection" in events
    assert not (tmp_path / "migrated-late-output").exists()
    current_envelope = hf_teacher_replay_cache.read_entry(cache_dir, current_request)
    assert current_envelope is not None
    assert (
        current_envelope["teacher_token_replay"]
        == legacy_envelope["teacher_token_replay"]
    )
    assert len(captured_receipts) == 1
    assert (
        preparer._verify_teacher_cache_import_receipt(
            captured_receipts[0],
            current_request_sha256=current_request["request_sha256"],
            current_envelope_sha256=current_envelope["envelope_sha256"],
        )
        == captured_receipts[0]
    )
    assert external_prompt not in json.dumps(captured_receipts[0], sort_keys=True)
    report_replay = tmp_path / "report-replay.json"
    report_plan = tmp_path / "report-plan.json"
    report_replay.write_text("{}\n", encoding="utf-8")
    report_plan.write_text("{}\n", encoding="utf-8")
    source_binding = preparer._source_binding(
        expected_tree_sha256=actual_tree["tree_sha256"],
        actual_tree=actual_tree,
        state_dict_sha256=STATE_SHA,
    )
    acquisition = {
        "schema": preparer.TEACHER_REPLAY_ACQUISITION_SCHEMA,
        "cache_mode": "require",
        "origin": "cache-import",
        "cache_result": captured_receipts[0]["current_cache_publication_result"],
        "cache_request_sha256": current_request["request_sha256"],
        "cache_envelope_sha256": current_envelope["envelope_sha256"],
        "cache_entry_relative_path": (
            f"{current_request['request_sha256']}/"
            f"{hf_teacher_replay_cache.CACHE_FILENAME}"
        ),
        "cache_verified_before_final_selection": True,
        "cache_import": captured_receipts[0],
    }
    usage = {
        "schema": preparer.PARENT_MODEL_USAGE_SCHEMA,
        "model_load_count": 1,
        "load_purpose": "fresh-targets",
        "teacher_replay_generated_this_run": False,
        "teacher_replay_reused_from_cache": True,
        "fresh_target_generation_performed": True,
    }
    behavior_binding = {
        "payload_sha256": "4" * 64,
        "mode": preparer.BEHAVIOR_MODE_FRESH,
        "causal_ab_eligible": True,
        "behavior_probe_count": preparer.BEHAVIOR_PROBE_COUNT,
        "generation_prompt_count": preparer.GENERATION_PROMPT_COUNT,
        "adapter_provenance": {
            "selection": {
                "external_candidate_source": dict(
                    preparer._load_external_fresh_reservoir_seed().receipt
                )
            }
        },
    }
    import_report = preparer._build_report(
        replay_path=report_replay,
        plan_path=report_plan,
        teacher_payload=teacher_payload,
        source_binding=source_binding,
        behavior_binding=behavior_binding,
        plan_payload={"plan_payload_sha256": "5" * 64},
        teacher_generation_device="cpu",
        recovery_device="cpu",
        teacher_replay_acquisition=acquisition,
        parent_model_usage=usage,
    )
    assert preparer.verify_build_report(import_report) == import_report

    events.clear()
    with pytest.raises(LateSelectionFailure, match="preserves migrated"):
        preparer.prepare_recovery_inputs(
            **base_args,
            output_dir=tmp_path / "current-hit-late-output",
            teacher_replay_cache_dir=cache_dir,
            teacher_replay_cache_mode="require",
        )
    assert "import-read" not in events
    assert "parent-model-fingerprint" in events

    def publish_variant(
        name: str,
        rows: list[dict[str, Any]],
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        variant_request = _cache_request_with_producer_rows(current_request, rows)
        variant_cache = tmp_path / name
        variant_envelope, _ = hf_teacher_replay_cache.publish_entry(
            cache_dir=variant_cache,
            request=variant_request,
            teacher_payload=teacher_payload,
            live_source_model_fingerprint=source_fingerprint,
        )
        return variant_cache, variant_request, variant_envelope

    bad_prep_hash = copy.deepcopy(legacy_rows)
    next(row for row in bad_prep_hash if row["role"] == "prepare_recovery_inputs")[
        "sha256"
    ] = "e" * 64
    bad_prep_bytes = copy.deepcopy(legacy_rows)
    next(row for row in bad_prep_bytes if row["role"] == "prepare_recovery_inputs")[
        "byte_count"
    ] += 1
    bad_nonprep = copy.deepcopy(legacy_rows)
    next(row for row in bad_nonprep if row["role"] == "hf_recovery")["sha256"] = (
        "e" * 64
    )
    for name, rows in (
        ("bad-prep-hash", bad_prep_hash),
        ("bad-prep-bytes", bad_prep_bytes),
        ("bad-nonprep", bad_nonprep),
    ):
        variant_cache, variant_request, variant_envelope = publish_variant(name, rows)
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="not semantically compatible",
        ):
            original_import(
                cache_dir=variant_cache,
                current_request=current_request,
                source_request_sha256=variant_request["request_sha256"],
                source_envelope_sha256=variant_envelope["envelope_sha256"],
            )

    pinned_current_runtime = copy.deepcopy(current_request["runtime_identity"])
    pinned_current_runtime["cudnn_version"] = 91_900
    pinned_current_request = _cache_request_with_runtime_identity(
        current_request,
        pinned_current_runtime,
    )
    pinned_legacy_runtime = copy.deepcopy(pinned_current_request["runtime_identity"])
    pinned_legacy_runtime["cudnn_version"] = None
    pinned_legacy_request = _cache_request_with_runtime_identity(
        pinned_current_request,
        pinned_legacy_runtime,
    )
    pinned_legacy_request = _cache_request_with_producer_rows(
        pinned_legacy_request,
        legacy_rows,
    )
    pinned_runtime_cache = tmp_path / "pinned-runtime-observation-import"
    pinned_runtime_envelope, _ = hf_teacher_replay_cache.publish_entry(
        cache_dir=pinned_runtime_cache,
        request=pinned_legacy_request,
        teacher_payload=teacher_payload,
        live_source_model_fingerprint=source_fingerprint,
    )
    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(
            preparer,
            "LEGACY_TEACHER_CACHE_IMPORT_RUNTIME_IDENTITY_SHA256",
            pinned_legacy_request["runtime_identity"]["sha256"],
        )
        runtime_patch.setattr(
            preparer,
            "LEGACY_TEACHER_CACHE_IMPORT_CURRENT_RUNTIME_IDENTITY_SHA256",
            pinned_current_request["runtime_identity"]["sha256"],
        )
        runtime_patch.setattr(
            preparer,
            "LEGACY_TEACHER_CACHE_IMPORT_CURRENT_CUDNN_VERSION",
            91_900,
        )
        pinned_import = original_import(
            cache_dir=pinned_runtime_cache,
            current_request=pinned_current_request,
            source_request_sha256=pinned_legacy_request["request_sha256"],
            source_envelope_sha256=pinned_runtime_envelope["envelope_sha256"],
        )
        runtime_receipt = pinned_import.pre_model_receipt[
            "runtime_identity_compatibility"
        ]
        assert runtime_receipt == {
            "mode": "pinned-legacy-missing-cudnn-version",
            "source_runtime_identity_sha256": pinned_legacy_request["runtime_identity"][
                "sha256"
            ],
            "current_runtime_identity_sha256": pinned_current_request[
                "runtime_identity"
            ]["sha256"],
            "source_cudnn_version": None,
            "current_cudnn_version": 91_900,
            "legacy_missing_cudnn_version_allowlist_applied": True,
            "only_cudnn_version_and_runtime_self_hash_changed": True,
        }
        pinned_current_envelope, pinned_publication = (
            hf_teacher_replay_cache.publish_entry(
                cache_dir=pinned_runtime_cache,
                request=pinned_current_request,
                teacher_payload=teacher_payload,
                live_source_model_fingerprint=source_fingerprint,
            )
        )
        pinned_final_receipt = preparer._finalize_teacher_cache_import_receipt(
            compatible_import=pinned_import,
            current_envelope=pinned_current_envelope,
            publication_result=pinned_publication,
        )
        assert (
            preparer._verify_teacher_cache_import_receipt(
                pinned_final_receipt,
                current_request_sha256=pinned_current_request["request_sha256"],
                current_envelope_sha256=pinned_current_envelope["envelope_sha256"],
            )
            == pinned_final_receipt
        )
        tampered_runtime_receipt = copy.deepcopy(pinned_final_receipt)
        tampered_runtime_receipt["runtime_identity_compatibility"][
            "current_cudnn_version"
        ] = 91_901
        tampered_body = dict(tampered_runtime_receipt)
        tampered_body.pop("payload_sha256")
        tampered_runtime_receipt["payload_sha256"] = preparer.canonical_sha256(
            tampered_body
        )
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="receipt failed authentication",
        ):
            preparer._verify_teacher_cache_import_receipt(
                tampered_runtime_receipt,
                current_request_sha256=pinned_current_request["request_sha256"],
                current_envelope_sha256=pinned_current_envelope["envelope_sha256"],
            )
        incompatible_runtime = copy.deepcopy(pinned_current_request["runtime_identity"])
        incompatible_runtime["cudnn_benchmark"] = True
        incompatible_current_request = _cache_request_with_runtime_identity(
            pinned_current_request,
            incompatible_runtime,
        )
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="runtime identity changed outside",
        ):
            original_import(
                cache_dir=pinned_runtime_cache,
                current_request=incompatible_current_request,
                source_request_sha256=pinned_legacy_request["request_sha256"],
                source_envelope_sha256=pinned_runtime_envelope["envelope_sha256"],
            )

    for request_sha, envelope_sha in (
        ("f" * 64, legacy_envelope["envelope_sha256"]),
        (legacy_request["request_sha256"], "f" * 64),
    ):
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="strict path/request/envelope authentication",
        ):
            original_import(
                cache_dir=cache_dir,
                current_request=current_request,
                source_request_sha256=request_sha,
                source_envelope_sha256=envelope_sha,
            )

    malformed_cache = tmp_path / "malformed-import-cache"
    malformed_entry = malformed_cache / legacy_request["request_sha256"]
    malformed_entry.mkdir(parents=True)
    (malformed_entry / hf_teacher_replay_cache.CACHE_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="strict path/request/envelope authentication",
    ):
        original_import(
            cache_dir=malformed_cache,
            current_request=current_request,
            source_request_sha256=legacy_request["request_sha256"],
            source_envelope_sha256=legacy_envelope["envelope_sha256"],
        )

    symlink_root = tmp_path / "symlink-import-cache"
    try:
        symlink_root.symlink_to(cache_dir, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="strict path/request/envelope authentication",
        ):
            original_import(
                cache_dir=symlink_root,
                current_request=current_request,
                source_request_sha256=legacy_request["request_sha256"],
                source_envelope_sha256=legacy_envelope["envelope_sha256"],
            )

    symlink_entry_root = tmp_path / "symlink-entry-import-cache"
    symlink_entry_root.mkdir()
    symlink_entry = symlink_entry_root / legacy_request["request_sha256"]
    try:
        symlink_entry.symlink_to(
            cache_dir / legacy_request["request_sha256"],
            target_is_directory=True,
        )
    except OSError:
        pass
    else:
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="strict path/request/envelope authentication",
        ):
            original_import(
                cache_dir=symlink_entry_root,
                current_request=current_request,
                source_request_sha256=legacy_request["request_sha256"],
                source_envelope_sha256=legacy_envelope["envelope_sha256"],
            )

    symlink_envelope_root = tmp_path / "symlink-envelope-import-cache"
    symlink_envelope_entry = symlink_envelope_root / legacy_request["request_sha256"]
    symlink_envelope_entry.mkdir(parents=True)
    real_envelope = tmp_path / "authorized-legacy-cache-envelope.json"
    _write_json(real_envelope, legacy_envelope)
    symlink_envelope = symlink_envelope_entry / hf_teacher_replay_cache.CACHE_FILENAME
    try:
        symlink_envelope.symlink_to(real_envelope)
    except OSError:
        pass
    else:
        with pytest.raises(
            preparer.RecoveryInputPreparationError,
            match="strict path/request/envelope authentication",
        ):
            original_import(
                cache_dir=symlink_envelope_root,
                current_request=current_request,
                source_request_sha256=legacy_request["request_sha256"],
                source_envelope_sha256=legacy_envelope["envelope_sha256"],
            )

    extra_child_root = tmp_path / "extra-child-import-cache"
    extra_child_entry = extra_child_root / legacy_request["request_sha256"]
    extra_child_entry.mkdir(parents=True)
    _write_json(
        extra_child_entry / hf_teacher_replay_cache.CACHE_FILENAME,
        legacy_envelope,
    )
    (extra_child_entry / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="strict path/request/envelope authentication",
    ):
        original_import(
            cache_dir=extra_child_root,
            current_request=current_request,
            source_request_sha256=legacy_request["request_sha256"],
            source_envelope_sha256=legacy_envelope["envelope_sha256"],
        )

    wrong_cache, wrong_request, wrong_envelope = publish_variant(
        "wrong-live-import", legacy_rows
    )
    wrong_fingerprint = copy.deepcopy(source_fingerprint)
    wrong_fingerprint["model_state_fingerprint"]["state_dict_sha256"] = "f" * 64
    wrong_args = dict(base_args)
    wrong_args["source_model_fingerprinter"] = lambda _model: wrong_fingerprint
    with pytest.raises(
        preparer.RecoveryInputPreparationError,
        match="source-model binding failed authentication",
    ):
        preparer.prepare_recovery_inputs(
            **wrong_args,
            output_dir=tmp_path / "wrong-live-import-output",
            teacher_replay_cache_dir=wrong_cache,
            teacher_replay_cache_mode="require",
            teacher_replay_cache_import_request_sha256=wrong_request["request_sha256"],
            teacher_replay_cache_import_envelope_sha256=wrong_envelope[
                "envelope_sha256"
            ],
        )
    assert hf_teacher_replay_cache.read_entry(wrong_cache, current_request) is None
