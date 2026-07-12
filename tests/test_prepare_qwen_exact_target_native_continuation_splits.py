from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_continuation_splits as builder  # noqa: E402
from neural_forge.hf_evaluation_provenance import (  # noqa: E402
    token_sequence_sha256,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        pieces = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)
        return {
            "input_ids": [
                3
                + int(hashlib.sha256(piece.encode()).hexdigest()[:7], 16)
                % 100_000
                for piece in pieces
            ]
        }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_text(role: str, index: int) -> str:
    return (
        f"legacy {role} identifier {index:04d} provides unique alpha beta gamma "
        "delta epsilon zeta eta theta context"
    )


def _prompt(tokenizer: _Tokenizer, text: str, source_line: int) -> dict[str, Any]:
    tokens = tokenizer(text)["input_ids"]
    digest = builder._token_sequence_sha256(tokens)
    return {
        "token_ids": tokens,
        "prompt_id": f"prompt-{digest}",
        "token_ids_sha256": digest,
        "source_line_number": source_line,
        "source_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _build_inputs(
    root: Path,
    *,
    fresh_count: int = 220,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    tokenizer = _Tokenizer()
    tokenizer_root = root / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer_json = tokenizer_root / "tokenizer.json"
    tokenizer_json.write_bytes(b"deterministic-test-tokenizer")

    split_dir = root / "original"
    split_dir.mkdir()
    split_hashes: dict[str, str] = {}
    split_prompts: dict[str, list[dict[str, Any]]] = {}
    source_line = 1
    for role in ("fit", "candidate", "sealed"):
        prompts = []
        for index in range(builder.ORIGINAL_COUNTS[role]):
            prompts.append(
                _prompt(tokenizer, _legacy_text(role, index), source_line)
            )
            source_line += 1
        split_prompts[role] = prompts
        payload = {
            "schema": builder.SPLIT_SCHEMA,
            "split": role,
            "sealed_before_model_access": role == "sealed",
            "prompts": prompts,
            "source": {"fixture": True},
            "raw_prompt_text_retained": False,
        }
        path = split_dir / f"{role}.json"
        path.write_bytes(_json_bytes(payload))
        split_hashes[role] = _file_sha256(path)

    old_texts = [
        _legacy_text(role, index)
        for role in ("fit", "candidate", "sealed")
        for index in range(builder.ORIGINAL_COUNTS[role])
    ]
    candidate_words = _legacy_text("candidate", 0).split()
    prefix_collision = " ".join(
        [*candidate_words[:8], "different", "continuation", "branch"]
    )
    fresh_texts = [
        (
            f"fresh corpus identifier {index:05d} requests unique topic alpha beta "
            "gamma delta epsilon zeta eta theta answer"
        )
        for index in range(fresh_count)
    ]
    corpus_rows = [*old_texts, prefix_collision, *fresh_texts]
    corpus = root / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for text in corpus_rows:
            handle.write(json.dumps({"prompt": text}, sort_keys=True) + "\n")

    return {
        "tokenizer": tokenizer,
        "tokenizer_root": tokenizer_root,
        "tokenizer_sha256": _file_sha256(tokenizer_json),
        "split_dir": split_dir,
        "split_hashes": split_hashes,
        "split_prompts": split_prompts,
        "corpus": corpus,
        "corpus_sha256": _file_sha256(corpus),
        "corpus_count": len(corpus_rows),
        "prefix_collision_text": prefix_collision,
    }


def _prepare(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    loader_calls: list[Path] = []

    def load(path: Path) -> _Tokenizer:
        loader_calls.append(path)
        return inputs["tokenizer"]

    report = builder.prepare_continuation_splits(
        output_dir=output,
        original_split_dir=inputs["split_dir"],
        corpus_path=inputs["corpus"],
        tokenizer_path=inputs["tokenizer_root"],
        original_split_sha256s=inputs["split_hashes"],
        corpus_sha256=inputs["corpus_sha256"],
        corpus_line_count=inputs["corpus_count"],
        tokenizer_json_sha256=inputs["tokenizer_sha256"],
        tokenizer_loader=load,
    )
    assert loader_calls == [inputs["tokenizer_root"].resolve()]
    return report


def _prepare_lexical(
    inputs: dict[str, Any],
    source: Path,
    output: Path,
) -> dict[str, Any]:
    source_hashes = {
        role: _file_sha256(source / f"{role}.json")
        for role in ("fit", "candidate", "sealed")
    }
    loader_calls: list[Path] = []

    def load(path: Path) -> _Tokenizer:
        loader_calls.append(path)
        return inputs["tokenizer"]

    report = builder.prepare_lexical_continuation_splits(
        output_dir=output,
        source_split_dir=source,
        corpus_path=inputs["corpus"],
        tokenizer_path=inputs["tokenizer_root"],
        source_split_sha256s=source_hashes,
        old_candidate_diagnostic_sha256=_file_sha256(
            source / "old_candidate_diagnostic.json"
        ),
        corpus_sha256=inputs["corpus_sha256"],
        corpus_line_count=inputs["corpus_count"],
        tokenizer_json_sha256=inputs["tokenizer_sha256"],
        tokenizer_loader=load,
    )
    assert loader_calls == [inputs["tokenizer_root"].resolve()]
    return report


def _prefixes(prompt: dict[str, Any]) -> set[tuple[int, ...]]:
    tokens = tuple(prompt["token_ids"])
    return {
        tokens[:length]
        for length in range(builder.MINIMUM_MEANINGFUL_PREFIX_TOKENS, len(tokens) + 1)
    }


def test_token_hash_exactly_matches_canonical_neural_forge_convention() -> None:
    tokens = (1, 2, 3, 151_935)
    assert builder._token_sequence_sha256(tokens) == token_sequence_sha256(tokens)


def test_builder_composes_exact_replay_fresh_and_diagnostic_domains(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs")
    output = tmp_path / "continuation"
    report = _prepare(inputs, output)

    assert {path.name for path in output.iterdir()} == {
        "fit.json",
        "candidate.json",
        "sealed.json",
        "old_candidate_diagnostic.json",
        "continuation_manifest.json",
        "build_report.json",
    }
    fit = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        (output / "candidate.json").read_text(encoding="utf-8")
    )
    sealed = json.loads((output / "sealed.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (output / "old_candidate_diagnostic.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "continuation_manifest.json").read_text(encoding="utf-8")
    )

    assert len(fit["prompts"]) == 96
    assert len(candidate["prompts"]) == 24
    assert len(sealed["prompts"]) == 32
    assert fit["prompts"][:24] == inputs["split_prompts"]["fit"]
    assert sealed["prompts"][:16] == inputs["split_prompts"]["sealed"]
    assert all("source_line_number" in value for value in fit["prompts"][24:])
    assert all("source_line_number" in value for value in candidate["prompts"])
    assert all("source_line_number" in value for value in sealed["prompts"][16:])

    assert diagnostic["schema"] == builder.OLD_CANDIDATE_DIAGNOSTIC_SCHEMA
    assert diagnostic["role"] == "diagnostic-only"
    assert diagnostic["prompts"] == inputs["split_prompts"]["candidate"]
    assert diagnostic["eligible_for_fit"] is False
    assert diagnostic["eligible_for_candidate_selection"] is False
    assert diagnostic["eligible_for_checkpoint_promotion"] is False
    assert diagnostic["schema"] != builder.SPLIT_SCHEMA

    active = {
        "fit": fit["prompts"],
        "candidate": candidate["prompts"],
        "sealed": sealed["prompts"],
    }
    seen_ids: set[str] = set()
    seen_prefixes: set[tuple[int, ...]] = set()
    for prompts in active.values():
        for prompt in prompts:
            assert prompt["prompt_id"] not in seen_ids
            assert not (_prefixes(prompt) & seen_prefixes)
            seen_ids.add(prompt["prompt_id"])
            seen_prefixes.update(_prefixes(prompt))
    old_candidate_ids = {
        value["prompt_id"] for value in inputs["split_prompts"]["candidate"]
    }
    old_candidate_prefixes = {
        prefix
        for value in inputs["split_prompts"]["candidate"]
        for prefix in _prefixes(value)
    }
    assert not (seen_ids & old_candidate_ids)
    assert not (seen_prefixes & old_candidate_prefixes)

    assert manifest["counts"] == {
        "active_output": {"candidate": 24, "fit": 96, "sealed": 32},
        "fresh": {"candidate": 24, "fit": 72, "sealed": 16},
        "old_candidate_diagnostic": 12,
        "original": {"candidate": 12, "fit": 24, "sealed": 16},
    }
    assert manifest["contract"]["old_candidate"] == "diagnostic-only"
    assert manifest["contract"]["original_fit_records_preserved_exactly"] is True
    assert (
        manifest["contract"]["original_sealed_records_preserved_exactly"] is True
    )
    assert manifest["model_access"] == {
        "model_class_imported": False,
        "model_created": False,
        "model_forward_calls": 0,
        "model_weights_loaded": False,
        "tokenizer_loaded": True,
    }
    assert report["model_accessed"] is False
    assert report["model_class_imported"] is False
    assert report["model_weights_loaded"] is False
    assert report["model_forward_calls"] == 0
    assert report["training_launched"] is False


def test_every_artifact_and_component_is_hash_bound_and_path_independent(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = _prepare(inputs, first)
    second_report = _prepare(inputs, second)

    assert first_report == second_report
    assert {path.name for path in first.iterdir()} == {
        path.name for path in second.iterdir()
    }
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    manifest_path = first / "continuation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_manifest_content = manifest.pop("manifest_content_sha256")
    assert declared_manifest_content == builder._canonical_sha256(manifest)
    for name, binding in manifest["artifacts"].items():
        artifact = first / binding["filename"]
        assert _file_sha256(artifact) == binding["sha256"], name
    for role in ("fit", "candidate", "sealed"):
        payload = json.loads((first / f"{role}.json").read_text(encoding="utf-8"))
        content_hash = payload.pop("split_content_sha256")
        assert content_hash == builder._canonical_sha256(payload)
    diagnostic = json.loads(
        (first / "old_candidate_diagnostic.json").read_text(encoding="utf-8")
    )
    diagnostic_hash = diagnostic.pop("diagnostic_content_sha256")
    assert diagnostic_hash == builder._canonical_sha256(diagnostic)
    assert first_report["continuation_manifest"]["sha256"] == _file_sha256(
        manifest_path
    )


def test_selection_explicitly_skips_old_exact_and_prefix_collisions(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs")
    originals = {
        role: builder._load_original_split(
            inputs["split_dir"],
            role,
            expected_sha256=inputs["split_hashes"][role],
        )
        for role in ("fit", "candidate", "sealed")
    }
    tokenizer = inputs["tokenizer"]
    exact_old_text = _legacy_text("fit", 0)
    collision_text = inputs["prefix_collision_text"]
    fresh_records = [
        {
            "source_line_number": index + 3,
            "source_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
        }
        for index, text in enumerate(
            [
                (
                    f"direct fresh identifier {fresh_index:05d} has unique alpha "
                    "beta gamma delta epsilon zeta eta theta response"
                )
                for fresh_index in range(140)
            ]
        )
    ]
    records = [
        {
            "source_line_number": 1,
            "source_prompt_sha256": hashlib.sha256(exact_old_text.encode()).hexdigest(),
            "text": exact_old_text,
        },
        {
            "source_line_number": 2,
            "source_prompt_sha256": hashlib.sha256(collision_text.encode()).hexdigest(),
            "text": collision_text,
        },
        *fresh_records,
    ]
    selected, stats = builder._select_fresh_domains(
        records,
        tokenizer,
        originals=originals,
        max_prompt_tokens=128,
    )

    assert stats["skipped_exact_prompt_collision"] == 1
    assert stats["skipped_meaningful_prefix_collision"] == 1
    assert {name: len(values) for name, values in selected.items()} == {
        "fit": 72,
        "candidate": 24,
        "sealed": 16,
    }
    prior_ids = {
        prompt.prompt_id
        for split in originals.values()
        for prompt in split.prompts
    }
    prior_prefixes = {
        prefix
        for split in originals.values()
        for prompt in split.prompts
        for prefix in prompt.prefixes
    }
    for prompts in selected.values():
        for prompt in prompts:
            assert prompt.prompt_id not in prior_ids
            assert not (prompt.prefixes & prior_prefixes)


def test_tamper_insufficient_corpus_and_existing_output_fail_closed(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs")
    with pytest.raises(
        builder.ContinuationSplitPreparationError,
        match="Original exact fit split hash changed",
    ):
        builder.prepare_continuation_splits(
            output_dir=tmp_path / "bad-original",
            original_split_dir=inputs["split_dir"],
            corpus_path=inputs["corpus"],
            tokenizer_path=inputs["tokenizer_root"],
            original_split_sha256s={
                **inputs["split_hashes"],
                "fit": "0" * 64,
            },
            corpus_sha256=inputs["corpus_sha256"],
            corpus_line_count=inputs["corpus_count"],
            tokenizer_json_sha256=inputs["tokenizer_sha256"],
            tokenizer_loader=lambda _path: inputs["tokenizer"],
        )

    with pytest.raises(
        builder.ContinuationSplitPreparationError,
        match="tokenizer JSON hash changed",
    ):
        builder.prepare_continuation_splits(
            output_dir=tmp_path / "bad-tokenizer",
            original_split_dir=inputs["split_dir"],
            corpus_path=inputs["corpus"],
            tokenizer_path=inputs["tokenizer_root"],
            original_split_sha256s=inputs["split_hashes"],
            corpus_sha256=inputs["corpus_sha256"],
            corpus_line_count=inputs["corpus_count"],
            tokenizer_json_sha256="f" * 64,
            tokenizer_loader=lambda _path: inputs["tokenizer"],
        )

    small_root = tmp_path / "small"
    small_inputs = _build_inputs(small_root, fresh_count=20)
    with pytest.raises(
        builder.ContinuationSplitPreparationError,
        match="Could not select the required fresh continuation prompts",
    ):
        _prepare(small_inputs, tmp_path / "insufficient")
    assert not (tmp_path / "insufficient").exists()

    output = tmp_path / "complete"
    _prepare(inputs, output)
    with pytest.raises(
        builder.ContinuationSplitPreparationError,
        match="must not already exist",
    ):
        _prepare(inputs, output)


def test_lexical_continuation_relabels_prior_candidate_and_reuses_sealed_exactly(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs", fresh_count=260)
    residual_attention = tmp_path / "residual-attention"
    _prepare(inputs, residual_attention)
    output = tmp_path / "lexical-continuation"
    report = _prepare_lexical(inputs, residual_attention, output)

    assert {path.name for path in output.iterdir()} == {
        "fit.json",
        "candidate.json",
        "lexical_continuation_manifest.json",
        "build_report.json",
    }
    prior_fit = json.loads(
        (residual_attention / "fit.json").read_text(encoding="utf-8")
    )
    prior_candidate = json.loads(
        (residual_attention / "candidate.json").read_text(encoding="utf-8")
    )
    prior_sealed_bytes = (residual_attention / "sealed.json").read_bytes()
    fit = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    candidate = json.loads((output / "candidate.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "lexical_continuation_manifest.json").read_text(encoding="utf-8")
    )

    assert len(fit["prompts"]) == 120
    assert fit["prompts"][:96] == prior_fit["prompts"]
    assert fit["prompts"][96:] == prior_candidate["prompts"]
    assert len(candidate["prompts"]) == 24
    assert not (output / "sealed.json").exists()
    assert (residual_attention / "sealed.json").read_bytes() == prior_sealed_bytes
    assert manifest["artifacts"]["sealed"] == {
        "copied_into_output": False,
        "filename": "sealed.json",
        "prompt_count": 32,
        "sha256": _file_sha256(residual_attention / "sealed.json"),
        "source_artifact_reused": True,
    }
    assert manifest["contract"]["prior_candidate_eligible_only_for_training"] is True
    assert manifest["contract"]["sealed_artifact_copied_or_rewritten"] is False
    assert report["sealed_artifact_reused_without_copy"] is True

    historical_prompts: list[dict[str, Any]] = []
    for name in ("fit", "candidate", "sealed"):
        historical_prompts.extend(
            json.loads(
                (residual_attention / f"{name}.json").read_text(encoding="utf-8")
            )["prompts"]
        )
    historical_prompts.extend(
        json.loads(
            (residual_attention / "old_candidate_diagnostic.json").read_text(
                encoding="utf-8"
            )
        )["prompts"]
    )
    historical_ids = {value["prompt_id"] for value in historical_prompts}
    historical_prefixes = {
        prefix for value in historical_prompts for prefix in _prefixes(value)
    }
    for prompt in candidate["prompts"]:
        assert prompt["prompt_id"] not in historical_ids
        assert not (_prefixes(prompt) & historical_prefixes)


def test_lexical_continuation_is_deterministic_and_fails_closed_on_source_tamper(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path / "inputs", fresh_count=260)
    residual_attention = tmp_path / "residual-attention"
    _prepare(inputs, residual_attention)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = _prepare_lexical(inputs, residual_attention, first)
    second_report = _prepare_lexical(inputs, residual_attention, second)
    assert first_report == second_report
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    source_hashes = {
        role: _file_sha256(residual_attention / f"{role}.json")
        for role in ("fit", "candidate", "sealed")
    }
    with pytest.raises(
        builder.ContinuationSplitPreparationError,
        match="Original exact sealed split hash changed",
    ):
        builder.prepare_lexical_continuation_splits(
            output_dir=tmp_path / "tampered",
            source_split_dir=residual_attention,
            corpus_path=inputs["corpus"],
            tokenizer_path=inputs["tokenizer_root"],
            source_split_sha256s={**source_hashes, "sealed": "0" * 64},
            old_candidate_diagnostic_sha256=_file_sha256(
                residual_attention / "old_candidate_diagnostic.json"
            ),
            corpus_sha256=inputs["corpus_sha256"],
            corpus_line_count=inputs["corpus_count"],
            tokenizer_json_sha256=inputs["tokenizer_sha256"],
            tokenizer_loader=lambda _path: inputs["tokenizer"],
        )
    assert not (tmp_path / "tampered").exists()


def test_builder_source_has_no_runner_trainer_or_model_dependency() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "run_qwen_exact_target_native_stage" not in source
    assert "hf_target_native_behavior_distillation" not in source
    assert "AutoModel" not in source
    assert "import torch" not in source
