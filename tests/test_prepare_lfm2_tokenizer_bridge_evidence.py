from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from experiments import prepare_lfm2_tokenizer_bridge_evidence as PREP  # noqa: E402
from experiments import run_lfm2_dynamic_catalog as CATALOG  # noqa: E402
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


def _write_wordlevel_tokenizer(
    directory: Path,
    vocabulary: dict[str, int],
) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    directory.mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel(
            vocabulary,
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return PREP.residual.load_local_tokenizer(directory)


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(PREP._canonical_json(value) + b"\n")


class _FakeParent:
    def __init__(self, output_token_id: int) -> None:
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.output_token_id = output_token_id
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.training = True
        self.generation_calls = 0

    def parameters(self) -> Any:
        return iter((self.parameter,))

    def eval(self) -> _FakeParent:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _FakeParent:
        self.training = mode
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_calls += 1
        input_ids = kwargs["input_ids"]
        continuation = torch.tensor(
            [[self.output_token_id]],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat((input_ids, continuation), dim=1)


class _FakeCache:
    def __init__(self, path: Path, parent_fingerprint: str) -> None:
        self.path = path
        self.cache_key = "a" * 64
        self.parent_fingerprint = parent_fingerprint
        self.corpus_fingerprint = "c" * 64


class _FakeFingerprint:
    def __init__(self, identity: str) -> None:
        self.identity_sha256 = identity

    def as_dict(self) -> dict[str, Any]:
        return {"identity_sha256": self.identity_sha256}


def _fixture(
    tmp_path: Path,
) -> tuple[
    PREP.BridgeEvidencePreparationConfig,
    Any,
    Any,
    Any,
    _FakeParent,
]:
    donor_directory = tmp_path / "donor"
    target_directory = tmp_path / "target"
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    (parent_directory / "config.json").write_text(
        json.dumps({"model_type": "lfm2", "vocab_size": 6}),
        encoding="utf-8",
    )
    donor = _write_wordlevel_tokenizer(
        donor_directory,
        {
            "<unk>": 0,
            "<bos>": 1,
            "B": 2,
            "<eos>": 3,
            "<pad>": 4,
            "4": 5,
        },
    )
    target = _write_wordlevel_tokenizer(
        target_directory,
        {
            "<bos>": 0,
            "B": 1,
            "4": 2,
            "<eos>": 3,
            "<pad>": 4,
            "<unk>": 5,
        },
    )
    artifact = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_directory,
        donor_row_capacity=6,
        target_row_capacity=6,
    )
    compilation_body = {
        "schema": "neural-forge-lfm2-target-tokenizer-compilation-v1",
        "status": "compiled",
        "target_tokenizer_directory": str(target_directory.resolve()),
        "bridge_artifact_sha256": artifact.artifact_sha256,
        "bridge_profile_sha256": artifact.profile_fingerprint,
        "donor_physical_row_capacity": 6,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "model_parameters_pruned": 0,
        "model_parameters_quantized": 0,
    }
    compilation_receipt = tmp_path / "compilation-receipt.json"
    _write_canonical(
        compilation_receipt,
        {
            **compilation_body,
            "receipt_sha256": PREP._canonical_sha256(compilation_body),
        },
    )
    trajectory_rows = [
        {
            "schema": PREP.TRAJECTORY_TEXT_SCHEMA,
            "id": "selection-0",
            "split": "selection",
            "input_text": "B",
            "forced_decode_text": "4",
        },
        {
            "schema": PREP.TRAJECTORY_TEXT_SCHEMA,
            "id": "fit-0",
            "split": "fit",
            "input_text": "B B",
            "forced_decode_text": "4",
        },
    ]
    trajectory_path = tmp_path / "trajectory-texts.json"
    trajectory_path.write_text(
        json.dumps(trajectory_rows),
        encoding="utf-8",
    )
    original_rows = tmp_path / "original-parent-rows.json"
    original_rows.write_text(
        json.dumps(
            [
                {
                    "id": "math",
                    "prompt": "What is two plus two?",
                    "category": "math",
                    "expected_response": "4",
                    "parent_output_text": "4",
                    "parent_output_ids": [5],
                }
            ]
        ),
        encoding="utf-8",
    )
    original_envelope = tmp_path / "original-parent-envelope.json"
    tokenizer_fingerprint = CATALOG.fingerprint_local_tokenizer(
        donor_directory
    )
    original_receipt = PREP.seal_lfm2_cached_whole_output_envelope(
        original_rows,
        original_envelope,
        parent_cache_key="d" * 64,
        parent_state_fingerprint="b" * 64,
        tokenizer_fingerprint=tokenizer_fingerprint,
        generation_limit=4,
    )
    config = PREP.BridgeEvidencePreparationConfig(
        parent_model=parent_directory,
        donor_tokenizer_directory=donor_directory,
        target_tokenizer_directory=target_directory,
        target_tokenizer_compilation_receipt=compilation_receipt,
        target_tokenizer_compilation_receipt_sha256=PREP._file_sha256(
            compilation_receipt
        ),
        bridge_artifact_sha256=artifact.artifact_sha256,
        bridge_profile_sha256=artifact.profile_fingerprint,
        trajectory_texts=trajectory_path,
        trajectory_texts_sha256=PREP._file_sha256(trajectory_path),
        original_parent_outputs=original_envelope,
        original_parent_outputs_sha256=original_receipt["file_sha256"],
        expected_parent_state_fingerprint="b" * 64,
        output_directory=tmp_path / "prepared",
        device="cpu",
        dtype="float32",
        generation_limit=4,
        max_decode_steps=4,
    )
    return config, donor, target, artifact, _FakeParent(output_token_id=5)


def test_trajectory_texts_are_canonical_for_prompt_and_forced_path(
    tmp_path: Path,
) -> None:
    config, _donor, target, _artifact, _parent = _fixture(tmp_path)
    source_rows = PREP._read_trajectory_texts(config)
    target_rows = PREP._tokenize_trajectory_rows(
        source_rows,
        target,
        max_decode_steps=4,
    )
    path = tmp_path / "target-calibration.json"
    _write_canonical(path, target_rows)
    calibration = PREP.residual.load_calibration_batches(path)

    evidence = CATALOG._verify_target_calibration_canonical_tokenization(
        path,
        calibration,
        target,
    )

    assert len(evidence) == 64
    assert target_rows[0]["target_tokenization_mode"] == (
        "tokenizer-call-add-special-tokens"
    )
    assert target_rows[0]["target_forced_decode_tokenization_mode"] == (
        "tokenizer-call-no-special-tokens"
    )
    assert target_rows[0]["forced_decode_ids"] == [2]


def test_existing_parent_greedy_calibration_text_is_verified_before_conversion(
    tmp_path: Path,
) -> None:
    config, donor, _target, _artifact, _parent = _fixture(tmp_path)
    donor_input = donor.encode("B", add_special_tokens=True)
    rows = [
        {
            "id": "selection-legacy",
            "split": "selection",
            "prompt": "B",
            "input_ids": donor_input,
            "parent_output_text": "4",
            "parent_output_ids": [5],
            "forced_decode_ids": [5],
        },
        {
            "id": "fit-legacy",
            "split": "fit",
            "prompt": "B",
            "input_ids": donor_input,
            "parent_output_text": "4",
            "parent_output_ids": [5],
            "forced_decode_ids": [5],
        },
    ]
    source = tmp_path / "legacy-parent-greedy.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    legacy_config = replace(
        config,
        trajectory_texts=source,
        trajectory_texts_sha256=PREP._file_sha256(source),
    )

    converted = PREP._read_trajectory_texts(
        legacy_config,
        donor_tokenizer=donor,
    )

    assert converted[0]["input_text"] == "B"
    assert converted[0]["forced_decode_text"] == "4"
    assert converted[0]["source_mode"] == (
        "verified-existing-parent-greedy-text"
    )

    rows[1]["input_ids"] = [0]
    source.write_text(json.dumps(rows), encoding="utf-8")
    tampered = replace(
        config,
        trajectory_texts=source,
        trajectory_texts_sha256=PREP._file_sha256(source),
    )
    with pytest.raises(
        PREP.Lfm2TokenizerBridgeEvidenceError,
        match="canonical donor tokenization",
    ):
        PREP._read_trajectory_texts(
            tampered,
            donor_tokenizer=donor,
        )


def test_preflight_validates_inputs_without_parent_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _donor, _target, _artifact, _parent = _fixture(tmp_path)

    def forbidden_parent_load(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight must not load the parent")

    monkeypatch.setattr(
        PREP.residual,
        "load_local_lfm2_model",
        forbidden_parent_load,
    )

    report = PREP.preflight_bridge_evidence_inputs(config)

    assert report["status"] == "preflight_validated"
    assert report["trajectory_row_count"] == 2
    assert report["original_behavior_case_count"] == 1
    assert report["parent_model_loaded"] is False
    assert report["parent_forward_calls"] == 0
    assert report["parent_generation_calls"] == 0
    assert report["output_directory_created"] is False
    assert not config.output_directory.exists()


def test_preflight_rejects_lexical_capacity_bridge_for_physical_model_rows(
    tmp_path: Path,
) -> None:
    config, _donor, _target, _artifact, _parent = _fixture(tmp_path)
    (config.parent_model / "config.json").write_text(
        json.dumps({"model_type": "lfm2", "vocab_size": 7}),
        encoding="utf-8",
    )

    with pytest.raises(
        PREP.Lfm2TokenizerBridgeEvidenceError,
        match="physical vocabulary rows: 6 != 7",
    ):
        PREP.preflight_bridge_evidence_inputs(config)


def test_prepare_builds_once_then_exact_rerun_reuses_without_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _donor, _target, artifact, parent = _fixture(tmp_path)
    calls = {"parent_load": 0, "cache_build": 0, "cache_open": 0}
    cache_holder: dict[str, _FakeCache] = {}

    def load_parent(*args: Any, **kwargs: Any) -> _FakeParent:
        calls["parent_load"] += 1
        return parent

    def fingerprint(_model: Any) -> _FakeFingerprint:
        assert _model.config._attn_implementation == "eager"
        return _FakeFingerprint(config.expected_parent_state_fingerprint)

    def build_cache(
        root: Path,
        _model: Any,
        batches: Any,
        **kwargs: Any,
    ) -> _FakeCache:
        calls["cache_build"] += 1
        assert len(tuple(batches)) == 2
        assert kwargs["config"].capture_causal_circuit_evidence is True
        path = Path(root) / ("a" * 64)
        path.mkdir(parents=True)
        cache = _FakeCache(
            path,
            config.expected_parent_state_fingerprint,
        )
        cache_holder["cache"] = cache
        return cache

    def open_cache(path: Path, *, verify: bool) -> _FakeCache:
        calls["cache_open"] += 1
        assert verify
        assert Path(path) == cache_holder["cache"].path
        return cache_holder["cache"]

    def resolve_runtime(
        bridge_config: Any,
        *,
        donor_tokenizer: Any,
        parent_cache: Any,
        calibration_splits: Any,
    ) -> Any:
        del donor_tokenizer, parent_cache
        calibration = PREP.residual.load_calibration_batches(
            bridge_config.target_tokenizer_calibration
        )
        assert tuple(calibration_splits) == tuple(calibration.splits)
        adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
            artifact,
            CATALOG._target_tokenized_records(calibration),
            expected_artifact_sha256=artifact.artifact_sha256,
        )
        canonical_evidence = (
            CATALOG._verify_target_calibration_canonical_tokenization(
                bridge_config.target_tokenizer_calibration,
                calibration,
                PREP.residual.load_local_tokenizer(
                    config.target_tokenizer_directory
                ),
            )
        )
        return CATALOG.ResolvedProductionTokenizerBridge(
            artifact=artifact,
            target_tokenizer=PREP.residual.load_local_tokenizer(
                config.target_tokenizer_directory
            ),
            target_tokenizer_artifact_fingerprint=(
                CATALOG.fingerprint_local_tokenizer(
                    config.target_tokenizer_directory
                )
            ),
            trace_adapter=adapter,
            target_calibration=calibration,
            paired_input_evidence=(),
            evidence_sha256=adapter.evidence_sha256,
            canonical_tokenization_evidence_sha256=canonical_evidence,
        )

    monkeypatch.setattr(
        PREP.residual,
        "load_local_lfm2_model",
        load_parent,
    )
    monkeypatch.setattr(PREP, "fingerprint_lfm2_parent", fingerprint)
    monkeypatch.setattr(
        PREP,
        "build_lfm2_circuit_trace_cache",
        build_cache,
    )
    monkeypatch.setattr(PREP, "open_lfm2_circuit_trace_cache", open_cache)
    monkeypatch.setattr(
        PREP.catalog,
        "resolve_production_tokenizer_bridge",
        resolve_runtime,
    )

    first = PREP.prepare_bridge_evidence(config)
    second = PREP.prepare_bridge_evidence(config)

    assert first["status"] == "prepared"
    assert first["reused_existing_exact_identity"] is False
    assert first["normalization_control"]["passed"] is True
    assert first["execution_audit"] == {
        "parent_model_loads": 1,
        "parent_execution_phases": 1,
        "parent_trace_cache_build_calls": 1,
        "parent_trace_capture_forward_calls": 8,
        "causal_circuit_installability_evidence_requested": True,
        "parent_behavior_generation_calls": 1,
        "canonical_parent_behavior_generation_calls": 0,
        "parent_outputs_reused_after_seal": True,
        "child_models_constructed": 0,
        "fold_invocations": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_parameters_pruned": 0,
        "model_parameters_quantized": 0,
        "network_access": False,
    }
    assert first["fold_runner_arguments"][
        "--sealed-parent-cache-directory"
    ] == str(cache_holder["cache"].path)
    assert second["reused_existing_exact_identity"] is True
    assert second["parent_model_loads_this_invocation"] == 0
    assert second["parent_generation_calls_this_invocation"] == 0
    assert calls == {
        "parent_load": 1,
        "cache_build": 1,
        "cache_open": 1,
    }
    assert parent.generation_calls == 1


def test_existing_partial_output_fails_closed(tmp_path: Path) -> None:
    config, _donor, _target, _artifact, _parent = _fixture(tmp_path)
    config.output_directory.mkdir()

    with pytest.raises(
        PREP.Lfm2TokenizerBridgeEvidenceError,
        match="partial",
    ):
        PREP.prepare_bridge_evidence(config)


def test_production_preparation_cannot_disable_causal_circuit_evidence(
    tmp_path: Path,
) -> None:
    config, _donor, _target, _artifact, _parent = _fixture(tmp_path)

    with pytest.raises(
        PREP.Lfm2TokenizerBridgeEvidenceError,
        match="capture_causal_circuit_evidence=True",
    ):
        replace(
            config,
            capture_causal_circuit_evidence=False,
        )
