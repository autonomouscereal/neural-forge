from __future__ import annotations

import errno
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    save_qwen3_codebook_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "replay_qwen_unified_v8_commit.py"
SPEC = importlib.util.spec_from_file_location("replay_qwen_unified_v8_commit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, size: int = 32) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _runtime(seed: int = 941):
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
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
    tokenizer = _Tokenizer()
    runtime = prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=2,
        neuron_prototypes_per_neuron=2,
    )
    rows = [int(item) for item in runtime.parent_token_rows.tolist()]
    contract = {
        "tokenizer_name_or_path": "tiny-replay-test",
        "vocab_size": len(tokenizer),
        "vocab_sha256": hashlib.sha256(
            MODULE._canonical_json(tokenizer.get_vocab())
        ).hexdigest(),
        "special_token_ids": {
            name: getattr(runtime.config, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": hashlib.sha256(
            MODULE._canonical_json(rows)
        ).hexdigest(),
    }
    return runtime, contract


def test_real_v8_evidence_bundle_binds_exact_selected_transaction() -> None:
    proof = MODULE.validate_v8_evidence_bundle(
        ROOT / "runs" / "unified-v8-checkpoint-failure-20260711"
    )

    assert proof.redirects == MODULE.V8_REDIRECTS
    assert (
        proof.selected_proposal_sha256s
        == MODULE.EXPECTED_SELECTED_VALIDATED_PROPOSAL_SHA256S
    )
    assert (
        proof.proposed_accounting["neuron_registry_sha256"]
        == MODULE.EXPECTED_NEURON_REGISTRY_SHA256
    )
    assert proof.proposed_accounting["child_unique_parameters"] == 3_995_516_416
    assert proof.proposed_accounting["deployable_bytes"] == 7_994_478_724


def test_v8_evidence_bundle_rejects_one_byte_artifact_drift(tmp_path: Path) -> None:
    source = ROOT / "runs" / "unified-v8-checkpoint-failure-20260711"
    copied = tmp_path / "evidence"
    shutil.copytree(source, copied)
    gate = copied / (
        "candidate-000001-"
        f"{MODULE.GATE_CANDIDATE_STATE_SHA256}.gate.json"
    )
    gate.write_bytes(gate.read_bytes() + b" ")

    with pytest.raises(MODULE.QwenUnifiedV8ReplayError, match="SHA-256 mismatch"):
        MODULE.validate_v8_evidence_bundle(copied)


def test_exact_mapping_diff_reports_values_and_types_without_tolerance() -> None:
    assert MODULE._mapping_diff({"same": 1}, {"same": 1}) == {}
    assert MODULE._mapping_diff(
        {"same_number_wrong_type": 1, "changed": "old", "missing": 3},
        {"same_number_wrong_type": 1.0, "changed": "new", "extra": 4},
    ) == {
        "changed": {
            "expected": "old",
            "observed": "new",
            "expected_type": "str",
            "observed_type": "str",
        },
        "extra": {
            "expected": "<missing>",
            "observed": 4,
            "expected_type": "str",
            "observed_type": "int",
        },
        "missing": {
            "expected": 3,
            "observed": "<missing>",
            "expected_type": "int",
            "observed_type": "str",
        },
        "same_number_wrong_type": {
            "expected": 1,
            "observed": 1.0,
            "expected_type": "int",
            "observed_type": "float",
        },
    }


def test_exact_replay_rolls_back_uncommitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _contract = _runtime()
    before = MODULE.model_content_identity(runtime)
    before_snapshot = MODULE.neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
    before_report = getattr(runtime, MODULE.surgery_allocator._REPORT_ATTRIBUTE)
    original_sync = MODULE.surgery_allocator._sync_parameter_report
    synchronized_reports = []

    def track_single_sync(value):
        result = original_sync(value)
        synchronized_reports.append(result)
        return result

    monkeypatch.setattr(
        MODULE.surgery_allocator,
        "_sync_parameter_report",
        track_single_sync,
    )

    with MODULE.ExactNeuronMedoidReplay(runtime, ((1, 4, 3),)):
        staged = MODULE.neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
        assert staged.prototype_count == before_snapshot.prototype_count - 1
        assert staged.registry_sha256 != before_snapshot.registry_sha256
        assert len(synchronized_reports) == 1
        # Ragged per-neuron prototype depth is encoded exactly as the original
        # allocator did on its one synchronization call.  Calling the private
        # sync twice is intentionally not used as an assertion: it normalizes
        # this field from None to 0 and would no longer replay v8 exactly.
        assert synchronized_reports[0].neuron_prototypes_per_neuron == 0

    assert MODULE.model_content_identity(runtime) == before
    assert (
        MODULE.neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
        == before_snapshot
    )
    assert getattr(runtime, MODULE.surgery_allocator._REPORT_ATTRIBUTE) == before_report


def test_exact_replay_save_reload_content_accounting_and_parent_free_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, contract = _runtime(947)
    original_sync = MODULE.surgery_allocator._sync_parameter_report
    synchronized_reports = []

    def track_sync(value):
        result = original_sync(value)
        synchronized_reports.append(result)
        return result

    monkeypatch.setattr(MODULE.surgery_allocator, "_sync_parameter_report", track_sync)
    with MODULE.ExactNeuronMedoidReplay(runtime, ((1, 4, 3),)) as transaction:
        expected_content = MODULE.model_content_identity(runtime)
        general_report_hash = MODULE._sha256(
            MODULE.qwen3_codebook_parameter_report(runtime).as_dict()
        )
        expected_accounting = MODULE.exact_accounting(runtime, contract)
        assert len(synchronized_reports) == 2
        synchronized_report = synchronized_reports[-1]
        assert expected_accounting.parameter_report_sha256 == MODULE._sha256(
            synchronized_report.as_dict()
        )
        assert expected_accounting.parameter_report_sha256 != general_report_hash
        checkpoint = tmp_path / "child"
        save_qwen3_codebook_checkpoint(
            runtime,
            checkpoint,
            tokenizer_contract=contract,
        )
        reloaded = load_qwen3_codebook_checkpoint(checkpoint, device="cpu")
        assert MODULE.model_content_identity(reloaded) == expected_content
        assert MODULE.exact_accounting(reloaded, contract) == expected_accounting
        smoke = MODULE._parent_free_smoke(reloaded)
        assert smoke["parent_model_loaded"] is False
        assert smoke["first_step_finite"] is True
        assert smoke["cached_step_finite"] is True
        transaction.commit()

    assert MODULE.model_content_identity(runtime) == expected_content


@pytest.mark.parametrize(
    "rename_error",
    [
        PermissionError(errno.EACCES, "CIFS denied directory rename"),
        OSError(errno.EXDEV, "cross-device directory rename"),
    ],
)
def test_checkpoint_publish_falls_back_to_marked_verified_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rename_error: OSError,
) -> None:
    staged = tmp_path / "staged"
    (staged / "nested").mkdir(parents=True)
    (staged / "model.safetensors").write_bytes(b"weights" * 64)
    (staged / "nested" / "manifest.json").write_text(
        '{"schema":"test"}\n', encoding="utf-8"
    )
    destination = tmp_path / "published"
    expected = MODULE._artifact_tree(staged)
    original_copytree = MODULE.shutil.copytree
    marker_seen_during_copy = []

    def fail_rename(_source, _destination):
        raise rename_error

    def observe_marker(source, target, *args, **kwargs):
        if Path(target) == destination:
            marker_seen_during_copy.append(
                (Path(target) / MODULE._INCOMPLETE_PROMOTION_MARKER).is_file()
            )
        return original_copytree(source, target, *args, **kwargs)

    monkeypatch.setattr(MODULE.os, "replace", fail_rename)
    monkeypatch.setattr(MODULE.shutil, "copytree", observe_marker)

    promotion = MODULE._publish_staged_checkpoint(
        staged, destination, expected_tree=expected
    )

    assert promotion["mode"] == "verified-copy-fallback"
    assert promotion["verified_copy_fallback"] is True
    assert promotion["atomic_directory_replace"] is False
    assert promotion["staged_tree_sha256"] == expected["tree_sha256"]
    assert promotion["published_tree_sha256"] == expected["tree_sha256"]
    assert promotion["total_bytes"] == expected["total_bytes"]
    assert marker_seen_during_copy == [True]
    assert not (destination / MODULE._INCOMPLETE_PROMOTION_MARKER).exists()
    assert MODULE._artifact_tree(destination) == expected
    assert MODULE._artifact_tree(staged) == expected


def test_checkpoint_copy_corruption_fails_and_removes_only_new_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.safetensors").write_bytes(b"exact-weights")
    destination = tmp_path / "published"
    sibling = tmp_path / "user-owned"
    sibling.mkdir()
    marker = sibling / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    original_copytree = MODULE.shutil.copytree

    def deny_rename(_source, _destination):
        raise PermissionError(errno.EACCES, "CIFS denied directory rename")

    def corrupt_copy(source, target, **kwargs):
        result = original_copytree(source, target, **kwargs)
        (Path(target) / "model.safetensors").write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(MODULE.os, "replace", deny_rename)
    monkeypatch.setattr(MODULE.shutil, "copytree", corrupt_copy)

    with pytest.raises(
        MODULE.QwenUnifiedV8ReplayError,
        match="differs from the staged artifact tree",
    ):
        MODULE._publish_staged_checkpoint(staged, destination)

    assert not destination.exists()
    assert (staged / "model.safetensors").read_bytes() == b"exact-weights"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_checkpoint_partial_copy_exception_is_scoped_and_nonfallback_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.safetensors").write_bytes(b"exact-weights")
    destination = tmp_path / "published"

    def deny_rename(_source, _destination):
        raise PermissionError(errno.EACCES, "CIFS denied directory rename")

    def partial_copy(_source, target, **_kwargs):
        (Path(target) / "partial.bin").write_bytes(b"partial")
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(MODULE.os, "replace", deny_rename)
    monkeypatch.setattr(MODULE.shutil, "copytree", partial_copy)

    with pytest.raises(OSError, match="copy failed"):
        MODULE._publish_staged_checkpoint(staged, destination)
    assert not destination.exists()
    assert staged.is_dir()

    def fail_without_fallback(_source, _destination):
        raise OSError(errno.EIO, "rename I/O failure")

    monkeypatch.setattr(MODULE.os, "replace", fail_without_fallback)
    with pytest.raises(OSError, match="rename I/O failure"):
        MODULE._publish_staged_checkpoint(staged, destination)
    assert not destination.exists()
