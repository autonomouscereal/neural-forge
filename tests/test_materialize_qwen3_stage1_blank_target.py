from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "materialize_qwen3_stage1_blank_target.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_qwen3_stage1_blank_target", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_materialization(staged: Path) -> dict:
    staged.mkdir()
    (staged / "config.json").write_text(
        json.dumps({"model_type": "qwen3", **MODULE.CANONICAL_CONFIG}) + "\n",
        encoding="utf-8",
    )
    (staged / "model.safetensors").write_bytes(b"fake-stage-one-weights")
    (staged / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    report = {
        "seed": MODULE.CANONICAL_SEED,
        "torch_dtype": MODULE.CANONICAL_DTYPE,
        "shape": dict(MODULE.CANONICAL_CONFIG),
        "parameter_count": {
            "formula": MODULE.EXACT_PARAMETER_COUNT,
            "actual_unique_model_parameters": MODULE.EXACT_PARAMETER_COUNT,
            "matches": True,
        },
        "random_initialization": {
            "pretrained_child_weights_loaded": False,
            "pretrained_model_loader_called": False,
            "state_dict_sha256": "a" * 64,
        },
        "training": {
            "performed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "tokenizer": {"length": 151_669},
    }
    (staged / MODULE.TARGET_REPORT_FILENAME).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _patch_lightweight_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace()
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    def materialize(_plan, staged, _tokenizer, **kwargs):
        assert kwargs == {
            "seed": 0,
            "torch_dtype": "float16",
            "safe_serialization": True,
        }
        return _fake_materialization(Path(staged))

    monkeypatch.setattr(MODULE, "materialize_qwen3_target", materialize)
    monkeypatch.setattr(
        MODULE,
        "_fresh_reload_certificate",
        lambda staged, report: {
            "performed": True,
            "checkpoint_source": str(staged),
            "exact_parameter_count": MODULE.EXACT_PARAMETER_COUNT,
            "constructor_state_dict_sha256": report["random_initialization"][
                "state_dict_sha256"
            ],
            "constructor_and_reload_state_exact": True,
        },
    )


def test_canonical_plan_is_exactly_the_documented_stage_one_shape() -> None:
    plan = MODULE.canonical_plan()

    assert plan["parameter_count"] == 1_410_567_168
    assert plan["config"] == {
        "vocab_size": 151_936,
        "hidden_size": 1_536,
        "intermediate_size": 5_120,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "attention_bias": False,
    }
    assert MODULE.CANONICAL_SEED == 0
    assert MODULE.CANONICAL_DTYPE == "float16"


@pytest.mark.parametrize("occupied", ["staging", "destination", "receipt"])
def test_existing_singleton_path_fails_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied: str,
) -> None:
    staging_root = tmp_path / "staging-root"
    artifact_root = tmp_path / "artifact-root"
    staging_root.mkdir()
    artifact_root.mkdir()
    paths = {
        "staging": staging_root / MODULE.CANONICAL_STAGING_NAME,
        "destination": artifact_root / MODULE.CANONICAL_ARTIFACT_NAME,
        "receipt": artifact_root
        / f"{MODULE.CANONICAL_ARTIFACT_NAME}{MODULE.RECEIPT_SUFFIX}",
    }
    paths[occupied].mkdir() if occupied != "receipt" else paths[occupied].write_text(
        "occupied", encoding="utf-8"
    )
    called = []
    monkeypatch.setattr(
        MODULE,
        "materialize_qwen3_target",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(MODULE.Stage1BlankArtifactError, match="must be absent"):
        MODULE.materialize_stage1_blank(
            staging_root=staging_root,
            artifact_root=artifact_root,
            tokenizer_model="cached-tokenizer",
        )

    assert called == []


def test_verified_copy_publication_removes_staging_only_after_tree_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging-root"
    artifact_root = tmp_path / "artifact-root"
    staging_root.mkdir()
    artifact_root.mkdir()
    _patch_lightweight_materialization(monkeypatch)
    original_replace = MODULE.os.replace

    def fail_directory_replace(source, destination):
        if Path(source).name == MODULE.CANONICAL_STAGING_NAME:
            raise OSError(errno.EXDEV, "cross-device")
        return original_replace(source, destination)

    monkeypatch.setattr(MODULE.os, "replace", fail_directory_replace)

    receipt = MODULE.materialize_stage1_blank(
        staging_root=staging_root,
        artifact_root=artifact_root,
        tokenizer_model="cached-tokenizer",
    )

    destination = artifact_root / MODULE.CANONICAL_ARTIFACT_NAME
    staging = staging_root / MODULE.CANONICAL_STAGING_NAME
    receipt_path = artifact_root / (
        MODULE.CANONICAL_ARTIFACT_NAME + MODULE.RECEIPT_SUFFIX
    )
    assert receipt["publication"]["mode"] == "verified-copy-fallback"
    assert receipt["fallback_staging_removed_after_verification"] is True
    assert destination.is_dir()
    assert not staging.exists()
    assert not (destination / ".neural-forge-promotion-incomplete").exists()
    assert (destination / MODULE.CERTIFICATE_FILENAME).is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert MODULE._artifact_tree(destination) == receipt["complete_artifact_tree"]


def test_atomic_publication_moves_staging_and_writes_complete_tree_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging-root"
    artifact_root = tmp_path / "artifact-root"
    staging_root.mkdir()
    artifact_root.mkdir()
    _patch_lightweight_materialization(monkeypatch)

    receipt = MODULE.materialize_stage1_blank(
        staging_root=staging_root,
        artifact_root=artifact_root,
        tokenizer_model="cached-tokenizer",
    )

    destination = artifact_root / MODULE.CANONICAL_ARTIFACT_NAME
    assert receipt["publication"]["mode"] == "atomic-replace"
    assert receipt["fallback_staging_removed_after_verification"] is False
    assert not (staging_root / MODULE.CANONICAL_STAGING_NAME).exists()
    assert MODULE._artifact_tree(destination) == receipt["complete_artifact_tree"]


def test_known_existing_staging_is_hash_bound_and_never_rematerialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging-root"
    artifact_root = tmp_path / "artifact-root"
    staging_root.mkdir()
    artifact_root.mkdir()
    staged = staging_root / MODULE.KNOWN_EXISTING_STAGING_NAME
    report = _fake_materialization(staged)
    model_sha256 = MODULE._file_sha256(staged / "model.safetensors")
    monkeypatch.setattr(MODULE, "KNOWN_EXISTING_MODEL_SHA256", model_sha256)
    monkeypatch.setattr(
        MODULE,
        "materialize_qwen3_target",
        lambda *_args, **_kwargs: pytest.fail("must not rematerialize"),
    )
    monkeypatch.setattr(
        MODULE,
        "_fresh_reload_certificate",
        lambda path, loaded: {
            "performed": True,
            "checkpoint_source": str(path),
            "constructor_state_dict_sha256": loaded["random_initialization"][
                "state_dict_sha256"
            ],
            "constructor_and_reload_state_exact": True,
        },
    )

    receipt = MODULE.materialize_stage1_blank(
        staging_root=staging_root,
        artifact_root=artifact_root,
        tokenizer_model="not-used",
        existing_staged=staged,
    )

    assert report["parameter_count"]["formula"] == MODULE.EXACT_PARAMETER_COUNT
    assert receipt["source_mode"] == "verify-and-publish-known-existing-staging"
    assert receipt["known_existing_model_safetensors_sha256"] == model_sha256
    assert not staged.exists()


def test_known_existing_staging_rejects_changed_model_before_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging-root"
    artifact_root = tmp_path / "artifact-root"
    staging_root.mkdir()
    artifact_root.mkdir()
    staged = staging_root / MODULE.KNOWN_EXISTING_STAGING_NAME
    _fake_materialization(staged)
    monkeypatch.setattr(MODULE, "KNOWN_EXISTING_MODEL_SHA256", "0" * 64)
    called = []
    monkeypatch.setattr(
        MODULE,
        "_fresh_reload_certificate",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(MODULE.Stage1BlankArtifactError, match="recorded SHA-256"):
        MODULE.materialize_stage1_blank(
            staging_root=staging_root,
            artifact_root=artifact_root,
            tokenizer_model="not-used",
            existing_staged=staged,
        )

    assert called == []
    assert staged.is_dir()


def test_changed_fallback_staging_is_never_deleted(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "stage"
    staged.mkdir()
    (staged / "weights").write_bytes(b"before")
    expected = MODULE._artifact_tree(staged)
    (staged / "weights").write_bytes(b"after")

    with pytest.raises(MODULE.Stage1BlankArtifactError, match="tree changed"):
        MODULE._remove_verified_staging(
            staged,
            expected_tree=expected,
            publication={"mode": "verified-copy-fallback"},
        )

    assert staged.is_dir()
    assert (staged / "weights").read_bytes() == b"after"


def test_remote_wrapper_is_foreground_offline_cpu_and_canonical() -> None:
    wrapper = (
        Path(__file__).parents[1]
        / "experiments"
        / "remote_materialize_qwen3_stage1_blank_20260712.sh"
    ).read_text(encoding="utf-8")

    assert "neural-forge-src" in wrapper
    assert "neural-forge-checkpoint-staging" in wrapper
    assert "/mnt/ai_data/neural-forge/canonical_blank_targets" in wrapper
    assert 'export CUDA_VISIBLE_DEVICES=""' in wrapper
    assert "HF_HUB_OFFLINE=1" in wrapper
    assert 'exec "$PYTHON"' in wrapper
    assert "nohup" not in wrapper
    assert "--local-files-only" in wrapper
    assert "--existing-staged" in wrapper
    assert MODULE.KNOWN_EXISTING_STAGING_NAME in wrapper


def test_module_reuses_existing_certification_and_publication_primitives() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from neural_forge.hf_blank_checkpoint import (" in source
    assert "_parameter_counts" in source
    assert "_state_dict_sha256" in source
    assert "from experiments.replay_qwen_unified_v8_commit import (" in source
    assert "_publish_staged_checkpoint" in source
    assert "_artifact_tree" in source
    assert "pydantic" not in source.lower()
    assert "sqlalchemy" not in source.lower()
