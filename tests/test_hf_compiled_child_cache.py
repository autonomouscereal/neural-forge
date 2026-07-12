from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge.hf_compiled_child_cache import (
    CACHE_RECEIPT_FILENAME,
    COMPILER_REPORT_FILENAME,
    CompiledChildCacheIntegrityError,
    CompiledChildCachePublicationError,
    CompilerIdentity,
    CompilerIdentityError,
    load_compiled_child_cache,
    model_state_fingerprint,
    publish_compiled_child_cache,
    verify_compiled_child_cache,
)


class _TinyHFModel(nn.Module):
    load_calls = 0

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.arange(width * 3, dtype=torch.float32).reshape(3, width) / 7.0
        )
        self.register_buffer("scale", torch.linspace(0.5, 1.5, width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weight.T

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps({"width": self.width}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls, source: Path, *, local_files_only: bool = True
    ) -> _TinyHFModel:
        assert local_files_only is True
        cls.load_calls += 1
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model = cls(width=int(config["width"]))
        state = load_file(source / "model.safetensors", device="cpu")
        model.load_state_dict(state, strict=True)
        return model


def _sha(character: str) -> str:
    return character * 64


def _identity(*, parent: str = "a", option: float = 0.1) -> CompilerIdentity:
    return CompilerIdentity(
        parent_artifact_tree_sha256=_sha(parent),
        blank_target_artifact_tree_sha256=_sha("b"),
        tokenizer_sha256=_sha("c"),
        compiler_input_sha256s={
            "canonical_validation_partition": _sha("e"),
            "fit_partition": _sha("d"),
        },
        compiler_implementation_sha256s={
            "hf_blank_bootstrap.py": _sha("2"),
            "hf_parent_compiler.py": _sha("1"),
        },
        compiler_options={
            "bootstrap_kwargs": {
                "heldout_gate_policy": "output_only",
                "max_operator_delta_ratio": option,
            },
            "capture_batch_size": 1,
        },
        runtime={
            "attention_implementation": "eager",
            "dtype": "float16",
            "torch_version": "2.13.0",
            "transformers_version": "4.56.0",
        },
        seed=0,
    )


def _cache_path(root: Path, identity: CompilerIdentity) -> Path:
    return root / identity.compiler_identity_sha256


def _report(model: nn.Module) -> dict[str, object]:
    state = model_state_fingerprint(model)
    count = state["unique_parameter_count"]
    return {
        "schema": "neural-forge-hf-parent-compiler-v1",
        "bootstrap_accepted": True,
        "target_budget_exact": True,
        "target_unique_parameter_count": count,
        "target_logical_parameter_count": state["logical_parameter_count"],
        "target_parameter_budget": count,
        "target_state_sha256_after": state["compiler_state_sha256"],
        "execution_graph_fingerprint": _sha("9"),
        "capability_proven": False,
    }


def test_compiler_identity_is_canonical_and_rejects_trainer_dependencies() -> None:
    first = _identity()
    second = CompilerIdentity(
        parent_artifact_tree_sha256=_sha("a"),
        blank_target_artifact_tree_sha256=_sha("b"),
        tokenizer_sha256=_sha("c"),
        compiler_input_sha256s={
            "fit_partition": _sha("d"),
            "canonical_validation_partition": _sha("e"),
        },
        compiler_implementation_sha256s={
            "hf_parent_compiler.py": _sha("1"),
            "hf_blank_bootstrap.py": _sha("2"),
        },
        compiler_options={
            "capture_batch_size": 1,
            "bootstrap_kwargs": {
                "max_operator_delta_ratio": 0.1,
                "heldout_gate_policy": "output_only",
            },
        },
        runtime={
            "transformers_version": "4.56.0",
            "torch_version": "2.13.0",
            "dtype": "float16",
            "attention_implementation": "eager",
        },
        seed=0,
    )
    assert first.as_dict() == second.as_dict()
    assert first.as_dict()["trainer_identity_included"] is False
    assert first.compiler_identity_sha256 == second.compiler_identity_sha256
    assert CompilerIdentity.from_dict(first.as_dict()).as_dict() == first.as_dict()

    with pytest.raises(CompilerIdentityError, match="trainer-dependent"):
        CompilerIdentity(
            parent_artifact_tree_sha256=_sha("a"),
            blank_target_artifact_tree_sha256=_sha("b"),
            tokenizer_sha256=_sha("c"),
            compiler_input_sha256s={"fit": _sha("d")},
            compiler_implementation_sha256s={"trainer_sha256": _sha("1")},
            compiler_options={},
            runtime={"dtype": "float16"},
            seed=0,
        )
    with pytest.raises(CompilerIdentityError, match="trainer-dependent"):
        CompilerIdentity(
            parent_artifact_tree_sha256=_sha("a"),
            blank_target_artifact_tree_sha256=_sha("b"),
            tokenizer_sha256=_sha("c"),
            compiler_input_sha256s={"fit": _sha("d")},
            compiler_implementation_sha256s={"compiler": _sha("1")},
            compiler_options={
                "module": "neural_forge.hf_target_native_behavior_distillation"
            },
            runtime={"dtype": "float16"},
            seed=0,
        )
    with pytest.raises(CompilerIdentityError, match="trainer-dependent"):
        CompilerIdentity(
            parent_artifact_tree_sha256=_sha("a"),
            blank_target_artifact_tree_sha256=_sha("b"),
            tokenizer_sha256=_sha("c"),
            compiler_input_sha256s={"fit": _sha("d")},
            compiler_implementation_sha256s={"compiler": _sha("1")},
            compiler_options={"training_schedule": {"steps": 24}},
            runtime={"dtype": "float16"},
            seed=0,
        )


def test_atomic_publish_verify_and_load_round_trip(tmp_path: Path) -> None:
    _TinyHFModel.load_calls = 0
    model = _TinyHFModel()
    identity = _identity()
    report = _report(model)
    cache = _cache_path(tmp_path, identity)

    receipt = publish_compiled_child_cache(
        cache,
        model=model,
        compiler_report=report,
        compiler_identity=identity,
    )
    assert cache.is_dir()
    assert (cache / CACHE_RECEIPT_FILENAME).is_file()
    assert (cache / COMPILER_REPORT_FILENAME).is_file()
    assert _TinyHFModel.load_calls == 1
    payload = receipt.as_dict()
    assert payload["integrity_verified_immutable_by_contract"] is True
    assert payload["fresh_reload_verified"] is True
    assert payload["trainer_identity_included"] is False
    assert payload["compiler_identity_sha256"] == (identity.compiler_identity_sha256)
    assert payload["model_state"] == model_state_fingerprint(model)
    assert payload["artifact_tree"]["file_count"] == 3
    assert payload["artifact_tree"]["receipt_excluded_from_tree"] is True

    verified = verify_compiled_child_cache(
        cache,
        expected_identity=identity,
        expected_receipt_sha256=receipt.receipt_sha256,
        expected_compiler_report_sha256=payload["compiler_report"]["canonical_sha256"],
        expected_state_dict_sha256=payload["model_state"]["state_dict_sha256"],
    )
    assert dict(verified.compiler_report) == report
    loaded = load_compiled_child_cache(
        cache,
        expected_identity=identity,
        model_loader=lambda path: _TinyHFModel.from_pretrained(path),
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert model_state_fingerprint(loaded.model) == model_state_fingerprint(model)
    assert loaded.receipt.as_dict() == receipt.as_dict()
    assert _TinyHFModel.load_calls == 2

    before = (cache / CACHE_RECEIPT_FILENAME).read_bytes()
    with pytest.raises(CompiledChildCachePublicationError, match="already exists"):
        publish_compiled_child_cache(
            cache,
            model=model,
            compiler_report=report,
            compiler_identity=identity,
        )
    assert (cache / CACHE_RECEIPT_FILENAME).read_bytes() == before


@pytest.mark.parametrize("target", ["model.safetensors", COMPILER_REPORT_FILENAME])
def test_file_or_report_tamper_fails_before_model_loader(
    tmp_path: Path, target: str
) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)
    publish_compiled_child_cache(
        cache,
        model=model,
        compiler_report=_report(model),
        compiler_identity=identity,
    )
    path = cache / target
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)
    calls = 0

    def forbidden_loader(_path: Path) -> nn.Module:
        nonlocal calls
        calls += 1
        return _TinyHFModel()

    with pytest.raises(CompiledChildCacheIntegrityError):
        load_compiled_child_cache(
            cache,
            expected_identity=identity,
            model_loader=forbidden_loader,
        )
    assert calls == 0


def test_extra_file_and_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)
    publish_compiled_child_cache(
        cache,
        model=model,
        compiler_report=_report(model),
        compiler_identity=identity,
    )
    calls = 0

    def loader(_path: Path) -> nn.Module:
        nonlocal calls
        calls += 1
        return _TinyHFModel()

    with pytest.raises(CompiledChildCacheIntegrityError, match="identity"):
        load_compiled_child_cache(
            cache,
            expected_identity=_identity(parent="7"),
            model_loader=loader,
        )
    assert calls == 0

    (cache / "payload.bin").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(CompiledChildCacheIntegrityError, match="unapproved"):
        load_compiled_child_cache(
            cache,
            expected_identity=identity,
            model_loader=loader,
        )
    assert calls == 0

    (cache / "payload.bin").unlink()
    (cache / "unexpected-empty-directory").mkdir()
    with pytest.raises(CompiledChildCacheIntegrityError, match="must be flat"):
        verify_compiled_child_cache(cache, expected_identity=identity)


def test_loader_state_mismatch_is_rejected_after_metadata_verification(
    tmp_path: Path,
) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)
    publish_compiled_child_cache(
        cache,
        model=model,
        compiler_report=_report(model),
        compiler_identity=identity,
    )

    def wrong_loader(path: Path) -> nn.Module:
        loaded = _TinyHFModel.from_pretrained(path)
        with torch.no_grad():
            loaded.weight.add_(1.0)
        return loaded

    with pytest.raises(CompiledChildCacheIntegrityError, match="cached state"):
        load_compiled_child_cache(
            cache,
            expected_identity=identity,
            model_loader=wrong_loader,
        )


@pytest.mark.parametrize("tamper", ["content", "encoding"])
def test_receipt_tamper_is_rejected(tmp_path: Path, tamper: str) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)
    publish_compiled_child_cache(
        cache,
        model=model,
        compiler_report=_report(model),
        compiler_identity=identity,
    )
    receipt_path = cache / CACHE_RECEIPT_FILENAME
    if tamper == "content":
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["integrity_verified_immutable_by_contract"] = False
        receipt_path.write_text(
            json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    with pytest.raises(
        CompiledChildCacheIntegrityError, match="self-hash|canonical byte"
    ):
        verify_compiled_child_cache(cache, expected_identity=identity)


def test_fresh_reload_mismatch_never_promotes_staging(tmp_path: Path) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)

    def wrong_loader(path: Path) -> nn.Module:
        loaded = _TinyHFModel.from_pretrained(path)
        with torch.no_grad():
            loaded.weight.mul_(2.0)
        return loaded

    with pytest.raises(CompiledChildCachePublicationError, match="Freshly reloaded"):
        publish_compiled_child_cache(
            cache,
            model=model,
            compiler_report=_report(model),
            compiler_identity=identity,
            model_loader=wrong_loader,
        )
    assert not cache.exists()
    assert not list(tmp_path.glob(f".{identity.compiler_identity_sha256}.*.staging"))
    assert not list(tmp_path.glob(".*.compile-cache.lock"))


def test_failed_staging_is_removed_and_destination_never_appears(
    tmp_path: Path,
) -> None:
    model = _TinyHFModel()
    identity = _identity()
    cache = _cache_path(tmp_path, identity)

    def broken_saver(_model: nn.Module, staging: Path) -> None:
        (staging / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("injected save failure")

    with pytest.raises(CompiledChildCachePublicationError, match="atomic promotion"):
        publish_compiled_child_cache(
            cache,
            model=model,
            compiler_report=_report(model),
            compiler_identity=identity,
            model_saver=broken_saver,
            model_loader=lambda _path: copy.deepcopy(model),
        )
    assert not cache.exists()
    assert not list(tmp_path.glob(f".{identity.compiler_identity_sha256}.*.staging"))
    assert not list(tmp_path.glob(".*.compile-cache.lock"))


def test_rejected_or_nonfinite_child_cannot_be_cached(tmp_path: Path) -> None:
    model = _TinyHFModel()
    identity = _identity()
    rejected = _report(model)
    rejected["bootstrap_accepted"] = False
    cache = _cache_path(tmp_path, identity)
    with pytest.raises(CompiledChildCacheIntegrityError, match="accepted"):
        publish_compiled_child_cache(
            cache,
            model=model,
            compiler_report=rejected,
            compiler_identity=identity,
        )
    assert not cache.exists()

    poisoned = copy.deepcopy(model)
    with torch.no_grad():
        poisoned.weight.reshape(-1)[0] = torch.nan
    with pytest.raises(CompiledChildCacheIntegrityError, match="NaN"):
        publish_compiled_child_cache(
            cache,
            model=poisoned,
            compiler_report=_report(model),
            compiler_identity=identity,
        )
    assert not cache.exists()


def test_publication_requires_identity_path_and_never_removes_foreign_lock(
    tmp_path: Path,
) -> None:
    model = _TinyHFModel()
    identity = _identity()
    with pytest.raises(CompiledChildCachePublicationError, match="identity SHA-256"):
        publish_compiled_child_cache(
            tmp_path / "arbitrary-name",
            model=model,
            compiler_report=_report(model),
            compiler_identity=identity,
        )
    cache = _cache_path(tmp_path, identity)
    lock = tmp_path / f".{identity.compiler_identity_sha256}.compile-cache.lock"
    foreign = b"foreign-publisher-token\n"
    lock.write_bytes(foreign)
    with pytest.raises(CompiledChildCachePublicationError, match="lock already exists"):
        publish_compiled_child_cache(
            cache,
            model=model,
            compiler_report=_report(model),
            compiler_identity=identity,
        )
    assert lock.read_bytes() == foreign
    assert not cache.exists()


def test_same_size_different_model_cannot_reuse_compiler_report(tmp_path: Path) -> None:
    source = _TinyHFModel()
    different = copy.deepcopy(source)
    with torch.no_grad():
        different.weight.add_(0.5)
    identity = _identity()
    cache = _cache_path(tmp_path, identity)
    with pytest.raises(CompiledChildCacheIntegrityError, match="state hash"):
        publish_compiled_child_cache(
            cache,
            model=different,
            compiler_report=_report(source),
            compiler_identity=identity,
        )
    assert not cache.exists()
