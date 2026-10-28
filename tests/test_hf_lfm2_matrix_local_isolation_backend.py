from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neural_forge import hf_lfm2_matrix_local_isolation_backend as backend


class _FakeEvidence:
    selection_sha256 = "s" * 64
    fit_row_count = 7
    heldout_row_count = 3


class _FakeKey:
    phase_key_sha256 = "k" * 64


class _FakeEntry:
    def __init__(self, payload_path: Path, result: dict[str, object]) -> None:
        self.payload_path = payload_path
        self.result = result
        self.completion_sha256 = "c" * 64


class _FakeTransaction:
    def __init__(self, payload_path: Path) -> None:
        self.payload_path = payload_path

    def __enter__(self) -> _FakeTransaction:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def complete(self, *, result: dict[str, object]) -> _FakeEntry:
        return _FakeEntry(self.payload_path, result)


class _FakePhaseCache:
    def __init__(self, payload_path: Path) -> None:
        self.payload_path = payload_path

    def lookup(self, _key: object) -> None:
        return None

    def begin(self, _key: object) -> _FakeTransaction:
        return _FakeTransaction(self.payload_path)


def test_site_evidence_loader_selects_balanced_basis_split_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    evidence = _FakeEvidence()
    binding = {
        "selection_sha256": evidence.selection_sha256,
        "parent_identity_sha256": "p" * 64,
    }

    def load_site_evidence(**kwargs: object) -> tuple[_FakeEvidence, dict[str, str]]:
        observed.update(kwargs)
        return evidence, binding

    monkeypatch.setattr(
        backend,
        "_runner_module",
        lambda: SimpleNamespace(load_site_evidence=load_site_evidence),
    )
    monkeypatch.setattr(
        backend,
        "_site_evidence_phase_key",
        lambda *_args, **_kwargs: (_FakeKey(), "t" * 64),
    )
    monkeypatch.setattr(
        backend,
        "DynamicFoldPhaseCache",
        lambda _root: _FakePhaseCache(tmp_path / "payload"),
    )
    monkeypatch.setattr(backend, "_EVIDENCE_MEMORY", {})

    loaded, _, reused, memory_reused, *_ = backend._load_cached_site_evidence(
        SimpleNamespace(),
        layer_index=2,
        endpoints=("residual.pre",),
        evidence_policy={
            "cache_root": tmp_path / "cache",
            "parent_trace_cache": tmp_path / "trace",
            "basis_phase_cache_root": tmp_path / "basis",
            "basis_phase_key_sha256": "b" * 64,
            "basis_phase_hmac_key_file": tmp_path / "key",
            "basis_phase_hmac_key_id": "test-key",
            "target_hidden_size": 816,
        },
    )

    assert loaded is evidence
    assert reused is False
    assert memory_reused is False
    assert observed["evidence_split_manifest"] is None


def test_site_evidence_parent_identity_must_match_live_parent() -> None:
    digest = "a" * 64
    assert (
        backend._require_site_evidence_parent_identity(
            {"parent_identity_sha256": digest}, {"identity_sha256": digest}
        )
        == digest
    )
    with pytest.raises(
        backend.Lfm2MatrixLocalBackendError,
        match="different parent identity",
    ):
        backend._require_site_evidence_parent_identity(
            {"parent_identity_sha256": "b" * 64},
            {"identity_sha256": digest},
        )


def test_legacy_cache_uses_physical_attention_layer_indices() -> None:
    key0 = torch.ones(1, 2, 3, 4)
    value0 = torch.full_like(key0, 2)
    key1 = torch.full_like(key0, 3)
    value1 = torch.full_like(key0, 4)

    rows = backend._extract_cache(
        ((key0, value0), (key1, value1)),
        attention_layer_indices=(2, 6),
        physical_layer_count=8,
    )

    assert tuple(index for index, _key, _value in rows) == (2, 6)
    assert rows[0][1] is key0
    assert rows[1][2] is value1


def test_cache_count_mismatch_fails_closed() -> None:
    tensor = torch.ones(1, 2, 3, 4)
    with pytest.raises(
        backend.Lfm2MatrixLocalBackendError,
        match="differs from physical attention topology",
    ):
        backend._extract_cache(
            ((tensor, tensor),),
            attention_layer_indices=(2, 6),
            physical_layer_count=8,
        )
