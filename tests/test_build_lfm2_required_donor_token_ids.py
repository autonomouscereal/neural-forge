from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments import build_lfm2_required_donor_token_ids as BUILD  # noqa: E402
from experiments.compile_lfm2_target_tokenizer_bridge import (  # noqa: E402
    load_required_donor_token_ids,
)


class _Tensor:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def reshape(self, size: int) -> _Tensor:
        assert size == -1
        return self

    def tolist(self) -> list[int]:
        return list(self._values)


class _Payload:
    def __init__(
        self,
        batch_index: int,
        input_ids: list[int],
        forced_ids: list[int],
        session_sha256: str,
    ) -> None:
        self.batch_index = batch_index
        self.input_sha256 = f"{batch_index + 3:x}" * 64
        self.ordered_tensor_names = BUILD.SELECTED_TENSOR_NAMES
        self.logical_bytes = 8 * (len(input_ids) + len(forced_ids))
        self.selection_sha256 = f"{batch_index + 5:x}" * 64
        self.session_sha256 = session_sha256
        self.tensors = {
            "input.forced_decode_ids": _Tensor(forced_ids),
            "input.input_ids": _Tensor(input_ids),
        }
        self.released = False


class _Session:
    def __init__(
        self,
        cache: _Cache,
        batches: list[tuple[list[int], list[int]]],
    ) -> None:
        self.cache_key = cache.cache_key
        self.batch_count = len(batches)
        self.evidence_sha256 = cache.session_sha256
        self._cache = cache
        self._batches = batches
        self.loaded_names: list[tuple[str, ...]] = []
        self.payloads: list[_Payload] = []
        self.closed = False

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.closed = True

    def load_selected_tensors(
        self,
        index: int,
        names: tuple[str, ...],
    ) -> _Payload:
        self.loaded_names.append(names)
        input_ids, forced_ids = self._batches[index]
        payload = _Payload(
            index,
            input_ids,
            forced_ids,
            self.evidence_sha256,
        )
        self.payloads.append(payload)
        return payload

    def release_selected_tensors(self, payload: _Payload) -> None:
        payload.released = True

    def verify(self) -> _Session:
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "manifest_sha256": self._cache.manifest_sha256,
            "completion_sha256": self._cache.completion_sha256,
            "batch_count": self.batch_count,
            "session_sha256": self.evidence_sha256,
        }


class _Cache:
    def __init__(
        self,
        path: Path,
        *,
        key_digit: str,
        parent_digit: str = "a",
        batches: list[tuple[list[int], list[int]]],
    ) -> None:
        self.path = path
        self.cache_key = key_digit * 64
        self.parent_fingerprint = parent_digit * 64
        self.corpus_fingerprint = "c" * 64
        self.manifest_sha256 = "d" * 64
        self.completion_sha256 = "e" * 64
        self.session_sha256 = "f" * 64
        self.batch_count = len(batches)
        self.manifest = {
            "manifest_sha256": self.manifest_sha256,
            "parent": {"identity_sha256": self.parent_fingerprint},
            "corpus": {"corpus_sha256": self.corpus_fingerprint},
        }
        self.session = _Session(self, batches)

    def selective_authenticated_session(self) -> _Session:
        return self.session


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cache_union_reads_only_token_tensors_and_writes_loadable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "cache-a"
    second_root = tmp_path / "cache-b"
    first_root.mkdir()
    second_root.mkdir()
    caches = {
        first_root.resolve(): _Cache(
            first_root.resolve(),
            key_digit="1",
            batches=[([2, 7], [9])],
        ),
        second_root.resolve(): _Cache(
            second_root.resolve(),
            key_digit="2",
            batches=[([7, 11], [15])],
        ),
    }
    monkeypatch.setattr(BUILD, "_open_trace_cache", caches.__getitem__)
    output = tmp_path / "required-token-ids.json"

    receipt = BUILD.write_required_donor_token_ids_manifest(
        [second_root, first_root],
        output,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    body = dict(manifest)
    seal = body.pop("manifest_sha256")
    assert seal == BUILD._canonical_sha256(body)
    assert receipt["output_file_sha256"] == _sha256(output)
    assert manifest["required_donor_token_ids"] == [2, 7, 9, 11, 15]
    assert manifest["source_cache_count"] == 2
    assert [source["cache_key"] for source in manifest["source_caches"]] == [
        "1" * 64,
        "2" * 64,
    ]
    for cache in caches.values():
        assert cache.session.loaded_names == [BUILD.SELECTED_TENSOR_NAMES]
        assert cache.session.closed is True
        assert all(payload.released for payload in cache.session.payloads)

    ids, loader_report = load_required_donor_token_ids(
        output,
        expected_file_sha256=receipt["output_file_sha256"],
    )
    assert ids == (2, 7, 9, 11, 15)
    assert loader_report["required_donor_token_ids"] == [2, 7, 9, 11, 15]


def test_cache_union_rejects_mixed_parent_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "cache-a"
    second_root = tmp_path / "cache-b"
    first_root.mkdir()
    second_root.mkdir()
    caches = {
        first_root.resolve(): _Cache(
            first_root.resolve(),
            key_digit="1",
            parent_digit="a",
            batches=[([2], [3])],
        ),
        second_root.resolve(): _Cache(
            second_root.resolve(),
            key_digit="2",
            parent_digit="b",
            batches=[([4], [5])],
        ),
    }
    monkeypatch.setattr(BUILD, "_open_trace_cache", caches.__getitem__)

    with pytest.raises(
        BUILD.Lfm2RequiredDonorTokenIdsError,
        match="one authenticated parent identity",
    ):
        BUILD.build_required_donor_token_ids_manifest(
            [first_root, second_root]
        )
