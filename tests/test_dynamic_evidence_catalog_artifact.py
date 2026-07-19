from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from neural_forge.dynamic_evidence_catalog_artifact import (
    CATALOG_FILENAME,
    MANIFEST_FILENAME,
    DynamicEvidenceCatalogArtifactIntegrityError,
    DynamicEvidenceCatalogArtifactPublicationError,
    DynamicEvidenceCatalogArtifactStore,
    DynamicEvidenceCatalogBinding,
)


def _sha(character: str) -> str:
    return character * 64


def _binding(*, suffix: str = "a") -> DynamicEvidenceCatalogBinding:
    return DynamicEvidenceCatalogBinding(
        catalog_kind="test-selective-catalog",
        source_sha256=_sha(suffix),
        provider_sha256=_sha("b"),
        authenticated_input_sha256=_sha("c"),
        codec_sha256=_sha("d"),
        bindings={
            "cache_key": _sha("e"),
            "layer_count": 2,
            "phase_order": ["prefill", "decode.000"],
        },
    )


def _payload() -> dict:
    return {
        "schema": "test-dynamic-evidence-payload-v1",
        "catalog_sha256": _sha("f"),
        "layers": [
            {"layer_index": 0, "evidence_sha256": _sha("1")},
            {"layer_index": 1, "evidence_sha256": _sha("2")},
        ],
        "tensor_payloads_resident": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }


def _canonical_write(path, value) -> None:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    path.write_bytes(encoded + b"\n")


def test_cold_publish_warm_load_and_immutable_collision(tmp_path) -> None:
    store = DynamicEvidenceCatalogArtifactStore(tmp_path / "catalogs")
    binding = _binding()
    assert store.load(binding) is None

    cold = store.publish(binding, _payload())
    assert cold.disposition == "published"
    assert set(path.name for path in cold.path.iterdir()) == {
        CATALOG_FILENAME,
        MANIFEST_FILENAME,
    }
    assert dict(cold.payload) == _payload()
    assert cold.verify().manifest_sha256 == cold.manifest_sha256
    assert cold.as_dict()["tensor_payloads_resident"] is False
    assert cold.as_dict()["parent_forward_calls"] == 0
    assert cold.as_dict()["training_steps"] == 0

    warm = store.load(binding)
    assert warm is not None
    assert warm.disposition == "loaded"
    assert warm.catalog_sha256 == cold.catalog_sha256
    assert warm.manifest_sha256 == cold.manifest_sha256
    assert dict(warm.payload) == _payload()
    assert store.publish(binding, _payload()).disposition == "loaded"

    changed = _payload()
    changed["layers"][0]["evidence_sha256"] = _sha("3")
    with pytest.raises(
        DynamicEvidenceCatalogArtifactPublicationError,
        match="another payload",
    ):
        store.publish(binding, changed)


def test_existing_tamper_fails_closed_without_running_builder(tmp_path) -> None:
    store = DynamicEvidenceCatalogArtifactStore(tmp_path / "catalogs")
    binding = _binding()
    artifact = store.publish(binding, _payload())
    document_path = artifact.path / CATALOG_FILENAME
    document = json.loads(document_path.read_text(encoding="ascii"))
    document["payload"]["layers"][0]["layer_index"] = 99
    _canonical_write(document_path, document)

    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return _payload()

    with pytest.raises(
        DynamicEvidenceCatalogArtifactIntegrityError,
        match="seal|binding",
    ):
        store.load_or_build(binding, builder)
    assert calls == 0
    assert artifact.path.is_dir()


def test_exact_inventory_and_binding_mismatch_fail_closed(tmp_path) -> None:
    store = DynamicEvidenceCatalogArtifactStore(tmp_path / "catalogs")
    original = store.publish(_binding(), _payload())
    (original.path / "unexpected.bin").write_bytes(b"not allowed")
    with pytest.raises(
        DynamicEvidenceCatalogArtifactIntegrityError,
        match="inventory",
    ):
        store.load(_binding())

    other = _binding(suffix="9")
    shutil.copytree(original.path, store.path_for(other))
    (store.path_for(other) / "unexpected.bin").unlink()
    with pytest.raises(
        DynamicEvidenceCatalogArtifactIntegrityError,
        match="binding",
    ):
        store.load(other)


def test_concurrent_identical_publish_has_one_winner_and_no_clobber(
    tmp_path,
) -> None:
    root = tmp_path / "catalogs"
    stores = [DynamicEvidenceCatalogArtifactStore(root) for _ in range(8)]
    binding = _binding()
    barrier = threading.Barrier(len(stores))

    def publish(store):
        barrier.wait()
        return store.publish(binding, _payload())

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        artifacts = tuple(executor.map(publish, stores))

    assert [value.disposition for value in artifacts].count("published") == 1
    assert [value.disposition for value in artifacts].count("loaded") == 7
    assert len({value.manifest_sha256 for value in artifacts}) == 1
    assert len({value.catalog_sha256 for value in artifacts}) == 1
    assert set(path.name for path in artifacts[0].path.iterdir()) == {
        CATALOG_FILENAME,
        MANIFEST_FILENAME,
    }
    assert not tuple(root.glob(".dynamic-evidence-catalog-stage-*"))
