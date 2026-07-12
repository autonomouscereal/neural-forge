from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_tree_digest_is_name_and_content_bound(tmp_path: Path) -> None:
    module = _load(
        "retire_remote_source_copies",
        "experiments/retire_remote_neural_forge_source_copies.py",
    )
    root = tmp_path / "tree"
    (root / "empty").mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "nested" / "value.bin").write_bytes(b"fold")

    file_digest = hashlib.sha256(b"fold").hexdigest()
    manifest = (
        b"D\tempty\nD\tnested\n" + f"F\tnested/value.bin\t4\t{file_digest}\n".encode()
    )
    assert module.tree_sha256(root) == hashlib.sha256(manifest).hexdigest()

    (root / "nested" / "value.bin").write_bytes(b"fail")
    assert module.tree_sha256(root) != hashlib.sha256(manifest).hexdigest()


def test_remote_retirement_manifest_names_only_neural_forge_roots() -> None:
    remote = _load(
        "retire_remote_manifest",
        "experiments/retire_remote_neural_forge_source_copies.py",
    )
    assert len(remote.COPIES) == 8
    assert len({copy.name for copy in remote.COPIES}) == 8
    assert all(len(copy.archive_commit) == 40 for copy in remote.COPIES)
    assert all(copy.name.startswith("neural-forge-") for copy in remote.COPIES)
