from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "consolidate_remote_checkout_paths.py"
)
SPEC = importlib.util.spec_from_file_location(
    "consolidate_remote_checkout_paths", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_consolidate_rewrites_stale_root_without_changing_newlines(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    target = experiments / "launch.sh"
    stale = module.STALE_ROOTS[0]
    target.write_bytes(b"ROOT=" + stale + b"\r\nnext\r\n")

    preview = module.consolidate(tmp_path, write=False)
    assert preview[0]["replacement_count"] == 1
    assert target.read_bytes() == b"ROOT=" + stale + b"\r\nnext\r\n"

    written = module.consolidate(tmp_path, write=True)
    assert written[0]["written"] is True
    assert target.read_bytes() == b"ROOT=" + module.CANONICAL_ROOT + b"\r\nnext\r\n"


def test_copy_inventory_scripts_are_intentionally_excluded(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    target = experiments / "remote_inventory_neural_forge_sources_20260712.sh"
    target.write_bytes(module.STALE_ROOTS[0])

    assert module.consolidate(tmp_path, write=True) == []
    assert target.read_bytes() == module.STALE_ROOTS[0]
