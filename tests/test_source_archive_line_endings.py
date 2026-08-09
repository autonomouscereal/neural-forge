from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMPT_RELATIVE_PATH = Path(
    "research/lfm2_semantic_whole_response_prompts_v1.json"
)
EXPECTED_PROMPT_BYTES = 8_646
EXPECTED_PROMPT_SHA256 = (
    "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "core.autocrlf=true",
            *arguments,
        ],
        check=True,
        capture_output=True,
    )


def test_windows_git_archive_preserves_frozen_manual_prompt_bytes(
    tmp_path: Path,
) -> None:
    source = REPOSITORY_ROOT / PROMPT_RELATIVE_PATH
    source_bytes = source.read_bytes()
    assert len(source_bytes) == EXPECTED_PROMPT_BYTES
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_PROMPT_SHA256
    assert b"\r\n" not in source_bytes

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitattributes").write_bytes(
        (REPOSITORY_ROOT / ".gitattributes").read_bytes()
    )
    archived_source = repository / PROMPT_RELATIVE_PATH
    archived_source.parent.mkdir(parents=True)
    archived_source.write_bytes(source_bytes)
    _git(
        repository,
        "add",
        "--",
        ".gitattributes",
        PROMPT_RELATIVE_PATH.as_posix(),
    )
    tree_object_id = _git(repository, "write-tree").stdout.decode("ascii").strip()

    archive = _git(
        repository,
        "archive",
        "--format=tar",
        tree_object_id,
        "--",
        PROMPT_RELATIVE_PATH.as_posix(),
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        member = tar.getmember(PROMPT_RELATIVE_PATH.as_posix())
        extracted = tar.extractfile(member)
        assert extracted is not None
        archived_bytes = extracted.read()

    assert archived_bytes == source_bytes
    assert len(archived_bytes) == EXPECTED_PROMPT_BYTES
    assert hashlib.sha256(archived_bytes).hexdigest() == EXPECTED_PROMPT_SHA256
    assert b"\r\n" not in archived_bytes
