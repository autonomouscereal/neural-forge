#!/usr/bin/env python3
"""Launch one experiment in a new session without retaining control-channel FDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_environment(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        key, separator, value = str(item).partition("=")
        if not separator or not key or "\0" in key or "=" in key:
            raise ValueError(f"invalid environment assignment: {item!r}")
        values[key] = value
    return values


def _linux_process_start_ticks(pid: int) -> int | None:
    path = Path(f"/proc/{int(pid)}/stat")
    try:
        tail = path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def launch_detached(
    command: list[str],
    *,
    log_path: Path,
    pid_path: Path,
    identity_path: Path,
    cwd: Path | None,
    environment: dict[str, str],
) -> dict[str, object]:
    if not command:
        raise ValueError("command cannot be empty.")
    if pid_path.exists() or identity_path.exists():
        raise FileExistsError("pid or identity file already exists; use a unique run tag.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    child_environment.update(environment)
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command,
            cwd=None if cwd is None else str(cwd),
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    time.sleep(0.05)
    if process.poll() is not None:
        raise RuntimeError(f"detached command exited during launch with code {process.returncode}.")
    identity = {
        "schema": "neural-forge-detached-process-v1",
        "pid": int(process.pid),
        "start_ticks": _linux_process_start_ticks(process.pid),
        "argv": command,
        "argv_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "cwd": None if cwd is None else str(cwd.resolve()),
        "log": str(log_path.resolve()),
        "environment": dict(sorted(environment.items())),
        "launcher_pid": os.getpid(),
    }
    _atomic_write_text(pid_path, f"{process.pid}\n")
    _atomic_write_text(identity_path, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    identity_path = args.identity_file or args.pid_file.with_suffix(args.pid_file.suffix + ".identity.json")
    identity = launch_detached(
        command,
        log_path=args.log,
        pid_path=args.pid_file,
        identity_path=identity_path,
        cwd=args.cwd,
        environment=_parse_environment(args.env),
    )
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
