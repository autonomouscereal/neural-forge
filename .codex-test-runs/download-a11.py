from __future__ import annotations

import sys
from pathlib import Path

SKILL = Path(r"C:\Users\cereal\.agents\skills\server-manager")
sys.path.insert(0, str(SKILL))

from ssh_client import SSHClient, ServerConfig  # noqa: E402


REMOTE = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "neural_forge_sparse_semantic_bridge_20260730/"
    "a11-k32-split-cout-lambda0001"
)
LOCAL = Path(
    "C:/Users/cereal/Documents/ai training/neural-forge/reports/"
    "sparse-semantic-a11-k32-split-cout-20260730"
)
FILES = (
    "base-codebook-override-authentication.json",
    "cached-hidden-diagnostic.json",
    "numeric-dtype-contract.json",
    "preflight-and-phase-summary.json",
    "functional-output-refit-authentication.json",
    "whole-response-dossier.json",
    "run-receipt.json",
)


def main() -> None:
    config = ServerConfig()
    client = SSHClient.from_config(config.get("ai"))
    if not client.connect():
        raise SystemExit("server-manager connection failed")
    try:
        for name in FILES:
            if not client.download_file(str(REMOTE / name), str(LOCAL)):
                raise SystemExit(f"download failed: {name}")
            print(f"downloaded={name}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
