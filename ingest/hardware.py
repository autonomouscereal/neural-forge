"""ingest.hardware -- GPU cards of the host this runs on (nvidia-smi) -> host, gpu.

The UUID is the join key; gpu_index is refreshed to the current nvidia-smi (PCI-ordered) index, which
is what the status files of the campaign name.  Run on the AI server; on a host without nvidia-smi
the batch records zero rows.
"""
from __future__ import annotations

import shutil
import subprocess

from . import common as C
from .common import db

SOURCE = "hardware"
QUERY = "index,uuid,name,compute_cap,memory.total,driver_version"


def parse_nvidia_smi(text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=<QUERY> --format=csv,noheader`."""
    cards = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        mem = parts[4].split()[0]
        cards.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "sm": parts[3],
                      "vram_mib": int(mem) if mem.isdigit() else None, "driver": parts[5]})
    return cards


def run(conn, batch: C.Batch, args) -> None:
    host_name = db.this_host()
    if not shutil.which("nvidia-smi"):
        batch.skip(f"no nvidia-smi on {host_name}")
        return
    out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    cards = parse_nvidia_smi(out)
    drivers = sorted({c["driver"] for c in cards})
    hid = db.ensure_host(conn, host_name, {"nvidia_driver": drivers[0] if len(drivers) == 1 else drivers,
                                           "gpu_count": len(cards)})
    for c in cards:
        batch.saw()
        db.upsert_gpu(conn, hid, c["index"], c["uuid"], c["name"], c["sm"], c["vram_mib"])
        batch.wrote()
