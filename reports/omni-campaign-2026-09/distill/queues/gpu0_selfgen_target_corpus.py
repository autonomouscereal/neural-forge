#!/usr/bin/env python3
"""gpu0 lane, job 1 input: the self-gen rescoring corpus with a TARGET mask.

`selfgen_sampler.py export-rescore` (1.2.0) writes the rescoring corpus with an
input-indexed `store_mask` ([prompt_len - 1, length - 2] per row: its own shift).
qtext_teacher_cache 1.0.4 (finding L2) takes a target-indexed `target_mask` and shifts
it itself.  This script adds `target_mask` to a copy of the export, derived a second,
independent way -- from the sealed self-gen shards' own row records (prompt_len,
response_len), not from the export's mask -- and checks that the export's ids are the
shard tokens.  The copy carries BOTH masks, so qtext 1.0.4 refuses the build unless
the producer's shift and the tool's shift agree at every position.

    python gpu0_selfgen_target_corpus.py <selfgen run dir> <export-rescore .safetensors> <out .safetensors>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

TOOLS = Path(os.environ.get("DISTILL_TOOLS", "/home/cereal/omni-campaign/distill/tools"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    run_dir, export, out = (Path(a) for a in sys.argv[1:4])
    spec = importlib.util.spec_from_file_location("selfgen_sampler", TOOLS / "selfgen_sampler.py")
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not sg.check_seal(manifest, "manifest_sha256"):
        print("REFUSED: the self-gen manifest does not match its own seal")
        return 2
    tensors = load_file(str(export))
    ids, lengths, store = tensors["input_ids"], tensors["lengths"], tensors["store_mask"]
    target = np.zeros_like(store)
    n = 0
    for name in manifest["shard_read_order"]:
        loaded = sg.read_shard(run_dir / name)
        for row in loaded["record"]["rows"]:
            p, r = int(row["prompt_len"]), int(row["response_len"])
            tokens = np.asarray(loaded["tokens"][row["offset"]: row["offset"] + p + r], dtype=np.int64)
            if int(lengths[n]) != p + r or not np.array_equal(ids[n, : p + r].astype(np.int64), tokens):
                print(f"REFUSED: export row {n} is not shard {name} row {row.get('index')} (length or ids differ)")
                return 2
            target[n, p : p + r] = 1
            n += 1
    if n != ids.shape[0]:
        print(f"REFUSED: the shards hold {n} rows, the export {ids.shape[0]}")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    save_file({**tensors, "target_mask": target}, str(tmp),
              metadata={"source_export_sha256": sha256(export), "source_manifest_sha256": manifest["manifest_sha256"],
                        "target_rule": "target_mask[n, t] = 1 for t in [prompt_len, prompt_len + response_len) from the shard row records",
                        "made_by": "distill/queues/gpu0_selfgen_target_corpus.py"})
    os.replace(tmp, out)
    print(f"target corpus {out} sha256 {sha256(out)} rows {n} width {ids.shape[1]} "
          f"targets {int(target.sum())} export_store {int(store.sum())} max_length {int(lengths.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
