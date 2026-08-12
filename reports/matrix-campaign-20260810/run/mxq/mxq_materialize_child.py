#!/usr/bin/env python3
"""Write a sealed composed child out as a loadable HF model directory.

ADDENDUM 8 asks for a continuous intelligence-per-compression measure, which
means running a real benchmark harness against the child.  lm-eval loads a model
directory, and the campaign's children exist only as "the parent plus these N
folded tensors", so this materializes that sum -- and materializes it under the
SAME rule the frozen review uses, or the benchmark would not be about the object
the review graded:

* `run_lfm2_composition_frozen_review.py` reconstitutes at **float32** (its
  `--dtype` default) -- parent bf16 upcast to fp32, folded fp32 tensors copied
  in -- so the written directory is float32 and the parent baseline is
  benchmarked at float32 too.  Same treatment on both sides or the comparison is
  between two numeric regimes rather than between two models.
* the .pt file digest is checked against `composed_child_tensors_sha256`, and
  every folded tensor is checked against the per-tensor digest the receipt
  sealed, using `_tensor_sha256`'s exact convention.  A directory that cannot
  prove it is the sealed child is not written.

Heavy output goes under /mnt/ai_data only: a float32 230M child is ~920 MB and
the root filesystem has ~7 GB free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(
    0, "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8/src"
)

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sealed_tensor_digests(state_digest: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    for entries in (state_digest.get("installed_tensor_digests") or {}).values():
        for entry in entries:
            key, _, digest = str(entry).partition("=")
            found[key] = digest
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.cell_directory
    receipt = json.loads((root / "cell-receipt.json").read_text(encoding="utf-8"))
    composition = receipt["composition"]
    tensors_path = root / "composed-child-tensors.pt"

    declared = composition.get("composed_child_tensors_sha256")
    observed = file_sha256(tensors_path)
    if declared is not None and str(declared) != observed:
        raise SystemExit(
            f"REFUSED: {tensors_path} digests {observed}, the receipt sealed {declared}."
        )

    state_digest = json.loads(
        (root / "composed-child-state-digest.json").read_text(encoding="utf-8")
    )
    expected = sealed_tensor_digests(state_digest)
    folded = torch.load(tensors_path, weights_only=True)
    if sorted(folded) != sorted(expected):
        raise SystemExit(
            f"REFUSED: saved tensors {sorted(folded)} are not the sealed folded "
            f"set {sorted(expected)}."
        )
    for key in sorted(folded):
        live = tensor_sha256(folded[key])
        if live != expected[key]:
            raise SystemExit(
                f"REFUSED: {key} digests {live}, the receipt sealed {expected[key]}."
            )

    weights = load_file(str(PARENT / "model.safetensors"))
    state = {key: value.to(torch.float32) for key, value in weights.items()}
    missing = [key for key in folded if key not in state]
    if missing:
        raise SystemExit(f"REFUSED: parent has no tensor {missing}.")
    for key, value in folded.items():
        target = state[key]
        value = value.to(torch.float32)
        if tuple(value.shape) != tuple(target.shape):
            raise SystemExit(
                f"REFUSED: {key} is {tuple(value.shape)}, parent is {tuple(target.shape)}."
            )
        state[key] = value.contiguous()

    arguments.out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(arguments.out / "model.safetensors"), metadata={"format": "pt"})
    # copyfile, not copy2: the parent model directory is read-only, and copy2
    # carries its mode bits across so the config rewrite below fails with EACCES
    # on the second file it touches.  The sidecars are data, not permissions.
    for name in SIDECARS:
        source = PARENT / name
        if source.is_file() and name != "config.json":
            destination = arguments.out / name
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
    config = json.loads((PARENT / "config.json").read_text(encoding="utf-8"))
    config["dtype"] = "float32"
    config.pop("torch_dtype", None)
    (arguments.out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = {
        "schema": "neural-forge-lfm2-matrix-materialized-child-v1",
        "cell_id": receipt["cell_id"],
        "receipt_sha": receipt["receipt_sha"],
        "composed_child_tensors_sha256": observed,
        "install_order": composition["install_order"],
        "folded_tensor_count": len(folded),
        "folded_tensor_digests_verified": sorted(folded),
        "net_stored_ratio": composition["storage"]["net_ratio"],
        "child_parameter_count": composition["storage"]["child_parameter_count"],
        "parent_parameter_count": composition["storage"]["parent_parameter_count"],
        "materialized_dtype": "float32",
        "materialization_rule": (
            "Parent bf16 weights upcast to float32 and the sealed folded float32 "
            "tensors copied in -- byte-for-byte the reconstitution "
            "run_lfm2_composition_frozen_review.py performs at its float32 "
            "default, so a benchmark of this directory is a benchmark of the "
            "object the frozen review graded.  The parent baseline is benchmarked "
            "from its own directory at the same float32 dtype."
        ),
        "written_model_safetensors_sha256": file_sha256(
            arguments.out / "model.safetensors"
        ),
    }
    (arguments.out / "materialization.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{receipt['cell_id']}: {len(folded)} folded tensors verified against the "
        f"seal -> {arguments.out}"
    )
    print(f"  net {composition['storage']['net_ratio']:.6f}x  "
          f"model.safetensors {provenance['written_model_safetensors_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
