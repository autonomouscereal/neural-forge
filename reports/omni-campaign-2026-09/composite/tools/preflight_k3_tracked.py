"""Read-only AI-server preflight; print a plan, never allocate or start a GPU.

Run on the host before the root agent starts the printed command through the
shared GPU allocator. The read-only source mounts must stay unchanged afterward.
The digest receipt proves its dated verification, not a new 1.56 TB hash pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TOOL = "reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py"
LANE = "reports/omni-campaign-2026-09/k3/"
SOURCE_HASHES = {
    TOOL: "cce7ac8a4f3a39dfe07904f3e0321f472eb8a52ca17b99b057a4b829eeb5f3cf",
    LANE + "k3_common.py": "3f5e457ced99b68345ab85642fd4f75b23e9444c55c09f9c1a5f7bb53dd8fc3f",
    LANE + "k3_layer.py": "07269c9196ab6e45c3299d671f972e6d2da296157fe5194e4cee5fd7a61c5390",
    LANE + "k3_stream.py": "789cb7fe18cbfa0a5b83b3addd5fc8edd5ccf094edbc5f55766bd4e5dc29b5bf",
    LANE + "c14_union_coreset.py": "3356780e918af111a108d40ef6b2b2658667441256014a9eb3443fa1e78cd535",
    LANE + "k3_receipt.py": "9242d42cc77a398265a319161e0d8e2fa06a6e94aedd87c2ea6fe19abba7dcd2",
    LANE + "k3_review_stream.py": "94c2005b80d5ba963cbf3f73ae76be57b1381d74ae984f873a0d48719d35af76",
}
CORPUS_HASHES = {
    "k3_input_ids.safetensors": "bb3b16573665d049fdc6f1368182fbffb2aabee2b83eff771b9f35c5c839ed38",
    "k3_review_input_ids.safetensors": "4b34c87d5d6a3aa9dceede2b21df991202614c5490ef9b6c8d75ac2a84ee8732",
    "k3_review_native_input_ids.safetensors": "5c53c84f9587b6a356209849fb578ed7c339a644c6f510460daf82e0ac3e1de7",
}
DIGEST_SHA = "58bc54f0c635ec78f53a4575155d249ebd10bde4ea302f65872bebe253d7a91f"
L0_SHA = "62d61d0741d1d78864fbe2f14b1aa0253564bef6f017523d71482ceea6d7d966"


def verify_file(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"hash mismatch: {path}: {actual} != {expected}")


def build_plan(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("--image must be the inspected immutable local image ID")
    paths = {name: Path(getattr(args, name)).resolve()
             for name in ("nf", "chimera", "parent", "corpus", "digest", "layer0", "output")}
    output = paths["output"]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
           for name, p in paths.items() if name != "output"):
        raise ValueError("output must not overlap a source/input mount")
    legacy = Path("/mnt/lexar/omni-campaign/k3").resolve()
    if output == legacy or output.is_relative_to(legacy) or legacy.is_relative_to(output):
        raise ValueError("the accepted Claude K3 output and snapshots must remain untouched")
    if args.layers != "0-1":
        raise ValueError("this preflight prepares only a fresh 0-1 run; review its receipt before continuation")
    if output.exists() and (not output.is_dir() or next(output.iterdir(), None) is not None):
        raise ValueError("fresh output must be absent or an empty directory; never overwrite a prior run")
    for rel, expected in SOURCE_HASHES.items():
        verify_file(paths["nf"] / rel, expected)
    for name, expected in CORPUS_HASHES.items():
        verify_file(paths["corpus"] / name, expected)
    verify_file(paths["digest"], DIGEST_SHA)
    verify_file(paths["layer0"], L0_SHA)
    verify_file(paths["parent"] / "config.json",
                "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213")
    verify_file(paths["parent"] / "model.safetensors.index.json",
                "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd")
    if not (paths["chimera"] / "src/chimera_omni/__init__.py").is_file():
        raise ValueError("--chimera must be the reviewed extracted Chimera 5114cdfd project root")
    digest = json.loads(paths["digest"].read_text(encoding="utf-8"))
    if not digest.get("complete") or not digest.get("all_verified") or len(digest["files"]) != 98:
        raise ValueError("digest receipt is not the complete 98-file verification")
    for row in digest["files"]:
        shard = paths["parent"] / row["file"]
        if shard.stat().st_size != row["local_size"]:
            raise ValueError(f"source size changed since the digest receipt: {shard}")
    mounts = [{"source": str(paths[name]), "target": target, "read_only": True}
              for name, target in (("nf", "/code/neural-forge"), ("chimera", "/code/chimera"),
                                   ("parent", "/parent"), ("corpus", "/inputs/corpus"),
                                   ("digest", "/inputs/digest.json"), ("layer0", "/inputs/layer0.json"))]
    mounts.append({"source": str(output), "target": "/run/k3", "read_only": False})
    command = ["env", f"OMP_NUM_THREADS={args.threads}", f"MKL_NUM_THREADS={args.threads}",
               f"OPENBLAS_NUM_THREADS={args.threads}", "python", "/code/neural-forge/" + TOOL,
               "--src-dir", "/parent", "--input-ids", "/inputs/corpus/k3_input_ids.safetensors",
               "--review-input-ids", "/inputs/corpus/k3_review_input_ids.safetensors",
               "--review-input-ids", "/inputs/corpus/k3_review_native_input_ids.safetensors",
               "--out", "/run/k3/fold", "--receipt-dir", "/run/k3/receipts", "--body-dir", "/run/k3/body",
               "--chimera-src", "/code/chimera/src", "--digest-receipt", "/inputs/digest.json",
               "--layer0-import-receipt", "/inputs/layer0.json", "--no-share-read",
               "--device", "cuda:0", "--experts-device", args.experts_device, "--resid-device", "cpu",
               "--compute-dtype", "auto", "--kda-backend", "torch", "--layers", args.layers,
               "--rung", "17408", "--k-pool-factor", "1", "--row-chunk", "1",
               "--token-chunk", "1024", "--chunk", "1024", "--holdout-blocks", "24",
               "--max-gram-gb", "160", "--max-device-gb", "-1",
               "--snapshot-every", "12", "--snapshot-keep", "2", "--acceptance", "gate"]
    return {"action": "plan_only_no_allocation_or_launch", "image_id": args.image,
            "required_gpu_uuid": "GPU-f5a09535-fc85-a027-50f7-734b344747dd",
            "mounts": mounts, "command_argv": command, "source_hashes": SOURCE_HASHES,
            "corpus_hashes": CORPUS_HASHES, "digest_receipt_sha256": DIGEST_SHA,
            "digest_scope": "dated full verification plus current file-size checks; not a new full hash pass",
            "chimera_identity": "root must verify the mounted tree against reviewed 5114cdfd archive",
            "next": "create only the new empty output directory; root allocates, runs canaries, then launches",
            "continuation": "review fresh 0-1, then keep image/mounts/threads/corpus unchanged and resume 2-12"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nf", "chimera", "parent", "corpus", "digest", "layer0", "output", "image"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", type=int, choices=(22, 44), default=22)
    parser.add_argument("--experts-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--layers", default="0-1")
    args = parser.parse_args()
    try:
        plan = build_plan(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"preflight refused: {exc}\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
