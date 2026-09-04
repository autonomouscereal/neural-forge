#!/usr/bin/env python3
"""s0_import_27b.py -- S0 of COMPOSITE step 1: the exact Qwen3.5/3.8 -> Chimera Omni import.

A thin CLI around the IMPORT_LANE_STATUS section-7 snippet
(`load_qwen35(dir, cfg, device='cpu', dtype=torch.float32, allow_unverified_output_gate=True)`
-> `.to(torch.bfloat16)` -> `write_safetensors(..., chimera_commit=...)`) with the
`music_codes` band declared through `ChimeraConfig.ledger_bands`.

What it produces under `--out`:

    model-*.safetensors            bf16 shards (Chimera parameter names)
    model.safetensors.index.json   weight_map + per-tensor sha256 (safetensors_io.write_safetensors)
    config.json                    chimera_omni.checkpoint.v1 (manifest with the bands declared)
    import_receipt.json            chimera_omni.import.qwen35.v1 (+ .sha256 sidecar)

Receipt contents beyond the loader's own record: the skipped `mtp.*` /
`model.visual.*` tensors with sha256, the random-init interface tensors, the
declared band(s) with their measured header growth and a zero-row check,
`peak_rss_bytes` from `resource.getrusage`, the chimera tree commit the code
ran from, the directory-vs-index shard set agreement (what `oc_seal` will
count), and -- with `--compare-to <checkpoint>` -- a `lane_comparison` block
naming every identical / differing tensor against that checkpoint.

Refusals (exit code 3): `--chimera-commit` does not match the tree the code
runs from (`git rev-parse HEAD` when the tree has a `.git`, otherwise the
`COMMIT` file written next to a git-archive extraction), the tree's
`chimera_omni` is not the one that was imported, `--out` already holds
safetensors (anywhere under it) without `--overwrite`, a non-positive
`--shard-bytes`, `--dtype-out fp16` without `--allow-fp16-narrowing`, a band
row that is not zero, a non-finite value after narrowing. Every one of those
fires BEFORE the write: nothing is written. Should a post-write check ever
refuse (the written shard set is not what the index names, a header element
count is not the config's), the tool unlinks what it wrote and leaves
`REFUSED.json` in `--out` naming the reason, so a checkpoint without a receipt
can never be sealed by `oc_seal`.

`--overwrite` unlinks the previous `model-*.safetensors`,
`model.safetensors.index.json`, `config.json`, `import_receipt.json(.sha256)`
and `REFUSED.json` at the top of `--out` before writing (recorded in the
receipt as `checkpoint.overwrite_removed`); it never touches a subdirectory
and never deletes anything else.

Profile: `--profile auto` (default) selects `chimera_omni_27b_profile` when the
source config carries the verified Qwen3.8-27B geometry and
`qwen35_chimera_config` otherwise; both are explicit choices too. The tensor
bytes are profile-independent -- only `config.json`'s `config.metadata` differs
(`profile`, `geometry_source`) -- which `test_s0_import_27b.py` proves.

Reproducibility: the model is constructed on CPU in fp32 after
`torch.manual_seed(--seed)`, where the default seed is torch's own default
generator seed (67280421310721). That makes the 16 random-init interface tensors
identical across the tool's own runs. The import lane's 0.8B bf16 checkpoint was
written by a process whose RNG history no seed reproduces (it built the HF
reference first), so its random-init tensors differ from this tool's; every
source-derived tensor, every band row and the tied alias are byte-identical.
Downstream sealing keys on THIS tool's `state_sha256`, never on the R-Q receipt's.

Standing rules honoured: no quantization (bf16 is the campaign's storage dtype,
declared as `--dtype-out`; fp16 narrowing is refused unless waived and then
range-checked), dense only, storage counted in parameters from the written
safetensors headers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s0_import_27b"
TOOL_VERSION = "1.1.0"
RECEIPT_SCHEMA = "chimera_omni.import.qwen35.v1"
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"
COMMIT_FILENAME = "COMMIT"
RECEIPT_FILENAME = "import_receipt.json"
REFUSED_FILENAME = "REFUSED.json"
INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"
# torch's default CPU generator seed (c10/core/GeneratorImpl.h: default_rng_seed_val).
TORCH_DEFAULT_SEED = 67280421310721
EXIT_REFUSED = 3
DTYPE_OUT = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
PROFILE_CHOICES = ("auto", "qwen35", "chimera_omni_27b")
# fp16's largest finite value; anything above it in fp32 overflows to inf on cast.
FP16_MAX = 65504.0


class Refusal(RuntimeError):
    """A precondition the tool will not run past; exit code 3."""


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# tree / commit
# --------------------------------------------------------------------------


def _is_hex(text: str) -> bool:
    return len(text) >= 7 and all(ch in "0123456789abcdef" for ch in text)


def read_commit_file(tree: Path) -> str | None:
    path = tree / COMMIT_FILENAME
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0].lower() if line.strip() else ""
        if token and not token.startswith("#"):
            return token
    return None


def resolve_tree_commit(tree: str | Path) -> tuple[str, str]:
    """`(commit, how)` for the chimera-omni tree at `tree`.

    A checkout answers with `git rev-parse HEAD`; a git-archive extraction (no
    `.git`) answers with the `COMMIT` file written next to it. Neither present
    is a refusal, not a guess.
    """

    root = Path(tree).resolve()
    if (root / ".git").exists():
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
        except (OSError, subprocess.CalledProcessError) as error:
            raise Refusal(f"{root} has a .git but `git rev-parse HEAD` failed: {error}") from error
        if not _is_hex(head):
            raise Refusal(f"{root}: git rev-parse HEAD returned {head!r}")
        return head, "git rev-parse HEAD"
    recorded = read_commit_file(root)
    if recorded is None:
        raise Refusal(
            f"{root} is neither a git checkout nor a git-archive tree with a {COMMIT_FILENAME} file; "
            f"write the extracted commit's hash to {root / COMMIT_FILENAME} first"
        )
    if not _is_hex(recorded):
        raise Refusal(f"{root / COMMIT_FILENAME} does not hold a hex commit: {recorded!r}")
    return recorded, f"{COMMIT_FILENAME} file"


def commits_match(declared: str, actual: str) -> bool:
    """Prefix match either way (a short hash names its full hash), 7+ hex chars."""

    left = declared.strip().lower()
    right = actual.strip().lower()
    if not (_is_hex(left) and _is_hex(right)):
        return False
    return left.startswith(right) or right.startswith(left)


def bind_chimera_tree(tree: str | Path, declared_commit: str) -> dict[str, Any]:
    """Put `<tree>/src` first on sys.path, import chimera_omni from it, check the commit."""

    root = Path(tree).resolve()
    src = root / "src"
    if not (src / "chimera_omni" / "__init__.py").is_file():
        raise Refusal(f"{root} has no src/chimera_omni package")
    actual, how = resolve_tree_commit(root)
    if not commits_match(declared_commit, actual):
        raise Refusal(
            f"--chimera-commit {declared_commit!r} does not name the tree the code runs from: "
            f"{root} is at {actual} ({how})"
        )
    sys.path.insert(0, str(src))
    import chimera_omni  # noqa: WPS433 (deliberately late: the path was just bound)

    loaded = Path(chimera_omni.__file__).resolve().parents[1]
    if loaded != src.resolve():
        raise Refusal(
            f"chimera_omni was imported from {loaded}, not from the bound tree {src}; "
            "clear PYTHONPATH / installed copies before running"
        )
    return {"tree": str(root), "src": str(src), "commit": actual, "commit_source": how}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def parse_band(text: str) -> tuple[str, int]:
    name, sep, width = text.partition(":")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"--ledger-band wants NAME:WIDTH, got {text!r}")
    try:
        value = int(width)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--ledger-band width must be an integer: {text!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"--ledger-band width must be positive: {text!r}")
    return name.strip(), value


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=1 << 22) as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_receipt(path: Path, document: dict[str, Any]) -> str:
    text = canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def text_config(hf_config: dict[str, Any]) -> dict[str, Any]:
    """The trunk config: Qwen3.5/3.8 nest it under `text_config`, older dumps do not."""

    nested = hf_config.get("text_config")
    return dict(nested) if isinstance(nested, dict) else dict(hf_config)


def resolve_profile(hf_config: dict[str, Any], requested: str, verified_27b_geometry: dict[str, Any]) -> str:
    """`auto` -> `chimera_omni_27b` iff the source carries every verified 27B geometry field."""

    if requested != "auto":
        if requested not in PROFILE_CHOICES:
            raise Refusal(f"unknown --profile {requested!r}")
        return requested
    trunk = text_config(hf_config)
    is_27b = all(key in trunk and trunk[key] == value for key, value in verified_27b_geometry.items())
    return "chimera_omni_27b" if is_27b else "qwen35"


def directory_shard_files(out_dir: Path) -> list[str]:
    """Every `*.safetensors` under `out_dir`, recursively, as sorted relative POSIX names.

    The same file set `oc_seal.checkpoint_shards` walks (its `_digest_files` is a
    recursive `os.walk`), so the receipt can state whether the directory will be
    counted as exactly the index's shards.
    """

    return sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*.safetensors") if p.is_file())


def index_shard_files(index: dict[str, Any]) -> list[str]:
    return sorted(set(str(name) for name in index["weight_map"].values()))


def prepare_out_dir(out: Path, *, overwrite: bool) -> list[str]:
    """Refuse an occupied `--out`, or (with `--overwrite`) clear the previous layout.

    Returns the relative names of the files removed. Only the tool's own
    top-level products are ever unlinked; a shard in a subdirectory is not the
    tool's to delete and is refused either way, because `oc_seal` would count it.
    """

    if not out.exists():
        return []
    present = directory_shard_files(out)
    nested = [name for name in present if "/" in name]
    if nested:
        raise Refusal(
            f"{out} holds safetensors in a subdirectory ({', '.join(nested[:5])}"
            f"{', ...' if len(nested) > 5 else ''}); oc_seal would count them and the tool "
            "will not delete what it did not write -- move them out first"
        )
    if present and not overwrite:
        raise Refusal(f"{out} already holds safetensors shards; pass --overwrite to replace them")
    removed: list[str] = []
    if overwrite:
        candidates = sorted(out.glob("model-*.safetensors")) + [
            out / INDEX_FILENAME,
            out / CONFIG_FILENAME,
            out / RECEIPT_FILENAME,
            out / (RECEIPT_FILENAME + ".sha256"),
        ]
        for path in candidates:
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        leftover = directory_shard_files(out)
        if leftover:
            raise Refusal(
                f"{out} still holds safetensors after --overwrite cleared the tool's own layout: "
                f"{', '.join(leftover[:5])}; not the tool's files, move them out first"
            )
    marker = out / REFUSED_FILENAME
    if marker.is_file():
        # The tool's own marker from an earlier refused run; a fresh run replaces it.
        marker.unlink()
        removed.append(marker.name)
    return removed


def cleanup_after_refusal(out: Path, index: dict[str, Any] | None, reason: str, argv: list[str]) -> dict[str, Any]:
    """Unlink what this run wrote and leave `REFUSED.json` naming why.

    Called only for a refusal raised AFTER `write_safetensors` returned. The
    shards named by the index, the index and the config are the run's products;
    nothing else is touched.
    """

    removed: list[str] = []
    names: list[str] = []
    if index is not None:
        names.extend(index_shard_files(index))
    # A receipt beside no checkpoint would be a lie; it goes too (a run's own receipt is
    # only written after these checks pass, so any receipt here predates this run).
    names.extend([INDEX_FILENAME, CONFIG_FILENAME, RECEIPT_FILENAME, RECEIPT_FILENAME + ".sha256"])
    for name in names:
        path = out / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    marker = {
        "schema": "chimera_omni.import.refused.v1",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "argv": list(argv)},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "removed": removed,
        "remaining_safetensors": directory_shard_files(out),
    }
    (out / REFUSED_FILENAME).write_text(canonical_json(marker), encoding="utf-8")
    return marker


def header_element_counts(out_dir: Path) -> dict[str, int]:
    """`{tensor name: elements}` from the written shard headers alone (the S0 gate reads headers)."""

    from chimera_omni.importing.safetensors_io import read_header, shard_paths

    counts: dict[str, int] = {}
    for shard in shard_paths(out_dir):
        header, _base = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            elements = 1
            for extent in meta["shape"]:
                elements *= int(extent)
            counts[name] = counts.get(name, 0) + elements
    return counts


def verify_written_checkpoint(out: Path, index: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """The post-write checks: shard set on disk == index, vocab-table headers == config.

    Raises `Refusal` on any disagreement; returns the facts the receipt records.
    """

    on_disk = directory_shard_files(out)
    named = index_shard_files(index)
    if on_disk != named:
        extra = sorted(set(on_disk) - set(named))
        missing = sorted(set(named) - set(on_disk))
        raise Refusal(
            f"{out} holds a different safetensors set than its index names "
            f"(extra: {extra or '-'}; missing: {missing or '-'}); oc_seal would count the directory, "
            "not the index"
        )
    counts = header_element_counts(out)
    hidden = int(cfg.hidden_size)
    expected_embed = int(cfg.vocab_size) * hidden
    embed_elements = counts.get("embed.weight", 0)
    head_elements = counts.get("lm_head.weight", 0)
    if embed_elements != expected_embed:
        raise Refusal(f"embed.weight header holds {embed_elements} elements, expected {expected_embed}")
    if not cfg.tie_word_embeddings and head_elements != expected_embed:
        raise Refusal(f"lm_head.weight header holds {head_elements} elements, expected {expected_embed}")
    if cfg.tie_word_embeddings and "lm_head.weight" in counts:
        raise Refusal("lm_head.weight is in the headers of a tied checkpoint; the alias must be index-only")
    return {
        "counts": counts,
        "directory_shard_files": on_disk,
        "index_shard_files": named,
        "directory_shard_files_match_index": True,
    }


def interface_tensors(initialised: list[str]) -> list[dict[str, Any]]:
    """The random-init Chimera-only tensors, parsed out of the loader's `initialised` lines."""

    out: list[dict[str, Any]] = []
    for line in initialised:
        if "random init" not in line:
            continue
        name, _space, rest = line.partition(" ")
        shape_text = rest.split(" random init", 1)[0].strip()
        try:
            shape = [int(x) for x in shape_text.strip("()").split(",") if x.strip()]
        except ValueError:
            shape = []
        elements = 1
        for extent in shape:
            elements *= extent
        out.append({"name": name, "shape": shape, "elements": elements})
    return sorted(out, key=lambda item: item["name"])


def json_diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every leaf that differs between two JSON documents."""

    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                out.append(path)
            else:
                out.extend(json_diff_paths(left[key], right[key], path))
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        out = []
        for position, (a, b) in enumerate(zip(left, right)):
            out.extend(json_diff_paths(a, b, f"{prefix}[{position}]"))
        return out
    return [] if left == right else [prefix or "<root>"]


def narrowing_report(model: Any, out_dtype: Any) -> dict[str, Any]:
    """Per-tensor range facts for a cast below bf16 (fp16): max|w|, overflow, flush, non-finite."""

    import torch

    finfo = torch.finfo(out_dtype)
    per_tensor: dict[str, dict[str, Any]] = {}
    nonfinite_total = 0
    overflow_total = 0
    flushed_total = 0
    seen: set[int] = set()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            key = tensor.data_ptr()
            if key in seen:
                per_tensor[name] = {"alias": True}
                continue
            seen.add(key)
            source = tensor.detach()
            cast = source.to(out_dtype)
            max_abs = float(source.abs().max().item()) if source.numel() else 0.0
            nonfinite = int((~torch.isfinite(cast)).sum().item())
            overflow = int((source.abs() > finfo.max).sum().item())
            flushed = int(((source != 0) & (cast == 0)).sum().item())
            per_tensor[name] = {
                "max_abs_fp32": max_abs,
                "nonfinite_after_cast": nonfinite,
                "overflow_count": overflow,
                "flushed_to_zero_count": flushed,
            }
            nonfinite_total += nonfinite
            overflow_total += overflow
            flushed_total += flushed
    return {
        "target_dtype": str(out_dtype),
        "target_max": float(finfo.max),
        "target_smallest_normal": float(finfo.tiny),
        "nonfinite_after_cast": nonfinite_total,
        "overflow_count": overflow_total,
        "flushed_to_zero_count": flushed_total,
        "per_tensor": per_tensor,
    }


def lane_comparison(
    out: Path, index: dict[str, Any], mine_digest: str, random_init: set[str], compare_to: Path
) -> dict[str, Any]:
    """Compare the written checkpoint with another one written by `write_safetensors`.

    Per-tensor sha256 comes from both indexes; the other checkpoint's
    `state_sha256` is recomputed from its shards (tied aliases restored from
    its index) so the two digests are computed the same way. Differences are
    classified as random-init (interface tensors, seed-dependent) or
    source-derived (a real mismatch). The block is a record; it refuses nothing.
    """

    from chimera_omni.importing.safetensors_io import read_index, read_safetensors, state_sha256

    lane_index = read_index(compare_to)
    if lane_index is None:
        raise Refusal(f"--compare-to {compare_to} has no {INDEX_FILENAME}")
    lane_sha = dict(lane_index["metadata"].get("tensor_sha256", {}))
    mine_sha = dict(index["metadata"]["tensor_sha256"])
    shared = sorted(set(lane_sha) & set(mine_sha))
    identical = [name for name in shared if lane_sha[name] == mine_sha[name]]
    differing = [name for name in shared if lane_sha[name] != mine_sha[name]]
    only_in_lane = sorted(set(lane_sha) - set(mine_sha))
    only_in_mine = sorted(set(mine_sha) - set(lane_sha))

    views = read_safetensors(compare_to, verify=True)
    lane_state = {name: view.torch() for name, view in views.items()}
    for alias, owner in dict(lane_index["metadata"].get("tied_weights", {})).items():
        if owner in lane_state:
            lane_state[alias] = lane_state[owner]
    lane_digest = state_sha256(lane_state)
    del lane_state, views

    mine_config = json.load(open(out / CONFIG_FILENAME, encoding="utf-8"))
    lane_config_path = compare_to / CONFIG_FILENAME
    lane_config = json.load(open(lane_config_path, encoding="utf-8")) if lane_config_path.is_file() else None
    return {
        "checkpoint": str(compare_to),
        "index_sha256": file_sha256(compare_to / INDEX_FILENAME),
        "config_sha256": file_sha256(lane_config_path) if lane_config_path.is_file() else None,
        "tensors_in_lane": len(lane_sha),
        "tensors_in_mine": len(mine_sha),
        "shared_count": len(shared),
        "identical_count": len(identical),
        "identical": identical,
        "differing_count": len(differing),
        "differing": differing,
        "differing_random_init": sorted(name for name in differing if name in random_init),
        "differing_source_derived": sorted(name for name in differing if name not in random_init),
        "only_in_lane": only_in_lane,
        "only_in_mine": only_in_mine,
        "tied_weights_equal": dict(lane_index["metadata"].get("tied_weights", {})) == index["metadata"]["tied_weights"],
        "weight_map_equal": dict(lane_index["weight_map"]) == dict(index["weight_map"]),
        "total_size_equal": int(lane_index["metadata"].get("total_size", -1)) == int(index["metadata"]["total_size"]),
        "state_sha256_lane": lane_digest,
        "state_sha256_mine": mine_digest,
        "state_sha256_equal": lane_digest == mine_digest,
        "config_json_differing_paths": (
            json_diff_paths(lane_config, mine_config) if lane_config is not None else None
        ),
        "byte_identical": (
            lane_digest == mine_digest and not differing and not only_in_lane and not only_in_mine
        ),
    }


# --------------------------------------------------------------------------
# the import
# --------------------------------------------------------------------------


def run_import(args: argparse.Namespace) -> dict[str, Any]:
    started_wall = time.time()
    tree = bind_chimera_tree(args.chimera_src, args.chimera_commit)
    log(f"chimera tree {tree['tree']} @ {tree['commit']} ({tree['commit_source']})")

    import torch

    from chimera_omni.importing.profiles import (
        QWEN38_27B_VERIFIED_GEOMETRY,
        chimera_omni_27b_profile,
        qwen35_chimera_config,
    )
    from chimera_omni.importing.qwen35 import load_qwen35
    from chimera_omni.importing.safetensors_io import (
        DEFAULT_SHARD_BYTES,
        state_sha256,
        write_safetensors,
    )
    from chimera_omni.ledger import DEFAULT_FIXED_BANDS

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not (src / "config.json").is_file():
        raise Refusal(f"{src} has no config.json")

    # ---- every refusal below fires before anything is written -------------
    if args.shard_bytes is not None and int(args.shard_bytes) <= 0:
        raise Refusal(f"--shard-bytes must be positive, got {args.shard_bytes}")
    shard_bytes = int(DEFAULT_SHARD_BYTES if args.shard_bytes is None else args.shard_bytes)
    if args.dtype_out == "fp16" and not args.allow_fp16_narrowing:
        raise Refusal(
            "--dtype-out fp16 narrows bf16-range weights (max 65504, subnormals flushed); the campaign "
            "stores bf16. Pass --allow-fp16-narrowing to accept a range-checked fp16 write"
        )
    compare_to = Path(args.compare_to).resolve() if args.compare_to else None
    if compare_to is not None and not (compare_to / INDEX_FILENAME).is_file():
        raise Refusal(f"--compare-to {compare_to} has no {INDEX_FILENAME}")
    removed = prepare_out_dir(out, overwrite=bool(args.overwrite))
    if removed:
        log(f"--overwrite removed {len(removed)} file(s) from {out}: {', '.join(removed[:6])}{', ...' if len(removed) > 6 else ''}")

    extra_bands = tuple(args.ledger_band or ())
    default_names = {name for name, _ in DEFAULT_FIXED_BANDS}
    for name, _width in extra_bands:
        if name in default_names:
            raise Refusal(f"--ledger-band {name!r} collides with a DEFAULT_FIXED_BANDS band")
    bands = tuple(DEFAULT_FIXED_BANDS) + extra_bands

    with open(src / "config.json", "r", encoding="utf-8") as handle:
        hf_config = json.load(handle)
    profile = resolve_profile(hf_config, args.profile, dict(QWEN38_27B_VERIFIED_GEOMETRY))
    builder = chimera_omni_27b_profile if profile == "chimera_omni_27b" else qwen35_chimera_config
    log(f"profile {profile} (requested {args.profile})")
    try:
        cfg = builder(
            hf_config,
            ledger_bands=bands,
            allow_unverified_output_gate=args.allow_unverified_output_gate,
        )
    except ValueError as error:
        # The profile's own refusals (the declared output gate, a non-dense MLP, ...)
        # are refusals here too: nothing is written and the reason is the profile's.
        raise Refusal(f"config profile refused {src}: {error}") from error
    hidden = int(cfg.hidden_size)
    base_vocab = int(cfg.ledger_base_vocab_size)
    default_rows = sum(width for _, width in DEFAULT_FIXED_BANDS)
    extra_rows = sum(width for _, width in extra_bands)
    tables = 1 if cfg.tie_word_embeddings else 2

    # The RNG state is the whole story of the 16 interface tensors; pin it.
    torch.manual_seed(int(args.seed))
    log(f"seed {args.seed}; importing {src} on {args.device} in float32")
    started = time.time()
    model, load_receipt = load_qwen35(
        src,
        cfg,
        device=args.device,
        dtype=torch.float32,
        allow_unverified_output_gate=args.allow_unverified_output_gate,
    )
    model.eval()
    import_seconds = round(time.time() - started, 2)
    log(f"import done in {import_seconds}s; converting to {DTYPE_OUT[args.dtype_out]}")

    out_dtype = getattr(torch, DTYPE_OUT[args.dtype_out])
    narrowing: dict[str, Any] | None = None
    if args.dtype_out == "fp16":
        narrowing = narrowing_report(model, out_dtype)
        if narrowing["nonfinite_after_cast"] or narrowing["overflow_count"]:
            raise Refusal(
                f"fp16 narrowing would overflow: {narrowing['overflow_count']} value(s) above {FP16_MAX}, "
                f"{narrowing['nonfinite_after_cast']} non-finite after the cast; nothing written"
            )
        log(
            f"fp16 narrowing checked: overflow 0, flushed_to_zero {narrowing['flushed_to_zero_count']}, "
            "non-finite 0"
        )
    model = model.to(dtype=out_dtype)

    with torch.no_grad():
        embed_band_zero = bool((model.embed.weight[base_vocab:] == 0).all().item())
        head_band_zero = True
        if not cfg.tie_word_embeddings:
            head_band_zero = bool((model.lm_head.weight[base_vocab:] == 0).all().item())
        extra_embed_zero = bool((model.embed.weight[base_vocab + default_rows :] == 0).all().item())
    band_rows_zero = embed_band_zero and head_band_zero
    if not band_rows_zero:
        raise Refusal("band rows are not zero after the import; the checkpoint would lie about its bands")

    log("hashing the state")
    digest = state_sha256(model)

    # ---- the write; a refusal past this point unlinks the products ---------
    log(f"writing {out}")
    started = time.time()
    index: dict[str, Any] | None = None
    try:
        index = write_safetensors(
            model.state_dict(),
            out,
            config=cfg.to_manifest(),
            ledger=cfg.ledger().to_manifest(),
            # Exactly the import lane's source record, so a band-free run reproduces
            # its config.json up to `chimera_commit`.
            source={"import_source": "qwen3_5", "model_dir": str(src)},
            chimera_commit=tree["commit"],
            shard_bytes=shard_bytes,
        )
        write_seconds = round(time.time() - started, 2)
        verified = verify_written_checkpoint(out, index, cfg)
    except Refusal as error:
        marker = cleanup_after_refusal(out, index, str(error), list(args.argv))
        log(f"post-write refusal; removed {marker['removed']} and wrote {out / REFUSED_FILENAME}")
        raise
    counts = verified["counts"]
    header_total = sum(counts.values())
    model_total = sum(p.numel() for p in model.parameters())

    shards = verified["index_shard_files"]
    shard_files = {
        name: {"bytes": (out / name).stat().st_size, "sha256": file_sha256(out / name)} for name in shards
    }
    random_init = interface_tensors(list(load_receipt.get("initialised", [])))
    comparison: dict[str, Any] | None = None
    if compare_to is not None:
        log(f"comparing with {compare_to}")
        comparison = lane_comparison(out, index, digest, {item["name"] for item in random_init}, compare_to)
        log(
            f"lane comparison: identical {comparison['identical_count']}/{comparison['shared_count']}, "
            f"differing random-init {len(comparison['differing_random_init'])}, "
            f"differing source-derived {len(comparison['differing_source_derived'])}, "
            f"state_sha256 equal {comparison['state_sha256_equal']}"
        )

    peak = peak_rss_bytes()
    receipt: dict[str, Any] = dict(load_receipt)
    receipt.update(
        {
            "schema": RECEIPT_SCHEMA,
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "argv": list(args.argv)},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chimera": tree,
            "chimera_commit": tree["commit"],
            "profile": profile,
            "profile_requested": args.profile,
            "profile_note": "tensor bytes are profile-independent; only config.json config.metadata differs",
            "seed": int(args.seed),
            "import_dtype": "torch.float32",
            "dtype_out": DTYPE_OUT[args.dtype_out],
            "narrowing": narrowing,
            "output_gate_waiver": (
                None
                if str(cfg.metadata.get("attention_output_gate", "sigmoid")) == "sigmoid"
                else {
                    "flag": "--allow-unverified-output-gate",
                    "config_metadata_attention_output_gate": str(cfg.metadata.get("attention_output_gate")),
                    "q1_observation": str(cfg.metadata.get("attention_output_gate_observation", "")),
                }
            ),
            "interface_tensors": random_init,
            "ledger_bands": {
                "default": [[name, width] for name, width in DEFAULT_FIXED_BANDS],
                "declared_extra": [[name, width] for name, width in extra_bands],
                "all": [[name, width] for name, width in bands],
                "base_vocab": base_vocab,
                "vocab_size": int(cfg.vocab_size),
                "hidden_size": hidden,
                "tables": tables,
                "tie_word_embeddings": bool(cfg.tie_word_embeddings),
                "extra_rows": extra_rows,
                "extra_band_elements_declared": extra_rows * hidden * tables,
                "band_rows_zero": band_rows_zero,
                "extra_band_rows_zero": extra_embed_zero,
            },
            "checkpoint": {
                "directory": str(out),
                "schema": index["metadata"]["schema"],
                "shards": shards,
                "shard_bytes": shard_bytes,
                "shard_files": shard_files,
                "directory_shard_files": verified["directory_shard_files"],
                "index_shard_files": verified["index_shard_files"],
                "directory_shard_files_match_index": verified["directory_shard_files_match_index"],
                "overwrite": bool(args.overwrite),
                "overwrite_removed": removed,
                "total_size": int(index["metadata"]["total_size"]),
                "tensors": len(index["weight_map"]),
                "tied_weights": index["metadata"]["tied_weights"],
                "header_element_count": header_total,
                "header_element_count_source": "derived-from-safetensors-headers",
                "model_parameter_count": model_total,
                "state_sha256": digest,
                "tensor_sha256_count": len(index["metadata"]["tensor_sha256"]),
                "index_sha256": file_sha256(out / INDEX_FILENAME),
                "config_sha256": file_sha256(out / CONFIG_FILENAME),
            },
            "lane_comparison": comparison,
            "timing": {
                "import_seconds": import_seconds,
                "write_seconds": write_seconds,
                "wall_seconds": round(time.time() - started_wall, 2),
            },
            "peak_rss_bytes": peak,
            "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss * 1024",
        }
    )
    receipt_path = Path(args.receipt) if args.receipt else out / RECEIPT_FILENAME
    receipt_sha = write_receipt(receipt_path, receipt)
    summary = {
        "out": str(out),
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "chimera_commit": tree["commit"],
        "profile": profile,
        "header_element_count": header_total,
        "extra_band_elements_declared": extra_rows * hidden * tables,
        "band_rows_zero": band_rows_zero,
        "state_sha256": digest,
        "tensors": len(index["weight_map"]),
        "directory_shard_files_match_index": True,
        "lane_byte_identical": None if comparison is None else comparison["byte_identical"],
        "peak_rss_bytes": peak,
    }
    print(json.dumps(summary, sort_keys=True))
    return receipt


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="exact Qwen3.5/3.8 -> Chimera Omni import with the ledger band(s) declared",
    )
    parser.add_argument("--src", required=True, help="HF Qwen3.5/3.8 checkpoint directory")
    parser.add_argument("--out", required=True, help="Chimera checkpoint directory to write")
    parser.add_argument(
        "--chimera-commit",
        required=True,
        help="commit the chimera-omni tree must be at (refused otherwise; prefix match)",
    )
    parser.add_argument(
        "--chimera-src",
        default=os.environ.get("CHIMERA_OMNI_SRC", DEFAULT_CHIMERA_SRC),
        help="chimera-omni tree root (contains src/chimera_omni; $CHIMERA_OMNI_SRC)",
    )
    parser.add_argument("--allow-unverified-output-gate", action="store_true")
    parser.add_argument(
        "--ledger-band",
        action="append",
        type=parse_band,
        default=None,
        metavar="NAME:WIDTH",
        help="extra fixed band appended above DEFAULT_FIXED_BANDS (repeatable), e.g. music_codes:65536",
    )
    parser.add_argument("--dtype-out", choices=sorted(DTYPE_OUT), default="bf16")
    parser.add_argument(
        "--allow-fp16-narrowing",
        action="store_true",
        help="accept --dtype-out fp16; the receipt then records per-tensor max|w|, overflow, flush and non-finite counts",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="auto",
        help=(
            "config builder: auto (chimera_omni_27b_profile when the source carries the verified 27B geometry, "
            "else qwen35_chimera_config), or name one explicitly"
        ),
    )
    parser.add_argument("--seed", type=int, default=TORCH_DEFAULT_SEED)
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=None,
        help="bytes per shard (default: safetensors_io.DEFAULT_SHARD_BYTES); 0 or negative is refused",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        metavar="CHECKPOINT",
        help="another write_safetensors checkpoint to compare with; writes the receipt's lane_comparison block",
    )
    parser.add_argument("--receipt", default=None, help=f"receipt path (default <out>/{RECEIPT_FILENAME})")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="unlink the previous model-*.safetensors / index / config / receipt / REFUSED.json at the top of --out first",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    try:
        run_import(args)
    except Refusal as error:
        print(f"[{TOOL_NAME}] REFUSED: {error}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
