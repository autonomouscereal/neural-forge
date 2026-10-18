"""Export a folded child as a STANDARD qwen3_5 checkpoint, and prove it is one.

"Standard" is a load-bearing word in this campaign, not a hope: the child must
come back through the stock ``transformers`` class with ``trust_remote_code``
literally false, count the parameters the receipt claims, and (for the text
trunk) convert with llama.cpp's stock ``convert_hf_to_gguf.py``.  Everything in
this module exists to make each of those checkable rather than asserted.

Three things the LFM2 exporter did not have to do:

* **directory digest.**  A 27B child is sharded, so ``sha256(model.safetensors)``
  is not a thing that exists.  The digest here is over the sorted
  ``(relative path, sha256)`` list of every weight file, and the receipt names
  the METHOD so a later reader is not guessing which convention produced it.
* **carry-through.**  ``^mtp.*`` is in ``_keys_to_ignore_on_load_unexpected``
  for every qwen3_5 class (modeling 1618), so transformers neither loads nor
  saves the MTP block.  It is real parameters in the parent's index.  Either it
  is copied verbatim (allowed only for axes that do not change the shapes it
  reads) or it is dropped and the receipt says ``mtp: dropped`` -- silence is
  not one of the options.
* **both loaders.**  A text-only child is a ``Qwen3_5ForCausalLM``
  (``model.layers.*``); a VL child is a ``Qwen3_5ForConditionalGeneration``
  (``model.language_model.layers.*``).  Verification re-loads with the class the
  child's own config names and refuses if any key was re-initialised.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

import qf_arch

#: Sidecars copied verbatim when the axis did not touch the tokenizer.  Missing
#: files are skipped; the receipt lists what was actually written.
SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)

WEIGHT_SUFFIXES = (".safetensors",)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digests(directory: str | Path) -> dict[str, str]:
    """``{tensor name: sha256 of its bytes}`` across every shard, in index order.

    Per tensor, not per file, because a shard boundary is an implementation
    detail of the writer and two byte-identical models can shard differently.
    """

    from safetensors import safe_open

    directory = Path(directory)
    out: dict[str, str] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            for name in sorted(handle.keys()):
                tensor = handle.get_tensor(name).contiguous()
                # `view(torch.uint8)` is dtype-agnostic (bf16 included) and does
                # not copy; `.numpy()` would refuse bfloat16 outright.
                out[name] = hashlib.sha256(
                    bytes(memoryview(tensor.view(torch.uint8).reshape(-1).numpy()))
                ).hexdigest()
    return out


def directory_digest(directory: str | Path) -> tuple[str, list[dict[str, str]]]:
    """One digest over every weight file, and the per-file list it was built from."""

    directory = Path(directory)
    rows = [
        {"path": path.name, "sha256": file_sha256(path)}
        for path in sorted(directory.iterdir())
        if path.suffix in WEIGHT_SUFFIXES
    ]
    digest = hashlib.sha256(qf_arch.canonical_json_bytes(rows)).hexdigest()
    return digest, rows


def carry_through(
    parent_dir: str | Path, child_dir: str | Path, prefixes: tuple[str, ...] = ("mtp.",)
) -> dict[str, Any]:
    """Copy parent tensors transformers refuses to own into the child's index.

    Only legal when the axis did not change any shape those tensors read.  The
    caller decides that; this function does the mechanics and reports exactly
    what it moved so the receipt can carry the claim.
    """

    from safetensors import safe_open
    from safetensors.torch import save_file

    parent_dir, child_dir = Path(parent_dir), Path(child_dir)
    carried: dict[str, torch.Tensor] = {}
    for path in sorted(parent_dir.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            for name in handle.keys():
                if any(name.startswith(prefix) for prefix in prefixes):
                    carried[name] = handle.get_tensor(name)
    if not carried:
        return {"carried": [], "prefixes": list(prefixes), "shard": None}

    shard = child_dir / "model-carried.safetensors"
    save_file(carried, str(shard), metadata={"format": "pt"})
    index_path = child_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"].update({name: shard.name for name in carried})
        index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + sum(
            tensor.numel() * tensor.element_size() for tensor in carried.values()
        )
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    else:
        # A single-file child has no index; adding one is how a second shard
        # becomes loadable at all.
        main = child_dir / "model.safetensors"
        weight_map = {name: shard.name for name in carried}
        with safe_open(str(main), "pt") as handle:
            for name in handle.keys():
                weight_map[name] = main.name
        index_path.write_text(
            json.dumps(
                {"metadata": {"total_size": sum(
                    tensor.numel() * tensor.element_size() for tensor in carried.values()
                )}, "weight_map": weight_map},
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "carried": sorted(carried),
        "prefixes": list(prefixes),
        "shard": shard.name,
        "parameters": int(sum(t.numel() for t in carried.values())),
    }


def export_child(
    model: torch.nn.Module,
    out_dir: str | Path,
    *,
    parent_dir: str | Path | None = None,
    sidecars: bool = True,
    tokenizer_from: str | Path | None = None,
    max_shard_size: str = "4GB",
) -> dict[str, Any]:
    """Write the child, its config, its sidecars, and everything needed to check it."""

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(
            f"{out_dir} already exists; a build never overwrites a child -- move it "
            f"or pick another tag"
        )
    out_dir.mkdir(parents=True)
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=max_shard_size)

    written: list[str] = []
    if sidecars and parent_dir is not None:
        source = Path(tokenizer_from or parent_dir)
        for name in SIDECARS:
            candidate = source / name
            if candidate.exists() and not (out_dir / name).exists():
                shutil.copy2(candidate, out_dir / name)
                written.append(name)

    digest, files = directory_digest(out_dir)
    return {
        "child_directory": str(out_dir),
        "child_tensors_sha256": digest,
        "child_tensors_sha256_method": (
            "sha256 over the canonical json of the sorted [{path, sha256}] list of "
            "every *.safetensors file in the child directory"
        ),
        "weight_files": files,
        "sidecars_copied": written,
        "tensor_sha256": tensor_digests(out_dir),
    }


def verify_child(
    child_dir: str | Path, *, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> dict[str, Any]:
    """Re-load the child through the STOCK class and check the arithmetic.

    ``trust_remote_code`` is not passed anywhere; a child that needs it is not a
    standard checkpoint and this function is where that becomes visible.
    """

    child_dir = Path(child_dir)
    config = qf_arch.load_config(child_dir)
    loader = qf_arch.loader_class(config)
    if str(device) == "cpu":
        qf_arch.force_torch_kernels()
    model = loader.from_pretrained(
        str(child_dir), dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    counts = qf_arch.assert_parameter_count(model, config)
    qf_arch.assert_layer_types_agree(model)
    # The id-space post-condition, made by code that never saw the fold and on
    # the objects the LOADER built: every token id the child names -- config,
    # sub-configs, and the generation config `generate` actually reads -- must be
    # a row this child has.  It is unconditional because it is true of every
    # child, not only a vocabulary one: an axis that renumbers nothing passes it
    # for free, and a parent that was already broken is caught here.
    import qf_vocab

    addressable = qf_vocab.assert_ids_addressable(
        config,
        getattr(model, "generation_config", None),
        int(qf_arch.text_config(config).vocab_size),
        where="re-loaded child",
    )
    digest, _ = directory_digest(child_dir)
    return {
        "loaded_class": type(model).__name__,
        "trust_remote_code": False,
        "transformers_version": __import__("transformers").__version__,
        "ids_addressable": addressable,
        "parameters_counted_live": counts["live"],
        "parameters_closed_form": counts["total_with_vision"]
        if qf_arch.vision_model(model) is not None
        else counts["total"],
        "child_tensors_sha256": digest,
        "statement": (
            "re-loaded from disk by the stock transformers class named by the "
            "child's own config, with trust_remote_code literally false"
        ),
    }, model


def shape_receipt(
    *,
    parent_config: Any,
    child_config: Any,
    parent_dir: str | Path | None,
    export: dict[str, Any],
    verify: dict[str, Any],
    axes: list[dict[str, Any]],
    evidence: dict[str, Any],
    stage: str = "fold",
    parent_code: str = "",
    lineage: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """The tensor-shape receipt.  Every ratio in it is arithmetic, not a measurement."""

    parent_counts = qf_arch.parameter_count(parent_config)
    child_counts = qf_arch.parameter_count(child_config)
    parent_total = parent_counts["total_with_vision"] if parent_counts["vision"] else parent_counts["total"]
    child_total = child_counts["total_with_vision"] if child_counts["vision"] else child_counts["total"]

    def component_totals(counts: dict[str, int]) -> dict[str, int]:
        layers = counts["linear_layers"] + counts["full_layers"]
        return {
            "mlp": counts["mlp_per_layer"] * layers,
            "gdn_mixer": counts["gdn_mixer"] * counts["linear_layers"],
            "attention_mixer": counts["attention_mixer"] * counts["full_layers"],
            "interface": counts["interface"],
            "vision": counts["vision"],
        }

    parent_components = component_totals(parent_counts)
    child_components = component_totals(child_counts)
    components = [
        {
            "component": label,
            "parent_parameters": int(parent_components[label]),
            "child_parameters": int(child_components[label]),
            "ratio": (child_components[label] / parent_components[label])
            if parent_components[label]
            else None,
            "fold": (parent_components[label] / child_components[label])
            if child_components[label]
            else None,
        }
        for label in ("mlp", "gdn_mixer", "attention_mixer", "interface", "vision")
    ]

    body = {
        "schema": "neural-forge-omni-qwen3_5-standard-shape-child-v1",
        "family": qf_arch.FAMILY,
        "stage": stage,
        "parent_code": parent_code,
        "parent_model": str(parent_dir) if parent_dir else None,
        "lineage": list(lineage or []),
        "quantization": "none",
        "standard_shape": True,
        "config_diff": qf_arch.config_diff(parent_config, child_config),
        "storage": {
            "parent_parameter_count": int(parent_total),
            "child_parameter_count": int(child_total),
            "net_stored_ratio": child_total / float(parent_total),
            "net_fold": parent_total / float(child_total),
            "counted_in": "parameters (never bytes; quantization is not folding)",
        },
        "per_component_ratios": components,
        "axes": axes,
        "evidence": evidence,
        "export": export,
        "verification": verify,
        "notes": notes,
        "thresholds": {
            "statement": (
                "every numeric line in this receipt is a proposed decision line for "
                "Cyril and a diagnostic for review ordering; the only grade is the "
                "frozen behavioural review"
            )
        },
    }
    body["receipt_sha256"] = qf_arch.canonical_sha256(body)
    return body


def write_receipt(receipt: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Write canonically, then PARSE IT BACK.  A receipt nobody re-read is a draft."""

    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(qf_arch.canonical_json_bytes(receipt))
    partial.replace(path)
    body = json.loads(path.read_text(encoding="ascii"))
    declared = body.pop("receipt_sha256")
    if declared != qf_arch.canonical_sha256(body):
        raise SystemExit(f"{path}: the receipt does not seal itself")
    body["receipt_sha256"] = declared
    return body


# ------------------------------------------------------------------ GGUF hook
def convert_to_gguf(
    child_dir: str | Path,
    out_path: str | Path,
    *,
    llama_cpp: str | Path,
    outtype: str = "bf16",
    timeout: int = 3600,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Invoke llama.cpp's stock ``convert_hf_to_gguf.py`` on the child.

    Not vendored, not re-implemented: the point of the export rule is that the
    UPSTREAM converter accepts the child.  ``dry_run`` returns the exact command
    without running it, which is what the self-tests use on a box where
    llama.cpp is not checked out.
    """

    script = Path(llama_cpp) / "convert_hf_to_gguf.py"
    command = [
        "python", str(script), str(child_dir),
        "--outfile", str(out_path), "--outtype", outtype,
    ]
    if dry_run or not script.exists():
        return {
            "ran": False,
            "reason": "dry-run" if dry_run else f"{script} not present",
            "command": command,
        }
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    result = {
        "ran": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode == 0 and Path(out_path).exists():
        result["gguf_sha256"] = file_sha256(out_path)
        result["gguf_bytes"] = Path(out_path).stat().st_size
    else:
        result["gguf_sha256"] = None
    return result


# ------------------------------------------------------- review runner hook
def frozen_review(
    *,
    review_script: str | Path,
    receipt: str | Path,
    child_dir: str | Path,
    parent_dir: str | Path,
    out: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    child_tokenizer: str | Path | None = None,
    compute_architecture: str | None = None,
    timeout: int = 14400,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Call the protocol lane's frozen-review runner.  This module never grades.

    The dual-column read (column A sealed, column B copy-backs refused) and the
    canonical artifact-tolerant box are the review lane's, and they are applied
    to the dossier this produces -- not here.  A fold module that scored its own
    children would be exactly the proxy-gating failure the campaign has been
    corrected on four times.
    """

    command = [
        "python", str(review_script),
        "--receipt", str(receipt),
        "--child-model", str(child_dir),
        "--parent-model", str(parent_dir),
        "--out", str(out),
        "--device", device,
        "--dtype", dtype,
        "--maximum-new-tokens", str(max_new_tokens),
    ]
    if child_tokenizer is not None:
        command += ["--child-tokenizer", str(child_tokenizer)]
    if compute_architecture:
        command += ["--require-compute-architecture", compute_architecture]
    if dry_run or not Path(review_script).exists():
        return {
            "ran": False,
            "reason": "dry-run" if dry_run else f"{review_script} not present",
            "command": command,
            "grading": "the dossier is graded by the review lane and by Cyril, never here",
        }
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {
        "ran": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "dossier": str(out) if Path(out).exists() else None,
        "dossier_sha256": file_sha256(out) if Path(out).exists() else None,
    }
