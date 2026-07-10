#!/usr/bin/env python3
"""Snapshot and detach a zero-gradient Qwen pair-composition evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from experiments.launch_detached import launch_detached
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.launch_detached import launch_detached


REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_BEHAVIOR_PROBES = 1000
EXACT_GENERATION_PROMPTS = 1000
EXACT_CHOICE_PROBES = 250
MANUAL_PROMPTS = (
    "Q: What is 6+4? A:",
    "Q: If a box has 3 red balls and 2 blue balls, how many balls are there? A:",
    "Write a Python function that adds two numbers.",
    "Briefly explain tokenizer mismatch in cross-family model merging.",
    "What should happen before pruning neurons in a merge-first compression pipeline?",
    "A person hears thunder after seeing lightning. What should they do?",
    "Explain one risk when merging two neural models.",
)
_FORBIDDEN_COMPOSE_FLAG_PARTS = (
    "recover",
    "train",
    "optimizer",
    "learning-rate",
    "steps",
    "epochs",
)


@dataclass(frozen=True)
class ComposeSettings:
    """Closed-form controls accepted by the future pair-composition command."""

    ridge: float = 1e-3
    max_length: int = 160
    max_tokens_per_layer: int = 4096
    max_anchors: int = 4096
    max_calibration_texts: int = 256
    basis_mode: str = "activation-pca"
    basis_preprocess: str = "uncentered-rowrms"
    activation_domain_balance: str = "keyword"
    fit_gate_validation_fraction: float = 0.25
    fit_gate_min_validation_rows: int = 128
    fit_gate_min_relative_improvement: float = 0.02
    fit_gate_max_cosine_drop: float = 0.0
    dtype: str = "float16"
    candidate_dtype: str = "float32"
    device: str = "cuda:0"

    def validate(self) -> None:
        if not math.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be non-negative.")
        for name in (
            "max_length",
            "max_tokens_per_layer",
            "max_anchors",
            "max_calibration_texts",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be positive.")
        if not 0.0 < self.fit_gate_validation_fraction < 1.0:
            raise ValueError(
                "fit-gate-validation-fraction must be between zero and one."
            )
        if self.fit_gate_min_validation_rows <= 0:
            raise ValueError("fit-gate-min-validation-rows must be positive.")
        if (
            not math.isfinite(self.fit_gate_min_relative_improvement)
            or self.fit_gate_min_relative_improvement < 0
        ):
            raise ValueError("fit-gate-min-relative-improvement must be non-negative.")
        if (
            not math.isfinite(self.fit_gate_max_cosine_drop)
            or self.fit_gate_max_cosine_drop < 0
        ):
            raise ValueError("fit-gate-max-cosine-drop must be non-negative.")
        if not self.dtype or not self.candidate_dtype or not self.device:
            raise ValueError("dtype, candidate-dtype, and device cannot be empty.")
        if self.basis_mode not in {"activation-pca", "embedding-grouped"}:
            raise ValueError("basis-mode must be activation-pca or embedding-grouped.")
        if self.basis_preprocess not in {
            "centered-rowrms",
            "uncentered-rowrms",
            "uncentered",
        }:
            raise ValueError("unsupported basis-preprocess.")
        if self.activation_domain_balance not in {"none", "keyword"}:
            raise ValueError("activation-domain-balance must be none or keyword.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argv_sha256(command: Iterable[str]) -> str:
    return hashlib.sha256(
        "\0".join(str(item) for item in command).encode("utf-8")
    ).hexdigest()


def _assert_no_recovery_or_training_flags(command: Iterable[str]) -> None:
    for token in command:
        if not str(token).startswith("--"):
            continue
        normalized = str(token).lower()
        if any(part in normalized for part in _FORBIDDEN_COMPOSE_FLAG_PARTS):
            raise ValueError(
                f"pair composition cannot accept recovery or training flag {token!r}."
            )


def build_compose_command(
    *,
    python: str | Path,
    source: Path,
    blank_candidate: Path,
    output: Path,
    calibration_file: Path,
    settings: ComposeSettings,
) -> list[str]:
    """Build the closed-form compose argv without any optimization controls."""

    settings.validate()
    command = [
        str(python),
        "-m",
        "neural_forge.cli",
        "hf-qwen-pair-compose",
        str(source),
        str(blank_candidate),
        str(output),
        "--ridge",
        str(settings.ridge),
        "--max-length",
        str(settings.max_length),
        "--max-tokens-per-layer",
        str(settings.max_tokens_per_layer),
        "--max-anchors",
        str(settings.max_anchors),
        "--basis-mode",
        settings.basis_mode,
        "--basis-preprocess",
        settings.basis_preprocess,
        "--activation-domain-balance",
        settings.activation_domain_balance,
        "--calibration-file",
        str(calibration_file),
        "--max-calibration-texts",
        str(settings.max_calibration_texts),
        "--fit-gate-validation-fraction",
        str(settings.fit_gate_validation_fraction),
        "--fit-gate-min-validation-rows",
        str(settings.fit_gate_min_validation_rows),
        "--fit-gate-min-relative-improvement",
        str(settings.fit_gate_min_relative_improvement),
        "--fit-gate-max-cosine-drop",
        str(settings.fit_gate_max_cosine_drop),
        "--dtype",
        settings.dtype,
        "--candidate-dtype",
        settings.candidate_dtype,
        "--device",
        settings.device,
    ]
    _assert_no_recovery_or_training_flags(command)
    return command


def build_eval_command(
    *,
    python: str | Path,
    model: Path,
    report: Path,
    eval_suite: Path,
    device: str,
    dtype: str,
) -> list[str]:
    """Build the one allowed child evaluation at the fixed evidence counts."""

    return [
        str(python),
        "-m",
        "neural_forge.cli",
        "hf-eval",
        str(model),
        str(report),
        "--eval-suite",
        str(eval_suite),
        "--max-behavior-probes",
        str(EXACT_BEHAVIOR_PROBES),
        "--max-generation-prompts",
        str(EXACT_GENERATION_PROMPTS),
        "--max-choice-probes",
        str(EXACT_CHOICE_PROBES),
        "--max-text-evals",
        "128",
        "--max-new-tokens",
        "96",
        "--device",
        device,
        "--dtype",
        dtype,
    ]


def _copy_with_reflink_fallback(source: Path, target: Path) -> str:
    """Create an independent copy, using Linux CoW cloning when supported."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # type: ignore[import-not-found]

        ficlone = 0x40049409
        with source.open("rb") as source_handle, target.open("xb") as target_handle:
            fcntl.ioctl(target_handle.fileno(), ficlone, source_handle.fileno())
        shutil.copystat(source, target, follow_symlinks=True)
        return "reflink"
    except (ImportError, OSError):
        target.unlink(missing_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
        return "copy"


def _snapshot_file(
    source: Path,
    target: Path,
    *,
    output_root: Path,
    role: str,
) -> dict[str, object]:
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"snapshot input is not a regular file: {source}")
    method = _copy_with_reflink_fallback(source, target)
    return {
        "role": role,
        "source": str(source),
        "path": target.resolve().relative_to(output_root.resolve()).as_posix(),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
        "materialization": method,
    }


def _snapshot_tree(
    source: Path,
    target: Path,
    *,
    output_root: Path,
    role: str,
    exclude_bytecode: bool = False,
) -> list[dict[str, object]]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"snapshot input is not a directory: {source}")
    records: list[dict[str, object]] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source)
        if exclude_bytecode and (
            "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"snapshot tree contains a non-regular file: {path}")
        records.append(
            _snapshot_file(
                resolved,
                target / relative,
                output_root=output_root,
                role=role,
            )
        )
    if not records:
        raise ValueError(f"snapshot tree is empty: {source}")
    return records


def _tree_sha256(records: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_checkpoint(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise ValueError(
            f"{label} must be a local checkpoint directory containing config.json."
        )
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    if not any(any(resolved.glob(pattern)) for pattern in weight_patterns):
        raise ValueError(f"{label} does not contain local checkpoint weights.")
    return resolved


def _validate_nonempty_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty file.")
    return resolved


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without dereferencing a venv executable symlink."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _validate_cached_parent(path: Path) -> Path:
    resolved = _validate_nonempty_file(path, "parent-eval")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("parent-eval must be a valid cached JSON report.") from exc
    if (
        not isinstance(report, dict)
        or "behavior_suite" not in report
        or "choice_accuracy" not in report
    ):
        raise ValueError("parent-eval is not a completed cached HF evaluation report.")
    return resolved


def _validate_tag(tag: str) -> str:
    if not tag or tag in {".", ".."} or "/" in tag or "\\" in tag or "\0" in tag:
        raise ValueError("tag must be one non-empty path component.")
    return tag


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _render_manual_program() -> str:
    prompts_json = json.dumps(MANUAL_PROMPTS)
    return f"""import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_path, device, dtype_name = sys.argv[1:5]
prompts = {prompts_json}
dtype = getattr(torch, dtype_name)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=dtype,
    local_files_only=True,
    trust_remote_code=False,
).to(device)
model.eval()
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id
items = []
for prompt in prompts:
    encoded = tokenizer(prompt, return_tensors=\"pt\").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    items.append({{\"prompt\": prompt, \"output\": tokenizer.decode(generated[0], skip_special_tokens=True)}})
Path(out_path).write_text(
    json.dumps({{\"model\": model_dir, \"manual_generations\": items}}, indent=2) + \"\\n\",
    encoding=\"utf-8\",
)
"""


def render_run_script(
    *,
    python: Path,
    output: Path,
    snapshot_manifest: Path,
    snapshot_manifest_sha256: str,
    snapshot_pythonpath: Path,
    compose_command: list[str],
    eval_command: list[str],
    output_model: Path,
    child_eval: Path,
    manual_report: Path,
    cached_parent_eval: Path,
    comparison_script: Path,
    retention_report: Path,
    device_uuid: str,
    device: str,
    dtype: str,
    wait_pids: list[int],
    wait_poll_seconds: int,
) -> str:
    """Render the complete immutable compose/eval/manual/retention chain."""

    verifier = r"""import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
expected_manifest_sha256 = sys.argv[3]
manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if manifest_digest != expected_manifest_sha256:
    raise SystemExit("snapshot manifest hash mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = (output / item["path"]).resolve()
    try:
        path.relative_to(output)
    except ValueError as exc:
        raise SystemExit(f"snapshot path escaped output root: {path}") from exc
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise SystemExit(f"snapshot file missing or resized: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise SystemExit(f"snapshot hash mismatch: {path}")
print(json.dumps({"snapshot_verified": len(manifest["files"]), "tree_sha256": manifest["tree_sha256"]}))
"""
    wait_values = " ".join(str(pid) for pid in wait_pids)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "umask 077",
        f"OUT={shlex.quote(str(output))}",
        f"PY={shlex.quote(str(python))}",
        f"SNAPSHOT_MANIFEST={shlex.quote(str(snapshot_manifest))}",
        f"SNAPSHOT_MANIFEST_SHA256={shlex.quote(snapshot_manifest_sha256)}",
        f"OUTPUT_MODEL={shlex.quote(str(output_model))}",
        f"CHILD_EVAL={shlex.quote(str(child_eval))}",
        f"MANUAL_REPORT={shlex.quote(str(manual_report))}",
        f"CACHED_PARENT_EVAL={shlex.quote(str(cached_parent_eval))}",
        f"RETENTION_REPORT={shlex.quote(str(retention_report))}",
        f"WAIT_PIDS={shlex.quote(wait_values)}",
        f"WAIT_POLL_SECONDS={int(wait_poll_seconds)}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(device_uuid)}",
        f"export PYTHONPATH={shlex.quote(str(snapshot_pythonpath))}",
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export TOKENIZERS_PARALLELISM=false",
        'echo "pair_compose_chain_start=$(date -Is)"',
        '"$PY" - "$SNAPSHOT_MANIFEST" "$OUT" "$SNAPSHOT_MANIFEST_SHA256" <<\'PY\'',
        verifier.rstrip(),
        "PY",
        "for wait_pid in $WAIT_PIDS; do",
        '  while kill -0 "$wait_pid" 2>/dev/null; do',
        '    echo "$(date -Is) waiting_for_prior_pid=$wait_pid"',
        '    sleep "$WAIT_POLL_SECONDS"',
        "  done",
        "done",
        f'echo "pair_compose_start=$(date -Is) device_uuid={device_uuid}"',
        shlex.join(compose_command),
        '[[ -s "$OUTPUT_MODEL/config.json" ]] || { echo "missing composed model config" >&2; exit 2; }',
        'echo "pair_compose_done=$(date -Is)"',
        shlex.join(eval_command),
        '[[ -s "$CHILD_EVAL" ]] || { echo "missing child evaluation" >&2; exit 2; }',
        '"$PY" - "$OUTPUT_MODEL" "$MANUAL_REPORT" '
        + shlex.quote(device)
        + " "
        + shlex.quote(dtype)
        + " <<'PY'",
        _render_manual_program().rstrip(),
        "PY",
        shlex.join(
            [
                str(python),
                str(comparison_script),
                str(cached_parent_eval),
                str(child_eval),
                "--out",
                str(retention_report),
                "--max-samples",
                "12",
            ]
        ),
        '[[ -s "$MANUAL_REPORT" && -s "$RETENTION_REPORT" ]] || { echo "missing post-compose evidence" >&2; exit 2; }',
        'echo "pair_compose_chain_done=$(date -Is)"',
        "",
    ]
    script = "\n".join(lines)
    if script.count("hf-eval") != 1:
        raise AssertionError(
            "run script must contain exactly one child hf-eval command."
        )
    if "hf-qwen-pair-compose" not in script:
        raise AssertionError("run script is missing hf-qwen-pair-compose.")
    _assert_no_recovery_or_training_flags(compose_command)
    return script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "--source-model", dest="source", type=Path, required=True
    )
    parser.add_argument(
        "--blank-candidate",
        "--candidate",
        dest="blank_candidate",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument(
        "--eval-suite", "--eval-suite-file", dest="eval_suite", type=Path, required=True
    )
    parser.add_argument("--parent-eval", type=Path, required=True)
    parser.add_argument(
        "--device-uuid", "--gpu-uuid", dest="device_uuid", required=True
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-tokens-per-layer", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=4096)
    parser.add_argument("--max-calibration-texts", type=int, default=256)
    parser.add_argument("--basis-mode", default="activation-pca")
    parser.add_argument("--basis-preprocess", default="uncentered-rowrms")
    parser.add_argument("--activation-domain-balance", default="keyword")
    parser.add_argument("--fit-gate-validation-fraction", type=float, default=0.25)
    parser.add_argument("--fit-gate-min-validation-rows", type=int, default=128)
    parser.add_argument("--fit-gate-min-relative-improvement", type=float, default=0.02)
    parser.add_argument("--fit-gate-max-cosine-drop", type=float, default=0.0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--candidate-dtype", default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tag = _validate_tag(args.tag)
    if (
        not args.device_uuid
        or any(character.isspace() for character in args.device_uuid)
        or "\0" in args.device_uuid
    ):
        raise ValueError("device-uuid must be one non-empty CUDA device identifier.")
    if args.wait_poll_seconds <= 0:
        raise ValueError("wait-poll-seconds must be positive.")
    if any(pid <= 0 for pid in args.wait_pid):
        raise ValueError("wait-pid values must be positive.")

    source = _validate_checkpoint(args.source, "source")
    blank_candidate = _validate_checkpoint(args.blank_candidate, "blank-candidate")
    if source == blank_candidate:
        raise ValueError("source and blank-candidate must be distinct checkpoints.")
    run_root = args.run_root.resolve(strict=True)
    if not run_root.is_dir():
        raise ValueError("run-root must be an existing directory.")
    calibration_file = _validate_nonempty_file(
        args.calibration_file, "calibration-file"
    )
    eval_suite = _validate_nonempty_file(args.eval_suite, "eval-suite")
    parent_eval = _validate_cached_parent(args.parent_eval)
    python = _lexical_absolute(args.python)
    if not python.is_file():
        raise ValueError("python must resolve to an interpreter file.")

    settings = ComposeSettings(
        ridge=args.ridge,
        max_length=args.max_length,
        max_tokens_per_layer=args.max_tokens_per_layer,
        max_anchors=args.max_anchors,
        max_calibration_texts=args.max_calibration_texts,
        basis_mode=args.basis_mode,
        basis_preprocess=args.basis_preprocess,
        activation_domain_balance=args.activation_domain_balance,
        fit_gate_validation_fraction=args.fit_gate_validation_fraction,
        fit_gate_min_validation_rows=args.fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement=args.fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop=args.fit_gate_max_cosine_drop,
        dtype=args.dtype,
        candidate_dtype=args.candidate_dtype,
        device=args.device,
    )
    settings.validate()

    output = run_root / tag
    unresolved_output = output.resolve()
    if unresolved_output.is_relative_to(source) or unresolved_output.is_relative_to(
        blank_candidate
    ):
        raise ValueError(
            "run output cannot be nested inside either checkpoint snapshot source."
        )
    output.mkdir(exist_ok=False)
    reports = output / "reports"
    reports.mkdir()
    snapshot = output / "snapshot"
    snapshot.mkdir()
    snapshot_inputs = snapshot / "inputs"
    snapshot_code = snapshot / "code"

    records: list[dict[str, object]] = []
    records.extend(
        _snapshot_tree(
            source,
            snapshot_inputs / "source",
            output_root=output,
            role="source_checkpoint",
        )
    )
    records.extend(
        _snapshot_tree(
            blank_candidate,
            snapshot_inputs / "blank_candidate",
            output_root=output,
            role="blank_candidate_checkpoint",
        )
    )
    snapshot_calibration = snapshot_inputs / "calibration" / calibration_file.name
    snapshot_eval_suite = snapshot_inputs / "eval_suite" / eval_suite.name
    snapshot_parent_eval = snapshot_inputs / "cached_parent_eval" / parent_eval.name
    records.append(
        _snapshot_file(
            calibration_file,
            snapshot_calibration,
            output_root=output,
            role="calibration_file",
        )
    )
    records.append(
        _snapshot_file(
            eval_suite, snapshot_eval_suite, output_root=output, role="eval_suite"
        )
    )
    records.append(
        _snapshot_file(
            parent_eval,
            snapshot_parent_eval,
            output_root=output,
            role="cached_parent_eval",
        )
    )

    package_source = REPO_ROOT / "src" / "neural_forge"
    comparison_source = REPO_ROOT / "experiments" / "compare_eval_retention.py"
    records.extend(
        _snapshot_tree(
            package_source,
            snapshot_code / "src" / "neural_forge",
            output_root=output,
            role="neural_forge_code",
            exclude_bytecode=True,
        )
    )
    snapshot_comparison = snapshot_code / "experiments" / "compare_eval_retention.py"
    records.append(
        _snapshot_file(
            comparison_source,
            snapshot_comparison,
            output_root=output,
            role="retention_code",
        )
    )
    records.append(
        _snapshot_file(
            Path(__file__),
            snapshot_code / "experiments" / Path(__file__).name,
            output_root=output,
            role="launcher_code",
        )
    )
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        records.append(
            _snapshot_file(
                pyproject,
                snapshot_code / "pyproject.toml",
                output_root=output,
                role="project_metadata",
            )
        )

    snapshot_source = snapshot_inputs / "source"
    snapshot_blank = snapshot_inputs / "blank_candidate"
    output_model = output / "model"
    child_eval = reports / "eval_1000x1000x250.json"
    manual_report = reports / "manual_generations.json"
    retention_report = reports / "retention_vs_cached_parent.json"
    compose_command = build_compose_command(
        python=python,
        source=snapshot_source,
        blank_candidate=snapshot_blank,
        output=output_model,
        calibration_file=snapshot_calibration,
        settings=settings,
    )
    eval_command = build_eval_command(
        python=python,
        model=output_model,
        report=child_eval,
        eval_suite=snapshot_eval_suite,
        device=settings.device,
        dtype=settings.dtype,
    )

    snapshot_manifest_path = snapshot / "snapshot_manifest.json"
    snapshot_manifest = {
        "schema": "neural-forge-qwen-pair-compose-snapshot-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "tree_sha256": _tree_sha256(records),
    }
    snapshot_manifest_path.write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_manifest_sha256 = _sha256(snapshot_manifest_path)
    run_script = output / "run_snapshot.sh"
    script_text = render_run_script(
        python=python,
        output=output,
        snapshot_manifest=snapshot_manifest_path,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        snapshot_pythonpath=snapshot_code / "src",
        compose_command=compose_command,
        eval_command=eval_command,
        output_model=output_model,
        child_eval=child_eval,
        manual_report=manual_report,
        cached_parent_eval=snapshot_parent_eval,
        comparison_script=snapshot_comparison,
        retention_report=retention_report,
        device_uuid=args.device_uuid,
        device=settings.device,
        dtype=settings.dtype,
        wait_pids=list(args.wait_pid),
        wait_poll_seconds=args.wait_poll_seconds,
    )
    run_script.write_text(script_text, encoding="utf-8", newline="\n")
    os.chmod(run_script, 0o500)

    environment = {
        "CUDA_VISIBLE_DEVICES": args.device_uuid,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    identity = launch_detached(
        ["bash", str(run_script)],
        log_path=output / "run.log",
        pid_path=output / "pid",
        identity_path=output / "process_identity.json",
        cwd=output,
        environment=environment,
    )
    manifest = {
        "schema": "neural-forge-qwen-pair-compose-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(REPO_ROOT),
        "source": str(source),
        "blank_candidate": str(blank_candidate),
        "calibration_file": str(calibration_file),
        "eval_suite": str(eval_suite),
        "cached_parent_eval": str(parent_eval),
        "device_uuid": args.device_uuid,
        "settings": asdict(settings),
        "compose_command": compose_command,
        "compose_argv_sha256": _argv_sha256(compose_command),
        "eval_command": eval_command,
        "eval_argv_sha256": _argv_sha256(eval_command),
        "exact_eval": {
            "behavior_probes": EXACT_BEHAVIOR_PROBES,
            "generation_prompts": EXACT_GENERATION_PROMPTS,
            "choice_probes": EXACT_CHOICE_PROBES,
        },
        "cached_parent_policy": "required-snapshot-no-parent-evaluation",
        "zero_gradient_contract": {
            "command": "hf-qwen-pair-compose",
            "recovery_flags": False,
            "training_flags": False,
        },
        "snapshot_manifest": str(snapshot_manifest_path),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "snapshot_tree_sha256": snapshot_manifest["tree_sha256"],
        "run_script": str(run_script),
        "run_script_sha256": _sha256(run_script),
        "environment": environment,
        "process": identity,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  run_manifest.json\n", encoding="ascii"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
