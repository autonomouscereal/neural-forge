"""Explicit in-container D1 check/run; no host-device or allocator fallback."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import uuid

try:
    from . import d1_allocated_contract as R
except ImportError:
    import d1_allocated_contract as R


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    result.add_argument("--runtime-assignment", required=True)
    result.add_argument("--runtime-assignment-sha256", required=True)
    result.add_argument("--candidate-receipt-sha256", required=True)
    result.add_argument("--execution-id", type=uuid.UUID, required=True)
    return result


def mount_access(mounts, mountinfo):
    """Independently inspect actual namespace mount access without writing probes."""
    observed = {}
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise R.Error("invalid live mountinfo")
        point = fields[4].replace("\\040", " ").replace("\\134", "\\")
        observed[point] = "ro" in fields[5].split(",")
    for item in mounts:
        if observed.get(item["target"]) is not item["read_only"]:
            raise R.Error("actual in-container mount missing/access differs: " + item["target"])


def cuda_evidence():
    import torch
    count = torch.cuda.device_count()
    uuids, finite = [], True
    for ordinal in range(count):
        properties = torch.cuda.get_device_properties(ordinal)
        uuids.append(str(properties.uuid))
        with torch.cuda.device(ordinal):
            value = torch.ones((2, 2), device=f"cuda:{ordinal}", dtype=torch.bfloat16)
            finite = finite and bool(torch.isfinite(value @ value).all())
            torch.cuda.synchronize(ordinal)
        del value
    return dict(device_count=count, ordinals=list(range(count)), uuids=uuids,
                torch_version=str(torch.__version__), bf16_matmul_finite=finite)


def candidate_record(expected, root=Path("/candidate")):
    record = R.read(root / "candidate_receipt.json", expected)
    if record.get("schema") != "chimera_recovery.d1_allocated_candidate.v1":
        raise R.Error("D1 allocated candidate schema differs")
    for relative, digest in record["files"].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or R.sha(root / relative) != digest:
            raise R.Error("D1 allocated candidate code pin differs: " + relative)
    return record


def trainer_argv(candidate, assignment_sha):
    source = candidate["successor"]
    mapped = lambda path: R.map_source(candidate["source_path_map"], path)
    return ["--child", "/models/child", "--parent", "/models/parent", "--preregistration", mapped(source["preregistration"]),
            "--ids", mapped(source["ids"]), "--seq-len", "4096", "--lengths-key", "lengths", "--held-rows", str(source["held_rows"]),
            "--source-manifest", mapped(source["manifest"]), "--source-manifest-sha256", source["manifest_sha256"],
            "--source-validator-sha256", candidate["files"]["tools/d1_source_contract.py"],
            "--source-path-map", "/candidate/source_path_map.json", "--source-path-map-sha256", candidate["files"]["source_path_map.json"],
            "--runtime-assignment", "/lease/assignment.json", "--runtime-assignment-sha256", assignment_sha,
            "--devices", "cuda:0,cuda:1", "--dtype", "bfloat16", "--train-dtype", "float32", "--master-dtype", "float32",
            "--epochs", "2", "--tokens-per-step", "32768", "--micro-tokens", "8192", "--lr", "2e-5", "--warmup-steps", "10",
            "--min-lr-frac", "0.1", "--clip", "1.0", "--seed", "0", "--loss-site", "residual", "--base-digest", "full",
            "--min-free-gib", "60", "--resume", "--layers", "all", "--out", "/runout/trainer", "--delta-out", "/deltaout/trainer"]


def resume_guard(path, runtime_record):
    receipt = Path(path) / "d1_receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("identity", {}).get("allocated_runtime") != runtime_record:
            raise R.Error("D1 existing output belongs to another runtime identity; no implicit retrain/overwrite")


def model_preflight(trainer, args, source_record, assignment):
    import torch
    tree = trainer.bind_chimera()
    child, parent = trainer.open_checkpoint(args.child), trainer.open_checkpoint(args.parent)
    trainer.guard_out(Path(args.out), [child.root, parent.root], True)
    trainer.guard_delta_out(Path(args.delta_out), Path(args.out), [child.root, parent.root])
    layout = trainer.union_layout(child, args.trunk_width, args.k3_channels)
    if ((layout.width, layout.trunk, layout.k3, layout.pad) != (16640, 4352, 12288, 0)
            or child.hidden != parent.hidden or len(child.text_blocks) != 64 or len(parent.text_blocks) != 64
            or parent.charted_blocks or str(child.cfg.activation_kind) != "silu"):
        raise R.Error("D1 real parent/clone geometry differs from selected D1 recipe")
    for j, k in zip(child.text_blocks, parent.text_blocks):
        if child.cfg.block_kind(j) != parent.cfg.block_kind(k):
            raise R.Error("D1 parent/clone layer kinds differ")
    ids, lengths, _ = trainer.load_ids(args.ids, args.seq_len, args.max_rows, args.lengths_key)
    valid = torch.arange(ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    text_rows = min(trainer.TEXT_ROWS, int(child.cfg.ledger_band_begin()), int(parent.cfg.ledger_band_begin()))
    if int(ids[valid].min()) < 0 or int(ids[valid].max()) >= text_rows:
        raise R.Error("D1 source IDs exceed native text vocabulary")
    free = {"run": trainer.disk_floor(Path(args.out), 60, 64 * 134_000_000),
            "delta": trainer.disk_floor(Path(args.delta_out), 60, 8_560_000_000)}
    # Reuse the actual campaign digest algorithm, copied and source-pinned in the candidate.
    from oc_seal import directory_digest
    digest, entries = directory_digest(child.root)
    if digest != assignment["clone_directory_sha256"] or len(entries) != 18:
        raise R.Error("D1 actual clone full bytes/count differ from authentic sealed clone")
    return dict(chimera=tree, clone_directory_sha256=digest, clone_files=len(entries),
                child=str(child.root), parent=str(parent.root), layout=layout.__dict__, source=source_record,
                host_buffers_bytes=2 * int(ids.numel()) * int(child.hidden) * 2,
                free_gib=free, real_fit_tokens=int(lengths[:-args.held_rows].sum()),
                real_held_tokens=int(lengths[-args.held_rows:].sum()), verdict="check pass; no optimizer constructed")


def run(args):
    runtime = R.validate_step(args)
    assignment = runtime["assignment"]
    if (args.runtime_assignment != "/lease/assignment.json"
            or args.candidate_receipt_sha256 != assignment["candidate_receipt_sha256"]):
        raise R.Error("D1 entry requires exact read-only lease and candidate pins")
    candidate = candidate_record(args.candidate_receipt_sha256)
    mount_access(assignment["mounts"], Path("/proc/self/mountinfo").read_text())
    if R.tree_digest("/chimera") != assignment["chimera_source_sha256"]:
        raise R.Error("D1 in-container native source tree differs from authentic root binding")
    # CUDA ordinals belong to the container; host GPU numbers are never forwarded here.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CHIMERA_SRC"], os.environ["CHIMERA_OMNI_SRC"] = "/chimera/src", "/chimera"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "16"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path[:0] = ["/candidate/tools", "/chimera/src"]
    cuda = cuda_evidence()
    if (cuda["device_count"] != 2 or cuda["uuids"] != [gpu["uuid"] for gpu in assignment["gpus"]]
            or cuda["torch_version"] != assignment["torch_version"] or not cuda["bf16_matmul_finite"]):
        raise R.Error("D1 actual container CUDA/Torch/BF16 proof differs")
    trainer = importlib.import_module("d1_trunk_recovery")
    argv = trainer_argv(candidate, args.runtime_assignment_sha256)
    training_args = trainer.build_parser().parse_args(argv)
    preregistration = trainer.preregistration_record(training_args)
    source = importlib.import_module("d1_source_contract").validate_args(training_args, Path(trainer.__file__))
    result = model_preflight(trainer, training_args, source, assignment)
    result.update(schema="chimera_recovery.d1_allocated_preflight.v1", cuda=cuda, runtime=runtime,
                  preregistration=preregistration, mode="run" if args.run else "check-only", training_launched=False)
    R.validate_step(args)  # Full byte verification can take minutes; root monitor must still be live.
    resume_guard(training_args.out, runtime)
    if args.run:
        # This is the only route to the original optimizer, gated again at every apply.
        result["training_launched"] = True
        trainer.run(argv)
    return result


def main():
    args = parser().parse_args()  # Unknown/no mode refuses before filesystem, Torch or CUDA.
    # One physical runner within the already-owned container, independent of host GPU locks.
    import fcntl
    lock_path = Path("/runout/.allocated-entry.lock")
    with lock_path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = Path("/runout/runtime-status.json")
        initial = dict(completed=False, execution_id=str(args.execution_id), assignment_sha256=args.runtime_assignment_sha256,
                       mode="run" if args.run else "check-only", started_at=datetime.now(timezone.utc).isoformat())
        status_path.write_bytes(R.canonical(initial) + b"\n")
        try:
            with Path("/runout/runtime-log.txt").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run(args)
            status = dict(completed=True, exit_code=0, result=result)
        except Exception as exc:
            status = dict(completed=True, exit_code=2, error_type=type(exc).__name__, error=str(exc))
        status.update(execution_id=str(args.execution_id), assignment_sha256=args.runtime_assignment_sha256, mode="run" if args.run else "check-only",
                      ended_at=datetime.now(timezone.utc).isoformat())
        temporary = Path("/runout/runtime-status.json.incoming")
        temporary.write_bytes(R.canonical(status) + b"\n")
        temporary.replace("/runout/runtime-status.json")
        print(json.dumps(status, sort_keys=True))
        raise SystemExit(status["exit_code"])


if __name__ == "__main__":
    main()
