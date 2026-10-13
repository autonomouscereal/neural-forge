"""Strict secretless D1 two-GPU container and fresh root-attested lease contract."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import uuid

SCHEMA = "chimera_recovery.d1_allocated_runtime.v1"
PROOF = "chimera_recovery.d1_allocator_live.v1"
Error = ValueError
OWNER = "codex-chimera-recovery-01a1326f"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path, expected):
    if not re.fullmatch(r"[a-f0-9]{64}", expected or "") or sha(path) != expected:
        raise Error("D1 allocated artifact missing or changed")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree_digest(root):
    root = Path(root)
    paths = sorted((root / "src").rglob("*.py"))
    if (root / "COMMIT").is_file():
        paths.append(root / "COMMIT")
    if not paths or any(path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise Error("D1 genuine native source tree missing/aliased")
    return hashlib.sha256(canonical(sorted([path.relative_to(root).as_posix(), sha(path)] for path in paths))).hexdigest()


def assignment_record(path, expected):
    value = read(path, expected)
    if (value.get("schema") != SCHEMA or value.get("server") != "ai" or value.get("owner") != OWNER
            or value.get("stage") != "D1" or value.get("native_task") != "01a1326f-4f44-7ef0-a2fa-2a64a706fedc"
            or value.get("active") is not True or not re.fullmatch(r"[a-f0-9]{64}", value.get("container_id", ""))
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", value.get("image_id", ""))
            or value.get("container_devices") != ["cuda:0", "cuda:1"]
            or value.get("lease_proof_target") != "/lease/live.json" or value.get("lease_mount_read_only") is not True):
        raise Error("D1 requires exact active root-owned two-GPU container assignment")
    uuid.UUID(value["allocation_group_id"])
    gpus = value.get("gpus", [])
    if (len(gpus) != 2 or [gpu.get("host_index") for gpu in gpus] != [0, 4]
            or [gpu.get("container_ordinal") for gpu in gpus] != [0, 1]
            or any(not re.fullmatch(r"GPU-[a-fA-F0-9-]{36}", gpu.get("uuid", "")) for gpu in gpus)
            or len({gpu["uuid"] for gpu in gpus}) != 2):
        raise Error("D1 assignment must bind two distinct physical GPU0/GPU4 UUIDs to native ordinals0/1")
    mounts = value.get("mounts", [])
    targets = [item.get("target") for item in mounts]
    if len(targets) != len(set(targets)) or any(not PurePosixPath(item.get("source", "")).is_absolute()
                                              or not PurePosixPath(item.get("target", "")).is_absolute()
                                              or type(item.get("read_only")) is not bool for item in mounts):
        raise Error("D1 exact explicit mount mapping required")
    roles = {item.get("role"): item for item in mounts}
    if (len(roles) != len(mounts) or set(roles) != {"child", "parent", "candidate", "chimera", "lease", "successor", "run", "delta"}
            or any(".." in PurePosixPath(target).parts or target == "/" for target in targets)
            or any(PurePosixPath(a) in PurePosixPath(b).parents for a in targets for b in targets if a != b)):
        raise Error("D1 mount roles must be unique with no nested/escaping targets")
    # A writable source must never offer another view of a protected input.
    protected = [PurePosixPath(item["source"]) for item in mounts if item["read_only"]]
    writable = [PurePosixPath(item["source"]) for item in mounts if not item["read_only"]]
    if any(a == b or a in b.parents or b in a.parents for a in protected for b in writable):
        raise Error("D1 writable mount aliases protected source/code/lease")
    for role, target in (("child", "/models/child"), ("parent", "/models/parent"), ("candidate", "/candidate"),
                         ("chimera", "/chimera"), ("lease", "/lease")):
        if role not in roles or roles[role]["target"] != target or roles[role]["read_only"] is not True:
            raise Error("D1 model/code/lease mount must be explicitly read-only: " + role)
    if roles["child"]["source"] != "/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1":
        raise Error("D1 requires the actual independent frozen clone, no source alias")
    if roles["child"]["source"] == roles["parent"]["source"]:
        raise Error("D1 teacher and child cannot alias")
    if ("successor" not in roles or roles["successor"]["read_only"] is not True
            or roles["successor"]["source"] != roles["successor"]["target"]):
        raise Error("D1 successor manifest uses exact unchanged absolute read-only paths")
    for role, target in (("run", "/runout"), ("delta", "/deltaout")):
        if role not in roles or roles[role]["target"] != target or roles[role]["read_only"] is not False:
            raise Error("D1 requires dedicated output mounts")
    if value.get("clone_directory_sha256") != "8edf50c42090d6d556d71d6c6c71e2405569b5082adaf133f4265bad1d901264":
        raise Error("D1 frozen clone full verified digest differs")
    if value.get("mem_available_launch_floor_gib") != 100 or value.get("max_proof_age_seconds") != 45:
        raise Error("D1 launch memory/freshness policy differs")
    if (not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,100}", value.get("container_name", ""))
            or not re.fullmatch(r"2\.11\.0(?:\+[a-zA-Z0-9.]+)?", value.get("torch_version", ""))
            or value.get("image_id") != "sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164"
            or not re.fullmatch(r"[a-f0-9]{64}", value.get("chimera_source_sha256", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", value.get("root_controller_sha256", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", value.get("candidate_receipt_sha256", ""))):
        raise Error("D1 exact reviewed Torch2.11 image/candidate/name pins required")
    return value


def verify_pg(assignment, rows):
    group = assignment["allocation_group_id"]
    uuids = [gpu["uuid"] for gpu in assignment["gpus"]]
    if len(rows) != 2 or {row.get("gpu_uuid") for row in rows} != set(uuids):
        raise Error("D1 live PostgreSQL allocation must contain the exact two members")
    for row in rows:
        details = row.get("details", {})
        if isinstance(details, str):
            details = json.loads(details)
        if (str(row.get("group_id") or row.get("allocation_id")) != group or row.get("state") != "running"
                or row.get("server") != "ai" or row.get("owner") != OWNER or row.get("kind") != "generic"
                or row.get("name") != assignment["container_name"] or row.get("container_id") != assignment["container_id"]
                or details.get("gpu_uuids") != uuids
                or row.get("gpu_index") != next(gpu["host_index"] for gpu in assignment["gpus"] if gpu["uuid"] == row["gpu_uuid"])):
            raise Error("D1 live PostgreSQL lease/owner/container/group differs")


def verify_docker(assignment, docker):
    group = assignment["allocation_group_id"]
    uuids = [gpu["uuid"] for gpu in assignment["gpus"]]
    labels = docker.get("Config", {}).get("Labels", {})
    requests = docker.get("HostConfig", {}).get("DeviceRequests", [])
    if (docker.get("Id") != assignment["container_id"] or docker.get("Image") != assignment["image_id"]
            or docker.get("Name", "").lstrip("/") != assignment["container_name"] or docker.get("State", {}).get("Running") is not True
            or labels.get("ai.local.gpu-controller") != "v1" or labels.get("ai.local.allocation") != group
            or labels.get("ai.local.owner") != OWNER or labels.get("ai.local.kind") != "generic"
            or json.loads(labels.get("ai.local.gpu-uuids", "[]")) != uuids
            or len(requests) != 1 or requests[0].get("DeviceIDs") != uuids or requests[0].get("Driver") != "nvidia"):
        raise Error("D1 immutable Docker image/device/allocator labels differ")
    actual_mounts = docker.get("Mounts", [])
    if len(actual_mounts) != len(assignment["mounts"]) or any(item.get("Type") != "bind" for item in actual_mounts):
        raise Error("D1 undeclared container mounts could alias protected inputs")
    observed = {(item["Source"], item["Destination"], not item["RW"]) for item in actual_mounts}
    expected = {(item["source"], item["target"], item["read_only"]) for item in assignment["mounts"]}
    if observed != expected:
        raise Error("D1 actual Docker mount access differs from explicit assignment")


def verify_host(assignment, host):
    if host.get("chimera_source_sha256") != assignment["chimera_source_sha256"]:
        raise Error("D1 native source tree changed after runtime assignment")
    if (host.get("k3_running") is not False or host.get("foreign_gpu_processes")
            or host.get("k3_passes_complete") is not True or host.get("k3_sealed_manifests", 0) < 4
            or host.get("selected_gpu_queues_running") is not False):
        raise Error("D1 cannot overlap K3 host memory or foreign selected-GPU processes")
    if host.get("phase") == "launch" and host.get("mem_available_gib", 0) < 100:
        raise Error("D1 initial MemAvailable below100GiB")
    if host.get("phase") not in {"launch", "training"}:
        raise Error("D1 live proof phase must be explicit")


def attest(assignment, assignment_sha, rows, docker, cuda, host, now=None):
    """Verify actual bounded root PG/Docker/CUDA evidence, then emit live proof."""
    now = now or datetime.now(timezone.utc)
    group = assignment["allocation_group_id"]
    uuids = [gpu["uuid"] for gpu in assignment["gpus"]]
    verify_pg(assignment, rows)
    verify_docker(assignment, docker)
    verify_host(assignment, host)
    if (cuda.get("device_count") != 2 or cuda.get("uuids") != uuids or cuda.get("torch_version") != assignment["torch_version"]
            or cuda.get("bf16_matmul_finite") is not True or cuda.get("ordinals") != [0, 1]):
        raise Error("D1 actual in-container CUDA ordinals/UUIDs/BF16 runtime differ")
    return dict(schema=PROOF, valid=True, assignment_sha256=assignment_sha, captured_at=now.isoformat(),
                allocation_group_id=group, container_id=assignment["container_id"], image_id=assignment["image_id"],
                gpu_uuids=uuids, phase=host["phase"], mem_available_gib=host["mem_available_gib"],
                chimera_source_sha256=assignment["chimera_source_sha256"], postgres_live=True, docker_live=True,
                root_controller_sha256=assignment["root_controller_sha256"],
                proof_producer="root_raw_pg_and_server_manager", training_launch_authorized=False)


def validate_step(args, *, now=None, proof_path=None, candidate_root=None):
    assignment = assignment_record(args.runtime_assignment, args.runtime_assignment_sha256)
    proof = json.loads(Path(proof_path or assignment["lease_proof_target"]).read_text(encoding="utf-8"))
    now = now or datetime.now(timezone.utc)
    captured = datetime.fromisoformat(proof["captured_at"])
    if captured.tzinfo is None or now.tzinfo is None:
        raise Error("D1 allocator proof timestamps must include UTC offset")
    age = (now - captured).total_seconds()
    if (not 0 <= age <= assignment["max_proof_age_seconds"]
            or proof.get("schema") != PROOF or proof.get("valid") is not True
            or proof.get("assignment_sha256") != args.runtime_assignment_sha256
            or proof.get("allocation_group_id") != assignment["allocation_group_id"]
            or proof.get("container_id") != assignment["container_id"] or proof.get("image_id") != assignment["image_id"]
            or proof.get("gpu_uuids") != [gpu["uuid"] for gpu in assignment["gpus"]]
            or proof.get("chimera_source_sha256") != assignment["chimera_source_sha256"]
            or proof.get("root_controller_sha256") != assignment["root_controller_sha256"]
            or proof.get("postgres_live") is not True or proof.get("docker_live") is not True
            or proof.get("phase") not in {"launch", "training"}
            or proof.get("proof_producer") != "root_raw_pg_and_server_manager"):
        raise Error("D1 allocator attestation stale/revoked/foreign; refusing optimizer update")
    candidate_root = Path(candidate_root or "/candidate")
    candidate = read(candidate_root / "candidate_receipt.json", assignment["candidate_receipt_sha256"])
    if (candidate.get("schema") != "chimera_recovery.d1_allocated_candidate.v1"
            or sha(__file__) != candidate["files"].get("tools/d1_allocated_contract.py")):
        raise Error("D1 actual runtime validator code differs from pinned candidate")
    if hasattr(args, "devices"):
        required = dict(devices="cuda:0,cuda:1", child="/models/child", parent="/models/parent", dtype="bfloat16",
                        train_dtype="float32", master_dtype="float32", base_digest="full",
                        source_manifest_sha256=candidate["successor"]["manifest_sha256"],
                        epochs=2, tokens_per_step=32768, micro_tokens=8192, lr=2e-5, warmup_steps=10,
                        min_lr_frac=0.1, clip=1.0, seed=0, loss_site="residual", layers="all",
                        out="/runout/trainer", delta_out="/deltaout/trainer", min_free_gib=60,
                        max_rows=None, max_steps=None, stop_after_layer=None)
        if any(getattr(args, key, None) != value for key, value in required.items()):
            raise Error("D1 direct trainer arguments violate allocated runtime/source contract")
        if sha(candidate_root / "tools/d1_trunk_recovery.py") != candidate["files"]["tools/d1_trunk_recovery.py"]:
            raise Error("D1 actual optimizer source changed")
    return dict(assignment_sha256=args.runtime_assignment_sha256, assignment=assignment,
                validator_sha256=sha(Path(__file__)), live_proof_scope="fresh root PG/Docker attestation before every optimizer apply")


def patch_trainer(raw):
    # Exact reviewed v4 source; never patch a production file in place.
    if hashlib.sha256(raw).hexdigest() != "c6b2525b70e1d8c74aeb5a40434e00cc25e43b1647c1afc903046a8da3c4ed03":
        raise Error("D1 allocated successor requires exact reviewed v4 trainer")
    text = raw.decode()
    def change(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise Error("D1 allocated trainer source anchor differs")
        text = text.replace(before, after, 1)
    change('TOOL_VERSION = "1.2.6-source-successor"', 'TOOL_VERSION = "1.2.7-allocated-successor"')
    change("    source_record = source_contract.validate_args(args, Path(__file__))",
           "    import d1_allocated_contract as runtime_contract\n    runtime_record = runtime_contract.validate_step(args)\n"
           "    source_record = source_contract.validate_args(args, Path(__file__))")
    change("        upd = trainer.apply(host, lr)", "        runtime_contract = __import__('d1_allocated_contract')\n"
           "        runtime_contract.validate_step(rs.args)\n        upd = trainer.apply(host, lr)")
    change('"preregistration": preregistration, "source_contract": source_record,\n                "hyper":',
           '"preregistration": preregistration, "source_contract": source_record, "allocated_runtime": runtime_record,\n                "hyper":')
    change('"preregistration": preregistration, "source_contract": source_record,\n                  "rows":',
           '"preregistration": preregistration, "source_contract": source_record, "allocated_runtime": runtime_record,\n                  "rows":')
    anchor = '    ap.add_argument("--source-validator-sha256", default=None)'
    change(anchor, anchor + '\n    ap.add_argument("--runtime-assignment", required=True)\n'
           '    ap.add_argument("--runtime-assignment-sha256", required=True)')
    compile(text, "d1_trunk_recovery.py", "exec")
    return text.encode()
