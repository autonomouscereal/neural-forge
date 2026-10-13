"""Prepare an immutable D1 allocated-runtime candidate; never allocate or launch."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

try:
    from . import d1_allocated_contract as R
    from . import d1_allocated_entry as E
    from . import d1_source_contract as S
except ImportError:
    import d1_allocated_contract as R
    import d1_allocated_entry as E
    import d1_source_contract as S

OWNER = "codex-chimera-recovery-01a1326f"
IMAGE = "sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164"
CLONE = "/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"
PREPARATION = "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013"
CLONE_DIGEST = "8edf50c42090d6d556d71d6c6c71e2405569b5082adaf133f4265bad1d901264"
CLONE_RECEIPT = "7737e7050ab181d743d019beded2c86a03e50ef8330c02bf28439b9ce02a7162"
REGISTRY_RECEIPT = "3e3e466ec25f707422c883f4177eede58425bbc4687792e827c2ebac5f65a929"


def plan(runtime_candidate, lease_source, run_source, delta_source):
    mounts = [dict(role=role, source=source, target=target, read_only=readonly) for role, source, target, readonly in (
        ("child", CLONE, "/models/child", True),
        ("parent", "/mnt/ai_data/omni-campaign/s0-probe-sep04", "/models/parent", True),
        ("candidate", runtime_candidate, "/candidate", True),
        ("chimera", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9", "/chimera", True),
        ("successor", PREPARATION, "/inputs/successor", True),
        ("lease", lease_source, "/lease", True),
        ("run", run_source, "/runout", False), ("delta", delta_source, "/deltaout", False))]
    return dict(schema="chimera_recovery.d1_allocation_plan.v1", owner=OWNER, server="ai", stage="D1",
                native_task="01a1326f-4f44-7ef0-a2fa-2a64a706fedc", active=False, production_ready=False,
                required_host_gpu_indices=[0, 4], image_id=IMAGE, mounts=mounts,
                clone_directory_sha256=CLONE_DIGEST, mem_available_launch_floor_gib=100, max_proof_age_seconds=45,
                container_devices=["cuda:0", "cuda:1"], lease_proof_target="/lease/live.json", lease_mount_read_only=True,
                container_id=None, allocation_group_id=None, container_name=None,
                allocation_command_template=["C:/Python314/python.exe", "-B", "C:/Users/cereal/hermes.py", "run", "--gpus", "0", "4",
                                             "--owner", OWNER, "--name", "<root-chosen-exact-name>", "--image", IMAGE]
                + [value for item in mounts for value in ("--mount-ro" if item["read_only"] else "--mount-rw", item["source"], item["target"])]
                + ["--runtime-arg", "python3", "--runtime-arg=-c", "--runtime-arg", "import time; time.sleep(86400)"],
                blockers=["No actual named two-GPU allocator group/container has been supplied",
                          "Root must choose an idle resource window, then bind real PG/Docker/CUDA evidence",
                          "No production D1 run or GPU parity/memory-throughput test performed by this preparation"],
                training_launched=False, registration_scope="No automatic model/ledger registration; audit actual completed delta separately")


def portable_validator(raw):
    """Add resolution to the exact reviewed validator; retain original seals/pins."""
    if hashlib.sha256(raw).hexdigest() != "68a178ddf618b1d9b72e055bf6f1eda2a350e8d5e928cbaac7d1b5d62bf1f303":
        raise R.Error("Exact reviewed D1 successor validator required")
    text = raw.decode()
    def change(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise R.Error("Portable D1 validator source anchor differs")
        text = text.replace(before, after, 1)
    change('SCHEMA = "chimera_recovery.d1_source_successor.v1"',
           'SCHEMA = "chimera_recovery.d1_source_successor.v1"\n_SOURCE_MAP = None\n\n'
           'def resolved_source(path):\n'
           '    if _SOURCE_MAP is None:\n        return str(path)\n'
           '    from d1_allocated_contract import map_source\n    return map_source(_SOURCE_MAP, path)\n\n'
           'def configure_source_map(args):\n    global _SOURCE_MAP\n'
           '    path = getattr(args, "source_path_map", None)\n    digest = getattr(args, "source_path_map_sha256", None)\n'
           '    if bool(path) != bool(digest):\n        raise ValueError("D1 source mapping path/SHA must be supplied together")\n'
           '    if path:\n        from d1_allocated_contract import source_map_record\n'
           '        _SOURCE_MAP = source_map_record(path, digest)\n    else:\n        _SOURCE_MAP = None')
    change('    path = Path(pin["path"])', '    path = Path(resolved_source(pin["path"]))')
    change('    return Path(left).resolve() == Path(right).resolve()',
           '    return Path(resolved_source(left)).resolve() == Path(resolved_source(right)).resolve()')
    change('def manifest_record(path, expected_sha):\n', 'def manifest_record(path, expected_sha):\n    path = resolved_source(path)\n')
    change('    manifest = manifest_record(path, expected_sha)', '    path = resolved_source(path)\n    manifest = manifest_record(path, expected_sha)')
    change('def validate_args(args, trainer_source):\n', 'def validate_args(args, trainer_source):\n    configure_source_map(args)\n')
    change('    parser.add_argument("--sha256", required=True)',
           '    parser.add_argument("--sha256", required=True)\n    parser.add_argument("--source-path-map")\n'
           '    parser.add_argument("--source-path-map-sha256")')
    change('    args = parser.parse_args()\n', '    args = parser.parse_args()\n    configure_source_map(args)\n')
    compile(text, "d1_source_contract.py", "exec")
    return text.encode()


def prepare(baseline, source_manifest, source_sha, clone_receipt, registry_receipt, campaign_tools, campaign_ledger,
            runtime_candidate, lease_source, run_source, delta_source, out):
    for path in (runtime_candidate, lease_source, run_source, delta_source):
        if not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or any(c in path for c in "\n\r"):
            raise R.Error("Explicit nonescaping absolute server paths required")
    if out.exists():
        raise R.Error("D1 allocated preparation requires a new additive directory")
    receipt = R.read(clone_receipt, CLONE_RECEIPT)
    registry = R.read(registry_receipt, REGISTRY_RECEIPT)
    if (receipt.get("clone_directory") != CLONE or receipt.get("clone_mode") != "copy" or receipt.get("files") != 18
            or receipt.get("clone_directory_sha256") != CLONE_DIGEST or receipt.get("source_directory_sha256") != CLONE_DIGEST
            or registry.get("status") != "cloned" or registry["checkpoint"].get("directory_sha256") != CLONE_DIGEST
            or registry["checkpoint"].get("clone_receipt_sha") != receipt.get("receipt_sha")):
        raise R.Error("Actual independent clone receipt/registry evidence differs")
    manifest = S.manifest_record(source_manifest, source_sha)
    files = {"tools/d1_trunk_recovery.py": R.patch_trainer((baseline / "tools/d1_trunk_recovery.py").read_bytes()),
             "tools/d1_source_contract.py": portable_validator((baseline / "tools/d1_source_contract.py").read_bytes()),
             "tools/d1_allocated_contract.py": Path(R.__file__).read_bytes(),
             "tools/d1_allocated_entry.py": Path(E.__file__).read_bytes(),
             "tools/d1_allocated_root.py": Path(__file__).with_name("d1_allocated_root.py").read_bytes(),
             "tools/d1_allocated.py": Path(__file__).read_bytes(),
             "tools/stage2_training_receipt.py": (campaign_tools / "stage2_training_receipt.py").read_bytes(),
             "tools/oc_seal.py": (campaign_ledger / "oc_seal.py").read_bytes(),
             "evidence/actual-clone-receipt.json": Path(clone_receipt).read_bytes(),
             "evidence/actual-registry-clone.json": Path(registry_receipt).read_bytes(),
             "source_path_map.json": R.canonical(R.SOURCE_MAP) + b"\n"}
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    candidate = dict(schema="chimera_recovery.d1_allocated_candidate.v1", prepared_date="2026-10-13",
                     image_id=IMAGE, clone_directory_sha256=CLONE_DIGEST, actual_counts=manifest["actual_counts"],
                     source_path_map=R.SOURCE_MAP, original_source_manifest_bytes_unchanged=True,
                     successor=dict(manifest=str(PurePosixPath(manifest["dataset"]["path"]).parent / "manifest.json"),
                                    manifest_sha256=source_sha, seal_sha256=manifest["seal_sha256"],
                                    ids=manifest["dataset"]["path"], held_rows=manifest["actual_counts"]["held_rows"],
                                    preregistration=manifest["preregistration"]["path"]),
                     files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                     training_launched=False, allocator_binding_ready=False, production_ready=False,
                     update_gate="Fresh exact-group root raw-PG/Docker proof before every production trainer.apply",
                     teacher_scope="Direct frozen S0 parent hidden-state forwards; no qtext cache reuse",
                     resume_scope="Same source/run/container assignment only; foreign previous runtime identity refused",
                     registration_scope="Export trainer results only; no automatic ledger mutation or model publication")
    allocation_plan = plan(runtime_candidate, lease_source, run_source, delta_source)
    allocation_plan["candidate_receipt_sha256"] = hashlib.sha256(R.canonical(candidate) + b"\n").hexdigest()
    allocation_plan["root_controller_sha256"] = candidate["files"]["tools/d1_allocated_root.py"]
    allocation_plan["source_path_map_sha256"] = candidate["files"]["source_path_map.json"]
    files["candidate_receipt.json"] = R.canonical(candidate) + b"\n"
    files["allocation_plan.json"] = R.canonical(allocation_plan) + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
        directory = Path(temporary)
        for name, raw in files.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        directory.rename(out)
    return dict(candidate_receipt_sha256=R.sha(out / "candidate_receipt.json"), allocation_plan_sha256=R.sha(out / "allocation_plan.json"),
                out=str(out), training_launched=False, production_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("baseline", "source-manifest", "clone-receipt", "registry-receipt", "campaign-tools", "campaign-ledger", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("source-sha", "runtime-candidate", "lease-source", "run-source", "delta-source"):
        parser.add_argument("--" + key, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
