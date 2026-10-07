"""Run actual K3 later windows after separately validating their native prefix.

This PID1 uses only local subprocesses inside the already owned allocator
container. Root owns allocation, provider restoration and independent sampling.
Native sources, first-launch files and historical acceptance remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid
from k3_recovery_ancestry import RecoveryAncestry
from recorded_reader_reuse import reuse_saved35_readers
from k3_capacity_boundary import (CapacityDenied, MemoryCapacityUnavailable, object_sha as capacity_object_sha,
    await_decision as capacity_await_decision, read_decision as capacity_read_decision,
    publish_sha as capacity_publish_sha)

sys.dont_write_bytecode = True
# Same explicit task-local CUDA allocator request as the successful recovery.
# This changes only this PID1 environment and its child processes, never Docker
# configuration, provider settings, global defaults or native producer argv.
ALLOC_REQUEST = "expandable_segments:True"
ALLOC_INHERITED = os.environ.get("PYTORCH_ALLOC_CONF")
os.environ["PYTORCH_ALLOC_CONF"] = ALLOC_REQUEST
OUT = Path("/run-output")
META = Path("/run-control-output")
RUN = "__UNRESOLVED_NEXT_RUN_UUID__"
IDENTITY_SHA = "__UNRESOLVED_NEXT_IDENTITY_SHA256__"
CAP_BYTES = None  # exact newly funded49-60 cap; never inferred from88GiB
CAP_CPUS = None  # exact newly funded49-60 CPU quota
SESSION_GENERATION = None
SESSION_ID = "__UNRESOLVED_NEXT_GENUINE_NATIVE_SESSION__"
PREVIOUS_NAS = "/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model"
NAS_PREFIX = Path("/resume-window37-48")
HOST_META = "/home/cereal/omni-campaign/outputs/k3-nas-window49-60-oct07-" + RUN + "/control-output"
HOST_WORKING = "/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-" + RUN + "/model"
PREFIX = Path("/resume-prefix")
HOST_PREFIX = "/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318"
HOST_FAILED = "/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1"
RETENTION_REASON = "preserve all existing and new native checkpoints"
RETENTION_IDENTITY = "--snapshot-keep 0: " + RETENTION_REASON
CONTROL = Path("/control")
NF = Path("/code/neural-forge")
PARENT = Path("/parent")
CORPUS = Path("/inputs/corpus")
DIGEST = Path("/inputs/digest-receipts/k3_lexar_digest_verify.json")
TOOL = NF / "reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py"
NATIVE_ID = "01a11108-009c-72f3-b2ba-e0d48fca1435"
RELAY_ID = "95f6c2bd-7139-46f6-b5d6-228d1b9753f7"
GPU = "GPU-f5a09535-fc85-a027-50f7-734b344747dd"
TOOL_SHA = "44ccfba059f0da1a5b2c63dfcbbb54e2f4e2b641fed5d0273cc00437cb6d8dde"
BASELINE_SHA = "c54e2fd0ade6fb8bbc9a111cb76fc5469abc1c197001b137e70f1c1156e976ff"
SCANNER_SHA = "e2670b38547947aa5c021ba46677b5d363131e18f8ef0a11d418ce0352737e02"
ORIGINAL_SCANNER_SHA = "593db85349e841825f5e9f254a7963de2139f854d4ee1392285f3bb9b3d98127"
VERIFIER_SHA = "f8a256186c4d6869747ff3c5558cf5d137db0cf8d6cfc3843b67446a5f914ecf"


class ReaderContractMismatch(ValueError):
    """Actual checker cannot read a native requested-rung/installed-width pair."""

ADAPTER_SHA = "de4a24f94897660875f6535b8114c7f65141fbc0a8c6dd7822e61e535fa40623"
CONFIG_SHA = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
WINDOWS = [(5, 12)] + [(s, min(s + 11, 92)) for s in range(13, 93, 12)]
GIB = 1 << 30


def canonical(value):
    return json.dumps(value, indent=1, sort_keys=True, default=str)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def operational_argv(first_argv, layers):
    require(first_argv.count("--snapshot-keep") == 1
            and first_argv[first_argv.index("--snapshot-keep") + 1] == "2",
            "preserved original checkpoint retention2 required")
    require(first_argv.count("--layers") == 1
            and first_argv[first_argv.index("--layers") + 1] == "5-12",
            "preserved original first layer argument required")
    argv = list(first_argv)
    argv[argv.index("--layers") + 1] = layers
    argv[argv.index("--snapshot-keep") + 1] = "0"
    return argv


def retention_override(first_plan_sha):
    return {"flag": "--snapshot-keep", "original_value": "2", "current_value": "0",
            "reason": RETENTION_REASON, "original_record_plan_sha256": first_plan_sha}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    # Native receipts can contain undefined derived diagnostics. Their values
    # do not substitute for the scanner's actual stored tensor finite checks.
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def pinned(path, expected):
    path = Path(path)
    require(re.fullmatch(r"[0-9a-f]{64}", expected or ""), "complete SHA256 required")
    require(path.is_file() and not path.is_symlink(), "regular pinned file required: " + str(path))
    require(sha(path) == expected, "pinned file hash mismatch: " + str(path))
    return path


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def current_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def header(path):
    with Path(path).open("rb") as handle:
        lead = handle.read(8)
        require(len(lead) == 8, "truncated actual safetensors header")
        length = struct.unpack("<Q", lead)[0]
        require(2 <= length <= 16 << 20 and length + 8 <= Path(path).stat().st_size,
                "bounded complete actual safetensors header required")
        return json.loads(handle.read(length), object_pairs_hook=unique)


def resource_sample():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    cgroup = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.current",
                 "memory.swap.max", "memory.events", "cpu.max"):
        text = (Path("/sys/fs/cgroup") / name).read_text().strip()
        if name == "memory.events":
            cgroup[name] = {parts[0]: int(parts[1]) for parts in map(str.split, text.splitlines())}
        elif name == "cpu.max":
            quota, period = text.split()
            cgroup[name] = {"quota": quota, "period": int(period),
                            "cpus": None if quota == "max" else int(quota) / int(period)}
        else:
            cgroup[name] = text if text == "max" else int(text)
    disk = shutil.disk_usage(OUT)
    return {"at_utc": now(), "host_memory": memory, "cgroup": cgroup,
            "output_disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}}


class Driver:
    pinned = staticmethod(pinned)
    read = staticmethod(read)
    unique = staticmethod(unique)
    header = staticmethod(header)
    canonical = staticmethod(canonical)
    def __init__(self, config_path, config_sha):
        require(re.fullmatch(r"[0-9a-f-]{36}", RUN) is not None and str(uuid.UUID(RUN)) == RUN,
                "unresolved next-phase RUN; proposal cannot execute")
        require(re.fullmatch(r"[0-9a-f]{64}", IDENTITY_SHA) is not None,
                "unresolved next-phase exact identity source pin")
        require(type(CAP_BYTES) is int and CAP_BYTES > 0 and type(CAP_CPUS) is int and CAP_CPUS > 0
                and type(SESSION_GENERATION) is int and SESSION_GENERATION > 0
                and str(uuid.UUID(SESSION_ID)) == SESSION_ID,
                "new actual resource cap and genuine native binding required")
        self.config_path=pinned(config_path,config_sha);self.config_sha=config_sha
        self.config=read(self.config_path)
        require(self.config.get("schema")=="neural_forge.k3_nas_closed48_worker_config.v1", "exact one-window NAS config required")
        self.identity=self.config["identity"]
        require(sha(CONTROL/"identity.json")==IDENTITY_SHA and read(CONTROL/"identity.json")==self.identity,"new exact phase identity required")
        require(self.identity["run_uuid"]==RUN and self.identity["native_thread_id"]==NATIVE_ID and self.identity["private_relay_thread"]==RELAY_ID and self.identity["gpu_uuid"]==GPU,"genuine exact owner/device/run required")
        require(self.identity["native_working_output_root"]==HOST_WORKING and self.identity["controller_output_root"]==HOST_META and self.identity["readonly_resume_source_roots"]=={"legacy_closed36": HOST_PREFIX, "closed37_48": PREVIOUS_NAS},"RO prefix and unique NAS output identity differs")
        require(self.identity["native_binding"]["session_generation"]==SESSION_GENERATION and self.identity["native_binding"]["session_id"]==SESSION_ID,"new actual supported native binding required")
        require(self.config["window"]==[49,60] and self.config["later_windows_admitted"] is False,"only legal49-60 window required")
        self.sources=self.config["source_pins"];self.tool_sha=self.config["native_tool_sha256"]
        self.first_ref=self.config["first_plan"];self.first_path=pinned(self.first_ref["path"],self.first_ref["sha256"])
        require(self.first_path==PREFIX/"ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b/inputs/original_record_plan.json", "exact retained original record-plan role required")
        self.first=read(self.first_path);self.first_argv=self.first["command_argv"]
        operational_argv(self.first_argv,"49-60")
        for flag in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            require(flag + "=" + str(CAP_CPUS) in self.first_argv,
                    "new exact CPU cap must retain the source selected native thread count")
        self.resume_argv=self.config["resume_argv"]
        from ro_resume_runtime import nas_resume_from_arguments
        from argparse import Namespace
        flags={}
        require(len(self.resume_argv)==14 and len(set(self.resume_argv[::2]))==7,"exact seven explicit RO source flags required")
        for flag,value in zip(self.resume_argv[::2],self.resume_argv[1::2]):
            require(flag.startswith("--resume-source-"),"unexpected resume source flag")
            flags[flag[2:].replace("-","_")]=value
        self.resolver,self.guard=nas_resume_from_arguments(Namespace(**flags),TOOL)
        self.current_context=self.guard.current_context
        require(self.guard.current_sha==self.tool_sha,"actual revised native pin differs")
        self.baseline_ref=self.config["baseline_receipt"];self.baseline_path=pinned(self.baseline_ref["path"],BASELINE_SHA);self.baseline=read(self.baseline_path)
        self.prep_ref=self.config["prep_receipt"];self.prep_path=pinned(self.prep_ref["path"],self.prep_ref["sha256"])
        self.source_check()
        cfg=read(pinned(PARENT/"config.json",CONFIG_SHA));self.model_cfg=cfg.get("text_config",cfg)
        self.budget=self.config["budget"]
        require(self.budget["memory_limit_bytes"]==CAP_BYTES and self.budget["cpus"]==CAP_CPUS and self.budget["host_headroom_bytes"]==16*GIB,"new source-funded exact49-60 cap and noSwap budget required")
        self.root=META/"continuation"/self.config["continuation_uuid"];uuid.UUID(self.config["continuation_uuid"])
        require(not self.root.exists() and not self.root.is_symlink(),"exclusive new controller receipt root required");self.root.mkdir(parents=True)
        self.allocation=None;self.window_path=None;self.initial_cgroup_checked=False
        self.status={"schema":"neural_forge.k3_actual_continuation_worker_status.v1","identity":self.identity,"allocation":None,"config_sha256":config_sha,"started_at":now(),"pid":os.getpid(),"python":sys.executable,"driver_receipts":str(self.root),"capability_accepted":False,"training_eligible":False,"numerical_targets_enforced":False,"completed_windows":[],"full_prefix_layers":48,"saved_valid_prefix_layers":48,"window_scope":"49-60","later_windows_admitted":False,"task_local_allocator_env_requested":{"PYTORCH_ALLOC_CONF":ALLOC_REQUEST},"RO_mapping":self.resolver.record()}
        new_json(self.root/"config.json",self.config)

    def source_check(self):
        pinned(self.config_path,self.config_sha);pinned(self.first_path,self.first_ref["sha256"]);pinned(TOOL,self.tool_sha)
        for name,expected in self.sources.items():pinned(CONTROL/name,expected)
        for name in ("ro_resume_paths.py","ro_resume_runtime.py","k3_recovery_ancestry.py"):
            pinned(TOOL.parent/name,self.sources[name])
        for field in ("resume_source_manifest","resume_source_observation","resume_source_identity","recovery_plan"):
            item=self.config[field];pinned(item["path"],item["sha256"])

    def event(self, stage, **values):
        record = {"at_utc": now(), "stage": stage, **values}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.status.update(record)
        current_json(META / "continuation_status.json", self.status)
        if self.window_path is not None:
            current_json(self.window_path / "status.json", self.status)

    def samples(self, stage):
        sample = resource_sample()
        with (self.root / "resources.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, **sample}, sort_keys=True) + "\n")
        return sample

    def run_command(self, argv, log, phase):
        require(isinstance(argv, list) and all(isinstance(x, str) and x and "\0" not in x for x in argv),
                "structured exact subprocess argv required")
        self.source_check()
        require(os.environ.get("PYTORCH_ALLOC_CONF") == ALLOC_REQUEST,
                "explicit task-local inherited allocator request changed")
        start = now()
        before = self.samples(phase + "_before")
        self.event(phase + "_started", command_argv=argv, command_started_at=start, log_path=str(log))
        peak = before["cgroup"]["memory.current"]
        with Path(log).open("xb") as handle:
            process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
            self.event(phase + "_running", child_pid=process.pid)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    sample = self.samples(phase + "_running")
                    peak = max(peak, sample["cgroup"]["memory.current"])
                    self.event(phase + "_running", child_pid=process.pid,
                               actual_resources=sample, observed_phase_cgroup_peak_bytes=peak)
        after = self.samples(phase + "_terminal")
        peak = max(peak, after["cgroup"]["memory.current"])
        result = {"phase": phase, "argv": argv, "pid": process.pid, "started_at": start,
                  "ended_at": now(), "exit_code": code, "log_path": str(log), "log_sha256": sha(log),
                  "resources_before": before, "resources_after": after,
                  "observed_phase_cgroup_peak_bytes": peak,
                  "cgroup_peak_is_container_lifetime": True,
                  "task_local_allocator_env_requested": {"PYTORCH_ALLOC_CONF": ALLOC_REQUEST},
                  "successful_recovery_allocator_proof": self.status.get("successful_recovery_allocator"),
                  "allocator_observation_scope": "task-local inherited request; no fresh segment observation or causal proof claimed"}
        new_json(Path(log).with_suffix(".execution.json"), result)
        self.event(phase + "_terminal", actual_command_result=result)
        require(code == 0, phase + " actual subprocess exit " + str(code) + "; preserved log " + str(log))
        self.source_check()
        return result

    def observe(self, plan_path, destination, phase):
        self.run_command([sys.executable, "-B", str(CONTROL / "observe_native.py"), str(plan_path),
                          "--image", self.identity["image"], "--gpu-uuid", GPU], destination, phase)
        observation = read(destination)
        require(observation.get("actual_resume_context") == self.current_context,
                "actually measured runtime/source/corpus differs from original resume context")
        require(observation.get("image_id") == self.identity["image"]
                and observation.get("gpu_uuid") == GPU
                and observation.get("cuda_canary", {}).get("finite") is True,
                "actual pinned runtime/CUDA observation failed")
        return observation

    def scanner_request(self, receipt, lo, hi):
        specs = []
        cfg = self.model_cfg
        for layer in range(lo, hi + 1):
            entry = receipt["layers"][str(layer)]["ffn"]
            path = Path(entry["path"])
            require(path == OUT / f"fold/dense/L{layer:02d}/ffn.safetensors"
                    and path.is_file() and not path.is_symlink(), "actual canonical emitted FFN required")
            hdr = header(path)
            meta = hdr["__metadata__"]
            width = int(meta["width_installed"])
            requested_rung = int(self.first_argv[self.first_argv.index("--rung") + 1])
            require(0 < width == entry["width_installed"]
                    and int(meta["K"]) == requested_rung
                    and int(meta["layer"]) == layer
                    and int(meta["latent_dim"]) == cfg["routed_expert_hidden_size"]
                    and int(meta["hidden"]) == cfg["hidden_size"]
                    and int(meta["shared_channels"]) == cfg["num_shared_experts"] * cfg["moe_intermediate_size"],
                    "actual FFN metadata/native receipt dimensions differ")
            specs.append({"layer": layer, "path": str(path), "sha256": entry["sha256"],
                          "width": width, "latent": cfg["routed_expert_hidden_size"],
                          "hidden": cfg["hidden_size"],
                          "shared_channels": cfg["num_shared_experts"] * cfg["moe_intermediate_size"],
                          "source_routed_channels": cfg["num_experts"] * cfg["moe_intermediate_size"],
                          "source_experts": cfg["num_experts"], "source_expert_channels": cfg["moe_intermediate_size"],
                          "requested_rung": requested_rung})
        return {"schema": "neural_forge.k3_functional_ffn_request.v2", "ffns": specs}

    def validate_window(self, window, plan_path, prior):
        lo, hi = window
        label = f"{lo}-{hi}"
        self.window_path = self.root / "windows" / label
        self.window_path.mkdir(parents=True, exist_ok=False)
        self.event("window_readback_started", completed_window=label)
        receipt_path = OUT / "receipts/k3_dense_pass_receipt.json"
        receipt_sha = sha(receipt_path)
        receipt = read(receipt_path)
        require(receipt.get("arguments", {}).get("snapshot_keep") == 0
                and receipt.get("snapshots", {}).get("keep") == 0,
                "actual native checkpoint retention0 required")
        require(receipt.get("resume_context") == self.current_context, "terminal native resume identity changed")
        require(set(receipt.get("installed_ffn_sha256", {})) == {str(i) for i in range(1, hi + 1)},
                "actual terminal installed prefix coverage mismatch")
        terminal_copy = self.window_path / "terminal_receipt.json"
        shutil.copy2(receipt_path, terminal_copy)
        require(not terminal_copy.samefile(receipt_path) and sha(terminal_copy) == receipt_sha,
                "actual terminal receipt copy mismatch")
        request_path = self.window_path / "ffn_request.json"
        new_json(request_path, self.scanner_request(receipt, lo, hi))
        scanner_path = self.window_path / "ffn_validation.json"
        try:
            self.run_command([sys.executable, "-B", str(CONTROL / "validate_k3_ffn.py"),
                              "--request", str(request_path), "--request-sha256", sha(request_path),
                              "--out", str(scanner_path)], self.window_path / "scanner.log", "ffn_full_scan")
        except ValueError:
            # Preserve the actual sealed checker refusal before requesting a
            # reader repair. This proves neither tensor corruption nor finite
            # coefficients in the unread lower-width artifact.
            if scanner_path.is_file():
                failed_scan = read(scanner_path)
                specs = {x["layer"]: x for x in read(request_path)["ffns"]}
                failures = [x for x in failed_scan.get("ffns", []) if not x.get("functional_artifact_usable")]
                if failures and all(x.get("error") == "native metadata mismatch: K"
                                    and specs[x["layer"]]["width"] != specs[x["layer"]]["requested_rung"] for x in failures):
                    self.event("reader_contract_mismatch", actual_checker_receipt=str(scanner_path),
                               actual_checker_receipt_sha256=sha(scanner_path),
                               tensor_finite_readback_incomplete=True, tensor_corruption_claimed=False)
                    raise ReaderContractMismatch("sealed e267 reader conflates requested K with actual installed width; actual refusal preserved at " + str(scanner_path))
            raise
        scan = read(scanner_path)
        require(scan.get("schema") == "neural_forge.k3_functional_ffn_receipt.v2"
                and scan.get("functional_artifacts_usable") is True
                and all(x.get("actual_library_load_verified") is True
                        and x.get("all_coefficients_finite") is True for x in scan["ffns"]),
                "actual finite full-byte safetensors scanner proof failed")
        runtime = self.window_path / "runtime_readback.json"
        self.observe(plan_path, runtime, "native_readback_observation")
        seal = OUT / "ancestry" / ("window-" + label + "-" + self.config["continuation_uuid"])
        command = [sys.executable, "-B", str(CONTROL / "verify_later_window.py"),
                   "--output", str(OUT), "--first-plan", str(self.first_path),
                   "--first-plan-sha256", self.first_ref["sha256"],
                   "--baseline-receipt", str(self.baseline_path),
                   "--runtime", str(runtime), "--runtime-sha256", sha(runtime),
                   "--nf", str(NF), "--parent", str(PARENT), "--corpus", str(CORPUS),
                   "--digest", str(DIGEST), "--receipt", str(receipt_path), "--receipt-sha256", receipt_sha,
                   "--prep-receipt", str(self.prep_path), "--prep-receipt-sha256", self.prep_ref["sha256"],
                   "--completed-window", label, "--seal-dir", str(seal),
                   "--ffn-validation", str(scanner_path), "--ffn-validation-sha256", sha(scanner_path)]
        if prior is not None:
            command += ["--prior-continuity", str(prior), "--prior-continuity-sha256", sha(prior)]
        command += self.resume_argv
        self.run_command(command, self.window_path / "verifier.log", "window_ancestry_verification")
        continuity_path = seal / "window_continuity.json"
        continuity = read(continuity_path)
        require(continuity.get("completed_window") == label and continuity.get("full_prefix_layers") == hi
                and continuity.get("ffn_full_scan_complete") is True, "actual validated window seal mismatch")
        self.status["completed_windows"].append({"layers": label, "continuity_path": str(continuity_path),
                                                  "continuity_sha256": sha(continuity_path),
                                                  "terminal_receipt_sha256": receipt_sha})
        self.event("window_validated", full_prefix_layers=hi, actual_resources=self.samples("window_validated"))
        new_json(self.window_path / "terminal_status.json", self.status)
        return continuity_path, continuity, receipt

    def admit(self, receipt, next_window, destination, resume_layer=None):
        require(next_window == (49, 60) and resume_layer is None,
                "only full49-60 phase admission required")
        require(receipt.get("resume_context") == self.guard.previous_context
                and set(receipt.get("installed_ffn_sha256", {})) == {str(i) for i in range(1, 49)}
                and receipt.get("layers_processed_this_run") == list(range(37, 49)),
                "authentic actual closed48 native producer receipt required")
        sample = self.samples("next_window_admission")
        group = sample["cgroup"]
        require(group["memory.max"] == CAP_BYTES and group["memory.swap.max"] == group["memory.swap.current"] == 0
                and group["cpu.max"]["cpus"] == CAP_CPUS,
                "new actual owned source-selected cap and noSwap required")
        additional = max(0, CAP_BYTES - group["memory.current"])
        host_need = additional + 16 * GIB
        # Exact read-map hashing/stat guards bind all48 FFN/body files. Older held
        # payloads need no recopy/read: their actual closed36 maximum is preserved
        # in the pinned historical resource proof and MAXed with all12 new rows.
        historical = pinned(self.config["closed36_disk_projection"]["path"],
                            self.config["closed36_disk_projection"]["sha256"])
        prior = read(historical)
        require(prior["next_window"] == "37-48" and prior["disk_estimate"]["required_new_bytes"] == 103035350992,
                "unaltered authentic closed36 historical layer maximum proof required")
        max_ffn = max(self.resolver.read_path(f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors").stat().st_size
                      for layer in range(1, 49))
        max_body = max(self.resolver.read_path(f"/run-output/body/L{layer:02d}/body.safetensors").stat().st_size
                       for layer in range(1, 49))
        held_sizes = [prior["disk_estimate"]["max_actual_held_rows_bytes"]]
        for layer in range(37, 49):
            held = receipt["layers"][str(layer)]["held_rows"]
            logical = f"/run-output/fold/resid/L{layer:02d}/held_rows.safetensors"
            require(held["path"] == logical, "actual new37-48 held-row source role required")
            path = self.resolver.read_path(logical)
            require(path.stat().st_size == held["bytes"] and self.resolver.old[logical]["sha256"] == held["sha256"],
                    "actual held-row fullSHA/extent differs from latest sealed producer")
            held_sizes.append(path.stat().st_size)
        boundary = self.resolver.read_path("/run-output/fold/snapshots/before_L49.safetensors")
        h = header(boundary)
        tokens = self.baseline["corpus"]["tokens"]
        hidden = self.model_cfg["hidden_size"]
        dtype = h["prefix_sum"]["dtype"]
        require(dtype == "BF16" and h["prefix_sum"]["shape"] == [tokens, hidden]
                and h["block_residual"]["dtype"] == dtype
                and h["block_residual"]["shape"] == [tokens, 5, hidden]
                and self.model_cfg["attn_res_block_size"] == 12,
                "actual before49 BF16/block5 geometry required")
        blocks = math.ceil(61 / self.model_cfg["attn_res_block_size"])
        dual = 2 * tokens * hidden * (1 + blocks) * 2
        from ro_resume_paths import NativeResumeReads
        terminal = NativeResumeReads(self.resolver).sealed_metadata("k3_dense_pass_receipt.json")
        require(terminal is not None and read(terminal) == receipt,
                "unique latest sealed48 full receipt required for metadata extent")
        metadata = terminal.stat().st_size * 16
        max_held = max(held_sizes)
        projection = {"new_layers": 12, "boundary_copies_budgeted": 3,
                      "dual_boundary_bytes": dual, "max_actual_ffn_bytes": max_ffn,
                      "max_actual_body_bytes": max_body, "max_actual_held_rows_bytes": max_held,
                      "metadata_bytes": metadata, "saved_head_bytes": 0,
                      "required_new_bytes": 12 * (max_ffn + max_body + max_held) + 3 * dual + metadata,
                      "method": "actual whole48 RO FFN/body maxima and historical held maximum MAX actual12 newheld; BF16/block6 three boundary pairs plus actual sealed48 receipt metadata",
                      "old_prefix_tensor_copy_bytes": 0}
        admission = {"schema": "neural_forge.k3_actual_next_window_resource_admission.v1",
                     "next_window": "49-60", "actual_resources": sample,
                     "current_owned_memory_limit_bytes": CAP_BYTES,
                     "current_owned_resident_bytes": group["memory.current"],
                     "additional_owned_budget_bytes": additional, "host_headroom_bytes": 16 * GIB,
                     "host_available_required_bytes": host_need, "disk_estimate": projection,
                     "new_output_filesystem": "NAS", "old_prefix_readonly": True,
                     "memory_admitted": sample["host_memory"]["MemAvailable_bytes"] >= host_need,
                     "disk_admitted": sample["output_disk"]["free_bytes"] >= projection["required_new_bytes"],
                     "numerical_quality_gate": False, "overall_92_completion_claimed": False}
        new_json(destination, admission)
        self.event("next_window_resource_admission", resource_admission=admission)
        if not admission["memory_admitted"]:
            raise CapacityDenied("actual RAM insufficient for new own growth plus16GiB headroom")
        if not admission["disk_admitted"]:
            raise CapacityDenied("actual NAS free insufficient for full49-60 new writes")
        self.own_capacity_admit(admission, destination, None)

    def own_capacity_admit(self, admission, admission_path, resume_layer):
        # Compute and pin the exact projection now; publish no capacity request
        # until the real native runtime observer has completed.
        self.pending_capacity_admission = {"path": Path(admission_path),
                                           "sha256": sha(admission_path), "resume_layer": resume_layer}

    def own_capacity_publish(self, admission, admission_path, resume_layer):
        caller = self.config["capacity_caller"]
        require(set(caller) == {"host_caller_config_sha256", "source_scripts", "source_pins"},
                "exact frozen own capacity caller config/scripts/source roster required")
        attempt = str(uuid.uuid4())
        requests, responses = META / "capacity_requests", META / "capacity_responses"
        for path in (requests, responses):
            path.mkdir(parents=True, exist_ok=True)
            require(path.is_dir() and not path.is_symlink() and path.resolve() == path,
                    "canonical own capacity metadata directory required")
        projection = admission["disk_estimate"]
        request = {"schema": "neural_forge.k3_driver_capacity_request.v1", "at": now(),
                   "attempt_uuid": attempt, "request_boundary_id": attempt,
                   "run_uuid": self.identity["run_uuid"], "allocation_id": self.allocation["allocation_id"],
                   "container_id": self.allocation["container_id"], "driver_sha256": sha(__file__),
                   "config_sha256": self.config_sha, "next_window": admission["next_window"],
                   "resume_layer": resume_layer, "own_remaining_write_bytes": projection["required_new_bytes"],
                   "remaining_copy_bytes": 0, "native_tool_sha256": self.tool_sha,
                   "projection": projection, "projection_sha256": capacity_object_sha(projection),
                   "admission_sha256": sha(admission_path), "identity_sha256": sha(CONTROL / "identity.json"),
                   "source_pins_sha256": capacity_object_sha(self.sources),
                   "allocation_receipt_sha256": self.status["allocation_receipt_sha256"],
                   "recovery_packet_sha256": self.guard.packet_sha}
        request_path = requests / ("launch-request-" + attempt + ".json")
        new_json(request_path, request)
        request_sha = sha(request_path)
        capacity_publish_sha(request_path.with_suffix(".sha256"), request_sha)
        self.source_check()
        self.event("owner_capacity_requested", request_path=str(request_path), request_sha256=request_sha,
                   next_window=admission["next_window"], actual_remaining_write_bytes=request["own_remaining_write_bytes"])
        self.pending_capacity = {"request_path": request_path, "request_sha256": request_sha,
                                 "responses": responses, "admission_path": Path(admission_path),
                                 "admission_sha256": request["admission_sha256"]}
        proof = capacity_await_decision(request_path, request_sha, responses, caller, self.identity)
        proof_path = Path(admission_path).with_name(Path(admission_path).stem + ".owner_capacity.json")
        new_json(proof_path, proof)
        self.event("owner_capacity_admitted", owner_capacity_proof=str(proof_path), owner_capacity_proof_sha256=sha(proof_path))

    def own_capacity_recheck(self, observation, plan_path):
        # Publish and await a new actual host request only after the native
        # observer, then fully re-read the decision immediately before compute.
        retry = 0
        while True:
            self.source_check()
            held = self.pending_capacity_admission
            admission = read(pinned(held["path"], held["sha256"]))
            if retry:
                observation = self.observe(plan_path, self.window_path / f"runtime_before_capacity_retry_{retry:05d}.json",
                                           "native_prelaunch_observation")
            try:
                self.own_capacity_publish(admission, held["path"], held["resume_layer"])
            except MemoryCapacityUnavailable as error:
                path = self.window_path / f"memory_capacity_denied_{retry:05d}.json"
                new_json(path, error.proof)
                self.event("owner_capacity_waiting_for_memory", denial_path=str(path), denial_sha256=sha(path),
                           retry_in_seconds=30, concrete_reason=str(error), native_fitter_started=False)
                retry += 1
                time.sleep(30)
                continue
            break
        pending = self.pending_capacity
        pinned(pending["admission_path"], pending["admission_sha256"])
        proof = capacity_read_decision(pending["request_path"], pending["request_sha256"], pending["responses"],
                                       self.config["capacity_caller"], self.identity)
        path = pending["admission_path"].with_name(pending["admission_path"].stem + ".owner_capacity.rechecked.json")
        new_json(path, proof)
        self.event("owner_capacity_rechecked_before_native", owner_capacity_recheck=str(path), owner_capacity_recheck_sha256=sha(path))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(observation["observed_utc"].replace("Z", "+00:00"))).total_seconds()
        require(0 <= age <= 300, "actual prelaunch runtime observation is stale after owner capacity wait")

    def execute(self):
        self.event("continuation_worker_ready",actual_resources=self.samples("worker_ready"))
        barrier=META/"continuation.ready"
        while not barrier.exists():time.sleep(1)
        armed=read(barrier);require(armed["config_sha256"]==self.config_sha,"current frozen barrier/config mismatch")
        allocation_path=pinned(armed["allocation_receipt_path"],armed["allocation_receipt_sha256"]);attestation=read(allocation_path)
        require(allocation_path==META/"continuation_allocation.json" and attestation["identity"]==self.identity and attestation["config_sha256"]==self.config_sha and attestation["image_id"]==self.identity["image"],"exact own allocation/source/image required")
        self.allocation=attestation["allocation"];uuid.UUID(self.allocation["allocation_id"])
        require(self.allocation["owner"]==self.identity["owner"] and self.allocation["name"]==self.config["expected_allocator_name"] and self.allocation["gpu_uuid"]==GPU and self.allocation["container_id"]==armed["container_id"] and self.allocation["allocation_id"]==armed["allocation_id"],"actual own unique allocation differs")
        hostname=Path("/etc/hostname").read_text().strip();require(hostname in (self.allocation["container_id"],self.allocation["container_id"][:12]),"actual PID1 container identity mismatch")
        mounts=self.allocation["details"]["mounts"]
        require(mounts==self.config["current_mounts"] and len(mounts)==10 and next(m for m in mounts if m["target"]==str(OUT))=={"source":HOST_WORKING,"target":str(OUT),"read_only":False} and next(m for m in mounts if m["target"]==str(PREFIX))=={"source":HOST_PREFIX,"target":str(PREFIX),"read_only":True} and next(m for m in mounts if m["target"]==str(NAS_PREFIX))=={"source":PREVIOUS_NAS,"target":str(NAS_PREFIX),"read_only":True},"exact ten disjoint two-origin RO/fresh NAS mounts required")
        self.status.update(allocation=self.allocation,allocation_receipt_sha256=armed["allocation_receipt_sha256"])
        self.source_check();self.window_path=self.root/"window49_60_launch";self.window_path.mkdir(exist_ok=False)
        prior=pinned(self.config["prior_continuity"]["path"],self.config["prior_continuity"]["sha256"])
        closed=read(prior);require(closed["full_prefix_layers"]==48 and closed["completed_window"]=="37-48" and str(prior)==self.guard.prior["path"] and sha(prior)==self.guard.prior["sha256"],"actual sealed48 boundary required")
        from ro_resume_paths import NativeResumeReads
        receipt_path=NativeResumeReads(self.resolver).sealed_metadata("k3_dense_pass_receipt.json")
        require(receipt_path is not None,"actual unique latest sealed48 terminal receipt required")
        receipt=read(receipt_path)
        plan_ref=self.config["recovery_plan"];plan_path=pinned(plan_ref["path"],plan_ref["sha256"]);plan=read(plan_path)
        expected=operational_argv(self.first_argv,"49-60")+self.resume_argv
        require(plan["command_argv"]==expected and plan["next_layers"]=="49-60" and plan["mounts"]==mounts and plan["image_id"]==self.identity["image"] and plan["required_gpu_uuid"]==GPU,"exact source-backed49-60 command/identity required")
        require(not any((OUT/leaf).exists() for leaf in ("fold","body","receipts","ancestry")),"new NAS output must have no preexisting model artifacts")
        self.admit(receipt,(49,60),self.window_path/"resource_admission.json")
        self.run_command([sys.executable,"-B",str(CONTROL/"native_nas_io_qualification.py"),"--identity",str(CONTROL/"identity.json"),"--identity-sha256",sha(CONTROL/"identity.json"),"--out",str(OUT)],self.window_path/"NAS_io_qualification.json","NAS_io_qualification")
        observation=self.observe(plan_path,self.window_path/"runtime_before_launch.json","native_prelaunch_observation")
        self.own_capacity_recheck(observation,plan_path)
        self.event("window_launch_armed",active_window="49-60",command_argv=expected,plan_sha256=sha(plan_path),previous_continuity_sha256=sha(prior),RO_mapping=self.resolver.record(),actual_resources=self.samples("window_launch"))
        self.run_command(expected,self.window_path/"fitter-49-60.log","native_fitter")
        continuity_path,continuity,new_receipt=self.validate_window((49,60),plan_path,prior)
        require(new_receipt.get("layers_processed_this_run")==list(range(49,61)) and new_receipt.get("start",{}).get("layer")==49 and new_receipt["start"].get("installed_layers")==list(range(1,49)),"actual49-60 processing and preserved48 resume required")
        self.event("bounded_window_complete",full_prefix_layers=60,final_continuity_path=str(continuity_path),final_continuity_sha256=sha(continuity_path),ended_at=now(),later_windows_admitted=False)
        new_json(self.root/"terminal_result.json",self.status)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    driver = None
    try:
        driver = Driver(args.config, args.config_sha256)
        return driver.execute()
    except Exception as exc:
        error_stage = "continuation_resource_refused" if isinstance(exc, CapacityDenied) else ("continuation_reader_contract_mismatch" if isinstance(exc, ReaderContractMismatch) else "continuation_functional_error")
        error = {"at_utc": now(), "stage": error_stage, "error_type": type(exc).__name__,
                 "error": str(exc), "partial_artifacts_preserved": True,
                 "provider_restore_owned_by_root": True, "capability_accepted": False,
                 "config_path": str(args.config), "config_sha256": args.config_sha256,
                 "actual_driver_sha256": sha(__file__)}
        if driver is not None and hasattr(driver, "root") and driver.root.exists():
            try:
                driver.event(error_stage, **{k: v for k, v in error.items() if k not in {"at_utc", "stage"}})
                new_json(driver.root / "error.json", error)
            except Exception as receipt_error:
                error["receipt_error"] = str(receipt_error)
        else:
            try:
                new_json(META / ("continuation_worker_error-" + str(uuid.uuid4()) + ".json"), error)
            except Exception as receipt_error:
                error["receipt_error"] = str(receipt_error)
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
