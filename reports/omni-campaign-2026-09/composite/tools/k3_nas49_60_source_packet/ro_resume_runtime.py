"""Source-only49-60 candidate; authentic sealed48 and fresh identity are mandatory."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from pathlib import Path

from ro_resume_paths import ROResumePaths, NativeResumeReads, pinned_json, require, _unique
from k3_recovery_ancestry import RecoveryAncestry

PREVIOUS_RUN = "764524b1-96a9-4636-be74-942f9c54754c"
PREVIOUS_NATIVE = "9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69"
COMPLETED_NATIVE = "720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243"
ANCESTRY_SOURCE = "88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41"
PREVIOUS_PACKET = "5b9559b8ac20428de7e4ac09201f71839f4d03113a985715ca81d3aedd01246c"
CLOSED36 = "dd0ab8e821307e352096cdded457934dfd6df3434cfbe40d74e24b09a215870b"
SEALED36 = "/run-output/ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b"
LEXAR_ROOT = "/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318"
NAS37_48_ROOT = "/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model"


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def add_resume_source_arguments(parser):
    for name in ("manifest", "observation", "identity"):
        parser.add_argument("--resume-source-" + name)
        parser.add_argument("--resume-source-" + name + "-sha256")
    parser.add_argument("--resume-source-root")
    return parser


def context_equal(actual, expected):
    # Preserve JSON scalar types as well as values (True must not equal1).
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return False


def validate_closed48_metadata(closed, scanner, terminal, manifest, paths, expected_context):
    """Join authenticated full-byte metadata; this function creates no finite proof."""
    require(closed.get("schema") == "neural_forge.k3_actual_window_continuity.v1"
            and closed.get("completed_window") == "37-48" and closed.get("full_prefix_layers") == 48
            and closed.get("actual_layers_processed_this_terminal_run") == list(range(37, 49))
            and closed.get("ffn_full_scan_complete") is True
            and closed.get("authorized_NAS_window_only") == "37-48"
            and context_equal(closed.get("resume_context"), expected_context)
            and closed.get("prior_window_continuity_sha256") == CLOSED36,
            "actual complete sealed48/current720 context and closed36 predecessor required")
    prefix = closed.get("prefix")
    require(isinstance(prefix, list) and len(prefix) == 48, "complete actual48 prefix receipt required")
    installed = {}
    for layer, row in enumerate(prefix, 1):
        ffn = f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors"
        done = ffn.replace("ffn.safetensors", "done.json")
        require(row.get("layer") == layer and row.get("ffn_sha256") == paths.old[ffn]["sha256"]
                and row.get("done_sha256") == paths.old[done]["sha256"],
                "closed48 prefix/canonical origin byte pins differ")
        installed[str(layer)] = row["ffn_sha256"]
    require(context_equal(terminal.get("resume_context"), expected_context)
            and terminal.get("layers_processed_this_run") == list(range(37, 49))
            and terminal.get("installed_ffn_sha256") == installed and terminal.get("refusal") is None
            and terminal.get("start", {}).get("layer") == 37
            and terminal["start"].get("installed_layers") == list(range(1, 37)),
            "actual complete37-48 terminal/context/installed chain required")
    require(scanner.get("schema") == "neural_forge.k3_functional_ffn_receipt.v2"
            and scanner.get("functional_artifacts_usable") is True
            and isinstance(scanner.get("ffns"), list)
            and [row.get("layer") for row in scanner["ffns"]] == list(range(37, 49))
            and all(row.get("actual_library_load_verified") is True
                    and row.get("all_coefficients_finite") is True
                    and row.get("header_and_payload_complete") is True
                    and row.get("sha256") == installed[str(row["layer"])] for row in scanner["ffns"]),
            "authenticated complete37-48 actual full raw finite/library scanner required")
    endpoints = closed.get("endpoint_readback")
    require(isinstance(endpoints, list) and len(endpoints) == 2
            and {row.get("stream") for row in endpoints} == {"teacher", "folded"},
            "both authentic before49 endpoint readbacks required")
    for row in endpoints:
        suffix = "" if row["stream"] == "teacher" else "_folded"
        logical = f"/run-output/fold/snapshots/before_L49{suffix}.safetensors"
        recorded = manifest.get("files", {}).get(logical, {})
        end_ref = terminal.get("end_of_window_snapshot", {}).get("files", {}).get(row["stream"], {})
        require(row.get("layer") == 49 and row.get("actual_library_load_verified") is True
                and row.get("all_coordinates_finite") is True
                and row.get("sha256") == paths.old[logical]["sha256"] == recorded.get("sha256") == end_ref.get("sha256")
                and end_ref.get("path") == logical and recorded.get("layer") == 49
                and recorded.get("stream") == row["stream"] and recorded.get("installed") == installed
                and recorded.get("run_id") == terminal.get("run_id")
                and terminal.get("end_of_window_snapshot", {}).get("layer") == 49,
                "authentic before49 finite/library/manifest/terminal/origin lineage differs")
    return installed


class Closed48Ancestry:
    def __init__(self, paths, identity, pins, tool_path):
        self.paths, self.identity, self.input_pins = paths, identity, pins
        self.packet_path, self.packet_sha = Path(pins["manifest"]["path"]), pins["manifest"]["sha256"]
        self.current_sha = file_sha(tool_path)
        code = paths.proof["source_pins"]
        for basename, path in (("ro_resume_paths.py", Path(__file__).with_name("ro_resume_paths.py")),
                               ("ro_resume_runtime.py", Path(__file__)),
                               ("k3_recovery_ancestry.py", Path(__file__).with_name("k3_recovery_ancestry.py"))):
            require(file_sha(path) == code[basename], "actual shared compatibility source changed: " + basename)
        require(self.current_sha == code["k3_dense_pass.py"], "genuine executing derivative source changed")
        require(code["k3_recovery_ancestry.py"] == ANCESTRY_SOURCE, "original ancestry88 source identity required")
        prior_packet, prior_native = paths.proof["previous_recovery_packet"], paths.proof["previous_native_source"]
        completed_native = paths.proof["completed_native_source"]
        require(prior_packet["sha256"] == PREVIOUS_PACKET and prior_native["sha256"] == PREVIOUS_NATIVE
                and completed_native["sha256"] == COMPLETED_NATIVE,
                "exact legacy saved35/9086/720 source ancestry required")
        require(file_sha(prior_native["path"]) == PREVIOUS_NATIVE
                and file_sha(completed_native["path"]) == COMPLETED_NATIVE,
                "authentic old producer source bytes changed")
        self.original = RecoveryAncestry(prior_packet["path"], PREVIOUS_PACKET, prior_native["path"])
        self.legacy_context, self.closed36_context = self.original.legacy_context, self.original.current_context
        closed36_path = paths.verify_old_recorded_reference(SEALED36 + "/window_continuity.json", CLOSED36)
        closed36 = pinned_json(Path(closed36_path).read_bytes(), CLOSED36)
        require(closed36.get("full_prefix_layers") == 36 and closed36.get("completed_window") == "25-36"
                and closed36.get("ffn_full_scan_complete") is True
                and context_equal(closed36.get("resume_context"), self.closed36_context),
                "genuine sealed36 predecessor context required")
        self.previous_context = copy.deepcopy(self.closed36_context)
        self.previous_context["software"]["tool_sha256"] = COMPLETED_NATIVE
        closed_ref = paths.proof["closed48_continuity"]
        closed_path = paths.verify_old_recorded_reference(closed_ref["path"], closed_ref["sha256"])
        closed = pinned_json(Path(closed_path).read_bytes(), closed_ref["sha256"])
        seal = "/run-output/" + paths.closed48_sealed_root
        scanner_path = paths.verify_old_recorded_reference(seal + "/inputs/full_ffn_validation.json", closed.get("ffn_validation_sha256"))
        receipt_path = paths.verify_old_recorded_reference(seal + "/receipts/k3_dense_pass_receipt.json", closed.get("receipt_sha256"))
        manifest_path = paths.verify_old_recorded_reference(seal + "/fold/snapshots/snapshots_manifest.json", closed.get("snapshot_manifest_sha256"))
        scanner = pinned_json(Path(scanner_path).read_bytes(), closed["ffn_validation_sha256"])
        terminal = pinned_json(Path(receipt_path).read_bytes(), closed["receipt_sha256"])
        manifest = pinned_json(Path(manifest_path).read_bytes(), closed["snapshot_manifest_sha256"])
        self.installed = validate_closed48_metadata(closed, scanner, terminal, manifest, paths, self.previous_context)
        self.current_context = copy.deepcopy(self.previous_context)
        self.current_context["software"]["tool_sha256"] = self.current_sha
        require(self.current_sha not in (COMPLETED_NATIVE, PREVIOUS_NATIVE, self.legacy_context["software"]["tool_sha256"]),
                "new derivative must use its genuine new source identity")
        self.packet = {"current_mounts": copy.deepcopy(paths.proof["current_mounts"])}
        self.prior = {"path": str(closed_path), "sha256": closed_ref["sha256"]}

    def logical(self, path):
        value = str(path)
        for source in self.paths.sources.values():
            prefix = str(source) + "/"
            if value.startswith(prefix):
                return "/run-output/" + value[len(prefix):]
        if value.startswith("/run-output/"):
            return value
        raise ValueError("context file is outside exact two-origin RO prefix/new output")

    def pin(self, path, failure=False):
        if str(path).startswith("/recovery-source/"):
            return self.original.pin(path, failure=failure)
        logical = self.logical(path)
        require(logical in self.paths.old, "old context is not an exact authenticated prefix48 file")
        self.paths.read_path(logical)
        return self.paths.old[logical]

    def matches(self, producer_context, path):
        logical = self.logical(path)
        if context_equal(producer_context, self.current_context):
            return logical in self.paths.writes and logical not in self.paths.old
        if logical not in self.paths.old:
            return False
        row = self.paths.old[logical]
        expected = (self.closed36_context, self.legacy_context) if row["source_origin"] == "legacy_closed36" else (self.previous_context,)
        if not any(context_equal(producer_context, context) for context in expected):
            return False
        self.pin(path)
        return True

    def matches_stamp(self, producer_context_json, path):
        try:
            producer = json.loads(producer_context_json, object_pairs_hook=_unique)
        except (TypeError, ValueError):
            return False
        return self.matches(producer, path)

    def historical_path(self, relative):
        return self.original.historical_path(relative)

    def map_read(self, logical):
        return self.paths.read_path(logical)

    def record(self):
        return {"schema": "neural_forge.k3_closed48_two_origin_RO_NAS_ancestry.v1",
                "current_tool_sha256": self.current_sha, "completed_producer_tool_sha256": COMPLETED_NATIVE,
                "closed36_producer_tool_sha256": PREVIOUS_NATIVE,
                "legacy_tool_sha256": self.legacy_context["software"]["tool_sha256"],
                "closed48_continuity": copy.deepcopy(self.prior), "closed36_continuity_sha256": CLOSED36,
                "previous_recovery_packet_sha256": PREVIOUS_PACKET, "input_pins": copy.deepcopy(self.input_pins),
                "only_context_change": "software.tool_sha256", "producer_contexts_rewritten": False,
                "failed_layer_preserved": 36, "saved_prefix_through_layer": 48, "new_layer_range": [49, 60],
                "path_mapping": self.paths.record(), "runtime_admission_claimed": False}


def nas_resume_from_arguments(args, tool_path):
    pins, docs = {}, {}
    for name in ("manifest", "observation", "identity"):
        path = getattr(args, "resume_source_" + name, None)
        expected = getattr(args, "resume_source_" + name + "_sha256", None)
        require(isinstance(path, str) and bool(path) and isinstance(expected, str),
                "explicit full-pinned two-origin resume metadata required")
        p = Path(path)
        require(p.is_file() and not p.is_symlink(), "regular explicit RO metadata input required")
        raw = p.read_bytes()
        docs[name] = (raw, pinned_json(raw, expected))
        pins[name] = {"path": path, "sha256": expected, "bytes": len(raw)}
    identity = docs["identity"][1]
    run = identity.get("run_uuid")
    require(isinstance(run, str) and run != PREVIOUS_RUN, "genuine distinct future controller UUID required")
    uuid.UUID(run)
    require(identity.get("new_layers") == [49, 60] and identity.get("only_this_window_scope") is True
            and identity.get("resume_source_mounted_read_only") is True
            and re.fullmatch(r"[0-9a-f]{64}", identity.get("closed48_continuity_sha256") or "") is not None
            and isinstance(identity.get("native_binding"), dict)
            and type(identity["native_binding"].get("session_generation")) is int
            and identity["native_binding"]["session_generation"] >= 6,
            "actual sealed48/full current binding and bounded49-60 identity required")
    require(identity.get("readonly_resume_source_roots") == {"legacy_closed36": LEXAR_ROOT, "closed37_48": NAS37_48_ROOT},
            "exact preserved Lexar and currentNAS RO source roots required")
    acting = {"owner": identity["owner"], "native_context": identity["native_thread_id"], "run_uuid": run,
              "image": identity["image"], "control_binding": identity["native_binding"]}
    paths = ROResumePaths(docs["manifest"][0], pins["manifest"]["sha256"], docs["observation"][0], pins["observation"]["sha256"], acting_identity=acting)
    require(getattr(args, "resume_source_root", None) == "/resume-prefix"
            and paths.first == 49 and paths.last == 60
            and paths.proof["closed48_continuity"]["sha256"] == identity["closed48_continuity_sha256"]
            and {origin: root["host_path"] for origin, root in paths.source_roots.items()} == identity["readonly_resume_source_roots"]
            and paths.proof["output_root"]["host_path"] == identity["native_working_output_root"],
            "actual two-origin roots/new NAS phase and closed48 pin differ")
    mounts = {row["target"]: row for row in paths.proof["current_mounts"]}
    require(mounts.get("/run-control-output") == {"target": "/run-control-output", "source": identity["controller_output_root"], "read_only": False},
            "separate actual new controller mount required")
    return paths, Closed48Ancestry(paths, identity, pins, tool_path)
