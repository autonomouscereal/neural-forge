"""One manifest-bound transition from the genuine failed K3 diagnostic source.

This module never changes a producer context. Old-context files are accepted
only by their exact full hash in the pinned import packet. New runtime identity
keeps the actually executing source hash. The packet is task-local environment
input; it is not a global or automatic compatibility policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

LEGACY_TOOL_SHA = "44ccfba059f0da1a5b2c63dfcbbb54e2f4e2b641fed5d0273cc00437cb6d8dde"
SCHEMA = "neural_forge.k3_diagnostic_source_recovery_ancestry.v1"
PACKET_ENV = "K3_DIAGNOSTIC_RECOVERY_PACKET"
PACKET_SHA_ENV = "K3_DIAGNOSTIC_RECOVERY_PACKET_SHA256"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate recovery JSON key: " + key)
        result[key] = value
    return result


def read(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(result, dict), "recovery JSON object required")
    return result


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "explicit recovery SHA256 required")
    return value


def canonical_path(value):
    require(isinstance(value, str), "recovery path string required")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts
            and any(path.is_relative_to(PurePosixPath(root)) and str(path) != root
                    for root in ("/run-output", "/recovery-source")),
            "canonical recovery file under actual output or readonly ancestry mount required")
    return value


def only_tool_hash_changed(legacy, current, current_sha):
    if not isinstance(legacy, dict) or not isinstance(current, dict):
        return False
    expected = copy.deepcopy(legacy)
    if expected.get("software", {}).get("tool_sha256") != LEGACY_TOOL_SHA:
        return False
    expected["software"]["tool_sha256"] = current_sha
    return current_sha != LEGACY_TOOL_SHA and expected == current


class RecoveryAncestry:
    def __init__(self, packet_path, packet_sha, current_source):
        self.packet_path = Path(packet_path)
        self.packet_sha = digest(packet_sha)
        require(self.packet_path.is_file() and not self.packet_path.is_symlink(), "regular recovery packet required")
        require(sha(self.packet_path) == self.packet_sha, "recovery packet full hash mismatch")
        self.packet = read(self.packet_path)
        p = self.packet
        require(p.get("schema") == SCHEMA and p.get("legacy_tool_sha256") == LEGACY_TOOL_SHA
                and p.get("prefix_through_layer") == 35 and p.get("failed_layer") == 36,
                "only genuine saved35 / failed36 recovery packet supported")
        self.current_sha = digest(p.get("current_tool_sha256"))
        require(Path(current_source).is_file() and not Path(current_source).is_symlink()
                and sha(current_source) == self.current_sha, "actual revised source full hash mismatch")
        self.legacy_context = p["legacy_resume_context"]
        self.current_context = p["current_resume_context"]
        require(only_tool_hash_changed(self.legacy_context, self.current_context, self.current_sha),
                "recovery may change only software.tool_sha256 in the actual resume context")
        require(self.legacy_context.get("version") == "k3_dense_pass.resume_context.v1"
                and self.legacy_context["model"]["source_directory"] == "/parent"
                and self.legacy_context["runtime"]["divergence_rows"] == 0,
                "genuine unchanged all-row context required")
        self.files = self._rows(p["imported_files"])
        require(all(path.startswith("/run-output/") for path in self.files), "active byte copies must use /run-output")
        self.historical_files = self._rows(p["historical_files"])
        self.failure_files = self._rows(p["failure_ancestry_files"])
        require(p.get("historical_root") == "/recovery-source" and p.get("historical_source_read_only") is True
                and self.failure_files and all(path.startswith("/recovery-source/")
                    for path in (*self.historical_files, *self.failure_files)),
                "genuine failed attempt must remain under readonly historical source mount")
        require(not (set(self.files) & set(self.failure_files)), "active import aliases failure ancestry")
        failed_required = {"/recovery-source/fold/dense/L36/ffn.safetensors",
                           "/recovery-source/body/L36/body.safetensors",
                           "/recovery-source/fold/resid/L36/held_rows.safetensors",
                           "/recovery-source/receipts/k3_dense_pass_receipt.partial.json",
                           "/recovery-source/fold/snapshots/snapshots_manifest.json"}
        require(failed_required <= set(self.failure_files)
                and any(path.endswith("/windows/13-24/fitter-25-36.log") for path in self.failure_files)
                and any(path.endswith("/windows/13-24/fitter-25-36.execution.json") for path in self.failure_files)
                and any(path.endswith("/error.json") for path in self.failure_files),
                "genuine L36 partial files, saved35 partial, log, execution and error must remain failure ancestry")
        for path in self.files:
            require(not re.search(r"/(?:dense|resid|body)/L(?:3[6-9]|[4-9][0-9])/", path),
                    "failed36 or later cannot be imported as valid saved prefix")
        self.prior = p["prior_continuity"]
        canonical_path(self.prior["path"])
        digest(self.prior["sha256"])
        require(self.historical_files.get(self.prior["path"], {}).get("sha256") == self.prior["sha256"],
                "preceding24 continuity must be a full-hash imported file")
        required = {f"/run-output/fold/dense/L{i:02d}/{name}" for i in range(1, 36)
                    for name in ("done.json", "ffn.safetensors")}
        required |= {"/run-output/fold/snapshots/before_L36.safetensors",
                     "/run-output/fold/snapshots/before_L36_folded.safetensors",
                     "/run-output/fold/snapshots/snapshots_manifest.json",
                     "/run-output/receipts/k3_dense_pass_receipt.json",
                     "/run-output/receipts/k3_dense_pass_receipt.partial.json"}
        require(required <= set(self.files), "full saved35 markers/FFNs and nativebefore36 pair required")
        self.checked = {}

    @staticmethod
    def _rows(rows):
        require(isinstance(rows, list), "recovery file list required")
        result = {}
        for row in rows:
            require(isinstance(row, dict), "recovery file row required")
            path = canonical_path(row["path"])
            require(path not in result and type(row.get("bytes")) is int and row["bytes"] >= 0,
                    "unique recovery file with exact byte count required")
            digest(row["sha256"])
            result[path] = row
        return result

    @classmethod
    def from_environment(cls, current_source, required=False):
        path, expected = os.environ.get(PACKET_ENV, ""), os.environ.get(PACKET_SHA_ENV, "")
        require(bool(path) == bool(expected), "task-local recovery packet path/hash must be supplied together")
        if not path:
            require(not required, "explicit task-local recovery packet required")
            return None
        return cls(path, expected, current_source)

    def pin(self, path, failure=False):
        logical = canonical_path(str(path))
        row = (self.failure_files if failure else {**self.files, **self.historical_files}).get(logical)
        require(row is not None, "old producer file is not authorized by recovery packet: " + logical)
        actual = Path(logical)
        require(actual.is_file() and not actual.is_symlink()
                and actual.resolve(strict=True).is_relative_to(Path("/recovery-source" if logical.startswith("/recovery-source/")
                    else "/run-output").resolve(strict=True)),
                "recovery file escapes actual output mount")
        before = actual.stat()
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if self.checked.get(logical) != stamp:
            require(before.st_size == row["bytes"] and sha(actual) == row["sha256"],
                    "full byte-identical recovery import/failure hash mismatch: " + logical)
            after = actual.stat()
            require(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "recovery file changed during full hash")
            self.checked[logical] = stamp
        return row

    def matches(self, producer_context, path):
        if producer_context == self.current_context:
            return True
        if producer_context != self.legacy_context:
            return False
        require(str(path) in self.files or str(path) in self.historical_files,
                "legacy producer file must be explicitly listed in manifest")
        self.pin(path)
        return True

    def matches_stamp(self, producer_context_json, path):
        try:
            context = json.loads(producer_context_json, object_pairs_hook=unique)
        except (TypeError, ValueError):
            return False
        return self.matches(context, path)

    def preflight(self):
        """One actual full import/failure-ancestry pass before any fitter launch."""
        for path in self.files:
            self.pin(path)
            row = self.files[path]
            source = row.get("source_path", "/recovery-source/" + path.removeprefix("/run-output/"))
            canonical_path(source)
            require(source.startswith("/recovery-source/"), "original import source must be readonly failed-attempt mount")
            original = Path(source)
            require(original.is_file() and not original.is_symlink()
                    and original.resolve(strict=True).is_relative_to(Path("/recovery-source").resolve(strict=True))
                    and not original.samefile(Path(path))
                    and original.stat().st_size == row["bytes"] and sha(original) == row["sha256"],
                    "actual original and active byte copy differ: " + source)
        for path in self.historical_files:
            self.pin(path)
        for path in self.failure_files:
            self.pin(path, failure=True)
        prior = read(self.prior["path"])
        require(prior.get("full_prefix_layers") == 24 and prior.get("completed_window") == "13-24"
                and prior.get("ffn_full_scan_complete") is True and prior.get("resume_context") == self.legacy_context,
                "genuine sealed completed24 context and scanner proof required")
        partial = read("/run-output/receipts/k3_dense_pass_receipt.partial.json")
        require(Path("/run-output/receipts/k3_dense_pass_receipt.partial.json").stat().st_mtime_ns
                > Path("/run-output/receipts/k3_dense_pass_receipt.json").stat().st_mtime_ns,
                "byte copies must preserve saved35 partial as strictly newest native resume receipt")
        require(not Path("/recovery-source/fold/dense/L36/done.json").exists(),
                "genuine failed36 must have no reusable done marker")
        require(partial.get("resume_context") == self.legacy_context and partial.get("refusal") is None
                and set(partial.get("layers", {})) == {str(i) for i in range(36)},
                "actual saved partial must contain exactly L00..L35")
        installed = {}
        for layer in range(1, 36):
            entry = partial["layers"][str(layer)]
            marker_path = f"/run-output/fold/dense/L{layer:02d}/done.json"
            marker = read(marker_path)
            ffn = entry["ffn"]
            require(marker.get("resume_context") == self.legacy_context and marker.get("layer") == layer
                    and marker.get("rung") == 17408 and marker.get("moe") is True
                    and marker.get("corpus_sha256") == self.legacy_context["corpus"]
                    and marker.get("artefact") == ffn["path"]
                    and marker.get("artefact_sha256") == ffn["sha256"] == self.files[ffn["path"]]["sha256"]
                    and entry.get("prefix_lineage") == marker.get("prefix_lineage") ==
                    {"version": "k3_dense_pass.prefix_lineage.v1", "upstream_ffn_sha256": dict(installed)},
                    "saved prefix marker/native partial lineage mismatch")
            require(ffn.get("done_marker_sha256", self.files[marker_path]["sha256"]) == self.files[marker_path]["sha256"],
                    "saved partial done-marker full hash mismatch")
            installed[str(layer)] = ffn["sha256"]
        prior_prefix = prior.get("prefix", [])
        require(len(prior_prefix) == 24 and all(row.get("layer") == layer
                and row.get("ffn_sha256") == self.files[f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors"]["sha256"]
                and row.get("done_sha256") == self.files[f"/run-output/fold/dense/L{layer:02d}/done.json"]["sha256"]
                for layer, row in enumerate(prior_prefix, 1)),
                "authentic sealed24 prefix differs from freshly hashed original/active bytes")
        # Reuse only the actual retained full coefficient proofs, after both
        # original and active artifacts have just been full-hashed above.
        prior5_path = self.prior["path"].replace("window-13-24-", "window-5-12-")
        self.pin(prior5_path)
        prior5 = read(prior5_path)
        require(self.historical_files[prior5_path]["sha256"] == prior.get("prior_window_continuity_sha256")
                and prior5.get("full_prefix_layers") == 12 and prior5.get("ffn_full_scan_complete") is True
                and prior5.get("resume_context") == self.legacy_context,
                "authentic retained completed12 ancestry must close completed24")
        proofs = [("/recovery-source/prefix_readback/bbce7113-be33-4132-983b-e3b74606e050/scanner_result.json",
                   "ec48b5252113575d68bc78d2222a6653cc25537d424865548037434f34198ba8"),
                  (str(Path(prior5_path).parent / "inputs/full_ffn_validation.json"), prior5["ffn_validation_sha256"]),
                  (str(Path(self.prior["path"]).parent / "inputs/full_ffn_validation.json"), prior["ffn_validation_sha256"])]
        covered = {}
        retained_scans = []
        for path, expected in proofs:
            row = self.pin(path)
            require(row["sha256"] == expected, "retained actual scanner hash differs from authentic ancestry")
            scan = read(path)
            require(scan.get("schema") in ("neural_forge.k3_functional_ffn_receipt.v1", "neural_forge.k3_functional_ffn_receipt.v2")
                    and scan.get("functional_artifacts_usable") is True, "retained actual prefix scanner must be usable")
            for record in scan["ffns"]:
                layer = record["layer"]
                require(type(layer) is int and 1 <= layer <= 24 and layer not in covered
                        and record.get("actual_library_load_verified") is True and record.get("all_coefficients_finite") is True
                        and record.get("functional_artifact_usable") is True
                        and record.get("sha256") == installed[str(layer)]
                        and record.get("bytes") == self.files[f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors"]["bytes"],
                        "retained scanner differs from byte-identical saved prefix")
                covered[layer] = record["sha256"]
            retained_scans.append({"path": path, "sha256": expected})
        require(set(covered) == set(range(1, 25)), "authentic retained full coefficient proofs must cover exactly1..24")
        return {**self.record(), "full_import_and_failure_hashes_verified": True,
                "saved_prefix_sha256": installed, "processed_layer_expected": 36,
                "retained_actual1_24_scanner_proofs": retained_scans,
                "retained_scanner_reuse_after_original_and_active_fullhash": True}

    def record(self):
        return {"schema": SCHEMA, "packet_path": str(self.packet_path), "packet_sha256": self.packet_sha,
                "legacy_tool_sha256": LEGACY_TOOL_SHA, "current_tool_sha256": self.current_sha,
                "saved_prefix_through_layer": 35, "failed_layer_preserved": 36,
                "only_context_change": "software.tool_sha256", "producer_contexts_rewritten": False,
                "prior_continuity_sha256": self.prior["sha256"]}

    def historical_path(self, relative):
        value = canonical_path("/recovery-source/" + str(relative))
        self.pin(value)
        return Path(value)


def producer_context_matches(current, producer, path, guard):
    return producer == current if guard is None else guard.matches(producer, path)


def snapshot_context_matches(current_json, producer_json, path, guard):
    return producer_json == current_json if guard is None else guard.matches_stamp(producer_json, path)
