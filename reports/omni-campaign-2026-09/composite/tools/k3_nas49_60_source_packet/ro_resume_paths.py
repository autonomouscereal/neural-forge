"""Candidate only: exact sealed reads and separately declared future writes.

The installed owner must pin this module, both manifests, actual mount/cap
evidence, and its current session before adoption. This module neither allocates
nor writes files. It never changes strings in historical receipts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import stat
import uuid
from pathlib import Path, PurePosixPath

CATALOG_SCHEMA = "chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1"
CONTRACT_SCHEMA = "chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1"
LOGICAL = "/run-output"
SOURCE_ALIAS = "/resume-prefix"
PREFIX_END = 48
ORIGIN_ALIASES = {"legacy_closed36": "/resume-prefix", "closed37_48": "/resume-window37-48"}
PARENT_CATALOG_SHA = "7a6b254fd7822bb2b868be149f2246b892868a7b2bbb0daa5f37db5ae1fe5cd2"
PARENT_ROWS_SHA = "c5d06a6f17377abdb50e948a830861f0b1539f83f4ca404c64d4a1166f3c7eb8"
NATIVE = "01a11108-009c-72f3-b2ba-e0d48fca1435"
OWNER = "codex-native-" + NATIVE
IMAGE = "sha256:d4ef16f161e51df2bd51df62c1aaa4ea7d9fc65fefcbaef8382027a527ec3ba6"
STAT_FIELDS = ("dev", "ino", "size", "mtime_ns", "ctime_ns", "mode")
ROOT_FIELDS = ("dev", "ino", "mode")


def require(value, message):
    if value is not True:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def pinned_json(raw, expected):
    require(isinstance(raw, bytes) and re.fullmatch(r"[0-9a-f]{64}", expected or "") is not None,
            "raw pinned JSON and complete SHA256 required")
    require(hashlib.sha256(raw).hexdigest() == expected, "source/observation manifest SHA differs")
    return json.loads(raw, object_pairs_hook=_unique)


def logical_relative(value):
    require(isinstance(value, str) and "\\" not in value and "\x00" not in value,
            "canonical POSIX logical string required")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts
            and path.is_relative_to(LOGICAL) and str(path) != LOGICAL,
            "canonical descendant of /run-output required")
    return path.relative_to(LOGICAL)


def _old_role(relative, role, origin, sealed_root):
    s = str(relative)
    if origin not in ORIGIN_ALIASES:
        return False
    lo, hi = (1, 36) if origin == "legacy_closed36" else (37, 48)
    if role in ("closed_prefix_ffn.safetensors", "closed_prefix_done.json"):
        m = re.fullmatch(r"fold/dense/L(\d{2})/(ffn\.safetensors|done\.json)", s)
        return m is not None and lo <= int(m[1]) <= hi and m[2] == role.removeprefix("closed_prefix_")
    if role == "closed_prefix_verbatim_body":
        m = re.fullmatch(r"body/L(\d{2})/body\.safetensors", s)
        return m is not None and lo <= int(m[1]) <= hi
    if role == "closed_prefix_retained_held_rows" and origin == "closed37_48":
        m = re.fullmatch(r"fold/resid/L(\d{2})/held_rows\.safetensors", s)
        return m is not None and 37 <= int(m[1]) <= 48
    if origin == "legacy_closed36":
        if role == "original_dense_L00_or_globals":
            return s in {"body/model_level.safetensors", "body/L00/body.safetensors",
                         "fold/dense/L00/mlp.safetensors", "fold/dense/L00/done.json"}
        if role == "native_before37_dual_boundary":
            return s in {"fold/snapshots/before_L37.safetensors", "fold/snapshots/before_L37_folded.safetensors"}
        if role == "sealed25_36_ancestry":
            prefix = "ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b/"
            return s.startswith(prefix) and s != prefix
    else:
        if role == "native_before49_dual_boundary":
            return s in {"fold/snapshots/before_L49.safetensors", "fold/snapshots/before_L49_folded.safetensors"}
        if role == "sealed37_48_ancestry":
            return s.startswith(sealed_root + "/") and _future_role(relative, 37, 48)
    return False


def _future_role(relative, first, last):
    s = str(relative)
    m = re.fullmatch(r"(?:fold/dense/L(\d{2})/(?:ffn\.safetensors|done\.json)|"
                     r"body/L(\d{2})/body\.safetensors|"
                     r"fold/resid/L(\d{2})/held_rows\.safetensors)", s)
    if m:
        layer = next(int(x) for x in m.groups() if x is not None)
        return first <= layer <= last
    m = re.fullmatch(r"fold/dense/L(\d{2})/L(\d{2})_K17408_formE\.safetensors", s)
    if m:
        return int(m[1]) == int(m[2]) and first <= int(m[1]) <= last
    m = re.fullmatch(r"fold/snapshots/before_L(\d{2})(?:_folded)?\.safetensors", s)
    if m:
        # Existing before37 is a read-only initial condition, never a new write.
        return first < int(m[1]) <= last + 1
    if s == "fold/snapshots/snapshots_manifest.json":
        return True
    if s in {"receipts/k3_dense_pass_receipt.json", "receipts/k3_dense_pass_receipt.partial.json",
             "receipts/k3_dense_pass_receipt.sha256"}:
        return True
    # Caller must still provide a finite, exact path allowlist. No directory
    # wildcard, inherited arbitrary ancestry path, or copied old prefix applies.
    m = re.fullmatch(r"ancestry/(window-\d+-\d+-[0-9a-f-]{36})/(.+)", s)
    if m:
        lo, hi = (int(x) for x in m[1].split("-")[1:3])
        inner = m[2]
        metadata_names = {"original_record_plan.json", "original_accepted_receipt.json",
                          "native_runtime_observation.json", "separate_copy_receipt.json",
                          "full_ffn_validation.json", "previous_window_continuity.json"}
        if inner in {"window_continuity.json", "ffn_request.json"} or inner in {"inputs/"+x for x in metadata_names}:
            return first <= lo <= hi <= last
        return first <= lo <= hi <= last and _future_role(PurePosixPath(inner), first, last)
    return False


class LocalFiles:
    """Production backend; tests inject explicitly synthetic filesystem doubles."""
    @staticmethod
    def identity(path):
        s = Path(path).lstat()
        return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size,
                    mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)

    @staticmethod
    def hash(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def exists(path):
        return Path(path).exists() or Path(path).is_symlink()


class ROResumePaths:
    def __init__(self, catalog_raw, catalog_sha256, observation_raw, observation_sha256,
                 *, acting_identity, backend=None):
        self.fs = backend if backend is not None else LocalFiles()
        self.catalog = pinned_json(catalog_raw, catalog_sha256)
        self.proof = pinned_json(observation_raw, observation_sha256)
        require(self.catalog.get("schema") == CATALOG_SCHEMA and self.proof.get("schema") == CONTRACT_SCHEMA,
                "exact two-origin catalog and observation schemas required")
        require(self.proof.get("source_catalog_sha256") == catalog_sha256,
                "physical observation bound to different source catalog")
        identity = self.proof.get("acting_identity")
        require(isinstance(identity, dict) and identity == acting_identity
                and all(identity.get(k) for k in ("owner", "native_context", "run_uuid", "image", "control_binding")),
                "exact current acting owner/native/run/image/full binding required")
        require(identity["owner"] == OWNER and identity["native_context"] == NATIVE and identity["image"] == IMAGE
                and identity["control_binding"].get("harness_id") == NATIVE,
                "fixed genuine K3 owner/native/immutable image required")
        uuid.UUID(identity["run_uuid"])
        self.acting_identity = copy.deepcopy(identity)
        self.first, self.last = self.proof.get("first_new_layer"), self.proof.get("last_new_layer")
        require(type(self.first) is int and type(self.last) is int and self.first == 49 and self.last == 60,
                "exact one-window49-60 suffix required")
        self.source_roots = self.proof.get("source_roots")
        require(isinstance(self.source_roots, dict) and set(self.source_roots) == set(ORIGIN_ALIASES),
                "exact Lexar closed36 and NAS closed37-48 origins required")
        self.sources = {origin: Path(root["path"]) for origin, root in self.source_roots.items()}
        self.source = self.sources["legacy_closed36"]
        self.output = Path(self.proof["output_root"]["path"])
        roots = list(self.source_roots.values()) + [self.proof["output_root"]]
        mounts = self.proof.get("current_mounts")
        require(isinstance(mounts, list) and len({row["target"] for row in mounts}) == len(mounts),
                "unique actual mounted targets required")
        mounts = {row["target"]: row for row in mounts}
        for origin, alias in ORIGIN_ALIASES.items():
            root = self.source_roots[origin]
            require(root.get("alias") == alias and root.get("read_only") is True,
                    "exact read-only origin alias required")
            if backend is None:
                require(str(self.sources[origin]) == alias, "production origin alias differs")
            require(mounts.get(alias) == {"target": alias, "source": root.get("host_path"), "read_only": True},
                    "actual full RO origin mount differs")
        output = self.proof["output_root"]
        require(output.get("alias") == LOGICAL and output.get("read_only") is False,
                "new output must be the exact own RW alias")
        if backend is None:
            require(str(self.output) == LOGICAL, "production new output alias differs")
        require(mounts.get(LOGICAL) == {"target": LOGICAL, "source": output.get("host_path"), "read_only": False},
                "actual full new RW output mount differs")
        host_output = PurePosixPath(output.get("host_path", ""))
        require(host_output.is_relative_to("/mnt/ai_data") and not host_output.is_relative_to("/mnt/ai_data/neural-forge-archives"),
                "separate own NAS output host mapping required")
        for root in roots:
            path, host = PurePosixPath(root.get("path", "")), PurePosixPath(root.get("host_path", ""))
            require(path.is_absolute() and host.is_absolute() and str(path) == root["path"]
                    and str(host) == root["host_path"] and ".." not in path.parts and ".." not in host.parts,
                    "canonical absolute roots required")
        for i, left in enumerate(roots):
            for right in roots[i + 1:]:
                for key in ("path", "host_path"):
                    a, b = PurePosixPath(left[key]), PurePosixPath(right[key])
                    require(a != b and not a.is_relative_to(b) and not b.is_relative_to(a),
                            "two RO origins and new output must be disjoint")
                require((left["stat"]["dev"], left["stat"]["ino"]) != (right["stat"]["dev"], right["stat"]["ino"]),
                        "different roots cannot name the same physical directory")
        self._check_roots()
        closed = self.proof.get("closed48_continuity")
        require(isinstance(closed, dict) and re.fullmatch(r"[0-9a-f]{64}", closed.get("sha256") or "") is not None,
                "unresolved actual closed48 continuity is not deployable")
        closed_rel = logical_relative(closed.get("path"))
        m = re.fullmatch(r"ancestry/(window-37-48-[0-9a-f-]{36})/window_continuity\.json", str(closed_rel))
        require(m is not None, "actual unique sealed37-48 continuity logical path required")
        uuid.UUID(m[1].split("window-37-48-", 1)[1])
        self.closed48_sealed_root = "ancestry/" + m[1]
        rows = self.catalog.get("input_catalog")
        require(isinstance(rows, list) and len(rows) > 127
                and self.catalog.get("parent_catalog_sha256") == PARENT_CATALOG_SHA,
                "complete unchanged127 ancestry plus actual37-48 input rows required")
        inherited = [{key: value for key, value in row.items() if key != "source_origin"} for row in rows[:127]]
        require(all(row.get("source_origin") == "legacy_closed36" for row in rows[:127])
                and hashlib.sha256(json.dumps(inherited, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == PARENT_ROWS_SHA,
                "original127 catalog row values must remain unchanged")
        self.old = {}
        for index, row in enumerate(rows):
            logical = row.get("recorded_logical_path", LOGICAL + "/" + row.get("relative_path", ""))
            rel = logical_relative(logical)
            origin = row.get("source_origin")
            require((origin == "legacy_closed36") == (index < 127),
                    "only unchanged127 rows may use the original Lexar origin")
            require(logical not in self.old and _old_role(rel, row.get("role"), origin, self.closed48_sealed_root),
                    "duplicate/unapproved origin/source role or source future-layer alias")
            root = self.source_roots[origin]
            require(re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None
                    and row.get("prospective_RO_input_path") == root["alias"] + "/" + str(rel)
                    and row.get("unchanged_host_input_path") == str(PurePosixPath(root["host_path"]) / rel),
                    "complete source SHA and exact origin alias/host mappings required")
            self.old[logical] = copy.deepcopy(row)
        required = {f"/run-output/fold/dense/L{i:02d}/{name}" for i in range(1, 49) for name in ("ffn.safetensors", "done.json")}
        required |= {f"/run-output/body/L{i:02d}/body.safetensors" for i in range(49)}
        required |= {f"/run-output/fold/resid/L{i:02d}/held_rows.safetensors" for i in range(37, 49)}
        required |= {"/run-output/body/model_level.safetensors", "/run-output/fold/dense/L00/mlp.safetensors", "/run-output/fold/dense/L00/done.json"}
        required |= {f"/run-output/fold/snapshots/before_L49{suffix}.safetensors" for suffix in ("", "_folded")}
        seal = LOGICAL + "/" + self.closed48_sealed_root
        required |= {seal + "/" + inner for inner in ("window_continuity.json", "inputs/full_ffn_validation.json", "receipts/k3_dense_pass_receipt.json", "receipts/k3_dense_pass_receipt.partial.json", "receipts/k3_dense_pass_receipt.sha256", "fold/snapshots/snapshots_manifest.json")}
        require(required <= set(self.old) and self.old[closed["path"]]["sha256"] == closed["sha256"],
                "complete actual prefix48 and latest closed48 metadata required")
        observed = self.proof.get("source_files")
        require(isinstance(observed, dict) and set(observed) == set(self.old), "all actual origin observations must join exactly")
        for logical, row in self.old.items():
            origin = row["source_origin"]
            fact, root = observed[logical], self.source_roots[origin]
            require(fact.get("source_origin") == origin and fact.get("source_sha256") == row["sha256"]
                    and fact.get("path") == str(self.sources[origin] / logical_relative(logical))
                    and set(fact.get("stat", {})) == set(STAT_FIELDS)
                    and fact["stat"]["dev"] == root["stat"]["dev"] and stat.S_ISREG(fact["stat"]["mode"]),
                    "observed full SHA/path/origin/device/regular identity differs")
            if "bytes" in row:
                require(fact["stat"]["size"] == row["bytes"], "source catalog byte extent differs")
        self.writes = set()
        declared = self.proof.get("new_write_paths")
        require(isinstance(declared, list) and declared and len(set(declared)) == len(declared), "finite exact new write paths required")
        for logical in declared:
            require(logical not in self.old and _future_role(logical_relative(logical), self.first, self.last),
                    "old/unlisted/outside49-60 write role refused")
            self.writes.add(logical)
        self._verified, self._issued_new, self._created_new = {}, set(), {}

    def _check_roots(self):
        roots = [(self.sources[origin], root) for origin, root in self.source_roots.items()]
        roots.append((self.output, self.proof["output_root"]))
        for path, root in roots:
            got, want = self.fs.identity(path), root["stat"]
            require(stat.S_ISDIR(got["mode"]) and all(got[k] == want[k] for k in ROOT_FIELDS),
                    "actual root device/inode/type changed")
            for parent in path.parents:
                require(not stat.S_ISLNK(self.fs.identity(parent)["mode"]), "root ancestor symlink refused")

    def _check_descendant(self, path, root, expected_device, *, may_not_exist=False):
        relative = path.relative_to(root)
        current = root
        parts = relative.parts
        for i, part in enumerate(parts):
            current = current / part
            if not self.fs.exists(current):
                require(may_not_exist, "missing declared source/new file; no fallback")
                continue
            got = self.fs.identity(current)
            require(got["dev"] == expected_device and not stat.S_ISLNK(got["mode"]),
                    "descendant device or symlink alias differs")
            require(stat.S_ISDIR(got["mode"]) if i < len(parts)-1 else stat.S_ISREG(got["mode"]),
                    "descendant special object or wrong type")

    def read_path(self, logical):
        """Old roles use their single pinned RO origin; new roles use fresh output."""
        rel = logical_relative(logical)
        self._check_roots()
        if logical in self.old:
            origin = self.old[logical]["source_origin"]
            root = self.sources[origin]
            path = root / rel
            self._check_descendant(path, root, self.source_roots[origin]["stat"]["dev"])
            observed = self.proof["source_files"][logical]
            before = self.fs.identity(path)
            require(before == observed["stat"], "source physical identity changed")
            token = tuple(before[k] for k in STAT_FIELDS)
            if self._verified.get(logical) != token:
                require(self.fs.hash(path) == self.old[logical]["sha256"], "source body SHA differs")
                require(self.fs.identity(path) == before, "source changed during body verification")
                self._verified[logical] = token
            return path
        require(logical in self.writes, "unlisted read role refused")
        path = self.output / rel
        self._check_descendant(path, self.output, self.proof["output_root"]["stat"]["dev"])
        return path

    def write_path(self, logical):
        """Returns a declared destination only; this method never opens/writes it."""
        rel = logical_relative(logical)
        self._check_roots()
        require(logical in self.writes and logical not in self.old,
                "writes to old prefix or unlisted destination refused")
        path = self.output / rel
        self._check_descendant(path, self.output, self.proof["output_root"]["stat"]["dev"], may_not_exist=True)
        if self.fs.exists(path):
            require(logical in self._created_new, "existing undeclared-ownership output must be preserved")
            current = self.fs.identity(path)
            require(all(current[k] == self._created_new[logical][k] for k in ("dev", "ino", "mode")),
                    "own new output identity changed before rewrite")
        else:
            self._issued_new.add(logical)
        return path

    def note_created_output(self, logical):
        """Records a caller-created new output; supplies no creation/admission proof."""
        require(logical in self._issued_new and logical in self.writes,
                "no own issued new output path exists")
        path = self.read_path(logical)
        self._created_new[logical] = self.fs.identity(path)

    def verify_old_recorded_reference(self, logical, recorded_sha256):
        require(logical in self.old and recorded_sha256 == self.old[logical]["sha256"],
                "old marker/receipt hash differs from frozen source role")
        return self.read_path(logical)

    def record(self):
        return {"schema": "chimera.k3.RO_multi_origin_resume_mapping_provenance.v1",
                "acting_identity": copy.deepcopy(self.acting_identity),
                "source_roots": copy.deepcopy(self.source_roots),
                "output_root": copy.deepcopy(self.proof["output_root"]),
                "prefix_through": PREFIX_END, "new_layer_range": [self.first, self.last],
                "closed48_continuity": copy.deepcopy(self.proof["closed48_continuity"]),
                "old_receipt_strings_rewritten": False, "runtime_admission_claimed": False,
                "verified_in_this_instance": sorted(self._verified)}


class NativeResumeReads:
    """Explicit native call-site candidates; writer methods must use write_path."""
    def __init__(self, resolver):
        self.paths = resolver

    def marker(self, layer):
        return self.paths.read_path(f"/run-output/fold/dense/L{layer:02d}/done.json")

    def ffn_reference(self, entry):
        return VerifierReads(self.paths).recorded(entry["path"], entry["sha256"])

    def sealed_metadata(self, basename):
        require(basename in {"k3_dense_pass_receipt.json", "k3_dense_pass_receipt.partial.json", "snapshots_manifest.json", "k3_dense_pass_receipt.sha256"},
                "explicit sealed native metadata role required")
        root = LOGICAL + "/" + self.paths.closed48_sealed_root + "/"
        choices = [logical for logical, row in self.paths.old.items()
                   if row["role"] == "sealed37_48_ancestry" and logical.startswith(root)
                   and PurePosixPath(logical).name == basename]
        require(len(choices) == 1, "exact unique latest closed48 sealed native metadata required")
        return self.paths.read_path(choices[0])

    def snapshot(self, layer, stream):
        require(stream in ("teacher", "folded"), "exact native snapshot stream required")
        suffix = "" if stream == "teacher" else "_folded"
        return self.paths.read_path(f"/run-output/fold/snapshots/before_L{layer:02d}{suffix}.safetensors")


class VerifierReads:
    def __init__(self, resolver):
        self.paths = resolver

    def recorded(self, logical, digest):
        if logical in self.paths.old:
            return self.paths.verify_old_recorded_reference(logical, digest)
        path = self.paths.read_path(logical)
        require(self.paths.fs.hash(path) == digest, "new emitted body SHA differs")
        return path


class ControllerReads:
    def __init__(self, resolver):
        self.paths = resolver

    def existing_layer(self, layer, role):
        require(type(layer) is int and 0 <= layer <= PREFIX_END, "exact closed-prefix layer required")
        require(role in ("ffn", "body", "held"), "explicit budget-source role required")
        if role == "held":
            require(37 <= layer <= 48, "only actual new37-48 held rows are budget inputs")
            logical = f"/run-output/fold/resid/L{layer:02d}/held_rows.safetensors"
        else:
            logical = (f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors" if role == "ffn"
                       else f"/run-output/body/L{layer:02d}/body.safetensors")
        return self.paths.read_path(logical)


class AncestryReads(VerifierReads):
    """Only an adapter candidate; old RecoveryAncestry policy stays unchanged."""
