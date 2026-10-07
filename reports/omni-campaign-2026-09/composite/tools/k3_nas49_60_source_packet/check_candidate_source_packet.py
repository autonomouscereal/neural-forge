"""Portable check of the reviewed source-only bundle; no native/Torch execution.

Candidate-root holds the seven exact flattened source basenames. Evidence-root
is the original recovery root, or a mirror with unique original proof basenames.
The original six test functions and data construction are kept unchanged; only
their module imports and two evidence paths are bound explicitly to supplied
roots. This adapter supplies no future48, resource or execution proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import unittest

ROSTER = ("recovery_resource_review/nas49_60_continuity_feasibility_v1/SOURCE_ONLY_49_60_PROPOSAL_ROSTER_V1.json",
          "61e9b20751f9d7d81bb217ca4d61df0b43646fcaa7b7428beea27b485b29cb8a")
AUDIT = ("nas49_60_independent_composition_review_v1/FINAL15_ROSTER_CURRENT_PIN_REVIEW_V1.json",
         "a491ea7b05a7287a3c2b3098909b6eec5107c9cd7ff1d0c32e2c97bdff25a90f")
INDEPENDENT = ("nas49_60_independent_composition_review_v1/INDEPENDENT_NAS49_60_COMPOSITION_SOURCE_REVIEW_V1.json",
               "734104e6aae8b2a4110cbcb5ff1fb6d20b74ba046126d174ba22aa33db503119")
TEST = ("nas49_60_proposal/test_path_seams.py", "817d14d0600a48ee51c2e28d2199ce517d9bf5e70371a1accebde55922fca5f0")
CATALOG = ("nas37_48_resume_review/ro_prefix_catalog_v2.json", "7a6b254fd7822bb2b868be149f2246b892868a7b2bbb0daa5f37db5ae1fe5cd2")
CLOSED36 = ("generation5_capacity_wait/actual_closed36_metadata_20261007T101629064667Z/window_continuity.json",
            "dd0ab8e821307e352096cdded457934dfd6df3434cfbe40d74e24b09a215870b")
QUALIFIED = ("nas37_48_resume_review/k3_dense_pass_9086.py", "9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69")
ANCESTRY = "88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41"
CANDIDATE_NAMES = {
    "native": "k3_dense_pass.py", "resolver": "ro_resume_paths.py", "runtime_guard": "ro_resume_runtime.py",
    "verifier": "verify_later_window.py", "controller_unbound": "continuation_worker49_60_template_v3.py",
    "qualification_unbound": "native_nas_io_qualification49_60_template_v2.py",
}


def require(ok, reason):
    if ok is not True:
        raise ValueError(reason)


def pinned(path, expected, size=None):
    require(path.is_file() and not path.is_symlink(), "regular supplied source/proof required: " + str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "full-byte SHA differs: " + str(path))
    require(size is None or len(raw) == size, "byte extent differs: " + str(path))
    return raw


def evidence_path(root, relative):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "bounded proof-relative path required")
    nested, flat = root.joinpath(*rel.parts), root / rel.name
    path = nested if nested.is_file() else flat
    require(path.resolve().is_relative_to(root.resolve()), "proof escaped supplied evidence root")
    return path


def recorded_relative(value):
    normalized = str(value).replace("\\", "/")
    require(normalized.count("/recovery_oct07/") == 1, "reviewed recovery-relative evidence path required")
    return normalized.split("/recovery_oct07/", 1)[1]


def json_pin(root, pin):
    path = evidence_path(root, pin[0])
    return path, json.loads(pinned(path, pin[1]))


def functions(raw):
    result = {}
    def collect(scope, prefix=""):
        for node in scope.body:
            if isinstance(node, ast.ClassDef):
                collect(node, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = ast.dump(node, include_attributes=False)
    collect(ast.parse(raw))
    return result


def exact_import(name, path, expected):
    # Execute these authenticated bytes, avoiding an implicit loader reread or
    # import of any same-named original proposal module from sys.path.
    raw = pinned(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None, "explicit candidate import spec required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(Path(module.__file__).resolve() == path.resolve(), "import used a different source location")
    return module


class TestEvidencePaths:
    """Only the two unchanged test data references; no arbitrary path resolver."""
    def __init__(self, paths):
        self.paths = paths

    def __truediv__(self, relative):
        require(relative in self.paths, "unexpected original test evidence path")
        return self.paths[relative]


def run_existing_six(test_path, raw, candidate_root, evidence, pins):
    names = ("k3_recovery_ancestry", "ro_resume_paths", "ro_resume_runtime")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        ancestry = exact_import(names[0], candidate_root / "k3_recovery_ancestry.py", ANCESTRY)
        paths = exact_import(names[1], candidate_root / "ro_resume_paths.py", pins["resolver"]["sha256"])
        runtime = exact_import(names[2], candidate_root / "ro_resume_runtime.py", pins["runtime_guard"]["sha256"])
        require(runtime.RecoveryAncestry is ancestry.RecoveryAncestry and runtime.ROResumePaths is paths.ROResumePaths,
                "runtime did not bind supplied candidate dependencies")
        tree = ast.parse(raw)
        declarations_before = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        namespace = {"__name__": "portable_candidate_source_seams", "__file__": str(test_path),
                     "HERE": candidate_root, "RECOVERY": TestEvidencePaths(evidence), "ro_resume_paths": paths}
        kept, removed = [], {"local_imports": 0, "path_assignments": 0, "sys_path_calls": 0, "main": 0}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in ("ro_resume_paths", "ro_resume_runtime"):
                module = paths if node.module == "ro_resume_paths" else runtime
                for alias in node.names:
                    namespace[alias.asname or alias.name] = getattr(module, alias.name)
                removed["local_imports"] += 1
            elif isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["ro_resume_paths"]:
                removed["local_imports"] += 1
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("HERE", "RECOVERY"):
                removed["path_assignments"] += 1
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and ast.unparse(node.value.func) in ("sys.path.insert", "sys.path.append"):
                removed["sys_path_calls"] += 1
            elif isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
                removed["main"] += 1
            else:
                kept.append(node)
        require(removed == {"local_imports": 3, "path_assignments": 2, "sys_path_calls": 2, "main": 1},
                "original six-test adapter seam changed")
        tree.body = kept
        require(declarations_before == {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))},
                "existing test function/class AST changed")
        exec(compile(tree, str(test_path), "exec"), namespace)
        require(namespace["ROResumePaths"] is paths.ROResumePaths and namespace["Closed48Ancestry"] is runtime.Closed48Ancestry,
                "tests would execute different original modules")
        suite = unittest.TestLoader().loadTestsFromTestCase(namespace["PathSeams"])
        require(suite.countTestCases() == 6, "exact existing six cases required")
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        require(result.wasSuccessful(), "existing candidate seam checks failed:\n" + stream.getvalue())
        return {"tests_run": result.testsRun, "successful": True, "output": stream.getvalue(),
                "explicit_imports": {name: str(candidate_root / (name + ".py")) for name in names},
                "test_function_and_class_ASTs_unchanged": True}
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def check(candidate_root, evidence_root):
    candidate_root, evidence_root = candidate_root.resolve(), evidence_root.resolve()
    require(candidate_root.is_dir() and evidence_root.is_dir(), "actual supplied candidate and evidence directories required")
    _, roster = json_pin(evidence_root, ROSTER)
    _, audit = json_pin(evidence_root, AUDIT)
    _, independent = json_pin(evidence_root, INDEPENDENT)
    require(roster.get("kind") == "SOURCE_ONLY_NOT_EXECUTION_SOURCE_CLOSURE" and len(roster["pins"]) == 15
            and audit.get("verdict") == "PASS_ALL15_CURRENT_ROSTER_PINS_SOURCE_ONLY"
            and independent.get("independent_AST_unchanged_qualified_count") == 99,
            "exact reviewed15 source-only roster and independent math review required")
    reviewed = {row["role"]: row for row in audit["checks"]}
    require(set(reviewed) == set(roster["pins"]) and all(row.get("passed") is True for row in reviewed.values()),
            "independent reviewed15 exact roles differ")
    physical, candidate_raw = {}, {}
    for role, pin in roster["pins"].items():
        require(pin["sha256"] == reviewed[role]["sha256"] and pin["bytes"] == reviewed[role]["bytes"],
                "roster role differs from final reviewed SHA/extent")
        path = candidate_root / CANDIDATE_NAMES[role] if role in CANDIDATE_NAMES else evidence_path(evidence_root, recorded_relative(pin["path"]))
        raw = pinned(path, pin["sha256"], pin["bytes"])
        if path.suffix == ".py":
            compile(raw, str(path), "exec")
        physical[role] = {"path": str(path), "sha256": pin["sha256"], "bytes": len(raw)}
        if role in CANDIDATE_NAMES:
            candidate_raw[role] = raw
    pinned(candidate_root / "k3_recovery_ancestry.py", ANCESTRY, 17493)
    compile((candidate_root / "k3_recovery_ancestry.py").read_bytes(), str(candidate_root / "k3_recovery_ancestry.py"), "exec")
    validation = json.loads(pinned(Path(physical["native_resolver_validation"]["path"]), roster["pins"]["native_resolver_validation"]["sha256"]))
    qualified_path = evidence_path(evidence_root, QUALIFIED[0])
    qualified = functions(pinned(qualified_path, QUALIFIED[1], 187275))
    candidate = functions(candidate_raw["native"])
    names = validation["original_qualified_functions_AST_unchanged"]
    require(len(names) == 99 and len(set(names)) == 99 and all(name in qualified and candidate.get(name) == qualified[name] for name in names),
            "all99 reviewed mathematical function ASTs must equal pinned9086")
    test_path = evidence_path(evidence_root, TEST[0]); test_raw = pinned(test_path, TEST[1], 11948)
    fixtures = {}
    for relative, digest in (CATALOG, CLOSED36):
        path = evidence_path(evidence_root, relative)
        pinned(path, digest)
        fixtures[relative] = path
    checks = run_existing_six(test_path, test_raw, candidate_root, fixtures, roster["pins"])
    return {"schema": "neural_forge.k3_NAS49_60_portable_source_packet_check.v1",
            "candidate_root": str(candidate_root), "evidence_root": str(evidence_root),
            "reviewed15_actual_pins": physical, "ancestry_source_sha256": ANCESTRY,
            "qualified_source_sha256": QUALIFIED[1], "qualified_math_AST_unchanged_count": 99,
            "existing_six_checks": checks, "compile_passed": True,
            "future_actual48_or_admission_claimed": False, "native_Torch_remote_resource_actions": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.candidate_root, args.evidence_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
