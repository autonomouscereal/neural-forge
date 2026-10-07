"""Portable check of the reviewed source-only bundle; no native/Torch execution.

Candidate-root holds the seven core and three capacity template basenames. Evidence-root
is the original recovery root, or a mirror with unique original proof basenames.
V4 keeps the ten-module closure and adopts only the reviewed current-peer preflight.
V3 compiles the complete ten-module source-only candidate closure.
V2 added the exact native resume I/O correction and unbound DriverV4.
The original six test functions and data construction are kept unchanged; only
their module imports and two evidence paths are bound explicitly to supplied
roots. The three authorized actual resume_point AST checks are also run.
This adapter supplies no future48, resource or execution proof.
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
    "verifier": "verify_later_window.py", "controller_unbound": "continuation_worker49_60_template_v4.py",
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


def check_core(candidate_root, evidence_root):
    candidate_root, evidence_root = candidate_root.resolve(), evidence_root.resolve()
    require(candidate_root.is_dir() and evidence_root.is_dir(), "actual supplied candidate and evidence directories required")
    _, roster = json_pin(evidence_root, ROSTER)
    _, audit = json_pin(evidence_root, AUDIT)
    _, independent = json_pin(evidence_root, INDEPENDENT)
    require(roster.get("kind") == "SOURCE_ONLY_NOT_EXECUTION_SOURCE_CLOSURE" and len(roster["pins"]) == 15
            and audit.get("verdict") == "PASS_ALL15_CURRENT_ROSTER_PINS_SOURCE_ONLY"
            and independent.get("independent_AST_unchanged_qualified_count") == 99,
            "exact historical reviewed15 source-only roster required")
    reviewed = {row["role"]: row for row in audit["checks"]}
    require(set(reviewed) == set(roster["pins"]) and all(row.get("passed") is True for row in reviewed.values()),
            "historical reviewed15 exact roles differ")
    native_validation_pin = ("nas49_60_proposal/native_boundary_v2/NATIVE_BOUNDARY_V2_SOURCE_VALIDATION.json",
                             "aee1c90f49a16f3cedcb90c9ca6ceabfffc792fccf03ab7496a76f26d2eaa5fd")
    driver_composition_pin = ("recovery_resource_review/nas49_60_continuity_feasibility_v1/admission_caller_proposal_v4/SOURCE_ONLY_49_60_EXACT_ADMISSION_CALLER_COMPOSITION_V4.json",
                              "8bd0ebb3678220b9b8ab6375dd4b2ed1efec53ac634ccc3ab89c4cdb94adb023")
    _, native_validation = json_pin(evidence_root, native_validation_pin)
    _, driver_composition = json_pin(evidence_root, driver_composition_pin)
    replacements = {
        "native": {"sha256": "5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007", "bytes": 190685},
        "controller_unbound": {"sha256": "c0881e4ac6b707dd0a34fccf9c1fd197a1bde20426f5060cdf5dc4d196168973", "bytes": 43555},
    }
    require(native_validation["candidate_source"]["sha256"] == replacements["native"]["sha256"]
            and native_validation["changed_existing_function_ASTs"] == ["DualStreamRun.resume_point"]
            and native_validation["qualified99_roster_already_excludes_resume_point"] is True
            and native_validation["deployable"] is False,
            "exact corrected native I/O source validation required")
    require(driver_composition["schema"] == "neural_forge.k3_source_only_49_60_exact_admission_caller_composition.v4"
            and driver_composition["new_source_templates"]["driver"]["sha256"] == replacements["controller_unbound"]["sha256"]
            and driver_composition["protocol_unchanged"]["boundary_keys"] == 16
            and driver_composition["deployable"] is False,
            "exact UNBOUND DriverV4 source composition required")
    historical, physical, candidate_raw, original_raw = {}, {}, {}, {}
    effective_pins = dict(roster["pins"])
    for role, pin in roster["pins"].items():
        require(pin["sha256"] == reviewed[role]["sha256"] and pin["bytes"] == reviewed[role]["bytes"],
                "historical roster differs from final historical review")
        path = evidence_path(evidence_root, recorded_relative(pin["path"])) if role in replacements or role not in CANDIDATE_NAMES else candidate_root / CANDIDATE_NAMES[role]
        old_raw = pinned(path, pin["sha256"], pin["bytes"])
        if path.suffix == ".py":
            compile(old_raw, str(path), "exec")
        historical[role] = {"path": str(path), "sha256": pin["sha256"], "bytes": len(old_raw)}
        if role in replacements:
            original_raw[role] = old_raw
            selected = replacements[role]
            path = candidate_root / CANDIDATE_NAMES[role]
            selected_raw = pinned(path, selected["sha256"], selected["bytes"])
            compile(selected_raw, str(path), "exec")
            effective_pins[role] = selected
        else:
            selected, selected_raw = pin, old_raw
        physical[role] = {"path": str(path), "sha256": selected["sha256"], "bytes": len(selected_raw)}
        if role in CANDIDATE_NAMES:
            candidate_raw[role] = selected_raw
    ancestry_raw = pinned(candidate_root / "k3_recovery_ancestry.py", ANCESTRY, 17493)
    compile(ancestry_raw, str(candidate_root / "k3_recovery_ancestry.py"), "exec")
    original_functions, current_functions = functions(original_raw["native"]), functions(candidate_raw["native"])
    require(set(original_functions) == set(current_functions)
            and [name for name in original_functions if original_functions[name] != current_functions[name]] == ["DualStreamRun.resume_point"],
            "native must change only the resume_point I/O guard")
    old_driver, new_driver = functions(original_raw["controller_unbound"]), functions(candidate_raw["controller_unbound"])
    require(set(old_driver) == set(new_driver)
            and [name for name in old_driver if old_driver[name] != new_driver[name]] == ["Driver.admit"],
            "DriverV4 must change only admission projection, preserving dispatcher/protocol")
    validation = json.loads(pinned(Path(historical["native_resolver_validation"]["path"]), roster["pins"]["native_resolver_validation"]["sha256"]))
    qualified_path = evidence_path(evidence_root, QUALIFIED[0])
    qualified = functions(pinned(qualified_path, QUALIFIED[1], 187275))
    names = validation["original_qualified_functions_AST_unchanged"]
    require(len(names) == 99 and len(set(names)) == 99 and "DualStreamRun.resume_point" not in names
            and all(name in qualified and current_functions.get(name) == qualified[name] for name in names),
            "all99 actual reviewed other qualified ASTs must equal pinned9086")
    test_path = evidence_path(evidence_root, TEST[0]); test_raw = pinned(test_path, TEST[1], 11948)
    fixtures = {}
    for relative, digest in (CATALOG, CLOSED36):
        path = evidence_path(evidence_root, relative)
        pinned(path, digest)
        fixtures[relative] = path
    checks = run_existing_six(test_path, test_raw, candidate_root, fixtures, effective_pins)
    boundary_test_pin = ("nas49_60_proposal/native_boundary_v2/test_resume_boundary_v2.py",
                         "b98bb493f4a75fa6d55a23583af033dcebe0ffc4926576438d2df658fa3ab13e")
    boundary_test_path = evidence_path(evidence_root, boundary_test_pin[0])
    boundary_raw = pinned(boundary_test_path, boundary_test_pin[1], 2648)
    namespace = {"__name__": "portable_actual_resume_point_tests", "__file__": str(boundary_test_path)}
    exec(compile(boundary_raw, str(boundary_test_path), "exec"), namespace)
    boundary_checks = namespace["run_actual_method_checks"](candidate_raw["native"])
    require(boundary_checks["tests_run"] == 3 and boundary_checks["successful"] is True,
            "exact three actual-candidate resume I/O checks required")
    return {"schema": "neural_forge.k3_NAS49_60_portable_source_packet_check.v2",
            "candidate_root": str(candidate_root), "evidence_root": str(evidence_root),
            "historical_reviewed15_actual_pins": historical, "current_candidate_actual_pins": physical,
            "exact_replacement_source_proofs": {"native": native_validation_pin, "driver": driver_composition_pin},
            "ancestry_source_sha256": ANCESTRY, "qualified_source_sha256": QUALIFIED[1],
            "qualified_other_AST_unchanged_count": 99, "resume_point_is_excluded_from_actual99_roster": True,
            "native_only_changed_function": "DualStreamRun.resume_point", "driver_only_changed_function": "Driver.admit",
            "existing_six_checks": checks, "actual_resume_point_three_checks": boundary_checks,
            "compile_passed": True, "unbound_templates": True,
            "future_actual48_or_admission_claimed": False, "native_Torch_remote_resource_actions": False}


def check(candidate_root, evidence_root):
    candidate_root, evidence_root = candidate_root.resolve(), evidence_root.resolve()
    closure_pin = ("recovery_resource_review/nas49_60_continuity_feasibility_v1/admission_caller_proposal_v4/SOURCE_ONLY_49_60_EXACT_ADMISSION_CALLER_COMPOSITION_V4.json",
                   "8bd0ebb3678220b9b8ab6375dd4b2ed1efec53ac634ccc3ab89c4cdb94adb023")
    independent_pin = ("nas49_60_independent_composition_review_v1/FINAL_CALLER_V4_NATIVE_BOUNDARY_V2_INDEPENDENT_REVIEW.json",
                       "7b55298ccf462d966fd95e14fbc3e29d41b7188f54d435b74332ae2739b00f87")
    _, closure = json_pin(evidence_root, closure_pin)
    _, review = json_pin(evidence_root, independent_pin)
    require(closure["schema"] == "neural_forge.k3_source_only_49_60_exact_admission_caller_composition.v4"
            and closure["deployable"] is False
            and review["verdict"] == "PASS_CORRECTED_SOURCE_COMPOSITION_ONLY_UNBOUND"
            and all(row["passed"] is True for row in review["source_checks"])
            and review["selected_composition"]["sha256"] == closure_pin[1]
            and closure["genuine_new_native_source_candidate"]["sha256"] == "5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007"
            and review["selected_native"]["sha256"] == closure["genuine_new_native_source_candidate"]["sha256"],
            "whole current source closure and exact independent source-only review required")
    expected = {
        "facts": ("k3_owned_readonly_facts49_60_template_v2.py", "38e27e17c59bc1a6c5b1ed9dd55e3e7923330961534af54bed6ce739df01a0dd", 11667),
        "preflight": ("k3_owned_capacity_preflight49_60_template_v2.py", "c3c6359b79342824af5b77d5073e4dabde720501de8648de587defbecb2f102e", 113469),
        "host_caller": ("k3_capacity_host_loop49_60_template_v2.py", "089c84de8866d297038733c43ddf180192d339e9b6c73fad9df974a3a0101ada", 17687),
    }
    capacity_sources = {}
    for role, (name, digest, extent) in expected.items():
        selected = closure["new_source_templates"][role]
        historical_expected = (name, "89ba302e7969680c9b51f394726b51eff99959b2c669398ad831630aaf838177", 100924) if role == "preflight" else (name, digest, extent)
        require((selected["name"], selected["sha256"], selected["bytes"]) == historical_expected,
                "capacity source template differs from preserved exact parent closure")
        path = candidate_root / name
        source = pinned(path, digest, extent)
        compile(source, str(path), "exec")
        capacity_sources[role] = {"path": str(path), "sha256": digest, "bytes": extent}
    result = check_core(candidate_root, evidence_root)
    peer_relative = "nas49_60_proposal/current_peer_admission_v1/corrected_v3/"
    _, peer = json_pin(evidence_root, (peer_relative+"CURRENT_PEER_SOURCE_COMPOSITION_V3.json",
        "83a92e035459fe4a9be76f9238faa86c890c2518a2f7d35c34ec8150ac133cd2"))
    require(peer["schema"] == "neural_forge.k3_next49_current_peer_source_composition.v3"
            and peer["source_only"] is True and peer["selected_for_runtime"] is False
            and peer["source_module_current_peer_role"] == "current_peer_holder_source"
            and peer["outputs"]["preflight"]["sha256"] == capacity_sources["preflight"]["sha256"]
            and peer["outputs"]["preflight"]["bytes"] == capacity_sources["preflight"]["bytes"],
            "exact isolated current-peer source composition required")
    proof_raw = pinned(evidence_path(evidence_root, peer_relative+"CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json"),
        "1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9", 1797866)
    current_raw = pinned(candidate_root/expected["preflight"][0], expected["preflight"][1], expected["preflight"][2])
    prior_raw = pinned(candidate_root/"current_peer_evidence/k3_owned_capacity_preflight49_60_prior89.py",
        "89ba302e7969680c9b51f394726b51eff99959b2c669398ad831630aaf838177", 100924)
    prior_functions, new_functions = functions(prior_raw), functions(current_raw)
    require(set(prior_functions) <= set(new_functions)
            and [name for name in prior_functions if prior_functions[name] != new_functions[name]] == ["classify", "run"]
            and set(new_functions)-set(prior_functions) == {"current_peer_public_source", "current_qimg_native_holder"},
            "only exact current-peer source/classification seam may change")
    result["current_peer_actual_metadata_three_checks"] = run_current_peer_three(current_raw, proof_raw)
    result["current_peer_source_composition"] = (peer_relative+"CURRENT_PEER_SOURCE_COMPOSITION_V3.json",
        "83a92e035459fe4a9be76f9238faa86c890c2518a2f7d35c34ec8150ac133cd2")
    result["current_peer_metadata_role"] = {"role": "current_peer_holder_source",
        "name": "CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json",
        "sha256": "1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9", "bytes": 1797866}
    result["schema"] = "neural_forge.k3_NAS49_60_portable_source_packet_check.v4"
    result["capacity_candidate_sources_actual_pins"] = capacity_sources
    result["whole_selected_source_closure"] = closure_pin
    result["whole_selected_independent_source_review"] = independent_pin
    result["candidate_production_template_module_count"] = 10
    result["existing_test_cases_unchanged_total"] = 9
    result["current_peer_existing_actual_metadata_cases"] = 3
    result["total_source_checks"] = 12
    return result




def run_current_peer_three(source, proof_raw):
    """The existing three authentic-metadata cases; no physical holder fixture."""
    import copy
    packet = json.loads(proof_raw)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "current_peer_public_source")
    def strict(ok, message):
        if ok is not True: raise RuntimeError(message)
    env = {"require": strict, "json": json, "hashlib": hashlib, "GIB": 1 << 30,
           "CURRENT_PEER_SOURCE_SHA": "1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9",
           "QIMG_OWNER": "codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c",
           "ROOT_OWNER": "codex-native-01a10f94-259b-7623-9679-ce4e68f82b8c"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<selected authentic metadata join>", "exec"), env)
    def request(): return {"sources": {"current_peer_holder_source": {"sha256": env["CURRENT_PEER_SOURCE_SHA"]}}}
    req = request(); joined = env["current_peer_public_source"]({"current_peer_holder_source": packet}, req)
    require(len(packet["members"]) == 19
            and joined["allocation"]["container_id"] == "7fb0e676b730d2dded9294d5318a18ba347e67c7d096a759a65dbc5946fd610f"
            and joined["inspect"]["HostConfig"]["Memory"] == joined["inspect"]["HostConfig"]["NanoCpus"] == 0
            and req["ROOT1075_exact_completed_phase_future_once"]["all_ROOT_futures_discharged"] is False,
            "authentic19 raw source/release join and honest original uncapped barrier required")
    checks = [{"case": "authentic19_fullraw_source_terminal_release_join", "pass": True}]
    bad = copy.deepcopy(packet); bad["members"]["qimg_worker"]["raw_utf8"] += " "
    try: env["current_peer_public_source"]({"current_peer_holder_source": bad}, request())
    except RuntimeError as exc: checks.append({"case": "changed_actual_worker_bytes_refuse", "pass": True, "reason": str(exc)})
    else: raise ValueError("changed actual worker bytes accepted")
    bad = request(); bad["sources"]["current_peer_holder_source"]["sha256"] = "0" * 64
    try: env["current_peer_public_source"]({"current_peer_holder_source": packet}, bad)
    except RuntimeError as exc: checks.append({"case": "wrong_whole_public_proof_pin_refuses", "pass": True, "reason": str(exc)})
    else: raise ValueError("wrong public proof pin accepted")
    return {"successful": True, "tests_run": 3, "checks": checks,
            "actual_current_QIMG_cap_claimed": False, "fresh_physical_predicate_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.candidate_root, args.evidence_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
