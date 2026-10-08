"""Add only the authenticated pending ROOT text write to reviewed later callers.

Source composition and bounded compile/pin/AST checks only. No candidate imports,
fixtures, remote connections, runtime controls or model operations.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent
OLD = RECOVERY / "later73_85_control_integration_v1"
OLD_SHARED = RECOVERY / "nas61_launch_integration_v1/current_ROOT_peer_scope_v1"
OLD_COMPOSITION_SHA = "4dc2ccd66c202a64512d44158a6c2075e8e8343ad480154d9d4d2b21287f650d"
OLD_PROOF_SHA = "99a1dec0fddc46a6478cf8bb289856dfe1804838a973b35f1bc3545e4b474201"
OLD_PREFLIGHT_SHA = "0b402fa5b60ccc75a2e5df21daee49925b85b4ab83cffd9a4a38883203510e19"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw)}


def read_pin(row):
    path = Path(row["path"])
    assert pin(path) == row
    return path.read_bytes()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.read_bytes() == raw


def one(text, old, new):
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)


def assign(text, name, expression):
    found = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == name]
    assert len(found) == 1, name
    return one(text, ast.get_source_segment(text, found[0]), name + "=" + expression)


def exact_delta(target, old, new):
    a, b = old.splitlines(True), new.splitlines(True)
    for kind, i, j, k, l in reversed(difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()):
        if kind == "equal":
            continue
        before, after = "".join(a[i:j]), "".join(b[k:l])
        if before:
            target = one(target, before, after)
        else:
            anchor = "".join(a[max(0, i - 5):i])
            assert anchor
            target = one(target, anchor, anchor + after)
    return target


def functions(raw):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-contract", type=Path, required=True)
    parser.add_argument("--shared-contract-sha256", required=True)
    args = parser.parse_args()
    assert pin(args.shared_contract)["sha256"] == args.shared_contract_sha256
    shared = json.loads(args.shared_contract.read_bytes())
    assert shared["source_only"] is True
    assert shared["metadata_count_before_allocation"] == 67 and shared["metadata_count_after_allocation"] == 68
    assert shared["source_role"] == "ROOT_current_serial_scope"
    assert shared["source_inactive_future_replacement_once"]["new_ROOT_write_bytes"] == 10 * (1 << 30)
    assert shared["conservative_whole_ROOT_future_bytes_once"] == 49 * (1 << 30)
    assert shared["additive_current_text_delta"]["speech_ROOT20GiB_preserved"] is True
    assert shared["new_phase_current_actor_adoption"] is False
    old_pre = (OLD_SHARED / "k3_owned_capacity_preflight61_72_ROOT_serial_v1.py").read_bytes()
    assert sha(old_pre) == OLD_PREFLIGHT_SHA
    new_pre = read_pin(shared["preflight"])
    proof = read_pin(shared["proof"])
    read_pin(shared["binder"])
    old_composition = (OLD / "LATER_NORMAL_CONTROL_COMPOSITION_V9.json").read_bytes()
    assert sha(old_composition) == OLD_COMPOSITION_SHA
    basis = json.loads(old_composition)
    changed_shared_functions = {name for name in functions(new_pre)
                                if functions(new_pre)[name] != functions(old_pre).get(name)}
    assert "run" in changed_shared_functions
    records, observations, previous_binder = [], [], shared["binder"]
    for phase in basis["phases"]:
        lo, hi = phase["window"]
        label = f"{lo}_{hi}"
        old_leaf = OLD / f"phase{label}_candidate_v9"
        leaf = HERE / f"phase{label}_ROOT_text_v1"
        assert not leaf.exists()
        leaf.mkdir()
        selected = json.loads(read_pin(phase["selected_source_roster"]))
        original = copy.deepcopy(selected)
        pre = exact_delta(read_pin(selected["preflight"]).decode(), old_pre.decode(), new_pre.decode())
        compile(pre, "later text preflight", "exec")
        old_functions = functions(read_pin(selected["preflight"]))
        new_functions = functions(pre)
        assert set(old_functions) == set(new_functions)
        assert all(new_functions[name] == old_functions[name]
                   for name in old_functions if name not in changed_shared_functions)
        assert all(new_functions[name] == functions(new_pre)[name]
                   for name in changed_shared_functions if name != "run")
        prepath = leaf / f"k3_owned_capacity_preflight{label}_ROOT_text_v1.py"
        write(prepath, pre.encode())
        selected["preflight"] = pin(prepath)
        for role in original:
            assert role == "preflight" or selected[role] == original[role]
            compile(read_pin(selected[role]), selected[role]["path"], "exec")
        roster_path = leaf / "SELECTED_PHASE_ROOT_TEXT_SOURCE_PINS_V1.json"
        write(roster_path, encode(selected))
        write(leaf / shared["source_filename"], proof)
        for name in [f"validate_actual{lo-1}_source_metadata_v2.py", f"ACTUAL{lo}_INPUTS_UNRESOLVED_V2.json"]:
            write(leaf / name, (old_leaf / name).read_bytes())
        controls = {}
        for name, row in phase["controls"].items():
            target = leaf / "controls" / name
            write(target, read_pin(row))
            controls[name] = pin(target)
            assert controls[name]["sha256"] == row["sha256"]
        source = read_pin(phase["binder"]).decode()
        local_output = f"D:/codex-owned-source-receipts/k3-later73-92-ROOT-text-source-only/phase{label}"
        source = assign(source, "LOCAL_OUTPUTS", "Path(" + repr(local_output) + ")")
        source = assign(source, "SOURCE_PINS_SHA", repr(sha(roster_path.read_bytes())))
        source = assign(source, "PREVIOUS_BINDER", "Path(" + repr(previous_binder["path"]) + ")")
        source = assign(source, "PREVIOUS_BINDER_SHA", repr(previous_binder["sha256"]))
        source = source.replace("SELECTED_PHASE_SOURCE_PINS_V9.json", roster_path.name)
        assert source.count(OLD_PROOF_SHA) == 2
        source = source.replace(OLD_PROOF_SHA, shared["proof"]["sha256"])
        assert source.count(OLD_PREFLIGHT_SHA) == 1
        source = source.replace(OLD_PREFLIGHT_SHA, shared["preflight"]["sha256"])
        source = source.replace("ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_PUBLIC_PROOF_V1.json", shared["source_filename"])
        source = one(source, "serial['bytes']==53603", "serial['bytes']==" + str(shared["proof"]["bytes"]))
        source = one(source, "'current_ROOT_serial_future_write_bytes_once':9*GIB", "'current_ROOT_serial_future_write_bytes_once':10*GIB")
        source = one(source, "serial ROOT16/4/9 source role", "serial ROOT16/4/10 source role")
        source = one(source, "Selected eleven source templates are normal-tested D commit6700 and immutable\nexact promoted phase component packet;", "Selected sources use the full current pinned roster and authenticated promoted phase component packet;\n")
        compile(source, "later text binder", "exec")
        binder_path = leaf / f"bind_actual{label}_ROOT_text_v1.py"
        write(binder_path, source.encode())
        previous_binder = pin(binder_path)
        helpers, replacements = {}, {}
        for name, row in phase["helpers"].items():
            helper = read_pin(row).decode()
            newname = name.replace("ROOT_serial_v9", "ROOT_text_v1")
            assert newname != name
            replacements[name] = newname
            if name.startswith("read_fresh_prospective"):
                helper = one(helper, "['new_ROOT_write_bytes']==9*(1<<30)", "['new_ROOT_write_bytes']==10*(1<<30)")
                helper = one(helper, "ROOT serial16/4/9 future", "ROOT serial16/4/10 future")
            else:
                assert helper.encode() == read_pin(row)
            assert functions(helper) == functions(read_pin(row))
            compile(helper, newname, "exec")
            path = leaf / "launch" / newname
            write(path, helper.encode())
            helpers[newname] = pin(path)
        helper_manifest = leaf / "launch/LAUNCH_ROOT_TEXT_HELPER_SOURCE_PINS_V1.json"
        write(helper_manifest, encode({name: {key: row[key] for key in ("sha256", "bytes")}
                                       for name, row in helpers.items()}))
        wrapper = read_pin(phase["wrapper"]).decode()
        wrapper = assign(wrapper, "LOCAL_OUTPUTS", "pathlib.Path(" + repr(local_output) + ")")
        wrapper = assign(wrapper, "BINDER", "BIND_ROOT/" + repr(binder_path.name))
        wrapper = assign(wrapper, "BINDER_SHA", repr(previous_binder["sha256"]))
        wrapper = assign(wrapper, "HELPER_MANIFEST_SHA", repr(sha(helper_manifest.read_bytes())))
        wrapper = wrapper.replace("LAUNCH_HELPER_SOURCE_PINS_V9.json", helper_manifest.name)
        for oldname, newname in replacements.items():
            wrapper = wrapper.replace(oldname, newname)
        compile(wrapper, "later text wrapper", "exec")
        wrapper_path = leaf / "launch" / f"launch_actual{label}_ROOT_text_v1.py"
        write(wrapper_path, wrapper.encode())
        assert ast.literal_eval(next(n.value.args[0] for n in ast.parse(source).body
                                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LOCAL_OUTPUTS" for t in n.targets))) == local_output
        assert ast.literal_eval(next(n.value.args[0] for n in ast.parse(wrapper).body
                                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LOCAL_OUTPUTS" for t in n.targets))) == local_output
        record = copy.deepcopy(phase)
        record.update(binder=previous_binder, wrapper=pin(wrapper_path), controls=controls,
                      helpers=helpers, helper_manifest=pin(helper_manifest), selected_source_roster=pin(roster_path),
                      source_filename=shared["source_filename"], source_role="ROOT_current_serial_scope", source_proof=pin(leaf / shared["source_filename"]),
                      ROOT_pending_text_delta_bytes=1 << 30, ROOT_serial_future_write_bytes_once=10 * (1 << 30),
                      current_ROOT_actor_or_runtime_adopted=False, whole_ROOT_future_bytes_once=49 * (1 << 30),
                      independent_review=None)
        write(leaf / "ROOT_TEXT_PHASE_SOURCE_CONTRACT_V1.json", encode(record))
        records.append(record)
        observations.append({"window": [lo, hi], "ten_other_selected_role_bytes_unchanged": True,
                             "six_controls_byte_identical": True, "three_helpers_byte_identical": True,
                             "readback_only_ROOT_write_guard_9_to10": True,
                             "binder_wrapper_LOCAL_OUTPUTS_equal": local_output,
                             "native_math_and_phase_writer_extent_unchanged": True,
                             "metadata_count_before_A": record["metadata_before_A"], "metadata_count_with_A": record["metadata_with_A"],
                             "actual_previous_closure_UUID_FS_cap_allocation_A_admission": None})
    result = copy.deepcopy(basis)
    result.update(schema="neural_forge.k3_later73_85_ROOT_text_source_composition.v1", phases=records,
                  ROOT_serial_basis_contract=pin(args.shared_contract), source_pending_text_delta_bytes=1 << 30,
                  future_ROOT_serial_scope={"RAM_bytes_once": 16 * (1 << 30), "CPU_threads_once": 4,
                                            "ROOT_write_bytes_once": 10 * (1 << 30), "actual_actor_adopted": False},
                  whole_ROOT_future_bytes_once=49 * (1 << 30), speech_ROOT20GiB_retained=True,
                  inclusive_ROOT47GiB_not_added_as_new_promise=True, independent_review=None,
                  prior_normal_tested_commit="a20eaad121ecb529b3521aed6fbe34eb008d4234")
    write(HERE / "LATER_ROOT_TEXT_SOURCE_COMPOSITION_V1.json", encode(result))
    write(HERE / "LATER_ROOT_TEXT_FOCUSED_SOURCE_CHECK_V1.json", encode({
        "schema": "neural_forge.k3_later_ROOT_text_focused_source_check.v1",
        "verdict": "PASS_SOURCE_ONLY_PENDING_INDEPENDENT_REVIEW", "shared_contract": pin(args.shared_contract),
        "composition": pin(HERE / "LATER_ROOT_TEXT_SOURCE_COMPOSITION_V1.json"),
        "shared_changed_functions": sorted(changed_shared_functions), "observations": observations,
        "checks": ["full selected SHA and byte checks", "compile-only of all selected source", "only exact shared preflight delta",
                   "unchanged preflight functions outside shared delta", "shared pure functions AST equality", "new predecessor binder full pins",
                   "metadata counts unchanged", "helper functions unchanged", "local output path join", "fresh readback write10GiB"],
        "runtime_imports_or_entrypoints": False, "fixtures_executed": False, "remote_Relay_provider_allocator_model_actions": False}))
    print(json.dumps({"composition": pin(HERE / "LATER_ROOT_TEXT_SOURCE_COMPOSITION_V1.json"),
                      "focused_check": pin(HERE / "LATER_ROOT_TEXT_FOCUSED_SOURCE_CHECK_V1.json"),
                      "phases": [{key: row[key] for key in ("window", "binder", "wrapper", "selected_source_roster")} for row in records]}))


if __name__ == "__main__":
    main()
