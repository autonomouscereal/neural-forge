"""Synthetic receipt/runner mechanics; no source-image or numerical proof."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_broad_capture as tool


def dump(path, body):
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return tool.reference(path)


def dataset(root):
    batches = []
    sources = {role: {"path": "/original/" + role, "file_bytes": 100, "header_bytes": 10,
        "header_sha256": tool.capture.PARENT_HEADER_SHA256 if role == "diffusion" else str(i) * 64,
        "full_file_sha256": None, "scope": "header only; tensor bodies not read"}
        for i, role in enumerate(("diffusion", "vae", "text_encoder"), 1)}
    for number in range(1, 5):
        folder = root / str(number)
        folder.mkdir()
        plan = tool.parent.read_plan(tool.TOOLS / f"qimg_parent_broad_sep07_batch{number}.json")
        plan_ref = dump(folder / "plan.json", plan)
        images = []
        for index, entry in enumerate(plan["entries"]):
            image = folder / (entry["id"] + ".png")
            Image.new("RGB", (512, 512), (number * 40, index * 25, number + index)).save(image)
            images.append({**entry, "file": image.name, "sha256": tool.capture.file_sha(image),
                "decoded_pixels_sha256": "1" * 64, "native_sample_latent_sha256": "2" * 64,
                "provenance": "native Qwen-Image parent generated; no selection; no edits; not a photograph",
                "visual_review": "pending"})
        receipt = {"schema": "neural_forge.qimg_parent_images.v1", "status": "complete",
            "plan_sha256": plan_ref["sha256"], "sampling": plan["sampling"], "sources": sources,
            "source_code_sha256": {**tool.capture.SOURCE_FILES, **tool.prepare.SOURCE_FILES, **tool.parent.SAMPLING_SOURCE},
            "runtime": {"parent_class": "comfy.model_base.QwenImage", "manual_cast_dtype": "torch.bfloat16",
                        "fp8_optimizations": False, "full_residency_verified": True, "dynamic_patcher": False},
            "images": images}
        batches.append({"number": number, "generation_plan": plan_ref,
                        "parent_receipt": dump(folder / "parent_receipt.json", receipt)})
    excluded = {"schema": tool.capture.CAPTURE_SCHEMA, "status": "complete", "fixture": False,
                "captures": [{"image_sha256": f"{i:064x}"} for i in range(8)]}
    return batches, [dump(root / "excluded.json", excluded)], sources


@pytest.fixture
def data(tmp_path):
    return dataset(tmp_path)


def test_seal_exact32_images_128forwards_and_deterministic_explicit_seeds(tmp_path, data):
    batches, excluded, _ = data
    ref = tool.seal(batches, excluded, tmp_path / "sealed")
    bundle, plans = tool.validate_bundle(ref["path"], ref["sha256"])
    assert bundle["expected"]["forwards"] == 128
    assert bundle["expected"]["image_rows"] == {"fit": 98304, "holdout": 32768}
    assert len(plans) == 4 and all(len(p["entries"]) == 8 and p["sigmas"] == tool.SIGMAS for p in plans)
    seeds, identities = set(), set()
    for batch, plan in zip(batches, plans):
        source = tool.json_pinned(**dict(path=batch["generation_plan"]["path"], digest=batch["generation_plan"]["sha256"]))
        for expected, actual in zip(source["entries"], plan["entries"]):
            assert all(actual[key] == expected[key] for key in ("id", "split", "prompt"))
            assert actual["seed"] == tool.calibration_seed(batch["generation_plan"]["sha256"], expected)
            assert actual["image_provenance"]["generation_seed"] == expected["seed"]
            seeds.add(actual["seed"]); identities.add(actual["image"]["sha256"])
    assert len(seeds) == len(identities) == 32
    assert len(list((tmp_path / "sealed").iterdir())) == 5
    # The sealed output is accepted by the unchanged production preparation API.
    prepared = tool.prepare.read_plan(Path(bundle["encoding_plans"][0]["path"]))
    assert len(prepared["entries"]) == 8 and prepared["sigmas"] == tool.SIGMAS
    with pytest.raises(ValueError, match="no overwrite"):
        tool.seal(batches, excluded, tmp_path / "sealed")


@pytest.mark.parametrize("mutation", ["partial", "incomplete", "prompt", "split", "seed", "order", "old", "duplicate_png",
    "changed_png", "source", "source_code", "cast", "provenance", "traversal", "wrong_plan", "empty_exclusions", "missing_batch"])
def test_bridge_refuses_changed_or_incomplete_generation(tmp_path, data, mutation):
    batches, excluded, _ = data
    ref = batches[0]["parent_receipt"]
    body = tool.json_pinned(ref["path"], ref["sha256"])
    if mutation == "partial": body["images"].pop()
    if mutation == "incomplete": body["status"] = "incomplete"
    if mutation == "prompt": body["images"][0]["prompt"] += " changed"
    if mutation == "split": body["images"][0]["split"] = "holdout"
    if mutation == "seed": body["images"][0]["seed"] += 1
    if mutation == "order": body["images"].reverse()
    if mutation == "old":
        old = tool.json_pinned(excluded[0]["path"], excluded[0]["sha256"])
        old["captures"][0]["image_sha256"] = body["images"][0]["sha256"]
        excluded[0] = dump(Path(excluded[0]["path"]), old)
    if mutation == "duplicate_png":
        first, second = body["images"][:2]
        folder = Path(ref["path"]).parent
        (folder / second["file"]).write_bytes((folder / first["file"]).read_bytes())
        second["sha256"] = first["sha256"]
    if mutation == "changed_png": (Path(ref["path"]).parent / body["images"][0]["file"]).write_bytes(b"changed")
    if mutation == "source": body["sources"]["vae"]["header_sha256"] = "f" * 64
    if mutation == "source_code": body["source_code_sha256"] = {}
    if mutation == "cast": body["runtime"]["fp8_optimizations"] = True
    if mutation == "provenance": body["images"][0]["provenance"] = "photograph"
    if mutation == "traversal": body["images"][0]["file"] = "../other.png"
    if mutation == "wrong_plan": body["plan_sha256"] = "0" * 64
    if mutation == "empty_exclusions": excluded = []
    if mutation == "missing_batch": batches.pop()
    batches[0]["parent_receipt"] = dump(Path(ref["path"]), body)
    with pytest.raises(ValueError): tool.seal(batches, excluded, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_seal_hash_change_or_prompt_edit_cannot_be_hidden_by_rehash(tmp_path, data):
    batches, excluded, _ = data
    ref = tool.seal(batches, excluded, tmp_path / "sealed")
    bundle = tool.json_pinned(ref["path"], ref["sha256"])
    p = bundle["encoding_plans"][0]
    plan = tool.json_pinned(p["path"], p["sha256"])
    plan["entries"][0]["prompt"] += " change"
    bundle["encoding_plans"][0] = dump(Path(p["path"]), plan)
    ref = dump(Path(ref["path"]), bundle)
    with pytest.raises(ValueError, match="exact source receipts"):
        tool.validate_bundle(ref["path"], ref["sha256"])


def test_header_mount_remap_is_not_misrepresented_as_hash_change(data):
    _, _, sources = data
    changed = {**sources["vae"], "path": "/newmount/vae"}
    assert tool.source_content(changed) == tool.source_content(sources["vae"])
    changed["full_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="truthful header-only"):
        tool.source_content(changed)


def test_encoding_audit_counts_actual_valid_rows_and_exact_fp32_sigma(tmp_path, data, monkeypatch):
    batches, excluded, sources = data
    plans, _, _ = tool.derive_plans(batches, excluded)
    plan = plans[0]
    plan_ref = tool.exclusive_json(tmp_path / "encoding_plan.json", plan)
    out = tmp_path / "encoded"
    out.mkdir()
    entries, linked_entries, details, linkage = [], [], {}, {}
    valid_total = 0
    for i, entry in enumerate(plan["entries"]):
        valid = torch.ones((1, 90), dtype=torch.bool)
        valid[:, -i-1:] = False
        state = {"model_inputs": torch.zeros((4, 16, 1, 64, 64), dtype=torch.float32),
            "sigma": torch.tensor(tool.SIGMAS, dtype=torch.float32), "context": torch.zeros((1, 90, 3584), dtype=torch.float32),
            "context_mask": valid, "context_token_ids": torch.arange(90, dtype=torch.int64)[None]}
        record = tool.capture.write_tensors(out / (entry["id"] + ".safetensors"), state)
        actual = {k: entry[k] for k in ("id", "split", "prompt", "image")}
        actual.update(prompt_sha256=tool.hashlib.sha256(entry["prompt"].encode()).hexdigest(),
                      tensors={"path": record["file"], "sha256": record["file_sha256"]})
        linked_entries.append(actual)
        entries.append({"entry": actual})
        linkage[entry["id"]] = {"image_sha256": actual["image"]["sha256"], "prompt_sha256": actual["prompt_sha256"],
                                "tensor_file_sha256": record["file_sha256"]}
        details[entry["id"]] = {"image_provenance": entry["image_provenance"], "tensors": record,
                               "noise": {"seed": entry["seed"], "sigma_shift_applied": False, "noise_scale": 1.0}}
        if entry["split"] == "fit": valid_total += 4 * int(valid.sum())
    receipt = {"schema": "neural_forge.qimg_encoding_receipt.v1", "plan_sha256": plan_ref["sha256"],
               "source_code_sha256": tool.prepare.SOURCE_FILES, "vae_loaded_complete_key_shape_coverage": True,
               "entries": linkage, "details": details}
    receipt_ref = dump(out / "encoding_receipt.json", receipt)
    linked = {"manifest": str(out / "capture_inputs.json"), "encoding_receipt_sha256": receipt_ref["sha256"],
              "forwards": 32, "sources": {k: sources[k] for k in ("vae", "text_encoder")}, "entries": linked_entries}
    monkeypatch.setattr(tool.capture, "load_inputs", lambda *a, **k: (entries, linked))
    result = tool.audit_encoding(out, plan, plan_ref, sources)
    assert result["row_counts"]["txt"]["fit"] == valid_total
    assert result["row_counts"]["img"] == {"fit": 24576, "holdout": 8192}
    # Default float64 cannot alter identity checks or the stored F32 sigma grid.
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert tool.audit_encoding(out, plan, plan_ref, sources)["row_counts"] == result["row_counts"]
    finally: torch.set_default_dtype(old)
    receipt["details"][plan["entries"][0]["id"]]["noise"]["seed"] += 1
    linked["encoding_receipt_sha256"] = dump(out / "encoding_receipt.json", receipt)["sha256"]
    with pytest.raises(ValueError, match="noise seed"):
        tool.audit_encoding(out, plan, plan_ref, sources)


def mock_runner(monkeypatch, bundle, plans, *, low_rows=False, fail=False):
    monkeypatch.setattr(tool, "validate_bundle", lambda *a: (bundle, plans))
    monkeypatch.setattr(tool.capture, "read_header", lambda p: ({}, bundle["source_identities"][p.name]))
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        assert kwargs["check"] is True and "shell" not in kwargs
        if fail: raise subprocess.CalledProcessError(5, argv)
    monkeypatch.setattr(tool.subprocess, "run", execute)
    def encoded(out, plan, ref, sources):
        return {"inputs": {"manifest": str(out / "capture_inputs.json")},
                "row_counts": {"img": {"fit": 24576, "holdout": 8192}, "txt": {"fit": 10 if low_rows else 1800, "holdout": 600}}}
    monkeypatch.setattr(tool, "audit_encoding", encoded)
    def captured(out, plan, encoding, sources, previous_frozen):
        batch = int(out.name[-1])
        return {"row_counts": encoding["row_counts"], "forwards": 32, "source_frozen_sha256": "1" * 64,
                "forward_file_sha256": [f"{batch * 100 + i:064x}" for i in range(32)]}
    monkeypatch.setattr(tool, "audit_capture", captured)
    return calls


@pytest.mark.parametrize("case", ["complete", "row_floor", "subprocess_failure", "duplicate_forward"])
def test_runner_sequential_sealed_tools_no_launcher_and_fail_closed(tmp_path, data, monkeypatch, case):
    batches, excluded, sources = data
    ref = tool.seal(batches, excluded, tmp_path / "sealed")
    bundle, plans = tool.validate_bundle(ref["path"], ref["sha256"])
    calls = mock_runner(monkeypatch, bundle, plans, low_rows=case == "row_floor", fail=case == "subprocess_failure")
    if case == "duplicate_forward":
        original = tool.audit_capture
        def duplicates(*args, **kwargs):
            result = original(*args, **kwargs)
            result["forward_file_sha256"] = [f"{i:064x}" for i in range(32)]
            return result
        monkeypatch.setattr(tool, "audit_capture", duplicates)
    out = tmp_path / "run"
    argv = (ref["path"], ref["sha256"], out, Path("diffusion"), Path("vae"), Path("text_encoder"), tmp_path / "comfy", 2)
    if case == "complete":
        result = tool.run(*argv)
        body = tool.json_pinned(result["path"], result["sha256"])
        assert body["actual_forwards"] == 128
        assert body["actual_row_counts"]["txt"]["fit"] == 7200
        assert len(calls) == 8
        assert all(Path(a[1]).name == "c16_qimg_prepare_inputs.py" for a in calls[:4])
        assert all(Path(a[1]).name == "c16_qimg_capture.py" for a in calls[4:])
        assert all(a[0] == sys.executable and "docker" not in a for a in calls)
        with pytest.raises(ValueError, match="no overwrite"): tool.run(*argv)
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)): tool.run(*argv)
        assert (out / "broad_capture_failure.json").is_file()
        assert not (out / "broad_capture_run_receipt.json").exists()
        assert len(calls) == {"row_floor": 4, "subprocess_failure": 1, "duplicate_forward": 8}[case]
        if case != "duplicate_forward":
            assert all(Path(a[1]).name == "c16_qimg_prepare_inputs.py" for a in calls)


def test_row_cap_and_valid_fit_floor_are_not_token_estimates():
    report = {"row_counts": {"img": {"fit": 24576, "holdout": 8192}, "txt": {"fit": 1536, "holdout": 1}}}
    assert tool.require_rows([report] * 4)["txt"]["fit"] == 6144
    bad = copy.deepcopy(report); bad["row_counts"]["txt"]["fit"] = 1535
    with pytest.raises(ValueError, match="row floor"): tool.require_rows([bad] * 4)
    bad = copy.deepcopy(report); bad["row_counts"]["img"]["fit"] += 1
    with pytest.raises(ValueError, match="row cap"): tool.require_rows([bad] * 4)


@pytest.mark.parametrize("bad", [None, "partial", "sigma", "input", "ids", "order", "source_bytes"])
def test_capture_audit_delegates_full_context_and_checks_exact_linkage(tmp_path, data, monkeypatch, bad):
    # Tiny stand-ins isolate orchestration/identity joins. Existing fold/joint
    # suites own full tensor byte/shape/numeric checks; no source proof is claimed.
    batches, excluded, sources = data
    plan = tool.derive_plans(batches, excluded)[0][0]
    out = tmp_path / "capture"
    out.mkdir()
    valid = torch.ones((1, 2), dtype=torch.bool)
    input_entries = [{"id": e["id"], "tensors": {"path": e["id"] + ".safetensors", "sha256": "1" * 64}} for e in plan["entries"]]
    encoding = {"inputs": {"manifest": str(tmp_path / "inputs.json"), "entries": input_entries},
        "actual_rows": {e["id"]: {"context_valid_sha256": tool.capture.tensor_sha(valid)} for e in plan["entries"]},
        "row_counts": {"img": {"fit": 24576, "holdout": 8192}, "txt": {"fit": 48, "holdout": 16}}}
    records = []
    for entry in plan["entries"]:
        for i, sigma in enumerate(tool.F32_SIGMAS):
            records.append({"sample_id": entry["id"], "split": entry["split"], "image_sha256": entry["image"]["sha256"],
                "prompt_sha256": tool.hashlib.sha256(entry["prompt"].encode()).hexdigest(), "sigma": sigma,
                "file": f"{entry['id']}_sigma{i:02d}.safetensors", "file_sha256": f"{len(records)+50:064x}",
                "image_tokens": 1024, "fixture": False, "complete_context": True, "row_order": "text_then_image"})
    body = {"schema": tool.capture.CAPTURE_SCHEMA, "status": "complete", "fixture": False, "block": 30,
        "inputs": encoding["inputs"], "source_code_sha256": tool.capture.SOURCE_FILES,
        "source_header_stable_after_capture": True, "source": {"source_identity": sources["diffusion"]},
        "runtime": {"parent_class": "comfy.model_base.QwenImage", "full_residency_verified": True,
                    "selected_block_and_time_match_source_bytes": True, "manual_cast_dtype": "torch.bfloat16",
                    "fp8_optimizations": False, "dynamic_patcher": False},
        "frozen_source": {"file": "source.safetensors", "file_sha256": "a" * 64}, "captures": records}
    if bad == "partial": body["captures"] = records[:-1]
    if bad == "order": body["captures"] = records[::-1]
    dump(out / "capture_receipt.json", body)
    monkeypatch.setattr(tool.fold, "frozen_shapes", lambda *a: {"fixture": (1,)})
    monkeypatch.setattr(tool.capture, "verified_file", lambda *a, **k: tmp_path / "input.safetensors")
    encoded = {"model_inputs": torch.zeros((4, 16, 1, 64, 64)), "sigma": torch.tensor(tool.SIGMAS, dtype=torch.float32),
               "context": torch.zeros((1, 2, 3584)), "context_mask": valid, "context_token_ids": torch.tensor([[1, 2]])}
    monkeypatch.setattr(tool, "load_file", lambda *a, **k: encoded)
    checked = []
    monkeypatch.setattr(tool.joint, "context_contract", lambda state, record, metadata: checked.append(record["file"]))
    def tensors(root, record, limit):
        if record["file"] == "source.safetensors":
            return {"fixture": torch.ones(1).to(torch.float8_e4m3fn)}, None
        state = {k: torch.zeros(1) for k in tool.fold.CAPTURE_KEYS}
        state.update(sigma=torch.tensor([record["sigma"]], dtype=torch.float32), context_valid=valid,
                     model_input=encoded["model_inputs"][:1].bfloat16(), context_token_ids=encoded["context_token_ids"],
                     text_attention_mask=torch.zeros((1, 2), dtype=torch.bfloat16))
        if bad == "sigma": state["sigma"] = state["sigma"].double()
        if bad == "input": state["model_input"] = state["model_input"] + 1
        if bad == "ids": state["context_token_ids"] = state["context_token_ids"] + 1
        return state, None
    monkeypatch.setattr(tool.fold, "verified_tensors", tensors)
    if bad:
        with pytest.raises(ValueError):
            tool.audit_capture(out, plan, encoding, sources, "b" * 64 if bad == "source_bytes" else None)
    else:
        result = tool.audit_capture(out, plan, encoding, sources)
        assert result["forwards"] == 32 and result["row_counts"] == encoding["row_counts"]
        assert checked == [r["file"] for r in records]
