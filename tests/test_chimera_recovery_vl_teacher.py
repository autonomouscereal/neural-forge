"""Prefix/mask contracts with actual original collator; no parent-quality claim."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/vl_teacher.py"
spec = importlib.util.spec_from_file_location("recovery_vl_teacher_test", PATH)
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


def encoded(torch, *, assistant=True, video=False, body=2):
    ids = [1, 10, 11, 20] + ([30] * body + [40, 3] if assistant else [])
    out = {"input_ids": torch.tensor([ids]), "mm_token_type_ids": torch.tensor([[0, 1, 1, 0] + ([0] * (body + 2) if assistant else [])])}
    key, grid = ("pixel_values_videos", "video_grid_thw") if video else ("pixel_values", "image_grid_thw")
    out[key], out[grid] = torch.arange(12).reshape(2, 6).float(), torch.tensor([[1, 2, 2]])
    return out


def processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=40, decode=lambda tokens, **kwargs: "\n" if tokens == [3] else "bad"))


def test_expanded_image_and_video_prefix_masks_answer_eos_only():
    torch = pytest.importorskip("torch")
    for video in (False, True):
        full, before = encoded(torch, video=video), encoded(torch, assistant=False, video=video)
        original = {key: value.clone() for key, value in full.items()}
        result = V.anchor_assistant(processor(), {"row_id": "tiny"}, full, before)
        assert result["assistant_target_mask"].tolist() == [[False, False, False, False, True, True, True, False]]
        assert all(torch.equal(result[key], value) for key, value in original.items())


@pytest.mark.parametrize("change,reason", [("prefix", "prompt IDs"), ("media", "expansion tensors"),
                                          ("types", "placeholders/types"), ("eos", "exactly one"), ("trailing", "trailing content")])
def test_native_prefix_and_completion_drift_refused(change, reason):
    torch = pytest.importorskip("torch")
    full, before = encoded(torch), encoded(torch, assistant=False)
    if change == "prefix":
        before["input_ids"][0, 0] = 2
    elif change == "media":
        before["pixel_values"][0, 0] += 1
    elif change == "types":
        full["mm_token_type_ids"][0, 4] = 1
    elif change == "eos":
        full["input_ids"][0, 4] = 40
    else:
        full["input_ids"][0, -1] = 5
    with pytest.raises(V.Error, match=reason):
        V.anchor_assistant(processor(), {"row_id": "tiny"}, full, before)


def original_writer():
    local = Path("C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/tools/qtext_teacher_cache.py")
    remote = Path("/home/cereal/omni-campaign/distill/tools/qtext_teacher_cache.py")
    source = Path(os.environ.get("CHIMERA_QTEXT_WRITER", str(local if local.is_file() else remote)))
    if not source.is_file():
        pytest.skip("authentic streamed writer unavailable; set CHIMERA_QTEXT_WRITER")
    return V.load_writer(source, V.R.sha(source))


def test_original_collator_shift_right_padding_media_and_restoration():
    torch = pytest.importorskip("torch")
    writer = original_writer()
    collate = writer.collate_vl
    encode_original = writer.encode_vl_row
    calls = []
    def encode_fixture(proc, row, root, kwargs, generation=False):
        calls.append((copy.deepcopy(row), kwargs, generation))
        return encoded(torch, assistant=not generation, video=row.get("video", False), body=row.get("body", 2))
    writer.encode_vl_row = encode_fixture
    rows = [{"row_id": "image", "messages": [{"role": "user", "content": [{"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]},
            {"row_id": "video", "video": True, "body": 1, "messages": [{"role": "user", "content": [{"type": "video"}]}, {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]}]
    try:
        with V.assistant_mask_adapter(writer):
            prepared = [writer.encode_vl_row(processor(), row, Path.cwd(), {"enable_thinking": False}) for row in rows]
            batch = writer.collate_vl(prepared, [0, 1], 0)
            assert batch["store_mask"].tolist() == [[False, False, False, True, True, True, False, False],
                                                    [False, False, False, True, True, False, False, False]]
            assert int(batch["store_mask"].sum()) == 5
            assert torch.equal(batch["pixel_values"], prepared[0]["pixel_values"])
            assert torch.equal(batch["pixel_values_videos"], prepared[1]["pixel_values_videos"])
            assert batch["input_ids"][1, -1] == 0 and not batch["store_mask"][1, -1]
            with pytest.raises(V.Error, match="unsupported audio"):
                writer.encode_vl_row(processor(), {"messages": [{"role": "user", "content": [{"type": "audio"}]}]}, Path.cwd(), {"enable_thinking": False})
            with pytest.raises(V.Error, match="enable_thinking"):
                writer.encode_vl_row(processor(), rows[0], Path.cwd(), {"enable_thinking": True})
            assert all(call[1] == {"enable_thinking": False} for call in calls)
            assert len(rows[0]["messages"]) == 2
        assert writer.encode_vl_row is encode_fixture and writer.collate_vl is collate
        with pytest.raises(RuntimeError):
            with V.assistant_mask_adapter(writer):
                raise RuntimeError("fixture failure")
        assert writer.encode_vl_row is encode_fixture and writer.collate_vl is collate
    finally:
        writer.encode_vl_row = encode_original


def source_fixture(tmp_path):
    assets = tmp_path / "assets"
    records = []
    for index, (route, kind, split) in enumerate((("vision_grounding", "image", "train"), ("video_understand", "video", "test"), ("audio_understand", "audio", "validation"))):
        case = f"case{index}"
        folder = assets / case
        folder.mkdir(parents=True)
        name = {"image": "scene.png", "video": "synchronized.mp4", "audio": "sound.wav"}[kind]
        asset = folder / name
        asset.write_bytes(("real fixture bytes " + kind).encode())
        truth = folder / "groundtruth.json"
        truth.write_text('{"groundtruth":{"count":1}}')
        records.append({"schema": "recovery_corpus.v1", "id": case, "route": route, "split": split,
                        "provenance": {"case_id": case, "group_id": case, "split": split, "groundtruth_path": str(truth), "groundtruth_sha256": V.R.sha(truth)},
                        "media": [{"type": kind, "role": "input", "path": str(asset), "sha256": V.R.sha(asset), "bytes": asset.stat().st_size, "fps": 8, "duration_seconds": 1.}],
                        "messages": [{"role": "user", "content": "Count"}, {"role": "assistant", "content": '{"count":1}'}],
                        "quality": {"source_ready": True, "train_ready": True, "groundtruth_verified": True},
                        "supervision": {"policy": "assistant_only", "message_mask": [0, 1]}})
    source, report = tmp_path / "understanding.jsonl", tmp_path / "report.json"
    def seal():
        source.write_bytes(b"".join(V.R.canonical(row) + b"\n" for row in records))
        report.write_bytes(V.R.canonical({"artifacts": {source.name: {"sha256": V.R.sha(source), "rows": len(records)}}, "cross_split_media_conflicts": 0}))
    seal()
    return assets, records, source, report, seal


def test_real_asset_bytes_held_groups_and_audio_blocked_in_prepared_plan(tmp_path):
    assets, records, source, report, _ = source_fixture(tmp_path)
    out = tmp_path / "prepared"
    plan = V.prepare_fixtures(source, report, assets, out, "/private/vl", writer_path=Path("/authentic/writer.py"), writer_sha="f" * 64)
    assert plan["accepted_rows"] == 2 and len(plan["blocked"]) == 1
    assert plan["blocked"][0]["route"] == "audio_understand" and not plan["production_training_ready"]
    assert [row["rows"] for row in plan["sources"]] == [1, 0, 1]
    prepared = json.loads((out / "test.jsonl").read_text())
    assert prepared["messages"][0]["content"][0]["num_frames"] == 8
    assert prepared["messages"][1]["content"][0]["text"] == records[1]["messages"][1]["content"]
    assert V.load_plan(out / "manifest.json", V.R.sha(out / "manifest.json")) == plan
    (out / "assets/case0/scene.png").write_bytes(b"tampered")
    with pytest.raises(V.Error, match="asset changed"):
        V.load_plan(out / "manifest.json", V.R.sha(out / "manifest.json"))


@pytest.mark.parametrize("problem,reason", [("group", "connected groups"), ("truth", "groundtruth bytes"), ("mask", "source admission")])
def test_invalid_authored_admission_never_publishes_plan(tmp_path, problem, reason):
    assets, records, source, report, seal = source_fixture(tmp_path)
    if problem == "group":
        records[1]["provenance"]["group_id"] = records[0]["provenance"]["group_id"]
    elif problem == "mask":
        records[0]["supervision"]["message_mask"] = [1, 1]
    else:
        (assets / "case0/groundtruth.json").write_bytes(b"bad")
    seal()
    out = tmp_path / "prepared"
    with pytest.raises(V.Error, match=reason):
        V.prepare_fixtures(source, report, assets, out, "/private/vl", writer_path=Path("/writer.py"), writer_sha="f" * 64)
    assert not out.exists()
