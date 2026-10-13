"""Real byte/argument/source-boundary refusals, including actual candidate run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest
from experiments.chimera_recovery import d1_candidate as D, d1_source_contract as C

BASELINE = Path("C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/d1-candidate-v2")


def write_tensor(path, *, lengths=(3, 2, 4), invalid_padding=False, wrong_held_mask=False):
    width, rows = 4096, len(lengths)
    values = []
    for length in lengths:
        values.extend(list(range(1, length + 1)) + [0] * (width - length))
    if invalid_padding:
        values[5] = 99
    tensors = {"input_ids": ("I64", [rows, width], struct.pack("<" + "q" * len(values), *values)),
               "lengths": ("I64", [rows], struct.pack("<qqq", *lengths)),
               "fit_mask": ("BOOL", [rows], bytes([1, 1, 0])),
               "held_mask": ("BOOL", [rows], bytes([0, 1 if wrong_held_mask else 0, 1]))}
    header, data = {}, b""
    for key, (dtype, shape, raw) in tensors.items():
        header[key] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data) + len(raw)])
        data += raw
    encoded = C.canonical(header)
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def seal(path, value):
    value.pop("seal_sha256", None)
    value["seal_sha256"] = hashlib.sha256(C.canonical(value)).hexdigest()
    path.write_bytes(C.canonical(value))
    return C.sha(path)


def fixture(tmp_path):
    source, prereg, audit = [tmp_path / name for name in ("source.json", "prereg.md", "audit.json")]
    for path in (source, prereg, audit):
        path.write_text("fixture source evidence, not a recovered model", encoding="utf-8")
    ids, docs, manifest = [tmp_path / name for name in ("ids.safetensors", "documents.jsonl", "manifest.json")]
    write_tensor(ids)
    documents = [dict(id="train-a", group="family-a", split="train", length=3, start=0, end=3),
                 dict(id="train-b", group="family-b", split="train", length=2, start=3, end=5),
                 dict(id="held-a", group="family-held", split="validation", length=4, start=0, end=4)]
    docs.write_bytes(b"".join(C.canonical(row) + b"\n" for row in documents))
    def pin(path):
        return dict(path=str(path), sha256=C.sha(path))
    counts = dict(fit_rows=2, held_rows=1, total_rows=3, width=4096, fit_real_tokens=5, diagnostic_real_tokens=4,
                  total_real_tokens=9, padded_positions=12279, fit_documents=2, diagnostic_documents=1)
    value = dict(schema=C.SCHEMA, actual_counts=counts, sources=dict(original_source=pin(source)), dataset=pin(ids), documents=pin(docs),
                 preregistration=pin(prereg), source_audit=pin(audit), fit_groups=["family-a", "family-b"], diagnostic_groups=["family-held"],
                 template_sha256="a" * 64)
    expected = seal(manifest, value)
    args = dict(ids=str(ids), seq_len=4096, lengths_key="lengths", held_rows=1, preregistration=str(prereg))
    return manifest, expected, value, args


def test_real_lengths_exact_held_partition_and_source_changes(tmp_path):
    manifest, expected, value, args = fixture(tmp_path)
    record = C.validate_manifest(manifest, expected, **args)
    assert record["actual_counts"]["fit_real_tokens"] == 5
    assert record["qtext_teacher_cache_required"] is False
    for changed in (dict(held_rows=2), dict(lengths_key="real_lengths"), dict(seq_len=None), dict(max_rows=2), dict(preregistration="foreign.md")):
        with pytest.raises(ValueError, match="trainer arguments differ"):
            C.validate_manifest(manifest, expected, **dict(args, **changed))
    Path(value["sources"]["original_source"]["path"]).write_text("changed source")
    with pytest.raises(ValueError, match="source pin missing/changed"):
        C.validate_manifest(manifest, expected, **args)


@pytest.mark.parametrize("change,error", [("padding", "zero padding"), ("held_mask", "masks disagree"), ("lengths", "real lengths")])
def test_resealed_inner_dataset_cannot_hide_invalid_padding_mask_or_lengths(tmp_path, change, error):
    manifest, _, value, args = fixture(tmp_path)
    path = Path(value["dataset"]["path"])
    write_tensor(path, invalid_padding=change == "padding", wrong_held_mask=change == "held_mask", lengths=(2, 2, 4) if change == "lengths" else (3, 2, 4))
    value["dataset"]["sha256"] = C.sha(path)
    expected = seal(manifest, value)
    with pytest.raises(ValueError, match=error):
        C.validate_manifest(manifest, expected, **args)


def test_resealed_shared_group_and_wrong_document_boundary_refused(tmp_path):
    manifest, _, value, args = fixture(tmp_path)
    value["diagnostic_groups"] = ["family-a"]
    expected = seal(manifest, value)
    with pytest.raises(ValueError, match="overlapping"):
        C.validate_manifest(manifest, expected, **args)
    value["diagnostic_groups"] = ["family-held"]
    docs = Path(value["documents"]["path"])
    documents = [json.loads(line) for line in docs.read_text().splitlines()]
    documents[-1]["start"] = 5
    documents[-1]["end"] = 9
    docs.write_bytes(b"".join(C.canonical(row) + b"\n" for row in documents))
    value["documents"]["sha256"] = C.sha(docs)
    expected = seal(manifest, value)
    with pytest.raises(ValueError, match="source spans"):
        C.validate_manifest(manifest, expected, **args)


def test_manifest_reseal_does_not_replace_external_expected_sha(tmp_path):
    manifest, expected, value, args = fixture(tmp_path)
    value["actual_counts"]["held_rows"] = 2
    seal(manifest, value)
    with pytest.raises(ValueError, match="explicit launcher/trainer pin"):
        C.validate_manifest(manifest, expected, **args)


def test_actual_candidate_trainer_refuses_source_before_opening_checkpoint(tmp_path, monkeypatch):
    if not BASELINE.is_dir():
        pytest.skip("reviewed campaign predecessor is not installed; run with staged source")
    manifest, expected, value, args = fixture(tmp_path)
    out = tmp_path / "candidate"
    receipt = D.prepare(BASELINE, manifest, expected, "/reviewed/successor/manifest.json", "/reviewed/candidate", out)
    trainer_path = out / "tools/d1_trunk_recovery.py"
    spec = importlib.util.spec_from_file_location("actual_d1_source_candidate", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, trainer)
    spec.loader.exec_module(trainer)
    monkeypatch.setitem(sys.modules, "d1_source_contract", C)
    monkeypatch.setattr(trainer, "preregistration_record", lambda args: dict(fixture_only=True))
    entered = []
    monkeypatch.setattr(trainer, "bind_chimera", lambda: entered.append("opened-model"))
    Path(value["sources"]["original_source"]["path"]).write_text("changed after source preparation")
    argv = ["--child", "unopened-child", "--parent", "unopened-parent", "--ids", args["ids"], "--seq-len", "4096", "--held-rows", "1",
            "--preregistration", args["preregistration"], "--source-manifest", str(manifest), "--source-manifest-sha256", expected,
            "--source-validator-sha256", C.sha(C.__file__), "--out", str(tmp_path / "must-not-exist")]
    with pytest.raises(ValueError, match="source pin missing/changed"):
        trainer.run(argv)
    assert not entered and not (tmp_path / "must-not-exist").exists()
    launch = (out / "queues/d1_launch.sh").read_text()
    assert 'held = int(args.held_rows)' in launch
    assert 'fit_tokens": int(got_len[:R-held].sum())' in launch
    assert 'verify_source || { status "FAIL: source/code changed after resource wait"' in launch
    assert 'manifest["shards"]' not in launch and "text_rows0-977.safetensors" not in launch
    assert receipt["training_launched"] is False and receipt["qtext_optional_corpus_args"][-1] == "lengths"
    assert C.sha(trainer_path) == receipt["files"]["tools/d1_trunk_recovery.py"]
