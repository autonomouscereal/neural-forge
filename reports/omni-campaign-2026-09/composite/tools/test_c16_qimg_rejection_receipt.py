"""Synthetic CPU refusal receipts; no real QIMG numerical or media proof."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_fold as tool
from test_c16_qimg_fold import bundle, dump, bounded_threads


def rejection_case(tmp_path, monkeypatch, kind):
    source = tmp_path / "capture"
    source.mkdir()
    path, _, policy_path, policy = bundle(source)
    data = tool.load_captures([path], fixture=True)
    # Unit-only loader stub lets main() reach the injected historical exception;
    # genuine CLI fixture refusal remains covered by the development CLI suite.
    # This helper never emits an artifact or simulates genuine admission.
    monkeypatch.setattr(tool, "load_captures", lambda *args, **kwargs: data)
    # Legacy rejection evidence is intentionally injected to exercise durable
    # error persistence; current finite targets never produce this exception.
    measured = {"status": "rejected_" + kind, "historical_test_injection": True,
                "measurement": {"rel_l2": .9, "cos": .2, "stored_update_scale": 2.0}}
    def injected(*args, **kwargs):
        raise tool.FoldRejected("injected historical refusal, not current policy", copy.deepcopy(measured))
    monkeypatch.setattr(tool, "fold_loaded", injected)
    return path, policy_path, policy, measured


@pytest.mark.parametrize("kind", ["numerical_policy", "source_replay", "stored_update"])
def test_real_refusals_persist_original_measurements_and_identities_without_candidate(tmp_path, monkeypatch, kind):
    path, policy_path, policy, measured = rejection_case(tmp_path, monkeypatch, kind)
    source_hashes = {p.name: tool.file_sha(p) for p in path.parent.iterdir() if p.is_file()}
    out = tmp_path / "fold"
    with pytest.raises(tool.FoldRejected) as exc:
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    receipt_path = tmp_path / "fold.rejected.json"
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == exc.value.receipt
    assert persisted["status"] == "rejected_" + kind
    assert all(persisted[k] == v for k, v in measured.items())
    assert persisted["policy"] == policy
    assert persisted["policy_sha256"] == tool.file_sha(policy_path)
    assert persisted["capture_receipts"][0]["sha256"] == tool.file_sha(path)
    assert persisted["source_frozen_file_sha256"] == tool.file_sha(path.parent / "frozen.safetensors")
    assert persisted["candidate_written"] is False
    assert persisted["rejection_receipt"] == str(receipt_path.resolve())
    assert persisted["implementation_sha256"]["c16_qimg_fold.py"] == tool.file_sha(Path(tool.__file__))
    assert not out.exists()
    assert {p.name: tool.file_sha(p) for p in path.parent.iterdir() if p.is_file()} == source_hashes


@pytest.mark.parametrize("stdout_loss", ["discarded", "broken_pipe"])
def test_cli_receipt_survives_stdout_loss(tmp_path, monkeypatch, stdout_loss):
    path, policy_path, _, _ = rejection_case(tmp_path, monkeypatch, "numerical_policy")
    out = tmp_path / "fold"
    receipt = tmp_path / "diagnostics" / "refusal.json"
    monkeypatch.setattr(sys, "argv", ["c16_qimg_fold.py", "--captures", str(path), "--policy", str(policy_path),
                                     "--policy-sha256", tool.file_sha(policy_path), "--out", str(out),
                                     "--rejection-receipt", str(receipt)])
    def lost_print(*a, **k):
        if stdout_loss == "broken_pipe":
            raise BrokenPipeError("synthetic detached stdout")
    monkeypatch.setattr("builtins.print", lost_print)
    expected = BrokenPipeError if stdout_loss == "broken_pipe" else SystemExit
    with pytest.raises(expected) as exc:
        tool.main()
    if stdout_loss == "discarded":
        assert exc.value.code == 2
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["status"] == "rejected_numerical_policy"
    assert persisted["candidate_written"] is False
    assert not out.exists()


@pytest.mark.parametrize("existing", ["file", "directory", "dangling_symlink"])
def test_existing_rejection_destination_refuses_before_policy_or_tensor_load(tmp_path, monkeypatch, existing):
    source = tmp_path / "capture"
    source.mkdir()
    path, _, policy_path, _ = bundle(source)
    receipt = tmp_path / "fold.rejected.json"
    if existing == "file":
        receipt.write_bytes(b"preserve old diagnostics")
    elif existing == "directory":
        receipt.mkdir()
        (receipt / "preserve").write_text("old")
    else:
        try:
            receipt.symlink_to(tmp_path / "missing.json")
        except OSError as exc:
            pytest.skip("host cannot create symlinks: " + str(exc))
    before = receipt.read_bytes() if existing == "file" else None
    monkeypatch.setattr(tool, "load_policy", lambda *a: pytest.fail("unsafe destination reached policy/tensor load"))
    with pytest.raises(ValueError, match="rejection receipt.*new file"):
        tool.convert([path], policy_path, "0" * 64, tmp_path / "fold", fixture=True)
    if before is not None:
        assert receipt.read_bytes() == before
    elif existing == "directory":
        assert (receipt / "preserve").read_text() == "old"
    else:
        assert receipt.is_symlink() and not receipt.exists()
    assert not (tmp_path / "fold").exists()


@pytest.mark.parametrize("unsafe", ["candidate", "candidate_child", "candidate_parent", "capture", "new_capture_parent", "candidate_in_capture", "policy"])
def test_input_or_candidate_overlap_refuses_before_fit_without_creating_parents(tmp_path, monkeypatch, unsafe):
    source = tmp_path / "capture"
    source.mkdir()
    path, _, policy_path, _ = bundle(source)
    source = path.parent
    out = tmp_path / "fold"
    receipt = {"candidate": out, "candidate_child": out / "refusal.json", "candidate_parent": tmp_path / "new_parent",
               "capture": source / "refusal.json", "new_capture_parent": source / "new_parent" / "refusal.json",
               "candidate_in_capture": tmp_path / "refusal.json", "policy": policy_path}[unsafe]
    if unsafe == "candidate_parent":
        out = receipt / "fold"
    elif unsafe == "candidate_in_capture":
        out = source / "new_parent" / "fold"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(tool, "load_policy", lambda *a: pytest.fail("unsafe destination reached policy/tensor load"))
    with pytest.raises(ValueError):
        tool.convert([path], policy_path, "0" * 64, out, rejection_receipt=receipt, fixture=True)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_receipt_created_during_fit_is_never_overwritten(tmp_path, monkeypatch):
    path, policy_path, _, _ = rejection_case(tmp_path, monkeypatch, "numerical_policy")
    out = tmp_path / "fold"
    receipt = tmp_path / "fold.rejected.json"
    original = tool.fold_loaded
    def concurrent_receipt(*a, **k):
        receipt.write_text("another run's diagnostics", encoding="utf-8")
        return original(*a, **k)
    monkeypatch.setattr(tool, "fold_loaded", concurrent_receipt)
    with pytest.raises((ValueError, FileExistsError)):
        tool.convert([path], policy_path, tool.file_sha(policy_path), out, fixture=True)
    assert receipt.read_text(encoding="utf-8") == "another run's diagnostics"
    assert not out.exists()


@pytest.mark.parametrize("validate_only", [False, True])
def test_success_or_validation_does_not_create_rejection_artifact_or_parents(tmp_path, validate_only):
    source = tmp_path / "capture"
    source.mkdir()
    path, _, policy_path, _ = bundle(source)
    out = tmp_path / "fold"
    receipt = tmp_path / "diagnostics" / "refusal.json"
    result = tool.convert([path], policy_path, tool.file_sha(policy_path), out,
                          rejection_receipt=receipt, fixture=True, validate_only=validate_only)
    assert result.get("validated") is True if validate_only else result["numerical_policy_passed"] is True
    assert not receipt.parent.exists()
    assert out.exists() is not validate_only


def test_ordinary_validation_error_is_not_mislabeled_as_numerical_rejection(tmp_path, monkeypatch):
    path, policy_path, _, _ = rejection_case(tmp_path, monkeypatch, "numerical_policy")
    monkeypatch.setattr(tool, "row_preflight", lambda *a: (_ for _ in ()).throw(ValueError("synthetic invalid row contract")))
    with pytest.raises(ValueError, match="invalid row contract"):
        tool.convert([path], policy_path, tool.file_sha(policy_path), tmp_path / "fold", fixture=True)
    assert not (tmp_path / "fold").exists()
    assert not (tmp_path / "fold.rejected.json").exists()
