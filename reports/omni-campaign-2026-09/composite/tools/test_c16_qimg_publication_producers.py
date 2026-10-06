"""Actual stack producer readback with rename consuming each pending stage."""
import errno
import json
from pathlib import Path
import sys

import pytest
import torch

import c16_qimg_atomic_publication as publication
import c16_qimg_stack_capture as capture
import c16_qimg_stack_fold as fold
from test_c16_qimg_stack_fold import test_actual_full60_capture_fold_consolidate_native_load_and_forward as full_integration


@pytest.fixture
def unsupported_link(monkeypatch):
    assert sys.platform=="linux","genuine Linux libc/kernel required; no skipped acceptance"
    def refuse(*args,**kwargs):raise OSError(errno.EPERM,"measured exfat unsupported-link fixture")
    monkeypatch.setattr(publication.os,"link",refuse)


def test_json_exact_reference_after_consumed_stage(tmp_path,unsupported_link):
    target=tmp_path/"receipt.json"
    result=capture._json_exclusive(target,{"schema":"fixture","id":4})
    assert set(result)=={"path","sha256"}
    assert capture._json(result["path"],result["sha256"])["id"]==4
    assert not list(tmp_path.glob("*.pending"))and not list(tmp_path.glob(".*.pending"))
    existing=target.read_bytes()
    with pytest.raises(OSError)as caught:capture._json_exclusive(target,{"id":5})
    assert caught.value.errno==errno.EEXIST and target.read_bytes()==existing
    pending=list(tmp_path.glob(".*.pending"))
    assert len(pending)==1 and json.loads(pending[0].read_text())["id"]==5


def test_tensor_target_readback_after_consumed_stage(tmp_path,unsupported_link):
    target=tmp_path/"tensors.safetensors"
    record=capture._tensors_exclusive(target,{"real":torch.arange(16,dtype=torch.float32).reshape(4,4)})
    assert record["path"]==str(target.resolve())and record["publication"]["method"]=="renameat2_noreplace"
    assert record["publication"]["source_retained"]is False
    loaded=capture._verify_tensors(record,[tmp_path])
    assert torch.equal(loaded["real"],torch.arange(16,dtype=torch.float32).reshape(4,4))
    assert not list(tmp_path.glob(".*.pending"))


def test_fsync_failure_never_publishes_json_or_tensor(tmp_path,monkeypatch):
    def fail(*args):raise OSError(errno.EIO,"flush refused")
    monkeypatch.setattr(capture.os,"fsync",fail)
    for name,write in (("receipt.json",lambda p:capture._json_exclusive(p,{"id":1})),
                       ("tensors.safetensors",lambda p:capture._tensors_exclusive(p,{"real":torch.ones(4)}))):
        target=tmp_path/name
        with pytest.raises(OSError)as caught:write(target)
        assert caught.value.errno==errno.EIO and not target.exists()
    assert len(list(tmp_path.glob(".*.pending")))==2


def test_failed_kernel_publication_preserves_pending_and_refuses_receipt(tmp_path,unsupported_link,monkeypatch):
    def fail(*args):raise OSError(errno.ENOSYS,"unsupported renameat2")
    monkeypatch.setattr(publication,"_rename_noreplace",fail)
    for name,write in (("receipt.json",lambda p:capture._json_exclusive(p,{"id":1})),
                       ("tensors.safetensors",lambda p:capture._tensors_exclusive(p,{"real":torch.ones(4)}))):
        target=tmp_path/name
        with pytest.raises(OSError)as caught:write(target)
        assert caught.value.errno==errno.ENOSYS and not target.exists()
    assert len(list(tmp_path.glob(".*.pending")))==2


def test_actual_full60_capture_fold_consolidation_native_with_rename(tmp_path,unsupported_link):
    # Real tiny60 source forwards, all120 fits, physical1933 tensor readback,
    # actual native loading/forward; no mocked fit/model/storage/rename calls.
    full_integration(tmp_path)
    path=tmp_path/"folded/stack_fold_manifest.json"
    body=capture._json(path,capture.single.file_sha(path))
    assert body["publication_backend"]==publication.policy()
    assert body["implementation_sha256"]["c16_qimg_atomic_publication.py"]==publication.implementation_sha256()
    state=body["standalone_state"]
    assert state["publication"]["method"]=="renameat2_noreplace"and state["publication"]["source_retained"]is False
    assert state["tensor_count"]==1933 and state["shared_timestep_instances_in_standalone_state"]==1
    assert not list((tmp_path/"capture").rglob("*.pending"))and not list((tmp_path/"capture").rglob(".*.pending"))
    assert not list((tmp_path/"folded").rglob("*.pending"))and not list((tmp_path/"folded").rglob(".*.pending"))
