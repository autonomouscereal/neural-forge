"""D4 image-edit source guard precedes loss; no second optimizer callback."""
import json
from pathlib import Path
import pytest
from experiments.chimera_recovery import consolidation_dispatch as D,consolidation_objectives as O,image_edit as E
from test_chimera_image_edit_stage import fixture

def test_distinct_objective_replaces_no_guided_image_route():
    assert D.LANES['qimg']==O.GUIDED
    assert D.LANES['image_edit']==O.EDIT==E.OBJECTIVE
    assert O.EDIT in O.OBJECTIVES
    assert D.SOURCES[O.EDIT]==E.SCHEMA and D.EDIT_STAGE==E.STAGE
    assert D.replay_lineage_order().index(E.STAGE)==D.replay_lineage_order().index(D.GEOMETRY_STAGE)+1

def test_actual_native_callback_scalar_denominator_and_window_scaling(tmp_path):
    torch,M,construct,batch,compute=fixture(tmp_path)
    from experiments.chimera_recovery import train_image_edit as Q
    h1,h2=construct(),construct();unit=h1.layout[0].container
    engines=[Q.ImageEditEngine(M,h,unit,Q.OPTIMIZER,{'fixture':True},total_steps=2) for h in (h1,h2)]
    loss=O.image_edit_backward(M,h1,batch,compute,4*64,1.)
    scaled=O.image_edit_backward(M,h2,batch,compute,4*64,.125)
    assert abs(loss-scaled)<1e-6
    for name,p in engines[0].census.items():
        assert p.grad is not None,name
        torch.testing.assert_close(engines[1].census[name].grad,p.grad*.125,rtol=3e-4,atol=2e-5)
    assert all(not engine.optimizer.state for engine in engines)
    with pytest.raises(O.Error,match='denominator'):O.image_edit_backward(M,h1,batch,compute,4*64+1,1.)

def test_changed_source_refused_before_any_backward_or_device_transfer(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from safetensors.torch import save_file
    tensor=tmp_path/'batch.safetensors';save_file({'input_ids':torch.tensor([[1]])},str(tensor))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'fixture':True}))
    item={'id':'fixture-edit','tensors':D.I.pin(tensor),'teacher_receipt':D.I.pin(receipt)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'batches':[item]}))
    row={'objective':O.EDIT,'lane':'image_edit','input':D.I.pin(tensor),'target':D.I.pin(receipt),
         'source_manifest':D.I.pin(manifest),'source_batch_id':item['id'],'supervised_units':65536}
    reached=[]
    class Reject:
        def __init__(self,*args,**kwargs):pass
        def bind(self,batch):
            assert batch['input_ids'].device.type=='cpu'
            raise E.Error('changed genuine source proof')
    monkeypatch.setattr(E,'ImageEditAdapter',Reject)
    monkeypatch.setattr(O,'image_edit_backward',lambda *a,**k:reached.append('backward'))
    callback=D.NativeBackward(None,tmp_path,{})
    with pytest.raises(E.Error,match='changed genuine source'):callback(object(),row,1.)
    assert reached==[]
