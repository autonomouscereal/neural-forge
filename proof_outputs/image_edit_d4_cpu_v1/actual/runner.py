import json,sys
from pathlib import Path
import pytest
from experiments.chimera_recovery import image_edit as E,train_image_edit as D,consolidation_dispatch as C
root=Path('/recovery');aliases=json.loads((root/'inputs/aliases.json').read_text())
cache=root/'native-edit-v1/manifest.json'
data=D.cache_manifest(cache,'8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573',aliases=aliases)
assert len(data['batches'])==32
assert C.LANES['image_edit']==E.OBJECTIVE and C.LANES['qimg']!=E.OBJECTIVE
tests=['test_chimera_image_edit.py','test_chimera_image_edit_stage.py',
       'test_chimera_image_edit_consolidation.py','test_chimera_geometry_dispatch.py']
rc=pytest.main(['/candidate/source/tests/'+name for name in tests]+['-q'])
assert rc==0
proof={'schema':'chimera_recovery.image_edit_d4_actual_cpu.v1','passed':True,
 'source_native_manifest_sha256':E.N.H.sha(cache),'source_authenticated_batches':32,
 'fixture_environment':{'CHIMERA_STREAMED_BASELINE':'/baseline/streamed_kd_trainer.py',
   'CHIMERA_D3_HYBRID':'/baseline/d3_hybrid_module.py','CHIMERA_OMNI_SRC':'/foundation/src'},
 'image_edit_objective':E.OBJECTIVE,'guided_qimg_objective':C.LANES['qimg'],
 'dispatcher_sha256':E.N.H.sha(Path(C.__file__)),
 'production_training_started':False,'gpu_exposed':False,'native_parent_full_parity':False}
Path('/candidate/proof.json').write_bytes(E.R.canonical(proof)+b'\n');print(json.dumps(proof))
