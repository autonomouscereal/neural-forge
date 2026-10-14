"""Actual selected-v2 32-batch 2026 geometry producer/consumer interoperability."""
import copy
import json
import os
from pathlib import Path
import pytest
from experiments.chimera_recovery import train_geometry as G


def actual():
    path=Path(os.environ.get('CHIMERA_GEOMETRY_ACTUAL_MANIFEST',
        'C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/geometry-flow-cpu-v5/manifest.json'))
    if not path.is_file():pytest.skip('actual immutable selected geometry metadata not supplied')
    assert G.I.sha(path)=='9534b01d346bab0e60aaf16b2c230128173545f40eaaa249c2c0d51c5a6d163f'
    return json.loads(path.read_text())


def test_actual_canonical_grid_and_source_seal_reach_source_traversal():
    data=actual();assert 'sigmas' not in data
    assert data['sigma_grid']==[.05,.3,.7,1.] and len(data['batches'])==32
    assert G.checked_cache_header(data,data['amendment']['sha256']) is data
    for key,value in [('sigma_grid',[.05]),('adapter_sha256','f'*64),('training_admission_selected',False)]:
        wrong=copy.deepcopy(data);wrong[key]=value
        # Even an internally resealed altered header cannot replace genuine
        # source settings/helper identity or actual selected admission.
        wrong['manifest_sha256']=G.I.digest({k:v for k,v in wrong.items() if k!='manifest_sha256'})
        with pytest.raises(G.Error,match='complete selected'):
            G.checked_cache_header(wrong,wrong['amendment']['sha256'])
    wrong=copy.deepcopy(data);wrong['sigmas']=wrong.pop('sigma_grid')
    wrong['manifest_sha256']=G.I.digest({k:v for k,v in wrong.items() if k!='manifest_sha256'})
    with pytest.raises(G.Error):G.checked_cache_header(wrong,wrong['amendment']['sha256'])
