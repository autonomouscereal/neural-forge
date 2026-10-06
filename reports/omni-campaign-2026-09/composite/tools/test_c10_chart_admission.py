"""Invented tiny bytes exercise admission; no real-model capability evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c10_union_install as c10
from test_c10_union_install import native_contract_fixture, HIDDEN, K3_HIDDEN


def chart_bytes(root):
    doc = native_contract_fixture(root)
    folder = root / 'charts' / 'L00'
    folder.mkdir(parents=True)
    chart = folder / 'ridge.safetensors'
    save_file({'C_out': torch.ones(HIDDEN, K3_HIDDEN) / 100,
               'C_in': torch.ones(K3_HIDDEN, HIDDEN) / 100,
               'C_out_bias': torch.zeros(K3_HIDDEN), 'C_in_bias': torch.zeros(HIDDEN)}, str(chart))
    doc.update({'schema': 'neural-forge.c17-native-chart-candidate.v1',
                'trunk_layer': 0, 'trunk_taps': {'rows_key': 'ffn_input'},
                'files': {'ridge': {'file_sha256': c10.sha256_file(str(chart))}},
                'status': 'candidate_only', 'production_install_eligible': False,
                'acceptance': {'exact_installed_computation_gate': 'not_run'}})
    doc['arms']['ridge'].update({'convention': c10.C17_CONVENTION,
                                'production_install_eligible': False})
    return folder, doc


@pytest.mark.parametrize('declaration', ['candidate', 'removed', 'affirmative'])
def test_native_chart_cannot_gain_production_admission_by_metadata(tmp_path, declaration):
    folder, doc = chart_bytes(tmp_path)
    # The observed native boundary contract itself is well formed; it is not acceptance.
    checked = c10.native_chart_boundaries(doc, str(folder / 'candidate_layer.json'), 0, 'ridge',
                                        trunk_hidden=HIDDEN, k3_hidden=K3_HIDDEN)
    assert checked['source_layers'] == [1]
    if declaration == 'removed':
        for key in ('schema', 'status', 'production_install_eligible', 'acceptance'):
            doc.pop(key)
        doc['arms']['ridge'].pop('production_install_eligible')
    elif declaration == 'affirmative':
        doc['schema'] = 'unreviewed.installation.receipt'
        doc['status'] = 'accepted'
        doc['production_install_eligible'] = True
        doc['arms']['ridge']['production_install_eligible'] = True
        doc['acceptance']['exact_installed_computation_gate'] = 'passed'
    # Copying or relabeling the candidate into C10's discovery name must not promote it.
    (folder / 'candidate_layer.json').write_text(json.dumps(copy.deepcopy(doc)), encoding='utf-8')
    (folder / 'layer.json').write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(SystemExit, match='production native chart admission is unavailable'):
        c10.load_chart(str(folder.parent), 0, K3_HIDDEN, HIDDEN, False)


def test_marked_tiny_fixture_still_exercises_chart_mechanics(tmp_path):
    folder, doc = chart_bytes(tmp_path)
    (tmp_path / c10.FIXTURE_MARKER).write_text(json.dumps({'schema': 'neural-forge.c10-synthetic-fixture.v1'}))
    (folder / 'layer.json').write_text(json.dumps(doc), encoding='utf-8')
    chart = c10.load_chart(str(folder.parent), 0, K3_HIDDEN, HIDDEN, False, fixture_root=str(tmp_path))
    assert chart.native_boundary_provenance['fixture'] is True
    assert chart.trunk_to_k3.shape == (K3_HIDDEN, HIDDEN)


def test_production_refusal_precedes_chart_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('production admission touched chart files before refusing')
    monkeypatch.setattr(c10.os.path, 'isfile', forbidden)
    with pytest.raises(SystemExit, match='production native chart admission is unavailable'):
        c10.load_chart(str(tmp_path / 'missing'), 0, K3_HIDDEN, HIDDEN, False)
