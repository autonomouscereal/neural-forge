import importlib.util
from pathlib import Path
import sys
import pytest

source = Path(__file__).resolve().parents[1] / 'experiments/chimera_recovery/audio_replay_index.py'
spec = importlib.util.spec_from_file_location('audio_replay_index_fixture', source)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def inputs():
    admitted = {'tone': {'train_admitted': True, 'route': 'audio_understand'},
        'speech': {'train_admitted': True, 'route': 'speech_understand'},
        'held': {'train_admitted': False, 'route': 'speech_understand'}}
    ce = [{'source_id': key, 'supervised_units': 9} for key in ('tone', 'speech')]
    flow = [{'source_id': 'speech', 'sigma': sigma, 'supervised_units': 10688} for sigma in (.05, .3, .7, 1.)]
    return admitted, ce, flow


def test_replay_coverage_distinguishes_sources_and_sigma_pairs():
    admitted, ce, flow = inputs()
    assert M.coverage(admitted, ce, flow, [.05, .3, .7, 1.])
    with pytest.raises(ValueError, match='CE replay'):
        M.coverage(admitted, ce + ce[:1], flow, [.05, .3, .7, 1.])
    with pytest.raises(ValueError, match='source/sigma'):
        M.coverage(admitted, ce, flow[:-1] + flow[:1], [.05, .3, .7, 1.])


def test_new_global_exclusion_cannot_reenter_earlier_sealed_flow():
    admitted, ce, flow = inputs()
    admitted['speech']['train_admitted'] = False
    with pytest.raises(ValueError, match='source/sigma'):
        M.coverage(admitted, ce[:1], flow, [.05, .3, .7, 1.])
    with pytest.raises(ValueError, match='CE replay'):
        M.coverage(admitted, ce + [{'source_id': 'held', 'supervised_units': 7}], flow, [.05, .3, .7, 1.])


def test_replay_denominator_cannot_be_count_only_or_empty():
    admitted, ce, flow = inputs()
    ce[0]['supervised_units'] = 0
    with pytest.raises(ValueError, match='measured positive'):
        M.coverage(admitted, ce, flow, [.05, .3, .7, 1.])
    ce[0]['supervised_units'] = float('nan')
    with pytest.raises(ValueError, match='measured positive'):
        M.coverage(admitted, ce, flow, [.05, .3, .7, 1.])


def test_flow_scalar_denominator_is_actual_stereo_native_rows_and_rejects_weight_changes():
    torch = pytest.importorskip('torch')
    batch = {'native_media_type_ids': torch.full((1, 334), 3, dtype=torch.int64),
        'native_media_target_mask': torch.ones(1, 334)}
    assert M.flow_units(batch) == 10688
    batch['native_media_target_mask'][0, 0] = .5
    with pytest.raises(ValueError, match='unit weights'):
        M.flow_units(batch)
    batch = {'native_media_type_ids': torch.full((1, 333), 3, dtype=torch.int64),
        'native_media_target_mask': torch.ones(1, 333)}
    with pytest.raises(ValueError, match='stereo'):
        M.flow_units(batch)


def test_partial_manifest_or_cuda_cannot_create_a_seal(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    manifest = tmp_path / 'partial.json'
    manifest.write_text('{"sealed":false}')
    out = tmp_path / 'index.json'
    with pytest.raises(ValueError, match='manifest pin'):
        M.audit(manifest, '0' * 64, tmp_path, manifest, '0' * 64, tmp_path, '1' * 64, out)
    assert not out.exists()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU'):
        M.audit(manifest, M.sha(manifest), tmp_path, manifest, M.sha(manifest), tmp_path, '1' * 64, out)
    assert not out.exists()
