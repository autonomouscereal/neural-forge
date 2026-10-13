import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/chimera_recovery/audio_paths.py'
spec = importlib.util.spec_from_file_location('recovery_audio_paths_test', PATH)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)
fixture = Path(__file__).with_name('test_chimera_audio_cache.py')
fs = importlib.util.spec_from_file_location('recovery_audio_path_fixtures', fixture)
F = importlib.util.module_from_spec(fs)
sys.modules[fs.name] = F
fs.loader.exec_module(F)

def test_path_only_derivation_retains_original_bytes_and_source_semantics(tmp_path):
    original_root = tmp_path / 'original'
    original_root.mkdir()
    _, row, source, rows, original, manifest, _ = F.fixture_admission(original_root)
    out = tmp_path / 'derived'
    result = P.rebind(manifest, P.R.sha(manifest), out, '/work/audio-derived')
    assert (out / 'original' / 'manifest.json').read_bytes() == manifest.read_bytes()
    assert (out / 'original' / 'admission.jsonl').read_bytes() == rows.read_bytes()
    changed = json.loads((out / 'admission.jsonl').read_text())
    assert P.projection(changed) == P.projection(row)
    assert changed['asset_path'].startswith('/work/audio-derived/files/')
    assert result['sources']['source']['sha256'] == original['sources']['source']['sha256']
    assert P.R.digest({k: v for k, v in result.items() if k != 'manifest_sha256'}) == result['manifest_sha256']
    # Resolve the explicit execution prefix to a private test directory without
    # changing the receipt, then prove the real reader refuses semantic edits.
    original_pin = P.C.T.pinned_file
    def pinned(ref, root):
        ref = copy.deepcopy(ref)
        if ref['path'].startswith('/work/audio-derived/'):
            ref['path'] = str(out / ref['path'].removeprefix('/work/audio-derived/'))
        return original_pin(ref, root)
    old = P.C.T.pinned_file
    P.C.T.pinned_file = pinned
    try:
        _, admitted = P.C.admission(P.C.pin(out / 'manifest.json'), out, source)
        assert admitted[row['id']]['train_admitted'] is True
        changed['train_admitted'] = False
        (out / 'admission.jsonl').write_bytes(P.R.canonical(changed) + b'\n')
        result['outputs']['admission']['sha256'] = P.R.sha(out / 'admission.jsonl')
        (out / 'manifest.json').write_bytes(P.R.canonical(result) + b'\n')
        with pytest.raises(P.Error, match='semantics'):
            P.C.admission(P.C.pin(out / 'manifest.json'), out, source)
    finally:
        P.C.T.pinned_file = old
