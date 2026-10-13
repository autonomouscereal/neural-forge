"""Actual native audio contract checks; fixtures make no production claim."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/chimera_recovery/audio_cache.py'
spec = importlib.util.spec_from_file_location('recovery_audio_cache_test', PATH)
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)

def record(asset='a' * 64):
    return {'id': 'fixture-audio', 'route': 'audio_understand', 'split': 'train',
        'messages': [{'role': 'user', 'content': 'Describe the observed sound.'}, {'role': 'assistant', 'content': 'A tone.'}],
        'media': [{'type': 'audio', 'role': 'input', 'sha256': asset}],
        'supervision': {'policy': 'assistant_only', 'message_mask': [False, True]}}

def proof(frames=40):
    samples, duration = frames * 800, frames * .025
    return {'schema': 'chimera_recovery.h3_standalone_audio_probe.v1', 'probe_passed': True,
        'teacher_unchanged': True, 'model_space_verified': True,
        'input_contract': {'target_samples': samples, 'target_rate': 32000, 'duration_seconds': duration, 'no_input_padding_or_trim': True},
        'measurement': {'raw_latent_shape': [1, 32, 2, frames], 'packed_shape': [2 * frames, 32],
            'decoded_shape': [1, 2, samples], 'measured_samples_per_latent_frame': 800, 'nominal_hop_seconds': .025,
            'encoder_repeat_bitwise_equal': True, 'channel_major_pack_bitwise_invertible': True,
            'latent_padded_frames': 0, 'latent_trimmed_frames': 0,
            'nominal_frame_interval_seconds': [[i * 800 / 32000, (i + 1) * 800 / 32000] for i in range(frames)]},
        'decode': {'waveform_padded_samples': 0, 'waveform_trimmed_samples': 0, 'first_pts_seconds': 0.,
            'last_end_seconds': duration, 'target_samples': samples, 'resampler_flushed': True}}

def test_native_pack_exact_band_stereo_scatter_read_mask_and_answer_ownership():
    torch = pytest.importorskip('torch')
    row, p = record(), proof(167)
    media, ids = list(range(2, 336)), [11, 12] + [C.D.PLACEHOLDER] * 334 + [13, 14, 15, 16]
    dialogue = {'source_record_sha256': C.R.digest(row), 'native_text_projection_bitwise_equal': True,
        'policy': C.D.POLICY, 'input_ids': ids, 'media_indices': media,
        'prompt_length': 338, 'assistant_target_mask': [False] * 338 + [True, True]}
    latent = torch.arange(334 * 32, dtype=torch.float32).reshape(334, 32)
    batch, chart = C.build_batch(row, dialogue, p, latent, proof_sha='b' * 64, amendment_sha='c' * 64, charted_attention_sha='d' * 64)
    assert batch['native_media_features'].shape == (1, 334, 6144)
    assert torch.equal(batch['native_media_features'][0, :, 480:512], latent)
    assert not batch['native_media_features'][..., :480].any() and not batch['native_media_features'][..., 512:].any()
    assert chart['latent_frames_per_channel'] == 167 and chart['native_parent_xyz_parity'] is False
    assert batch['assistant_target_mask'].sum() == 2
    assert not batch['h3_prompt_read_mask'][0, 338:].any() and batch['h3_prompt_read_mask'][0, :338].all()
    assert torch.equal(batch['native_media_positions'][0, :167, 3], torch.zeros(167))
    assert torch.equal(batch['native_media_positions'][0, 167:, 3], torch.ones(167))
    with pytest.raises(C.Error, match='latent'):
        C.build_batch(row, dialogue, p, latent[:, :31], proof_sha='b' * 64, amendment_sha='c' * 64, charted_attention_sha='d' * 64)
    bad = dict(dialogue, source_record_sha256='e' * 64)
    with pytest.raises(C.Error, match='dialogue'):
        C.build_batch(row, bad, p, latent, proof_sha='b' * 64, amendment_sha='c' * 64, charted_attention_sha='d' * 64)

def fixture_admission(tmp_path):
    asset = tmp_path / 'asset.wav'
    asset.write_bytes(b'fixture raw bytes; format not an encoder proof')
    row = record(C.R.sha(asset))
    source = tmp_path / 'source.jsonl'
    source.write_bytes(C.R.canonical(row) + b'\n')
    sources = {'source': C.pin(source)}
    for name in ('eval_rows', 'normalized', 'frozen_exclusions'):
        path = tmp_path / (name + '.json')
        path.write_bytes(b'[]\n')
        sources[name] = C.pin(path)
    admission_row = {'id': row['id'], 'split': 'train', 'source_row_sha256': C.R.digest(row),
        'asset_sha256': row['media'][0]['sha256'], 'asset_path': str(asset),
        'authored_answer_sha256': C.R.digest(row['messages'][-1]['content']), 'recovery_group_id': 'fixture-group',
        'source_train_eligible': True, 'train_admitted': True, 'blockers': []}
    path = tmp_path / 'admission.jsonl'
    path.write_bytes(C.R.canonical(admission_row) + b'\n')
    manifest = {'schema': 'recovery_audio_admission.v1', 'sealed': True, 'complete_source_traversal': True,
        'frozen_exclusions_verified': True, 'connected_groups_verified': True, 'sources': sources,
        'outputs': {'admission': C.pin(path)}}
    outer = tmp_path / 'manifest.json'
    outer.write_bytes(C.R.canonical(manifest) + b'\n')
    return row, admission_row, source, path, manifest, outer, asset

def test_admission_exact_join_rejects_held_and_source_asset_tamper(tmp_path):
    _, admitted, source, rows, manifest, outer, asset = fixture_admission(tmp_path)
    _, ready = C.admission(C.pin(outer), tmp_path, source)
    assert ready['fixture-audio']['train_admitted'] is True
    admitted['split'] = 'validation'
    rows.write_bytes(C.R.canonical(admitted) + b'\n')
    manifest['outputs']['admission'] = C.pin(rows)
    outer.write_bytes(C.R.canonical(manifest) + b'\n')
    with pytest.raises(C.Error, match='join'):
        C.admission(C.pin(outer), tmp_path, source)
    admitted['split'] = 'train'
    rows.write_bytes(C.R.canonical(admitted) + b'\n')
    manifest['outputs']['admission'] = C.pin(rows)
    outer.write_bytes(C.R.canonical(manifest) + b'\n')
    asset.write_bytes(b'tampered')
    with pytest.raises(C.Error, match='changed'):
        C.admission(C.pin(outer), tmp_path, source)

def test_incomplete_admission_never_grants_source_readiness(tmp_path):
    _, _, source, _, manifest, outer, _ = fixture_admission(tmp_path)
    manifest['connected_groups_verified'] = False
    outer.write_bytes(C.R.canonical(manifest) + b'\n')
    with pytest.raises(C.Error, match='globally'):
        C.admission(C.pin(outer), tmp_path, source)


def encoder_fixture(tmp_path, monkeypatch, *, speech=False, held=False):
    row, admitted, source, rows, manifest, outer, asset = fixture_admission(tmp_path)
    if speech:
        row['route'] = 'speech_understand'
        row['provenance'] = {'independent_asr_exact_match': True, 'synthetic_official_voice': True,
                             'user_reference_voice_used': False}
    if held:
        row['split'] = admitted['split'] = 'test'
        admitted.update(source_train_eligible=False, train_admitted=False, blockers=['heldout'])
    source.write_bytes(C.R.canonical(row) + b'\n')
    admitted['source_row_sha256'] = C.R.digest(row)
    rows.write_bytes(C.R.canonical(admitted) + b'\n')
    manifest['sources']['source'] = C.pin(source)
    manifest['outputs']['admission'] = C.pin(rows)
    outer.write_bytes(C.R.canonical(manifest) + b'\n')
    calls = []
    def encode(args):
        # Only orchestration is substituted here; this is not an H3 proof.
        calls.append(args)
        args.out.write_bytes(b'{"fixture":true}\n')
        return {'probe_passed': True, 'model_space_verified': True}
    monkeypatch.setattr(C.P, 'run', encode)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return row, source, outer, calls


def test_admitted_encoder_keeps_held_rows_diagnostic_without_lifting_admission(tmp_path, monkeypatch):
    row, source, seal, calls = encoder_fixture(tmp_path, monkeypatch, speech=True, held=True)
    result = C.encode_admitted_source(source, seal, C.R.sha(seal), Path('stage'), Path('comfy'), Path('vae'), tmp_path / 'encoded')
    assert len(calls) == len(result['entries']) == 1
    assert result['entries'][0]['source_row_sha256'] == C.R.digest(row)
    assert calls[0].audio_sha256 == row['media'][0]['sha256']
    assert result['complete_source_traversal'] is True
    assert result['native_admission_granted'] is False and result['training_started'] is False


def test_admitted_speech_encoder_refuses_missing_independent_asr(tmp_path, monkeypatch):
    row, source, seal, calls = encoder_fixture(tmp_path, monkeypatch, speech=True)
    # Rebind a genuinely different source snapshot; the ASR policy still must reject it.
    row['provenance']['independent_asr_exact_match'] = False
    source.write_bytes(C.R.canonical(row) + b'\n')
    data = C.T.read_json(seal)
    admission_rows = Path(data['outputs']['admission']['path'])
    admitted = json.loads(admission_rows.read_text())
    admitted['source_row_sha256'] = C.R.digest(row)
    admission_rows.write_bytes(C.R.canonical(admitted) + b'\n')
    data['sources']['source'], data['outputs']['admission'] = C.pin(source), C.pin(admission_rows)
    seal.write_bytes(C.R.canonical(data) + b'\n')
    with pytest.raises(C.Error, match='independent ASR'):
        C.encode_admitted_source(source, seal, C.R.sha(seal), Path('stage'), Path('comfy'), Path('vae'), tmp_path / 'encoded')
    assert not calls and not (tmp_path / 'encoded/manifest.json').exists()


def test_admitted_encoder_partial_failure_cannot_seal(tmp_path, monkeypatch):
    _, source, seal, _ = encoder_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(C.P, 'run', lambda args: {'probe_passed': False, 'model_space_verified': False})
    with pytest.raises(C.Error, match='partial outputs diagnostic'):
        C.encode_admitted_source(source, seal, C.R.sha(seal), Path('stage'), Path('comfy'), Path('vae'), tmp_path / 'encoded')
    assert not (tmp_path / 'encoded/manifest.json').exists()
