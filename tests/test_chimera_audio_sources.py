"""Real PCM and full tiny global admission fixtures; no native/model claim."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import wave

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/chimera_recovery/audio_sources.py'
spec = importlib.util.spec_from_file_location('audio_sources_test', PATH)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)


@pytest.fixture
def fixture(tmp_path):
    helper = Path(os.environ.get('CHIMERA_AUDIO_ADMISSION_HELPER', str(PATH.with_name('audio_admission.py'))))
    if not helper.exists():
        pytest.skip('actual source-pinned global admission helper not installed')
    A = S.helper(helper, S.R.sha(helper))
    def write(name, value, lines=False):
        path = tmp_path / name
        path.write_bytes(b''.join(A.canonical(r) + b'\n' for r in value) if lines else A.canonical(value) + b'\n')
        return {'path': str(path), 'sha256': A.file_sha(path), 'rows': len(value) if lines else None}
    def wav(name, rate, sample):
        path = tmp_path / name
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
            handle.writeframes(sample * rate)
        return path
    tone, speech = wav('tone.wav', 16000, b'\x01\x00'), wav('speech.wav', 24000, b'\x02\x00')
    truth = {'contains_speech': False, 'count': 1, 'events': [{'kind': 'tone', 'start_seconds': .25, 'frequency_hz': 1000}]}
    groundtruth = write('truth.json', {'groundtruth': truth})
    record = {'schema': 'recovery_corpus.v1', 'id': 'tone2026', 'route': 'audio_understand', 'split': 'train',
        'messages': [{'role': 'user', 'content': 'Describe these observed sounds.'}, {'role': 'assistant', 'content': json.dumps(truth)}],
        'media': [{'type': 'audio', 'role': 'input', 'path': str(tone), 'sha256': A.file_sha(tone), 'bytes': tone.stat().st_size,
                   'sample_rate': 16000, 'duration_seconds': 1.}],
        'supervision': {'policy': 'assistant_only', 'message_mask': [0, 1]}, 'quality': {'source_ready': True, 'groundtruth_verified': True},
        'provenance': {'source': 'authored2026', 'source_year': 2026, 'source_verified': True, 'source_code_sha256': 'a' * 64,
            'group_id': 'tonegroup2026', 'split': 'train', 'source_ids': ['tone2026'],
            'groundtruth_path': groundtruth['path'], 'groundtruth_sha256': groundtruth['sha256']}}
    user = {'schema': 'recovery_corpus.v1', 'id': 'speech2026', 'route': 'speech_understand', 'split': 'train',
        'messages': [{'role': 'user', 'content': 'Transcribe the observed speech.'},
                     {'role': 'assistant', 'content': 'The gardener moved several clay pots beneath the covered porch.'}],
        'media': [{'type': 'audio', 'role': 'input', 'path': str(speech), 'sha256': A.file_sha(speech),
                   'source_media_sha256': A.file_sha(speech)}],
        'supervision': {'policy': 'assistant_only', 'message_mask': [0, 1]},
        'provenance': {'source': 'speech2026', 'source_year': 2026, 'group_id': 'speechgroup2026', 'split': 'train',
            'source_ids': ['speech2026'], 'speaker': 'Ryan', 'independent_asr_exact_match': True,
            'synthetic_official_voice': True, 'user_reference_voice_used': False,
            'original_audio_path': str(speech), 'original_audio_sha256': A.file_sha(speech),
            'plan_row_sha256': 'b' * 64, 'tts_row_sha256': 'c' * 64, 'asr_row_sha256': 'd' * 64}}
    measure, _ = A.pcm(speech)
    user['provenance']['minimal_tail_derivation'] = {'original': measure, 'derived': measure, 'tail_padding_samples': 0,
        'original_asset_sha256': A.file_sha(speech), 'derived_asset_sha256': A.file_sha(speech)}
    unrelated = {'schema': 'recovery_corpus.v1', 'id': 'registry2026', 'split': 'train', 'media': [],
        'messages': [{'role': 'user', 'content': 'Explain this unrelated orbital mechanics experiment.'}],
        'provenance': {'group_id': 'registry2026', 'recovery_group_id': 'registry2026', 'source_ids': ['orbital2026']}}
    normalized = write('normalized.jsonl', [unrelated], True)
    eval_rows = write('eval.jsonl', [], True)
    registry = write('registry.jsonl', [unrelated], True)
    corpus = write('corpus.json', {'schema': 'recovery_corpus.manifest.v1', 'outputs': {'normalized': normalized},
                                  'eval_sources': [eval_rows], 'sources': [registry]})
    prior = {'schema': 'recovery_media_admission.v1', 'connected_groups_verified': True, 'eval_disjoint': True,
        'sources': {'normalized': normalized, 'corpus_manifest': corpus, 'eval_rows': eval_rows},
        'outputs': {'admission': write('prior-rows.jsonl', [], True), 'cross_route_training_exclusions': write('prior-exclusions.jsonl', [], True)}}
    prior['manifest_sha256'] = A.digest(prior)
    common = {'normalized': normalized, 'eval_rows': eval_rows, 'corpus_manifest': corpus,
              'prior_admission': write('prior.json', prior), 'frozen_exclusions': write('frozen.json', [])}
    source = write('procedural-source.jsonl', [record], True)
    inventory = write('procedural-inventory.jsonl', [record], True)
    report = write('report.json', {'schema': 'operator_media_seed_report.v1', 'generated_at': '2026-10-13',
        'source_code_sha256': 'a' * 64, 'examples': 1, 'cross_split_media_conflicts': 0, 'verified_assets': 1,
        'artifacts': {'procedural-source.jsonl': source, 'procedural-inventory.jsonl': inventory}})
    m, _ = A.pcm(tone)
    pf_row = {**{k: v for k, v in m.items() if k != 'pcm_sha256'}, 'id': record['id'], 'split': 'train',
        'source_row_sha256': A.digest(record), 'recovery_group_id': record['provenance']['group_id'], 'waveform_sha256': A.file_sha(tone),
        'groundtruth_sha256': groundtruth['sha256'], 'authored_answer_sha256': A.digest(truth)}
    pf = write('preflight.json', {'schema': 'chimera_recovery.audio_understanding_preflight.v1', 'source_ready': True,
        'source_sha256': source['sha256'], 'source_report_sha256': report['sha256'], 'audio_rows': 1, 'rows': [pf_row]})
    proc_specs = {**common, 'source': source, 'source_inventory': inventory, 'source_report': report, 'source_preflight': pf}
    proc_candidates, _ = A.procedural_candidates({k: Path(v['path']) for k, v in proc_specs.items()}, proc_specs, tmp_path)
    A.seal_admission(specs=proc_specs, candidates=proc_candidates, inventory=[record], output=tmp_path / 'procedural-admission')
    def speech_admission(extra=None):
        inv = [user] + ([extra] if extra else [])
        speech_specs = {**common, 'source': write('speech-source.jsonl', [user], True),
                        'source_inventory': write('speech-inventory.jsonl', inv, True)}
        candidate = {'record': user, 'asset_path': str(speech), 'asset_sha256': A.file_sha(speech), 'measure': measure,
            'teacher_evidence': {**{k: user['provenance'][k] for k in ('plan_row_sha256', 'tts_row_sha256', 'asr_row_sha256')},
                                 'independent_asr_exact_match': True}}
        A.seal_admission(specs=speech_specs, candidates=[candidate], inventory=inv, output=tmp_path / 'speech-admission')
        return tmp_path / 'speech-admission/manifest.json'
    return A, helper, record, user, tmp_path / 'procedural-admission/manifest.json', speech_admission


def run_prepare(tmp_path, fixture, speech):
    A, helper, _, _, procedural, _ = fixture
    return S.prepare(procedural, speech, procedural_sha=A.file_sha(procedural), speech_sha=A.file_sha(speech),
        admission_helper=helper, helper_sha=A.file_sha(helper), procedural_asset_root=tmp_path, out=tmp_path / 'joint')


def test_full_union_preserves_source_rows_and_recomputes_each_family(tmp_path, fixture):
    A, _, record, user, _, speech_builder = fixture
    result = run_prepare(tmp_path, fixture, speech_builder())
    assert result['source_candidates'] == result['full_original_inventory'] == result['train_admitted'] == 2
    assert result['admitted_by_route'] == {'audio_understand': 1, 'speech_understand': 1}
    assert {A.digest(r) for r in A.rows(tmp_path / 'joint/source_ready.jsonl')} == {A.digest(record), A.digest(user)}
    assert result['native_train_ready'] is False and result['training_started'] is False


def test_new_cross_source_holdout_asset_excludes_previously_admitted_procedural_row(tmp_path, fixture):
    A, _, record, _, _, speech_builder = fixture
    held = copy.deepcopy(record)
    held.update(id='held_other_route2026', route='video_understand', split='test')
    held['provenance'].update(group_id='held_video2026', split='test', source_ids=['held_other2026'])
    result = run_prepare(tmp_path, fixture, speech_builder(held))
    actual = {r['id']: r for r in A.rows(tmp_path / 'joint/global_admission/admission.jsonl')}
    assert result['train_admitted'] == 1 and result['full_original_inventory'] == 3
    assert actual[record['id']]['train_admitted'] is False and 'source_family_holdout' in actual[record['id']]['blockers']
    assert actual['speech2026']['train_admitted'] is True


def test_joint_admission_retains_prior_nonheld_exclusion(tmp_path, fixture):
    A, _, _, _, procedural, speech_builder = fixture
    data = json.loads(procedural.read_text())
    path = Path(data['outputs']['admission']['path'])
    row = next(A.rows(path))
    row.update(train_admitted=False, source_train_eligible=False, blockers=['explicit_prior_exclusion2026'])
    path.write_bytes(A.canonical(row) + b'\n')
    data['outputs']['admission']['sha256'] = A.file_sha(path)
    data.pop('manifest_sha256'); data['manifest_sha256'] = A.digest(data)
    procedural.write_bytes(A.canonical(data) + b'\n')
    result = run_prepare(tmp_path, fixture, speech_builder())
    row = next(A.rows(tmp_path / 'joint/global_admission/admission.jsonl'))
    assert result['train_admitted'] == 1 and row['train_admitted'] is False
    assert 'explicit_prior_exclusion2026' in row['blockers']


def test_joint_refuses_different_global_frozen_snapshot(tmp_path, fixture):
    A, _, _, _, _, speech_builder = fixture
    speech = speech_builder()
    data = json.loads(speech.read_text())
    path = tmp_path / 'different-frozen.json'
    path.write_bytes(A.canonical(['Distinct held sentence from another frozen source.']) + b'\n')
    data['sources']['frozen_exclusions'] = {'path': str(path), 'sha256': A.file_sha(path)}
    speech.write_bytes(A.canonical(data) + b'\n')
    with pytest.raises(S.Error, match='global normalized/eval/prior/frozen'):
        run_prepare(tmp_path, fixture, speech)
    assert not (tmp_path / 'joint').exists()


def test_joint_source_original_pcm_tamper_refuses_before_promotion(tmp_path, fixture):
    _, _, _, user, _, speech_builder = fixture
    speech = speech_builder()
    Path(user['provenance']['original_audio_path']).write_bytes(b'corrupted2026')
    with pytest.raises((S.Error, ValueError), match='changed'):
        run_prepare(tmp_path, fixture, speech)
    assert not (tmp_path / 'joint').exists()


def test_explicit_original_admission_sha_pin_cannot_be_replaced(tmp_path, fixture):
    A, helper, _, _, procedural, speech_builder = fixture
    speech = speech_builder()
    with pytest.raises(S.Error, match='changed'):
        S.prepare(procedural, speech, procedural_sha='0' * 64, speech_sha=A.file_sha(speech),
            admission_helper=helper, helper_sha=A.file_sha(helper), procedural_asset_root=tmp_path, out=tmp_path / 'joint')
    assert not (tmp_path / 'joint').exists()


def test_incomplete_original_encoder_inventory_cannot_be_composed(tmp_path, fixture):
    A, _, _, _, _, speech_builder = fixture
    run_prepare(tmp_path, fixture, speech_builder())
    source = tmp_path / 'joint/source_ready.jsonl'
    admission = tmp_path / 'joint/global_admission/manifest.json'
    clean = tmp_path / 'incomplete-clean.json'
    clean.write_bytes(A.canonical({'schema': 'chimera_recovery.clean_audio_cache.v1', 'sealed': True,
        'complete_source_traversal': True, 'source': S.C.pin(source), 'entries': []}) + b'\n')
    with pytest.raises(S.Error, match='original clean proof inventory'):
        S.compose_cache(source, admission, A.file_sha(admission), [S.C.pin(clean)], tmp_path / 'composed')
    assert not (tmp_path / 'composed').exists()
