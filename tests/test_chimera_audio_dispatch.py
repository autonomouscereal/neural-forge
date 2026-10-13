"""Distinct audio production dispatch, fail-closed contract and safety."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/chimera_recovery/train_audio.py'
spec = importlib.util.spec_from_file_location('recovery_audio_dispatch_test', PATH)
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)

def test_audio_entry_cannot_alias_vision_or_guided_generation(tmp_path):
    config = tmp_path / 'config.json'
    config.write_bytes(A.R.canonical({'schema': 'chimera_recovery.native_stage_config.v1', 'stage': 'D3d'}) + b'\n')
    result = A.preflight(config)
    assert not result['production_ready'] and not result['training_started']
    assert result['blockers'] == ['explicit D3d_audio_ce config is required']
    with pytest.raises(A.Error, match='preflight blocked'):
        A.train(config, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()

def test_missing_sources_surface_concrete_blockers_without_loading_engine(tmp_path):
    config = tmp_path / 'config.json'
    config.write_bytes(A.R.canonical({'schema': 'chimera_recovery.audio_stage_config.v1', 'stage': A.STAGE}) + b'\n')
    result = A.preflight(config)
    assert not result['production_ready'] and not result['training_started']
    assert any('preregistration' in blocker for blocker in result['blockers'])
    assert any('batches_path' in blocker for blocker in result['blockers'])

def test_selected_speaking_successor_replaces_old_direct_d4_order_without_accepting_missing_sources(tmp_path):
    config = tmp_path / 'config.json'
    value = {'schema': 'chimera_recovery.audio_stage_config.v1', 'stage': A.STAGE,
        'objective': A.OBJECTIVE, 'read_policy': A.READ_POLICY, 'dtype': 'float32',
        'after_stage': 'D3d', 'before_stage': 'D3d_speech_flow', 'safety': A.SAFETY, 'seed': 20261013}
    config.write_bytes(A.R.canonical(value))
    result = A.preflight(config)
    assert not result['production_ready'] and not result['training_started']
    assert any('base:' in blocker for blocker in result['blockers'])
    assert not any('audio CE must follow' in blocker or 'audio objective, stage order' in blocker for blocker in result['blockers'])
    for after, before in [('D3d', 'D4'), ('D3b', 'D3d_speech_flow')]:
        config.write_bytes(A.R.canonical(dict(value, after_stage=after, before_stage=before)))
        refused = A.preflight(config)
        assert not refused['production_ready']
        assert any('audio CE must follow completed vision D3d and precede selected D3d_speech_flow' in blocker
                   for blocker in refused['blockers'])

def test_finite_guard_does_not_advance_onepass_on_retry_and_halts_eighth():
    fixture = SimpleNamespace(step=2, consecutive_skips=0, skip_window=[])
    result = None
    for _ in range(8):
        result = A.AudioCEEngine.nonfinite_skip(fixture)
    assert result == {'loss': None, 'skipped': True, 'step': 2, 'halt': True}
    assert fixture.step == 2 and len(fixture.skip_window) == 8
    fixture.consecutive_skips, fixture.skip_window = 0, [True] * 20 + [False] * 79
    assert A.AudioCEEngine.nonfinite_skip(fixture)['halt'] is True

def test_declared_audio_optimizer_cannot_change_silently():
    expected = {'unit_lr': 2e-5, 'bridge_lr': 2e-5, 'lora_lr': 1e-4, 'betas': [.9, .95], 'eps': 1e-8,
        'weight_decay': 0., 'clip': 1., 'warmup_steps': 20, 'min_lr_ratio': .1}
    assert A.audio_optimizer(expected) == expected
    with pytest.raises(A.Error, match='successor amendment'):
        A.audio_optimizer(dict(expected, bridge_lr=1e-4))

def test_dispatch_calls_actual_audio_engine_with_native_shift_inputs(monkeypatch):
    calls = []
    sentinel, mask = object(), object()
    monkeypatch.setattr(A.T, 'make_segment', lambda batch, **kwargs: sentinel)
    engine = SimpleNamespace(hybrid=SimpleNamespace(media_placeholder=248832),
        train_step=lambda *args, **kwargs: calls.append((args, kwargs)) or {'step': 1})
    result = A.dispatch_step(engine, {'assistant_target_mask': mask}, {'prompt_length': 101, 'completion_eos_id': 151645})
    assert result['step'] == 1
    assert calls == [((sentinel, mask), {'prompt_length': 101, 'completion_eos_id': 151645})]
