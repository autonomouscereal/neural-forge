"""Actual tiny H3/text generation; target independence and bounded EOS behavior."""
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

S = load(HERE.parent / 'experiments/chimera_recovery/audio_sampling.py', 'audio_sampling_test')
F = load(HERE / 'test_chimera_recovery_train_audio.py', 'audio_sampling_native_fixture')


def fixture(tmp_path):
    torch, _, engine, _, segment, _, kwargs = F.setup(tmp_path)
    prefix = kwargs['prompt_length']
    media = set(segment['indices'][0].tolist())
    text = [i for i in range(segment['ids'].shape[1]) if i not in media]
    segment['source_positions']['h3'][0, text, 0] = torch.arange(len(text)).float()
    return torch, engine, segment, prefix


def test_genuine_free_answer_ignores_reference_tail_and_never_updates_parameters(tmp_path):
    torch, engine, segment, prefix = fixture(tmp_path)
    before = {name: value.clone() for name, value in engine.census.items()}
    ids, features, xyz = segment['ids'].clone(), segment['features'].clone(), segment['source_positions']['h3'].clone()
    one = S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=7, max_new_tokens=3)
    # These invalid reference-tail values would fail a full native forward.
    # They are discarded entirely before this independent inference call.
    segment['ids'][0, prefix:] = -999
    segment['source_positions']['h3'][:, prefix:] = float('nan')
    two = S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=7, max_new_tokens=3)
    assert one == two and one['teacher_forced'] is False and one['review_pass_granted'] is False
    assert one['generated_ids'] and all(0 <= value < engine.hybrid.base_vocab for value in one['generated_ids'])
    assert all(torch.equal(value, before[name]) and value.grad is None for name, value in engine.census.items())
    assert engine.step == engine.sampler_counter == 0 and 'unit_write' not in engine.hybrid.__dict__
    assert torch.equal(segment['ids'][:, :prefix], ids[:, :prefix]) and torch.equal(segment['features'], features)
    assert torch.equal(segment['source_positions']['h3'][:, :prefix], xyz[:, :prefix])


def test_native_eos_stops_first_token_and_explicit_budget_stops_without_eos(tmp_path):
    torch, engine, segment, prefix = fixture(tmp_path)
    # Authentic fixture heads are read-only memory maps. Replace only this
    # mathematical fixture's attribute; never write into the mapped base.
    engine.hybrid.engine.head_weight = torch.zeros_like(engine.hybrid.engine.head_weight)
    eos = S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=0, max_new_tokens=5)
    assert eos['generated_ids'] == [0] and eos['stopped_on_eos'] is True
    bounded = S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=1, max_new_tokens=2)
    assert bounded['generated_ids'] == [0, 0] and bounded['stopped_on_eos'] is False


def test_nonfinite_native_logits_refuse_and_restore_bounded_joint_read_override(tmp_path):
    torch, engine, segment, prefix = fixture(tmp_path)
    engine.hybrid.engine.head_weight = engine.hybrid.engine.head_weight.clone()
    with torch.no_grad():engine.hybrid.engine.head_weight[0].fill_(float('nan'))
    with pytest.raises(S.Error, match='nonfinite'):
        S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=7, max_new_tokens=1)
    assert 'unit_write' not in engine.hybrid.__dict__ and engine.step == 0


def test_sampling_rejects_missing_clean_observation_or_unamended_text_positions(tmp_path):
    _, engine, segment, prefix = fixture(tmp_path)
    segment['sigma'][0, 0] = .1
    with pytest.raises(S.Error, match='clean type3'):
        S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=7, max_new_tokens=1)
    segment['sigma'].zero_()
    segment['source_positions']['h3'][0, 0, 0] = 99
    with pytest.raises(S.Error, match='ordinal chart'):
        S.sample_answer(engine, segment, prompt_length=prefix, completion_eos_id=7, max_new_tokens=1)
