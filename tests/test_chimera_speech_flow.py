"""Separate speaking condition, sealed noise, loss/ownership and exact resume."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('speech_flow_test', ROOT / 'experiments/chimera_recovery/train_speech_flow.py')
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)
F = S.F

def row():
    return {'id': 'speech-fixture', 'route': 'speech_understand', 'split': 'train',
        'messages': [{'role': 'user', 'content': 'Transcribe the audio.'}, {'role': 'assistant', 'content': 'The boat crossed the lake.'}],
        'supervision': {'message_mask': [False, True]},
        'provenance': {'speaker': 'Ryan', 'independent_asr_exact_match': True,
                       'synthetic_official_voice': True, 'user_reference_voice_used': False}}

def tokenizer():
    # Deliberate protocol fixture; no actual-parent claim.
    spec = importlib.util.spec_from_file_location('speech_prefix_tokenizer_fixture', ROOT / 'tests/test_chimera_recovery_audio_dialogue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Tokenizer()

def test_exact_generation_prefix_distinct_role_and_source_semantics():
    source, native = row(), tokenizer()
    before = copy.deepcopy(source)
    packed = F.pack_prefix(source, native, media_rows=80, max_length=2048)
    assert source == before and packed['speaker'] == 'Ryan' and packed['transcript'] == before['messages'][-1]['content']
    assert packed['input_ids'][:packed['prompt_length']] == packed['parent_text_ids']
    assert packed['media_indices'][0] == packed['prompt_length']
    assert all(v == F.D.PLACEHOLDER for v in packed['input_ids'][packed['prompt_length']:])
    assert packed['messages'][0]['role'] == 'user' and 'synthetic Ryan voice' in packed['messages'][0]['content']
    assert 'assistant_target_mask' not in packed and packed['policy'] != F.D.POLICY
    assert F.noise_seed(F.R.digest(source), .3) == F.noise_seed(F.R.digest(source), .3)
    assert F.noise_seed(F.R.digest(source), .3) != F.noise_seed(F.R.digest(source), .7)
    with pytest.raises(F.Error, match='no waveform'):
        F.pack_prefix(source, native, media_rows=80, max_length=5)

@pytest.mark.parametrize('field,value', [('speaker', 'unverified-person'), ('independent_asr_exact_match', False),
                                       ('synthetic_official_voice', False), ('user_reference_voice_used', True)])
def test_unsealed_voice_or_transcript_cannot_become_speaking_condition(field, value):
    source = row()
    source['provenance'][field] = value
    with pytest.raises(F.Error, match='source-sealed speech'):
        F.generation_prompt(source)

def test_positive_sigma_noise_target_separation_and_source_positions(tmp_path):
    torch = pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('speech_flow_encoder_shape_fixture', ROOT / 'tests/test_chimera_audio_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, proof = row(), module.proof(167)
    prefix = F.pack_prefix(source, tokenizer(), media_rows=334, max_length=2048)
    clean = torch.randn(334, 32, generator=torch.Generator().manual_seed(31))
    kwargs = dict(proof_sha='a' * 64, amendment_sha='b' * 64, charted_attention_sha='c' * 64)
    batch, chart, seed = F.build_batch(source, prefix, proof, clean, sigma=.3, **kwargs)
    x_t, target, noise = batch['native_media_features'][..., 480:512], batch['native_media_targets'][..., 224:256], batch['flow_noise']
    assert torch.equal(x_t, .7 * target + .3 * noise)
    assert torch.equal(target[0], clean) and not torch.equal(x_t, target)
    assert not batch['native_media_features'][..., :480].any() and not batch['native_media_features'][..., 512:].any()
    assert chart['read_policy'] == F.READ_POLICY and 'clean_sigma' not in chart
    assert chart['native_parent_xyz_parity'] is False and chart['latent_frames_per_channel'] == 167
    compact, storage = F.T.C.compact_native_batch(batch, {3: {'feature': (480, 512), 'emission': (224, 256)}})
    assert storage == F.STORAGE and 'native_media_features' not in compact and 'native_media_targets' not in compact
    inflated = F.T.C.inflate_native_batch(compact, storage)
    assert set(inflated) == set(batch) and all(torch.equal(inflated[key], value) for key, value in batch.items())
    full_noise, _, same_seed = F.build_batch(source, prefix, proof, clean + 9., sigma=1., **kwargs)
    earlier, _, _ = F.build_batch(source, prefix, proof, clean, sigma=1., **kwargs)
    assert torch.equal(full_noise['native_media_features'], earlier['native_media_features'])
    assert torch.equal(full_noise['charted_source_positions'], earlier['charted_source_positions'])
    assert same_seed != seed
    with pytest.raises(F.Error, match='positive sigma'):
        F.build_batch(source, prefix, proof, clean, sigma=0., **kwargs)

def setup(tmp_path, *, native_inheritance=False):
    torch = pytest.importorskip('torch')
    source = os.environ.get('CHIMERA_D3_BASELINE')
    if not source:
        pytest.skip('set CHIMERA_D3_BASELINE for genuine native tiny speech fixture')
    foundation = os.environ.get('CHIMERA_OMNI_SRC')
    if foundation:
        sys.path.insert(0, str(Path(foundation) / 'src'))
    torch.set_num_threads(1)
    module = S.T.load_module(Path(source), 'd3_hybrid_speech_fixture')
    # The baseline toy AV allocation reserves four audio feature columns for
    # two emission columns. This standalone speaking fixture explicitly declares
    # a two-channel native latent at feature20:22 and emission10:12; reserved
    # feature22:24 stays unused. The genuine importer, units and full-depth
    # engine remain unchanged. Production still requires exact32/32 allocation.
    module.TINY_BANDS = dict(module.TINY_BANDS)
    module.TINY_BANDS[3] = ((20, 22), (10, 12))
    built = module.build_tiny_child(tmp_path / 'tiny-base', seed=17)
    optimizer = {**S.OPTIMIZER, 'unit_lr': .001, 'bridge_lr': .001, 'emission_lr': .001, 'lora_lr': .001,
                 'warmup_steps': 0, 'min_lr_ratio': 1.}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        model = module.HybridD3(built['dir'], built['parent_config'], module.HybridOptions(dtype=torch.float32, device='cpu', kernels='torch',
            lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        if native_inheritance:
            context = {'fixture': True, 'base_sha256': 'a' * 64, 'parent_config_sha256': 'b' * 64, 'hybrid_module_sha256': 'c' * 64}
            S.T.I.initialize_stage(model, {'mode': 'initialize', 'context': context}, tmp_path)
            S.T.N.initialize_stage(model, {'mode': 'initialize', 'context': context}, tmp_path)
        return S.SpeechFlowEngine(module, model, 'tiny_h3', optimizer, {'fixture': True, 'seed': 17}, total_steps=3)
    engine = construct()
    segment = module.tiny_segment(engine.hybrid.cfg, 'h3', prompt=6, rows=6, seed=8)
    segment['types'].fill_(3)
    segment['features'].zero_()
    flo, fhi = engine.hybrid.bands[3]['feature']
    elo, ehi = engine.hybrid.bands[3]['emission']
    clean = torch.randn(1, 6, fhi - flo, generator=torch.Generator().manual_seed(33))
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(35))
    segment['features'][..., flo:fhi] = .7 * clean + .3 * noise
    target = torch.zeros_like(segment['features'])
    target[..., elo:ehi] = clean
    segment['sigma'].fill_(.3)
    # Mathematical fixture only: actual production charts require genuine proof.
    segment['source_positions'] = {'h3': torch.zeros(1, segment['ids'].shape[1], 3)}
    weights = torch.ones(1, 6)
    return torch, module, engine, construct, segment, target, weights

def test_speaking_full_depth_matches_independent_eager_velocity_gradient(tmp_path):
    torch, module, engine, _, segment, target, weights = setup(tmp_path)
    model = engine.hybrid
    context = model.prepare(segment, media_grad=True)
    x, contexts = context['x'], model.contexts(context['x'])
    with S.A.prompt_only_h3(model, engine.placement.container, segment['ids'].shape[1]):
        for index in range(model.num_layers):
            layer = copy.deepcopy(model.engine.layer(index, index + 1 if index + 1 < model.num_layers else None))
            model.bank.bind(layer, index)
            x = model.engine._run(layer, index, x, contexts)
            x = model.apply_units(x, index, context)
    emission = model.emission(model.final_norm(x), context)
    lo, hi = model.bands[3]['emission']
    eager = ((emission[..., lo:hi].float() - target[..., lo:hi]) / segment['sigma'].unsqueeze(-1)).square().mean()
    eager.backward()
    expected = {k: v.grad.clone() for k, v in engine.census.items()}
    engine.optimizer.zero_grad(set_to_none=True)
    def objective(normed, ctx):
        return S.R.flow_velocity_loss(model.emission(normed, ctx), target, segment['types'], weights, model.bands,
            x_t=segment['features'], sigma=segment['sigma'], noisy_feature_bands={3: model.bands[3]['feature']})
    with S.A.prompt_only_h3(model, engine.placement.container, segment['ids'].shape[1]):
        loss, _, _ = module.full_depth_forward_backward(model, segment, objective)
    assert abs(loss - float(eager.detach())) < 1e-5
    for name, value in engine.census.items():
        assert torch.allclose(value.grad, expected[name], atol=2e-5, rtol=2e-4), name

def test_speaking_native_ownership_update_exact_resume_and_no_foreign_code(tmp_path):
    torch, _, engine, construct, segment, target, weights = setup(tmp_path)
    before = {k: v.detach().clone() for k, v in engine.census.items()}
    result = engine.train_step(segment, target, weights, prompt_length=6)
    assert result['guided_teacher'] is False and result['speech_codec_ids_used'] is False
    for group in ('unit', 'bridge', 'emission', 'lora'):
        assert result['group_norms'][group] > 0
        assert any(not torch.equal(v, before[k]) for k, v in engine.census.items() if k.startswith(group + '.'))
    S.A.save_audio_checkpoint(engine, tmp_path / 'interrupted')
    expected = engine.train_step(segment, target, weights, prompt_length=6)
    resumed = construct()
    S.A.resume_audio_checkpoint(resumed, tmp_path / 'interrupted')
    actual = resumed.train_step(segment, target, weights, prompt_length=6)
    assert actual == expected and all(torch.equal(v, resumed.census[k]) for k, v in engine.census.items())
    segment['ids'][0, 0] = engine.hybrid.base_vocab + 1
    with pytest.raises(S.Error, match='complete native text prefix'):
        engine.train_step(segment, target, weights, prompt_length=6)

def test_speaking_preflight_refuses_missing_real_sources_and_selected_amendment(tmp_path):
    config = tmp_path / 'missing.json'
    config.write_text('{"schema":"chimera_recovery.speech_flow_stage_config.v1","stage":"D3d_speech_flow"}')
    result = S.preflight(config)
    assert result['production_ready'] is False and result['training_started'] is False
    assert any('preregistration' in b for b in result['blockers']) and any('optimizer' in b for b in result['blockers'])


def test_speaking_sampler_discards_any_input_feature_values_and_never_accepts_target(tmp_path):
    torch, _, engine, _, segment, _, _ = setup(tmp_path)
    before = segment['features'].clone()
    latent, receipt = S.sample_latents(engine, segment, prompt_length=6, seed=7)
    segment['features'].fill_(91.)
    changed, same = S.sample_latents(engine, segment, prompt_length=6, seed=7)
    assert torch.equal(latent, changed) and receipt == same
    assert receipt['clean_target_supplied_to_sampler'] is False and receipt['decoded_waveform_ready'] is False
    assert len(receipt['trajectory']) == 32 and receipt['trajectory'][-1]['next_sigma'] == 0.
    assert latent.shape == (1, 6, engine.hybrid.bands[3]['feature'][1] - engine.hybrid.bands[3]['feature'][0])
    with pytest.raises(TypeError):
        S.sample_latents(engine, segment, prompt_length=6, seed=7, clean_target=before)


def test_speaking_cumulative_native_adapter_keeps_both_interface_gradients_and_resume(tmp_path):
    torch, _, engine, construct, segment, target, weights = setup(tmp_path, native_inheritance=True)
    for group in ('bridge', 'emission'):
        values = [v for k, v in engine.census.items() if k.startswith(group + '.')]
        assert values and all(v.is_leaf and v.dtype == torch.float32 for v in values)
    first = engine.train_step(segment, target, weights, prompt_length=6)
    assert all(first['group_norms'][g] > 0 for g in ('unit', 'bridge', 'emission', 'lora'))
    engine.save(tmp_path / 'native-interrupted')
    engine.train_step(segment, target, weights, prompt_length=6)
    resumed = construct()
    resumed.resume(tmp_path / 'native-interrupted')
    resumed.train_step(segment, target, weights, prompt_length=6)
    assert all(torch.equal(v, resumed.census[k]) for k, v in engine.census.items())
