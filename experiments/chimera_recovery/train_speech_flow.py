"""Distinct waveform-supervised speaking driver, full depth and exact resume.

The current endpoint head is differentiated through its canonical velocity
conversion. Genuine H3 clean targets and source-admitted transcript/voice pairs
are mandatory. This module never launches a teacher or acquires a GPU.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import time

try:
    from . import speech_flow as F
except ImportError:
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('chimera_speech_flow_driver_cache', Path(__file__).with_name('speech_flow.py'))
    F = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = F
    spec.loader.exec_module(F)
T, R, Error, HERE = F.T, F.R, F.Error, F.HERE
A = T.load_module(HERE / 'train_audio.py', 'chimera_speech_flow_audio_common')
STAGE, OBJECTIVE, READ_POLICY = F.STAGE, F.OBJECTIVE, F.READ_POLICY
SAFETY = A.SAFETY
OPTIMIZER = {'unit_lr': 2e-5, 'bridge_lr': 2e-5, 'emission_lr': 2e-5, 'lora_lr': 1e-4,
    'betas': [.9, .95], 'eps': 1e-8, 'weight_decay': 0., 'clip': 1., 'warmup_steps': 20, 'min_lr_ratio': .1}


def options(value, *, fixture=False):
    checked = dict(value)
    bridge = checked.pop('bridge_lr', None)
    if type(bridge) not in (float, int) or not math.isfinite(bridge) or bridge <= 0:
        raise Error('speaking optimizer needs an explicit positive bridge rate')
    T.optimizer_config(checked)
    if not fixture and value != OPTIMIZER:
        raise Error('speaking initial optimizer must match the separately selected dated amendment')
    return dict(value)


def validate_inputs(model, segment, target, weights, prompt_length):
    import torch
    ids, types, indices, features, sigma = (segment[key] for key in ('ids', 'types', 'indices', 'features', 'sigma'))
    n, width = types.numel(), model.cfg.native_feature_dim
    flo, fhi = model.bands[3]['feature']
    elo, ehi = model.bands[3]['emission']
    if (type(prompt_length) is not int or prompt_length < 1 or ids.shape != (1, prompt_length + n) or
            ids.dtype != torch.int64 or types.shape != (1, n) or types.dtype != torch.int64 or bool((types != 3).any()) or
            indices.shape != types.shape or indices.dtype != torch.int64 or
            not torch.equal(indices[0], torch.arange(prompt_length, ids.shape[1], device=indices.device)) or
            bool(((ids[:, :prompt_length] < 0) | (ids[:, :prompt_length] >= model.base_vocab)).any()) or
            bool((ids[:, prompt_length:] != model.media_placeholder).any()) or
            features.shape != (1, n, width) or target.shape != features.shape or weights.shape != types.shape or
            features.requires_grad or target.requires_grad or sigma.shape != types.shape or
            bool(((sigma <= 0) | (sigma > 1)).any()) or fhi - flo != ehi - elo or
            not bool((weights == 1).all())):
        raise Error('speaking requires complete native text prefix, positive-sigma type3 noisy inputs and detached emission targets')
    outside_input, outside_target = features.clone(), target.clone()
    outside_input[..., flo:fhi], outside_target[..., elo:ehi] = 0, 0
    source = segment.get('source_positions', {}).get('h3')
    if (bool(outside_input.any()) or bool(outside_target.any()) or source is None or source.shape != (*ids.shape, 3) or
            any(not bool(torch.isfinite(v).all()) for v in (features, target, sigma, source, segment['positions'], weights))):
        raise Error('speaking noisy feature/clean emission allocation or source coordinates invalid')


def sample_latents(engine, segment, *, prompt_length, seed, steps=32):
    """Target-free Euler flow from Gaussian noise, returning model-space audio.

    The input waveform/target is never an argument. Any supplied feature values
    are discarded before the first forward. Physical time/chart and native text
    prefix remain fixed. Actual VAE decode and independent ASR/listening review
    are separate source-pinned diagnostics, not invented here.
    """
    import torch
    if type(seed) is not int or not 0 <= seed < 2 ** 63 or type(steps) is not int or steps != 32:
        raise Error('initial speaking sampler requires explicit seed and amended 32 Euler steps')
    model = engine.hybrid
    sample = {key: value.detach().clone() if hasattr(value, 'detach') else value for key, value in segment.items()}
    sample['source_positions'] = {key: value.detach().clone() for key, value in segment.get('source_positions', {}).items()}
    sample['features'].zero_()
    sample['sigma'] = torch.ones_like(segment['sigma'])
    n = sample['types'].numel()
    flo, fhi = model.bands[3]['feature']
    elo, ehi = model.bands[3]['emission']
    noisy = torch.randn((1, n, fhi - flo), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32).to(model.device)
    sample['features'][..., flo:fhi] = noisy.to(sample['features'].dtype)
    dummy = torch.zeros_like(sample['features'])
    validate_inputs(model, sample, dummy, torch.ones_like(sample['sigma']), prompt_length)
    trajectory = []
    for number in range(steps):
        sigma, next_sigma = 1. - number / steps, 1. - (number + 1) / steps
        sample['sigma'].fill_(sigma)
        sample['features'][..., flo:fhi] = noisy.to(sample['features'].dtype)
        with A.prompt_only_h3(model, engine.placement.container, sample['ids'].shape[1]):
            predicted = model.forward(sample)['emission'][..., elo:ehi].float()
        if not bool(torch.isfinite(predicted).all()):
            raise Error('speaking sampler predicted nonfinite endpoint')
        velocity = (noisy - predicted) / sigma
        noisy = noisy + (next_sigma - sigma) * velocity
        if not bool(torch.isfinite(noisy).all()):
            raise Error('speaking sampler produced nonfinite native latent')
        trajectory.append({'sigma': sigma, 'next_sigma': next_sigma, 'latent_sha256': R.tensor_sha256(noisy.cpu())})
    return noisy.detach(), {'schema': 'chimera_recovery.speech_flow_sample.v1', 'seed': seed, 'steps': steps,
        'integration': 'Euler in canonical dx/dsigma velocity, descending sigma1to0', 'trajectory': trajectory,
        'objective': OBJECTIVE, 'clean_target_supplied_to_sampler': False, 'prediction_kind': 'x0',
        'foreign_codec_ids_used': False, 'decoded_waveform_ready': False, 'native_parent_xyz_parity': False}


class SpeechFlowEngine(T.D3bEngine):
    checkpoint_schema = 'chimera_recovery.speech_flow_checkpoint.v1'

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        self.module, self.hybrid = module, hybrid
        self.options = options(optimizer, fixture=identity.get('fixture') is True)
        self.identity = {**identity, 'stage': STAGE, 'objective': OBJECTIVE, 'read_policy': READ_POLICY,
                         'optimizer_sha256': R.digest(optimizer), 'total_steps': total_steps}
        self.placement = next((p for p in hybrid.layout if p.container == unit and p.spec.source == 'h3' and 3 in p.type_ids), None)
        if self.placement is None or hybrid.head_deltas or hybrid.bridge_deltas or type(total_steps) is not int or total_steps < 1:
            raise Error('speaking needs a fresh genuine H3 unit and exact positive update budget')
        if identity.get('fixture') is not True:
            for key in ('base_sha256', 'batch_manifest_sha256', 'amendment_sha256', 'hybrid_module_sha256'):
                R.require_sha(identity.get(key), 'speaking ' + key)
            if (identity.get('heldout_groups_excluded') is not True or hybrid.base_vocab != 248320 or
                    hybrid.bands[3]['feature'] != (480, 512) or hybrid.bands[3]['emission'] != (224, 256)):
                raise Error('speaking production requires exact current type3 allocations and source-group admission')
        for native_unit in hybrid.units.values():
            native_unit.requires_grad_(False)
        hybrid.unit_parameters([unit])
        bridges, heads = hybrid.add_bridge_deltas([3]), hybrid.add_emission_deltas([3])
        for value in hybrid.bank.tensors.values():
            value.requires_grad_(True)
        self.census = {f'unit.{unit}.{name}': value for name, value in hybrid.units[unit].named_parameters() if value.requires_grad}
        self.census.update({f'bridge.{d.name}': d.delta for d in bridges})
        self.census.update({f'emission.{d.name}': d.delta for d in heads})
        self.census.update({'lora.' + name: value for name, value in sorted(hybrid.bank.tensors.items()) if value.requires_grad})
        groups = [{'name': kind, 'params': [v for k, v in self.census.items() if k.startswith(kind + '.')],
                   'lr': self.options[kind + '_lr'], 'base_lr': self.options[kind + '_lr']}
                  for kind in ('unit', 'bridge', 'emission', 'lora')]
        if any(not group['params'] for group in groups):
            raise Error('speaking ownership requires native unit, input bridge, emission and shared LoRA')
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options['betas']), eps=self.options['eps'], weight_decay=self.options['weight_decay'])
        self.ownership = R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False

    def train_step(self, segment, target, weights, *, prompt_length):
        import torch
        if self.step >= self.total_steps:
            raise Error('speaking exhausted exact one-pass row/sigma budget')
        validate_inputs(self.hybrid, segment, target, weights, prompt_length)
        self.optimizer.zero_grad(set_to_none=True)
        def objective(normed, context):
            prediction = self.hybrid.emission(normed, context)
            return R.flow_velocity_loss(prediction, target, segment['types'], weights, self.hybrid.bands,
                param='x0', target_kind='x0', x_t=segment['features'], sigma=segment['sigma'],
                noisy_feature_bands={3: self.hybrid.bands[3]['feature']})
        try:
            # All sequence slots are generation prompt or noisy audio. This
            # bounded override enforces type0/type3 only, with no target text.
            with A.prompt_only_h3(self.hybrid, self.placement.container, segment['ids'].shape[1]):
                loss, _, _ = self.module.full_depth_forward_backward(self.hybrid, segment, objective)
        except self.module.D3Error as exc:
            if 'not finite' not in str(exc):
                raise
            return A.AudioCEEngine.nonfinite_skip(self)
        missing = [name for name, value in self.census.items() if value.grad is None]
        if missing:
            raise Error('speaking declared trainable received no gradient: ' + repr(missing[:3]))
        if not math.isfinite(loss) or any(not bool(torch.isfinite(v.grad).all()) for v in self.census.values()):
            return A.AudioCEEngine.nonfinite_skip(self)
        norms = {g['name']: float(torch.sqrt(sum(v.grad.float().square().sum() for v in g['params']))) for g in self.optimizer.param_groups}
        if any(v <= 0 for v in norms.values()):
            raise Error('speaking declared optimizer group received zero gradient')
        factor = self.module.trainer_module().lr_factor(self.step, self.options['warmup_steps'], self.total_steps, self.options['min_lr_ratio'])
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options['clip'], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips, self.skip_window = 0, (self.skip_window + [False])[-100:]
        return {'loss': loss, 'step': self.step, 'skipped': False, 'group_norms': norms, 'gradient_norm': float(norm),
                'objective': OBJECTIVE, 'guided_teacher': False, 'speech_codec_ids_used': False}

    def export(self, out):
        raise Error('speaking requires finalized shared-r16 and cumulative native export, not a generic D3b slice')


def batches(path, amendment_sha):
    data = T.read_json(path)
    if (data.get('schema') != F.SCHEMA or data.get('stage') != STAGE or data.get('objective') != OBJECTIVE or
            any(data.get(k) is not True for k in ('sealed', 'complete_source_traversal', 'heldout_groups_excluded')) or
            data.get('amendment_sha256') != amendment_sha or data.get('adapter_sha256') != R.sha(HERE / 'speech_flow.py') or
            data.get('dependencies') != F.dependency_pins() or
            data.get('sigma_grid') != F.SIGMAS or data.get('seed') != F.SEED or data.get('guided_teacher') is not False or
            data.get('read_policy') != READ_POLICY or data.get('prediction_kind') != 'x0' or
            data.get('native_parent_xyz_parity') is not False or data.get('split') != 'train' or not data.get('batches')):
        raise Error('speaking requires complete source-admitted native supervised-flow batches')
    source = T.pinned_file(data['source'], path.parent)
    _, admitted = F.C.admission(data['source_admission'], path.parent, source)
    T.pinned_file(data['clean_manifest'], path.parent)
    T.pinned_file(data['charted_attention'], path.parent)
    seen, source_sigma = set(), set()
    for item in data['batches']:
        if not item.get('id') or item['id'] in seen or item.get('route') != 'speech_generate':
            raise Error('speaking row/sigma identities must be unique exact generation rows')
        seen.add(item['id'])
        T.pinned_file(item['tensors'], path.parent)
        receipt = T.read_json(T.pinned_file(item['teacher_receipt'], path.parent))
        if (receipt.get('id') != item['id'] or receipt.get('source_id') != item.get('source_id') or
                receipt.get('recovery_group_id') != item.get('recovery_group_id') or
                receipt.get('tensors_sha256') != item['tensors']['sha256'] or receipt.get('amendment_sha256') != amendment_sha):
            raise Error('speaking cache manifest/row receipt differs')
        pair = (item['source_id'], receipt.get('sigma'))
        if pair in source_sigma or receipt.get('sigma') not in F.SIGMAS:
            raise Error('speaking source/sigma ownership duplicates or differs')
        source_sigma.add(pair)
    expected = {(ident, sigma) for ident, row in admitted.items() if row['train_admitted'] for sigma in F.SIGMAS}
    excluded = {row['id'] for row in data.get('excluded', [])}
    if source_sigma != expected or excluded != {ident for ident, row in admitted.items() if not row['train_admitted']}:
        raise Error('speaking must cover every admitted source/sigma once and explicitly exclude all holdouts')
    return data


def preflight(config_path):
    config, root = T.read_json(config_path), config_path.parent
    checked, blockers = {}, []
    def check(label, fn):
        try:
            checked[label] = fn()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            blockers.append(label + ': ' + str(exc))
    if config.get('schema') != 'chimera_recovery.speech_flow_stage_config.v1' or config.get('stage') != STAGE:
        blockers.append('explicit distinct D3d_speech_flow config required')
    check('optimizer', lambda: options(config['optimizer']))
    check('preregistration', lambda: T.verify_preregistration(config['preregistration'], root))
    for name, key in (('hybrid', 'hybrid_module'), ('base', 'immutable_base_receipt'), ('parity', 'parity_receipt'), ('runtime', 'runtime_assignment')):
        check(name, lambda key=key: str(T.pinned_file(config[key], root)) if key == 'hybrid_module' else T.read_json(T.pinned_file(config[key], root)))
    try:
        if (config.get('objective') != OBJECTIVE or config.get('read_policy') != READ_POLICY or config.get('after_stage') != 'D3d_audio_ce' or
                config.get('before_stage') != 'D4' or config.get('seed') != F.SEED or config.get('safety') != SAFETY or
                config.get('dtype') not in {'float32', 'bfloat16'} or config.get('sampling_policy') != 'one_pass_row_sigma'):
            raise Error('speaking objective/read/order/seed/safety/sampling selection differs')
        raw = T.pinned_file(config['preregistration'], root).read_bytes().replace(b'\r\n', b'\n')
        tail = raw[raw.find(b'\n## Amendments'):]
        for token in (STAGE, OBJECTIVE, READ_POLICY, F.CHART_POLICY, F.INSERTION_POLICY,
                      'speech_sigma_grid=[0.05,0.3,0.7,1.0]', 'speech_euler_steps=32'):
            if token.encode() not in tail:
                raise Error('dated preregistration has not selected speaking contract: ' + token)
        base = checked['base']
        if (base.get('immutable_clone') is not True or base.get('registered') is not True or
                base.get('verified_equal_to_registered_parent') is not True or base.get('base_sha256') != base.get('clone_sha256') or
                Path(base.get('clone_path', '')) != Path(config['omni'])):
            raise Error('speaking immutable current-base identity differs')
        R.require_sha(base['base_sha256'], 'speaking base_sha256')
        checked['shared_r16'] = T.I.preflight_config(config, root, base['base_sha256'])
        checked['native_state'] = T.N.preflight_config(config, root, checked['shared_r16'])
        producer = T.read_json(T.pinned_file(config['native_state']['source'], root))
        if producer.get('stage') != 'D3d_audio_ce':
            raise Error('speaking must inherit the completed audio CE cumulative state')
        if (checked['parity'].get('verdict') != 'PASS' or checked['parity'].get('base_sha256') != base['base_sha256'] or
                checked['parity'].get('speech_generation_read_policy') != READ_POLICY or
                checked['parity'].get('speech_flow_full_depth_gradients') is not True):
            raise Error('speaking needs actual current-base amended full-depth/read-policy parity')
        if checked['runtime'].get('active') is not True or checked['runtime'].get('device') != config.get('device'):
            raise Error('speaking requires active exclusive allocator assignment')
        if (config.get('lora', {}).get('rank') != 16 or config['lora'].get('alpha') != 16 or
                config['lora'].get('k3_rank') != 0 or config['lora'].get('gain') is not False):
            raise Error('speaking requires the existing shared r16/alpha16/noK3/no gain bank')
        checked['batches_path'] = str(T.pinned_file(config['batches_manifest'], root))
        checked['batches'] = batches(Path(checked['batches_path']), checked['preregistration']['amendment_sha256'])
        if type(config.get('steps')) is not int or config['steps'] != len(checked['batches']['batches']):
            raise Error('speaking budget must equal the admitted row/sigma inventory once')
    except (KeyError, ValueError, TypeError, OSError) as exc:
        blockers.append(str(exc))
    return {'schema': 'chimera_recovery.speech_flow_preflight.v1', 'stage': STAGE, 'production_ready': not blockers,
        'training_started': False, 'blockers': blockers, 'config_sha256': R.sha(config_path), 'config': config,
        'verified': checked, 'driver_sha256': R.sha(Path(__file__)), 'native_parent_xyz_parity': False, 'guided_teacher': False}


def train(config_path, out, resume=None):
    result = preflight(config_path)
    if not result['production_ready']:
        raise Error('speaking preflight blocked: ' + '; '.join(result['blockers']))
    if out.exists():
        raise Error('preserve speaking run; use fresh output and explicit exact checkpoint resume')
    import torch
    config, checked = result['config'], result['verified']
    module = T.load_module(Path(checked['hybrid']), 'd3_hybrid_speech_production')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, trunk_ffn_width=config['lora']['trunk_ffn_width'], gain=False)
    hybrid = module.HybridD3(config['omni'], config['parent_config'], module.HybridOptions(dtype=getattr(torch, config['dtype']),
        device=config['device'], kernels=config['kernels'], lora=plan, lora_seed=config['seed'], plan=config.get('compose_plan')))
    shared = T.I.initialize_stage(hybrid, checked['shared_r16'], config_path.parent)
    native = T.N.initialize_stage(hybrid, checked['native_state'], config_path.parent)
    identity = {'stage': STAGE, 'objective': OBJECTIVE, 'read_policy': READ_POLICY, 'config_sha256': result['config_sha256'],
        'base_sha256': checked['base']['base_sha256'], 'batch_manifest_sha256': config['batches_manifest']['sha256'],
        'amendment_sha256': checked['preregistration']['amendment_sha256'], 'hybrid_module_sha256': config['hybrid_module']['sha256'],
        'driver_sha256': result['driver_sha256'], 'cache_adapter_sha256': R.sha(HERE / 'speech_flow.py'),
        'heldout_groups_excluded': True, 'shared_r16': shared, 'native_state': native}
    engine = SpeechFlowEngine(module, hybrid, config['unit'], config['optimizer'], identity, total_steps=config['steps'])
    # Shared guard binds the complete identity and accepted review; its contents
    # remain stage-specific, and the engine checkpoint schema is distinct.
    accepted_review = A.resume_audio_checkpoint(engine, resume) if resume else None
    out.mkdir(parents=True)
    (out / 'preflight.json').write_bytes(R.canonical(result) + b'\n')
    A.save_audio_checkpoint(engine, out / 'checkpoint-initial', accepted_review)
    (out / 'initial_receipt.json').write_bytes(R.canonical({'schema': 'chimera_recovery.speech_flow_initial.v1',
        'identity': engine.identity, 'trainable_census': engine.ownership, 'step': engine.step, 'sampler_counter': engine.sampler_counter}) + b'\n')
    items, root = checked['batches']['batches'], Path(checked['batches_path']).parent
    last_safety = time.monotonic()
    while engine.step < engine.total_steps:
        quarter = max(1, math.ceil(engine.total_steps * .25))
        if engine.step >= quarter and not engine.decode_review_passed:
            review_path = config.get('speech_decode_review')
            path = Path(review_path) if review_path else None
            path = path if path is None or path.is_absolute() else config_path.parent / path
            review = T.read_json(path) if path and path.is_file() else {}
            if (review.get('schema') != 'chimera_recovery.speech_flow_decode_review.v1' or review.get('passed') is not True or
                    review.get('run_identity_sha256') != R.digest(engine.identity) or not quarter <= review.get('trained_step', -1) <= engine.step or
                    review.get('clean_target_supplied_to_sampler') is not False or review.get('heldout_only') is not True or
                    review.get('actual_h3_waveform_decode') is not True or not review.get('samples')):
                A.save_audio_checkpoint(engine, out / f'checkpoint-speech-review-{engine.step:08d}', accepted_review)
                return {'stage': STAGE, 'status': 'awaiting_held_speech_sampling_review', 'step': engine.step, 'training_complete': False}
            engine.decode_review_passed, accepted_review = True, review
        if engine.sampler_counter >= len(items):
            raise Error('speaking one-pass row/sigma sampler exhausted early')
        item = items[engine.sampler_counter]
        receipt = T.read_json(T.pinned_file(item['teacher_receipt'], root))
        cpu = T.load_native_batch(item, root, 'cpu')
        try:
            F.validate_batch(cpu, receipt, root)
            batch = {key: value.to(config['device']) for key, value in cpu.items()}
            segment = T.make_segment(batch, media_placeholder=hybrid.media_placeholder)
            step = engine.train_step(segment, batch['native_media_targets'], batch['native_media_target_mask'], prompt_length=receipt['prompt_length'])
        except Exception:
            A.save_audio_checkpoint(engine, out / f'checkpoint-failure-{engine.step:08d}', accepted_review)
            raise
        with (out / 'steps.jsonl').open('ab') as stream:
            stream.write(R.canonical(step) + b'\n')
        if step.get('halt') is True:
            A.save_audio_checkpoint(engine, out / f'checkpoint-skip-halt-{engine.step:08d}', accepted_review)
            raise Error('speaking persistent nonfinite skips require a ruling')
        if time.monotonic() - last_safety >= 1800:
            A.save_audio_checkpoint(engine, out / f'checkpoint-safety-{engine.step:08d}', accepted_review)
            T.prune_safety_checkpoints(out, keep=2)
            last_safety = time.monotonic()
    A.save_audio_checkpoint(engine, out / 'checkpoint-final', accepted_review)
    bank = T.I.export_from_engine(engine, out / 'checkpoint-final', out / 'shared-r16', checked['shared_r16']['context'], STAGE, config['unit'])
    native = T.N.export_from_engine(engine, out / 'checkpoint-final', out / 'native-state', checked['shared_r16']['context'], STAGE, bank)
    return {'schema': 'chimera_recovery.speech_flow_run.v1', 'stage': STAGE, 'training_complete': True, 'step': engine.step,
        'registered': False, 'shared_r16': bank, 'native_state': native, 'guided_teacher': False, 'native_parent_xyz_parity': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'train'):
        command = sub.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'train':
            command.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.config) if args.command == 'preflight' else train(args.config, args.out, args.resume)
    if args.command == 'preflight':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(R.canonical(result) + b'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in {'config', 'verified'}}))
    return 0 if result.get('production_ready', result.get('training_complete', False)) else 2


if __name__ == '__main__':
    raise SystemExit(main())
