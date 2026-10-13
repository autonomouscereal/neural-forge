"""Independently reconstruct sealed audio/speech batches before replay indexing.

The source-pinned producer adapters are explicit inputs. This CPU audit never
loads Chimera weights, creates an optimizer, or grants production-state parity.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

SCHEMA = 'chimera_recovery.audio_speech_replay_index.v1'
CE = 'authored_audio_assistant_ce'
FLOW = 'supervised_speech_rectified_flow'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def coverage(admitted, ce_entries, flow_entries, sigma_grid):
    """Reject dropped, duplicated, held or newly excluded replay identities."""
    expected_ce = {ident for ident, row in admitted.items() if row['train_admitted']}
    ce_ids = [row['source_id'] for row in ce_entries]
    expected_flow = {(ident, sigma) for ident, row in admitted.items()
        if row['train_admitted'] and row['route'] == 'speech_understand' for sigma in sigma_grid}
    flow_pairs = [(row['source_id'], row['sigma']) for row in flow_entries]
    if len(ce_ids) != len(set(ce_ids)) or set(ce_ids) != expected_ce:
        raise ValueError('CE replay must cover every globally admitted source exactly once')
    if len(flow_pairs) != len(set(flow_pairs)) or set(flow_pairs) != expected_flow:
        raise ValueError('speech replay must cover every globally admitted source/sigma exactly once')
    if any(type(row.get('supervised_units')) not in (int, float) or
            not math.isfinite(row['supervised_units']) or row['supervised_units'] <= 0
            for row in ce_entries + flow_entries):
        raise ValueError('replay requires measured positive supervised units')
    return True


def flow_units(batch):
    import torch
    types, weights = batch['native_media_type_ids'], batch['native_media_target_mask']
    if (types.dtype != torch.int64 or types.ndim != 2 or types.shape[0] != 1 or
            weights.shape != types.shape or types.numel() < 2 or types.numel() % 2 or
            bool((types != 3).any()) or not bool(torch.isfinite(weights).all()) or
            not bool((weights == 1).all())):
        raise ValueError('speech replay units require complete stereo type3 rows with actual unit weights')
    return int(weights.sum()) * 32


def audit(ce_manifest, ce_sha, ce_source_dir, flow_manifest, flow_sha, flow_source_dir, amendment_sha, out):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or out.exists():
        raise ValueError('native replay audit requires CUDA-hidden CPU and fresh output')
    if sha(ce_manifest) != ce_sha or sha(flow_manifest) != flow_sha:
        raise ValueError('native replay manifest pin differs')
    import torch
    torch.set_num_threads(8)
    A = load(ce_source_dir / 'train_audio.py', 'actual_joint_audio_replay_driver')
    C = A.T.load_module(ce_source_dir / 'audio_cache.py', 'actual_joint_audio_replay_cache')
    W = load(flow_source_dir / 'train_speech_flow.py', 'actual_speech_replay_driver')
    F = W.F
    ce = A.audio_batches(ce_manifest, amendment_sha)
    flow = W.batches(flow_manifest, amendment_sha)
    source = A.T.pinned_file(ce['source'], ce_manifest.parent)
    _, admitted = C.admission(ce['source_admission'], ce_manifest.parent, source)
    records = C.source_rows(source)
    excluded = [item['id'] for item in ce.get('excluded', [])]
    if (len(excluded) != len(set(excluded)) or
            set(excluded) != {ident for ident, row in admitted.items() if not row['train_admitted']}):
        raise ValueError('joint CE must explicitly retain every excluded source identity')
    flow_source = W.T.pinned_file(flow['source'], flow_manifest.parent)
    flow_records = F.C.source_rows(flow_source)
    if (any(ident not in records or C.R.digest(record) != C.R.digest(records[ident])
            for ident, record in flow_records.items()) or
            {ident for ident, row in admitted.items() if row['route'] == 'speech_understand'} != set(flow_records)):
        raise ValueError('separate speaking source differs from complete joint source identities')
    clean_path = A.T.pinned_file(ce['clean_manifest'], ce_manifest.parent)
    clean = A.T.read_json(clean_path)
    if (clean.get('schema') != 'chimera_recovery.clean_audio_cache.v1' or clean.get('sealed') is not True or
            clean.get('complete_source_traversal') is not True or
            A.T.pinned_file(clean['source'], clean_path.parent) != source or
            len(clean['entries']) != len(records) or {item['id'] for item in clean['entries']} != set(records)):
        raise ValueError('joint clean cache must retain every source including held diagnostics')
    proofs = []
    for item in clean['entries']:
        record = records[item['id']]
        if item.get('source_row_sha256') != C.R.digest(record):
            raise ValueError('clean source row identity changed')
        proof_path, proof, tensors = C.verified_probe(item['encoder_proof'], clean_path.parent, record)
        C.load_latent(proof, tensors)
        proofs.append({'id': item['id'], 'encoder_proof': pin(proof_path), 'source_tensors': pin(tensors),
            'raw_shape': proof['measurement']['raw_latent_shape'], 'model_space_verified': True,
            'split': admitted[item['id']]['split'], 'train_admitted': admitted[item['id']]['train_admitted']})
        if len(proofs) % 32 == 0:
            print(json.dumps({'phase': 'encoder_tensor_reconstruction', 'verified_rows': len(proofs)}), flush=True)
    ce_entries, flow_entries = [], []
    text_vocabulary_contract = SimpleNamespace(base_vocab=248320)
    for item in ce['batches']:
        receipt = A.T.read_json(A.T.pinned_file(item['teacher_receipt'], ce_manifest.parent))
        batch = A.T.load_native_batch(item, ce_manifest.parent, 'cpu')
        C.validate_batch(batch, receipt, ce_manifest.parent)
        segment = A.T.make_segment(batch, media_placeholder=C.D.PLACEHOLDER)
        targets, count = A.assistant_ce_targets(text_vocabulary_contract, segment, batch['assistant_target_mask'],
            receipt['prompt_length'], receipt['completion_eos_id'])
        if count != receipt['assistant_targets'] or int(targets['ce_w'].sum()) != count:
            raise ValueError('recorded CE denominator differs from authentic next-token ownership')
        ident = item['id']
        ce_entries.append({**item, 'objective': CE, 'source_id': ident, 'source_row_sha256': C.R.digest(records[ident]),
            'source_manifest': pin(ce_manifest), 'source_admission': ce['source_admission'],
            'union_recovery_group_id': admitted[ident]['recovery_group_id'],
            'supervised_units': count, 'denominator_kind': 'assistant_next_token_positions_including_completion_eos',
            'prompt_length': receipt['prompt_length'], 'input_ids_length': batch['input_ids'].numel(),
            'native_audio_rows': batch['native_media_type_ids'].numel(),
            'latent_frames_per_channel': batch['native_media_type_ids'].numel() // 2,
            'amendment_sha256': amendment_sha, 'source_reconstruction_bitwise_equal': True})
        if len(ce_entries) % 16 == 0:
            print(json.dumps({'phase': 'native_ce_reconstruction', 'verified_rows': len(ce_entries)}), flush=True)
    for item in flow['batches']:
        receipt = W.T.read_json(W.T.pinned_file(item['teacher_receipt'], flow_manifest.parent))
        batch = W.T.load_native_batch(item, flow_manifest.parent, 'cpu')
        F.validate_batch(batch, receipt, flow_manifest.parent)
        ident = item['source_id']
        if admitted[ident]['train_admitted'] is not True:
            raise ValueError('earlier speaking batch was excluded by the final global source union')
        count = flow_units(batch)
        flow_entries.append({**item, 'objective': FLOW, 'source_row_sha256': C.R.digest(records[ident]),
            'source_manifest': pin(flow_manifest), 'source_admission': flow['source_admission'],
            'union_source_admission': ce['source_admission'],
            'union_recovery_group_id': admitted[ident]['recovery_group_id'], 'sigma': receipt['sigma'],
            'noise_seed': receipt['noise_seed'], 'supervised_units': count,
            'denominator_kind': 'native_weighted_audio_rows_times32_emission_channels',
            'prompt_length': receipt['prompt_length'], 'input_ids_length': batch['input_ids'].numel(),
            'native_audio_rows': batch['native_media_type_ids'].numel(),
            'latent_frames_per_channel': batch['native_media_type_ids'].numel() // 2,
            'amendment_sha256': amendment_sha, 'source_reconstruction_bitwise_equal': True})
        if len(flow_entries) % 16 == 0:
            print(json.dumps({'phase': 'native_speech_flow_reconstruction', 'verified_rows': len(flow_entries)}), flush=True)
    coverage(admitted, ce_entries, flow_entries, F.SIGMAS)
    if torch.cuda.is_initialized():
        raise ValueError('CUDA initialized during CPU-only replay reconstruction')
    result = {'schema': SCHEMA, 'sealed': True, 'passed': True, 'verified': True, 'complete_source_traversal': True,
        'source_manifests': {'audio_ce': pin(ce_manifest), 'speech_flow': pin(flow_manifest), 'joint_clean': pin(clean_path)},
        'source_admissions': {'joint': ce['source_admission'], 'speech': flow['source_admission']},
        'source': ce['source'], 'amendment_sha256': amendment_sha, 'proofs': proofs,
        'entries': ce_entries + flow_entries, 'objectives': {
            CE: {'batches': len(ce_entries), 'supervised_units': sum(row['supervised_units'] for row in ce_entries)},
            FLOW: {'batches': len(flow_entries), 'supervised_units': sum(row['supervised_units'] for row in flow_entries)}},
        'held_source_rows': len(records) - len(ce_entries), 'source_reconstruction_bitwise_equal': True,
        'ce_adapter': pin(ce_source_dir / 'audio_cache.py'), 'flow_adapter': pin(flow_source_dir / 'speech_flow.py'),
        'audit_adapter': pin(Path(__file__)), 'model_forward_performed': False, 'training_started': False,
        'gpu_exposed': False, 'production_training_ready': False,
        'remaining': ['accepted preceding native/shared stage state', 'actual current-base full-depth objective parity',
            'exclusive training runtime', 'held free-answer/speech decode reviews during training']}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name('.' + out.name + '.staging')
    if temporary.exists():
        raise ValueError('replay audit staging path already exists')
    temporary.write_bytes(C.R.canonical(result) + b'\n')
    temporary.rename(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ce-manifest', 'ce-source-dir', 'flow-manifest', 'flow-source-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('ce-sha256', 'flow-sha256', 'amendment-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = audit(args.ce_manifest, args.ce_sha256, args.ce_source_dir, args.flow_manifest,
        args.flow_sha256, args.flow_source_dir, args.amendment_sha256, args.out)
    print(json.dumps({'sealed': result['sealed'], 'objectives': result['objectives'], 'training_started': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
