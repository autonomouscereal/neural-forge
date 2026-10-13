"""Sealed actual H3 clean observations -> native assistant CE batches.

Encoding is an explicit CPU operation; preparation never loads model weights.
Source admission, exact actual encoder proof, parent text and the amended chart
are independent requirements. No foreign codec or generated answer is inferred.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
T = load('train_media.py', 'chimera_recovery_audio_cache_common')
D = load('audio_dialogue.py', 'chimera_recovery_audio_cache_dialogue')
H = load('h3_audio_chart.py', 'chimera_recovery_audio_cache_chart')
P = load('h3_audio_probe.py', 'chimera_recovery_audio_cache_probe')
R, Error = T.R, T.Error
STAGE = 'D3d_audio_ce'
SCHEMA = 'chimera_recovery.audio_batches.v1'

def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': R.sha(path)}

def source_rows(source):
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [row for row in rows if row.get('route') in {'audio_understand', 'speech_understand'}]
    if not rows or len({row.get('id') for row in rows}) != len(rows):
        raise Error('audio source needs unique actual audio/speech understanding rows')
    return {row['id']: row for row in rows}

def admission(entry, root, source):
    path = T.pinned_file(entry, root)
    manifest = T.read_json(path)
    if (manifest.get('schema') != 'recovery_audio_admission.v1' or manifest.get('sealed') is not True or
            any(manifest.get(key) is not True for key in ('complete_source_traversal', 'frozen_exclusions_verified', 'connected_groups_verified'))):
        raise Error('audio needs complete connected, globally frozen-excluded sealed source admission')
    sources = manifest.get('sources', {})
    # Verify actual source files, including exclusion evidence; flags alone do not seal them.
    if not {'source', 'eval_rows', 'normalized', 'frozen_exclusions'}.issubset(sources):
        raise Error('audio admission must pin source, actual frozen eval and original normalized registry')
    checked = {name: T.pinned_file(ref, path.parent) for name, ref in sources.items()}
    if R.sha(checked['source']) != R.sha(source):
        raise Error('audio admission source bytes differ')
    rows_path = T.pinned_file(manifest['outputs']['admission'], path.parent)
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if 'source_path_binding' in manifest:
        binding = manifest['source_path_binding']
        paths = load('audio_paths.py', 'chimera_audio_verify_paths')
        original = T.read_json(T.pinned_file(binding['original_manifest'], path.parent))
        old_path = T.pinned_file(binding['original_admission'], path.parent)
        old_rows = [json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if (binding.get('policy') != paths.POLICY or binding.get('only_paths_rebound') is not True or
                binding.get('original_admission', {}).get('sha256') != original['outputs']['admission']['sha256'] or
                paths.projection(rows) != paths.projection(old_rows) or
                R.digest(paths.projection(rows)) != binding.get('semantic_rows_sha256') or
                {name: ref['sha256'] for name, ref in sources.items()} != {name: ref['sha256'] for name, ref in original['sources'].items()}):
            raise Error('derived audio path binding changed source admission semantics or original digests')
        removed = {'sources', 'outputs', 'manifest_sha256', 'source_path_binding'}
        if {key: value for key, value in manifest.items() if key not in removed} != {key: value for key, value in original.items() if key not in removed}:
            raise Error('derived audio path binding changed global admission policy/traversal')
    expected = source_rows(source)
    if len(rows) != len(expected) or {row.get('id', row.get('item_id')) for row in rows} != set(expected):
        raise Error('audio admission must retain every source row exactly once')
    result, groups = {}, {}
    for row in rows:
        ident = row.get('id', row.get('item_id'))
        record = expected[ident]
        media = record.get('media', [])
        if len(media) != 1 or media[0].get('type') != 'audio' or media[0].get('role') != 'input':
            raise Error('audio admission requires one actual source waveform')
        if (row.get('source_row_sha256') != R.digest(record) or row.get('asset_sha256') != media[0].get('sha256') or
                row.get('authored_answer_sha256') != R.digest(record['messages'][-1]['content']) or
                row.get('split') != record.get('split') or not row.get('recovery_group_id') or
                type(row.get('train_admitted')) is not bool or not isinstance(row.get('blockers'), list)):
            raise Error('audio admission exact row/waveform/answer/split/group join differs')
        if row['train_admitted'] and (row['split'] != 'train' or row['blockers'] or row.get('source_train_eligible') is not True):
            raise Error('audio holdout or blocked row cannot enter optimizer data')
        T.pinned_file({'path': row['asset_path'], 'sha256': row['asset_sha256']}, path.parent)
        group = row['recovery_group_id']
        if group in groups and groups[group] != row['split']:
            raise Error('connected audio group crosses source splits')
        groups[group] = row['split']
        result[ident] = row
    return manifest, result

def verified_probe(entry, root, record):
    proof_path = T.pinned_file(entry, root)
    proof = T.read_json(proof_path)
    H.validate_encoder_proof(proof)
    waveform = record['media'][0]
    if (proof['input_contract'].get('asset_sha256') != waveform['sha256'] or
            proof.get('teacher_sha256_before') != P.VAE_SHA or proof.get('teacher_sha256_after') != P.VAE_SHA or
            proof.get('stage_c_decode_bitwise_equal') is not True or proof.get('cpu_only') is not True or
            proof.get('sources', {}).get('stage_c_sha256') != P.STAGE_SHA or
            proof.get('sources', {}).get('comfy_sources') != P.SOURCE_PINS or
            proof.get('runtime', {}).get('replacement_modules_used') is not False):
        raise Error('audio actual encoder/source/model-space proof does not bind this waveform')
    tensor_path = T.pinned_file(proof['tensors'], proof_path.parent)
    return proof_path, proof, tensor_path

def build_batch(record, dialogue, proof, latent, *, proof_sha, amendment_sha, charted_attention_sha):
    import torch
    H.validate_encoder_proof(proof)
    frames = proof['measurement']['raw_latent_shape'][-1]
    if (latent.shape != (2 * frames, 32) or latent.dtype != torch.float32 or latent.requires_grad or
            not bool(torch.isfinite(latent).all()) or dialogue.get('source_record_sha256') != R.digest(record) or
            dialogue.get('native_text_projection_bitwise_equal') is not True or
            dialogue.get('policy') != D.POLICY or len(dialogue['media_indices']) != 2 * frames):
        raise Error('audio native pack needs exact parent dialogue and genuine finite32-channel latent')
    ids, media = dialogue['input_ids'], dialogue['media_indices']
    text = [i for i in range(len(ids)) if i not in set(media)]
    chart = H.chart(dialogue['prompt_length'] - len(media), frames, len(ids), media, text,
        clean_encoder_receipt_sha256=proof_sha, amendment_sha256=amendment_sha,
        source_pins={'stage_c': P.STAGE_SHA, 'comfy_audio_vae': P.AUDIO_SOURCE_SHA,
                     'comfy_sd': P.SOURCE_PINS['comfy/sd.py'], 'charted_attention': charted_attention_sha}, encoder_proof=proof)
    features = torch.zeros(1, len(media), R.FEATURES, dtype=torch.float32)
    features[0, :, 480:512] = latent
    batch = {'input_ids': torch.tensor([ids], dtype=torch.int64),
        'assistant_target_mask': torch.tensor([dialogue['assistant_target_mask']], dtype=torch.bool),
        'native_media_features': features, 'native_media_indices': torch.tensor([media], dtype=torch.int64),
        'native_media_type_ids': torch.full((1, len(media)), 3, dtype=torch.int64),
        'native_media_valid_mask': torch.ones(1, len(media), dtype=torch.bool),
        'native_media_noise_levels': torch.zeros(1, len(media), dtype=torch.float32),
        'native_media_positions': torch.tensor(chart['native_media_positions'], dtype=torch.float32),
        'charted_source_positions': torch.tensor(chart['charted_source_positions'], dtype=torch.float32),
        'h3_prompt_read_mask': torch.tensor(chart['h3_prompt_read_mask'], dtype=torch.bool)}
    return batch, chart['contract']

def load_latent(proof, tensor_path):
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(tensor_path), device='cpu')
    raw, model, latent = (tensors[key] for key in ('x0_raw_audio', 'x0_model_audio', 'packed_model_audio'))
    waveform, decoded = (tensors[key] for key in ('input_waveform_stereo32k', 'decoded_waveform'))
    expected_pack = model[0].permute(1, 2, 0).reshape(-1, 32).contiguous()
    calls = proof['measurement'].get('encoder_input_calls', [])
    if (len(calls) != 2 or any(call.get('sha256') != P.tensor_sha(waveform) or call.get('shape') != list(waveform.shape) for call in calls) or
            list(raw.shape) != proof['measurement']['raw_latent_shape'] or list(decoded.shape) != proof['measurement']['decoded_shape'] or
            list(waveform.shape) != [1, 2, proof['input_contract']['target_samples']] or
            P.tensor_sha(raw) != proof['measurement']['raw_latent_sha256'] or P.tensor_sha(decoded) != proof['measurement']['decoded_sha256'] or
            any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in tensors.values()) or
            not torch.equal(raw, model) or not torch.equal(expected_pack, latent)):
        raise Error('audio clean waveform/raw/model/decoded tensor bytes differ from actual encoder proof')
    return latent

def validate_batch(batch, receipt, root):
    import torch
    record_path = T.pinned_file(receipt['source_record'], root)
    record = T.read_json(record_path)
    _, admitted = admission(receipt['source_admission'], root, T.pinned_file(receipt['source'], root))
    row = admitted.get(record['id'])
    if row is None or row.get('train_admitted') is not True or row['recovery_group_id'] != receipt['recovery_group_id']:
        raise Error('audio batch lost global training admission')
    proof_path, proof, tensor_path = verified_probe(receipt['encoder_proof'], root, record)
    latent = load_latent(proof, tensor_path)
    dialogue = T.read_json(T.pinned_file(receipt['dialogue'], root))
    actual, contract = build_batch(record, dialogue, proof, latent, proof_sha=R.sha(proof_path),
        amendment_sha=receipt['amendment_sha256'], charted_attention_sha=receipt['charted_attention_sha256'])
    if contract != receipt['chart'] or set(actual) != set(batch) or any(not torch.equal(actual[key], batch[key].detach().cpu()) for key in actual):
        raise Error('audio native batch differs from independently reconstructed sealed recipe')
    return True

def encode_source(source, source_preflight, source_preflight_sha, asset_root, stage_c, comfy_root, audio_vae, out):
    """Executable full traversal using the genuine CPU probe per source waveform."""
    if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise Error('encoding needs a fresh output and explicit CUDA-hidden CPU runtime')
    if R.sha(source_preflight) != source_preflight_sha:
        raise Error('audio preflight changed')
    preflight = T.read_json(source_preflight)
    rows = source_rows(source)
    if preflight.get('source_ready') is not True or preflight.get('source_sha256') != R.sha(source):
        raise Error('audio encoder source/preflight differs')
    known = {row['id']: row for row in preflight['rows']}
    if set(known) != set(rows):
        raise Error('audio encoder must visit every preflight source row')
    out.mkdir(parents=True)
    entries = []
    for number, (ident, record) in enumerate(rows.items()):
        if known[ident]['source_row_sha256'] != R.digest(record):
            raise Error('audio source row changed')
        case = record['provenance']['case_id']
        if Path(case).name != case or case in {'.', '..'}:
            raise Error('audio case path escapes asset root')
        audio = asset_root / case / 'sound.wav'
        destination = out / f'{number:06d}.json'
        result = P.run(SimpleNamespace(audio=audio, audio_sha256=record['media'][0]['sha256'],
            stage_c=stage_c, comfy_root=comfy_root, audio_vae=audio_vae,
            audio_source_sha256=P.AUDIO_SOURCE_SHA, out=destination))
        if result.get('probe_passed') is not True or result.get('model_space_verified') is not True:
            raise Error('actual audio encoder failed; partial outputs are diagnostic only: ' + ident)
        entries.append({'id': ident, 'source_row_sha256': R.digest(record), 'encoder_proof': pin(destination)})
    manifest = {'schema': 'chimera_recovery.clean_audio_cache.v1', 'sealed': True, 'complete_source_traversal': True,
        'source': pin(source), 'source_preflight': pin(source_preflight), 'entries': entries,
        'adapter_sha256': R.sha(Path(__file__)), 'encoder_adapter_sha256': R.sha(HERE / 'h3_audio_probe.py'),
        'native_admission_granted': False, 'training_started': False}
    (out / 'manifest.json').write_bytes(R.canonical(manifest) + b'\n')
    return manifest

def prepare(source, clean_manifest, clean_sha, admission_path, admission_sha, tokenizer_dir, amendment_sha, charted_attention, out, *, max_length=65536):
    if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise Error('native preparation needs fresh output and CUDA-hidden CPU runtime')
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    for name, expected in D.TOKENIZER_PINS.items():
        if R.sha(tokenizer_dir / name) != expected:
            raise Error('actual parent tokenizer changed: ' + name)
    R.require_sha(amendment_sha, 'audio_amendment_sha256')
    chart_sha = R.sha(charted_attention)
    manifest_path = T.pinned_file({'path': str(clean_manifest), 'sha256': clean_sha}, Path('.'))
    clean = T.read_json(manifest_path)
    if clean.get('schema') != 'chimera_recovery.clean_audio_cache.v1' or clean.get('sealed') is not True or clean.get('complete_source_traversal') is not True:
        raise Error('audio native preparation refuses partial/unsealed encoder output')
    if R.sha(T.pinned_file(clean['source'], manifest_path.parent)) != R.sha(source):
        raise Error('clean audio source changed')
    admission_pin = {'path': str(admission_path.resolve()), 'sha256': admission_sha}
    _, admitted = admission(admission_pin, Path('.'), source)
    rows = source_rows(source)
    entries = {entry['id']: entry for entry in clean['entries']}
    if len(entries) != len(clean['entries']) or set(entries) != set(rows):
        raise Error('clean encoder proof must cover every actual source row once')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.audio-prepare-', dir=out.parent))
    batches, excluded = [], []
    try:
        for number, (ident, record) in enumerate(rows.items()):
            entry, row = entries[ident], admitted[ident]
            if entry.get('source_row_sha256') != R.digest(record):
                raise Error('clean encoder source row identity differs')
            proof_path, proof, tensor_path = verified_probe(entry['encoder_proof'], manifest_path.parent, record)
            latent = load_latent(proof, tensor_path)
            dialogue = D.pack_dialogue(record, tokenizer, media_rows=latent.shape[0], max_length=max_length)
            batch, chart = build_batch(record, dialogue, proof, latent, proof_sha=R.sha(proof_path),
                amendment_sha=amendment_sha, charted_attention_sha=chart_sha)
            if not row['train_admitted']:
                excluded.append({'id': ident, 'split': row['split'], 'blockers': row['blockers'], 'recovery_group_id': row['recovery_group_id']})
                continue
            basename = f'{number:06d}'
            target = temporary / (basename + '.safetensors')
            save_file(batch, str(target))
            record_path, dialogue_path = temporary / (basename + '.source.json'), temporary / (basename + '.dialogue.json')
            record_path.write_bytes(R.canonical(record) + b'\n')
            dialogue_path.write_bytes(R.canonical(dialogue) + b'\n')
            # Final absolute paths, digest of exact temporary bytes. Atomic directory promotion keeps all pins valid.
            def final_pin(path):
                return {'path': str((out.resolve() / path.name)), 'sha256': R.sha(path)}
            receipt = {'schema': 'chimera_recovery.audio_cache.v1', 'sealed': True, 'all_finite': True,
                'id': ident, 'stage': STAGE, 'objective': 'authored_audio_assistant_ce', 'split': 'train', 'held': False,
                'source': pin(source), 'source_record': final_pin(record_path), 'dialogue': final_pin(dialogue_path),
                'source_admission': admission_pin, 'encoder_proof': pin(proof_path), 'source_tensors': pin(tensor_path),
                'corpus_sha256': admission_sha, 'tensors_sha256': R.sha(target), 'source_row_sha256': R.digest(record),
                'recovery_group_id': row['recovery_group_id'], 'amendment_sha256': amendment_sha,
                'charted_attention_sha256': chart_sha, 'chart': chart, 'prompt_length': dialogue['prompt_length'],
                'completion_eos_id': dialogue['completion_eos_id'], 'assistant_targets': dialogue['assistant_targets'],
                'clean_encoder_receipt_sha256': R.sha(proof_path), 'assistant_only': True, 'teacher_logits_used': False,
                'tokenizer_files': {name: pin(tokenizer_dir / name) for name in D.TOKENIZER_PINS},
                'adapter_sha256': R.sha(Path(__file__)), 'dialogue_adapter_sha256': R.sha(HERE / 'audio_dialogue.py'),
                'chart_adapter_sha256': R.sha(HERE / 'h3_audio_chart.py')}
            receipt_path = temporary / (basename + '.receipt.json')
            receipt_path.write_bytes(R.canonical(receipt) + b'\n')
            batches.append({'id': ident, 'route': record['route'], 'recovery_group_id': row['recovery_group_id'],
                'tensors': final_pin(target), 'teacher_receipt': final_pin(receipt_path)})
        if not batches:
            raise Error('no globally admitted training audio rows')
        result = {'schema': SCHEMA, 'stage': STAGE, 'objective': 'authored_audio_assistant_ce', 'sealed': True,
            'split': 'train', 'complete_source_traversal': True, 'heldout_groups_excluded': True,
            'source': pin(source), 'source_admission': admission_pin, 'clean_manifest': pin(manifest_path),
            'source_manifest_sha256': clean_sha, 'corpus_sha256': admission_sha, 'amendment_sha256': amendment_sha,
            'charted_attention': pin(charted_attention), 'batches': batches, 'excluded': excluded,
            'encoder_receipt_inventory_sha256': R.digest([entry['encoder_proof']['sha256'] for entry in clean['entries']]),
            'adapter_sha256': R.sha(Path(__file__)), 'training_started': False, 'native_parent_xyz_parity': False}
        (temporary / 'manifest.json').write_bytes(R.canonical(result) + b'\n')
        temporary.rename(out)
        return result
    except BaseException:
        # Failed staging has no manifest/promoted admission. Preserve it for diagnosis.
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    encode = sub.add_parser('encode-source')
    for name in ('source', 'source-preflight', 'asset-root', 'stage-c', 'comfy-root', 'audio-vae', 'out'):
        encode.add_argument('--' + name, type=Path, required=True)
    encode.add_argument('--source-preflight-sha256', required=True)
    native = sub.add_parser('prepare')
    for name in ('source', 'clean-manifest', 'admission', 'tokenizer-dir', 'charted-attention', 'out'):
        native.add_argument('--' + name, type=Path, required=True)
    for name in ('clean-manifest-sha256', 'admission-sha256', 'amendment-sha256'):
        native.add_argument('--' + name, required=True)
    native.add_argument('--max-length', type=int, default=65536)
    args = parser.parse_args(argv)
    if args.command == 'encode-source':
        result = encode_source(args.source, args.source_preflight, args.source_preflight_sha256, args.asset_root,
            args.stage_c, args.comfy_root, args.audio_vae, args.out)
    else:
        result = prepare(args.source, args.clean_manifest, args.clean_manifest_sha256, args.admission, args.admission_sha256,
            args.tokenizer_dir, args.amendment_sha256, args.charted_attention, args.out, max_length=args.max_length)
    print(json.dumps({'sealed': result['sealed'], 'training_started': False, 'entries': len(result.get('entries', result.get('batches', [])))}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
