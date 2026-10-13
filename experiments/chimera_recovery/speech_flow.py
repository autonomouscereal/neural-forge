"""Genuine waveform targets for a separately amended supervised speech flow.

Source speech understanding rows stay unchanged. The derived generation prompt
and positive-sigma flow contract are explicit; this is not guided parent KD.
No speech codec index, clean target, reference answer token or ASR logit enters
the model input. Encoding and global source admission are separate operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

try:
    from . import audio_cache as C
except ImportError:
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('chimera_speech_cache_common', Path(__file__).with_name('audio_cache.py'))
    C = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = C
    spec.loader.exec_module(C)
T, R, D, H, Error = C.T, C.R, C.D, C.H, C.Error
HERE = Path(__file__).resolve().parent
STAGE = 'D3d_speech_flow'
OBJECTIVE = 'supervised_speech_rectified_flow'
READ_POLICY = 'h3_generation_prompt_and_noisy_audio_only'
INSERTION_POLICY = 'native_noisy_audio_after_assistant_generation_prefix_v1'
CHART_POLICY = 'audio40hz_generation_prompt_offset_categorical_stereo_v1'
SCHEMA = 'chimera_recovery.speech_flow_batches.v1'
SIGMAS = [.05, .3, .7, 1.]
SEED = 20261013
VOICES = {'Ryan', 'Aiden', 'Vivian', 'Serena'}
DEPENDENCIES = ('audio_cache.py', 'audio_dialogue.py', 'h3_audio_chart.py', 'h3_audio_probe.py', 'training.py', 'media_cache.py')
STORAGE = {'schema': 'chimera_recovery.native_band.v1', 'width': 6144,
           'bands': {'3': {'feature': [480, 512], 'emission': [224, 256]}}, 'channel_loss_mask': False}


def dependency_pins():
    return {name: R.sha(HERE / name) for name in DEPENDENCIES}


def generation_prompt(record):
    p, messages = record.get('provenance', {}), record.get('messages', [])
    if (record.get('route') != 'speech_understand' or p.get('speaker') not in VOICES or
            p.get('independent_asr_exact_match') is not True or p.get('synthetic_official_voice') is not True or
            p.get('user_reference_voice_used') is not False or
            [m.get('role') for m in messages] != ['user', 'assistant'] or
            record.get('supervision', {}).get('message_mask') != [False, True] or record.get('tools') or
            any(message.get('tool_calls') for message in messages)):
        raise Error('speaking targets require source-sealed speech transcript, exact ASR and official synthetic voice')
    transcript = messages[-1].get('content')
    if (not isinstance(transcript, str) or not transcript.strip() or
            any(token in transcript for token in ('<|im_start|>', '<|im_end|>'))):
        raise Error('speaking condition requires the exact unambiguous source transcript')
    text = f'Generate speech in the synthetic {p["speaker"]} voice, saying exactly:\n{transcript}'
    return {'messages': [{'role': 'user', 'content': text}], 'transcript': transcript,
            'speaker': p['speaker'], 'source_row_sha256': R.digest(record),
            'transcript_sha256': R.digest(transcript), 'prompt_sha256': R.digest(text),
            'route': 'speech_generate', 'policy': INSERTION_POLICY}


def pack_prefix(record, tokenizer, *, media_rows, max_length):
    prompt = generation_prompt(record)
    if type(media_rows) is not int or media_rows < 1:
        raise Error('speech generation needs a real positive native row count')
    options = dict(add_generation_prompt=True, enable_thinking=False, tools=None)
    rendered = tokenizer.apply_chat_template(prompt['messages'], tokenize=False, **options)
    ids = R.native_token_ids(tokenizer.apply_chat_template(prompt['messages'], tokenize=True, **options))
    if (not isinstance(rendered, str) or not ids or
            ids != R.native_token_ids(tokenizer(rendered, add_special_tokens=False)) or
            any(type(v) is not int or not 0 <= v < R.TEXT_VOCAB for v in ids) or not getattr(tokenizer, 'chat_template', None)):
        raise Error('speech generation must preserve the exact authentic parent generation-prefix IDs')
    if type(max_length) is not int or len(ids) + media_rows > max_length:
        raise Error('speech generation exceeds bounded context; no waveform or prompt truncation')
    return {**prompt, 'schema': 'chimera_recovery.speech_generation_prefix.v1',
        'input_ids': ids + [D.PLACEHOLDER] * media_rows, 'media_indices': list(range(len(ids), len(ids) + media_rows)),
        'parent_text_ids': ids, 'prompt_length': len(ids), 'native_text_projection_bitwise_equal': True,
        'template_sha256': R.digest(tokenizer.chat_template),
        'rendered_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest()}


def noise_seed(source_row_sha256, sigma):
    digest = R.digest({'source_row_sha256': source_row_sha256, 'sigma': sigma, 'seed': SEED,
                       'policy': 'source_row_sigma_sha256_first63bits_v1'})
    return int(digest[:16], 16) & ((1 << 63) - 1)


def build_batch(record, prefix, proof, latent, *, proof_sha, amendment_sha, charted_attention_sha, sigma):
    import torch
    H.validate_encoder_proof(proof)
    frames = proof['measurement']['raw_latent_shape'][-1]
    if (sigma not in SIGMAS or type(sigma) not in (int, float) or latent.shape != (2 * frames, 32) or
            latent.dtype != torch.float32 or latent.requires_grad or not bool(torch.isfinite(latent).all()) or
            prefix.get('source_row_sha256') != R.digest(record) or prefix.get('policy') != INSERTION_POLICY or
            prefix.get('native_text_projection_bitwise_equal') is not True):
        raise Error('speech flow needs exact detached native latent, generation prefix and declared positive sigma')
    expected_prompt = generation_prompt(record)
    if any(prefix.get(key) != expected_prompt[key] for key in expected_prompt):
        raise Error('speech prompt/transcript/voice/source semantic derivation changed')
    n, length = latent.shape[0], len(prefix['input_ids'])
    text_count, indices = prefix['prompt_length'], prefix['media_indices']
    if (text_count < 1 or prefix['input_ids'][:text_count] != prefix['parent_text_ids'] or
            indices != list(range(text_count, length)) or len(indices) != n or
            prefix['input_ids'][text_count:] != [D.PLACEHOLDER] * n):
        raise Error('noisy generation rows must follow the complete native generation prefix')
    source_pins = {'stage_c': C.P.STAGE_SHA, 'comfy_audio_vae': C.P.AUDIO_SOURCE_SHA,
                   'comfy_sd': C.P.SOURCE_PINS['comfy/sd.py'], 'charted_attention': charted_attention_sha}
    # Reuse the proved nominal-lattice mathematics, explicitly derive a new
    # noisy-generation contract; the clean observation contract is not selected.
    mapped = H.chart(text_count, frames, length, indices, list(range(text_count)),
        clean_encoder_receipt_sha256=proof_sha, amendment_sha256=amendment_sha,
        source_pins=source_pins, encoder_proof=proof)
    chart = {'schema': 'chimera_recovery.speech_generation_chart.v1', 'policy': CHART_POLICY,
        'read_policy': READ_POLICY, 'insertion_policy': INSERTION_POLICY,
        'coordinate_derivation': 'source-pinned nominal H3 lattice with new generation prompt offset; categorical stereo axis',
        'nominal_chart_math_adapter_sha256': R.sha(HERE / 'h3_audio_chart.py'),
        'source_pins': source_pins, 'clean_encoder_receipt_sha256': proof_sha,
        'amendment_sha256': amendment_sha, 'sigma': sigma, 'native_parent_xyz_parity': False,
        'latent_frames_per_channel': frames, 'sample_rate': 32000, 'samples_per_frame': 800,
        'spec_type_ids': [2, 3], 'actual_type_id': 3, 'prompt_length': text_count,
        'input_read_scope': 'exact generation prompt plus noisy audio; no assistant answer or clean waveform target'}
    seed = noise_seed(R.digest(record), sigma)
    epsilon = torch.randn(latent.shape, dtype=torch.float32, generator=torch.Generator(device='cpu').manual_seed(seed))
    noisy = (1. - sigma) * latent + sigma * epsilon
    features, targets = torch.zeros(1, n, 6144), torch.zeros(1, n, 6144)
    features[0, :, 480:512], targets[0, :, 224:256] = noisy, latent
    batch = {'input_ids': torch.tensor([prefix['input_ids']], dtype=torch.int64),
        'native_media_features': features, 'native_media_targets': targets,
        'native_media_indices': torch.tensor([indices], dtype=torch.int64),
        'native_media_type_ids': torch.full((1, n), 3, dtype=torch.int64),
        'native_media_noise_levels': torch.full((1, n), sigma, dtype=torch.float32),
        'native_media_valid_mask': torch.ones(1, n, dtype=torch.bool),
        'native_media_target_mask': torch.ones(1, n),
        'native_media_positions': torch.tensor(mapped['native_media_positions'], dtype=torch.float32),
        'charted_source_positions': torch.tensor(mapped['charted_source_positions'], dtype=torch.float32),
        'flow_noise': epsilon.unsqueeze(0), 'h3_generation_read_mask': torch.ones(1, length, dtype=torch.bool)}
    return batch, chart, seed


def validate_batch(batch, receipt, root):
    source = T.pinned_file(receipt['source'], root)
    _, admitted = C.admission(receipt['source_admission'], root, source)
    record = T.read_json(T.pinned_file(receipt['source_record'], root))
    row = admitted.get(record['id'], {})
    if (receipt.get('schema') != 'chimera_recovery.speech_flow_cache.v1' or receipt.get('sealed') is not True or
            receipt.get('objective') != OBJECTIVE or receipt.get('guided_teacher') is not False or
            receipt.get('held') is not False or receipt.get('split') != 'train' or row.get('train_admitted') is not True or
            row.get('recovery_group_id') != receipt.get('recovery_group_id') or
            receipt.get('source_row_sha256') != R.digest(record) or receipt.get('adapter_sha256') != R.sha(Path(__file__)) or
            receipt.get('dependencies') != dependency_pins() or
            receipt.get('storage') != STORAGE or
            receipt.get('input_layout') != 'feature' or receipt.get('target_layout') != 'emission' or
            receipt.get('prediction_kind') != 'x0' or receipt.get('target_kind') != 'x0' or
            receipt.get('noisy_feature_bands') != {'3': [480, 512]} or receipt.get('no_clean_target_in_inputs') is not True):
        raise Error('speaking cache/source/admission/flow contract differs')
    proof_path, proof, tensor_path = C.verified_probe(receipt['encoder_proof'], root, record)
    latent = C.load_latent(proof, tensor_path)
    prefix = T.read_json(T.pinned_file(receipt['prefix'], root))
    for name, ref in receipt['tokenizer_files'].items():
        if D.TOKENIZER_PINS.get(name) != ref['sha256']:
            raise Error('speech parent tokenizer pins differ')
        T.pinned_file(ref, root)
    if set(receipt['tokenizer_files']) != set(D.TOKENIZER_PINS):
        raise Error('speech parent tokenizer inventory incomplete')
    expected, chart, seed = build_batch(record, prefix, proof, latent, proof_sha=R.sha(proof_path),
        amendment_sha=receipt['amendment_sha256'], charted_attention_sha=receipt['charted_attention_sha256'], sigma=receipt['sigma'])
    import torch
    if chart != receipt['chart'] or seed != receipt['noise_seed'] or set(batch) != set(expected) or any(
            not torch.equal(batch[key].cpu(), value) for key, value in expected.items()):
        raise Error('speech cache differs from exact source/noise/prompt/native reconstruction')
    return True


def prepare(source, clean_manifest, clean_sha, admission_path, admission_sha, tokenizer_dir, amendment_sha, charted_attention, out, *, max_length=65536):
    if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise Error('speaking preparation needs fresh output and CUDA-hidden CPU runtime')
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    R.require_sha(amendment_sha, 'speech_flow_amendment_sha256')
    for name, sha in D.TOKENIZER_PINS.items():
        if R.sha(tokenizer_dir / name) != sha:
            raise Error('actual speech condition parent tokenizer changed: ' + name)
    clean_path = T.pinned_file({'path': str(clean_manifest), 'sha256': clean_sha}, Path('.'))
    clean = T.read_json(clean_path)
    if (clean.get('schema') != 'chimera_recovery.clean_audio_cache.v1' or clean.get('sealed') is not True or
            clean.get('complete_source_traversal') is not True or
            R.sha(T.pinned_file(clean['source'], clean_path.parent)) != R.sha(source)):
        raise Error('speaking needs a complete sealed genuine waveform encoder cache')
    admission_pin = {'path': str(admission_path.resolve()), 'sha256': admission_sha}
    _, admitted = C.admission(admission_pin, Path('.'), source)
    rows = C.source_rows(source)
    if any(record['route'] != 'speech_understand' for record in rows.values()):
        raise Error('speaking preparation requires distinct source-sealed speech rows')
    entries = {item['id']: item for item in clean['entries']}
    if len(entries) != len(clean['entries']) or set(entries) != set(rows):
        raise Error('speaking encoder proof inventory must cover every source row once')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    chart_sha = R.sha(charted_attention)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.speech-flow-', dir=out.parent))
    batches, excluded = [], []
    for number, (ident, record) in enumerate(rows.items()):
        entry, row = entries[ident], admitted[ident]
        if entry['source_row_sha256'] != R.digest(record):
            raise Error('speaking encoder source row changed')
        proof_path, proof, tensor_path = C.verified_probe(entry['encoder_proof'], clean_path.parent, record)
        latent = C.load_latent(proof, tensor_path)
        prefix = pack_prefix(record, tokenizer, media_rows=latent.shape[0], max_length=max_length)
        if not row['train_admitted']:
            excluded.append({'id': ident, 'split': row['split'], 'blockers': row['blockers'], 'recovery_group_id': row['recovery_group_id']})
            continue
        basename = f'{number:06d}'
        source_path, prefix_path = temporary / (basename + '.source.json'), temporary / (basename + '.prefix.json')
        source_path.write_bytes(R.canonical(record) + b'\n')
        prefix_path.write_bytes(R.canonical(prefix) + b'\n')
        def final_pin(path):
            return {'path': str(out.resolve() / path.name), 'sha256': R.sha(path)}
        for index, sigma in enumerate(SIGMAS):
            batch, chart, seed = build_batch(record, prefix, proof, latent, proof_sha=R.sha(proof_path),
                amendment_sha=amendment_sha, charted_attention_sha=chart_sha, sigma=sigma)
            target = temporary / (basename + f'.sigma{index}.safetensors')
            stored, storage = T.C.compact_native_batch(batch, {3: {'feature': (480, 512), 'emission': (224, 256)}})
            if storage != STORAGE:
                raise Error('speaking compact storage disagrees with audited native type3 allocation')
            save_file(stored, str(target))
            receipt = {'schema': 'chimera_recovery.speech_flow_cache.v1', 'sealed': True, 'all_finite': True,
                'id': ident + f':sigma{index}', 'source_id': ident, 'stage': STAGE, 'objective': OBJECTIVE,
                'split': 'train', 'held': False, 'guided_teacher': False, 'teacher_logits_used': False,
                'no_clean_target_in_inputs': True, 'input_layout': 'feature', 'target_layout': 'emission',
                'prediction_kind': 'x0', 'target_kind': 'x0', 'noisy_feature_bands': {'3': [480, 512]},
                'source': C.pin(source), 'source_record': final_pin(source_path), 'prefix': final_pin(prefix_path),
                'source_admission': admission_pin, 'encoder_proof': C.pin(proof_path), 'source_tensors': C.pin(tensor_path),
                'source_row_sha256': R.digest(record), 'recovery_group_id': row['recovery_group_id'],
                'corpus_sha256': admission_sha, 'tensors_sha256': R.sha(target), 'amendment_sha256': amendment_sha,
                'charted_attention_sha256': chart_sha, 'chart': chart, 'sigma': sigma, 'noise_seed': seed,
                'prompt_length': prefix['prompt_length'], 'tokenizer_files': {name: C.pin(tokenizer_dir / name) for name in D.TOKENIZER_PINS},
                'adapter_sha256': R.sha(Path(__file__)), 'dependencies': dependency_pins(), 'storage': storage,
                'native_parent_xyz_parity': False}
            receipt_path = temporary / (basename + f'.sigma{index}.receipt.json')
            receipt_path.write_bytes(R.canonical(receipt) + b'\n')
            batches.append({'id': receipt['id'], 'source_id': ident, 'route': 'speech_generate',
                'recovery_group_id': row['recovery_group_id'], 'tensors': final_pin(target),
                'teacher_receipt': final_pin(receipt_path), 'storage': storage})
    if not batches:
        raise Error('no globally admitted speaking targets; diagnostic encodings do not grant admission')
    result = {'schema': SCHEMA, 'sealed': True, 'stage': STAGE, 'objective': OBJECTIVE, 'split': 'train',
        'complete_source_traversal': True, 'heldout_groups_excluded': True, 'source': C.pin(source),
        'source_admission': admission_pin, 'clean_manifest': C.pin(clean_path), 'corpus_sha256': admission_sha,
        'amendment_sha256': amendment_sha, 'charted_attention': C.pin(charted_attention), 'batches': batches,
        'excluded': excluded, 'sigma_grid': SIGMAS, 'noise_seed_policy': 'source_row_sigma_sha256_first63bits_v1',
        'seed': SEED, 'read_policy': READ_POLICY, 'prediction_kind': 'x0', 'guided_teacher': False,
        'training_started': False, 'native_parent_xyz_parity': False, 'adapter_sha256': R.sha(Path(__file__)),
        'dependencies': dependency_pins()}
    (temporary / 'manifest.json').write_bytes(R.canonical(result) + b'\n')
    temporary.rename(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'clean-manifest', 'admission', 'tokenizer-dir', 'charted-attention', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('clean-manifest-sha256', 'admission-sha256', 'amendment-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-length', type=int, default=65536)
    args = parser.parse_args(argv)
    result = prepare(args.source, args.clean_manifest, args.clean_manifest_sha256, args.admission, args.admission_sha256,
        args.tokenizer_dir, args.amendment_sha256, args.charted_attention, args.out, max_length=args.max_length)
    print(json.dumps({'sealed': result['sealed'], 'batches': len(result['batches']), 'training_started': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
