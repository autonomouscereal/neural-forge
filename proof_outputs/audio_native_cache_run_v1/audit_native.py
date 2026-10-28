import importlib.util
import json
import os
from pathlib import Path
import sys
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('audit is CPU only')
root = Path('/work')
spec = importlib.util.spec_from_file_location('actual_audio_native_audit', root / 'experiments/chimera_recovery/audio_cache.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
import torch
torch.set_num_threads(4)
clean_path = root / 'clean-audio-cache-v1/manifest.json'
native_path = root / 'native-audio-batches-v1/manifest.json'
clean, native = C.T.read_json(clean_path), C.T.read_json(native_path)
if not clean['sealed'] or not clean['complete_source_traversal'] or len(clean['entries']) != 24:
    raise ValueError('full24 actual encoder traversal missing')
source = C.source_rows(C.T.pinned_file(clean['source'], clean_path.parent))
proofs = []
for item in clean['entries']:
    proof_path, proof, tensors = C.verified_probe(item['encoder_proof'], clean_path.parent, source[item['id']])
    C.load_latent(proof, tensors)
    proofs.append({'id': item['id'], 'proof_sha256': C.R.sha(proof_path), 'tensors_sha256': C.R.sha(tensors),
        'raw_shape': proof['measurement']['raw_latent_shape'], 'model_space_verified': proof['model_space_verified'],
        'waveform_snr_db': proof['measurement']['waveform_snr_db']})
rows = []
for item in native['batches']:
    receipt = C.T.read_json(C.T.pinned_file(item['teacher_receipt'], native_path.parent))
    batch = C.T.load_native_batch(item, native_path.parent, 'cpu')
    C.validate_batch(batch, receipt, native_path.parent)
    rows.append({'id': item['id'], 'recovery_group_id': item['recovery_group_id'], 'split': receipt['split'],
        'input_ids_shape': list(batch['input_ids'].shape), 'features_shape': list(batch['native_media_features'].shape),
        'assistant_targets': int(batch['assistant_target_mask'].sum()), 'tensors_sha256': item['tensors']['sha256'],
        'source_reconstruction_bitwise_equal': True, 'clean_encoder_receipt_sha256': receipt['clean_encoder_receipt_sha256']})
if len(rows) != 8 or len(native['excluded']) != 16 or torch.cuda.is_initialized():
    raise ValueError('actual admitted8/held16 or CUDA audit differs')
result = {'schema': 'chimera_recovery.actual_audio_native_audit.v1', 'passed': True,
    'clean_manifest_sha256': C.R.sha(clean_path), 'native_manifest_sha256': C.R.sha(native_path),
    'source_admission_sha256': native['source_admission']['sha256'], 'amendment_sha256': native['amendment_sha256'],
    'proofs': proofs, 'batches': rows, 'supervised_targets': sum(row['assistant_targets'] for row in rows),
    'actual_encoder_rows': len(proofs), 'native_train_batches': len(rows), 'holdout_rows_excluded': len(native['excluded']),
    'source_reconstruction_bitwise_equal': True, 'model_forward_performed': False,
    'gpu_exposed': False, 'training_started': False, 'production_training_ready': False,
    'remaining': ['completed native/shared preceding D3d state', 'actual current-base amended full-depth/read-policy parity',
                  'exclusive training allocator runtime', 'held free-answer review during training'],
    'adapter_sha256': C.R.sha(root / 'experiments/chimera_recovery/audio_cache.py')}
Path('/audit/result.json').write_bytes(C.R.canonical(result) + b'\n')
print(json.dumps({key: value for key, value in result.items() if key not in {'batches', 'proofs'}}))
