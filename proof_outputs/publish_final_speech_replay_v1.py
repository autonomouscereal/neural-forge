"""Copy exact sealed receipts and write a dated scope successor, without altering producers."""
import hashlib
import json
import shutil
from pathlib import Path

OWN = Path(__file__).resolve().parents[1]
OLD = OWN.parent / 'chimera-trainer-readiness-oct13-3e25f2da3b32'
BASE = Path('C:/Users/cereal/Documents/ai training')
SEALED = OLD / 'proof_outputs/native_seals_v3/codex-native-audio-consumer-modality-v2'
OUT = BASE / 'reviews/2026-10-13-folding/recovery-preparation/speech-replay-final-v1'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pin(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}

def copy_exact(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(src) != sha(dst):
            raise RuntimeError('immutable destination differs: ' + str(dst))
    else:
        shutil.copyfile(src, dst)
    if sha(src) != sha(dst):
        raise RuntimeError('copy verification failed: ' + str(dst))
    return {'source': pin(src), 'copied': pin(dst)}

index = json.loads((SEALED / 'replay-index.json').read_text())
if sha(SEALED / 'replay-index.json') != '3c99cfb737b1d7ff0b0e987687cf4fca6b60910da60e9ab3d94fdbf719e8152c':
    raise RuntimeError('unexpected replay seal')
if any(index.get(k) is not True for k in ('sealed', 'passed', 'verified', 'complete_source_traversal', 'source_reconstruction_bitwise_equal')):
    raise RuntimeError('incomplete replay index')
if index.get('training_started') is not False or index.get('model_forward_performed') is not False:
    raise RuntimeError('unexpected production execution claim')
counts = {}
for entry in index['entries']:
    obj = entry['objective']
    value = counts.setdefault(obj, {'batches': 0, 'supervised_units': 0})
    value['batches'] += 1
    value['supervised_units'] += entry['supervised_units']
if counts != index['objectives']:
    raise RuntimeError('measured replay denominators do not match entries')
expected = {
    'authored_audio_assistant_ce': {'batches': 194, 'supervised_units': 2833},
    'supervised_speech_rectified_flow': {'batches': 744, 'supervised_units': 8776960},
}
if counts != expected or len(index['proofs']) != 225 or index['held_source_rows'] != 31:
    raise RuntimeError('unexpected replay coverage')

copies = []
for src in sorted(SEALED.rglob('*')):
    if src.is_file():
        copies.append(copy_exact(src, OUT / 'actual' / src.relative_to(SEALED)))
archives = OLD / 'proof_outputs/native_audio_consumer_v2/downloaded-seals'
for name in ('collection-v3.tar.gz', 'collection-v3.tar.json'):
    copies.append(copy_exact(archives / name, OUT / name))

scope = json.loads((OWN / 'proof_outputs/owned_modality_readiness_oct13_v2/manifest.json').read_text())
scope['supersedes'] = pin(OWN / 'proof_outputs/owned_modality_readiness_oct13_v2/manifest.json')
scope['evidence']['audio_speech_final_replay'] = pin(OUT / 'actual/replay-index.json')
scope['lanes']['audio_ce']['full_replay_source_reconstruction_sealed'] = True
scope['lanes']['audio_ce']['measured_assistant_positions'] = 2833
scope['lanes']['speech_flow']['verification_state'] = 'complete: all744 actual source reconstructions bitwise authenticated'
scope['lanes']['speech_flow']['full_replay_source_reconstruction_sealed'] = True
scope['lanes']['speech_flow']['measured_weighted_native_scalar_units'] = 8776960
scope['last_audio_audit_status'] = {
    'exitcode': 0, 'encoder_proofs_verified': 225,
    'native_ce_batches_verified': 194, 'native_speech_flow_batches_verified': 744,
    'gpu_exposed': False, 'model_forward_performed': False, 'training_started': False,
}
scope['remaining_common'] = [x for x in scope['remaining_common'] if x != 'final744 speech replay source reconstruction seal']
scope_raw = json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode() + b'\n'
scope_path = OUT / 'scope-successor.json'
if scope_path.exists() and scope_path.read_bytes() != scope_raw:
    raise RuntimeError('scope successor differs from existing immutable artifact')
scope_path.write_bytes(scope_raw)
receipt = {
    'schema': 'chimera_recovery.speech_replay_collection.v1', 'dated': '2026-10-13',
    'index': pin(OUT / 'actual/replay-index.json'), 'scope': pin(scope_path),
    'copied_files': copies, 'all_copied_sha256_verified': True,
    'scope_boundary': 'genuine encoders, source-authenticated native batches and CPU mechanics; no current-model capability or production launch claim',
    'speech_scope': '186 exact-ASR retained synthetic train sources from48 authored English utterances/four official voices;15 separate held speech sources',
    'procedural_scope': '8 authored non-speech train sources;16 held observations',
    'storage': 'original tensor files remain in the source-pinned remote namespaces recorded by the unchanged manifests; this capsule copies receipts/index and the SHA-verified collected archive',
}
receipt_path = OUT / 'collection-proof.json'
raw = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode() + b'\n'
if receipt_path.exists() and receipt_path.read_bytes() != raw:
    raise RuntimeError('collection proof differs from existing immutable artifact')
receipt_path.write_bytes(raw)
print(json.dumps({'collection': pin(receipt_path), 'scope': pin(scope_path), 'index': pin(OUT / 'actual/replay-index.json'), 'copied_files': len(copies), 'objectives': counts}))
