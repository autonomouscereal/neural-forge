"""Preverify a complete bound source packet and extract only into this new own runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

RUN = I['run_uuid']
assert str(__import__('uuid').UUID(RUN)) == RUN and I['new_layers'] == [73,84]
assert I['run_uuid'] == RUN
assert I['runtime_root'] == '/home/cereal/omni-campaign/runtime/k3-nas-window73-84-oct07-' + RUN
assert I['output_root'] == '/home/cereal/omni-campaign/outputs/k3-nas-window73-84-oct07-' + RUN + '/control-output'
assert I['new_output_filesystem'] in ('NAS','Lexar')
assert I['native_working_output_root'] == ('/mnt/ai_data' if I['new_output_filesystem']=='NAS' else '/mnt/lexar')+'/omni-campaign/k3-nas-window73-84-oct07-' + RUN + '/model'
root, output = Path(I['runtime_root']), Path(I['output_root'])
archive = root / ARCHIVE_NAME

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

assert root.is_dir() and output.is_dir() and not root.is_symlink() and not output.is_symlink()
assert not (root / 'production_code').exists() and not (root / 'production_control').exists()
assert sha(archive) == ARCHIVE_SHA
verified = {}
with zipfile.ZipFile(archive) as packet:
    infos = packet.infolist()
    assert len(infos) == len(MEMBERS) and set(packet.namelist()) == set(MEMBERS)
    for info in infos:
        rel = PurePosixPath(info.filename)
        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in info.filename
        assert rel.parts[0] in ('production_code', 'production_control') and len(rel.parts) > 1
        assert not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
        raw = packet.read(info)
        pin = MEMBERS[info.filename]
        assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256']
        verified[info.filename] = raw
assert any(name.startswith('production_code/') for name in verified)
assert any(name.startswith('production_control/') for name in verified)
for name, raw in verified.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists() and not path.is_symlink()
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    assert sha(path) == MEMBERS[name]['sha256']
result = {'schema': 'neural_forge.k3_NAS73_84_owned_bound_source_stage.v1',
          'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'identity': I, 'archive_sha256': sha(archive), 'members': MEMBERS,
          'code_files': {p.relative_to(root/'production_code').as_posix(): sha(p) for p in (root/'production_code').rglob('*') if p.is_file()},
          'control_files': {p.relative_to(root/'production_control').as_posix(): sha(p) for p in (root/'production_control').rglob('*') if p.is_file()},
          'old_model_inputs_copied_bytes': 0, 'model_or_allocation_started': False}
with (output/'initial_source_stage73_84.json').open('x') as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(result))
