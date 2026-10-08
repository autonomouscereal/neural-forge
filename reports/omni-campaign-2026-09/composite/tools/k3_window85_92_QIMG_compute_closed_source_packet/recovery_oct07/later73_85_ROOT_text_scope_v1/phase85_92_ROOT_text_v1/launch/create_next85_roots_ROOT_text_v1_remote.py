"""Narrow derivative of the existing exclusive ROOT/NAS directory creator."""
import datetime, json, os
from pathlib import Path

RUN=I['run_uuid']
assert str(__import__('uuid').UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139'
assert I['run_uuid']==RUN and I['new_layers']==[85,92]
assert I['runtime_root']=='/home/cereal/omni-campaign/runtime/k3-nas-window85-92-oct07-'+RUN
assert I['output_root']==I['controller_output_root']=='/home/cereal/omni-campaign/outputs/k3-nas-window85-92-oct07-'+RUN+'/control-output'
assert I['new_output_filesystem'] in ('NAS','Lexar')
STORAGE='/mnt/ai_data' if I['new_output_filesystem']=='NAS' else '/mnt/lexar'
assert I['native_working_output_root']==STORAGE+'/omni-campaign/k3-nas-window85-92-oct07-'+RUN+'/model'
sources={role:Path(path) for role,path in I['readonly_resume_source_roots'].items()}
assert set(sources)=={'closed37_48', 'closed61_72', 'closed73_84', 'legacy_closed36', 'closed49_60'}
model=Path(I['native_working_output_root'])
for source in sources.values():
    assert source.is_dir() and source.resolve()==source and not source.is_symlink()
    assert source!=model and not source.is_relative_to(model) and not model.is_relative_to(source)
paths=[Path(I['runtime_root']),Path(I['controller_output_root']),model.parent]
for path in paths:
    assert path.is_absolute() and '..' not in path.parts and RUN in str(path) and not path.exists()
    allowed=Path(STORAGE+'/omni-campaign') if path.is_relative_to(STORAGE) else Path('/home/cereal/omni-campaign')
    assert path.is_relative_to(allowed)
    for parent in [path,*path.parents]:
        if parent.exists():assert not parent.is_symlink()
    path.mkdir(parents=True,exist_ok=False)
model.mkdir(exist_ok=False)
observer=Path(I['runtime_root'])/'observer_source';observer.mkdir(exist_ok=False)
assert sources['closed37_48'].stat().st_dev==sources['closed49_60'].stat().st_dev
assert sources['legacy_closed36'].stat().st_dev!=sources['closed37_48'].stat().st_dev
for role in ['closed61_72', 'closed73_84']:
    source=sources[role]
    domain='closed37_48' if source.is_relative_to('/mnt/ai_data') else 'legacy_closed36'
    assert source.is_relative_to('/mnt/ai_data/omni-campaign') or source.is_relative_to('/mnt/lexar/omni-campaign')
    assert source.stat().st_dev==sources[domain].stat().st_dev
assert model.stat().st_dev==sources['closed37_48' if I['new_output_filesystem']=='NAS' else 'legacy_closed36'].stat().st_dev
created=[*paths,model,observer]
record={'schema':'neural_forge.k3_exclusive_NAS85_92_model_root.v1',
    'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':I,
    'created_paths':[str(path) for path in created],'old_prefixes_modified':False,
    'new_payload_copy_bytes':0,'model_or_allocation_admitted':False,
    'roots':{str(path):{'device':path.stat().st_dev,'inode':path.stat().st_ino,
        'free_bytes':os.statvfs(path).f_bavail*os.statvfs(path).f_frsize} for path in created}}
target=model.parent/'OWN_NAMESPACE_V1.json'
raw=(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert target.read_bytes()==raw
print(json.dumps(record))
