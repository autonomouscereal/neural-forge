"""Verify and exec the reviewed zero-payload metadata collector in real SSH ancestry."""
import hashlib, json, os
from pathlib import Path

RUN=I['run_uuid']
assert str(__import__('uuid').UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139'
assert I['run_uuid']==RUN and I['new_layers']==[73,84]
root=Path(I['runtime_root']);source=root/'observer_source';control=root/'production_control'
assert root==Path('/home/cereal/omni-campaign/runtime/k3-nas-window73-84-oct07-'+RUN)
assert set(ASSETS)=={'collect_actual73_84_multi_origin.py','parent61_catalog.json','mounts.json'}
for name,pin in ASSETS.items():
    path=source/name
    assert path.is_file() and not path.is_symlink() and path.resolve()==path
    raw=path.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
assert ASSETS['collect_actual73_84_multi_origin.py']==COLLECTOR_PIN
assert ASSETS['parent61_catalog.json']['sha256']==I['parent_catalog_sha256']
assert ASSETS['mounts.json']['sha256']==MOUNTS_SHA
assert hashlib.sha256((control/'identity.json').read_bytes()).hexdigest()==IDENTITY_SHA
assert json.loads((control/'identity.json').read_bytes())==I
args=['/usr/bin/python3','-B',str(source/'collect_actual73_84_multi_origin.py'),
    '--identity',str(control/'identity.json'),'--identity-sha256',IDENTITY_SHA,
    '--parent-catalog',str(source/'parent61_catalog.json'),
    '--mounts',str(source/'mounts.json'),'--mounts-sha256',MOUNTS_SHA,
    '--output-directory',str(control)]
os.execv(args[0],args)
