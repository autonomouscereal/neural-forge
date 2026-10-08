"""Immutable whole later source checks; compile only, no fixtures/imports."""
import argparse,hashlib,json
from pathlib import Path
def check(root):
    roster=json.loads((root/'WHOLE_SOURCE_PACKET_ROSTER_V1.json').read_bytes())
    files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert files==set(roster['members'])|{'WHOLE_SOURCE_PACKET_ROSTER_V1.json'}
    for name,row in roster['members'].items():
        p=root/name;raw=p.read_bytes();assert not p.is_symlink() and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        if name.endswith('.py'):compile(raw,str(p),'exec')
    c=json.loads((root/'WHOLE_CONTROL_CONTRACT_V1.json').read_bytes())
    assert c['actual_prior_closure'] is c['actual_UUID'] is c['actual_FS'] is c['actual_cap'] is c['actual_A'] is None and c['can_arm'] is False and c['deployable'] is False
    assert c['metadata_before_A']==(70 if c['window']==[73,84] else 73) and c['metadata_with_A']==c['metadata_before_A']+1
    assert c['mount_count']==(12 if c['window']==[73,84] else 13) and c['whole_writer_once']==(180943168512 if c['window']==[73,84] else 40692831840)
    assert c['ROOT_serial_future_once']=={'RAM_bytes':16*(1<<30),'CPU_threads':4,'ROOT_write_bytes':9*(1<<30),'active_actor_adopted':False}
    for name,row in c['selected_members'].items():assert roster['members'][name]['sha256']==row['sha256'] and roster['members'][name]['bytes']==row['bytes']
    review=json.loads((root/'evidence/WHOLE_LATER_INDEPENDENT_REVIEW.json').read_bytes())
    assert review['verdict'].startswith('PASS_SOURCE_ONLY') and c['whole_independent_review_sha256']==hashlib.sha256((root/'evidence/WHOLE_LATER_INDEPENDENT_REVIEW.json').read_bytes()).hexdigest()
    return {'window':c['window'],'member_count':len(roster['members']),'all_fullSHA_and_compile_verified':True,'actual_admission_claimed':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--packet-root',type=Path,action='append',required=True);a=p.parse_args()
    print(json.dumps({'ok':True,'source_only':True,'packets':[check(r) for r in a.packet_root]},sort_keys=True))
if __name__=='__main__':main()
