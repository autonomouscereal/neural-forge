"""Meaningful immutable source-only checks; no fixture/native imports or writes."""
from pathlib import Path
import argparse,hashlib,json
def check(root):
    roster=json.loads((root/'PACKET_SOURCE_ROSTER_V1.json').read_text())
    found={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert found==set(roster['members'])|{'PACKET_SOURCE_ROSTER_V1.json'},'exact exclusive packet file roster'
    for name,row in roster['members'].items():
        p=root/name;raw=p.read_bytes();assert not p.is_symlink() and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name
        if p.suffix=='.py':compile(raw,str(p),'exec')
    requirements=json.loads((root/'CALLER_CONTROL_INTEGRATION_REQUIREMENTS_V1.json').read_text())
    assert requirements['actual_prior_closure'] is None and requirements['actual_caps'] is None and requirements['actual_allocation'] is None
    assert requirements['can_arm'] is False and requirements['deployable'] is False
    assert requirements['whole_phase_writer_once_bytes'] in (180943168512,40692831840)
    assert requirements['initial_native_boundary']==('before_L73' if requirements['new_window']==[73,84] else 'before_L85')
    assert requirements['RO_origin_count']==(4 if requirements['new_window']==[73,84] else 5)
    assert requirements['exact_sorted_mount_count']==(12 if requirements['new_window']==[73,84] else 13)
    composition=json.loads((root/'evidence/SOURCE_COMPOSITION.json').read_text())
    assert composition['source_only'] is True and composition['deployable'] is False
    assert len(composition['source_pins'])==8
    for name,row in composition['source_pins'].items():
        raw=(root/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']
    review=json.loads((root/'evidence/LATER73_92_INDEPENDENT_SOURCE_REVIEW_V1.json').read_text())
    assert review['verdict']=='PASS_SOURCE_ONLY_CONDITIONAL_NONDEPLOYABLE' and review['qualified99_OTHER_native_ASTs_match9086_each_window'] is True
    phase='-'.join(str(x) for x in requirements['new_window'])
    independently_reviewed={r['name']:r for r in review['source_pins'] if r['phase']==phase}
    assert set(independently_reviewed)==set(composition['source_pins'])
    assert all(composition['source_pins'][name]['sha256']==independently_reviewed[name]['sha256'] for name in independently_reviewed)
    assert review['writer_accounting'][phase]['whole_writer_once_bytes']==requirements['whole_phase_writer_once_bytes']
    return {'packet_root':str(root),'window':requirements['new_window'],'full_member_hashes_verified':len(roster['members']),
            'eight_reviewed_sources_compiled':True,'late_binding_stays_null':True,'can_arm':False,'runtime_actions':False}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet-root',type=Path,action='append',required=True)
    args=parser.parse_args();print(json.dumps({'ok':True,'source_only':True,'packets':[check(p) for p in args.packet_root]},sort_keys=True))
if __name__=='__main__':main()
