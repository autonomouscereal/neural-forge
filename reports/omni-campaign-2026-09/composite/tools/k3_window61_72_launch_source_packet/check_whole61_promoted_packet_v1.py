"""Existing reviewed whole61 source checks; no imports/actions/fixtures."""
from pathlib import Path
import argparse,ast,hashlib,json,zipfile

ROSTER_SHA='6433edaa87fed9ce481f2362a9d7e4e3a58d927f3c71009c605760f035bc2c3f'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def functions(raw):
    result={}
    for node in ast.parse(raw).body:
        if isinstance(node,ast.FunctionDef):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.ClassDef):
            for child in node.body:
                if isinstance(child,ast.FunctionDef):result[node.name+'.'+child.name]=ast.dump(child,include_attributes=False)
    return result
def run(root):
    root=root.resolve();raw=(root/'WHOLE61_REVIEWED_SOURCE_ROSTER_V1.json').read_bytes();assert sha(raw)==ROSTER_SHA
    roster=json.loads(raw);assert roster['source_only'] is True and roster['deployable'] is False
    assert all(roster[key] is None for key in ('actual60','next_UUID','filesystem','allocation','capacity_admission'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(roster['members'])|{'WHOLE61_REVIEWED_SOURCE_ROSTER_V1.json','check_whole61_promoted_packet_v1.py'}
    compiled=0
    for name,row in roster['members'].items():
        p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
        raw=p.read_bytes();assert sha(raw)==row['sha256'] and len(raw)==row['bytes'],name
        if name.endswith('.py'):compile(raw,name,'exec');compiled+=1
    b=root/'recovery_oct07/nas61_launch_integration_v1';s=b/'current_ROOT_peer_scope_v1'
    contract=json.loads((s/'ROOT_SERIAL_WHOLE_CALLER_SOURCE_CONTRACT_V1.json').read_bytes())
    assert contract['metadata_count_before_allocation']==67 and contract['metadata_count_after_allocation']==68 and contract['script_count']==5
    assert contract['source_only'] is True and contract['actual60'] is None and contract['fresh_capacity'] is None
    assert contract['new_source_normal_promoted_commit'] is None # retained reviewed historical declaration
    proof=json.loads((s/'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_PUBLIC_PROOF_V1.json').read_bytes())
    assert len(proof['members'])==8
    for role,row in proof['members'].items():
        raw=row['raw_utf8'].encode();assert len(raw)==row['bytes'] and sha(raw)==row['sha256'];json.loads(raw)
    checks=json.loads((s/'ACTUAL_ROOT_SERIAL_FOCUSED_CHECKS_V1.json').read_bytes())
    assert checks['result']=='PASS_SOURCE_ONLY' and len(checks['cases'])==4 and all(x['pass'] is True for x in checks['cases'])
    assert checks['actual_serial_scope']['RAM_bytes']==17179869184 and checks['actual_serial_scope']['CPU_threads']==4
    assert checks['actual_serial_scope']['new_ROOT_write_bytes']==9663676416 and checks['classify_and_unknown_actor_guards_unchanged'] is True
    for name in ('current_ROOT_serial_peer_scope_independent_review_v1/ROOT_SERIAL_WHOLE_CALLER_INDEPENDENT_STATIC_REVIEW_V1.json',
        'nas61_whole_integration_independent_review_v1/WHOLE61_INDEPENDENT_STATIC_REVIEW_V1.json'):
        value=json.loads((root/'recovery_oct07'/name).read_bytes());assert 'PASS' in json.dumps(value)
    component=b/'packet_source'
    selected=json.loads((b/'SELECTED61_PACKET_SOURCE_PINS_V1.json').read_bytes());assert len(selected)==11
    native=functions((component/Path(selected['native']['path']).name).read_bytes())
    qualified=functions((component/'evidence/k3_dense_pass_9086.py').read_bytes())
    names=json.loads((component/'evidence/SOURCE_ONLY_PROPOSAL_VALIDATION_V2.json').read_bytes())['original_qualified_functions_AST_unchanged']
    assert len(names)==len(set(names))==99 and all(native[name]==qualified[name] for name in names)
    before=functions((component/'k3_owned_capacity_preflight61_72_template_v3.py').read_bytes())
    after=functions((s/'k3_owned_capacity_preflight61_72_ROOT_serial_v1.py').read_bytes())
    assert set(after)-set(before)=={'current_ROOT_serial_scope'}
    assert {name for name in before if before[name]!=after[name]}=={'run'}
    before=functions((b/'bind_actual61_runtime_v3.py').read_bytes());after=functions((s/'bind_actual61_ROOT_serial_v1.py').read_bytes())
    assert set(before)==set(after) and {name for name in before if before[name]!=after[name]}=={'selected','bind','verify_bound_bundle','final_configs'}
    assert functions((b/'launch/launch_actual61_72_v3.py').read_bytes())==functions((s/'launch/launch_actual61_ROOT_serial_v1.py').read_bytes())
    assert roster['writer_bytes_once']==166556705792 and roster['NAS_if_selected_common_once']==858583311360
    assert roster['ROOT_serial_RAM_CPU_ROOT_write_once']==[17179869184,4,9663676416]
    assert max(692026605568+166556705792,690415992832+166556705792)==858583311360
    assert 25769803776+16*(1<<30)+166556705792==209506378752
    nested=0
    for p in (root/'provenance').glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
            assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in z.namelist())
            nested+=len(z.namelist())
    print(json.dumps({'schema':'neural_forge.k3_whole61_ROOT_serial_normal_promotion_source_check.v1','PASS':True,
        'raw_member_checks':len(roster['members']),'compiled_source_count':compiled,'qualified_other_AST_unchanged_count':99,
        'existing_pure_authenticated_ROOT_check_cases':4,'existing_full_public_ROOT_raw_members':8,
        'existing_independent_whole_source_reviews':2,'nested_historical_immutable_member_count':nested,
        'existing_actor_unknown_guard_AST_unchanged':True,'actual60_or_allocation_admission_or_runtime_actions':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate-root',type=Path,required=True);run(p.parse_args().candidate_root)
