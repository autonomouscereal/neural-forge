"""Three bounded checks of the selected source using authentic public metadata."""
from pathlib import Path
import ast, copy, hashlib, json
HERE=Path(__file__).resolve().parent
PREFLIGHT_SHA='c3c6359b79342824af5b77d5073e4dabde720501de8648de587defbecb2f102e'
PROOF_SHA='1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9'
def require(ok,msg):
    if ok is not True:raise RuntimeError(msg)
def main():
    source=(HERE/'k3_owned_capacity_preflight49_60_current_peer_v3.py').read_bytes()
    raw=(HERE/'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json').read_bytes()
    assert hashlib.sha256(source).hexdigest()==PREFLIGHT_SHA
    assert hashlib.sha256(raw).hexdigest()==PROOF_SHA
    compile(source,'<selected isolated preflight>','exec')
    packet=json.loads(raw)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='current_peer_public_source')
    env={'require':require,'json':json,'hashlib':hashlib,'GIB':1<<30,'CURRENT_PEER_SOURCE_SHA':PROOF_SHA,
         'QIMG_OWNER':'codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c',
         'ROOT_OWNER':'codex-native-01a10f94-259b-7623-9679-ce4e68f82b8c'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<selected actual metadata join>','exec'),env)
    def request():return {'sources':{'current_peer_holder_source':{'sha256':PROOF_SHA}}}
    req=request();joined=env['current_peer_public_source']({'current_peer_holder_source':packet},req)
    assert len(packet['members'])==19
    assert joined['allocation']['container_id']=='7fb0e676b730d2dded9294d5318a18ba347e67c7d096a759a65dbc5946fd610f'
    assert joined['inspect']['HostConfig']['Memory']==joined['inspect']['HostConfig']['NanoCpus']==0
    assert req['ROOT1075_exact_completed_phase_future_once']['all_ROOT_futures_discharged'] is False
    rows=[{'case':'authentic19_fullraw_source_terminal_release_join','pass':True}]
    bad=copy.deepcopy(packet);bad['members']['qimg_worker']['raw_utf8']+=' '
    try:env['current_peer_public_source']({'current_peer_holder_source':bad},request())
    except RuntimeError as exc:rows.append({'case':'changed_actual_worker_bytes_refuse','pass':True,'reason':str(exc)})
    else:raise AssertionError('changed actual worker bytes accepted')
    bad=request();bad['sources']['current_peer_holder_source']['sha256']='0'*64
    try:env['current_peer_public_source']({'current_peer_holder_source':packet},bad)
    except RuntimeError as exc:rows.append({'case':'wrong_whole_public_proof_pin_refuses','pass':True,'reason':str(exc)})
    else:raise AssertionError('wrong public proof pin accepted')
    result={'schema':'neural_forge.k3_next49_current_peer_actual_metadata_checks.v1',
            'source_only':True,'checks':rows,'preflight_sha256':PREFLIGHT_SHA,'proof_sha256':PROOF_SHA,
            'proof_members':19,'compile_pass':True,'fresh_host_frame_checked':False,
            'actual_current_QIMG_cap_claimed':False,'remote_model_resource_foreign_actions':False,
            'preserved_draft_failures':[
              {'revision':'V1','failure':'ready path used control; worker actually writes native-results'},
              {'revision':'V2','failure':'raw ledger13 columns differ from enriched allocation gpu_name/model_id/port'}],
            'new_whole_source_closure_binding_and_fresh_current_admission_required':True}
    out=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    target=HERE/'CURRENT_PEER_ACTUAL_METADATA_CHECKS_V1.json'
    with target.open('xb') as handle:handle.write(out)
    assert target.read_bytes()==out
    print(json.dumps({'path':str(target),'sha256':hashlib.sha256(out).hexdigest(),'bytes':len(out),'checks_passed':len(rows)}))
if __name__=='__main__':main()
