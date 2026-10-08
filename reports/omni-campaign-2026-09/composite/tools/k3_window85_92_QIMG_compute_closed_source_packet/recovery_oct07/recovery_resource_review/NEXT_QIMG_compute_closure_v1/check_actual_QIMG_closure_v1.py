"""Three focused pure metadata checks; no candidate main, runtime or DB call."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parents[1]
GIB=1<<30


def require(ok,message):
    if ok is not True:raise RuntimeError(message)


def integer(value,label):
    require(type(value) is int and value>=0,label)
    return value


def main():
    script=HERE/'qimg_native_compute_closure_consumer_v1.py'
    module=ast.parse(script.read_bytes())
    scope={'require':require,'integer':integer,'hashlib':hashlib,'json':json,
           'QIMG_OWNER':'codex-native-01a111e4-2e50-7eb0-916a-74e1058b3a9c'}
    module.body=[node for node in module.body if isinstance(node,ast.FunctionDef)]
    exec(compile(module,str(script),'exec'),scope)
    sources={};pins={}
    names={'QIMG_native_compute_closure':'QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json',
           'QIMG_native_terminal_raw_part_1':'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json',
           'QIMG_native_terminal_raw_part_2':'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json'}
    for role,name in names.items():
        raw=(HERE/name).read_bytes();sources[role]=json.loads(raw)
        pins[role]={'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    prior=RECOVERY/'recovery_resource_review/actual49_60_binding_proposal_v1/source_bundle_launch_v1/SOURCE_CLOSURE49_60.json'
    closure=json.loads(prior.read_bytes())
    # Actual source49 retained metadata only, plus the current reviewed ROOT proof.
    other=sum(ref['bytes'] for ref in closure['sources'].values())+191797
    request={'sources':pins,'source_metadata_bytes_except_QIMG_closure':other}
    result=scope['qimg_native_compute_closure'](sources,request)
    assert result['RAM_bytes']==result['CPU_threads']==0 and result['raw_member_count']==62
    assert result['retained_disk_credit_bytes']==0 and result['foreign_provider_scope_discharged'] is False
    checks=[{'name':'actual_full62_terminal_normal_release_ownerV37_join','passed':True,
             'existing_source49_metadata_bytes':other,'decoded_plus_existing_metadata_bytes':other+result['decoded_all_proof_bytes']}]
    bad=copy.deepcopy(sources)
    row=bad['QIMG_native_terminal_raw_part_1']['members'][0]
    raw=zlib.decompress(base64.b64decode(row['raw_zlib_base64']));raw=raw[:-1]+bytes([raw[-1]^1])
    row['raw_zlib_base64']=base64.b64encode(zlib.compress(raw,9)).decode()
    try:scope['qimg_native_compute_closure'](bad,request)
    except RuntimeError as e:checks.append({'name':'changed_actual_terminal_raw_fullSHA_refused','passed':True,'reason':str(e)})
    else:raise AssertionError('changed actual raw accepted')
    bad=copy.deepcopy(sources)
    row=bad['QIMG_native_compute_closure']['outer']['release']
    raw=zlib.decompress(base64.b64decode(row['raw_zlib_base64']));doc=json.loads(raw)
    doc['fresh_committed_released_row']['owner']='wrong-owner'
    row['raw_zlib_base64']=base64.b64encode(zlib.compress((json.dumps(doc)+'\n').encode(),9)).decode()
    try:scope['qimg_native_compute_closure'](bad,request)
    except RuntimeError as e:checks.append({'name':'changed_actual_normal_release_owner_bytes_refused','passed':True,'reason':str(e)})
    else:raise AssertionError('changed normal release accepted')
    output={'schema':'neural_forge.k3_NEXT_QIMG_closed_compute_focused_metadata_checks.v1',
            'checks':checks,'passed':True,'pure_consumer_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
            'input_source_pins':pins,'actual_current_holder_absence_proven':False,
            'actual_current_normal_row_reobserved':False,'actual_next_admission_claimed':False,
            'candidate_main_imported_or_executed':False,'runtime_or_database_or_Relay_actions':False,
            'source49_changed':False}
    path=HERE/'ACTUAL_QIMG_COMPUTE_CLOSURE_FOCUSED_CHECKS_V1.json'
    with path.open('x',encoding='utf-8') as f:f.write(json.dumps(output,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':True}))


if __name__=='__main__':main()
