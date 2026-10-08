"""Metadata-only final contract and preserved focused-check correction."""
from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
    b=p.read_bytes();return {'path':str(p.resolve()),'sha256':sha(b),'bytes':len(b)}
def enc(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def write(p,b):
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert p.read_bytes()==b
def main():
    old=HERE/'ROOT_TEXT_WHOLE_CALLER_SOURCE_CONTRACT_V3.json'
    v=json.loads(old.read_bytes());v['schema']='neural_forge.k3_actual61_ROOT_serial_text_source_caller_proposal.v4'
    v['preserved_contract_before_complete_helper_pin_reseal']=pin(old)
    v['helpers']=json.loads((HERE/'launch/LAUNCH_HELPER_SOURCE_PINS_V1.json').read_bytes())
    new=HERE/'ROOT_TEXT_WHOLE_CALLER_SOURCE_CONTRACT_V4.json';write(new,enc(v))
    failed=HERE/'check_actual_ROOT_text_source_v3.py'
    check=failed.read_text()
    assert check.count("changed==['run','current_ROOT_serial_scope']")==1
    check=check.replace("changed==['run','current_ROOT_serial_scope']","set(changed)=={'run','current_ROOT_serial_scope'}")
    check=check.replace('ACTUAL_ROOT_TEXT_FOCUSED_CHECKS_V3.json','ACTUAL_ROOT_TEXT_FOCUSED_CHECKS_V4.json')
    corrected=HERE/'check_actual_ROOT_text_source_v4.py';write(corrected,check.encode())
    failure={'schema':'neural_forge.k3_owned_source_check_preserved_failure.v1','failed_source':pin(failed),'failed_check':'two permitted AST differences were compared in the wrong source-order; actual order current_ROOT_serial_scope then run','source_code_or_runtime_failure':False,'corrected_source':pin(corrected),'contract':pin(new)}
    write(HERE/'FOCUSED_CHECK_ORDER_FAILURE_PRESERVED_V1.json',enc(failure))
    print(json.dumps({'contract':pin(new),'check':pin(corrected),'helper_manifest':pin(HERE/'launch/LAUNCH_HELPER_SOURCE_PINS_V1.json')}))
if __name__=='__main__':main()
