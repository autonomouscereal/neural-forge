"""Portable existing61 source checks only; no tensor/runtime/remote operations."""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path

ROSTER_SHA='f9f74e98e63f8655b11cd9babe3773aa05ec8f9e5c8a89571d1f55da704102c6'
def sha(x):return hashlib.sha256(x).hexdigest()
def functions(raw):
    tree=ast.parse(raw);out={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):out[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.ClassDef):
            for child in node.body:
                if isinstance(child,ast.FunctionDef):out[node.name+'.'+child.name]=ast.dump(child,include_attributes=False)
    return out
def run(root):
    root=root.resolve();raw=(root/'PACKET_SOURCE_ROSTER_V1.json').read_bytes();assert sha(raw)==ROSTER_SHA
    roster=json.loads(raw)
    assert roster['source_only'] is True and roster['deployable'] is False
    for name,row in roster['members'].items():
        path=root/name;assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
        content=path.read_bytes();assert sha(content)==row['sha256'] and len(content)==row['bytes'],name
        if name.endswith('.py'):compile(content,name,'exec')
    selected=roster['selected_sources'];assert len(selected)==11
    writer=json.loads((root/'evidence/ACTUAL_HEADER_61_72_WRITE_AND_ARRAY_COMPONENTS_V1.json').read_bytes())
    whole=writer['writer']['conservative_complete_source_scope_extent_bytes'];assert whole==166556705792
    assert max(692026605568+whole,690415992832+whole)==858583311360
    assert 25769803776+16*(1<<30)+whole==209506378752
    native=functions((root/selected['native']).read_bytes())
    qualified=functions((root/'evidence/k3_dense_pass_9086.py').read_bytes())
    validation=json.loads((root/'evidence/SOURCE_ONLY_PROPOSAL_VALIDATION_V2.json').read_bytes())
    names=validation['original_qualified_functions_AST_unchanged']
    assert len(names)==len(set(names))==99 and all(native[n]==qualified[n] for n in names)
    for role,allowed in [('facts',set()),('host',{'main'}),('preflight',{'host','current_NAS_scope','run'})]:
        old=ast.parse((root/('evidence/parent49_'+role+'.py')).read_bytes());new=ast.parse((root/selected[role]).read_bytes())
        f=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=f(old),f(new);assert set(before)==set(after) and {n for n in before if before[n]!=after[n]}==allowed
    for role in ('facts','host','preflight'):
        result=subprocess.run([sys.executable,'-B',str(root/selected[role]),'--help'],capture_output=True,text=True,timeout=15)
        assert result.returncode!=0 and 'badly formed hexadecimal UUID string' in result.stderr
    inputs=root/'binding_proposal/ACTUAL61_72_INPUTS_UNRESOLVED_V1.json';out=root/'must_not_be_created'
    assert not out.exists()
    result=subprocess.run([sys.executable,'-B',str(root/'binding_proposal/bind_actual61_72_source_proposal_v4.py'),
        '--inputs',str(inputs),'--inputs-sha256',sha(inputs.read_bytes()),'--output-directory',str(out)],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'new actual61 UUID remains unresolved' in result.stderr and not out.exists()
    print(json.dumps({'schema':'neural_forge.k3_window61_72_normal_source_packet_check.v1','PASS':True,
        'raw_member_checks':len(roster['members']),'selected_source_pin_and_compile_count':11,
        'qualified_other_AST_unchanged_count':99,'existing_caller_AST_guard_checks':3,
        'existing_unresolved_module_refusal_checks':3,'existing_unresolved_binder_refusal_checks':1,
        'authentic_header_arithmetic_check':True,'actual60_closure_or_cap_or_admission_claimed':False,
        'remote_tensor_runtime_resource_actions':False}))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate-root',type=Path,required=True)
    run(p.parse_args().candidate_root)
if __name__=='__main__':main()
