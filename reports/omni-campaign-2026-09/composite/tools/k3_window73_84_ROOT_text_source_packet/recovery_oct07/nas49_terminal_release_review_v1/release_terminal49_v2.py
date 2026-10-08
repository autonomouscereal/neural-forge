"""Existing normal exact-group terminal release; retain the physical terminal container."""
import asyncio, base64, hashlib, importlib.util, json, os, sys, uuid, zlib
from pathlib import Path
D=Path(__file__).resolve().parent
K3=D.parent.parent
PHASE=D.parent/"recovery_resource_review/actual49_60_binding_proposal_v1/launch_receipts_v1"
sys.path.insert(0,str(K3))
import infra
from infra import load, native_session, now, record, sha
infra.ROOT=D

def module(name,path,pin):
    assert sha(path)==pin,str(path)
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def decode_terminal_evidence(transport, allocation):
    assert transport['transport']=='zlib.base64.complete-owned-terminal-metadata.v1'
    assert transport['container_id']==allocation['container_id'] and transport['allocation_id']==allocation['allocation_id']
    for key in ('uncompressed_bytes','compressed_bytes'):
        assert type(transport[key]) is int and transport[key]>0,key
    packed=base64.b64decode(transport['compressed_base64'],validate=True)
    assert len(packed)==transport['compressed_bytes'] and hashlib.sha256(packed).hexdigest()==transport['compressed_sha256']
    inflater=zlib.decompressobj()
    expanded=inflater.decompress(packed,transport['uncompressed_bytes']+1)
    assert inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail
    assert len(expanded)==transport['uncompressed_bytes'] and hashlib.sha256(expanded).hexdigest()==transport['uncompressed_sha256']
    physical=json.loads(expanded)
    assert physical['ok'] is True and physical['container_id']==allocation['container_id']
    evidence=physical['terminal_evidence']
    assert evidence['complete'] is True and evidence['model_artifact_success_claimed'] is False and evidence['tensor_payload_reads']==0
    members=[*evidence['files'],*evidence['docker_logs'].values()]
    for member in members:
        raw=base64.b64decode(member['raw_base64'],validate=True)
        assert type(member['bytes']) is int and len(raw)==member['bytes']
        assert hashlib.sha256(raw).hexdigest()==member['sha256']
    assert len({row['path'] for row in evidence['files']})==len(evidence['files'])
    assert set(evidence['docker_logs'])=={'stdout','stderr'}
    return physical,expanded


async def main():
    assert sha(PHASE/'identity.json')=='1e434880c63de9afbfb7185a5daa151e8f39466133874e27aa14076c04408bd1'
    assert sha(PHASE/'actual_allocation.json')=='43524885b09a34bfb8bc87055b799ffd74b47d706bb09dc7d7f24568bc63c387'
    assert sha(K3/'initial_snapshot.json')=='48f4c5de740a8bffaa924680472601a0add04b522478f242c3cf35681ce98f0d'
    assert sha(PHASE/'freeze-complete.json')=='9826e148359c7073b84d36da321fb89ef9f98d210b029e1f592ca064f96567bd'
    i=load(PHASE/'identity.json')
    a=load(PHASE/'actual_allocation.json')
    frozen=load(PHASE/'freeze-complete.json')['result']['freeze']['result']
    assert i['run_uuid']=='03603908-570d-4c90-96cb-ee3a63a9a139'
    assert frozen['source_freeze_sha256']=='f17c6912d1e3ba5f62f7943ba9f8e004d9c8def3be54e5820bce174899110647'
    binding,binding_receipt=native_session(['connect'])
    assert {k:binding[k] for k in i['native_binding']}==i['native_binding']
    assert a['allocation_id']=='aebe07fe-2905-470c-8a10-f683a770faeb'
    assert a['container_id']=='f924bdf0fa81ff21e26b791623a115e6812bd579b3a1d048b69b451ddf0a7514'
    normal=module('k3_exact_installed_allocator','D:/relay/worktrees/local-hermes-collaboration/best/local-hermes/hermes.py','97b58088b555b4669b94c15649d370dfd4dacd28dd6e3a430ad417522424117e')
    memory=module('agent_memory','C:/Users/cereal/.claude/memory_backend/agent_memory.py','4de3abea6de7a58d4d78c354a692ffb1f125bcdd6571fa2f99898a1d4d6a92fa')
    contract=module('k3_existing_catalog',K3.parents[2]/'runtime_admission_consistency_oct07/k3_generation5_publication_and_no_ddl_review_v1/allocator_catalog_contract.py','fad65fc7ed687f24c64f411cfb1907629fef3dd290ce436d25231909184b48c6')
    wire=module('k3_reviewed_local_packed_wire',PHASE.parent/'local_launch49_60_v1/launch_actual49_60_transport_v4.py','f1b34b09a1eed028a6cd3440fffb7e96f54aed14c84faa6607002606b2156378')
    conn=await memory.connect()
    result={'at':now(),'native_binding':i['native_binding'],'binding_receipt':str(binding_receipt),'actual_allocation':a,'preserve_physical_container':True,'provider_mutation':False,'reconcile_or_stop_or_remove':False}
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL lock_timeout='10s'")
            await conn.execute("SET LOCAL statement_timeout='20s'")
            await conn.execute('SELECT pg_advisory_xact_lock(hashtext($1))','gpu-runtime:ai')
            catalog=dict(await conn.fetchrow(contract.CATALOG_SQL));contract.validate_catalog(catalog)
            result['catalog']=catalog
            rows=[dict(r) for r in await conn.fetch('SELECT * FROM gpu_runtime_allocations WHERE server=$1 AND COALESCE(group_id,allocation_id)=$2 FOR UPDATE','ai',uuid.UUID(a['allocation_id']))]
            assert len(rows)==1
            row=rows[0]
            for k in ('owner','name','container_id','gpu_uuid','gpu_index','kind','state','details'):
                assert row[k]==a[k],k
            assert row['allocation_id']==row['group_id']==uuid.UUID(a['allocation_id'])
            result['before']=rows
            assert sha(D/'release_terminal49_readonly_v2.py')=='b6e099ba0f2662cf68df17c4443afeb819d16b35011f4a53df10212d9815c814'
            transport,receipt,status=wire.packed_control(infra,D/'release_terminal49_readonly_v2.py',{'I':i,'A':a,'BASELINE':load(K3/'initial_snapshot.json'),'FROZEN':frozen},timeout=25,label='NAS49-60-exact-owned-terminal-physical-boundary')
            physical,expanded=decode_terminal_evidence(transport,a)
            expanded_path=D/'actual_terminal_evidence49_60_v2.json'
            with expanded_path.open('xb') as handle:
                handle.write(expanded);handle.flush();os.fsync(handle.fileno())
            assert expanded_path.read_bytes()==expanded and sha(expanded_path)==transport['uncompressed_sha256']
            result['complete_expanded_terminal_evidence']={'path':str(expanded_path),'bytes':len(expanded),'sha256':sha(expanded_path),'all_member_bytes_and_hashes_verified':True}
            result['physical_boundary']=physical;result['physical_receipt']=str(receipt)
            assert physical['ok'] is True and physical['terminal_evidence']['complete'] is True
            assert physical['container_id']==a['container_id'] and physical['terminal_evidence']['model_artifact_success_claimed'] is False
            precommit=record('NAS49-60-exact-own-normal-release-precommit',result)
            result['terminal_evidence_precommit']={'path':str(precommit),'sha256':sha(precommit)}
            await normal.release_group(conn,'ai',uuid.UUID(a['allocation_id']))
            result['transaction_after']=[dict(r) for r in await conn.fetch('SELECT * FROM gpu_runtime_allocations WHERE allocation_id=$1',uuid.UUID(a['allocation_id']))]
            assert result['transaction_after'][0]['state']=='released'
        result['committed_after']=[dict(r) for r in await conn.fetch('SELECT * FROM gpu_runtime_allocations WHERE allocation_id=$1',uuid.UUID(a['allocation_id']))]
        assert result['committed_after'][0]['state']=='released'
        result.update(at=now(),completed=True,normal_function='unchanged installed release_group')
        p=record('NAS49-60-exact-own-preserving-normal-release',result)
        with (D/'actual_terminal_release49_60_v2.json').open('x',encoding='utf-8') as handle:
            handle.write(json.dumps({'receipt':str(p),'sha256':sha(p),'result':result},indent=2,default=str)+'\n')
        print(json.dumps({'receipt':str(p),'sha256':sha(p),'completed':True,'allocation_id':a['allocation_id'],'container_preserved':True}))
    finally:await conn.close()

if __name__=='__main__':asyncio.run(main())
