"""Root-only D1 PG/container binding and lease monitor; never acquire GPUs.

Uses the installed raw-PostgreSQL memory connection and server-manager transport.
Bind only an existing exact-task generic allocation. Execute requires a separate
explicit check/run selector and maintains a fresh receipt until the job finishes.
No model registration, allocation release, process killing, or host CUDA fallback.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
import uuid

try:
    from . import d1_allocated_contract as R
    from . import d1_allocated as B
except ImportError:
    import d1_allocated_contract as R
    import d1_allocated as B

SELECT = """SELECT allocation_id,group_id,server,gpu_uuid,gpu_index,owner,name,kind,container_id,state,details
FROM gpu_runtime_allocations WHERE server=$1 AND owner=$2 AND COALESCE(group_id,allocation_id)=$3 ORDER BY gpu_index"""

# Only fixed metadata/own-job operations; no environment/private contexts returned.
REMOTE = r'''
import base64,hashlib,json,os,pathlib,re,subprocess,sys,time
r=json.loads(base64.b64decode(sys.argv[1])); cid=r['container_id']
if not re.fullmatch('[a-f0-9]{64}',cid): raise ValueError('full exact container ID required')
def cmd(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=15)
 if p.returncode: raise RuntimeError('bounded runtime operation failed: '+argv[0])
 return p.stdout
def root(role): return pathlib.Path(next(m['source'] for m in r['mounts'] if m['role']==role))
action=r['action']; out={}
if action=='snapshot':
 d=json.loads(cmd(['docker','inspect',cid]))[0]
 labels=d.get('Config',{}).get('Labels',{}) or {}
 out['docker']={k:d[k] for k in ['Id','Image','Name','State','Mounts']}
 out['docker']['Config']={'Labels':{k:v for k,v in labels.items() if k.startswith('ai.local.')}}
 out['docker']['HostConfig']={'DeviceRequests':d['HostConfig'].get('DeviceRequests',[])}
 for m in r['mounts']:
  if str(pathlib.Path(m['source']).resolve(strict=True))!=m['source']: raise ValueError('symlink/alias mount source')
 ownpids={int(x) for x in cmd(['docker','top',cid,'-eo','pid']).splitlines()[1:] if x.strip().isdigit()}
 processes=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).splitlines()
 foreign=[]
 for line in processes:
  fields=[x.strip() for x in line.split(',')]
  if len(fields)==2 and fields[0] in r['gpu_uuids'] and int(fields[1]) not in ownpids:
   foreign.append({'gpu_uuid':fields[0],'pid':int(fields[1])})
 def alive(pattern):
  p=subprocess.run(['pgrep','-f',pattern],capture_output=True,timeout=5)
  if p.returncode not in (0,1): raise RuntimeError('process census unavailable')
  return p.returncode==0
 mem={x.split(':')[0]:int(x.split()[1]) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
 tree=root('chimera');paths=sorted((tree/'src').rglob('*.py'))
 if (tree/'COMMIT').is_file(): paths.append(tree/'COMMIT')
 if not paths or any(p.is_symlink() or not p.resolve().is_relative_to(tree.resolve()) for p in paths): raise ValueError('native source alias/missing')
 pairs=sorted([p.relative_to(tree).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()] for p in paths)
 source_sha=hashlib.sha256(json.dumps(pairs,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 status=pathlib.Path('/home/cereal/omni-campaign/distill/queues/gpu6.status')
 states={}
 if status.is_file():
  if status.stat().st_size>1048576: raise ValueError('bounded K3 status size exceeded')
  for line in status.read_text().splitlines():
   fields=line.split()
   if len(fields)>2 and fields[1] in ['k3-pass-0','k3-pass-1','k3-pass-2','k3-pass-3']: states[fields[1]]=fields[2]
 sealed=0
 for path in pathlib.Path('/mnt/lexar/omni-campaign/teacher_cache/k3').rglob('manifest.json'):
  if path.stat().st_size<2097152 and json.loads(path.read_text()).get('verdict')=='sealed': sealed+=1
 out['host']={'phase':r['phase'],'mem_available_gib':mem['MemAvailable']/1048576,
  'k3_running':alive(r'k3_teacher_cache[^ /]*\.py run'),
  'selected_gpu_queues_running':alive(r'(^|[ /])gpu[04][^ /]*\.sh( |$)'),
  'k3_passes_complete':len(states)==4 and all(v in ('PASS','DONE') for v in states.values()),
  'k3_sealed_manifests':sealed,'foreign_gpu_processes':foreign,'chimera_source_sha256':source_sha}
 if r.get('probe_cuda'):
  code="import sys,json;sys.path.insert(0,'/candidate/tools');from d1_allocated_entry import cuda_evidence;print(json.dumps(cuda_evidence(),sort_keys=True))"
  out['cuda']=json.loads(cmd(['docker','exec',cid,'python3','-c',code]))
elif action=='publish':
 lease=root('lease')
 if not str(lease).startswith('/home/cereal/omni-campaign/distill/runtime_leases/D1-'): raise ValueError('exclusive D1 lease namespace required')
 for name,value in r['records'].items():
  if name not in ('assignment.json','live.json'): raise ValueError('bounded lease record name required')
  temporary=lease/(name+'.incoming');temporary.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
  os.chmod(temporary,0o600);os.replace(temporary,lease/name)
 out={'published':True}
elif action=='start':
 # Refuse another still-running entry after a controller crash; do not kill it.
 own=cmd(['docker','top',cid,'-eo','pid,args'])
 if 'd1_allocated_entry.py' in own: raise ValueError('existing D1 entry still active')
 argv=['docker','exec','-d',cid,'python3','/candidate/tools/d1_allocated_entry.py',r['mode'],
  '--runtime-assignment','/lease/assignment.json','--runtime-assignment-sha256',r['assignment_sha256'],
  '--candidate-receipt-sha256',r['candidate_receipt_sha256'],'--execution-id',r['execution_id']]
 cmd(argv);out={'started':True}
elif action=='status':
 path=root('run')/'runtime-status.json'
 if path.exists():
  if path.stat().st_size>4194304: raise ValueError('bounded own-job status size exceeded')
  value=json.loads(path.read_text())
  if value.get('execution_id')==r['execution_id']:
   out={k:value.get(k) for k in ('execution_id','assignment_sha256','mode','completed','exit_code','error_type','error','ended_at')}
else: raise ValueError('unsupported bounded D1 action')
print('D1_ROOT_RESULT='+json.dumps(out,sort_keys=True))
'''


def remote(request):
    """Server-manager is the sole transport; never invoke a raw SSH program."""
    sys.path.insert(0, os.environ.get("SERVER_MANAGER_SKILL_DIR", "C:/Users/cereal/.agents/skills/server-manager"))
    from ssh_client import SSHClient, ServerConfig
    token = base64.b64encode(R.canonical(request)).decode()
    with contextlib.redirect_stdout(io.StringIO()):
        connection = SSHClient.from_config(ServerConfig().get("ai"), control=True)
        if not connection.connect(timeout=15):
            raise R.Error("D1 root server-manager transport unavailable")
        try:
            code, output, _ = connection.execute(shlex.join(["python3", "-c", REMOTE, token]), timeout=25)
        finally:
            connection.close()
    if code:
        raise R.Error("D1 bounded server-manager operation failed; no live proof issued")
    values = [line[len("D1_ROOT_RESULT="):] for line in output.splitlines() if line.startswith("D1_ROOT_RESULT=")]
    if len(values) != 1:
        raise R.Error("D1 exact bounded remote result missing")
    return json.loads(values[0])


async def connection():
    sys.path.insert(0, os.environ.get("AGENT_MEMORY_BACKEND", "C:/Users/cereal/.claude/memory_backend"))
    import agent_memory
    # No controller.connect(): it creates/changes tables. This path is reads + advisory consumer lock only.
    return await asyncio.wait_for(agent_memory.connect(), 12)


async def allocation_rows(conn, group):
    rows = await conn.fetch(SELECT, "ai", B.OWNER, uuid.UUID(group), timeout=8)
    return [{key: str(value) if isinstance(value, uuid.UUID) else value for key, value in dict(row).items()} for row in rows]


def request(assignment, action, **extra):
    return dict(action=action, container_id=assignment["container_id"], mounts=assignment["mounts"],
                gpu_uuids=[gpu["uuid"] for gpu in assignment["gpus"]], **extra)


async def refresh(conn, assignment, assignment_sha, cuda, phase):
    if R.sha(__file__) != assignment["root_controller_sha256"]:
        raise R.Error("D1 root controller source changed; no live update proof")
    captured = datetime.now(timezone.utc)
    rows = await allocation_rows(conn, assignment["allocation_group_id"])
    # PG and Docker both have to prove the exact identity on every refresh.
    snapshot = await asyncio.to_thread(remote, request(assignment, "snapshot", phase=phase, probe_cuda=False))
    proof = R.attest(assignment, assignment_sha, rows, snapshot["docker"], cuda, snapshot["host"], now=captured)
    await asyncio.to_thread(remote, request(assignment, "publish", records={"live.json": proof}))
    return proof


async def guarded_cuda_probe(conn, assignment):
    # Never initialize CUDA until current PG membership, Docker access and host gates are proven.
    rows = await allocation_rows(conn, assignment["allocation_group_id"])
    R.verify_pg(assignment, rows)
    metadata = await asyncio.to_thread(remote, request(assignment, "snapshot", phase="launch", probe_cuda=False))
    R.verify_docker(assignment, metadata["docker"])
    R.verify_host(assignment, metadata["host"])
    R.verify_pg(assignment, await allocation_rows(conn, assignment["allocation_group_id"]))
    return await asyncio.to_thread(remote, request(assignment, "snapshot", phase="launch", probe_cuda=True))


async def bind(args):
    plan = R.read(args.plan, args.plan_sha256)
    if (plan.get("schema") != "chimera_recovery.d1_allocation_plan.v1" or plan.get("owner") != B.OWNER
            or plan.get("image_id") != B.IMAGE or plan.get("root_controller_sha256") != R.sha(__file__) or args.out.exists()):
        raise R.Error("Exact new reviewed D1 allocation binding required")
    conn = await connection()
    try:
        captured = datetime.now(timezone.utc)
        rows = await allocation_rows(conn, str(args.group))
        if len(rows) != 2:
            raise R.Error("Actual D1 allocator group must have exactly two rows")
        assignment = {key: plan[key] for key in ("owner", "server", "stage", "native_task", "image_id", "mounts",
                     "clone_directory_sha256", "mem_available_launch_floor_gib", "max_proof_age_seconds",
                     "container_devices", "lease_proof_target", "lease_mount_read_only", "candidate_receipt_sha256", "root_controller_sha256")}
        assignment.update(schema=R.SCHEMA, active=True, allocation_group_id=str(args.group),
                          container_id=rows[0]["container_id"], container_name=rows[0]["name"],
                          gpus=[dict(host_index=row["gpu_index"], uuid=row["gpu_uuid"], container_ordinal=i) for i, row in enumerate(rows)])
        R.verify_pg(assignment, rows)
        metadata = await asyncio.to_thread(remote, request(assignment, "snapshot", phase="launch", probe_cuda=False))
        assignment["chimera_source_sha256"] = metadata["host"]["chimera_source_sha256"]
        snapshot = await guarded_cuda_probe(conn, assignment)
        assignment["torch_version"] = snapshot["cuda"]["torch_version"]
        assignment["initial_cuda_evidence"] = snapshot["cuda"]
        rows = await allocation_rows(conn, str(args.group))
        # Validate the exact published schema, then cross-check authentic live evidence.
        raw = R.canonical(assignment) + b"\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pending = args.out.with_name(args.out.name + ".incoming")
        pending.write_bytes(raw)
        digest = R.sha(pending)
        try:
            R.assignment_record(pending, digest)
            proof = R.attest(assignment, digest, rows, snapshot["docker"], snapshot["cuda"], snapshot["host"], now=captured)
            await asyncio.to_thread(remote, request(assignment, "publish", records={"assignment.json": assignment, "live.json": proof}))
            pending.rename(args.out)
        finally:
            pending.unlink(missing_ok=True)
        return dict(assignment=str(args.out), assignment_sha256=digest, container_id=assignment["container_id"],
                    group=str(args.group), bound=True, training_launched=False)
    finally:
        await conn.close()


async def execute(args):
    assignment = R.assignment_record(args.assignment, args.sha256)
    if assignment["root_controller_sha256"] != R.sha(__file__):
        raise R.Error("D1 execution requires exact published root controller source")
    mode = "--run" if args.run else "--check-only"
    execution_id = str(uuid.uuid4())
    conn = await connection()
    locked = False
    started = False
    try:
        locked = await conn.fetchval("SELECT pg_try_advisory_lock(hashtext($1))", "d1-root-consumer:" + assignment["allocation_group_id"], timeout=8)
        if not locked:
            raise R.Error("Another root controller already monitors this exact D1 group")
        snapshot = await guarded_cuda_probe(conn, assignment)
        cuda = snapshot["cuda"]
        if cuda != assignment["initial_cuda_evidence"]:
            raise R.Error("D1 actual CUDA runtime differs from bound immutable assignment")
        await refresh(conn, assignment, args.sha256, cuda, "launch")
        await asyncio.to_thread(remote, request(assignment, "start", mode=mode, execution_id=execution_id,
                              assignment_sha256=args.sha256, candidate_receipt_sha256=assignment["candidate_receipt_sha256"]))
        started = True
        began = time.monotonic()
        seen = False
        while True:
            # Initial 100GiB floor was checked immediately before start. Host buffers legitimately consume it afterward.
            await refresh(conn, assignment, args.sha256, cuda, "training")
            status = await asyncio.to_thread(remote, request(assignment, "status", execution_id=execution_id))
            if status:
                seen = True
                if status.get("assignment_sha256") != args.sha256 or status.get("mode") != mode[2:]:
                    raise R.Error("D1 own-job completion belongs to another assignment/mode")
                if status.get("completed"):
                    return dict(status, training_launched=args.run, controller_scope="exact group; allocation remains root-owned")
            if not seen and time.monotonic() - began > 60:
                raise R.Error("D1 entry did not publish its own startup status")
            await asyncio.sleep(10)
    finally:
        if locked:
            # Revocation/monitor exit removes authority for further updates, including when transport or PG fails.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(remote, request(assignment, "publish", records={"live.json": dict(
                    schema=R.PROOF, valid=False, captured_at=datetime.now(timezone.utc).isoformat(),
                    assignment_sha256=args.sha256, reason="root monitor ended; no optimizer authority", started=started)}))
            with contextlib.suppress(Exception):
                await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", "d1-root-consumer:" + assignment["allocation_group_id"], timeout=8)
        await conn.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bind", allow_abbrev=False)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--group", type=uuid.UUID, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("execute", allow_abbrev=False)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    p.add_argument("--assignment", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    return result


def main():
    args = parser().parse_args()
    print(json.dumps(asyncio.run(bind(args) if args.command == "bind" else execute(args)), sort_keys=True))


if __name__ == "__main__":
    main()
