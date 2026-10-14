"""Prepare/run only the selected private D1 boundary canary; never allocate GPUs.

The original trainer bytes and complete successor dataset stay unchanged. A
separately pinned candidate admits just the selected layer/step limits through
the existing allocator proof before every original optimizer apply.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from . import d1_allocated_contract as R
except ImportError:
    import d1_allocated_contract as R

V8_SHA = '2070b494e1a6e165e0af1a931eaf66195c272d365c576fb8f4576fd4e131b2c3'
CONFIG_SHA = '3516b587ce80461745b772eb92bf1d3c0ffd40fc5e0d82d83c037ad872c99b02'
PROFILE = dict(schema='chimera_recovery.d1_private_canary.v1',selected_date='2026-10-14',
    parent_candidate_sha256=V8_SHA,clone_config_sha256=CONFIG_SHA,
    layers='0,3',stop_after_layer=3,max_steps=1,wall_seconds=1800,
    complete_source_rows=979,source_width=4096,held_rows=62,
    source_rows_unchanged=True,optimizer_arithmetic_unchanged=True,
    private_outputs_only=True,production_resume_allowed=False,accepted_model=False,
    no_intelligence_or_full_depth_claim=True)


def profile(candidate,root=Path('/candidate')):
    value=R.read(Path(root)/'canary_profile.json',candidate['files']['canary_profile.json'])
    if value!=PROFILE or candidate.get('canary_profile')!={'path':'canary_profile.json','sha256':candidate['files']['canary_profile.json']}:
        raise R.Error('exact selected private canary profile required')
    if R.sha(__file__)!=candidate['files'].get('tools/d1_canary.py'):
        raise R.Error('actual private canary helper bytes differ')
    counts=candidate['actual_counts']
    if counts.get('total_rows')!=979 or counts.get('width')!=4096 or counts.get('held_rows')!=62:
        raise R.Error('private canary must preserve complete 979x4096 source and62 held rows')
    return value


def overrides(candidate,root=Path('/candidate')):
    value=profile(candidate,root)
    return {name:value[name] for name in ('layers','stop_after_layer','max_steps')}


def arguments(argv,candidate,root=Path('/candidate')):
    result=list(argv);selected=overrides(candidate,root)
    if result.count('--layers')!=1 or result[result.index('--layers')+1]!='all':
        raise R.Error('private canary requires original unmodified production layer selector')
    if '--max-steps' in result or '--stop-after-layer' in result or '--max-rows' in result:
        raise R.Error('unexpected source/step selector in private canary argv')
    result[result.index('--layers')+1]=selected['layers']
    return result+['--max-steps',str(selected['max_steps']),'--stop-after-layer',str(selected['stop_after_layer'])]


def actual_geometry(candidate,child,parent,root=Path('/candidate')):
    profile(candidate,root)
    if R.sha(child.root/'config.json')!=CONFIG_SHA:
        raise R.Error('actual frozen clone config differs from collected first-GQA proof')
    for checkpoint in (child,parent):
        kinds=[str(checkpoint.cfg.block_kind(i)) for i in checkpoint.text_blocks]
        if kinds[0]!='kda' or next((i for i,k in enumerate(kinds) if k=='gated_gqa'),None)!=3:
            raise R.Error('actual first KDA/GQA boundaries differ; new explicit canary selection required')


def fresh_outputs(run=Path('/runout'),delta=Path('/deltaout')):
    if (Path(run)/'canary-base-before.json').exists():
        raise R.Error('private canary already attempted; new explicit output namespace required')
    for path in (Path(run)/'trainer',Path(delta)/'trainer'):
        if path.exists() and any(path.iterdir()):
            raise R.Error('private canary cannot resume or overwrite an existing trainer/delta')


def begin(candidate,args,result):
    profile(candidate)
    from oc_seal import directory_digest
    parent_digest,parent_files=directory_digest(Path(args.parent))
    record=dict(schema='chimera_recovery.d1_canary_base_digests.v1',
        clone_before=result['clone_directory_sha256'],parent_before=parent_digest,
        parent_files=len(parent_files),model_paths={'child':args.child,'parent':args.parent},
        source_manifest_sha256=args.source_manifest_sha256,production_accepted=False)
    path=Path('/runout/canary-base-before.json')
    if path.exists():raise R.Error('private canary before-receipt already exists; no implicit resume')
    path.write_bytes(R.canonical(record)+b'\n')
    return record


def finish(candidate,args,before):
    profile(candidate)
    from oc_seal import directory_digest
    child,_=directory_digest(Path(args.child));parent,_=directory_digest(Path(args.parent))
    record=dict(before,clone_after=child,parent_after=parent,
        bases_full_byte_digest_equal=child==before['clone_before'] and parent==before['parent_before'])
    receipt=Path(args.out)/'d1_receipt.json'
    if receipt.is_file():
        actual=json.loads(receipt.read_text(encoding='utf-8'))
        layers=actual.get('layers',[])
        trained=[r for r in layers if r.get('trained') is True]
        record['actual_trained_layer_steps']={str(r['text_layer']):r.get('record',{}).get('steps') for r in trained}
        record['complete_two_selected_updates']=record['actual_trained_layer_steps']=={'0':1,'3':1}
        record['trainer_receipt']={'path':str(receipt),'sha256':R.sha(receipt)}
        record['measured_scope']='original per-layer loss/gradient/held residual/VRAM/RSS; private canary through layer3'
    Path('/runout/canary-base-after.json').write_bytes(R.canonical(record)+b'\n')
    if not record['bases_full_byte_digest_equal']:raise R.Error('private canary changed a protected model base')
    return record


def replace_once(raw,before,after,name):
    text=raw.decode('utf-8')
    if text.count(before)!=1:raise R.Error('exact reviewed canary source anchor differs: '+name)
    result=text.replace(before,after,1).encode()
    compile(result,name,'exec')
    return result


def prepare(baseline,out,runtime_candidate,lease_source,run_source,delta_source):
    baseline=Path(baseline);out=Path(out)
    parent=R.read(baseline/'candidate_receipt.json',V8_SHA)
    if out.exists():raise R.Error('new immutable private canary namespace required')
    files={name:(baseline/name).read_bytes() for name in parent['files']}
    if any(hashlib.sha256(raw).hexdigest()!=parent['files'][name] for name,raw in files.items()):
        raise R.Error('original selected v8 source capsule changed')
    files['tools/d1_canary.py']=Path(__file__).read_bytes()
    files['canary_profile.json']=R.canonical(PROFILE)+b'\n'
    files['evidence/production-v8-candidate.json']=(baseline/'candidate_receipt.json').read_bytes()
    key='tools/d1_allocated_contract.py'
    files[key]=replace_once(files[key],
        '        if any(getattr(args, key, None) != value for key, value in required.items()):',
        '        if candidate.get("canary_profile"):\n            import d1_canary\n'
        '            required.update(d1_canary.overrides(candidate, candidate_root))\n'
        '        if any(getattr(args, key, None) != value for key, value in required.items()):',key)
    key='tools/d1_allocated_entry.py'
    files[key]=replace_once(files[key],
        '    return ["--child",', '    argv = ["--child",',key)
    files[key]=replace_once(files[key],
        '"--delta-out", "/deltaout/trainer"]\n',
        '"--delta-out", "/deltaout/trainer"]\n'
        '    if candidate.get("canary_profile"):\n        import d1_canary\n'
        '        argv = d1_canary.arguments(argv, candidate)\n    return argv\n',key)
    files[key]=replace_once(files[key],
        '    for j, k in zip(child.text_blocks, parent.text_blocks):',
        '    candidate = candidate_record(assignment["candidate_receipt_sha256"])\n'
        '    if candidate.get("canary_profile"):\n        import d1_canary\n'
        '        d1_canary.actual_geometry(candidate, child, parent)\n'
        '    for j, k in zip(child.text_blocks, parent.text_blocks):',key)
    files[key]=replace_once(files[key],
        '    candidate = candidate_record(args.candidate_receipt_sha256)\n',
        '    candidate = candidate_record(args.candidate_receipt_sha256)\n'
        '    if args.run and candidate.get("canary_profile"):\n        import d1_canary\n'
        '        d1_canary.fresh_outputs()\n',key)
    files[key]=replace_once(files[key],
        '        trainer.run(argv)',
        '        if candidate.get("canary_profile"):\n            import d1_canary\n'
        '            before = d1_canary.begin(candidate, training_args, result)\n'
        '            try:\n                trainer.run(argv)\n'
        '            finally:\n                result["private_canary"] = d1_canary.finish(candidate, training_args, before)\n'
        '        else:\n            trainer.run(argv)',key)
    key='tools/d1_allocated_root.py'
    files[key]=replace_once(files[key],
        " if 'd1_allocated_entry.py' in own: raise ValueError('existing D1 entry still active')",
        " if 'd1_allocated_entry.py' in own or 'd1_canary.py watch' in own: raise ValueError('existing private D1 entry/watch still active')",key)
    files[key]=replace_once(files[key],
        "'/candidate/tools/d1_allocated_entry.py',r['mode'],",
        "'/candidate/tools/d1_canary.py','watch',r['mode'],",key)
    candidate=dict(parent,prepared_date='2026-10-14',canary_only=True,production_ready=False,
        canary_profile={'path':'canary_profile.json','sha256':hashlib.sha256(files['canary_profile.json']).hexdigest()},
        files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
        registration_scope='Private capacity canary only; never register/accept/resume as productionD1')
    plan=R.read(baseline/'allocation_plan.json',R.sha(baseline/'allocation_plan.json'))
    replacements=dict(candidate=runtime_candidate,lease=lease_source,run=run_source,delta=delta_source)
    for item in plan['mounts']:
        if item['role'] in replacements:item['source']=replacements[item['role']]
    # Rebuild the unexecuted allocator argv from the exact new declared mounts.
    split=plan['allocation_command_template'].index('--mount-ro')
    oldargv=plan['allocation_command_template']
    tail=oldargv[oldargv.index('--runtime-arg'):]
    plan['allocation_command_template']=oldargv[:split]+[v for m in plan['mounts']
        for v in ('--mount-ro' if m['read_only'] else '--mount-rw',m['source'],m['target'])]+tail
    plan.update(candidate_receipt_sha256=hashlib.sha256(R.canonical(candidate)+b'\n').hexdigest(),
        root_controller_sha256=candidate['files']['tools/d1_allocated_root.py'],
        canary_only=True,canary_profile=candidate['canary_profile'],wall_seconds=1800)
    files['candidate_receipt.json']=R.canonical(candidate)+b'\n'
    files['allocation_plan.json']=R.canonical(plan)+b'\n'
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name+'.',dir=out.parent) as name:
        temporary=Path(name)
        for relative,raw in files.items():
            path=temporary/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        temporary.rename(out)
    return {'candidate':str(out),'candidate_sha256':R.sha(out/'candidate_receipt.json'),
        'plan_sha256':R.sha(out/'allocation_plan.json'),'training_launched':False,'production_ready':False}


def watch(argv):
    """Bound this own child process; preserve an explicit timeout result."""
    import d1_allocated_entry as E
    args=E.parser().parse_args(argv)
    candidate=E.candidate_record(args.candidate_receipt_sha256)
    selected=profile(candidate)
    try:
        process=subprocess.run([sys.executable,'/candidate/tools/d1_allocated_entry.py',*argv],
            timeout=selected['wall_seconds'],check=False)
        return process.returncode
    except subprocess.TimeoutExpired:
        value=dict(completed=True,exit_code=124,error_type='CanaryWallTimeout',
            error='Selected1800-second private canary limit reached; no accepted model produced',
            execution_id=str(args.execution_id),assignment_sha256=args.runtime_assignment_sha256,
            mode='run' if args.run else 'check-only',private_canary=True,production_accepted=False)
        path=Path('/runout/runtime-status.json.incoming');path.write_bytes(R.canonical(value)+b'\n')
        path.replace('/runout/runtime-status.json')
        return 124


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='watch':return watch(argv[1:])
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('baseline','out','runtime-candidate','lease-source','run-source','delta-source'):
        parser.add_argument('--'+name,required=True)
    try:print(json.dumps(prepare(**vars(parser.parse_args(argv))),sort_keys=True));return 0
    except (ValueError,OSError,KeyError) as exc:print(json.dumps({'refused':str(exc),'training_launched':False}));return 2


if __name__=='__main__':raise SystemExit(main())
