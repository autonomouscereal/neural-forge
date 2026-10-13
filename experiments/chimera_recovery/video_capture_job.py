"""Run one source-pinned bounded video teacher job inside a root-owned allocation.

This entry point never allocates a GPU or launches a container. The root supplies
an immutable reviewed request and the actual allocator receipt. Preparation and
teacher capture do not imply native student readiness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

try:
    from . import video_answer as B
    from . import video_stock_control as S
except ImportError:
    import video_answer as B
    import video_stock_control as S


def pinned_request(path,sha,job):
    p=B.A.pinned({'path':str(path),'sha256':sha},'reviewed bounded video request');r=json.loads(p.read_text())
    if (r.get('schema')!='recovery_video_teacher_request.v1' or r.get('sealed') is not True or
            r.get('training_started') is not False or r.get('gpu_launch_performed') is not False or
            B.A.digest({k:v for k,v in r.items() if k!='manifest_sha256'})!=r.get('manifest_sha256') or
            job not in r['jobs'] or r['jobs'][job]['command']!=job):
        raise ValueError('exact sealed capture/control request required')
    for spec in r['sources'].values():B.A.pinned(spec,'reviewed source capsule member')
    for spec in r['inputs'].values():B.A.pinned(spec,'reviewed teacher input')
    return r


def main(argv=None):
    import resource
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--job',choices=('capture','control'),required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.receipt.exists():raise ValueError('fresh actual execution receipt required')
    request=pinned_request(args.request,args.request_sha256,args.job);budget=request['resource_policy']
    started=time.perf_counter();failure=None;code=1;runtime={}
    def expired(*_):raise TimeoutError('1200-second bounded video job budget expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(budget['timeout_seconds'])
    try:
        import torch,transformers,accelerate,shutil,sysconfig
        if (os.getuid()!=1000 or torch.__version__!='2.11.0+cu128' or transformers.__version__!='5.12.1' or
                accelerate.__version__!='1.14.0' or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):
            raise ValueError('actual reviewed runtime and one allocated remapped BF16 GPU required')
        raw=str(torch.cuda.get_device_properties(0).uuid);gpu_uuid='GPU-'+str(uuid.UUID(raw.removeprefix('GPU-')))
        if gpu_uuid!=budget['gpu_uuid']:raise ValueError('actual sole GPU differs from reviewed allocation request')
        runtime=S.require_budget(S.memory_available(),torch.cuda.mem_get_info()[0])
        runtime.update(gpu_uuid=gpu_uuid,torch_uuid_representation=raw)
        compiler=shutil.which('gcc');header=Path(sysconfig.get_paths()['include'])/'Python.h'
        if not compiler or not header.is_file():raise ValueError('actual FLA compiler and Python headers required')
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        with torch.inference_mode():
            rng=torch.Generator(device='cuda').manual_seed(20261013)
            qkv=[torch.randn((1,64,2,128),device='cuda',dtype=torch.bfloat16,generator=rng) for _ in range(3)]
            gates=-torch.ones((1,64,2),device='cuda',dtype=torch.float32)
            beta=torch.full((1,64,2),.5,device='cuda',dtype=torch.float32)
            result,_=chunk_gated_delta_rule(*qkv,g=gates,beta=beta,use_qk_l2norm_in_kernel=True)
            torch.cuda.synchronize()
            if result.shape!=qkv[2].shape or not torch.isfinite(result).all():raise ValueError('actual FLA kernel preflight failed')
        runtime['actual_fla_kernel_preflight']=True;del qkv,gates,beta,result
        kwargs={k:(Path(v) if k in request['jobs'][args.job]['path_arguments'] else v)
            for k,v in request['jobs'][args.job]['arguments'].items()}
        result=B.capture(**kwargs) if args.job=='capture' else B.control(**kwargs)
        code=0;runtime.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        pinned_request(args.request,args.request_sha256,args.job)
    except Exception as error:failure={'type':type(error).__name__,'message':str(error)}
    finally:signal.alarm(0)
    proof={'schema':'recovery_video_teacher_execution.v1','source_year':2026,'exit_code':code,'job':args.job,
        'request':B.V.pin(args.request),'runtime':runtime,'failure':failure,'elapsed_seconds':time.perf_counter()-started,
        'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'native_train_ready':False,
        'training_started':False,'code_sha256':B.A.file_sha(Path(__file__))}
    args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_bytes(B.A.canonical(proof)+b'\n')
    print(json.dumps({'exit_code':code,'failure':failure,'native_train_ready':False}));return code


if __name__=='__main__':raise SystemExit(main())
