"""CPU-only actual video receiver plus native authored-fixture execution.

The explicit byte-identical mirror retains original source JSON/seals. On Linux
Windows paths resolve relative to this isolated mirror cwd. No receipt is edited.
This proves receiver/Torch binding and small-model forward-path plumbing, not
current 64-text-block OMNI1 execution or a production accepted-state transition.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import unittest

try:
    from . import train_video as D
except ImportError:
    import train_video as D
A=D.A


def verify_mirror(manifest,root):
    if (manifest.get('schema')!='recovery_video_cpu_mirror.v1' or manifest.get('sealed') is not True or
            A.digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest.get('manifest_sha256')):
        raise ValueError('source-pinned explicit video mirror required')
    root=root.resolve()
    if os.name=='nt':raise ValueError('this Linux source-preserving path mirror is not a Windows rewrite')
    aliases={}
    for entry in manifest['files']:
        original=entry['original_path'];relative=Path(entry['mirror_relative_path'])
        normalized=original.replace('\\','/')
        if not normalized.startswith('C:/Users/cereal/Documents/ai training/') or str(relative).replace('\\','/')!=normalized:
            raise ValueError('only explicit original owned workspace paths may be mirrored')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=entry['bytes'] or A.file_sha(path)!=entry['sha256']:
            raise ValueError('mirror escaped namespace or changed original source bytes')
        aliases[original]={'path':str(path),'sha256':entry['sha256']}
    return aliases


def run(*,manifest_path,manifest_sha256,mirror_root,baseline,native_root,output):
    manifest=D.N.json_pin(manifest_path,manifest_sha256)
    aliases=verify_mirror(manifest,mirror_root)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU job requires explicitly hidden CUDA before importing Torch')
    with D.N.source_aliases(aliases):
        return run_bound(manifest,manifest_path,manifest_sha256,mirror_root,baseline,native_root,output,len(aliases))


def run_bound(manifest,manifest_path,manifest_sha256,mirror_root,baseline,native_root,output,count):
    inputs=manifest['inputs'];adapter=D.VideoAdapter(**inputs)
    source_preflight=adapter.preflight()
    os.environ['CHIMERA_VIDEO_DRIVER_DIR']=str(mirror_root/manifest['driver_dir'].replace('\\','/'))
    os.environ['CHIMERA_D3_BASELINE']=str(baseline.resolve())
    os.environ['CHIMERA_OMNI_SRC']=str(native_root.resolve())
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    driver=D.vision_driver(adapter.amendment)
    positions=0;rows=[]
    for entry in adapter.entries.values():
        batch=load_file(str(D.N.pinned(entry,'actual native Torch row')),device='cpu');bound=adapter.bind(batch,entry['source_id'],driver)
        if (bound['segment']['ids'].device.type!='cpu' or bound['teacher_features'].requires_grad or
                bound['segment']['features'].shape!=(1,144,6144) or bound['teacher_features'].shape!=(1,144,1152)):
            raise ValueError('real temporal Torch receiver differs')
        positions+=bound['answer_positions'];rows.append(entry['source_id'])
    fixture=importlib.import_module('tests.test_chimera_recovery_train_video')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(fixture.TorchNativeVideoTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun!=1 or positions!=792:
        raise ValueError('actual native authored-fixture execution failed or skipped')
    proof={'schema':'recovery_video_cpu_consumer_proof.v1','sealed':True,'source_year':2026,
           'source_mirror':{'path':str(manifest_path),'sha256':manifest_sha256},'mirror_files_verified':count,
           'source_preflight':source_preflight,'actual_torch_rows':8,'assistant_positions':positions,'source_ids':rows,
           'actual_native_authored_fixture_tests':result.testsRun,'skips':0,'tests_passed':True,
           'fixture_scope':'actual native engine; four authored shared layers plus source units;144temporal rows',
           'current64_text_block_checkpoint_executed':False,'actual_current_full_depth_runtime_pending':True,
           'torch_version':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),
           'training_started':False,'production_accepted_state_created':False,'code_sha256':A.file_sha(Path(__file__))}
    if proof['cuda_initialized']:raise ValueError('CPU proof unexpectedly initialized CUDA')
    proof['manifest_sha256']=A.digest(proof)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(A.canonical(proof)+b'\n')
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest-path','mirror-root','baseline','native-root','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    result=run(**vars(parser.parse_args()));print(json.dumps({'rows':result['actual_torch_rows'],'positions':792,'tests':1,'skips':0}))


if __name__=='__main__':main()
