from pathlib import Path
from types import SimpleNamespace
import json,sys,time
sys.path.insert(0,'/source/experiments/chimera_recovery')
import h3_audio_probe as P
import torch
from safetensors.torch import load_file
base=dict(stage_c=Path('/previous/audio-cache-inputs-v1/stage_c_media_a.py'),comfy_root=Path('/opt/ComfyUI'),
audio_vae=Path('/opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors'),audio_source_sha256=P.AUDIO_SOURCE_SHA)
started=time.monotonic()
session=P.EncoderSession(SimpleNamespace(**base))
entries=[]
for i,case in enumerate([{'audio': '/previous/audio-cache-inputs-v1/assets/518a56a9484d3bceca4c6f3c/sound.wav', 'audio_sha256': 'a389090f2ca97297735a77b8e1b67fbc7094db1bbc074784a188e1983d40ad18', 'reference': '/previous/clean-audio-cache-v1/000000.json'}, {'audio': '/previous/audio-cache-inputs-v1/assets/22225ebd4b0131bf331642b2/sound.wav', 'audio_sha256': 'f5dbf03eee75b5a9baa03bf838334762fc54b17d321a062261b6b7d76a09883f', 'reference': '/previous/clean-audio-cache-v1/000001.json'}]):
    args=SimpleNamespace(**base,audio=Path(case['audio']),audio_sha256=case['audio_sha256'],out=Path('/output')/f'{i:06d}.json')
    result=P.run(args,session=session)
    assert result['probe_passed'], result.get('failure')
    reference=json.loads(Path(case['reference']).read_text())
    oldpath=Path('/previous/clean-audio-cache-v1')/Path(reference['tensors']['path']).name
    assert P.sha(oldpath)==reference['tensors']['sha256']
    old=load_file(str(oldpath),device='cpu')
    new=load_file(str(args.out.with_suffix('.safetensors')),device='cpu')
    assert set(old)==set(new) and all(torch.equal(old[k],new[k]) for k in old), 'actual separate-runtime tensors differ'
    entries.append({'asset_sha256':case['audio_sha256'],'proof':str(args.out),'proof_sha256':P.sha(args.out),
    'separate_runtime_reference_sha256':P.sha(Path(case['reference'])),'all_tensors_bitwise_equal':True})
sealed=session.seal()
assert sealed['actual_waveforms']==len(entries)==2
result={'schema':'chimera_recovery.h3_reusable_cpu_actual_proof.v1','passed':True,'session':sealed,'entries':entries,
'wall_seconds':time.monotonic()-started,'adapter_sha256':P.sha(Path(P.__file__)),'training_started':False,'gpu_exposed':False}
Path('/output/result.json').write_bytes(P.canonical(result)+b'\n')
print(json.dumps({'passed':True,'waveforms':2,'wall_seconds':result['wall_seconds'],'result_sha256':P.sha(Path('/output/result.json'))}),flush=True)
