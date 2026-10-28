from pathlib import Path
import json
import zipfile
from experiments.chimera_recovery import train_video as D

OWN=Path(__file__).resolve().parents[1]
MAIN=Path('C:/Users/cereal/Documents/ai training')
ROOT=MAIN/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
REV=MAIN/'reviews/2026-10-13-folding/recovery-preparation'
selection=REV/'temporal-video-selection-v2/selection.json'
amendment=REV/'temporal-video-selection-v2/TEMPORAL_VIDEO_AMENDMENT_2026-10-13.v2.json'
native=OWN/'proof_outputs/chimera_video_native_oct13_v2/manifest.json'
output=OWN/'proof_outputs/chimera_video_cpu_capsule_oct13_v2'
if output.exists():raise ValueError('fresh capsule required')
output.mkdir()
files={}
def add(p):
    p=Path(p).resolve()
    if p.is_file():files[p.as_posix()]=p
for folder in [OWN/'proof_outputs/chimera_video_native_oct13_v2',OWN/'proof_outputs/chimera_video_capture_review_oct13_v5',
               OWN/'proof_outputs/video-native-staged-inputs-v1/capture-inputs',
               OWN/'proof_outputs/video-native-staged-inputs-v1/features',OWN/'proof_outputs/video-native-staged-inputs-v1/ingress',
               REV/'video-k64-runtime-v1/capture-actual',REV/'video-k64-runtime-v1/control-actual']:
    for p in folder.rglob('*'):add(p)
for p in (ROOT/'experiments/chimera_recovery').glob('*.py'):add(p)
for p in (OWN/'experiments/chimera_recovery').glob('*.py'):add(p)
for p in [selection,amendment,OWN/'experiments/chimera_recovery/TEMPORAL_VIDEO_AMENDMENT_2026-10-13.v1.json']:
    add(p)
    if not p.is_file():raise ValueError('actual selected source missing')
entries=[]
for normalized,p in sorted(files.items()):
    sha=D.A.file_sha(p)
    for original in sorted(set((str(p),normalized))):
        entries.append({'original_path':original,'mirror_relative_path':normalized,'sha256':sha,'bytes':p.stat().st_size})
inputs={'manifest_path':str(native),'manifest_sha256':D.A.file_sha(native),'amendment_path':str(amendment),
        'amendment_sha256':D.A.file_sha(amendment),'selection_path':str(selection),'selection_sha256':D.A.file_sha(selection)}
# Actual positive local policy check uses the genuine owner selection before
# packaging. CPU native execution is independently recorded by video_cpu_job.
adapter=D.VideoAdapter(**inputs);proof=adapter.preflight()
manifest={'schema':'recovery_video_cpu_mirror.v1','sealed':True,'source_year':2026,'files':entries,'inputs':inputs,
          'driver_dir':str(ROOT/'experiments/chimera_recovery'),'original_receipts_modified':False,
          'actual_local_complete_preflight':proof,'runtime_policy':{'cuda_visible_devices':'','cpu_threads':8,
          'maximum_ram_gib':8,'timeout_seconds':180},'native_baseline':'/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py',
          'native_tree':'/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9',
          'current64_text_block_runtime_proven':False,'gpu_job_requested':False}
manifest['manifest_sha256']=D.A.digest(manifest)
manifest_path=output/'manifest.json';manifest_path.write_bytes(D.A.canonical(manifest)+b'\n')
archive=output/'video_cpu_capsule.v2.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(manifest_path,'manifest.json')
    for relative,p in sorted(files.items()):z.write(p,'mirror/'+relative)
    owned={'video_native.py','video_quantized_answer.py','train_video.py','video_cpu_job.py'}
    for p in (ROOT/'experiments/chimera_recovery').glob('*.py'):
        if p.name not in owned:z.write(p,'code/experiments/chimera_recovery/'+p.name)
    for name in ('video_native.py','video_quantized_answer.py','train_video.py','video_cpu_job.py'):
        z.write(OWN/'experiments/chimera_recovery'/name,'code/experiments/chimera_recovery/'+name)
    z.writestr('code/tests/__init__.py','')
    z.write(OWN/'tests/test_chimera_recovery_train_video.py','code/tests/test_chimera_recovery_train_video.py')
receipt={'archive':{'path':str(archive),'sha256':D.A.file_sha(archive),'bytes':archive.stat().st_size},
         'manifest':{'path':str(manifest_path),'sha256':D.A.file_sha(manifest_path)},'mirror_files':len(files),
         'alias_entries':len(entries),'source_original_selection':inputs,'all_mirrored_bytes_verified':True}
(output/'delivery.json').write_bytes(D.A.canonical(receipt)+b'\n')
print(json.dumps(receipt))
