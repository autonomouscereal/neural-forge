"""Immutable actual-source portable capsule; this performs no learning."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys,tarfile
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/cereal/Documents/ai training')
SOURCE=BASE/'proof_outputs/chimera_speech_combined_admission_oct13_v1'
OUT=HERE/'speech_native_prepare_v1'
if OUT.exists():raise ValueError('preserve prior speech capsule')
spec=importlib.util.spec_from_file_location('speech_portable_audio_paths',HERE.parent/'experiments/chimera_recovery/audio_paths.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
def sha(p):return A.R.sha(p)
admission=SOURCE/'global_admission/manifest.json'
expected='550ce5093621ff2b477297e6844b300d8c193af6dc6bac61c30a463f3e7ad590'
if sha(admission)!=expected:raise ValueError('actual combined admission changed')
OUT.mkdir()
inputs=OUT/'inputs';inputs.mkdir()
A.rebind(admission,expected,inputs/'admission','/work/inputs/admission')
receipt=json.loads((inputs/'admission/transfer_receipt.json').read_text())
for name in ('source_inventory.jsonl','source_ready.jsonl','preparation.json','quarantine.jsonl'):
    shutil.copyfile(SOURCE/name,inputs/name)
rows=[json.loads(line) for line in (inputs/'source_inventory.jsonl').read_text().splitlines()]
if len(rows)!=208:raise ValueError('original full208-source inventory required')
assets=inputs/'complete_teacher_assets';assets.mkdir()
inventory=[]
for row in rows:
    p=row['provenance']
    for role,path,digest in [('original',p['original_audio_path'],p['original_audio_sha256']),
                            ('derived',row['media'][0]['path'],row['media'][0]['sha256'])]:
        path=Path(path)
        if sha(path)!=digest:raise ValueError('full original/derived teacher asset changed: '+row['id'])
        target=assets/(digest+'.wav')
        if not target.exists():shutil.copyfile(path,target)
        inventory.append({'id':row['id'],'role':role,'original_path':str(path),'path':'/work/inputs/complete_teacher_assets/'+target.name,
                          'sha256':digest,'independent_asr_exact_match':p['independent_asr_exact_match']})
(inputs/'complete_teacher_assets.json').write_bytes(A.R.canonical({'schema':'chimera_recovery.full_speech_source_assets.v1',
    'original_inventory_rows':208,'entries':inventory,'optimizer_admission_changed':False})+b'\n')
stage=BASE/'neural-forge/reports/omni-campaign-2026-09/distill/tools/stage_c_media_a.py'
if sha(stage)!='fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf':raise ValueError('actual StageC source changed')
shutil.copyfile(stage,inputs/'stage_c_media_a.py')
charted=BASE/'chimera-omni/src/chimera_omni/modeling/charted.py'
shutil.copyfile(charted,inputs/'charted_attention.py')
amendment=BASE/'reviews/2026-10-13-folding/recovery-preparation/speech-preregistration-v1/PREREGISTRATION_SPEECH_RECOVERY_2026-10-13.md'
if sha(amendment)!='cf676e8d0b1c63052972921059fed97628122fb2257c6c4236ee421a923d9a91':raise ValueError('actual selected speech amendment changed')
shutil.copyfile(amendment,inputs/'selected-speech-preregistration.md')
files=[{'path':str(path.relative_to(inputs)).replace('\\','/'),'sha256':sha(path),'bytes':path.stat().st_size}
       for path in sorted(inputs.rglob('*')) if path.is_file()]
manifest={'schema':'chimera_recovery.final_speech_execution_inputs.v1','files':files,'source_manifest_sha256':expected,
    'source_preparation_sha256':sha(SOURCE/'preparation.json'),'original_tts_sources':208,'retained_native_sources':201,
    'train_admitted':186,'held_diagnostic':15,'quarantine':7,'selected_preregistration_sha256':sha(amendment),
    'amendment_sha256':'312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055','training_started':False}
(OUT/'inputs_manifest.json').write_bytes(A.R.canonical(manifest)+b'\n')
archive=OUT/'inputs.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as handle:
    for item in files:handle.add(inputs/item['path'],arcname=item['path'],recursive=False)
(OUT/'transfer.json').write_bytes(A.R.canonical({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,
    'inputs_manifest_sha256':sha(OUT/'inputs_manifest.json'),'uncompressed_bytes':sum(item['bytes'] for item in files)})+b'\n')
print(json.dumps({'inputs':len(files),'bytes':archive.stat().st_size,'native_rows':201,'original_tts_inventory':208,'training_started':False}))
