"""Prepare immutable globally resealed225 ingress; reuse only identical bytes."""
from pathlib import Path
import importlib.util,json,shutil,sys,tarfile
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/cereal/Documents/ai training')
ROOT=BASE/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
JOINT=HERE/'chimera_joint_audio_admission_oct13_v1'
OUT=HERE/'joint_audio_native_prepare_v1'
DONOR=HERE/'speech_native_prepare_v1/inputs'
REMOTE_DONOR='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/inputs'
if OUT.exists():raise ValueError('preserve prior joint capsule')
spec=importlib.util.spec_from_file_location('joint_portable_paths',ROOT/'experiments/chimera_recovery/audio_paths.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
def sha(path):return A.R.sha(path)
manifest=JOINT/'global_admission/manifest.json'
expected='6d4d2eac4644179c615f60b8288b11b582096f6080aac225a29fad42946eed43'
if sha(manifest)!=expected:raise ValueError('fulljoint225admissionchanged')
OUT.mkdir();inputs=OUT/'inputs';inputs.mkdir()
A.rebind(manifest,expected,inputs/'admission','/joint/inputs/admission')
for name in ('source_inventory.jsonl','source_ready.jsonl','preparation.json'):
    shutil.copyfile(JOINT/name,inputs/name)
for original,name,expected_sha in (
    (BASE/'chimera-omni/src/chimera_omni/modeling/charted.py','charted_attention.py',None),
    (BASE/'reviews/2026-10-13-folding/recovery-preparation/speech-preregistration-v1/PREREGISTRATION_SPEECH_RECOVERY_2026-10-13.md',
     'selected-speech-preregistration.md','cf676e8d0b1c63052972921059fed97628122fb2257c6c4236ee421a923d9a91')):
    if expected_sha and sha(original)!=expected_sha:raise ValueError('selected preregistration changed')
    shutil.copyfile(original,inputs/name)
donor_by_sha={sha(path):str(path.relative_to(DONOR)).replace('\\','/') for path in DONOR.rglob('*') if path.is_file()}
files=[];reuse=[];transferred=[]
for path in sorted(inputs.rglob('*')):
    if not path.is_file():continue
    item={'path':str(path.relative_to(inputs)).replace('\\','/'),'sha256':sha(path),'bytes':path.stat().st_size}
    files.append(item)
    if item['sha256'] in donor_by_sha:
        reuse.append({**item,'donor_path':REMOTE_DONOR+'/'+donor_by_sha[item['sha256']]})
    else:transferred.append(item)
receipt={'schema':'chimera_recovery.joint_audio_execution_inputs.v1','files':files,'verified_byte_reuse':reuse,
    'source_admission_sha256':expected,'source_preparation_sha256':sha(JOINT/'preparation.json'),
    'retained_source_rows':225,'full_original_inventory':352,'train_admitted':194,'held_excluded':31,
    'original_per_wave_proofs_preserved':True,'selected_preregistration_sha256':'cf676e8d0b1c63052972921059fed97628122fb2257c6c4236ee421a923d9a91',
    'amendment_sha256':'312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055','training_started':False}
(OUT/'inputs_manifest.json').write_bytes(A.R.canonical(receipt)+b'\n')
archive=OUT/'new_inputs.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as handle:
    for item in transferred:handle.add(inputs/item['path'],arcname=item['path'],recursive=False)
(OUT/'transfer.json').write_bytes(A.R.canonical({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,
    'inputs_manifest_sha256':sha(OUT/'inputs_manifest.json'),'input_files':len(files),'byte_verified_donor_files':len(reuse),
    'new_files':len(transferred),'new_bytes':sum(item['bytes'] for item in transferred),'source_bytes':sum(item['bytes'] for item in files)})+b'\n')
print(json.dumps({'new_archive_bytes':archive.stat().st_size,'input_files':len(files),'verified_donor_files':len(reuse),
                  'train_admitted':194,'held_excluded':31,'training_started':False}))
