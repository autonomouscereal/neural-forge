import hashlib,json
from pathlib import Path
base=Path('proof_outputs')
metadata=json.loads((base/'speech_official_model_metadata_oct13_v2.json').read_text(encoding='utf-8'))
api=json.loads((base/'speech_official_api_oct13.json').read_text(encoding='utf-8'))
models=[]
for source in metadata['rows']:
    if 'model_id' not in source:
        continue
    models.append({key:source[key] for key in ('model_id','revision','api_url','api_sha256','files')} |
                  {'configs':{name:{'sha256':item['sha256']} for name,item in source['configs'].items()}})
packages=[]
for source in api['rows']:
    if 'package' in source:
        packages.append({key:source[key] for key in ('package','version','requires_python','requires_dist','files')})
manifest={'schema':'chimera_recovery.speech_teacher_pins.v1','dated':'2026-10-13',
          'public_metadata_source_sha256':'e96417410cf957f5c7fb3236061e61b7e61b22771a60a95c58a881b03a7766f3',
          'api_source_sha256':'91e9e87a1b9af86c14a0da101cd835d472ec378fe4a7be2c41b898f1600b59d5',
          'verified_wheel_api_sha256':'ff7f81d144ad5dc777168d67829fddb6c6d1a004f9afd7b178163669eb7eda06',
          'models':models,'packages':packages,'native_training_admission_granted':False,
          'runtime_policy':'separate private environments; omit GUI dependencies; no global mutation; verify actual resolved imports',
          'readiness':'pins only; parent owns genuine download/runtime/inference receipts'}
target=Path('experiments/chimera_recovery/SPEECH_TEACHER_PINS_2026-10-13.json')
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'models':len(models),'packages':len(packages)}))
