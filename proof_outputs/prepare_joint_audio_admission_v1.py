from pathlib import Path
import hashlib,subprocess,sys
BASE=Path('C:/Users/cereal/Documents/ai training')
OWN=Path(__file__).resolve().parents[1]
ROOT=BASE/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
helper=ROOT/'experiments/chimera_recovery/audio_admission.py'
argv=[sys.executable,str(OWN/'experiments/chimera_recovery/audio_sources.py'),'prepare-source',
      '--procedural-admission',str(BASE/'proof_outputs/chimera_audio_source_admission_oct13_v1/manifest.json'),
      '--procedural-sha256','fa9486df16f53329d6f83ed02b90ab6e6901ceae4672f37f4c8b92ef81b0fe6b',
      '--speech-admission',str(BASE/'proof_outputs/chimera_speech_combined_admission_oct13_v1/global_admission/manifest.json'),
      '--speech-sha256','550ce5093621ff2b477297e6844b300d8c193af6dc6bac61c30a463f3e7ad590',
      '--admission-helper',str(helper),'--helper-sha256',hashlib.sha256(helper.read_bytes()).hexdigest(),
      '--procedural-asset-root',str(BASE/'proof_outputs/chimera_media_seed_oct13_v2/assets'),
      '--out',str(OWN/'proof_outputs/chimera_joint_audio_admission_oct13_v1')]
print('Starting actual full joint admission traversal',flush=True)
raise SystemExit(subprocess.run(argv).returncode)
