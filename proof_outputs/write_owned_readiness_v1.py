"""Current collected modality evidence; no missing state promoted as ready."""
import hashlib,json
from pathlib import Path
OWN=Path(__file__).resolve().parents[1]
OLD=OWN.parent/'chimera-trainer-readiness-oct13-3e25f2da3b32'
BASE=Path('C:/Users/cereal/Documents/ai training')
OUT=OWN/'proof_outputs/owned_modality_readiness_oct13_v2';OUT.mkdir()
def pin(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
evidence={
 'h3_native_actual':pin(OLD/'proof_outputs/native_seals_v3/codex-h3-native-modality-v3/proof.json'),
 'h3_native_batches':pin(OLD/'proof_outputs/native_seals_v3/codex-h3-native-modality-v3/native-h3-v1/manifest.json'),
 'image_edit_actual':pin(OLD/'proof_outputs/image_edit_cpu_v1/actual/proof.json'),
 'image_edit_batches':pin(OLD/'proof_outputs/image_edit_cpu_v1/actual/native-edit-v1/manifest.json'),
 'image_edit_d4_actual':pin(OWN/'proof_outputs/image_edit_d4_cpu_v1/actual/proof.json'),
 'unified297_admission':pin(OLD/'proof_outputs/chimera_all_media_source_admission_oct13_v2/global_admission/manifest.json'),
 'natural_speech_source_repair':pin(OWN/'proof_outputs/chimera_natural_speech_repair_oct13_v3/manifest.json'),
 'existing_audio_actual_decode':pin(OWN/'proof_outputs/natural_audio_readproof_v1/proof.json'),
 'decoded_speech_asr_canary':pin(BASE/'reviews/2026-10-13-folding/recovery-preparation/h3-decoded-speech-asr-v1/result.json')}
speech_status=json.loads((OWN/'proof_outputs/image_edit_d4_cpu_v1/latest_status.json').read_text())['audio_audit']
value={'schema':'chimera_recovery.owned_modality_readiness.v1','dated':'2026-10-13','evidence':evidence,
 'scope':'source preparation, authentic CPU ingress, mechanics and replay validation; not trained model capability',
 'lanes':{
  'audio_ce':{'objective':'authored_audio_assistant_ce','source_train_rows':194,'held_sources':31,
    'scope':'8 authored non-speech observations plus186 independently exact-ASR synthetic speech sources',
    'native_codec':'actual H3 raw/model-space32-channel stereo;40Hz;all original PCM retained with explicit minimal-tail derivations',
    'encoder_entries_verified':225,'native_ce_batches_verified':194,
    'full_current_model_forward_parity':False,'accepted_predecessor_state_available':False},
  'speech_flow':{'objective':'supervised_speech_rectified_flow','native_train_batches':744,
    'sources':186,'sigmas_per_source':4,'verification_state':'final full source reconstruction audit active',
    'requested_duration_is_input':True,'duration_prediction_or_audio_eos_learned':False,
    'foreign_speech_codec_ids_used':False,'guidance_teacher_kd':False,
    'scope':'48 authored English utterances with4 official synthetic voices; not broad natural speech competence',
    'decoded_speech_independent_asr_scope':'one genuine VAE roundtrip canary; not201 Chimera outputs'},
  'joint_av':{'objective':'guided_native_velocity','native_train_batches':32,'sources':8,
    'sigmas_per_source':4,'native_source_authentication':True,
    'source_window_seconds':{'numerator':22,'denominator':24},'video_source_frames':22,'video_source_fps':24,
    'video_resolution':96,'audio_encoded_latent_frames':36,'explicit_zero_latent_layout_frames':1,
    'audio_native_loss_clock':'own unscaled canonical velocity after exact dual-clock slope conversion',
    'parent_PackedLayout_and_packers_bitwise':True,
    'bridge_clock_measurement':'finite perturbation dominance; not exact receptive-field proof',
    'scope':'small synthetic same-mux fixtures, not general speech or AV skill'},
  'image_edit':{'stage':'D3b_image_edit_supervised_flow','objective':'supervised_image_edit_rectified_flow',
    'sources':8,'native_train_batches':32,'held_sources':16,'scalar_units_each':65536,
    'genuine_vae_before_after_pairs':24,'actual_native_source_authentication':True,
    'individual_stage_actual_cpu_tests':7,'canonical_env_d4_actual_cpu_tests':14,
    'one_physical_D4_optimizer':True,'full_parent_joint_RoPE_parity':False,
    'scope':'train recolor fixtures; held translation/removal; not broad image editing competence'},
  'video_understand':{'scope':'peer-proven24 input ingresses and8 genuine ViT13 target captures; answerK64 and selected temporal native objective are separate prerequisites',
    'still_T1_cache_reinterpreted_as_temporal':False,'full_parent_temporal_RoPE_parity':False},
  'natural_speech':{'original_inventory_rows':1763,'distinct_original_asset_hashes':1701,
    'asr_repaired_rows':884,'source_train_admitted':0,'original_data_date_verified':False,
    'collection_dates_promoted_to_original_dates':False,
    'scope':'existing source-preserving blocked ASR repair; LAION source is synthetic and SonicMaster is restoration/music'}},
 'last_audio_audit_status':speech_status,'all_modal_recovery_ready':False,'production_training_launched_by_this_preparation':False,
 'remaining_common':['final744 speech replay source reconstruction seal','accepted preceding stage/native/shared checkpoints',
   'actual current model full-depth objective parity','selected and source-pinned replay plan containing all distinct lanes',
   'graded free held answers/generated artifacts during recovery','broader source-backed modality coverage with valid original dates'],
 'immutability':'earlier producer manifests and failed launch receipts preserved; successor artifacts retain explicit source lineage'}
raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()+b'\n'
(OUT/'manifest.json').write_bytes(raw)
print(json.dumps(pin(OUT/'manifest.json')))
