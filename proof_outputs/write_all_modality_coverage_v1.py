import hashlib,json
from pathlib import Path
BASE=Path('C:/Users/cereal/Documents/ai training')
OWN=Path(__file__).resolve().parents[1]
ROOT=BASE/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
out=OWN/'proof_outputs/all_modality_coverage_audit_v1'
out.mkdir(exist_ok=True)
def pin(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
source=BASE/'proof_outputs/chimera_media_seed_oct13_v2/source_ready.jsonl'
rows=[json.loads(line) for line in source.read_text().splitlines()]
inspection=BASE/'reviews/2026-10-13-folding/recovery-preparation/vl-processor-inspection-v1/vl-processor-inspection-v1.json'
proof=json.loads(inspection.read_text())
plan=OWN/'proof_outputs/chimera_vl_jobs_oct13_v1/manifest.json'
facts={
 'schema':'chimera_recovery.all_modality_coverage_audit.v1',
 'scope':'current executable preparation source and actual2026CPU ingress; not trained student capability',
 'sources':{key:pin(path) for key,path in {
  'operator_inventory144':source,'operator_report':source.parent/'report.json',
  'vl_plan48':plan,'actual_vl_cpu48':inspection,
  'native_qimg_adapter':ROOT/'experiments/chimera_recovery/media_cache.py',
  'native_vision_adapter':ROOT/'experiments/chimera_recovery/vision_cache.py',
  'vl_teacher_adapter':ROOT/'experiments/chimera_recovery/vl_teacher.py',
  'actual_media_b_teacher':BASE/'reviews/2026-10-13-folding/live/ace_conditioning_sources/media_b_teacher_cache.py',
  'actual_comfy_model_base':BASE/'reviews/2026-10-13-folding/live/ace_conditioning_sources/model_base.py',
  'explicit_speaking_duration':OWN/'experiments/chimera_recovery/train_speech_flow.py'}.items()},
 'image_edit':{
  'actual_source_counts':{split:sum(row['route']=='image_edit' and row['split']==split for row in rows) for split in ('train','validation','test')},
  'existing_qimg_route':'image_generate','existing_qimg_reference_image_in_inputs':False,
  'existing_cache_has_before_image_conditioning':False,
  'existing_route_table_does_not_supply_native_adapter':True,
  'actual_parent_source_supports_reference_latents':True,
  'parent_source_claim_scope':'ComfyQwenImage.extra_conds passes ref_latents and ref_latents_method; currentStageCwriter/nativeadapterdo not capture this path',
  'remaining':[
   'global connected admission for unchanged before/after/instruction rows, including originalregistry/eval/priorheld/crossroute exclusions',
   'actualQIMG-VAE before/after encode and process_latent_in/out proofs',
   'pinned before-image conditioning source contract and exact independently noised after target; after metadata never input',
   'distinct supervised-edit objective/driver amendment OR actual reference-conditioned parentKD capture with complete conditioning and native packed layout',
   'mixed clean conditioning/noisy target row validator and target-only loss ownership; existingpositive-sigma-all-valid-row flow contract is insufficient',
   'current-child full-depth gradient parity and held target-free edited-image decode/preservation metrics'],
  'next_executable_inputs':{'source':str(source),'input_media_role':'input before.png','target_media_role':'target after.png',
   'raw_before_c11_api':'vision_cache.patchify after exactpinnedRGBresize; usegeometry_conditioning nativepatchparity recipe, nofullQwenweightload',
   'actual_vae_loader':'stage_c_media_a.load_vae withpinned qwen_image_vae.safetensors; verifyactualmodelspace transform',
   'parent_reference_capture_contract':'actualComfyQwenImage.extra_conds(reference_latents=[before_model_latent],reference_latents_method=explicitpinnedvalue)'},
  'native_training_prepared':False},
 'video_understanding':{
  'actual_source_counts':{split:sum(row['route']=='video_understand' and row['split']==split for row in rows) for split in ('train','validation','test')},
  'actual_teacher_ingress':{'passed':proof['passed'],'rows':len(proof['rows']),'model_weights_loaded':proof['model_weights_loaded'],
   'native_video_frames_requested':8,'source_frames':8,'full_coverage_policy':'all8sourceframes; originalprocessor temporal2 packing keeps4temporalpatch groups',
   'scope':'nativeprefix/mediaexpansion/EOSownership; neitherteacheranswersnorstudentunderstandingevaluated'},
  'current_d3d_native_converter':'PILstillimage,duplicatedframe,gridT1,cu_seqlens[0,N],source_image; no video batchconversion',
  'current_qvl_feature_writer':'qvl-run requiresqvl_images/PILstillimage and constructsCUDAvisionmodel; no nativevideo corpus route',
  'remaining':[
   'serialallocatedvl_teacher.py build againstall48supported fixtures toseal genuine248320-vocabK64 assistant targets withfull nativeprefixmask',
   'fresh authenticViT13 source features for actual24newimagesand24videos; old7492imagecache does notautomatically covertheseassetSHAs',
   'derive nativevideo C11rows from originalprocessor pixel_values_videos/grid_thw plusactualfullsampling/timestamps, notduplicatedstillimages',
   'captureandbind temporal cu_seqlens/position embeddings andnativebridge coordinates; prove currentQVLunit behavior oramendment ratherthanassumeparenttemporalparity',
   'join actualteacheranswer source format to vision_answer_source.v1/nativebatch recipe and completeglobal group admission',
   'held freeanswers graded against authoredflashcolor/count/onset/position labels; finitemediatensors alone are ingress evidence'],
  'next_executable_capture':{
   'adapter':'vl_teacher.py build','plan':str(plan),'plan_sha256':pin(plan)['sha256'],
   'inspection':str(inspection),'inspection_sha256':pin(inspection)['sha256'],
   'writer_sha256':proof['writer_sha256'],'model_dir':'/home/cereal/models/Qwen3.8-27B-Uncensored',
   'split_jobs':['train16','validation16','test16'],'top_k':64,'micro_batch':2,'head_chunk_rows':256,
   'allocation_policy':'queueuntilsingle27BQwenstreamerRAMbudgetavailable; nosecondconcurrentstreamer',
   'native_feature_capture_extension':'source-pinnedoriginalprocessorvideo tensors +separateactualQVL32visionmodel13pre/posthooks; noaudio route'},
  'native_training_prepared':False},
 'speech_output':{
  'objective':'supervised_speech_rectified_flow','foreign_codec_ids_used':False,
  'native_shape':'channel-major2Trowsby32 H3modelspacelatents',
  'sampler':'callerrequested_frames=T; seedGaussian;32Eulerstepssigma1to0; target-free',
  'duration_seconds':'nominalT/40; decodedlengthmeasuredseparately',
  'samples_per_stereo_channel':'800T at32000Hz',
  'duration_predicted_by_model':False,'audio_eos_predicted_by_model':False,
  'source_admission_scope':'186train/15held speech; globallyunionreproof194audio/speechtrain expecteduntilnativecacheaudit',
  'remaining':['actual201perwaveencoderseal','native744source/sigmapairseal','actualcurrent-childtrainingparity/precedingstates',
   'held target-freesampling→actualH3decode→independentASR/listening/voice/durationreview'],
  'unconstrained_length_speaking_prepared':False},
 'training_started':False,'gpu_launch_performed_by_audit':False,'all_modalities_production_ready':False}
(out/'report.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps({'report':str(out/'report.json'),'sha256':pin(out/'report.json')['sha256'],'native_training_prepared':False}))
