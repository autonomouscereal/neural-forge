import hashlib,json
from pathlib import Path
BASE=Path('C:/Users/cereal/Documents/ai training')
OWN=Path(__file__).resolve().parents[1]
parent=BASE/'reviews/2026-10-13-folding/recovery-preparation/speech-preregistration-v1/PREREGISTRATION_SPEECH_RECOVERY_2026-10-13.md'
raw=parent.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='cf676e8d0b1c63052972921059fed97628122fb2257c6c4236ee421a923d9a91'
out=OWN/'proof_outputs/h3_amendment_candidate_v3';out.mkdir()
append='''

### 2026-10-13 selected H3 same-mux guided_velocity preparation successor

This dated selection adds the authentic joint audio/video recovery lane to the
previous selected recovery amendments. The frozen D1-D4 body, earlier dated
audio/speech bodies, original independent preparation receipts and parent model
bytes remain unchanged. This document binds this new H3 route; earlier stage
configurations retain their independently selected preregistration files. No
training launch, accepted stage state, model quality or convergence is claimed.

The source scope is all24original joint_av procedural fixtures,8train and16held,
screened with the unchanged225audio/speech candidates and full352original
inventory against91984normalized rows,1948frozen evaluation rows,17203prior
admission rows and21081cross-route exclusions. The immutable new249candidate
admission is de642d15eb976cce615264453e20d49b2eb71369cb1870b8eb9ad57638527202.
Its all8train AV sources are eligible; earlier194native CE source admissions and
connected groups are unchanged. Held validation/test sources never enter the
optimizer and have no teacher targets in this initial capture.

The complete actual teacher scope is all32targets,4sigmas for each of all8train
sources, with no source, sigma, seed or quality ranking. Initial preparation
selects one successful update per admitted source/target pair, exactly one pass
of the complete eligible capture. The frozen historical300-pair H3 quota is
preserved as history and does not authorize repeats to fill a quota. Any D4
replay-budget successor must explicitly declare its complete eligible scope.

The teacher is the genuine BF16 MiniMax H3 checkpoint, fullweight SHA
a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4,
unchanged before/after capture. The sealed velocity manifest is
7770cf61b2b88a9dfce802009ca7e3d0f26be05ea9904f2ca47b9f002c90df17;
actual native geometry/packer proof is
2150acf1c6e30ac9b34076df43361b70b86c7dcb6cf52d02dc38db504fa62e77.
CFG1means one actual positive branch. For this capture guided_velocity means
matching that actual conditional parent's canonical velocity, without invented
unconditional evaluations, classifier-free amplification or foreign codecs.

Both streams decode the same physical mux and use start0, end22/24seconds,
22frames at24fps96x96, latest source-frame hold from all8source frames at8fps.
Measured audio/video origins agree; no waveform padding, silent fallback or PTS
gaps certify synchronization. The actual video model-space latent is24x7x6x6;
audio is32x2x37. The29333resampled audio samples have no waveform padding.
The genuine VAE emits36audio latent frames, and the original writer adds one
explicit zero-latent frame to satisfy the37-frame parent PackedLayout. The sealed
actual encode manifest is72aa63260aeadaa244255734d7b51ddefcb27400ced933b62670580eda3c885e.
Every source records audio_latent_frames_encoded=36 and audio_latent_padded=1;
the native converter checks the zero tail and the exact raw-encode to captured
model-space clean-source join. All37teacher-layout loss slots are retained,
including the explicit padding slot. This does not claim37actual encodes,
additional observed source time or exact receptive fields. The physical window
remains22/24seconds and the decoded waveform remains29333samples.

The video sigma uses shift12. Audio sigma uses shift3 at the same underlying
time; its exact derivative slope12to3 is source-pinned per target. Video target
is captured canonical v_video. Audio target is captured v_audio_native equal to
the flat guided audio velocity divided by that exact slope. Native loss compares
audio in its own sigma parameterization, with no additional slope factor. Raw
student parent-head signs are+1for video and audio, so canonical prediction is
minus raw for both. Input features are noisy video384:480 and audio480:512;
emission targets are video128:224 and audio224:256. Clean teacher latents and
authored assistant metadata never enter student conditioning.

The actual full sequence is native parent-template text prefix, channel-major
stereo audio74rows, then video63rows. Authentic MiniMax PackedLayout and packers
are bitwise-proved for all32targets. The prefix uses the genuine parent template
and only the user generation request plus its explicit native-window view.

The new native bridge uses a single shared clock from the measured finite
perturbation dominance groups [[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],
[13,14,15,16],[17],[18,19,20,21]], identical across all8sources. Video time is
each group's mean frame-center time; spatial grid endpoints are retained. Audio
uses actual40Hz channel-major centers and categorical stereo branch identity.
The bridge file SHA is f95cb494ac12fa491667aacc261b976241e3ce7b934b7719a9407fb2ee9f0591.
This measured coordinate selection is an explicit dated bridge convention;
finite perturbation dominance does not prove exclusive receptive fields. The
separate parent rotary PackedLayout parity remains bitwise-established.

The existing D3b h3_dit_L25_curve6 engine, shared-r16 inheritance and cumulative
native-state handoff own this stage. Only audited unit/emission/shared-r16
parameters train, with unit/emission LR2e-5 and r16 LR1e-4; AdamW betas .9/.95,
eps1e-8, weight decay0, clip1, warmup20, min LR ratio.1, seed20261013. One pass
and exact checkpoint resume preserve all source/target identities. Missing
accepted preceding state, exact-base parity, finite gradients, held generation
evaluation or exclusive runtime keeps production launch closed.
'''
for old,new in {
 'all24original':'all 24 original','8train':'8 train','16held':'16 held','unchanged225audio/speech':'unchanged 225 audio/speech',
 'full352original':'full 352 original','against91984normalized':'against 91,984 normalized','1948frozen':'1,948 frozen',
 '17203prior':'17,203 prior','21081cross-route':'21,081 cross-route','new249candidate':'new 249 candidate',
 'all8train':'all 8 train','earlier194native':'earlier 194 native','all32targets,4sigmas':'all 32 targets, 4 sigmas',
 'historical300-pair':'historical 300-pair','CFG1means':'CFG 1 means','start0, end22/24seconds':'start 0, end 22/24 seconds',
 '22frames at24fps96x96':'22 frames at 24 fps, 96x96','all8source frames at8fps':'all 8 source frames at 8 fps',
 'The29333resampled':'The 29,333 resampled','emits36audio':'emits 36 audio','the37-frame':'the 37-frame',
 'is72aa':'is 72aa','All37teacher-layout':'All 37 teacher-layout','claim37actual':'claim 37 actual',
 'remains22/24seconds':'remains 22/24 seconds','remains29333samples':'remains 29,333 samples','shift12':'shift 12','shift3':'shift 3',
 'slope12to3':'slope from shift 12 to shift 3','signs are+1for':'signs are +1 for','video384:480 and audio480:512':'video 384:480 and audio 480:512',
 'video128:224 and audio224:256':'video 128:224 and audio 224:256','audio74rows, then video63rows':'audio 74 rows, then video 63 rows',
 'all32targets':'all 32 targets','all8sources':'all 8 sources','actual40Hz':'actual 40 Hz','LR2e-5':'LR 2e-5','LR1e-4':'LR 1e-4',
 'decay0, clip1, warmup20, min LR ratio.1, seed20261013':'decay 0, clip 1, warmup 20, min LR ratio .1, seed 20261013',
 'eps1e-8':'eps 1e-8'}.items():append=append.replace(old,new)
path=out/'PREREGISTRATION_H3_RECOVERY_2026-10-13.md';path.write_bytes(raw+append.encode())
normalized=path.read_bytes().replace(b'\r\n',b'\n')
start=normalized.index(b'## 1. Scope');end=normalized.index(b'\n## Amendments',start)
body=normalized[start:end+1]
assert len(body)==15339 and hashlib.sha256(body).hexdigest()=='c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370'
receipt={'source':{'path':str(parent),'sha256':hashlib.sha256(raw).hexdigest()},
 'candidate':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
 'earlier_document_bytes_preserved_as_exact_prefix':True,'frozen_body_bytes':len(body),
 'frozen_body_sha256':hashlib.sha256(body).hexdigest(),'amendment_sha256':hashlib.sha256(normalized[end:]).hexdigest(),
 'selected_by_root':False,'training_started':False}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
