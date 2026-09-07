# H3 native patch-boundary readiness — 2026-09-07

This is a read-only capture contract, **not a producer, accepted chart, installed
child, or media-capability result**. No GPU was allocated or model forward run.
Tests use explicitly synthetic CPU tensors and do not replace source evidence.

## Fresh bounded source evidence

Read `/opt/ComfyUI/comfy/ldm/minimax/model.py` inside immutable existing runtime
`f9f6ed9b9e8cc2f28585fb4fecb08a1dbd9bc658383e7395943bb5c4aebde15d` through
server-manager. Its SHA256 is
`882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024`, matching
the real curve6 converter pin. This was a source read, not HY3D/H3 inference.

The source parent is
`/home/cereal/comfyui/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`.
A bounded 55,976-byte header plus four small projection payloads were read;
**no fresh whole-40.2GB SHA was computed**. Header SHA256:
`d55386f89bed7f9719ae315b4d70d104b9000e7a839d7702c82a97c1835324e6`.
The validator pins that header and all four freshly read raw-payload hashes.

| Native boundary | Exact contract |
|---|---|
| Video projection | FP32 weight `[5376,96]`, FP32 bias5376; video `[1,24,T,H,W]` patch `(1,2,2)`, row order T/H/W, feature order C/pt/ph/pw |
| Audio projection | FP32 weight `[5376,32]`, FP32 bias5376; audio `[1,32,2,T]`, channel-major stereo rows |
| Text | Actual Qwen3-VL layer50 hidden states `[1,L,5120]`, native condition projection plus token refiner, not Gaussian or a silently pre-refined5376 stand-in |
| Packed input | Full `[text|audio|video]` for this initial plain-T2VA contract; native block0 input cloned before in-place residual writes |
| Coordinates | Actual native `PackedLayout.position_ids [S,3]` float64, including text; area-normalized H/W, stereo W extremes, native nonuniform time clock |
| Time | Actual FP32 timestep/1000; video shift12, audio shift3, full original clean/noise/x rows; no sigma or frame relabeling |

The native positional clock `(1,4,4,4,4)*5/3` is **not** a substitute for VAE
ownership measurement. The latter is an argmax influence observation, not proof
of exclusive causal support. Every latent frame remains present even if its
argmax ownership group is empty.

## Existing evidence and remaining gap

Read the actual retained `h3_blocks_v2/capture_receipt.json` and
`h3-native-curve6-sep07/charted_block.json` on the server. The historical source
receipt records source file SHA `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4`
and FP32 compute/BF16 body storage. The converter preserves explicit deviations:
procedural C15 clips, Gaussian text states, and latent frames1–4 relabeled0–3.
It explicitly requires future host/source charts and explicit source coordinates.
Those artifacts remain preserved partial evidence, not full native-patch proof.

Original `media/c16_h3_capture.py` already runs the actual DiT and captures block
input/output, MLP input/output, attention output, time embedding/modulation and
segment rows. It does **not** save exact patch projection I/O or actual THW rows.
Its `--text-states` option is not a text encoder/provenance producer. Its existing
C15 latent reconstruction does not restore missing latent frames0/5. Group4/
latent-group1 raw3072→24 rows must not be relabeled as native96 patches.

## Smallest next real producer

1. Bind actual S1 original video/audio files and their original parent groups;
   split by both AV parents (or bipartite connected components) before capture.
   Every sigma/window of either original parent stays on the same side.
2. Run authentic video/audio VAE encodes with native normalization, preserving
   complete latents. Measure the full per-raw-frame influence matrix on those
   clips. Do not select only a group size or drop latent0/5. Save raw frame count,
   input hashes, full latent hashes, actual encoder source/weight-loading receipt.
3. Produce real Qwen3-VL layer50 text states for the exact prompt, with analogous
   source/weight and output provenance. This producer/evidence is still missing;
   a Gaussian fallback is not supported.
4. Extend the real H3 capture route with FP32 projection pre/post hooks, original
   full clean/noise/noisy input arrays, actual sigma values, refined text and
   block0 pre-hook clone. Observe actual `rope_freqs(position_ids, ...)` input;
   do not reconstruct it from child coordinates. Preserve the original instance
   attribute state when removing a temporary method override (avoid bound-method
   reference cycles). Retain existing block/time/write taps for independent replay.
5. Run the contract below, then independently replay parent block/time/write
   evidence and fit/gate paired host charts. No behavioral acceptance threshold
   or post-S3 host acceptance is invented here.

GPU/native-model mounts remain essential for new authentic forward captures.
CPU can validate pinned geometry, provenance, all-frame/stereo coverage and the
small source projections, but cannot manufacture the missing encoder/DiT evidence.

## Executable validator contract

`c16_h3_patch_contract.py --manifest /absolute/capture.json --manifest-sha256 SHA`
prints a report only, performs no writes and has **no production fixture switch**.
It loads reviewed geometry nodes verbatim from the hash-pinned native source via
AST; no Comfy shim or model constructor is executed. It checks the four source
FP32 projections in512-row CPU chunks. The fixed1e-5 relative-L2 projection check
is an arithmetic replay tolerance, not an intelligence/quality gate.

Top-level schema is `neural-forge.h3-native-patch-capture.v1`, `fixture:false`,
with `native_source`, `producer` (both `{path,sha256}`), absolute
`parent_checkpoint`, exact `projection_raw_bytes_sha256` from the code constants,
`max_capture_bytes` (explicit per-forward limit ≤8GiB), and `forwards`.

Each forward declares unique `id`, original `split:fit|held`, `route:plain_t2va`,
`minimax_payload:{}`, `transformer_options:{}`, `stimuli:s1_media`, `deviations:[]`,
`compute_dtype:float32`, `allow_tf32:false`, native `video_shape`, `audio_shape`,
`raw_frames`, `text_len`, both `video_asset`/`audio_asset` file refs and
`video_parent_group`/`audio_parent_group`, `tensors` file ref, `temporal_probe`
JSON ref and three `video|audio|text_encoder_receipt` JSON refs. Initial scope
rejects latent padding/cropping, keyframe/reference/tag payloads, schedule
overrides and alternate compute dtypes rather than silently simplifying them.

Safetensors preserve full `video|audio_clean`, `_noise`, `_x`, `_patch_input`,
`_patch_output`; `sigma_video`, `sigma_audio`, `timestep`; `text_states`,
`text_refined`, `block0_input` as FP32; `position_ids` as F64 and
`video_latent_frame_ids` as I64. Temporal probe includes complete `frames`,
`latent_frames`, `influence`, `owner_per_frame`, `groups`, actual video
`input_sha256` and `encoder_receipt_sha256`.

Encoder receipts use `kind:h3_native_<video|audio|text>_encode`,
`actual_parent_forward:true`, `source_code` and `source_identity_receipt` refs,
`output_raw_bytes_sha256`, `output_shape`, `output_dtype:float32`. VAEs also bind
`input_sha256`, `normalization:native_h3_model_input`, and video `raw_frames`.
Text binds `encoder_family:qwen3_vl`, `hidden_layer:50` and `prompt` file ref.
Identity receipts use `schema:neural-forge.h3-encoder-identity.v1`, `family`
(`minimax_h3_video_vae|minimax_h3_audio_vae|qwen3_vl`), `source_code_sha256`,
`actual_tensor_loader_verified:true`, absolute `checkpoint_path`, positive
`checkpoint_bytes`, `checkpoint_header_sha256`. These are requirements for the
new producer, not claims that such native input artifacts already exist.

The validator verifies receipt hashes and joins, not the truth of a producer's
entire encoder execution independently. Success explicitly reports
`encoders_independently_replayed:false`, `parent_dit_forward_replayed:false`,
`folded_child_proven:false`, `production_install_eligible:false`. Original
asset hashes, parent-group IDs, and exact clean-latent content hashes all enforce
split consistency; transformed near-duplicates are not claimed detected.
