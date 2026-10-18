# R6 — K3 + H3 HARNESS SPECS (Chimera Omni fold campaign, 2026-09-01)

Reader R6. Every fact is tagged [verified] (read in code / config / on the server this session) or
[assumed] (inferred, must be receipted before it is relied on). Every threshold below is a PROPOSED
DECISION LINE for Cyril, never a gate the harness invents. Nothing here edits neural-forge or
chimera-omni source; all new code lives under `/home/cereal/omni-campaign/<lane>/` on the server and
under this lane directory locally.

Sources read: plan `docs/CHIMERA_OMNI_FOLD_PLAN_2026-09-01.md` (all sections), plan inputs
`4_read5_server_inventory.md`, memory notes (matrix campaign, behavior-first grading), scratchpad
`k3_linear.py` (= K3 `modeling_kimi_linear.py`), `k3_modeling.py` (= `modeling_kimi_k3.py`),
`k3_layer_probe.py`; chimera-omni `native_media.py`, `native_segments.py`, `modeling/chimera.py`,
`modeling/layers.py`, `config.py`, `proof/h3_hybrid_av_flow.py`, `media/training.py`; neural-forge
`mx45_build.py` (Harness, prior_anchored, law_abiding_solve[_split], pivoted_columns[_gram],
install_narrow_mlp, narrow_mlp_site), `experiments/run_chimera_omni_toy_fold.py`,
`hf_causal_stable_latent_moe_fold.py`, `hf_closed_form_fit.py`, `seal_matrix_behavior_review.py`;
server (read-only, three bounded ssh_client calls): K3 `config.json`, shard-2 safetensors header,
`configuration_kimi_k3.py::is_kda_layer`, `compressed_tensors` 0.18.0 sources, the four ComfyUI custom
node dirs, comfy-core `ldm/minimax/{model,vae,audio_vae}.py`, `sd.py`, `latent_formats.py`,
`model_base.py` inside container `comfyui_h3_gpu4`, ACE-Step HF modeling/config, ACE VAE config,
`/home/cereal/comfyui/models/{vae,diffusion_models,checkpoints,text_encoders,loras}` top level.

---

## 0. Verified ground truth the specs stand on

### 0.1 Kimi K3 on disk — `/mnt/ai_data/models/moonshotai/Kimi-K3` [verified]
- `config.json` text_config: 93 layers, hidden 7168, `linear_attn_config`: `kda_layers` = 1..91 minus
  multiples of 4, `full_attn_layers` = 4,8,...,92 and 93, `num_heads` 96, `head_dim` 128,
  `short_conv_kernel_size` 4, `use_full_rank_gate` true, `gate_lower_bound` -5.0.
  `is_kda_layer(layer_idx) = (layer_idx + 1) in kda_layers` [verified configuration_kimi_k3.py:152-156]
  → 0-based layer index i is KDA iff (i+1) ∈ kda_layers: KDA at i ∈ {0,1,2, 4,5,6, ... , 88,89,90},
  gated-MLA at i ∈ {3,7,11,...,91,92} (24 MLA: 23 at i=4k−1 plus i=92). 69 KDA + 24 MLA = 93.
- `first_k_dense_replace` 1 → layer 0 has a dense `KimiMLP` (intermediate 33,792, SiTU); layers 1..92
  each have `block_sparse_moe` (`moe_layer_freq` 1). So 92 MoE layers, not 93.
- MoE: `num_experts` 896, `num_experts_per_token` 16, `num_shared_experts` 2, `moe_intermediate_size`
  3072, `routed_expert_hidden_size` 3584 (latent), `latent_moe_use_norm` true, router `sigmoid`,
  `moe_renormalize` true, `routed_scaling_factor` 1.0, `num_expert_group` 1, `topk_group` 1,
  `topk_method` noaux_tc (bias enters selection only, code line 723/750).
- MLA: `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64,
  `mla_use_nope` true, `mla_use_output_gate` true, `num_attention_heads` 96, `num_key_value_heads` 96.
  `v_head_dim` was past the 6000-char cut of my config read → [assumed 128]; receipt it from the
  `kv_b_proj` shape: rows = 96·(128 + v_head_dim).
- Activation `situ`, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0, `rms_norm_eps`
  1e-5, `attn_res_block_size` 12, vocab 163,840, `max_position_embeddings` 1,048,576, bf16.
- Quantization: compressed-tensors `mxfp4-pack-quantized`, group_size 32, symmetric float 4-bit,
  scale_dtype uint8, targets Linear, ignore = self_attn.*, shared_experts.*, mlp.(gate|up|down)_proj,
  lm_head, vision_tower, mm_projector (all bf16).
- Shard 2 (`model-00002-of-000096.safetensors`) header = 818,696 bytes, 5,404 tensors; layer 1
  (first MoE layer, KDA) lives entirely in shard 2. Per-layer tensor layout (prefix
  `language_model.model.layers.1.`):
  - experts.N.w1.weight_packed U8 [3072, 1792], w1.weight_scale U8 [3072, 112]; w3 same;
    w2.weight_packed U8 [3584, 1536], w2.weight_scale U8 [3584, 96]  (in-features 3584 / 3072 packed
    two per byte; scales = in/32 per output row → group along the INPUT axis).
  - gate.weight BF16 [896, 7168]; gate.e_score_correction_bias F32 [896];
    routed_expert_down_proj.weight BF16 [3584, 7168]; routed_expert_norm.weight BF16 [3584];
    routed_expert_up_proj.weight BF16 [7168, 3584]; shared_experts.{gate_proj,up_proj} BF16
    [6144, 7168], shared_experts.down_proj BF16 [7168, 6144].
  - self_attn (KDA): q_proj/k_proj/v_proj/g_proj BF16 [12288, 7168]; o_proj BF16 [7168, 12288];
    f_a_proj BF16 [128, 7168]; f_b_proj BF16 [12288, 128]; A_log F32 [128] (NOTE: 128 = head_dim,
    not num_heads — the modeling code declares A_log over num_heads=96 but the checkpoint stores 128;
    fla's `chunk_kda` takes A_log per head... this mismatch must be receipted at load: [assumed] the
    on-disk 128 is indexed by head_dim inside fla's gate `g = lower_bound·sigmoid(A_log.exp()·(g+dt_bias))`
    broadcasting over d; Chimera's `decay_log_scale` is per HEAD [verified layers.py:215] — one of the
    two readings is wrong and the exactness receipt (K3-1 step 6) decides); dt_bias F32 [12288];
    q_conv1d/k_conv1d/v_conv1d.weight F32 [12288, 1, 4]; o_norm.weight F32 [128]; b_proj BF16
    [96, 7168].
  - input_layernorm, post_attention_layernorm, self_attention_res_norm, mlp_res_norm BF16 [7168];
    self_attention_res_proj, mlp_res_proj BF16 [1, 7168].
  - MLA layer tensor names [assumed from modeling code]: self_attn.q_a_proj [1536,7168],
    q_a_layernorm [1536], q_b_proj [96·192=18432, 1536], kv_a_proj_with_mqa [512+64=576, 7168],
    kv_a_layernorm [512], kv_b_proj [96·(128+v)=24576 if v=128, 512], o_proj [7168, 96·v], g_proj
    [96·v, 7168]. NoPE: `rotary_emb = None`, no rotation is ever applied [verified k3_linear.py:403,
    439-440]; the 64 "rope" dims are plain extra shared key dims = Chimera `mla_extra_head_dim`.
  - Model-level names [assumed]: `language_model.model.embed_tokens.weight` [163840,7168],
    `language_model.model.norm.weight`, `language_model.model.output_attn_res_norm.weight`,
    `language_model.model.output_attn_res_proj.weight` [1,7168], `language_model.lm_head.weight`.
- Parameter arithmetic per MoE layer: one expert = 3·(3072·3584) = 33,030,144 params; 896 experts =
  29,595,009,024 (29.6B) = **59.2 GB at bf16** (the plan's "~30 GB per layer at bf16" is off by 2×;
  packed MXFP4 on disk ≈ 14.8 GB + 0.93 GB scales). KDA body ≈ 444M (q,k,v,g 352M + o 88M + decay/
  beta/conv ≈ 3M); MoE non-expert ≈ 190M (router 6.4M + down/up 51.4M + shared 132.1M); ≈ 634M
  non-expert per KDA layer [verified arithmetic from shapes].

### 0.2 MXFP4 decode — compressed_tensors 0.18.0 in the lab venv [verified]
- `compressed_tensors/compressors/nvfp4/helpers.py::unpack_fp4_from_uint8(a, m, n, dtype)`:
  `high = (a & 0xF0) >> 4; low = a & 0x0F; combined = torch.stack((low, high), dim=1).flatten()`
  → **low nibble = even column, high nibble = odd column** (matches `k3_layer_probe.dequant_mxfp4`).
  Sign = bit 3, magnitude index = bits 0-2 into `kE2M1ToFloat` (E2M1 table; the probe's
  `[0,.5,1,1.5,2,3,4,6]` [assumed identical — assert equality against `kE2M1ToFloat` in the receipt]).
- `quantization/utils/mxfp_utils.py::maybe_convert_from_mx_exp`: `scale = 2.0 ** (scale.int32 − 127)`
  (E8M0, bias 127) — matches the probe's `torch.ldexp(1, scale−127)`.
- `compressors/mxfp4/base.py::MXFP4PackedCompressor(NVFP4PackedCompressor)`: params
  (`weight_packed`, `weight_scale`), scale decompress = `decompress_mx_scale`. No global scale.
- fla-core 0.5.2 / flash-linear-attention 0.5.2 installed [verified]; `compressed_tensors` 0.18.0
  [verified]; `pydantic` got installed as a transitive dep of compressed-tensors — it is NOT used by
  any of our code (estate rule: no Pydantic in OUR code; the dependency is the vendor's).
- The lab venv torch 2.11+cu128 warns P40s (sm61) are unsupported — P40s are NOT usable from this
  venv; the Pascal venv `venv-pascal-cu118-torch260` is the P40 path [verified inventory].

### 0.3 Server state at 14:55 [verified]
- GPU0 RTX 3090 24 MiB used (free); GPU4 RTX 3090 17,731 MiB (Codex `comfyui_h3_gpu4`); GPU6 RTX
  8000 5 MiB used (idle, Codex-owned, claim pending); P40s idle. `p40probe` container running
  (unhealthy) on GPU1.
- `/home/cereal/omni-campaign/k3/`: `k3_layer_probe.py`, `pip_install.log`, `k3_layer1_cpu.log` whose
  only line is `layer 1: 5404 tensors in shards [...]` at 14:26 — i.e. the probe has been inside
  `read_tensors` on the CIFS share for ≥30 min (the <3 MB/s share). No receipt yet.
- Media: `/home/cereal/comfyui/models/vae/`: `minimax_h3_video_vae_fp16.safetensors` 5.21 GB,
  `minimax_h3_audio_vae_fp32.safetensors` 0.61 GB, `ace_1.5_vae.safetensors` 0.34 GB,
  `qwen_image_vae.safetensors` 0.25 GB, `Hunyuan3D-vae-v2-1-fp16.ckpt` 0.66 GB;
  `diffusion_models/`: `minimax_h3_fl2va_pruned_bf16.safetensors` 40.2 GB, `..._int8_convrot` 21.0 GB
  ×2 (fl2va/ref2va), `qwen_image_fp8_e4m3fn.safetensors` 20.4 GB, `acestep_v1.5_turbo.safetensors`
  4.79 GB, `acestep-v15-turbo/` dir; `checkpoints/hunyuan_3d_v2.1.safetensors` 7.37 GB;
  `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` 51.5 GB; `loras/h3/
  minimax_h3_turbo_v4_step600_ema.safetensors` 0.78 GB; ACE HF repo
  `/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo/{config.json, configuration_acestep_v15.py,
  modeling_acestep_v15_turbo.py, model.safetensors -> /workspace/ComfyUI/models/diffusion_models/
  acestep-v15-turbo/model.safetensors (container path; host = /home/cereal/comfyui/models/
  diffusion_models/acestep-v15-turbo/ [assumed]), silence_latent.pt}`; ACE VAE
  `/home/cereal/comfyui/models/ACE-Step/vae/config.json` = diffusers `AutoencoderOobleck`
  (audio_channels 2, decoder_input_channels 64, downsampling_ratios [2,4,4,6,10] → 1920 samples per
  latent frame, sampling_rate 48000 → 25 latent frames/s [arithmetic]).
- ComfyUI core is NOT on the host (`/home/cereal/comfyui/comfy/ldm` does not exist); it lives in the
  image `comfyui:codex-minimax-h3-runtime-env-r5-20260808` at `/opt/ComfyUI` (py3.10, torch
  2.6+cu124, diffusers 0.35.2, transformers 4.57.6) [verified via `docker exec` grep]. H3/ACE/
  Qwen-Image/Hunyuan3D VAE + DiT classes are comfy-core modules there:
  - `comfy/ldm/minimax/model.py`: `MiniMaxH3Model(hidden_size=5376, num_layers=50,
    token_refiner_num_layers=2, num_attention_heads=56, attention_head_dim=128,
    ffn_hidden_size=14336, latents_dim=24, audio_latents_dim=32, patch_size=(1,2,2), text_dim=5120,
    time_embed_dim=2688, sigma_shift_video=12.0, sigma_shift_audio=3.0, adaln_curve_grid=None, ...)`;
    `DiTBlock.forward(x, t_emb, mod_segments, rope_freqs, transformer_options)` =
    adaLN(shift,scale,gate ×2 from `adaln_proj(t_emb)`, 3 modalities) → `norm1` → `attn` (qkv_proj,
    q_norm/k_norm, rope) → gated add → `norm2` → `mlp` → gated add; `FinalLayer.forward(x, t_emb,
    video_seg, audio_seg)` → `video_out` (5376→96, fp32 island), `audio_out` (5376→32);
    `video_patch_proj` (96→5376, fp32), `audio_patch_proj` (32→5376, fp32), `condition_proj`
    (5120→5376) + `TokenRefiner` (2 blocks) for the Qwen3-VL text states; `PackedLayout(text_len,
    latent_t, latent_h, latent_w, audio_t, keyframes, refs, frame_count)` packs [text | video patches
    | audio frames] into ONE sequence `h` of shape [seq_len, 5376] (single stream, unbatched);
    `patchify_video(latent, (1,2,2))`, `pack_audio(latent)`; `rope_freqs(position_ids [S,3])` → [S,96].
    The pruned checkpoint has 20.11B elements (532 tensors) [inventory] vs the class default 50
    layers — the loader derives depth from the state dict [assumed]; the turbo node reads
    `h3_silu_temb_grid.safetensors` `silu_t_emb_grid` [1025, 2688] (adaLN curve form).
  - `comfy/ldm/minimax/vae.py`: `MiniMaxH3VideoVAE(in_channels=3, embed_dim=24, z_channels=24,
    ch_mult=(1,2,2,4,4,8), space_down=(2,2,2,2,1,1) → 16×, time_down=(1,2,2,1,1,1) → 4×,
    clip_length=17, tiling=True)`: `EncoderFCN3D` (causal Conv3d) → `quant_conv` (48→48 moments) →
    posterior mean; `post_quant_conv` (24→24) → `ViT3DDecoder(patch 16, patch_t 4, 36 layers, 32 heads
    ×64)`; `latents_mean/std` buffers; `pixel_mean/std` ImageNet. comfy `sd.py`: latent_channels 24,
    frames 17k+5 ↔ latents 5k+2, 16× spatial, working dtypes fp16/fp32, tiles internally.
  - `comfy/ldm/minimax/audio_vae.py`: `MiniMaxH3AudioVAE`: `encode(waveform [B,2,L] @32 kHz in
    [-1,1]) → normalized latents [B,32,2,T]`, 800 samples per latent frame (40 Hz), stereo channels
    encoded independently by a mono DAC encoder (`Encoder(64, rates (2,4,4,5,5), latent_dim 2048)` →
    `AttnProjection` 2048→32 → `mean_proj`); `decode(z [B,32,2,T]) → [B,2,L]` via `dec_in_proj`
    (32→2048) + `BigVGAN`. Posterior mean only, no sampling.
  - `latent_formats.py`: `MiniMaxH3Video` (24 ch, spatial 16, temporal 4, scale 1.0),
    `MiniMaxH3AV` (32 = max of the two streams), `Hunyuan3Dv2_1` (64 ch, 1-D, scale 1.00395),
    `ACEAudio15` (64 ch, 1-D, temporal_downscale 1764 = 44.1 kHz/25 Hz; note the ACE 1.5 VAE config
    says 48 kHz — comfy resamples [assumed]).
  - `model_base.py::MiniMaxH3.extra_conds`: `c_crossattn` = `preprocess_text_embeds` (condition_proj +
    token refiner once per run); payload dict carries `text_token_tags`, `keyframes`, `refs`,
    `cond_video_latents`, `cond_audio_latents`, noise-aug, seed, and the `PackedLayout` built from
    `latent_shapes` (video [B,24,T,H,W] rounded to 2×2 patch; audio length = latent_shapes[1][-1]).
  - `comfy/ldm/ace/ace_step15.py`: `AceStepDiTModel` (line 599), `AceStepDiTLayer` (249),
    `AceStepConditionGenerationModel` (1009), `FSQ`, `AceStepAudioTokenizer`; ACE 1.0 `MusicDCAE` in
    `ace/vae/`. `comfy/ldm/hunyuan3dv2_1/hunyuandit.py`: `HunYuanDiTPlain`, `HunYuanDiTBlock`,
    `MoEBlock`/`MoEGate` (Hunyuan3D 2.1's DiT has MoE FFN blocks [verified class names] — a C14-class
    problem, not a plain C16 block); `comfy/ldm/hunyuan3d/vae.py::ShapeVAE` (sd.py:820). Qwen-Image
    VAE = Wan 2.1-style `comfy/ldm/wan/vae.py::WanVAE` (sd.py:795; 16 latent ch, 8× spatial [assumed
    from the Wan branch]).
- Custom nodes (entry points) [verified]:
  - `ComfyUI-MiniMax-H3-Turbo/__init__.py`: classes `MiniMaxH3TurboSampler` (line 242, returns
    `comfy.samplers.KSAMPLER(_turbo_sampler)`, 4-step), `MiniMaxH3TurboLoRA` (440; `comfy.lora.load_lora`
    with key_map `diffusion_model.<module>.weight`, `_FrugalLoRA(comfy.weight_adapter.LoRAAdapter)`,
    `BypassInjectionManager`, wraps `WrappersMP.DIFFUSION_MODEL`); uses `comfy.model_sampling.
    ModelSamplingAV`; `h3_silu_temb_grid.safetensors`. No model loader of its own — the H3 model is
    loaded by comfy core (`UNETLoader` → `supported_models.MiniMaxH3`).
  - `ComfyUI_RH_ACE-Step/nodes.py`: `RunningHub_ACEStep_Loader` (line 9), `_GenerationParams`,
    `_Creator`, `_Artist`; `acestep/handler.py::load_lora`; `acestep/acestep_v15_pipeline.py` drives
    `AceStepHandler` (DiT) + `LLMHandler`.
  - `ComfyUI-kaola-ace-step/acestep_wrapper.py`: `_load_dit_model` = `AutoModel.from_pretrained(model_path,
    trust_remote_code)` with the HF dir on `sys.path` (lines 116-187); `_load_vae` =
    `AutoencoderOobleck.from_pretrained(vae_path)` (230-253); text encoder `AutoModel.from_pretrained`
    (260-271); `nodes.py` NODE_CLASS_MAPPINGS `ACE_STEP_TextToMusic/Cover/Repaint/.../LoRALoader`;
    `acestep_repo/` submodule dir exists.
  - `ComfyUI-Hunyuan3d-2-1/nodes.py`: `Hy3D21VAELoader.loadmodel(model_name, vae_config)` (571) with
    default `vae_config={num_latents: 4096, embed_dim: 64, ...}` building `ShapeVAE(**vae_config)`
    (line ~617); `Hy3D21VAEConfig` (623), `Hy3D21VAEDecode.process(vae, latents, box_v,
    octree_resolution, mc_level, num_chunks, mc_algo, enable_flash_vdm, force_offload)` (676);
    `Hy3DMeshGenerator`; `hy3dshape/hy3dshape/models/autoencoders/model.py::ShapeVAE(VectsetVAE)`:
    `from_pretrained` (157), `encode(surface, sample_posterior=True)` (324), `decode(latents)` (336);
    `pipelines.py::Hunyuan3DDiTFlowMatchingPipeline` (711); `surface_loaders.py::load_surface(mesh,
    num_points=8192)`, `SharpEdgeSurfaceLoader`; `models/denoisers/hunyuan3ddit.py::Hunyuan3DDiT` (284).
- ACE-Step 1.5 turbo HF config [verified]: `AceStepConditionGenerationModel`, hidden 2048, 24 DiT
  layers (`layer_types` alternating sliding_attention(128)/full_attention), 16 heads × 128, 8 kv heads,
  intermediate 6144 SiLU (`Qwen3MLP`), `in_channels` 192, `audio_acoustic_hidden_dim` 64, `patch_size`
  2, rope_theta 1e6, `timestep_mu` -0.4, `timestep_sigma` 1.0, lyric encoder 8 layers, timbre encoder 4,
  `num_audio_decoder_hidden_layers` 24, FSQ levels [8,8,8,5,5,5]. `AceStepDiTLayer.forward(hidden_states,
  position_embeddings, temb, attention_mask, ..., encoder_hidden_states, encoder_attention_mask)`:
  `(scale_shift_table + temb).chunk(6)` → adaLN self-attn (gated) → cross-attn (ungated residual) →
  adaLN MLP (gated); returns `(hidden_states,)`. `AceStepDiTModel.forward(..., context_latents)`:
  `hidden_states = cat([context_latents, hidden_states], -1)` (64 src + 64 chunk_mask + 64 noisy =
  192) → pad → `proj_in` → `condition_embedder(encoder_hidden_states)` → layers loop (line 1460) →
  `norm_out * (1+scale) + shift` → `proj_out` → crop. `prepare_noise`: src_latents width =
  context_latents.shape[-1]//2 = 64.

### 0.4 Chimera Omni target (local checkout `C:/Users/cereal/Documents/ai training/chimera-omni`) [verified]
- `config.py`: `FeedForwardKind = Literal["dense"]` only (`validate` rejects anything else); block kinds
  kda/local/csa/hca/gated_mla; `residual_mode` none | block_attnres (`attnres_block_size` 12 default);
  `activation_kind` silu | situ (`situ_gate_beta` 4.0, `situ_up_beta` 25.0 = K3's values);
  `native_feature_dim` 3072 default, `native_position_dim` 4, `native_type_count` 8,
  `native_bidirectional_refinement` (gated_mla blocks only). `k3_aligned_test_config()`: hidden 128,
  intermediate 384, 5 layers, pattern (kda,kda,kda,gated_mla) + `force_final_gated_mla` → layers
  0,1,2 kda; 3,4 gated_mla; attnres_block_size 2; head dims 32; kda_decay_rank 32; mla latent 64,
  query latent 64, key 32, extra 8, value 32; situ; native_feature_dim 64; bidirectional refinement on.
- `layers.py`: `KimiDeltaAttention` (reference recurrence; `q_proj/k_proj/v_proj`, `q_conv/k_conv/
  v_conv` = `CausalDepthwiseShortConv` with bias + SiLU, `decay_down/decay_up` (= K3 f_a/f_b),
  `decay_bias` [heads, key_dim] (= dt_bias), `decay_log_scale` [heads] (= A_log), `write_strength`
  (= b_proj), `output_gate` (= g_proj), `output_norm` = `HeadwiseRMSNorm` (= o_norm), `out` (= o_proj));
  gate: `decay_fraction = sigmoid(exp(A)·(logits + bias))`, `retention = exp(lower_bound ·
  decay_fraction)`, q/k L2-normalized, `delta_rule_update`. `GatedMultiHeadLatentAttention`
  (`q_down/q_norm/q_up`, `kv_down/kv_norm/kv_up`, `gate`, `out`; NoPE; extra key dims broadcast).
  `DepthAttentionResidual(norm, query[hidden])` = K3 `_apply_attn_res` with `score_weight =
  norm.weight ⊙ proj.weight`. `FeedForward(up: hidden→2I fused [gate | up], down: I→out)`,
  `_activate` situ identical to K3 `SituAndMul` (β tanh(g/β)·σ(g)·β_l tanh(u/β_l)).
  `ChimeraBlock.attention_output(x, native_media_slot_mask, valid_media_mask,
  enable_native_media_refinement, token_valid_mask)` uses `prefix_media_attention_mask` on gated_mla
  blocks for the bidirectional media refinement; `feed_forward_output`.
- `chimera.py`: `NativeContinuousMediaBridge.embed(features, type_ids, positions, valid_mask,
  noise_levels)`: `x = feature_proj(features ⊙ (1 + type_feature_scale[type])) + type_embed[type] +
  position_proj(pos) + position_fourier_proj(fourier(pos)) [+ noise_proj(fourier(sigma))]` → `norm` →
  type affine. `UnifiedNativeMediaEmissionHead.forward(hidden, type_ids)`: `norm` → type affine →
  `out` (hidden → native_feature_dim). `ChimeraOmniModel._forward_trunk` block_attnres loop mirrors
  K3 `_forward_attn_residual` exactly (completed_blocks appended at layer % attnres_block_size == 0,
  prefix_sum reset). `forward(...)` media kwargs: `native_media_features/type_ids/positions/indices/
  valid_mask/update_mask/noise_levels/targets/loss_mask/target_kind ∈ {reconstruction, velocity, x0}/
  bidirectional_refinement`; `aux["native_media_x0"]`.
- `native_media.py`: `image_to_native_patches(image [B,C,H,W], patch=32, feature_dim=3072)`,
  `video_to_native_patches(video [B,T,C,H,W], patch=32, tubelet=1, feature_dim, fps)`
  (source_feature_dim = tubelet·C·patch²), `waveform_to_native_segments(waveform [B,C,S], segment=3072,
  stride, feature_dim, sample_rate_hz=32000)`, inverses `native_patches_to_video`,
  `native_segments_to_waveform` (require `feature_width_lossless`), `_fit_dim` pads or adaptive-pools
  to feature_dim. `native_segments.py`: type ids image 1, video 2, audio 3, music 4, tts 5, ocr 6;
  positions schema thwr_v1.
- `media/training.py`: `sample_joint_av_noise_levels(type_ids, video_shift=12, audio_shift=3)`,
  `joint_av_sampling_schedule`, `make_rectified_flow_batch`, `rectified_flow_x0_sample`,
  `_shift_sigma(σ, s) = s·σ/(1+(s−1)σ)` — the same shifts as H3's `sigma_shift_video/audio` (12/3).
- `proof/h3_hybrid_av_flow.py`: FRAMES 8, FPS 8, SIZE 32, PATCH 16, TUBELET 2, stereo 32 kHz 12,288
  samples, SEGMENT 1536, FEATURE_DIM 1536, 32 video + 16 audio media tokens; `build_model` =
  `k3_aligned_test_config` with `native_feature_dim=1536`, `max_sequence_length=128`; 3,185,932 params
  (inventory, v4 report cc67c49 all 13 gates true); train 4000 steps; `evaluate_generation` = x0
  sampling 16 steps + `measure_direction/pitch/onsets`; `save_checkpoint`/`hash_state_dict`.
- neural-forge kernels reusable as-is: `mx45_build.py::prior_anchored(A, Y, prior, rel)` (ridge
  anchored to a prior, Cholesky), `law_abiding_solve(A, Y, prior)` / `law_abiding_solve_split(A, Y,
  check, prior)` (λ ladder, update_scale ≤ 1 veto), `pivoted_columns(F, k)` / `pivoted_columns_gram(G,
  k)` (deterministic greedy pivoted Cholesky selection — Python loop over n per step: O(n·k) Python
  iterations, must be vectorized for n ≥ 1e5), `write_scores`, `install_narrow_mlp(layer, keep, sel,
  w2p, dtype)` (LFM2 w1/w3/w2 layout), `narrow_mlp_site(child, parent_block, layer, evidence,
  keep_ff, solve, ...)` (composed-reader exactness + identity precondition + pivot + law solve),
  `Harness.capture(targets, pristine)` (forward-pre-hooks, fp64 rows, fit/hld split).
  `hf_closed_form_fit.py::fit_ridge_linear(input_rows, target_rows, ridge, fit_bias, validation_*)`,
  `fit_diagonal_norm(..., kind="rms_norm", epsilon, has_bias)`, `normalize_rows`.
  `experiments/run_chimera_omni_toy_fold.py::ActivationTap` (forward hooks incl. the live
  completed_blocks snapshot), `build_residual_chart` (one global orthonormal chart by SVD of pooled
  residual rows), `ChildInstaller` (chart_transport / ridge_linear / diagonal_norm /
  install_depth_selector, execution-order recapture) — the exact-4x fold of the tiny Chimera.
  `hf_causal_stable_latent_moe_fold.py::capture_chimera_stable_latent_moe_layer_evidence /
  stable_latent_moe_routing / compile_hf_causal_stable_latent_moe_fold` target a
  `StableLatentMoEFeedForward` module that the CURRENT chimera-omni config no longer admits
  (`FeedForwardKind = "dense"`), and its contract forbids changing `expert_count` ("kind
  translation") — so it is NOT the C14 kernel; it is prior art for routing recovery and per-expert
  row budgets only. `experiments/seal_matrix_behavior_review.py` seals the frozen 15-case verdict
  (`--useful/--partial/--observed-miss`, `--dossier`), the review instrument to reuse.

---

## (a) K3-1 — the streamed per-layer calibration forward

Files: server `/home/cereal/omni-campaign/k3/k3_stream.py` (new; grows out of `k3_layer_probe.py`),
`/home/cereal/omni-campaign/k3/k3_receipt.py`; local mirror in this lane dir. Runtime: lab venv
`/home/cereal/model-surgery-lab/venv/bin/python` on GPU0 (3090, bf16) with fla Triton kernels;
CPU fallback = fp32 reference path (below). Everything wrapped in `timeout`; never walks `/mnt/ai_data`.

### a.1 Loading one `KimiDecoderLayer` from the shards via the index
1. `index = json.load(model.safetensors.index.json)["weight_map"]`; `keys = [k for k in index if
   k.startswith(f"language_model.model.layers.{i}.")]`; `shards = sorted(set(index[k]))` (layer 1 → one
   shard; other layers may straddle two — handle the list) [verified for layer 1].
2. `read_tensors(shard, wanted)` from the probe (header parse, offset-sorted sequential reads, 64 MiB
   buffer) — keep. Read ORDER = shard order, offset order; log MB/s per shard (this number prices the
   whole campaign; the probe's `receipt["steps"]["read"]["MB_per_s"]` is that receipt).
3. Split the layer dict: `attn = {k: v for k in keys if ".self_attn." in k}`, `res = {input_layernorm,
   post_attention_layernorm, *_res_norm, *_res_proj}`, `moe_body = {gate.*, routed_expert_*,
   shared_experts.*}`, `experts_packed = {experts.N.{w1,w2,w3}.{weight_packed,weight_scale}}` (896×6
   uint8 tensors, 14.8 GB + 0.93 GB) — keep PACKED in RAM, never materialize all 896 in bf16 at once
   (59.2 GB) except for the reference receipt.
4. Non-layer tensors (read once, cached to local disk `/home/cereal/omni-campaign/k3/body/`):
   `embed_tokens.weight` (2.35 GB bf16), `norm.weight`, `output_attn_res_{norm,proj}`, `lm_head.weight`
   (2.35 GB). Then stage the whole non-expert body (~100 GB: every layer's attn + res + moe_body +
   layer-0 dense mlp) to `/home/cereal/omni-campaign/k3/body/L{i:02d}.safetensors` as it streams by —
   the plan's "copy K3's non-expert body local" falls out of the first pass for free.

### a.2 MXFP4 dequant (and the cross-check against compressed_tensors)
- Reference kernel (probe's `dequant_mxfp4(packed, scale, group=32)`): `codes = stack((p & 0xF, p >> 4),
  -1).reshape(out, -1)`; `mag = E2M1[codes & 7]`, `sign = (codes & 8) ? -1 : 1`; `vals.reshape(out, in/32,
  32) * 2^(scale−127)[..., None]` → bf16. Matches compressed_tensors 0.18.0 layout [verified §0.2].
- Cross-check receipt (run once on expert 0 of layer 1): `from compressed_tensors.compressors.nvfp4.helpers
  import unpack_fp4_from_uint8, kE2M1ToFloat` and `from compressed_tensors.compressors.mx_utils import
  decompress_mx_scale` [module verified to exist; symbol name [assumed] — fall back to
  `quantization.utils.mxfp_utils.maybe_convert_from_mx_exp`]; `ref = unpack_fp4_from_uint8(packed,
  3072, 3584, torch.float32) * decompress_mx_scale(scale).float().repeat_interleave(32, dim=1)`;
  assert `torch.equal(E2M1, kE2M1ToFloat)` and `(ref − ours.float()).abs().max() == 0` (both are
  exact products of a 4-bit code and a power of two; bf16 cast of ±6·2^e is exact). Record sha256 of
  `ours` for expert 0 w1/w2/w3 in the receipt.
- GPU-resident packed experts (the design that makes the MoE half fast): move `experts_packed`
  (15.7 GB uint8) to GPU0 once per layer; dequantize per expert on demand into a bf16 [3072,3584]
  scratch (22 MB) with a vectorized torch kernel (the probe's ops are all elementwise/gather — they
  run on GPU as-is; Triton fusion is an optimization, not a requirement). GPU0 budget: packed 15.7 GB
  + KDA body 0.9 GB bf16 + activations for a 4k-token chunk ≈ 2 GB → fits 24 GB. On the RTX 8000 (46
  GB, claim pending) the same plus fp16 compute after an overflow check.

### a.3 Running the layer with fla on GPU0 (bf16) / CPU fallback
- Attention: instantiate the REAL classes from the repo's `modeling_kimi_linear.py`
  (`sys.path.insert(0, K3_DIR)`; `from configuration_kimi_k3 import KimiLinearConfig`;
  `KimiLinearConfig(**config["text_config"])`), `KimiDeltaAttention(cfg, layer_idx)` or
  `KimiMLAAttention(cfg, layer_idx)` built on `device="meta"` then `load_state_dict(attn, assign=True)`
  with the bf16 tensors (A_log/dt_bias/conv weights/o_norm fp32 as stored). Set
  `cfg._attn_implementation = "sdpa"` for MLA (flash_attn is NOT installed [verified inventory]; torch
  SDPA accepts q_head_dim 192 ≠ v_head_dim 128); we never construct `KimiLinearModel`, whose
  `__init__` force-sets flash_attention_2. transformers 5.12.1 vs the file's `>= 4.56` assert and its
  imports (`masking_utils.create_causal_mask`, `utils.generic.OutputRecorder/check_model_inputs`,
  `pytorch_utils.ALL_LAYERNORM_LAYERS`) [assumed importable; if not, use the transformers-4.57.3
  overlay `/home/cereal/model-surgery-lab/python_overlays_transformers_4_57_3` on PYTHONPATH].
- KDA on GPU: `KimiDeltaAttention.forward(hidden_states, attention_mask=None, cache_params=None,
  cu_seqlens=cu)` → `chunk_kda(q,k,v,g,beta,A_log,dt_bias, use_qk_l2norm_in_kernel, use_gate_in_kernel,
  use_beta_sigmoid_in_kernel, safe_gate=True, lower_bound=-5, transpose_state_layout=True, cu_seqlens)`
  [verified call]. Pack the calibration corpus as one [1, T_chunk, 7168] row with `cu_seqlens` (varlen)
  so sequences of 4k–32k tokens never leak state into each other; chunk T into 16k-token pieces to
  bound activation memory (q,k,v,g each [T,12288] bf16 = 0.4 GB per 16k).
- MLA on GPU: per sequence (or block-diagonal causal mask) through `KimiMLAAttention.forward`; KV cache
  off. NoPE — no rotary anywhere [verified].
- CPU fallback / fp32 reference (used for the exactness receipt and whenever GPU0 is not free): the
  chimera-omni reference `KimiDeltaAttention.forward_with_state` (layers.py:289) run with a weight
  remap: q_proj→q_proj, k_proj→k_proj, v_proj→v_proj, q_conv1d.weight [12288,1,4]→`q_conv.conv.weight`
  (fla `ShortConvolution` has bias=False by default [assumed] vs Chimera's Conv1d bias=True — zero the
  bias; both apply SiLU), f_a_proj→decay_down, f_b_proj→decay_up, dt_bias [12288]→decay_bias
  [96,128], A_log→decay_log_scale (SHAPE CONFLICT: disk [128] vs Chimera [96] — see §0.1; the receipt
  resolves which axis A_log indexes: try both broadcasts and keep the one that matches fla to bf16
  precision), b_proj→write_strength, g_proj→output_gate, o_norm [128]→output_norm.weight [96,128]
  (broadcast the shared 128-vector to every head), o_proj→out. This IS the D4 acceptance receipt
  "one K3 layer exact against the K3 reference code" — for free.
- MoE half (GPU, packed experts): `logits = X.float() @ gate.T`, `scores = sigmoid(logits)`,
  `topk_idx = topk(scores + e_score_correction_bias, 16)`, `w = scores.gather(idx)`; `w /= w.sum(-1)`;
  `× routed_scaling_factor` (1.0) [verified KimiMoEGate]. `L = X @ W_down.T` [T,3584]. Group tokens by
  expert (argsort, as `moe_infer`), for each touched expert dequantize w1,w3,w2 → `h = SiTU(L_e @ w1.T,
  L_e @ w3.T)` with K3's `SituAndMul` (fp32 inside, cat[gate|up] layout) → `y_e = h @ w2.T`; scatter-add
  `w_e · y_e` into `Y_lat` [T,3584] fp32. Then `Y_norm = RMSNorm(Y_lat)·γ_routed`, `routed = Y_norm @
  W_up.T`, `shared = SiTU(X@Wsg.T, X@Wsu.T) @ Wsd.T` (6144, full 7168 space; note `KimiMLP` situ path
  also uses `linear_beta` 25 on the up branch [verified]), `moe_out = routed + shared`. The probe used
  SiLU and no renorm-of-bias subtlety — replace with the exact SiTU path above.
- Layer 0 is the one dense layer (`mlp.gate_proj/up_proj/down_proj` 33,792 SiTU) — plain `KimiMLP`.

### a.4 AttnRes `block_residual` plumbing across layers (exact transcription of `_forward_attn_residual`)
State carried layer to layer: `prefix_sum` [T,7168] and `block_residual` [T, n_blocks, 7168].
```
block_residual = zeros(T, 0, 7168); prefix_sum = embed(ids)
for i in range(93):
    h = prefix_sum
    if block_residual.shape[1] > 0:
        h = attn_res(prefix_sum, block_residual, self_attention_res_proj, self_attention_res_norm)
    if i % 12 == 0:                      # i = 0,12,24,36,48,60,72,84  -> 8 snapshots
        block_residual = cat(block_residual, prefix_sum[:,None]); prefix_sum = None
    a = self_attn(input_layernorm(h))
    prefix_sum = a if prefix_sum is None else prefix_sum + a
    h = attn_res(prefix_sum, block_residual, mlp_res_proj, mlp_res_norm)
    m = moe_or_mlp(post_attention_layernorm(h))
    prefix_sum = prefix_sum + m
final = attn_res(prefix_sum, block_residual, output_attn_res_proj, output_attn_res_norm)
logits = lm_head(norm(final))
```
`attn_res(prefix, blocks, proj, norm)`: `v = cat(blocks, prefix[:,None])`, keys = RMS-normalized v
(no weight), `scores = (k · (norm.weight ⊙ proj.weight)).sum(-1)`, `probs = softmax(scores)`, out =
`probs @ v` (fp32, cast back) [verified `_apply_attn_res`]. Consequence for streaming: at layer 0 the
embedding goes INTO `block_residual` and `prefix_sum` restarts from the attention output — the
"residual stream" a folding harness must cache per layer is the PAIR (prefix_sum, block_residual
snapshot count), not one tensor. Chimera's `DepthAttentionResidual` is the same op with
`query = norm.weight ⊙ proj.weight` pre-multiplied [verified layers.py:520-538] — K3's trained
`*_res_proj` install as `query = res_norm.weight * res_proj.weight.squeeze(0)` (D4 note).

### a.5 Per-layer cache to local disk
Directory `/home/cereal/omni-campaign/k3/cache/T{T}/L{i:02d}/` (safetensors, bf16 unless noted) with a
sha256 receipt JSON per file:
- `prefix_in.bf16` [T,7168] (the prefix_sum entering the layer; None → absent flag), `blocks_count`
  (int; snapshots live once in `/cache/T{T}/blocks/B{k}.bf16`, written at layers 0,12,...,84),
- `attn_in.bf16` (post `attn_res` + `input_layernorm`, optional), `attn_out.bf16` [T,7168],
- `moe_x.bf16` [T,7168] = `post_attention_layernorm(h)` — the C14 input X,
- `router_topk_idx.int16` [T,16], `router_topk_w.bf16` [T,16], `router_scores.bf16` [T,896] (optional,
  0.46 GB at 256k — keep for mass statistics),
- `latent_in.bf16` [T,3584] = L, `y_lat.fp32` [T,3584] (pre-norm routed accumulator — the C14 ridge
  TARGET; fp32 because the ridge target must not be bf16-quantized), `moe_out.bf16` [T,7168],
- `prefix_out.bf16` [T,7168].
Sizes at T = 65,536: moe_x 0.94 GB, latent_in 0.47, y_lat 0.94 (fp32), moe_out 0.94, prefix_in/out 0.94
each, attn_out 0.94 → ≈ 6.1 GB/layer → 92 layers ≈ 560 GB (too close to the 615 GB free). Decision
line D-a1 for Cyril: first full pass at T = 32,768 (≈ 280 GB, all layers) OR T = 65,536 storing only
{moe_x, latent_in, y_lat, router_topk_*} (2.4 GB/layer → 220 GB) with prefix states regenerated on
demand. Recommendation: the second (C14 needs X, L, Y, router; the review needs prefix states only at
the layers being swapped, and those are recomputed by re-streaming the prefix). Router statistics per
layer (`expert_mass[e] = Σ_t w_e(t)`, `expert_count[e]`) are written to `router_stats.json`.
- Calibration corpus: the plan's ≥32k-token evidence corpus tokenized with K3's own tiktoken
  encoding (`encoding_k3.py` + `tiktoken` [assumed installed in the lab venv — receipt it]); fit /
  held-out split by sequence, sealed with a manifest sha256 as `split_by_batch` does.

### a.6 Exactness receipt vs the reference forward (K3-1 rung 0, the first deliverable)
`k3_receipt.py --layer 1 --tokens 256 --seed 0`:
1. Reference: full bf16 materialization of the 896 experts on CPU (59.2 GB RAM) inside the REAL
   `KimiDecoderLayer(cfg, 1)` built from `modeling_kimi_linear.py` (attention module on GPU0 with fla,
   `block_sparse_moe` on CPU with `moe_infer`), `_forward_attn_residual(hidden, block_residual=embeddings
   snapshot)` → (prefix_out_ref, block_residual).
2. Harness: `k3_stream.run_layer(1, ...)` on the same 256 tokens (GPU packed experts).
3. Compare: rel-L2 and mean cosine of `prefix_out`, `attn_out`, `moe_out`, `y_lat`; top-16 set equality
   per token; sha256 of both outputs (fp32 bytes). Proposed decision line D-a2: rel-L2 ≤ 2e-2 on
   `prefix_out` and top-16 agreement ≥ 99% of tokens (bf16 ties at the router boundary are the only
   admissible difference); anything larger is a harness bug, not a tolerance.
4. Also emit the fp32 reference-recurrence receipt (Chimera `KimiDeltaAttention` remap, §a.3) vs fla:
   rel-L2 on `attn_out` at 256 tokens — decides the A_log axis question and is the D4 acceptance.
5. Timings in the receipt: read MB/s, dequant s, attention s, MoE s (experts touched, max tokens on one
   expert), total. From these: hours per layer × 92 = the K3 calibration price.

Throughput expectations [arithmetic]: at T = 64k, attention ≈ 2·T·0.44B ≈ 56 TFLOP (≈ 2–5 s on the
3090), MoE ≈ 2·T·16·33M ≈ 68 TFLOP + dequant (≈ 10 s); the share read (16 GB packed+body per layer)
dominates: 3 MB/s → 90 min/layer, 100 MB/s (share idle) → 3 min/layer. Do not stage experts locally
(1.37 TB packed > 615 GB free, and the estate forbids deleting staged copies); wait for the Qwen
download to finish, then stream from the share at its idle speed. Stage only the non-expert body.

---

## (b) C14 — expert-union channel coreset solver (per MoE layer)

File: `/home/cereal/omni-campaign/k3/c14_union_coreset.py` (CPU fp64 Grams + GPU feature evaluation).
Inputs per layer from §a.5: X [T,7168] bf16, L [T,3584] bf16, Y = y_lat [T,3584] fp32,
topk_idx/topk_w, router_stats, and the layer's packed experts + moe_body.

### b.1 The object being folded (exact statement)
Routed path R(X) = W_up · RMSNorm_γ( Σ_{e∈top16(t)} w_e(t) · W2_e · SiTU(W1_e L_t, W3_e L_t) ),
L = W_down X. Dense replacement (rung K): R'(X) = W_up · RMSNorm_γ( W2' · SiTU(W1_S L, W3_S L) ) where
S is a set of K channels drawn from the union U = {(e, j): e < 896, j < 3072} (|U| = 2,752,512),
W1_S / W3_S = the VERBATIM rows W1_e[j], W3_e[j] (channel coreset, never SVD — LFM2 law 40/42), and
W2' [3584×K] is ridge-refit on the cached (features, Y). `W_down`, `routed_expert_norm`, `W_up` and
the two shared experts (6144 SiTU channels in the 7168 space) are kept VERBATIM. The router is
deleted. SiTU constants (β 4, β_l 25) are identical for every expert [verified config] so the child's
activation is one `SituAndMul`. Chimera install: `FeedForward.up.weight = cat(W1_S, W3_S)` (chunk(2)
gives gate first [verified layers.py:570]), `down.weight = W2'` — but see §b.6 for the two forms.

### b.2 Features and weights
- Routed feature of channel c=(e,j) at token t: `φ_c(t) = w_e(t)·1[e ∈ top16(t)] · SiTU(W1_e[j]·L_t,
  W3_e[j]·L_t)` (router mass built in; zero where not routed). This is what the parent actually
  writes: Y_t = Σ_c φ_c(t) W2_c.
- Dense feature (what the child computes): `ψ_c(t) = SiTU(W1_e[j]·L_t, W3_e[j]·L_t)` on EVERY token.
- Selection is ranked on φ (router-mass-weighted stream statistics, the plan's wording); the refit
  target is Y and the refit design matrix is ψ_S evaluated on ALL tokens (the honest dense
  objective). Prior for the ridge anchor: `W2_prior[:, c] = W2_e[:, j] · m_c` with `m_c =
  E_t[w_e(t) 1[e∈top16(t)]]` (the channel's mean routed weight, from router_stats) — recorded.

### b.3 Two-stage pivoted selection (so the Gram fits in RAM)
- Stage 1 (per expert, 896 solves): rows R_e = tokens routing to e (≈ T·16/896 = T/56; at T=65,536
  ≈ 1,170 rows; at 256k ≈ 4,680); `F_e = diag(w_e) · SiTU(L_{R_e} W1_e^T, L_{R_e} W3_e^T)` [n_e × 3072];
  Gram `G_e = F_e^T F_e` fp64 (3072² × 8 B = 75.5 MB; 896 Grams = 67.6 GB if all kept — keep only the
  chosen columns); `sel_e = pivoted_columns_gram(G_e, k_e)` with `k_e = round(K_pool · mass_e /
  Σ mass)` (router-mass-proportional allocation, floor 1). K_pool = pool size for stage 2.
  Cost: 896 × (n_e·3072² ≈ 11–44 GFLOP) — minutes on GPU0; the greedy loop on 3072 columns is fast
  enough as written (n·k ≈ 3072·128 Python iterations per expert ≈ 0.4M → seconds; vectorize
  `argmax` anyway).
- Stage 2 (union re-pivot): pool P = ∪_e sel_e, |P| = K_pool; build `F_P` [T × K_pool] as φ
  (routed-weighted) in GPU chunks of 8k tokens, accumulate `G_P = F_P^T F_P` fp64 on CPU; run a
  VECTORIZED pivoted Cholesky (`diag.argmax()` per step, rank-1 update of an n×k L; O(n·k²)) to K.
  Sizes: K_pool = 114,688 (896×128) → G fp64 = 105 GB (fits 251 GiB with headroom only when the
  layer's packed experts are on GPU and no bf16 expert copy is in RAM; fp32 52.6 GB) — O(n·k²) at
  K = 65,536: 4.9e14 flop → hours on CPU fp64, ~10 min on GPU fp32 (L = 115k×65k fp32 = 30 GB → GPU
  cannot hold it; CPU it is). K_pool = 57,344 (896×64) → G fp64 26 GB, K = 17,408: 1.7e13 → minutes.
- Practical rung policy (proposed D-b1): rung 65,536 = stage 1 ONLY with k_e allocated by mass
  (K_pool = K, no stage 2; the union Gram at 65k is 34 GB fp64 — affordable for the REFIT but the
  O(n·k²) re-pivot is not worth it when every pool column is kept); rungs 32,768 / 17,408 / 8,704 =
  stage 1 to K_pool = 2K (oversample ×2) then stage 2 to K. Record both arms at 32,768 (stage-1-only vs
  two-stage) once to price stage 2 behaviorally.

### b.4 Ridge refit of W2' (and the "up_proj" question)
- Design A = ψ_S [T × K] evaluated densely on the fit split in GPU chunks (T=65,536, K=65,536: 2·T·3584
  ·K·2 ≈ 6e13 flop, seconds); accumulate `G = AᵀA` (fp64 on CPU, K² × 8 B = 34 GB at 65,536; 8.6 GB at
  32,768; 2.4 GB at 17,408) and `B = AᵀY` [K × 3584]. Solve with `prior_anchored`-style Cholesky
  (`G + λI`, `λ = rel · mean diag G`, `B + λ · prior^T`) through `law_abiding_solve_split` (λ ladder
  chosen on the internal fit split; update_scale ≤ 1 veto) — reuse the mx45 functions verbatim
  (they take A, Y directly; add a Gram-input variant `prior_anchored_gram(G, B, prior, rel)` because A
  at [65k × 65k] fp64 = 34 GB is too big to hold as rows — the Gram form is identical arithmetic).
- Target = y_lat (pre-norm, fp32) so `routed_expert_norm` and `W_up` stay verbatim (exact downstream).
  `up_proj` is NOT refit in the exact-latent form. Held-out scores: `write_scores` (relL2, cos, cos_p05)
  of `ψ_S W2'ᵀ` vs y_lat on the held split, plus the same scores AFTER norm+up (the 7168 write) and
  the whole-layer `prefix_out` drift — all three recorded; none gates.
- SiTU handling: features are computed with K3's `SituAndMul` (fp32 internally) — identical to
  Chimera's `_activate` — so verbatim gate/up rows reproduce ψ exactly in the child; the coreset is on
  the nonlinear map (no linearization). The `linear_beta` tanh on the up branch is part of ψ.

### b.5 Storage arithmetic per layer (parameters, MoE block incl. shared, vs parent 29.785B)
| rung K | routed dense (3·K·3584) | + down/up 51.4M + shared 132.1M | ratio (routed only / whole MoE) |
|---|---|---|---|
| 65,536 | 704.6M | 888.1M | 42.0× / 33.5× |
| 32,768 | 352.3M | 535.8M | 84.0× / 55.6× |
| 17,408 | 187.2M | 370.7M | 158.1× / 80.4× |
| 8,704 | 93.6M | 277.1M | 316.2× / 107.5× |
All far past 4×. Whole model at K=17,408: 92 × 370.7M + 0.19B (layer-0 mlp) + 93 × 0.44B attn + 4.7B
embed/head ≈ 80B dense (the plan's "~93B" counted the shared experts twice or kept the router — use
80B, 160 GB bf16).

### b.6 The two child forms (both built, both measured)
- Form E (exact-latent, for the dense-K3 review K3-3): module `DenseLatentFFN` = shared SiTU-FFN
  (verbatim) + [down → dense SiTU(K) → W2' → RMSNorm_γ → up]. Not a single Chimera `FeedForward`.
- Form M (merged, standard-export form): gate/up rows lifted EXACTLY into the 7168 space
  (`W1_S · W_down` [K × 7168], `W3_S · W_down`), then the norm removed by refitting `W2''` [7168 × K]
  directly to the routed 7168-write `W_up RMSNorm_γ(y_lat)` (ridge, same A) — an approximation of the
  norm, measured; concatenated with the 6144 shared channels → ONE SiTU FFN with I = K + 6144 in
  Chimera's `FeedForward` / `kimi-k3` dense layout. Storage: 3·(K+6144)·7168 (e.g. K=17,408 → 506M —
  larger than form E's 371M because the lift trades the 3584 latent for 7168; report both).
Proposed D-b2: form E for K3-3 scoring; form M only for children that must export.

### b.7 Code plan (functions)
`load_layer_cache(i, T)`, `expert_masses(router_stats)`, `stage1_per_expert(experts_packed, L, topk,
k_alloc) -> sel_e`, `pool_features_chunked(P, L, topk_w, weighted: bool) -> Gram/B accumulators`,
`pivoted_cholesky_vectorized(G, k)`, `refit_w2_gram(G, B, prior, check_rows)` (wraps
`law_abiding_solve_split` in Gram form), `build_form_E(...)`, `build_form_M(...)`, `receipt(...)`
(shapes, param counts, sha256 of every installed tensor, held-out scores, λ, update_scale).

---

## (c) Streamed whole-model review with N layers replaced

Hard fact: K3's unreplaced MoE layers cannot decode from RAM (92 × 14.8 GB packed = 1.37 TB > 251 GiB),
and per-token expert fetches from the share (16 experts × 16.5 MB packed × 93 layers ≈ 24 GB per
generated token) are infeasible at any share speed measured here. Therefore:
- R-proxy (any N, streamed, one pass, diagnostic ONLY — never a gate): teacher-forced review. Inputs =
  the frozen 15-case + bank-v2 prompts concatenated with the PARENT's own answers (parent answers
  come from R-full on the all-dense child? No — from the proxy's first pass with N = 0 through the
  argmax path is not generation either). So: the prompts + a fixed reference continuation; the harness
  streams all 93 layers once with layers ∈ N swapped for their C14 children (re-running the cached
  prefix from the first swapped layer onward), and records at the answer positions: KL(parent ‖ child)
  logits, top-1/top-5 agreement, per-case drift of `final` hidden. File `k3_review_stream.py`. Cost =
  one K3-1 pass over ~20k tokens. Use: ORDER the layers/rungs (which layer's K3-2 child to build
  next), never to declare a child dead.
- R-full (the intelligence gate, Cyril-scored): only when EVERY MoE layer is dense (K3-3). Form-E
  dense-K3 at K=17,408 ≈ 80B → 160 GB bf16 RAM-resident; decode on the 88-CPU host in bf16 (≈ 1–3
  s/token [assumed]; 15+30 prompts × 256 tokens ≈ 3–10 h) via a `torch` CPU implementation of the
  streamed forward with KV/recurrent caches (`KimiDynamicCache` semantics: conv states + recurrent
  state per KDA layer, K/V per MLA layer; `fused_recurrent_kda` has no CPU kernel → use the Chimera
  reference recurrence `forward_with_state` per token). GPU split-decode (2×24 + 46 + 4×24 = 190 GB
  VRAM in fp16; P40s need the Pascal venv and fp16 overflow checks) is the fast path but is
  cross-venv — build the CPU path first. Output = the same dossier `seal_matrix_behavior_review.py`
  expects (verbatim responses, `--useful/--partial/--observed-miss`, artifact-tolerant: loops /
  cap-rides / missing EOS are NOT content misses).
- Growth of coverage ("install 3 sites at a time"): R-proxy for N ∈ {1, 3, 9, 27, 92}; R-full at N = 92
  (all rungs 65,536 → 8,704 by succession); then R-full re-runs when depth/width folds (K3-4) land.
  Proposed D-c1: the first R-full is at K=65,536 (the mildest rung, 888M/layer → ≈ 128B... too big for
  RAM at bf16 (256 GB)) → the first R-full must be at K ≤ 32,768 (≈ 96B, 192 GB) — or at 65,536 with
  the P40/RTX8000 split. Cyril's call.

---

## (d) C15 — VAE absorption into the boundary projection / emission head

Files: `/home/cereal/omni-campaign/h3/c15_vae_pairs.py` (runs INSIDE the comfy image because the
H3/Qwen-Image/Hunyuan3D VAE classes are comfy-core: `docker run --rm --gpus '"device=0"' -v
/home/cereal/comfyui/models:/opt/ComfyUI/models -v /home/cereal/omni-campaign:/work
comfyui:codex-minimax-h3-runtime-env-r5-20260808 /opt/environments/python/comfyui/bin/python
/work/h3/c15_vae_pairs.py ...` — a container WE start, never touching `comfyui_h3_gpu4`; the ACE
Oobleck VAE runs there too via diffusers 0.35.2), and `/home/cereal/omni-campaign/h3/c15_fit_maps.py`
(lab venv, fp64 ridge). Pair files are safetensors under `/home/cereal/omni-campaign/h3/pairs/<vae>/`.

### d.1 Obtaining (raw patch, latent) pairs
Common loader: `import comfy.sd, comfy.utils; vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))`
(the branch is picked by state-dict keys: H3 video by `decoder.transformer_blocks.0.scale1` +
`encoder.down.5.block.0.conv1.weight`; H3 audio by `pre_block.attn.zero_k_bias`; Wan/Qwen-Image by
`decoder.upsamples...`; Hunyuan3D `ShapeVAE` at sd.py:820) [verified sd.py]; `z = vae.encode(pixels)`,
`x = vae.decode(z)` [comfy `VAE.encode/decode` signatures: image [B,H,W,C] in [0,1]; video [B,T,H,W,C];
audio dict/waveform [B,C,S] — [assumed]; the H3 audio path documents encode input [B,2,samples] and
decode latent [B,32,2,T]]. Or call the `first_stage_model` directly (`vae.first_stage_model.encode`) to
skip comfy's tiling/normalization wrappers — do the latter and record `latents_mean/std` handling.
- H3 video VAE: raw clip [B, T=17k+5, 3, H, W] (fps as sampled) → latent [B, 24, 5k+2, H/16, W/16].
  DiT token = 2×2 latent patch (`patchify_video (1,2,2)`) = 96 numbers ↔ raw 32×32 px × (4 frames per
  latent frame; first latent frame ↔ 5 frames) × 3 = 12,288 raw values (aligned tubelet: `patch=32,
  tubelet=4`, Chimera `video_to_native_patches(..., patch=32, tubelet=4)` → `source_feature_dim` 12,288;
  positions from its thwr grid). Pair rows: (raw tubelet [12288], latent patch [96], plus a CONTEXT
  variant: the 3×3 spatial × 2 temporal neighbor tubelets [12288×18]). Corpus: procedural clips (the
  proof's `make_clip` scaled to 64–256 px), the compliant media bench clips, and H3's own outputs are
  OFF the table as training data (license) — use natural video from the curated dataset only if
  Cyril's internal-use ruling covers it (it covers running H3 internally; VAE encodes of OUR clips are
  our data).
- H3 audio VAE: waveform [B,2,L] 32 kHz → [B,32,2,T], 800 samples/frame. Aligned Chimera segment:
  `waveform_to_native_segments(segment=800, stride=800, sample_rate_hz=32000)` (one token per channel
  per frame; Chimera default segment 3072 does NOT align — the H3-aligned proof rung uses 800 or 3200
  = 4 frames ↔ 128 latents). Pairs: (segment [800] (+ ±2 context = 4000), latent [32]) per channel.
- ACE VAE (AutoencoderOobleck, 48 kHz stereo, 1920 samples/frame, 64 ch): pairs (stereo frame
  [2×1920 = 3840] (+ context), latent [64]). ACE DiT token = 1 latent frame (patch_size 2 acts on the
  proj_in conv [assumed]) → 64 dims, the `native_feature_dim` for the ACE-aligned media band.
- Qwen-Image VAE (Wan-style 16 ch, 8×): pairs (8×8×3 = 192 raw per latent pixel; DiT token = 2×2
  latents = 16×16×3 = 768 raw ↔ 64 latent) with 3×3 neighbor context (6912).
- Hunyuan3D 2.1 `ShapeVAE` (num_latents 4096 × embed_dim 64, vecset — latent tokens are learned
  queries over a point cloud, NOT spatially indexed): `encode(surface [B,N,3+3(+sharp)])` from
  `load_surface(mesh, 8192)` / `SharpEdgeSurfaceLoader`; `decode(latents)` → cross-attention decoder
  evaluated on query points → SDF grid (`Hy3D21VAEDecode` with octree_resolution, mc_level). No
  patch-local pairing exists; the C15 map for 3D is whole-object: E: PCA(64³ voxel-SDF grid) → 4096×64
  latents (ridge in a reduced basis), D: latents → SDF values at a fixed query grid = the decoder's
  own cross-attention read-out (which is bilinear in latents per query point — its per-point map
  decode(q) = Attn(q, latents) is linear in the VALUES but softmax in the keys). Proposed D-d1: 3D is
  last (D6) and its first rung is D-only (emit latents from the trunk, decode with the verbatim
  ShapeVAE decoder as a companion until absorbed).

### d.2 Fitting the closed-form maps
`c15_fit_maps.py`: fp64 ridge `E = argmin ||R E − Z||² + λ||E||²` (R = raw rows [n × d_raw], Z = latent
rows [n × d_lat]) via `fit_ridge_linear(R, Z, ridge, fit_bias=True, validation_*)` — Gram d_raw²: 12,288²
fp64 = 1.2 GB (fine); with 18-tubelet context 221k² = 390 GB (NOT feasible) → context variant uses a
random-feature / PCA-reduced context (PCA of neighbor tubelets to 2,048 dims → Gram 1.6 GB) or kernel
ridge on ≤ 50k rows (Gram n² = 20 GB). `D = argmin ||Z D − R||²` (Gram 96², trivial). Metrics: latent
relL2/cos (E), reconstruction PSNR of `D(E(R))` vs R and vs the VAE's own `decode(encode(R))` (the
ceiling), per patch geometry. Rungs: linear per patch → + context → + one nonlinearity (SiTU on a
random-feature lift, still closed-form) — proposed D-d2: a map is "absorbed" when PSNR(D∘E) ≥ PSNR(VAE
roundtrip) − 3 dB on held-out clips; below that, the trunk keeps a companion VAE for that modality.

### d.3 Where the maps go (composition into Chimera, exact linear algebra)
- Input side: `NativeContinuousMediaBridge.feature_proj.weight` [hidden × d_raw] ←
  `B_in · P_src · Eᵀ` where `P_src` = the source model's own latent→trunk patch projection
  (H3 `video_patch_proj` [5376×96], `audio_patch_proj` [5376×32], ACE `proj_in` [assumed linear part],
  Qwen-Image `img_in` [assumed]) and `B_in` = the C17 cross-width chart (5376 → hidden) fitted in (e);
  `type_feature_scale[type]` stays; the bias of E folds into `type_embed[type]` (`type_embed +=
  B_in P_src b_E`). `position_proj/position_fourier_proj` are Chimera's; H3's RoPE lives inside
  attention and is not absorbed here (H3-3).
- Output side: `UnifiedNativeMediaEmissionHead.out.weight` [d_raw × hidden] ← `Dᵀ · P_out · B_out`
  where `P_out` = H3 `FinalLayer.video_out` [96×5376] / `audio_out` [32×5376] (fp32 island), ACE
  `proj_out`; `B_out` = chart hidden → 5376; D's bias folds into the head's type affine shift
  (`type_affine[type][hidden:]`) only approximately (it sits before `out`) — record the residual.
- Because the flow objective is on RAW patches (Chimera x0 on native features) while H3's is on
  latents, the trunk trained after absorption predicts x0 in RAW space with heads that speak latent
  internally; alternatively switch the media band's `native_feature_dim` to the latent width and keep
  D as a companion decoder — both arms in H3-0 (§f).

---

## (e) C16 — DiT block transplant, ACE-Step's DiT first

Files: `/home/cereal/omni-campaign/h3/c16_ace_capture.py` (lab venv; ACE HF code via
`AutoModel.from_pretrained("/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo", trust_remote_code=
True, dtype=bf16)` exactly as `acestep_wrapper._load_dit_model` does [verified], with the transformers
4.57.3 overlay on PYTHONPATH if 5.12.1 rejects `GradientCheckpointingLayer`/`EncoderDecoderCache`
imports [assumed]), `c16_charts.py`, `c16_transplant.py`, `c16_coreset_block.py`.

### e.1 Block I/O capture (ACE)
- Hooks (`ActivationTap`-style, forward hooks with kwargs) on `model.decoder.layers[i]` (the 24
  `AceStepDiTLayer`s; attribute path [assumed `decoder`] — resolve by `named_modules()` filtering
  `isinstance(m, AceStepDiTLayer)`), capturing per call: `hidden_states` in/out [B, L, 2048],
  `temb` [B, 6·2048] (per-step), `position_embeddings` (cos/sin), `encoder_hidden_states` [B, L_c, 2048],
  masks, and the sampler step σ. Also hook `proj_in` input (context_latents ‖ noisy [B,L,192]) and
  `proj_out` output [B,L,64] and the `norm_out` modulation.
- Driver: the model's own generation path (`AceStepConditionGenerationModel.generate`-style code at
  lines 1788+: `prepare_condition` → `prepare_noise(context_latents, seed)` → step loop with
  `context_latents`, turbo 8 steps [assumed]) fed with (a) `silence_latent.pt` + text/lyric conditions
  from the 1.7B LM path (needs `LLMHandler`) or (b) the simplest legal stimulus: random text/lyric
  hidden states of the right widths (the file's own smoke test at lines 2004-2073 builds
  `text_hidden_states [2,77,1024]`, `lyric [2,123,1024]`, `hidden_states/src_latents [2,seq,64]`) —
  rung 0 uses (b) to get the plumbing bit-exact, rung 1 uses real conditions through the kaola
  wrapper (`AceStepHandler`) with our hooks attached to its `self.model`.
- Store per step per block: `x_in`, `x_out`, `temb`, `sigma`, token positions (25 Hz frame index) →
  `/home/cereal/omni-campaign/h3/ace_blocks/step{s}/L{i:02d}.safetensors` (bf16; per 10 s clip 250
  frames × 2048 × 2 B = 1 MB per tensor — tiny).

### e.2 Residual charts (B_in / B_out)
- Chimera side rows: run the Chimera media trunk (proof trunk or the Wave-F import) on the SAME
  clips represented as native tokens in the ACE-latent band (after §d: feature = the 64-d ACE latent
  per 25 Hz frame, type_id music 4, positions t = frame/25 s), with `native_media_target_kind="x0"`,
  `noise_levels` = the ACE step's σ mapped through Chimera's shift (record the σ→t map), capturing the
  residual at each block boundary (`ActivationTap.depth_values` / `blocks.k.norm1` input rows).
- Pairs: (x_chimera[k](t), x_ace[i](t)) matched by frame index and step. Chart: `B_in: R^{h} →
  R^{2048}` and `B_out: R^{2048} → R^{h}` by ridge (`fit_ridge_linear`, fp64) — when h = 2048 (a
  Qwen-shaped trunk is 5120; K3-shaped 7168; the proof trunk 128) use orthogonal Procrustes only for
  equal widths; otherwise ridge, with the orthonormal-chart variant of `build_residual_chart` (SVD of
  pooled rows) as the control arm. Chart quality columns: relL2, cos, retained energy — diagnostics.
- Proposed D-e1 chart pairing rule: pair Chimera layer k with ACE block i by matched DEPTH FRACTION
  (k/n_chimera ≈ i/24) first; alternative pairings are arms.

### e.3 Arms (the T1 protocol) and the 4× per-block coreset
- replace: Chimera block k → `x + B_out(ACE_i(B_in(x), temb) − B_in(x))` (the block's residual delta,
  charted; norms/gates inside ACE_i kept). insert: the same delta added as an extra block between k
  and k+1. control: a random-init block of identical shape through the same charts. identity: charts
  only, no block. random-chart: ACE_i behind random orthonormal charts.
- temb adapter: ACE's own `TimestepEmbedding` (`time_embed`, 256 → 2048 → 2048 SiLU [verified
  lines 1275-1276], ≈ 4.7M params, counted) driven by t(σ_chimera); `scale_shift_table` per block
  kept verbatim. Cross-attention to the ACE condition encoder is DROPPED in the transplant (Chimera has
  no cross-attention; the text condition is in the trunk prefix) — arm "xattn-off" (zero the cross
  branch) vs "xattn-folded" (its output at capture time ridge-folded into the MLP input as a bias
  field... only as a diagnostic).
- 4× channel coreset per transplanted block BEFORE insertion: MLP 6144 → 1536 by `pivoted_columns` on
  the captured `mlp_norm` output rows (gate/up verbatim rows, down `law_abiding_solve_split`) — the
  exact `narrow_mlp_site` recipe with Qwen3MLP's gate_proj/up_proj/down_proj names; attention: no
  head fold at rung 1 (heads are the expensive lane — LFM2 law), `o_proj`/`q/k/v` verbatim, so the
  block's ratio at rung 1 is MLP-only: 3·6144·2048 = 37.7M → 9.4M (4× on the MLP, 2.1× on the block).
  Rung 2 adds kv-head merge 8 → 4 via `joint_relation_solve` (T7) to reach ≥ 4× on the whole block —
  the ≥ 4× law is per COMPONENT; report both.
- Rungs: 1 block (i = 12), 4 blocks (i ∈ {4, 10, 16, 22}), 16 blocks; at 4× and 8× MLP. Measurement:
  the Chimera media objective (x0 MSE on held-out clips) parent vs child vs control, decoded WAVs via
  the ACE VAE decoder (D or verbatim), and Cyril's blind 1–5 — the only gate.

### e.4 H3 blocks (after ACE): identical harness with `DiTBlock.forward(x, t_emb, mod_segments,
rope_freqs)` hooks inside the comfy container (`model.model.diffusion_model.blocks[i]`), single-stream
packed sequence (text | video | audio) — pairs split by `PackedLayout` segment; hidden 5376, heads 56,
ffn 14336; the fp32 heads; needs GPU6 (46 GB) for bf16 (40 GB) or the int8-convrot 21 GB file on GPU0
(quant ops [assumed comfy-supported]). `mod_segments` gives per-modality adaLN — the joint-AV
modulation Cyril wants Chimera to inherit (H3-3).

---

## (f) H3-0 — the rung on the 3.19M proof trunk

Base: `chimera_omni/proof/h3_hybrid_av_flow.py` (v4, cc67c49) — `build_model` (k3_aligned_test_config,
native_feature_dim 1536), `make_batch`, `train` (joint AV x0), `evaluate_understanding`,
`evaluate_generation` (16-step x0, `measure_direction/pitch/onsets`), 13 gates, `freeze_before/after.pt`
sha receipts, MP4/WAV artifacts. File: `/home/cereal/omni-campaign/h3/h3_0_proof_rung.py` (imports the
proof's functions; never edits them). GPU: RTX 8000 was the proof's device (2614 s / 4000 steps);
GPU0 works (3090, bf16 not needed — fp32 3.19M).

- H3-0a (C15 on the proof): fixture scaled to the H3 VAE's grid (SIZE 64, FRAMES 17 → latents 4×4×5 =
  80 latent cells → 20 DiT tokens of 96; audio 12,288 samples @32 kHz → 15.4 → 16 latent frames × 2 ch
  of 32). Arms: (1) raw-patch trunk (the v4 baseline, re-run, same seed); (2) latent-band trunk:
  `native_feature_dim` = 96 (audio latents padded 32 → 96 via `_fit_dim`), features = H3 VAE latents
  (posterior mean) from the container, targets = latents, decode for scoring with the verbatim H3 VAE
  decoders; (3) absorbed trunk: raw features in, bridge `feature_proj` initialized as `P·Eᵀ` (E from
  §d fitted on the fixture), emission `out` initialized as `Dᵀ·P_out` (P, P_out random-orthonormal
  stand-ins at this scale, or the H3 patch projections charted down 5376 → 128 by SVD), trained
  identically; (4) blank-init control. Score: the 13 gates + `video_mse/audio_mse` + the same MP4/WAV
  artifacts for Cyril; proposed D-f1: arm (3) is "absorption carries" if it passes ≥ as many gates as
  (1) at equal steps and its decoded clips are not blind-scored worse.
- H3-0b (C16 on the proof): transplant ONE block into the proof trunk's gated_mla layer 3 (the
  bidirectional refinement block) — first an ACE block (2048; charts 128 ↔ 2048 by ridge on pairs from
  the fixture's audio tokens through ACE with hooks) then an H3 `DiTBlock` (5376; requires the H3 run
  on the fixture latents inside the container — 20 video + 32 audio + text tokens, cheap per forward;
  model residency 40 GB → GPU6 claim or int8 on GPU0). Arms replace / insert / control / identity /
  random-chart; the transplanted block's MLP coreset at 4× (ACE 6144 → 1536; H3 14336 → 3584) on the
  fixture rows; train the rest of the trunk (the transplant frozen: its weights are the fold, the trunk
  is 3.19M) and re-score the 13 gates. Honest scale note: one 4×-folded ACE block (≈ 26M) or H3 block
  (≈ 170M) is 8–50× the trunk — H3-0b tests the chart + arm + coreset MACHINERY (bit-exact charts,
  arms ordering, receipts), not the transplant's value; the size-appropriate first real target is ACE
  blocks into a ≥ 2048-wide trunk (the Wave-F import), which is (e).
- Receipts: every arm writes `report.json` in the proof's schema plus `state_sha256` before/after,
  chart sha256s, installed-tensor sha256s, param counts (unique), and the arm table; artifacts hashed.

---

## Open questions / receipts owed before code runs
1. A_log axis (disk [128] vs per-head [96]) — exactness receipt decides (§a.3).
2. `v_head_dim` of the MLA layers (kv_b_proj rows) — read layer 3's header.
3. transformers 5.12.1 importability of `modeling_kimi_linear.py` and of ACE's `modeling_acestep_v15_turbo.py`
   (else the 4.57.3 overlay).
4. `tiktoken` in the lab venv for K3's `encoding_k3.py`.
5. Share read speed after the Qwen download (prices K3-1); the running probe's receipt.
6. GPU6 claim (H3 bf16 residency) — D14; H3 int8-convrot loadability in our own container.
7. comfy `VAE.encode/decode` calling conventions for video/audio (use `first_stage_model` directly).
8. Whether `pydantic` (installed by compressed-tensors) is acceptable as a vendor transitive dep — our
   code never imports it.

---

## ADDENDUM A (second R6 pass, 2026-09-01 ~17:40) — receipts landed, state changes, corrections

Everything below was read this pass (four bounded `ssh_client` calls: K3 config + venv + custom-node
listing; compressed_tensors sources; `docker run --rm` on the H3 runtime IMAGE; a second image read).

### A.1 Open questions closed
- **Q5 (share speed / probe receipt) — CLOSED.** `k3_layer1_cpu.log` finished at 15:24: `read 16.99 GB in
  3207 s = 5.3 MB/s` (shard 2 only), `dequantized 896 experts in 245 s` (CPU), `MoE forward on 256 random
  tokens: 2.48 s on cpu; 192 of 896 experts touched`, receipt `/home/cereal/omni-campaign/k3/
  k3_layer1_probe_cpu.json` (940 B). Dequant sanity: expert0 w1 std 0.0247 vs shared gate_proj std 0.0188
  (same order — a wrong nibble/scale convention would not give this). Price now: 53 min READ per layer
  from the share → 92 MoE layers ≈ 81 h read-bound; compute per layer is minutes. Re-measure the share
  after the Qwen download finishes before scheduling the full pass.
- **Q2 (`v_head_dim`) — CLOSED: 128** [verified this pass from the full `config.json` read:
  `"v_head_dim": 128`, `"qk_nope_head_dim": 128`, `"qk_rope_head_dim": 64`, `"q_lora_rank": 1536`,
  `"kv_lora_rank": 512`, `"num_key_value_heads": 96`]. So `kv_b_proj` = [96·(128+128) = 24576, 512],
  `q_b_proj` = [96·192 = 18432, 1536], `kv_a_proj_with_mqa` = [576, 7168], `o_proj` = [7168, 12288],
  `g_proj` = [12288, 7168] (`mla_use_output_gate` true). Per-MLA-layer attention params ≈ 232M.
- **MXFP4 nibble order — VERIFIED at source**: `compressed_tensors/compressors/nvfp4/helpers.py:160-182`
  `unpack_fp4_from_uint8(a, m, n, dtype=bf16)`: `high = (a_flat & 0xF0) >> 4; low = a_flat & 0x0F;
  combined = torch.stack((low, high), dim=1).flatten()` → low nibble = even column. `pack_fp4_to_uint8`
  at helpers.py:109 (Triton variant :91). `compressors/mx_utils.py:33 decompress_mx_scale`:
  `scale_exp = scale.int32 − 127` → `2**scale_exp` (bias 127). `compressors/mxfp4/base.py::
  MXFP4PackedCompressor(NVFP4PackedCompressor)` param names `("weight_packed", "weight_scale")`,
  `can_compress` = num_bits 4, type float, group_size 32. The receipt's cross-check imports are exactly:
  `from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8` and
  `from compressed_tensors.compressors.mx_utils import decompress_mx_scale`.
- Lab venv [verified this pass]: compressed_tensors 0.18.0, fla 0.5.2 (fla-core + flash-linear-attention),
  triton 3.6.0, torch 2.11.0+cu128, transformers 5.12.1, einops 0.8.2, safetensors 0.8.0; `flash_attn`
  MISSING (so `cfg._attn_implementation = "eager"` or `"sdpa"` for `KimiMLAAttention`; never build
  `KimiLinearModel`, whose `__init__` forces flash_attention_2 [verified k3_linear.py:1110-1117]).
  torch warns every P40 (sm61) is unsupported by this build — P40s only via the Pascal venv.

### A.2 Server state CHANGED since §0.3 (nvidia-smi + docker ps at ~17:10)
- GPU0 RTX 3090: **21,550 MiB used** by container `gpu0_256k` (`ghcr.io/ggml-org/llama.cpp:server-cuda`,
  up 2 h, "unhealthy") — the 256k envelope test moved onto GPU0; GPU1 P40 21,381 MiB used by `p40_256k`
  (up 2 min). **GPU4 RTX 3090 is FREE (15 MiB)** — no `comfyui_h3_gpu4` container is running any more
  (docker ps shows NO comfy container at all). GPU6 RTX 8000 idle (5 MiB). GPU2/3/5 P40 idle.
  → K3-1 GPU path: pin by PCI bus id (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) and preflight-assert
  `torch.cuda.mem_get_info()` ≥ 22 GiB free on the chosen 3090 (the `seal_and_review.sh` preflight
  pattern); today that is GPU4 (reserved to neural-forge in `comfy-gpu-owners.json` [inventory]), not GPU0.
  Never stop `gpu0_256k`/`p40_256k` (not ours).
- Because no comfy container runs, §(d)/(e) container work uses `docker run --rm` on the IMAGE (verified
  working this pass: `docker run --rm --entrypoint sh comfyui:codex-minimax-h3-runtime-env-r5-20260808
  -c "..."` reads `/opt/ComfyUI/comfy/...`). Add `--gpus '"device=<bus>"'` and the two `-v` mounts from
  §(d) for real runs. Images present: `codex-minimax-h3-runtime-env-r3/r4/r5-20260808` (39.3 GB each),
  `codex-minimax-h3-v0.30.0-torch26(-r2)-20260804`, `codex-3d-hunyuan-paint-skintoken-torch26-20260618`
  (+restored variants), `codex-3d-hunyuan-skintoken-20260618`, `codex-extended-ltx-ace-20260522`.
- Local media files re-listed [verified, bytes]: `vae/minimax_h3_video_vae_fp16.safetensors`
  5,207,808,496; `vae/minimax_h3_audio_vae_fp32.safetensors` 605,254,808; `vae/ace_1.5_vae.safetensors`
  337,431,732 (+ `vae/vae/diffusion_pytorch_model.safetensors` 337,431,388 = the diffusers copy);
  `vae/qwen_image_vae.safetensors` 253,806,246; `vae/Hunyuan3D-vae-v2-1-fp16.ckpt` 655,648,152;
  `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` 40,225,724,176; `..._int8_convrot`
  20,970,379,616 (fl2va and ref2va); `qwen_image_fp8_e4m3fn.safetensors` 20,430,635,136;
  `acestep_v1.5_turbo.safetensors` 4,787,825,604 (= `acestep-v15-turbo/model.safetensors`);
  `checkpoints/hunyuan_3d_v2.1.safetensors` 7,365,943,290 (symlinked as `diffusion_models/
  hunyuan3d-dit-v2-1-fp16.{ckpt,safetensors}`); `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`
  51,506,295,256, `..._nvfp4_awq` 15,687,142,551, `qwen_1.7b_ace15` 3,708,523,360, `qwen_0.6b_ace15`
  1,191,588,248, `qwen_2.5_vl_7b_fp8_scaled` 9,384,670,680; `loras/h3/minimax_h3_turbo_v4_step600_ema.
  safetensors` 779,849,816. LTX files are present but OUT (D3).

### A.3 Comfy-core class facts re-verified from the image (line numbers = `/opt/ComfyUI/comfy/ldm/...`)
- `minimax/model.py`: `DiTBlock.__init__(hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu,
  adaln_dtype, ...)` (line 255) = `norm1`/`norm2` RMSNorm + `attn` + `mlp` + `adaln_proj = AdalnProj(t_dim,
  hidden, 6, 3)` (6 modulations × 3 modality rows); `forward(x, t_emb, mod_segments, rope_freqs)`:
  `h = _mod_scale_shift(norm1(x), shift_msa, scale_msa, mod_segments); x = _mod_gate(x, gate_msa,
  attn(h), mod_segments); h = _mod_scale_shift(norm2(x), ...); return _mod_gate(x, gate_mlp, mlp(h), ...)`.
  `FinalLayer(hidden, t_dim, video_dim, audio_dim, eps)` (275): `video_out`/`audio_out` fp32 Linear
  with bias; `forward(x, t_emb, video_seg, audio_seg)` with `(start, stop, timestep_row)` segments.
  `MiniMaxH3Model.__init__` defaults (412): hidden 5376, num_layers 50, token_refiner 2, heads 56 ×
  128, ffn 14336, latents_dim 24, audio_latents_dim 32, patch (1,2,2) → `video_patch_dim` 96, text_dim
  5120, time_embed_dim 2688, `sigma_shift_video` 12.0 / `sigma_shift_audio` 3.0, `adaln_curve_grid`
  (curve-form checkpoints replace `time_embedder` + full-width adaln with `adaln_t_table` [grid, 2688]);
  `video_patch_proj` (96→5376, fp32, bias), `audio_patch_proj` (32→5376), `condition_proj` (5120→5376),
  `rope.inv_freq` [16] → `rope_freqs(position_ids [S,3])` → [S,96] (t,h,w axes × 16 each, doubled).
  Depth of the PRUNED 20.11B checkpoint is derived from the state dict [assumed] — read
  `blocks.N` max index from the safetensors header before sizing GPU residency.
- `minimax/vae.py::MiniMaxH3VideoVAE` (323): `encode(x)` 654, `decode(z)` 680, `encode_temporal` 522,
  `decode_temporal` 578, `encode_tiled`/`decode_tiled` 673/677; `EncoderFCN3D` (113, causal Conv3d) and
  `ViT3DDecoder` (260). `minimax/audio_vae.py::MiniMaxH3AudioVAE` (373): `decode(z)` 415, `encode(
  waveform)` 426; DAC-style `Encoder` (194) + `BigVGAN` (307) decoder. `sd.py` VAE detection keys:
  video `decoder.transformer_blocks.0.scale1` + `encoder.down.5.block.0.conv1.weight` (line 942); audio
  `pre_block.attn.zero_k_bias` (965); Hunyuan3D `ShapeVAE()` (820); Oobleck `AudioOobleckVAE(**config)`
  (665). `supported_models.py:959 MiniMaxH3`: `latent_format = MiniMaxH3AV`, `sampling_settings shift
  12.0`, `memory_usage_factor 0.114`, dtypes bf16/fp32, conditioner detected at
  `text_encoders.qwen3vl_32b.transformer.` (Qwen3-VL-32B truncated to 50 layers, sd.py:1568).
  `latent_formats.py`: `MiniMaxH3Video` (570) 24 ch, `latent_dimensions 3`, spatial 16, temporal 4,
  scale 1.0 (+ `latent_rgb_factors`); `MiniMaxH3AV` (605) 32 ch; `ACEAudio15` (955) 64 ch, 1-D,
  `temporal_downscale_ratio 1764` (= 44,100/25 — the comfy ACE-1.5 path runs 44.1 kHz; the diffusers
  Oobleck config's 48 kHz/1920 from §0.3 is the HF path — both are 25 latent frames/s; receipt the rate
  the loader you use actually applies); `Hunyuan3Dv2_1` (937) 64 ch, 1-D, scale 1.0039506158752403.
- `ace/ace_step15.py`: `AceStepDiTLayer.__init__(hidden_size, num_heads, num_kv_heads, head_dim,
  intermediate_size, rms_norm_eps, layer_type ∈ {full_attention, sliding_attention}, sliding_window=128)`
  (249): `self_attn_norm` → `self_attn` (`AceStepAttention`, GQA, sliding or full) → `cross_attn_norm` →
  `cross_attn` → `mlp_norm` → `mlp` (`MLP(hidden, intermediate)`), `scale_shift_table` [1, 6, hidden]
  adaLN; `AceStepDiTModel.__init__(in_channels, hidden_size, num_layers, num_heads, num_kv_heads,
  head_dim, intermediate_size, patch_size, audio_acoustic_hidden_dim, condition_dim, layer_types,
  sliding_window, rms_norm_eps)` (599): `rotary_emb` (base 1e6), `proj_in = Conv1d(in_channels →
  hidden, kernel = stride = patch_size)`, `time_embed` and `time_embed_r` (`TimestepEmbedding(256,
  hidden)`); `AceStepConditionGenerationModel.forward(x, timestep, context, lyric_embed, refer_audio,
  audio_codes, is_covers, replace_with_null_embeds)` (1009/1114); `AceStepAudioTokenizer` (911) with
  `FSQ`/`ResidualFSQ`, `AudioTokenDetokenizer` (965) — the 5 Hz code path for the 1.7B LM (C18).
  `ace/vae/` = ACE-1.0 `MusicDCAE` (DCAE + log-mel + vocoder) — NOT the 1.5 VAE; 1.5 uses Oobleck.
- `qwen_image/model.py::QwenImageTransformer2DModel.__init__` defaults (341): patch_size 2, in_channels
  64, out_channels 16, num_layers 60, heads 24 × 128 (inner 3072), `joint_attention_dim` 3584 (Qwen2.5-VL
  7B), pooled 768, rope axes (16,56,56); `QwenImageTransformerBlock(dim, heads, head_dim)` (206) =
  MMDiT double stream: `img_mod`/`txt_mod` (SiLU → Linear dim→6·dim), `img_norm1/2`, `txt_norm1/2`
  (LayerNorm no affine), `img_mlp`/`txt_mlp` (GELU FeedForward), joint `attn`. Latent 16 ch, 8× (Wan
  VAE) → DiT token = 2×2 latent = 64 dims ↔ 16×16×3 = 768 raw values (Chimera image `patch=16`).
- `hunyuan3d/vae.py::ShapeVAE.__init__` defaults (899): `num_latents 4096`, `embed_dim 64`, `width
  1024`, `heads 16`, `num_decoder_layers 16`, `num_encoder_layers 8`, `pc_size 81920`,
  `pc_sharpedge_size 0`, `point_feats 4`, `downsample_ratio 20` (**4096 = 81920/20 — the latents are
  the FPS-downsampled surface points used as cross-attention queries in `PointCrossAttention`, so each
  latent DOES carry an anchor position** [assumed from the constructor arithmetic; verify in
  `PointCrossAttention.forward(point_cloud, features)` line 199 that the query set is the subsampled
  points] — this gives §d.1 a LOCAL pairing for 3D: latent_j ↔ the voxel-SDF patch around anchor j, a
  per-latent ridge like the other modalities, instead of the whole-object PCA map), `num_freqs 8`
  Fourier embedder, `scale_factor 1.0039506158752403`; `decode(latents)` (966): `post_kl` (64→1024) →
  `transformer` (16 layers) → `volume_decoder(latents, geo_decoder, bounds 1.01, num_chunks 8000,
  octree_resolution 256)` → grid logits; `encode(surface)` (978): `surface[:, :, :3]` points +
  `[:, :, 3:]` feats → `encoder` → `pre_kl` → `DiagonalGaussianDistribution.sample()`.
- Custom nodes [re-verified file lists]: `ComfyUI-MiniMax-H3-Turbo/__init__.py` only (classes
  `MiniMaxH3TurboSampler` :242, `_FrugalLoRA` :261, `MiniMaxH3TurboLoRA` :440; `h3_silu_temb_grid.
  safetensors` :181). `ComfyUI_RH_ACE-Step/nodes.py` (`RunningHub_ACEStep_Loader` :9, `_GenerationParams`
  :45, `_Creator` :81, `_Artist` :131) + `acestep/{handler.py (AceStepHandler :56, load_lora :139,
  AutoModel.from_pretrained :406/:417, AutoencoderOobleck.from_pretrained(<ckpt>/vae) :483,
  vae.encode(...).latent_dist.sample() :2555, vae.decode :2374), llm_inference.py (LLMHandler :26,
  AutoModelForCausalLM.from_pretrained :224), acestep_v15_pipeline.py, dit_alignment_score.py,
  inference.py, api_server.py, third_parts/nano-vllm}`. `ComfyUI-kaola-ace-step/{acestep_wrapper.py
  (ACEStepWrapper :13, _load_vae :230 → AutoencoderOobleck.from_pretrained(path, low_cpu_mem_usage=False)
  :242, model AutoModel.from_pretrained :151/:165, silence_latent torch.load :226), nodes.py
  (ACE_STEP_BASE :422, TEXT_TO_MUSIC :629, COVER :782, REPAINT :953, SIMPLE_MODE :1101, FORMAT_SAMPLE
  :1260, CREATE_SAMPLE :1335, UNDERSTAND :1405, LORA_LOADER :1537), nodes_base_features.py, nodes_vram.py,
  nodes_captioner.py, nodes_transcriber.py}`. `ComfyUI-Hunyuan3d-2-1/{__init__.py, nodes.py,
  hy3dshape/hy3dshape/{pipelines.py (Hunyuan3DDiTPipeline :137, from_pretrained :217, vae =
  instantiate_from_config(config['vae']) :192, Hunyuan3DDiTFlowMatchingPipeline :711),
  models/autoencoders/model.py (VectsetVAE :118, ShapeVAE :238, from_pretrained :157),
  models/autoencoders/{attention_blocks.py, volume_decoders.py (VanillaVolumeDecoder :141,
  HierarchicalVolumeDecoding :185, FlashVDMVolumeDecoding :280), surface_extractors.py (MC :101, DMC
  :127)}, models/denoisers/{hunyuan3ddit.py (Hunyuan3DDiT :284, DoubleStreamBlock :155,
  SingleStreamBlock :220), hunyuandit.py (HunYuanDiTPlain :465, HunYuanDiTBlock :301), moe_layers.py
  (MoEGate :43, MoEBlock :112)}, models/conditioner.py (DinoImageEncoder :127, DualImageEncoder :214)},
  hy3dshape/{minimal_demo.py, minimal_vae_demo.py}, hy3dpaint/{textureGenPipeline.py, demo.py,
  train.py, convert_utils.py}}`.

### A.4 Corrections / sharpenings to the body above
- §0.1 A_log: the modeling code declares `A_log` over `num_heads` (96) [verified k3_linear.py:520-521];
  the header read in the first pass reported [128]. Unresolved — the receipt (§a.6 step 4) decides;
  add to the receipt a `load_state_dict(strict=True)` attempt on `KimiDeltaAttention` built from
  `KimiLinearConfig(**text_config)`: a shape error there is the fastest disambiguation.
- §0.1 layer schedule: independent confirmation from Chimera — `k3_aligned_test_config` uses pattern
  (kda,kda,kda,gated_mla) with `force_final_gated_mla=True` [verified config.py:234,242], i.e. Codex
  aligned to the 1-based reading (0-based MLA at i = 4k−1 and i = 92).
- §a.2 / §a.6: the probe's `dequant_mxfp4` already matches the verified nibble order; keep the
  cross-check receipt anyway (one call, bit-exact expected).
- §a.5 sizes at T = 65,536 [arithmetic, this pass]: `moe_x` 0.94 GB, `latent_in` 0.47, `y_lat` fp32
  0.94, router idx int16 2 MB + w bf16 2 MB → the recommended minimal set is 2.35 GB/layer → 216 GB for
  92 layers (fits 615 GB with the ~100 GB body staging). `block_residual` in RAM at layer 84 = 8 ×
  0.94 GB = 7.5 GB (fine); restart snapshots at layers 0,12,…,84 total ≈ 34 GB on disk.
- §a.3 chunked KDA over long sequences: `KimiDeltaAttention.forward` with `cache_params=KimiDynamicCache`
  carries `conv_states` (q,k,v) and `recurrent_states` between chunks and stays in `chunk` mode for
  q_len > 1 [verified k3_linear.py:559-561, 574-578, 646-649] — so one ≥32k sequence can be run in
  8k-token chunks on a 24 GB card with exact state carry; fresh cache per sequence.
- §(b) Gram/RAM plan, exact numbers: candidate union |U| = 896·3072 = 2,752,512; stage-1 per-expert
  Gram 3072² fp64 = 75.5 MB (rows per expert ≈ T·16/896: 1,170 at T=65,536, 4,680 at 262,144 — prefer
  ≥131k tokens for the three study layers so stage-1 Grams are not rank-starved); stage-2 union Gram
  fp64: K_pool 34,816 → 9.7 GB; 65,536 → 34.4 GB; 131,072 → 137 GB (fp32 69 GB — only this last one
  strains 251 GiB; pick K_pool = 2K for the ≤32,768 rungs and K_pool = K at 65,536, as §b.3 says).
  Refit Gram `AᵀA` at K = 65,536 fp64 = 34.4 GB + `AᵀY` 1.9 GB; Cholesky 65,536³/3 ≈ 9.4e13 flop ≈
  10–20 min on 88 CPUs. Accumulate both in token blocks of 8k on GPU (features) → CPU fp64 (Grams).
- §(b) storage table stands (routed dense 3·K·3584: 704.6M / 352.3M / 187.2M / 93.6M = 42.0× / 84.0× /
  158.1× / 316.2× on the 29,595,009,024-param expert mass; router 6.4M + bias deleted).
- §(c): dense-K3 RAM residency [arithmetic]: K = 17,408 → per MoE layer 370.7M + attention (KDA ≈ 444M,
  MLA ≈ 232M) → 92·370.7M + 69·444M + 24·232M + layer-0 mlp 0.73B + embed/head 2·1.17B ≈ 74–80B →
  148–160 GB bf16: fits RAM (251 GiB) for CPU free-run; K = 32,768 ≈ 95B → 190 GB: tight; K = 65,536 ≈
  128B → 256 GB: does NOT fit — first R-full at K ≤ 32,768 (D-c1 as stated). CPU decode is memory-bound
  (~150 GB per token / ~50 GB/s ≈ 3 s/token [assumed bandwidth]) → 45 prompts × 256 tokens ≈ 10 h.
- §(d) H3 geometry: one H3 DiT video token (2×2 latent patch, 96 numbers) ↔ raw 32×32 px × 4 frames × 3
  = 12,288 values = Chimera `video_to_native_patches(patch=32, tubelet=4)` → `native_feature_dim`
  12,288 (bridge `feature_proj` 12,288×hidden = 63M at 5120 — acceptable). Audio: 800 samples/frame at
  32 kHz per §0.3 (40 Hz) → Chimera `segment=800` per channel; ACE: 1764 (44.1k) or 1920 (48k) per
  frame → `segment` = the loader's hop; stereo = 2 tokens per latent frame (Chimera emits one token
  per channel per window [verified native_media.py:360-361]) — the map stacks both channel tokens.
- §(f) proof geometry re-verified [h3_hybrid_av_flow.py:41-54]: FRAMES 8, FPS 8, SIZE 32, PATCH 16,
  TUBELET 2, SAMPLES 12,288 @ 32 kHz stereo, SEGMENT 1536, FEATURE_DIM = 2·3·16·16 = 1536,
  VIDEO_TOKENS = 4·4 = 16, AUDIO_TOKENS = 8·2 = 16, MEDIA_TOKENS 32 (the body's "32 video + 16 audio" is
  a slip — it is 16 + 16), VIDEO_SHIFT 12, AUDIO_SHIFT 3, TOK_MEDIA_PLACEHOLDER 3264, generation
  `input_ids` = 4 condition tokens + 32 placeholders; proof trunk checkpoint =
  `/home/cereal/chimera-proof-c5a5c18-v4/proof_outputs/h3_hybrid_av_flow_v4/freeze_after.pt` (torch.save
  dict `{"source_commit", "state_dict"}`, 12.78 MB, sha in report.json) [inventory + save_checkpoint].
- Rolling local staging of expert shards would require deleting OUR staged copies (estate rule: never
  delete on the server) → keep §a.6's "stream experts from the share, stage only the ~100 GB body"; if
  Cyril wants staging, that is decision line D-a3 (explicit permission to delete our own staged shards
  under `/home/cereal/omni-campaign/k3/stage/`).
