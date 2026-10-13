# H3 joint capture preparation, 2026-10-13

`h3_cache.py` supplies a strict executable cache-contract preflight, a separate
candidate of the exact Stage-C writer, and a numerical pair validator/packer.
It launches no teacher, GPU job or training. Existing Stage-C v1 cache seals do
not meet the joint contract: they omit actual `x_t`, native audio velocity,
sealed shard sidecars and measured audio/video clock evidence.

The pinned original is
`reports/omni-campaign-2026-09/distill/tools/stage_c_media_a.py`, SHA256
`fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf`.
The candidate preserves teacher computation and adds captures of the actual
`xt[0]`, `xt[1]`, `tgt.v_audio_native`, full video picks/times, decoded audio PTS,
both media hashes, and shard sidecar SHA256 seals. It refuses waveform silence,
padding, audio discontinuities, unmatched PTS origins, or a different MiniMax
`model.py` source from the chart recipe pin
`882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024`.
This is a private candidate for review and a bounded actual capture; its
decoder instrumentation has not been tested against the remote ComfyUI runtime.

The bounded source is the eight **training** joint fixtures in
`proof_outputs/chimera_modality_jobs_oct13_v1/manifest.json`. The preparation
checks the planned item seal and actual local media seals. It assigns both
streams to the identical physical muxed MP4. It does not borrow another WAV.
The requested physical window remains `[0,22/24)` seconds, 22 frames at 24fps,
96px, from the actual 1-second 8fps sources. The full source-to-native frame map
is measured by the writer and compared with the planned map. A decode refusal
is evidence that an alignment implementation is still required, not permission
to remove the check. Raw audio waveform padding is refused; latent padding is
retained as the Stage-C VAE convention and recorded separately by that writer.

Run preparation locally from this source checkout (PowerShell):

```powershell
C:/Python314/python.exe -B -m unittest discover -s tests -p test_chimera_h3_cache.py
C:/Python314/python.exe -B experiments/chimera_recovery/h3_cache.py patch-writer --source 'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/tools/stage_c_media_a.py' --out 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_h3_capture_preparation_oct13_v1/stage_c_media_a_h3_joint_candidate.py'
C:/Python314/python.exe -B experiments/chimera_recovery/h3_cache.py prepare-items --planner-manifest 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_modality_jobs_oct13_v1/manifest.json' --out 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_h3_capture_preparation_oct13_v1/h3_items.same_mux.train.jsonl'
```

The two preparation commands refuse existing output files. The writer command
also emits `.patch` and `.receipt.json`; the items command emits its receipt.
Root must run the four numerical Torch tests in its supported CPU Torch runtime
as well as the standard-library tests. Numerical fixtures are explicitly
synthetic contract checks and never serve as native teacher training data.
On the actual Linux runtime set `CHIMERA_H3_CAMPAIGN_SOURCE=/home/cereal/omni-campaign`
so the source/chart reference tests verify the authentic server mirror rather
than skip because the workstation's Windows path is absent.

For the future bounded capture, root's runtime owner must transfer the new
candidate and strict items using server-manager, recheck remote file SHA256s
and actual teacher identities, and retain the same eight muxed assets. Adapt
the three argv arrays in the original modality job manifest as follows:

1. Replace argv element 2 with the uploaded candidate path.
2. Set `--items` to the uploaded strict item file. Set `--cache-tag` to
   `recovery-same-mux-20261013` and use a new cache root.
3. Compute `sha16 = SHA256(strict item bytes)[:16]`. For the velocity job set
   `--encode-cache` to `CACHE_ROOT/h3-vae-recovery-same-mux-20261013/sha16`
   and `--te-cache` to
   `CACHE_ROOT/h3-te-qwen3vl_32b_minimax_h3_nvfp4_awq-recovery-same-mux-20261013/sha16`.
4. Keep the physical encode argv `--width 96 --height 96 --length 22
   --start-seconds 0 --max-drop-fraction 0`, full teacher hashes, CFG 1,
   four shifted sigmas per item, and `--parity-every 1`.

The original item bytes and cache namespace must not be reused: same-mux
preparation changes the corpus SHA256. The text and encode cache dependencies
must both use the new strict manifest. The thermal gate is removed by the human;
exclusive runtime ownership and actual source/teacher file seals still apply.

After the actual capture is sealed, run a **body-free tensor-header and metadata
preflight**, which reads only the specified teacher cache:

```text
python -B experiments/chimera_recovery/h3_cache.py preflight --cache ACTUAL_H3_VELOCITY_CACHE --writer-sha256 REVIEWED_CANDIDATE_SHA256 --out NEW_PREFLIGHT_RECEIPT.json
```

Passing `capture_contract_ok` still gives `native_train_ready:false`. Required
production binding is deliberately separate:

- Use `validate_pair(tensors, target_sidecar_row, row_index, cfg)` for **every**
  target. Join `target_item` to `item.index`. Rebuild noise with a CPU generator,
  seed and full batch-containing shapes, video then audio; compare the sealed
  noise digest and bit-exact actual noisy inputs. Use the exact Python sigma
  from the sidecar, not the rounded tensor scalar.
- The canonical video target is `v_video`; audio is `v_audio/slope_audio` in
  float64 then float32. Audio rows use the shifted audio sigma (12→3), not the
  video sigma. Both native parent raw signs are +1, so canonical velocity is
  **negative** native raw output. CFG branch equality uses bounded tolerance
  because guidance is performed on denoised values in float32.
- Call `pack_pair` only with authentic native chat prefix IDs, the genuine
  ledger's media token ID, and separately measured/pinned
  `native_media_positions[1,M,4]`. This helper refuses to invent bridge clock
  coordinates from the old pair bank's assumed four-frame stride.
- The adapter places video noisy rows in **384:480**, audio in **480:512**;
  emission targets are video **128:224**, audio **224:256**. It returns both
  `native_media_loss_weights` and the mask alias, positive on both streams,
  and explicit target-channel supervision. Clean latents are never inputs.
- Native ordering is `text | audio | video`. The complete
  `charted_source_positions[1,S,3]` includes the text prefix, both stereo audio
  streams, and video patches. Its exact float64→float32 recipe matches the
  authenticated 2026 `s8_media_smoke.h3_source_positions`; these RoPE coordinates
  are distinct from physical timestamps.
- The production driver must seal native batches and bind source group/split,
  exact corpus SHA, writer/teacher/parent identities, frozen exclusions, and
  `chimera_recovery.native_flow_cache.v1` with `target_kind:canonical_velocity`,
  `prediction_kind:raw`, explicit feature/emission layouts, noisy suballocations,
  parent signs `{2:+1,3:+1}`, and the entire measured AV alignment contract.

No actual H3 teacher cache, native training admission, trained recovery
checkpoint, speech transcription coverage or performance result is asserted by
these preparation files.
