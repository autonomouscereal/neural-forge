# Procedural media fixtures

`media_seed.py` creates 2026 operator-authored assets on the CPU using the
existing Pillow, numpy, ffmpeg, and ffprobe runtime. It does not load a model,
download media, manufacture latent tensors, or claim native model performance.

```powershell
python -m unittest discover -s tests -p test_chimera_media_seed.py -v
python experiments/chimera_recovery/media_seed.py --examples 144 --exclude-prompts actual_frozen_prompts.json --out new_media_tranche
```

The default 144 fixtures span 18 exclusive scenario families and six routes:
`image_edit`, `joint_av`, `volume_generate`, `audio_understand`,
`vision_grounding`, and `video_understand`. The generator refuses existing output
directories. Each file carries its actual SHA-256, size, and semantic role.
Groundtruth manifests carry a separate file hash and generator source hash.
Asset identities cannot cross splits.

Image edits preserve separate before/after roles and verify changed pixels and
an unchanged object. Audio fixtures contain measured non-speech tones, chirps,
and noise bursts; they provide no speech-recognition evidence. Video events and
audio pulses derive from one event timeline, are actually muxed, then decoded
and verified. Joint targets share the muxed source hash and interval. Meshes are
actual OBJ vertices and faces, reparsed and checked for closed manifold edges
and analytic volume, with an orthographic render and coordinate metadata.

Assistant targets contain the requested count, color, position, timing, sound,
and geometry answers. Instrumentation measurements and codec probes stay in
groundtruth/metadata rather than becoming text targets.

`source_ready.jsonl` includes all source fixtures. `understanding_ready.jsonl`
includes only the input-media/text-answer routes. Those are structurally useful
sources; their authentic native input processing remains pending. Generative
rows explicitly retain `train_ready=false` and `native_cache_status=pending`
until actual native teacher/cache provenance exists. No teacher checkpoint hash
is invented to make a corpus gate pass. `report.json` records route/split counts,
asset verification, source hashes, output hashes, and limitations.

These simple scenes supplement missing format coverage. They do not replace
natural-media grounding, speech/ASR, realistic image edits, or general 3D data.
