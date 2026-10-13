# Existing audio source audit and transcript-cue repair, 2026-10-13

The complete pinned 91984-row normalized registry includes 1763 OpenSLR-source
speech rows: 884 ASR rows and 879 paired generation rows. These are existing
inventory entries. They are not 1763 independent waveform examples or admitted
training examples. Their recorded 2026 collection date explicitly does not
establish the underlying source's original date.

`natural_speech.py` preserves original row hashes, registry shard/RID, license,
speaker/chapter annotations, source groups, split and waveform hashes. It replaces
the exact expected-transcript cue in supported ASR prompts with an observation
instruction and uses the original annotated transcript for assistant supervision.
The original answer object remains in provenance. Generation rows remain in the
complete inventory and receive no ASR rewrite. Every derivative is blocked from
training until source-date evidence, actual waveform/transcript verification,
complete global admission and authentic native encoder evidence are supplied.

Run `python -m experiments.chimera_recovery.natural_speech --source SOURCE.jsonl
--source-sha256 SHA256 --out FRESH_DIRECTORY`. Output includes blocked repaired
rows, every original speech inventory entry, quarantine and a sealed audit.
Audio inventory counting recognizes the original `type`, `kind` and `media`
aliases, including degraded-input/clean-target restoration pairs. Counting such
pairs does not convert them into speech-understanding observations.

The existing LAION Orpheus-tuning source is explicitly synthetic voice acting in
its [primary dataset card](https://huggingface.co/datasets/laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning).
Its [March 2025 source commit](https://huggingface.co/datasets/laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning/commit/4b450369600f5f05d877b4b6d45227c37777c9e2)
does not make those waveforms natural recorded speech. The
[SonicMaster source card](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset)
describes music restoration/mastering pairs and links its 2025 paper; those pairs
need their own observation/target contract and original-asset provenance.
Neither source closes the natural ASR gap merely by appearing in inventory.
