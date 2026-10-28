---
language_creators:
  - creator_1
language:
  - ach
  - aka
  - amh
  # - bam
  - bau
  - dag
  - dga
  - ewe
  - fat
  # - fuf
  - ful
  - hau
  - ibo
  - kik
  - kpo
  - lin
  - lug
  - luo
  - mas
  - mlg
  - nyn
  - orm
  - pcm
  - sid
  - sna
  - sog
  - swa
  - tir
  - twi
  - wal
  # - wol
  - yor
license:
  - cc-by-sa-4.0
  - cc-by-4.0
multilinguality:
  - multilingual
source_datasets:
  - UGSpeechData
  - DigitalUmuganda/AfriVoice
  - original
task_categories:
  - automatic-speech-recognition
  - text-to-speech
pretty_name: Waxal NLP Datasets
arxiv: 2602.02734
annotation_creators:
  - human-annotated
  - crowdsourced
tags:
  - audio
  - automatic-speech-recognition
  - text-to-speech
configs:
  - config_name: ach_asr
    data_files:
      - split: train
        path: data/ASR/ach/ach-train-*
      - split: validation
        path: data/ASR/ach/ach-validation-*
      - split: test
        path: data/ASR/ach/ach-test-*
      - split: unlabeled
        path: data/ASR/ach/ach-unlabeled-*
  - config_name: ach_tts
    data_files:
      - split: train
        path: data/TTS/ach/ach-train-*
      - split: validation
        path: data/TTS/ach/ach-validation-*
      - split: test
        path: data/TTS/ach/ach-test-*
  - config_name: aka_asr
    data_files:
      - split: train
        path: data/ASR/aka/aka-train-*
      - split: validation
        path: data/ASR/aka/aka-validation-*
      - split: test
        path: data/ASR/aka/aka-test-*
      - split: unlabeled
        path: data/ASR/aka/aka-unlabeled-*
  - config_name: amh_asr
    data_files:
      - split: train
        path: data/ASR/amh/amh-train-*
      - split: validation
        path: data/ASR/amh/amh-validation-*
      - split: test
        path: data/ASR/amh/amh-test-*
      - split: unlabeled
        path: data/ASR/amh/amh-unlabeled-*
  # - config_name: bam_tts
  #   data_files:
  #     - split: train
  #       path: data/TTS/bam/bam-train-*
  #     - split: validation
  #       path: data/TTS/bam/bam-validation-*
  #     - split: test
  #       path: data/TTS/bam/bam-test-*
  - config_name: bau_tts
    data_files:
      - split: train
        path: data/TTS/bau/bau-train-*
      - split: validation
        path: data/TTS/bau/bau-validation-*
      - split: test
        path: data/TTS/bau/bau-test-*
  - config_name: dag_asr
    data_files:
      - split: train
        path: data/ASR/dag/train-*
      - split: validation
        path: data/ASR/dag/validation-*
      - split: test
        path: data/ASR/dag/test-*
      - split: unlabeled
        path: data/ASR/dag/unlabeled-*
  - config_name: dga_asr
    data_files:
      - split: train
        path: data/ASR/dga/train-*
      - split: validation
        path: data/ASR/dga/validation-*
      - split: test
        path: data/ASR/dga/test-*
      - split: unlabeled
        path: data/ASR/dga/unlabeled-*
  - config_name: ewe_asr
    data_files:
      - split: train
        path: data/ASR/ewe/ewe-train-*
      - split: validation
        path: data/ASR/ewe/ewe-validation-*
      - split: test
        path: data/ASR/ewe/ewe-test-*
      - split: unlabeled
        path: data/ASR/ewe/ewe-unlabeled-*
  - config_name: ewe_tts
    data_files:
      - split: train
        path: data/TTS/ewe/ewe-train-*
      - split: validation
        path: data/TTS/ewe/ewe-validation-*
      - split: test
        path: data/TTS/ewe/ewe-test-*
  - config_name: fat_tts
    data_files:
      - split: train
        path: data/TTS/fat/fat-train-*
      - split: validation
        path: data/TTS/fat/fat-validation-*
      - split: test
        path: data/TTS/fat/fat-test-*
  - config_name: ful_asr
    data_files:
      - split: train
        path: data/ASR/ful/ful-train-*
      - split: validation
        path: data/ASR/ful/ful-validation-*
      - split: test
        path: data/ASR/ful/ful-test-*
      - split: unlabeled
        path: data/ASR/ful/ful-unlabeled-*
  # - config_name: fuf_tts
  #   data_files:
  #     - split: train
  #       path: data/TTS/fuf/fuf-train-*
  #     - split: validation
  #       path: data/TTS/fuf/fuf-validation-*
  #     - split: test
  #       path: data/TTS/fuf/fuf-test-*
  - config_name: ful_tts
    data_files:
      - split: train
        path: data/TTS/ful/ful-train-*
      - split: validation
        path: data/TTS/ful/ful-validation-*
      - split: test
        path: data/TTS/ful/ful-test-*
  - config_name: hau_tts
    data_files:
      - split: train
        path: data/TTS/hau/hau-train-*
      - split: validation
        path: data/TTS/hau/hau-validation-*
      - split: test
        path: data/TTS/hau/hau-test-*
  - config_name: ibo_tts
    data_files:
      - split: train
        path: data/TTS/ibo/ibo-train-*
      - split: validation
        path: data/TTS/ibo/ibo-validation-*
      - split: test
        path: data/TTS/ibo/ibo-test-*
  - config_name: kik_tts
    data_files:
      - split: train
        path: data/TTS/kik/kik-train-*
      - split: validation
        path: data/TTS/kik/kik-validation-*
      - split: test
        path: data/TTS/kik/kik-test-*
  - config_name: kpo_asr
    data_files:
      - split: train
        path: data/ASR/kpo/train-*
      - split: validation
        path: data/ASR/kpo/validation-*
      - split: test
        path: data/ASR/kpo/test-*
      - split: unlabeled
        path: data/ASR/kpo/unlabeled-*
  - config_name: lin_asr
    data_files:
      - split: train
        path: data/ASR/lin/lin-train-*
      - split: validation
        path: data/ASR/lin/lin-validation-*
      - split: test
        path: data/ASR/lin/lin-test-*
      - split: unlabeled
        path: data/ASR/lin/lin-unlabeled-*
  - config_name: lug_asr
    data_files:
      - split: train
        path: data/ASR/lug/lug-train-*
      - split: validation
        path: data/ASR/lug/lug-validation-*
      - split: test
        path: data/ASR/lug/lug-test-*
      - split: unlabeled
        path: data/ASR/lug/lug-unlabeled-*
  - config_name: lug_tts
    data_files:
      - split: train
        path: data/TTS/lug/lug-train-*
      - split: validation
        path: data/TTS/lug/lug-validation-*
      - split: test
        path: data/TTS/lug/lug-test-*
  - config_name: luo_tts
    data_files:
      - split: train
        path: data/TTS/luo/luo-train-*
      - split: validation
        path: data/TTS/luo/luo-validation-*
      - split: test
        path: data/TTS/luo/luo-test-*
  - config_name: mas_asr
    data_files:
      - split: train
        path: data/ASR/mas/mas-train-*
      - split: validation
        path: data/ASR/mas/mas-validation-*
      - split: test
        path: data/ASR/mas/mas-test-*
      - split: unlabeled
        path: data/ASR/mas/mas-unlabeled-*
  - config_name: mlg_asr
    data_files:
      - split: train
        path: data/ASR/mlg/mlg-train-*
      - split: validation
        path: data/ASR/mlg/mlg-validation-*
      - split: test
        path: data/ASR/mlg/mlg-test-*
      - split: unlabeled
        path: data/ASR/mlg/mlg-unlabeled-*
  - config_name: nyn_asr
    data_files:
      - split: train
        path: data/ASR/nyn/nyn-train-*
      - split: validation
        path: data/ASR/nyn/nyn-validation-*
      - split: test
        path: data/ASR/nyn/nyn-test-*
      - split: unlabeled
        path: data/ASR/nyn/nyn-unlabeled-*
  - config_name: nyn_tts
    data_files:
      - split: train
        path: data/TTS/nyn/nyn-train-*
      - split: validation
        path: data/TTS/nyn/nyn-validation-*
      - split: test
        path: data/TTS/nyn/nyn-test-*
  - config_name: orm_asr
    data_files:
      - split: train
        path: data/ASR/orm/orm-train-*
      - split: validation
        path: data/ASR/orm/orm-validation-*
      - split: test
        path: data/ASR/orm/orm-test-*
      - split: unlabeled
        path: data/ASR/orm/orm-unlabeled-*
  - config_name: pcm_tts
    data_files:
      - split: train
        path: data/TTS/pcm/train-*
      - split: validation
        path: data/TTS/pcm/validation-*
      - split: test
        path: data/TTS/pcm/test-*
  - config_name: sid_asr
    data_files:
      - split: train
        path: data/ASR/sid/sid-train-*
      - split: validation
        path: data/ASR/sid/sid-validation-*
      - split: test
        path: data/ASR/sid/sid-test-*
      - split: unlabeled
        path: data/ASR/sid/sid-unlabeled-*
  - config_name: sna_asr
    data_files:
      - split: train
        path: data/ASR/sna/sna-train-*
      - split: validation
        path: data/ASR/sna/sna-validation-*
      - split: test
        path: data/ASR/sna/sna-test-*
      - split: unlabeled
        path: data/ASR/sna/sna-unlabeled-*
  - config_name: tir_asr
    data_files:
      - split: train
        path: data/ASR/tir/tir-train-*
      - split: validation
        path: data/ASR/tir/tir-validation-*
      - split: test
        path: data/ASR/tir/tir-test-*
      - split: unlabeled
        path: data/ASR/tir/tir-unlabeled-*
  - config_name: sog_asr
    data_files:
      - split: train
        path: data/ASR/sog/sog-train-*
      - split: validation
        path: data/ASR/sog/sog-validation-*
      - split: test
        path: data/ASR/sog/sog-test-*
      - split: unlabeled
        path: data/ASR/sog/sog-unlabeled-*
  - config_name: swa_tts
    data_files:
      - split: train
        path: data/TTS/swa/swa-train-*
      - split: validation
        path: data/TTS/swa/swa-validation-*
      - split: test
        path: data/TTS/swa/swa-test-*
  - config_name: twi_tts
    data_files:
      - split: train
        path: data/TTS/twi/twi-train-*
      - split: validation
        path: data/TTS/twi/twi-validation-*
      - split: test
        path: data/TTS/twi/twi-test-*
  - config_name: yor_tts
    data_files:
      - split: train
        path: data/TTS/yor/yor-train-*
      - split: validation
        path: data/TTS/yor/yor-validation-*
      - split: test
        path: data/TTS/yor/yor-test-*
  - config_name: wal_asr
    data_files:
      - split: train
        path: data/ASR/wal/wal-train-*
      - split: validation
        path: data/ASR/wal/wal-validation-*
      - split: test
        path: data/ASR/wal/wal-test-*
      - split: unlabeled
        path: data/ASR/wal/wal-unlabeled-*
  # - config_name: wol_tts
  #   data_files:
  #     - split: train
  #       path: data/TTS/wol/wol-train-*
  #     - split: validation
  #       path: data/TTS/wol/wol-validation-*
  #     - split: test
  #       path: data/TTS/wol/wol-test-*
  - config_name: aka_asr_v2
    data_files:
      - split: train
        path: data/ASR/aka/aka-train-*
      - split: validation
        path: data/ASR/aka/aka-validation-*
      - split: test
        path: data/ASR/aka/aka-test-*
      - split: unlabeled
        path: data/ASR/aka/aka-unlabeled-*
  - config_name: amh_asr_v2
    data_files:
      - split: train
        path: data/ASR/amh/amh-train-*
      - split: validation
        path: data/ASR/amh/amh-validation-*
      - split: test
        path: data/ASR/amh/amh-test-*
      - split: unlabeled
        path: data/ASR/amh/amh-unlabeled-*
  - config_name: orm_asr_v2
    data_files:
      - split: train
        path: data/ASR/orm/orm-train-*
      - split: validation
        path: data/ASR/orm/orm-validation-*
      - split: test
        path: data/ASR/orm/orm-test-*
      - split: unlabeled
        path: data/ASR/orm/orm-unlabeled-*
  - config_name: sid_asr_v2
    data_files:
      - split: train
        path: data/ASR/sid/sid-train-*
      - split: validation
        path: data/ASR/sid/sid-validation-*
      - split: test
        path: data/ASR/sid/sid-test-*
      - split: unlabeled
        path: data/ASR/sid/sid-unlabeled-*
  - config_name: tir_asr_v2
    data_files:
      - split: train
        path: data/ASR/tir/tir-train-*
      - split: validation
        path: data/ASR/tir/tir-validation-*
      - split: test
        path: data/ASR/tir/tir-test-*
      - split: unlabeled
        path: data/ASR/tir/tir-unlabeled-*
  - config_name: wal_asr_v2
    data_files:
      - split: train
        path: data/ASR/wal/wal-train-*
      - split: validation
        path: data/ASR/wal/wal-validation-*
      - split: test
        path: data/ASR/wal/wal-test-*
      - split: unlabeled
        path: data/ASR/wal/wal-unlabeled-*
dataset_info:
  - config_name: ach_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ach_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: aka_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: amh_asr
    features: 
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  # - config_name: bam_tts
  #   features: 
  #     - name: id
  #       dtype: string
  #     - name: speaker_id
  #       dtype: string
  #     - name: text
  #       dtype: string
  #     - name: locale
  #       dtype: string
  #     - name: gender
  #       dtype: string
  #     - name: audio
  #       dtype: audio
  - config_name: bau_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: dag_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: dga_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ewe_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ewe_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: fat_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ful_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: fuf_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ful_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: hau_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: ibo_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: kik_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: kpo_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: lin_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: lug_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: lug_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: luo_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: mas_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: mlg_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: nyn_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: nyn_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: orm_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: pcm_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: sid_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: sna_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: sog_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: swa_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: tir_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: twi_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: wal_asr
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  # - config_name: wol_tts
  #   features:
  #     - name: id
  #       dtype: string
  #     - name: speaker_id
  #       dtype: string
  #     - name: text
  #       dtype: string
  #     - name: locale
  #       dtype: string
  #     - name: gender
  #       dtype: string
  #     - name: audio
  #       dtype: audio
  - config_name: yor_tts
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: text
        dtype: string
      - name: locale
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: aka_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: amh_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: orm_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: sid_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: tir_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
  - config_name: wal_asr_v2
    features:
      - name: id
        dtype: string
      - name: speaker_id
        dtype: string
      - name: transcription
        dtype: string
      - name: language
        dtype: string
      - name: gender
        dtype: string
      - name: audio
        dtype: audio
---

# Waxal Datasets

The WAXAL dataset is a large-scale multilingual speech corpus for African languages, introduced in the paper [WAXAL: A Large-Scale Multilingual African Language Speech Corpus](https://huggingface.co/papers/2602.02734).

## Table of Contents

-   [Dataset Description](#dataset-description)
    -   [ASR Dataset](#asr-dataset)
    -   [TTS Dataset](#tts-dataset)
    -   [How to Use](#how-to-use)
-   [Dataset Structure](#dataset-structure)
    -   [ASR Data Fields](#asr-data-fields)
    -   [TTS Data Fields](#tts-data-fields)
    -   [Data Splits](#data-splits)
    -   [ASR v2 splits](#asr-v2-splits-speaker-disjoint)
-   [Dataset Curation](#dataset-curation)
-   [Considerations for Using the Data](#considerations-for-using-the-data)
-   [Additional Information](#additional-information)
-   [Citation](#citation)

## Dataset Description

The Waxal project provides datasets for both Automated Speech Recognition (ASR)
and Text-to-Speech (TTS) for African languages. The goal of this dataset's
creation and release is to facilitate research that improves the accuracy and
fluency of speech and language technology for these underserved languages, and
to serve as a repository for digital preservation.

The Waxal datasets are collections acquired through partnerships with Makerere
University, The University of Ghana, Digital Umuganda, Media Trust, Loud and
Clear, and AIMS Senegal. Acquisition was funded by Google and the Gates
Foundation under an agreement to make the dataset openly accessible. The Senegalese
languages (Wolof and Pular) were provided by AIMS Senegal.

### ASR Dataset

The Waxal ASR dataset is a collection of data in 19 African languages. It
consists of approximately 1,250 hours of transcribed natural speech from a wide
variety of voices. The 19 languages in this dataset represent over 100 million
speakers across 40 Sub-Saharan African countries.

Provider            | Languages                                | License
:------------------ | :--------------------------------------- | :------------:
Makerere University | Acholi, Luganda, Masaaba, Nyankole, Soga                                  | `CC-BY-SA-4.0`
University of Ghana | Akan, Ewe, Dagbani, Dagaare, Ikposo                                       | `CC-BY-4.0`
Digital Umuganda    | Fula, Lingala, Shona, Malagasy, Amharic, Oromo, Sidama, Tigrinya, Wolaytta | `CC-BY-SA-4.0`

### TTS Dataset

The Waxal TTS dataset is a collection of text-to-speech data in 17 African
languages. It consists of over 180 hours of high-quality, single-speaker recordings reading phonetically balanced scripts.

Provider            | Languages                            | License
:------------------ | :----------------------------------- | :------------:
Makerere University | Acholi, Luganda, Kiswahili, Nyankole                            | `CC-BY-SA-4.0`
University of Ghana | Akan (Fante, Twi), Baoule, Ewe                                  | `CC-BY-4.0`
Media Trust         | Fula, Igbo, Hausa, Yoruba, Nigerian Pidgin                      | `CC-BY-SA-4.0`
Loud and Clear      | Kikuyu, Luganda, Luo, Swahili                                   | `CC-BY-SA-4.0`
AIMS Senegal        | Bambara, Pular, Wolof                                           | `CC-BY-SA-4.0`

### How to Use

The `datasets` library allows you to load and pre-process your dataset in pure
Python, at scale.

First, ensure you have the necessary dependencies installed to handle audio
data. You will need `ffmpeg` installed on your system.

**Google Colab / Ubuntu**
```bash
sudo apt-get install ffmpeg
pip install datasets[audio]
```

**macOS**
```bash
brew install ffmpeg
pip install datasets[audio]
```

**Windows**
Download and install from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's in your PATH.
```bash
pip install datasets[audio]
```

If you encounter `RuntimeError: Could not load libtorchcodec`, please ensure `ffmpeg` is correctly installed or check for compatibility between your `torch`, `torchaudio`, and `torchcodec` versions.

**Loading ASR Data**

To load ASR data for a specific language, specify the configuration name, e.g.
`sna_asr` for Shona ASR data.

```python
from datasets import load_dataset, Audio

# Load Shona (sna) ASR dataset
asr_data = load_dataset("google/WaxalNLP", "sna_asr")

# Access splits
train = asr_data['train']
val = asr_data['validation']
test = asr_data['test']

# Example: Accessing audio bytes and other fields
example = train[0]
print(f"Transcription: {example['transcription']}")
print(f"Sampling Rate: {example['audio']['sampling_rate']}")
# 'array' contains the decoded audio bytes as a numpy array
print(f"Audio Array Shape: {example['audio']['array'].shape}")
```

**Loading TTS Data**

To load TTS data for a specific language, specify the configuration name, e.g.
`swa_tts` for Swahili TTS data.

```python
from datasets import load_dataset

# Load Swahili (swa) TTS dataset
tts_data = load_dataset("google/WaxalNLP", "swa_tts")

# Access splits
train = tts_data['train']
```

## Dataset Structure

### ASR Data Fields

```python
{
  'id': 'sna_0',
  'speaker_id': '...',
  'audio': {
    'array': [...],
    'sample_rate': 16_000
  },
  'transcription': '...',
  'language': 'sna',
  'gender': 'Female',
}
```

*   **id**: Unique identifier.
*   **speaker_id**: Unique identifier for the speaker.
*   **audio**: Audio data.
*   **transcription**: Transcription of the audio.
*   **language**: ISO 639-2 language code.
*   **gender**: Speaker gender ('Male', 'Female', or empty).

### TTS Data Fields

```python
{
  'id': 'swa_0',
  'speaker_id': '...',
  'audio': {
    'array': [...],
    'sample_rate': 16_000
  },
  'text': '...',
  'locale': 'swa',
  'gender': 'Female',
}
```

*   **id**: Unique identifier.
*   **speaker_id**: Unique identifier for the speaker.
*   **audio**: Audio data.
*   **text**: Text script.
*   **locale**: ISO 639-2 language code.
*   **gender**: Speaker gender.

### Data Splits

For the **ASR Dataset**, the data with transcriptions is split as follows: *
**train**: 80% of labeled data. * **validation**: 10% of labeled data. *
**test**: 10% of labeled data.

The **unlabeled** split contains all samples that do not have a corresponding
transcription.

The **TTS Dataset** follows a similar structure, with data split into `train`,
`validation`, and `test` sets.

### ASR v2 splits (speaker-disjoint)

The ASR half also ships a second set of splits intended for benchmarking, under
`data/ASR_v2/`. They re-partition the same utterances so that **no speaker
appears in more than one split**, meaning a model is always evaluated on voices
it did not train on. The audio is untouched -- v2 is a re-labelling, not a new
release of the recordings.

| Across all 19 ASR languages | original | v2 |
|---|---|---|
| Eval speakers also present in train | up to 100 % | **0 %** |
| Eval utterances with an exact transcript twin in train | 1.09 % | **0.09 %** |
| Split sizes by duration | 80 / 10 / 10 nominal | 84.5 / 7.7 / 7.7 |

438,230 utterances, 2,242 hours. Gender is matched across splits for the
12 languages that carry gender labels. Per-language figures are in
`data/ASR_v2/metadata/split_comparison.csv`, and the full provenance -- seed,
source revision, every parameter -- in `metadata/splits_manifest.json`.

#### Already speaker-disjoint languages (Akan, Amharic, Oromo, Sidama, Tigrinya, Wolaytta)

Six languages were **already speaker-disjoint in the original (v1) datasets** and did not require a re-split:
- **Amharic (`amh`)**, **Oromo (`orm`)**, **Sidama (`sid`)**, **Tigrinya (`tir`)**, and **Wolaytta (`wal`)** (Digital Umuganda) already had 0.00% speaker leakage between `train` and `validation`/`test`.
- **Akan (`aka`)** (University of Ghana) already had 0.00% speaker leakage in its `test` split.

For these six languages, v2 directly points to the original dataset splits. You can load them directly via their v2 configs (`aka_asr_v2`, `amh_asr_v2`, `orm_asr_v2`, `sid_asr_v2`, `tir_asr_v2`, `wal_asr_v2`) or their original configs without the split map re-labelling procedure:

```python
from datasets import load_dataset

# Direct loading for already speaker-disjoint languages (e.g. Amharic)
ds_test = load_dataset("google/WaxalNLP", "amh_asr_v2", split="test")
# Or equivalently using the original config:
# ds_test = load_dataset("google/WaxalNLP", "amh_asr", split="test")
```

#### Reading v2 (for re-partitioned languages)

Read the existing ASR configs and re-label each row from the split map:

```python
import pandas as pd
from datasets import load_dataset

LANG, WANT = "sna", "test"
m = pd.read_csv(
    f"hf://datasets/google/WaxalNLP/data/ASR_v2/metadata/split_map/{LANG}.csv",
    dtype=str,
)
v2 = dict(zip(m["id"], m["v2_split"]))       # `id` is unique within a language

for v1_split in ("train", "validation", "test"):          # all three
    for row in load_dataset("google/WaxalNLP", f"{LANG}_asr",
                            split=v1_split, streaming=True):
        if v2.get(row["id"]) == WANT:
            ...
```

Three things worth knowing:

- **Read all three original labelled splits.** Utterances move between them:
  Acholi's v2 test set draws 397 of its 515 rows from the original *train*.
  Reading only the same-named split returns a fraction of the data, silently.
- **Skip `unlabeled`.** It carries no transcripts and takes no part in the
  re-split.
- **Two utterances are unmapped** -- `kpo_149159` has no `speaker_id` and
  `lin_9193` has an empty transcript. They are listed in
  `metadata/excluded_rows.csv`; `v2.get()` returns `None` for them.

Streaming bounds memory rather than download: filtering still transfers the
language's labelled half, 1.6 GB (`ach`) to 12.7 GB (`sid`).

## Dataset Curation

The data was gathered by multiple partners:

Provider            | Dataset                                                  | License
:------------------ | :------------------------------------------------------- | :------
University of Ghana | [UGSpeechData](https://doi.org/10.57760/sciencedb.22298) | `CC BY 4.0`
Digital Umuganda    | [AfriVoice](DigitalUmuganda/AfriVoice)                   | `CC-BY-SA 4.0`
Makerere University | [Yogera Dataset](https://doi.org/10.7910/DVN/BEROE0)     | `CC-BY-SA 4.0`
Media Trust         |                                                          | `CC-BY-SA 4.0`
Loud and Clear      |                                                          | `CC-BY-SA 4.0`
AIMS Senegal        |                                                          | `CC-BY-SA 4.0`

## Considerations for Using the Data

Please check the license for the specific languages you are using, as they may
differ between providers.

**Affiliation:** Google Research

## Version and Maintenance

-   **Current Version:** 2.0.0
-   **Last Updated:** 03/2026

## Citation

```bibtex
@article{waxal2026,
  title={WAXAL: A Large-Scale Multilingual African Language Speech Corpus},
  author={Anonymous},
  journal={arXiv preprint arXiv:2602.02734},
  year={2026}
}
```