from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave

from experiments.chimera_recovery import speech_admission as S
from experiments.chimera_recovery import audio_admission as A


class SpeechAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        text="A gardener moved clay pots beneath the porch after rain."
        self.plan={"schema":"chimera.speech-teacher-generation-plan.v1","synthetic":True,"original_user_voice_used":False,
                   "sampling":{"do_sample":False,"subtalker_dosample":False,"max_new_tokens":256},"rows":2,"unique_texts":1,
                   "records":[{"id":"speech2026-"+v.lower(),"text":text,"speaker":v,"language":"English","split":"train",
                               "group_id":"a"*64,"source":"authored2026"} for v in ("Ryan","Vivian")]}
        self.sha=A.digest(self.plan)
        self.tts=self.receipt("tts");self.asr=self.receipt("asr")
        for p in self.plan["records"]:
            directory=self.root/p["id"];directory.mkdir()
            original=directory/"original.wav";derived=directory/"h3-aligned.wav"
            for path,samples in ((original,99840),(derived,100200)):
                with wave.open(str(path),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)
                    w.writeframes(b"\x00\x10"*99840+b"\0\0"*(samples-99840))
            t={**p,"audio":"/work/speech-campaign/assets/"+p["id"]+"/original.wav","audio_sha256":A.file_sha(original),
               "h3_audio":"/work/speech-campaign/assets/"+p["id"]+"/h3-aligned.wav","h3_audio_sha256":A.file_sha(derived),
               "source_samples":99840,"source_rate":24000,"channels":1,"derived_samples":100200,"zero_tail_samples":360,
               "source_pcm_prefix_bitwise_preserved":True,"source_trimmed_samples":0,"h3_target_samples":133600,"expected_h3_frames":167,
               "codec_shape":[56,16],"hit_cap":False,"rms":.125,"peak":.125,"speech_teacher_revision":S.MODELS["tts"]["revision"]}
            self.tts["rows"].append(t)
            self.asr["rows"].append({"id":p["id"],"group_id":p["group_id"],"split":"train","text":text,
                                     "transcript":text,"language":"English","audio_sha256":t["h3_audio_sha256"],
                                     "transcript_exact_after_case_punctuation":True,"eligible_for_global_admission":True,"failure_reason":None})
        self.asr.update(eligible_count=2,quarantined_count=0)
        self.download={"schema":"chimera.speech-teacher-download.v1","complete":True,
                       "models":[{"model_id":m["model_id"],"revision":m["revision"],
                                  "files":[{"path":n,"sha256":h,"size":100,"upstream_verified":True} for n,h in m["files"].items()]} for m in S.MODELS.values()]}

    def tearDown(self):
        self.temp.cleanup()

    def receipt(self,mode):
        return {"schema":"chimera.speech-teacher-campaign.v1","mode":mode,"complete":True,"plan_sha256":self.sha,
                "teacher":S.MODELS[mode]["model_id"],"revision":S.MODELS[mode]["revision"],"synthetic":True,"training_started":False,"rows":[]}

    def build(self):
        return S.build_sources(plan=self.plan,plan_sha=self.sha,tts=self.tts,asr=self.asr,download=self.download,asset_root=self.root)

    def test_real_pcm_prefix_exact_asr_and_voice_group_provenance(self):
        c,inventory,q=self.build()
        self.assertEqual((len(c),len(inventory),len(q)),(2,2,0))
        self.assertEqual(inventory[0]["provenance"]["group_id"],inventory[1]["provenance"]["group_id"])
        self.assertEqual(inventory[0]["media"][0]["source_media_sha256"],self.tts["rows"][0]["audio_sha256"])
        self.assertEqual(c[0]["measure"]["samples"],100200)
        self.assertFalse(inventory[0]["quality"]["native_train_ready"])
        self.assertNotIn(inventory[0]["messages"][-1]["content"],inventory[0]["messages"][0]["content"])

    def test_transcript_mismatch_remains_unsupervised_not_polished(self):
        a=self.asr["rows"][0]
        a.update(transcript="The gardener went somewhere else.",transcript_exact_after_case_punctuation=False,
                 eligible_for_global_admission=False,failure_reason="independent teacher transcript mismatch")
        self.asr.update(eligible_count=1,quarantined_count=1)
        c,inventory,q=self.build()
        self.assertEqual(len(c),1)
        self.assertEqual(inventory[0]["supervision"]["message_mask"],[0,0])
        self.assertFalse(q[0]["supervised"])

    def test_partial_campaign_or_canary_cannot_admit_full_plan(self):
        self.tts["complete"]=False
        with self.assertRaisesRegex(ValueError,"incomplete"):
            self.build()
        self.tts["complete"]=True;self.tts["rows"].pop()
        with self.assertRaisesRegex(ValueError,"inventory must match exactly"):
            self.build()

    def test_claimed_exact_flag_is_recomputed(self):
        self.asr["rows"][0]["transcript"]="Seven instead of eleven."
        with self.assertRaisesRegex(ValueError,"exact-agreement receipt"):
            self.build()

    def test_edited_source_text_or_hash_is_not_accepted(self):
        self.tts["rows"][0]["text"]="A changed target sentence."
        with self.assertRaisesRegex(ValueError,"identity join"):
            self.build()
        self.tts["rows"][0]["text"]=self.plan["records"][0]["text"]
        self.tts["rows"][0]["h3_audio_sha256"]="f"*64
        self.asr["rows"][0]["audio_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"minimal zero-tail"):
            self.build()

    def test_official_codec_weight_pin_cannot_be_waived(self):
        self.download["models"][1]["files"][-1]["sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"official reviewed pin"):
            self.build()

    def test_recorded_amplitude_matches_actual_pcm(self):
        self.tts["rows"][0]["rms"]=.5
        with self.assertRaisesRegex(ValueError,"PCM amplitude"):
            self.build()

    def test_same_source_waveform_can_link_global_media_holdout(self):
        _,inventory,_=self.build()
        self.assertIn("asset:"+self.tts["rows"][0]["audio_sha256"],A.identities(inventory[0]))

    def test_near_utterances_keep_connected_identity_across_splits(self):
        p=self.plan["records"][1]
        p.update(text="A gardener moved clay pots beneath the porch after snow.",group_id="b"*64,split="test")
        self.plan["unique_texts"]=2;self.sha=A.digest(self.plan)
        self.tts["plan_sha256"]=self.asr["plan_sha256"]=self.sha
        self.tts["rows"][1].update(p)
        self.asr["rows"][1].update({k:p[k] for k in ("text","group_id","split")},transcript=p["text"])
        _,inventory,_=self.build()
        ids=[next(i for i in r["provenance"]["source_ids"] if i.startswith("near-utterance-group:")) for r in inventory]
        self.assertEqual(ids[0],ids[1])
        self.assertNotEqual(inventory[0]["provenance"]["group_id"],inventory[1]["provenance"]["group_id"])

    def test_held_only_plan_cannot_silently_supply_train_rows(self):
        self.plan["optimizer_use_forbidden"]=True
        with self.assertRaisesRegex(ValueError,"held-only"):
            self.build()

    def test_independent_campaign_near_utterances_propagate_holdout(self):
        _,records,_=self.build()
        held=copy.deepcopy(records[0]);held["id"]="separate-held-plan"
        held["split"]="test";held["provenance"]["group_id"]="b"*64
        held["provenance"]["source_ids"]=["separate-held-source"]
        held["media"]=[{"sha256":"c"*64,"source_media_sha256":"d"*64,"path":"independent-held.wav"}]
        held["messages"][-1]["content"]="A gardener moved clay pots beneath the porch after snow."
        rows=records+[held]
        self.assertFalse(A.identities(records[0]) & A.identities(held))
        S.link_campaign_groups(rows)
        groups=A.Groups()
        indices=[groups.add(A.identities(r)) for r in rows]
        self.assertEqual(len({groups.find(i) for i in indices}),1)
        self.assertEqual(held["split"],"test")
        self.assertEqual(records[0]["provenance"]["tts_row_sha256"],A.digest(self.tts["rows"][0]))

    def test_combined_campaign_duplicate_ids_refused(self):
        _,records,_=self.build()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            S.link_campaign_groups(records+[copy.deepcopy(records[0])])


if __name__ == "__main__":
    unittest.main()
