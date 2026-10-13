from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave

from experiments.chimera_recovery import audio_admission as A


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.asset = self.base / "sound.wav"
        self.write_wave(self.asset, 24000, 99840, b"\x01\x00")
        self.record = {"schema":"recovery_corpus.v1","id":"2026-audio-a","route":"audio_understand","split":"train",
                       "messages":[{"role":"user","content":"Describe these recorded sound events with onset and frequency."},
                                   {"role":"assistant","content":"A one kilohertz tone begins after a quarter second."}],
                       "media":[{"sha256":A.file_sha(self.asset),"path":str(self.asset),"type":"audio","role":"input"}],
                       "provenance":{"source":"authored2026","source_ids":["2026-event1"],"group_id":"tones2026","split":"train"}}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_wave(path, rate, frames, sample):
        with wave.open(str(path),"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(sample*frames)

    def write(self, name, body, jsonl=False):
        path = self.base/name
        path.write_bytes(b"".join(A.canonical(r)+b"\n" for r in body) if jsonl else A.canonical(body)+b"\n")
        return {"path":str(path),"sha256":A.file_sha(path),"rows":len(body) if jsonl else None}

    def setup_specs(self, *, normalized=None, eval_rows=None, prior_rows=None, prior_exclusions=None, frozen=None):
        n = copy.deepcopy(self.record)
        n.update(id="normalized2026",media=[],messages=[{"role":"user","content":"Explain the distinct orbital mechanics question."}])
        n["provenance"].update(group_id="registry2026",recovery_group_id="registry2026",source_ids=["different2026"])
        norm = self.write("normalized.jsonl", normalized if normalized is not None else [n], True)
        ev = self.write("eval.jsonl", eval_rows or [], True)
        registry = self.write("registry.jsonl",[n],True)
        corpus = self.write("corpus.json",{"schema":"recovery_corpus.manifest.v1","outputs":{"normalized":norm},"eval_sources":[ev],"sources":[registry]})
        prows = self.write("prior.jsonl",prior_rows or [],True)
        exclusions = self.write("exclusions.jsonl",prior_exclusions or [],True)
        prior = {"schema":"recovery_media_admission.v1","connected_groups_verified":True,"eval_disjoint":True,
                 "sources":{"normalized":norm,"corpus_manifest":corpus,"eval_rows":ev},
                 "outputs":{"admission":prows,"cross_route_training_exclusions":exclusions}}
        prior["manifest_sha256"] = A.digest(prior)
        return {"normalized":norm,"corpus_manifest":corpus,"eval_rows":ev,"prior_admission":self.write("prior.json",prior),
                "frozen_exclusions":self.write("frozen.json",frozen or [])}

    def run_seal(self,specs,record=None,inventory=None):
        r = record or self.record
        candidate = {"record":r,"asset_sha256":r["media"][0]["sha256"],"asset_path":r["media"][0]["path"]}
        out = self.base/("out"+str(len(list(self.base.glob("out*")))))
        manifest = A.seal_admission(specs=specs,candidates=[candidate],inventory=inventory or [r],output=out)
        return manifest,list(A.rows(out/"admission.jsonl"))[0]

    def test_admits_train_source_but_never_native_or_teacher(self):
        manifest,r = self.run_seal(self.setup_specs())
        self.assertTrue(r["train_admitted"])
        self.assertFalse(r["native_train_ready"])
        self.assertEqual(r["id"],r["item_id"])
        self.assertEqual(r["source_row_sha256"],A.digest(self.record))
        self.assertEqual(r["authored_answer_sha256"],A.digest(self.record["messages"][-1]["content"]))
        self.assertEqual(manifest["traversal_counts"]["normalized"],1)
        self.assertIn("original_registry_0",manifest["sources"])
        self.assertIn("prior_admission_rows",manifest["sources"])
        self.assertEqual(A.digest({k:v for k,v in manifest.items() if k!="manifest_sha256"}),manifest["manifest_sha256"])

    def test_all_route_holdout_asset_propagates_to_audio(self):
        held = copy.deepcopy(self.record)
        held.update(id="2026-otherroute-held",route="joint_av",split="test")
        held["provenance"].update(group_id="otherroute2026",split="test")
        _,r=self.run_seal(self.setup_specs(),inventory=[self.record,held])
        self.assertFalse(r["train_admitted"])
        self.assertIn("source_family_holdout",r["blockers"])

    def test_same_family_different_asset_is_still_held(self):
        held = copy.deepcopy(self.record)
        held.update(id="2026-family-held",split="validation")
        held["media"]=[{"sha256":"d"*64}]
        held["provenance"].update(source_ids=["different"],split="validation")
        _,r=self.run_seal(self.setup_specs(),inventory=[self.record,held])
        self.assertFalse(r["train_admitted"])
        self.assertIn("connected_group_not_train_exclusive",r["blockers"])

    def test_prior_media_exclusion_cannot_be_bypassed(self):
        prior={"item_id":"ace2026","asset_sha256":self.record["media"][0]["sha256"],"split":"train",
               "source_train_eligible":False,"blockers":["ace_captured_holdout"],"connected_source_groups":["registry2026"]}
        _,r=self.run_seal(self.setup_specs(prior_rows=[prior]))
        self.assertIn("ace_captured_holdout",r["blockers"])
        self.assertFalse(r["train_admitted"])

    def test_frozen_spoken_content_blocks_all_voice_variants(self):
        s=copy.deepcopy(self.record);s["route"]="speech_understand"
        s["messages"][-1]["content"]="A gardener moved the clay pots beneath the covered porch after rain."
        _,r=self.run_seal(self.setup_specs(frozen=[s["messages"][-1]["content"]]),record=s,inventory=[s])
        self.assertIn("frozen_eval_exact_text",r["blockers"])
        self.assertFalse(r["train_admitted"])

    def test_generic_observation_instruction_is_not_media_duplicate(self):
        held=copy.deepcopy(self.record);held.update(id="held_other2026",split="test")
        held["media"]=[{"sha256":"c"*64}]
        held["provenance"].update(group_id="noise2026",source_ids=["noise2026"],split="test")
        held["messages"][-1]["content"]="There are three separate bounded noise bursts."
        _,r=self.run_seal(self.setup_specs(),inventory=[self.record,held])
        self.assertTrue(r["train_admitted"])

    def test_changed_or_incomplete_global_source_refuses(self):
        specs=self.setup_specs()
        Path(specs["normalized"]["path"]).write_text("{}\n")
        with self.assertRaisesRegex(ValueError,"source pin"):
            self.run_seal(specs)

    def test_prior_row_count_tamper_refuses_even_with_resealed_manifest(self):
        specs=self.setup_specs()
        p=Path(specs["prior_admission"]["path"]);v=json.loads(p.read_text())
        v["outputs"]["admission"]["rows"]=1
        v.pop("manifest_sha256");v["manifest_sha256"]=A.digest(v)
        specs["prior_admission"]=self.write("prior.json",v)
        with self.assertRaisesRegex(ValueError,"traversal count"):
            self.run_seal(specs)

    def test_minimal_tail_prefix_and_explicit_scope(self):
        derived=self.base/"padded.wav"
        with wave.open(str(derived),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b"\x01\x00"*99840+b"\x00\x00"*360)
        receipt={"schema":"chimera.speech-h3-minimal-tail-padding.v1","source_sha256":A.file_sha(self.asset),
                 "derived_sha256":A.file_sha(derived),"source_rate":24000,"source_samples":99840,
                 "derived_samples":100200,"tail_padding_samples":360,"prefix_pcm_bitwise_equal":True,"trimmed_samples":0}
        result=A.validate_tail_derivation(receipt,self.asset,derived)
        self.assertEqual(result["tail_padding_samples"],360)
        self.assertFalse(result["native_train_ready"])
        self.assertAlmostEqual(result["derived_duration_seconds"],4.175)
        self.write_wave(derived,24000,100200,b"\x02\x00")
        receipt["derived_sha256"]=A.file_sha(derived)
        with self.assertRaisesRegex(ValueError,"identical PCM prefix"):
            A.validate_tail_derivation(receipt,self.asset,derived)

    def procedural(self):
        r=copy.deepcopy(self.record)
        answer={"contains_speech":False,"count":1,"events":[{"kind":"tone","start_seconds":.25,"duration_seconds":.25,"frequency_hz":1000}]}
        r["messages"][-1]["content"]=json.dumps(answer)
        truth=self.write("groundtruth.json",{"groundtruth":answer})
        r["provenance"].update(source_verified=True,source_year=2026,source_code_sha256="a"*64,
                               groundtruth_path=truth["path"],groundtruth_sha256=truth["sha256"])
        r["media"][0].update(bytes=self.asset.stat().st_size,sample_rate=24000,duration_seconds=4.16)
        r["quality"]={"source_ready":True,"groundtruth_verified":True}
        r["supervision"]={"policy":"assistant_only","message_mask":[0,1]}
        source=self.write("understanding.jsonl",[r],True);inventory=self.write("inventory.jsonl",[r],True)
        report=self.write("report.json",{"schema":"operator_media_seed_report.v1","generated_at":"2026-10-13", "source_code_sha256":"a"*64,
                                        "artifacts":{"understanding.jsonl":source,"inventory.jsonl":inventory},
                                        "examples":1,"cross_split_media_conflicts":0,"verified_assets":1})
        measure,_=A.pcm(self.asset)
        proof={**{k:v for k,v in measure.items() if k!="pcm_sha256"},"id":r["id"],"split":"train","recovery_group_id":"tones2026",
               "waveform_sha256":r["media"][0]["sha256"],"groundtruth_sha256":truth["sha256"],
               "authored_answer_sha256":A.digest(answer),"source_row_sha256":A.digest(r)}
        preflight=self.write("preflight.json",{"schema":"chimera_recovery.audio_understanding_preflight.v1","source_ready":True,
                                             "source_sha256":source["sha256"],"source_report_sha256":report["sha256"],"rows":[proof],"audio_rows":1})
        specs={"source":source,"source_inventory":inventory,"source_report":report,"source_preflight":preflight}
        return {k:Path(v["path"]) for k,v in specs.items()},specs

    def test_actual_pcm_groundtruth_preflight_exact_join(self):
        paths,specs=self.procedural()
        candidates,inventory=A.procedural_candidates(paths,specs,self.base)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["measure"]["samples"],99840)
        proof=json.loads(paths["source_preflight"].read_text())
        proof["rows"][0]["source_row_sha256"]="f"*64
        paths["source_preflight"].write_bytes(A.canonical(proof))
        with self.assertRaisesRegex(ValueError,"preflight.*join"):
            A.procedural_candidates(paths,specs,self.base)

    def test_changed_waveform_bytes_refuse_before_admission(self):
        paths,specs=self.procedural()
        self.write_wave(self.asset,24000,99840,b"\x02\x00")
        with self.assertRaisesRegex(ValueError,"asset path/hash/bytes"):
            A.procedural_candidates(paths,specs,self.base)


if __name__ == "__main__":
    unittest.main()
