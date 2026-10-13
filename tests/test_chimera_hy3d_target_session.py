import copy
from pathlib import Path
import tempfile
import unittest

from experiments.chimera_recovery import hy3d_target_session as S


class TargetSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.entries=[{"id":"geometry-corrected-media-seed-"+format(i,"024x"),"original_id":"media-seed-"+format(i,"024x"),
             "source_row_sha256":"a"*64,"source_admission_sha256":"b"*64,"sample_seed":S.SEED,
             "dense_resolution":16,"repeat_sample":i%8==0,"split":"train" if i<8 else "test",
             "source_train_eligible":i<8} for i in range(24)]
        self.plan={"schema":S.PLAN,"sealed":True,"source_year":2026,"rows":24,"sources":self.entries,
             "tool_pins":S.tool_pins(),"parent_source_pins":S.H.PINS,"original_whole_parent_sha256":S.H.WEIGHT_SHA,
             "policy":{"best_seed_selection":False,"clean_encoder_query_xyz_generation_input_allowed":False}}

    def tearDown(self):self.temp.cleanup()

    def seal(self,body):
        body=copy.deepcopy(body);body["plan_sha256"]=S.A.digest(body)
        path=self.root/"plan.json";path.write_bytes(S.A.canonical(body))
        return path,S.A.file_sha(path)

    def test_full_plan_and_fixed_seed_not_silent_small_subset(self):
        path,pin=self.seal(self.plan);self.assertEqual(len(S.validate_plan(path,pin)["sources"]),24)
        for kind in ("subset","seed","held","source","target_leak"):
            changed=copy.deepcopy(self.plan)
            if kind=="subset":changed["sources"].pop();changed["rows"]=23
            elif kind=="seed":changed["sources"][0]["sample_seed"]+=1
            elif kind=="held":changed["sources"][8]["source_train_eligible"]=True
            elif kind=="source":changed["parent_source_pins"]={}
            else:changed["policy"]["clean_encoder_query_xyz_generation_input_allowed"]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):S.validate_plan(*self.seal(changed))

    def test_changed_code_or_source_identity_does_not_get_hash_waiver(self):
        for key in ("tool_pins","original_whole_parent_sha256"):
            body=copy.deepcopy(self.plan);body[key]={} if key=="tool_pins" else "c"*64
            with self.subTest(key=key),self.assertRaises(ValueError):S.validate_plan(*self.seal(body))
        path,pin=self.seal(self.plan)
        with self.assertRaisesRegex(ValueError,"file pin"):S.validate_plan(path,"c"*64)

    def test_full_admission_join_rejects_resealed_holdout_promotion(self):
        rows=[]
        for entry in self.entries:
            row={"schema":S.G.ADMISSION,"id":entry["id"],"item_id":entry["id"],"source_row_sha256":"a"*64,
                 "split":entry["split"],"train_admitted":entry["source_train_eligible"],
                 "held_out":not entry["source_train_eligible"],"blockers":[] if entry["source_train_eligible"] else ["held"]}
            rows.append(row);entry["source_admission_sha256"]=S.A.digest(row)
        p=self.root/"admission.jsonl";p.write_bytes(b"".join(S.A.canonical(r)+b"\n" for r in rows))
        m=self.root/"manifest.json"
        m.write_bytes(S.A.canonical({"outputs":{"admission":{"path":"C:\\original\\admission.jsonl","sha256":S.A.file_sha(p)}}}))
        self.assertEqual(len(S.admission_join(self.plan,m,self.root)),24)
        rows[8]["train_admitted"]=True;self.entries[8]["source_train_eligible"]=True
        self.entries[8]["source_admission_sha256"]=S.A.digest(rows[8])
        p.write_bytes(b"".join(S.A.canonical(r)+b"\n" for r in rows))
        m.write_bytes(S.A.canonical({"outputs":{"admission":{"path":"admission.jsonl","sha256":S.A.file_sha(p)}}}))
        with self.assertRaisesRegex(ValueError,"frozen eligibility"):S.admission_join(self.plan,m,self.root)


if __name__=="__main__":unittest.main()
