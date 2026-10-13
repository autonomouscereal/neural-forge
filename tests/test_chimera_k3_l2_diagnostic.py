import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.chimera_recovery import k3_l2_diagnostic as D


class DiagnosticContractTests(unittest.TestCase):
    def fixtures(self):
        segments = [{"rows":24,"sequence_length_used":4096},{"rows":48,"sequence_length_used":512},
                    {"rows":40,"sequence_length_used":512}]
        lane = {k:v for k,v in D.PINS.items() if k not in ("k3_dense_pass.py","k3_teacher_cache-gpu6.py")}
        original = [{"rows":s["rows"],"sequence_length":s["sequence_length_used"],"file_sha256":str(i),
                     "input_ids_sha256":"ids"+str(i)} for i,s in enumerate(segments)]
        fold = {"tool_sha256":D.PINS["k3_dense_pass.py"],"lane_sha256":lane,
                "corpus":{"segments":original,"tokens":143360},
                "layers":{str(k):{"prefix_out_sha256":{"teacher":str(k)}} for k in range(3)}}
        probe = {"verdict":"sealed","problems":[],"tool_sha256":D.PINS["k3_teacher_cache-gpu6.py"],
                 "engine":{"sha256":D.PINS["k3_dense_pass.py"],"lane_sha256":lane},
                 "integrity":{"after_equals_before":True,"changed":[],"drift_vs_fold_receipt":[]},
                 "corpus":{"segments":segments,"positions":143360,"group_tokens":32768,"files":original},
                 "layers":{str(k):{"prefix_out_sha256":str(k) if k<2 else "different"} for k in range(3)}}
        return fold,probe

    def test_actual_shape_plan_is_not_parity_admission(self):
        result=D.diagnose_receipts(*self.fixtures())
        self.assertEqual(result["grouped_positions"],[32768,32768,32768,24576,20480])
        self.assertEqual(result["first_unequal_layer"],2)
        self.assertFalse(result["production_patch_proven"])
        self.assertFalse(result["parity_pass"])
        self.assertFalse(result["training_admitted"])

    def test_large_group_still_is_not_whole_corpus(self):
        fold,probe=self.fixtures();probe["corpus"]["group_tokens"]=143360
        self.assertEqual(D.diagnose_receipts(fold,probe)["grouped_positions"],[98304,24576,20480])

    def test_row_group_boundaries_do_not_crop_sequence_or_skip_tail(self):
        groups=D.make_groups([{"rows":3,"sequence_length_used":7}],10)
        self.assertEqual([(x["start"],x["stop"]) for x in groups],[(0,7),(7,14),(14,21)])

    def test_wrong_lane_or_engine_fails_before_replay(self):
        for key in ["lane_sha256","engine"]:
            fold,probe=self.fixtures()
            if key=="lane_sha256":fold[key]["k3_common.py"]="changed"
            else:probe[key]["sha256"]="changed"
            with self.subTest(key=key),self.assertRaises(ValueError):D.diagnose_receipts(fold,probe)

    def test_corpus_file_shape_and_coverage_mismatch_fail(self):
        cases=[lambda p:p["corpus"]["files"][0].update(file_sha256="tamper"),
               lambda p:p["corpus"]["segments"][0].update(sequence_length_used=4095),
               lambda p:p["corpus"].update(positions=143359)]
        for mutate in cases:
            fold,probe=self.fixtures();probe=copy.deepcopy(probe);mutate(probe)
            with self.assertRaises(ValueError):D.diagnose_receipts(fold,probe)

    def test_partial_or_changed_source_receipt_is_refused(self):
        for field,value in [("verdict","partial"),("problems",["failed"])]:
            fold,probe=self.fixtures();probe[field]=value
            with self.assertRaises(ValueError):D.diagnose_receipts(fold,probe)
        fold,probe=self.fixtures();probe["integrity"]["after_equals_before"]=False
        with self.assertRaises(ValueError):D.diagnose_receipts(fold,probe)

    def test_bad_group_budget_and_geometry_refused(self):
        for budget in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):D.make_groups([],budget)
        with self.assertRaises(ValueError):D.make_groups([{"rows":0,"sequence_length_used":1}],1)

    def test_source_tamper_is_not_an_allowed_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in D.PINS:(root/name).write_text("changed")
            with self.assertRaises(ValueError):D.verify_sources(root)


@unittest.skipUnless(os.environ.get("CHIMERA_K3_DIAGNOSTIC_SOURCES"),"genuine pinned capsule supplied for CPU controls")
class GenuineSourceCPUControls(unittest.TestCase):
    def test_reference_digest_matches_original_fp32_upcast_convention(self):
        D.load_engine(Path(os.environ["CHIMERA_K3_DIAGNOSTIC_SOURCES"]))
        import torch
        import k3_common
        # Exercise multiple hashing chunks and signed zero in the original dtype.
        tensor=torch.arange(70000,dtype=torch.float32).reshape(350,200).to(torch.bfloat16)
        tensor[0,0]=-0.0
        record=D.tensor_record(tensor,torch)
        self.assertEqual(record["fold_fp32_sha256"],k3_common.sha256_tensor(tensor))
        self.assertNotEqual(record["sha256"],record["fold_fp32_sha256"])
    def test_real_packed_mxfp4_ffn_arithmetic_repeats_and_preserves_weights(self):
        report=D.cpu_control(Path(os.environ["CHIMERA_K3_DIAGNOSTIC_SOURCES"]))
        self.assertTrue(all(report["repeats_equal"].values()))
        self.assertTrue(report["weights_unchanged"])
        self.assertFalse(report["real_parent_weights"])
        self.assertFalse(report["parent_parity_proven"])
        self.assertEqual(report["arms"]["whole_A"]["prefix_out"]["shape"],[257,64])
        self.assertEqual(report["arms"]["whole_A"]["y_lat"]["dtype"],"torch.float32")
        self.assertTrue(all(rec["finite"] for arm in report["arms"].values() for rec in arm.values()))

    def test_original_kda_attnres_zero_gain_and_moe_shared_pair_replay(self):
        report=D.cpu_pair_control(Path(os.environ["CHIMERA_K3_DIAGNOSTIC_SOURCES"]))
        self.assertTrue(report["repeats_bitwise_equal"])
        self.assertTrue(report["weights_unchanged"])
        self.assertFalse(report["parent_parity_proven"])
        self.assertTrue(report["record"]["input_unchanged"])
        self.assertEqual(set(report["record"]["comparisons"]),
                         {"carried","ffn_in","latent_in","router_topk_idx","router_topk_w","y_lat","prefix_out"})
        self.assertTrue(all(x["finite"] for arm in ("whole","grouped") for x in report["record"][arm].values()))


if __name__=="__main__":unittest.main()
