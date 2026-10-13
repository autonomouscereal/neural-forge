"""Boundaries of the source/job planner; these are not native model tests."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest

from experiments.chimera_recovery import modality_jobs as M


class JobPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.code = self.root / "source.py"
        self.code.write_text("# synthetic metadata unit fixture, 2026; no actual native cache\n")
        self.stage = self.root / "stage.py"
        self.stage.write_text("H3_FPS=24\nH3_SHIFT_VIDEO=12.0\nH3_SHIFT_AUDIO=3.0\n" +
                              "".join(f"def {name}(): pass\n" for name in
                                      ("run_h3_encode", "run_text", "run_velocity", "h3_align_frame_count", "read_items")))
        self.trainer = self.root / "runner/experiments/chimera_recovery"
        self.trainer.mkdir(parents=True)
        for name in ("training.py", "train_media.py", "media_cache.py", "ace_cache.py", "train_music.py"):
            (self.trainer / name).write_text("# labeled unit fixture only\n")
        self.bank = self.root / "frozen.json"
        M.write_json(self.bank, ["a different heldout frozen prompt"])
        self.rows = []
        for route in M.ROUTES:
            for split in sorted(M.SPLITS):
                ident = route + "-" + split
                family = ident
                directory = self.fixtures / "assets" / ident
                directory.mkdir(parents=True)
                roles = {"image_edit": [("image", "input"), ("image", "target")],
                         "joint_av": [("video", "target"), ("audio", "target")],
                         "volume_generate": [("image", "conditioning"), ("mesh", "target")],
                         "vision_grounding": [("image", "input")], "video_understand": [("video", "input")],
                         "audio_understand": [("audio", "input")]}[route]
                assets = []
                for number, (kind, role) in enumerate(roles):
                    path = directory / (str(number) + ".fixture")
                    path.write_bytes((ident + str(number) + " metadata unit fixture bytes").encode())
                    assets.append({**M.pin(path), "type": kind, "role": role, "source_verified": True,
                                   "representation": "raw"})
                gt = directory / "groundtruth.json"
                events = [{"frame": 1, "start_seconds": .125, "duration_seconds": .125, "frequency_hz": 440},
                          {"frame": 7, "start_seconds": .875, "duration_seconds": .125, "frequency_hz": 660},
                          {"frame": 7, "start_seconds": .95, "duration_seconds": .05, "frequency_hz": 880}]
                M.write_json(gt, {"family": family, "case": 0, "groundtruth": {"events": events}})
                row = {"schema": "recovery_corpus.v1", "id": ident, "route": route, "split": split,
                       "messages": [{"role": "user", "content": "unit fixture " + ident},
                                    {"role": "assistant", "content": '{"count":3}'}], "tools": [], "media": assets,
                       "supervision": {"policy": "assistant_only", "message_mask": [0, 1]},
                       "quality": {"source_ready": True, "assets_verified": True, "groundtruth_verified": True,
                                   "native_cache_status": "pending"},
                       "provenance": {"split": split, "group_id": family, "template_family": family,
                                      "source_ids": [ident], "source_year": 2026, "source_code_sha256": M.sha(self.code),
                                      "synthetic_operator_authored": True, "actual_assets_generated": True,
                                      "groundtruth_path": str(gt), "groundtruth_sha256": M.sha(gt)}}
                if route == "joint_av":
                    for asset in assets:
                        asset.update(source_media_sha256=assets[0]["sha256"], start_seconds=0., duration_seconds=1.)
                    assets[0]["fps"] = 8
                    assets[1]["sample_rate"] = 16000
                    row["av_alignment"] = {"same_origin": True}
                self.rows.append(row)
        self.reseal()

    def tearDown(self):
        self.tmp.cleanup()

    def reseal(self):
        artifact = M.write_rows(self.fixtures / "source_ready.jsonl", self.rows)
        M.write_json(self.fixtures / "report.json", {"schema": "operator_media_seed_report.v1",
            "generated_at": "2026-10-13T00:00:00Z", "source_code_sha256": M.sha(self.code),
            "artifacts": {"source_ready.jsonl": artifact}, "examples": len(self.rows),
            "routes": dict(Counter(r["route"] for r in self.rows)),
            "verified_assets": sum(len(r["media"]) for r in self.rows),
            "families": len({r["provenance"]["group_id"] for r in self.rows}), "frozen_eval_prompts_excluded": 1})

    def build(self, name="plan"):
        return M.build(self.fixtures, self.code, self.stage, self.trainer.parents[1], self.root / name,
                       "/home/operator/new-media-stage", self.bank)

    def test_plan_has_real_pins_and_no_native_readiness(self):
        result = self.build()
        self.assertEqual(result["heldout_probes"], 12)
        self.assertEqual(len(result["artifacts"]), 20)
        self.assertTrue(all(not lane["native_train_ready"] for lane in result["lanes"].values()))
        self.assertTrue(all(job["status"] == "prepared_not_launched" for job in result["jobs"]))
        h3 = M.read_rows(self.root / "plan/h3_items.train.jsonl")
        self.assertEqual(len(h3), 1)
        self.assertEqual(h3[0]["split"], "train")
        self.assertFalse(result["training_started"])

    def test_heldout_input_never_contains_answers_or_target_assets(self):
        row = next(r for r in self.rows if r["route"] == "image_edit" and r["split"] == "test")
        task = M.heldout_task(row)
        self.assertEqual([m["role"] for m in task["input"]["messages"]], ["user"])
        self.assertEqual([m["role"] for m in task["input"]["media"]], ["input"])
        self.assertEqual(task["oracle"]["expected_answer"], {"count": 3})
        self.assertNotIn(task["oracle"]["target_assets"][0]["sha256"], M.canonical(task["input"]))
        with self.assertRaisesRegex(ValueError, "training"):
            M.heldout_task(next(r for r in self.rows if r["split"] == "train"))

    def test_same_size_asset_mutation_refused(self):
        path = Path(self.rows[0]["media"][0]["path"])
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "digest"):
            self.build()

    def test_generator_source_change_refused(self):
        self.code.write_text("# different unit fixture source\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.build()

    def test_family_cross_split_refused(self):
        self.rows[1]["provenance"]["group_id"] = self.rows[0]["provenance"]["group_id"]
        self.reseal()
        with self.assertRaisesRegex(ValueError, "family crosses"):
            self.build()

    def test_media_cross_split_refused(self):
        target = self.rows[1]["media"][0]
        target.update(self.rows[0]["media"][0])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "bytes cross"):
            self.build()

    def test_frozen_prompt_collision_refused(self):
        M.write_json(self.bank, [self.rows[0]["messages"][0]["content"]])
        with self.assertRaisesRegex(ValueError, "collides"):
            self.build()

    def test_h3_native_window_recomputes_events_and_clips_end(self):
        row = next(r for r in self.rows if r["route"] == "joint_av" and r["split"] == "train")
        item = M.h3_item(row, self.fixtures.resolve(), PurePosixPath("/new-stage"))
        oracle = item["native_window_oracle"]
        self.assertEqual(oracle["event_count"], 2)  # .95s event falls outside22/24s.
        self.assertEqual(oracle["events"][0]["native_visible_frames"], [3, 4, 5])
        self.assertAlmostEqual(oracle["events"][1]["source_event"]["duration_seconds"], 22 / 24 - .875)
        self.assertFalse(oracle["raw_full_second_oracle_reused"])
        jobs = self.build()["jobs"]
        encode = jobs[0]["argv"]
        self.assertEqual(encode[encode.index("--length") + 1], "22")
        self.assertEqual(encode[encode.index("--start-seconds") + 1], "0")
        self.assertEqual(jobs[2]["depends_on"], ["h3-encode", "h3-text"])

    def test_h3_mismatched_window_refused(self):
        row = next(r for r in self.rows if r["route"] == "joint_av" and r["split"] == "train")
        row["media"][1]["duration_seconds"] = 1.1
        with self.assertRaisesRegex(ValueError, "clock/origin/duration"):
            M.h3_item(row, self.fixtures.resolve(), PurePosixPath("/new-stage"))

    def test_stage_c_clock_drift_refused(self):
        self.stage.write_text(self.stage.read_text().replace("H3_FPS=24", "H3_FPS=25"))
        with self.assertRaisesRegex(ValueError, "clock or schedule"):
            self.build()

    def test_existing_output_and_relative_remote_refused(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.build()
        with self.assertRaisesRegex(ValueError, "absolute target"):
            M.build(self.fixtures, self.code, self.stage, self.trainer.parents[1], self.root / "another",
                    "relative", self.bank)


if __name__ == "__main__":
    unittest.main()
