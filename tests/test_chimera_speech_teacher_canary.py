import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave

from experiments.chimera_recovery import speech_teacher_canary as C


class CanaryContractTests(unittest.TestCase):
    def fixture(self, root):
        asset = root / "original.wav"
        with wave.open(str(asset), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x01\x00" * 1200)
        reference = {"id": "source-a", "split": "train", "speaker": "Ryan", "text": "A small boat moves across the lake.",
                     "source_row_sha256": "a" * 64, "recovery_group_id": "group-a", "reference_source_duration_seconds": 0.05,
                     "optimizer_admitted": False, "baseline_original_audio": {"path": str(asset), "sha256": C.sha(asset), "samples": 1200, "sample_rate": 24000}}
        plan = {"rows": [reference]}
        plan["manifest_sha256"] = C.digest(plan)
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        request = {"plan": {"path": str(path), "sha256": C.sha(path)}, "comparison_plan_sha256": plan["manifest_sha256"],
                   "rows": [copy.deepcopy(reference)], "limits": {"threads": 4, "memory_cap_gib": 24, "task_timeout_seconds": 600, "max_new_tokens": 256}}
        request["request_sha256"] = C.digest(request)
        return request

    def reseal(self, request):
        request.pop("request_sha256", None)
        request["request_sha256"] = C.digest(request)

    def test_full_pcm_reference_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.fixture(Path(tmp))
            _, rows = C.validate_request(request)
            self.assertEqual(rows[0]["baseline_original_audio"]["samples"], 1200)

    def test_changed_transcript_voice_and_group_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(Path(tmp))
            for key, value in (("text", "Leaked answer"), ("speaker", "Vivian"), ("recovery_group_id", "group-b")):
                with self.subTest(key=key):
                    request = copy.deepcopy(base)
                    request["rows"][0][key] = value
                    self.reseal(request)
                    with self.assertRaisesRegex(ValueError, "changed original reference"):
                        C.validate_request(request)

    def test_trimmed_waveform_refused_despite_resealed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.fixture(root)
            path = root / "original.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x01\x00" * 1199)
            request["rows"][0]["baseline_original_audio"]["sha256"] = C.sha(path)
            self.reseal(request)
            with self.assertRaisesRegex(ValueError, "changed full original waveform identity"):
                C.validate_request(request)

    def test_resource_scope_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.fixture(Path(tmp))
            request["limits"]["threads"] = 8
            self.reseal(request)
            with self.assertRaisesRegex(ValueError, "resource/generation policy"):
                C.validate_request(request)

    def test_holdout_reference_cannot_enter_bounded_train_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.fixture(Path(tmp))
            path = Path(request["plan"]["path"])
            plan = json.loads(path.read_text())
            plan["rows"][0]["split"] = "validation"
            plan.pop("manifest_sha256")
            plan["manifest_sha256"] = C.digest(plan)
            path.write_text(json.dumps(plan))
            request["plan"]["sha256"] = C.sha(path)
            request["comparison_plan_sha256"] = plan["manifest_sha256"]
            self.reseal(request)
            with self.assertRaisesRegex(ValueError, "training references"):
                C.validate_request(request)

    def test_duplicate_rows_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.fixture(Path(tmp))
            request["rows"] *= 2
            self.reseal(request)
            with self.assertRaisesRegex(ValueError, "distinct sources"):
                C.validate_request(request)

    def test_scoring_preserves_verbatim_and_word_error_distinction(self):
        metrics = C.transcript_metrics("Blue boats move.", "blue boats move")
        self.assertFalse(metrics["verbatim_exact"])
        self.assertTrue(metrics["case_punctuation_exact"])
        self.assertEqual(metrics["wer"], 0)
        metrics = C.transcript_metrics("Blue boats move.", "blue ships move fast")
        self.assertEqual(metrics["word_edit_distance"], 2)
        self.assertEqual(metrics["wer"], 2 / 3)
        with self.assertRaisesRegex(ValueError, "empty reference"):
            C.transcript_metrics("", "anything")


if __name__ == "__main__":
    unittest.main()
