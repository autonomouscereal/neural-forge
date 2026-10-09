"""Synthetic CPU pipeline fixtures; GPU subprocess/audit side effects are mocked.

Authority/recipe/freeze/PNG/source-code/pin gates are real, never mocked. This
tests orchestration, not authentic model inference or full tensor replay.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
import c16_qimg_broad_capture as broad
import c16_qimg_fresh_capture as fresh
import c16_qimg_fresh_preregistration as gate
from test_c16_qimg_fresh_stages import Fixture, sha, write


class PipelineTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.f = Fixture(self.root)
        self.receipts = []
        self.sources = {role: {"path": "/synthetic/" + role, **ident}
                        for role, ident in self.f.generation["source_headers"].items()}
        code = {**broad.capture.SOURCE_FILES, **broad.prepare.SOURCE_FILES, **broad.parent.SAMPLING_SOURCE}
        for batch, plan_ref in enumerate(self.f.req["plans"], 1):
            plan = json.loads((self.root / plan_ref["path"]).read_text())
            target = self.root / f"parent{batch}"; target.mkdir()
            body = {"schema": "neural_forge.qimg_parent_images.v1", "status": "complete",
                    "test_fixture_only": "CPU-created PNGs and synthetic metadata; never actual parent generation",
                    "plan_sha256": plan_ref["sha256"], "sampling": plan["sampling"], "source_code_sha256": code,
                    "sources": self.sources, "runtime": {"parent_class": "comfy.model_base.QwenImage",
                        "manual_cast_dtype": "torch.bfloat16", "fp8_optimizations": False,
                        "full_residency_verified": True, "dynamic_patcher": False}, "images": []}
            for i, planned in enumerate(plan["entries"]):
                path = target / (planned["id"] + ".png")
                Image.new("RGB", (512, 512), (batch * 30, i * 25, 90 + i)).save(path)
                body["images"].append({**planned, "file": path.name, "sha256": broad.capture.file_sha(path),
                    "provenance": "native Qwen-Image parent generated; no selection; no edits; not a photograph",
                    "decoded_pixels_sha256": sha("syntheticpixels" + planned["id"]),
                    "native_sample_latent_sha256": sha("syntheticlatent" + planned["id"])})
            receipt = write(self.root, f"parent{batch}/receipt.json", body); self.receipts.append(receipt)
            for actual, entry in zip(self.f.images[(batch-1)*8:batch*8], body["images"]):
                actual.update(image_sha256=entry["sha256"], parent_receipt_sha256=receipt["sha256"])
        exclusion = {"schema": broad.capture.CAPTURE_SCHEMA, "status": "complete", "fixture": False,
                     "test_fixture_only": True, "captures": [{"image_sha256": sha("oldimage" + str(i))} for i in range(40)]}
        self.exclusions = [write(self.root, "old_capture_exclusion.json", exclusion)]
        self.f.refresh("precapture")

    def seal(self):
        self.bundle = fresh.seal(**self.f.args, parent_receipts=self.receipts, exclusions=self.exclusions, out=self.root / "sealed")
        return self.bundle

    def authorize(self):
        now = datetime.now(timezone.utc)
        body = {"schema": gate.AUTH_SCHEMA, "event_id": "synthetic_runtime", "action": "encode_and_capture_only",
                "contract_sha256": self.f.args["contract_sha"], "bundle_sha256": self.bundle["sha256"],
                "approved_at_utc": (now-timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                "not_after_utc": (now+timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                "runtime_image_id": "sha256:" + sha("synthetic-image"), "physical_gpu_uuid": "GPU-synthetic-no-device",
                "scope": "external coordinator manages and verifies runtime/resources; no fold acceptance"}
        auth = write(self.root, "runtime_auth.json", body)
        anchor = write(self.root, "runtime_anchor.json", {"schema": "neural_forge.qimg_capture_runtime_authority_anchor.v1",
            "authority_id": "synthetic-test-owner", "revision": "test2-after-seal",
            "source_authority_anchor_sha256": self.f.args["anchor_sha"], "contract_sha256": self.f.args["contract_sha"],
            "allowed_runtime_authorization_sha256": auth["sha256"]})
        self.run_args = dict(out=self.root / "run", weights="/synthetic/diffusion", vae="/synthetic/vae",
            text_encoder="/synthetic/text_encoder", comfy_root="/synthetic/comfy", threads=2,
            runtime_authorization_path=self.root / auth["path"], runtime_authorization_sha=auth["sha256"],
            runtime_anchor_path=self.root / anchor["path"], runtime_anchor_sha=anchor["sha256"])
        return body, auth, anchor

    def mocked_run(self, mutation=None):
        # No source authority or readiness gates are replaced. Only operations
        # that would require teacher/model tensors are explicit synthetic doubles.
        self.dispatches = []
        def dispatch(argv, **kwargs):
            self.dispatches.append(argv)
            Path(argv[argv.index("--out") + 1]).mkdir(exist_ok=False)
            if mutation and len(self.dispatches) == 1: mutation()
        def header(path): return {}, self.sources[path.name]
        def encoding(out, plan, plan_ref, sources):
            return {"inputs": {"manifest": str(out / "capture_inputs.json")},
                    "row_counts": {"img": {"fit": 24576, "holdout": 8192}, "txt": {"fit": 1600, "holdout": 200}},
                    "actual_rows": {}, "encoding_receipt": {"path": str(out / "encoding_receipt.json"), "sha256": sha(str(out))}}
        def capture(out, plan, encoded, sources, previous_frozen=None):
            return {"capture_receipt": {"path": str(out / "capture_receipt.json"), "sha256": sha(str(out))},
                    "row_counts": encoded["row_counts"], "forwards": 32, "source_frozen_sha256": sha("fixedsource"),
                    "forward_file_sha256": [sha(str(out) + str(i)) for i in range(32)]}
        self.encoding_auditor, self.capture_auditor = encoding, capture
        with patch.object(broad.subprocess, "run", side_effect=dispatch), patch.object(broad.capture, "read_header", side_effect=header), \
             patch.object(broad, "audit_encoding", side_effect=encoding), patch.object(broad, "audit_capture", side_effect=capture):
            self.run_ref = fresh.run(self.bundle["path"], self.bundle["sha256"], **self.run_args)
        return self.run_ref

    def post_args(self, run_ref=None, alter_start=None):
        run_ref = run_ref or self.run_ref
        body = json.loads(Path(run_ref["path"]).read_text())
        start = {"schema": gate.START_SCHEMA, "event_id": "synthetic_actual_pipeline_start",
                 "started_at_utc": body["started_at_utc"], "contract_sha256": self.f.args["contract_sha"],
                 "recipe_sha256": self.f.req["recipe_sha256"], "generated_images_sha256": gate.meta.canonical_sha(self.f.images),
                 "generation_chronology_sha256": self.f.req["generation_chronology"]["sha256"],
                 "postgeneration_qa_sha256": self.f.post_ref["sha256"], "run_receipt_sha256": run_ref["sha256"]}
        if alter_start: alter_start(start)
        ref = write(self.root, "external_start.json", start)
        req = copy.deepcopy(self.f.req); req["capture_start_chronology"] = ref
        request = write(self.root, "postcapture_request.json", req)
        anchor = copy.deepcopy(self.f.anchor); anchor["allowed_raw_sha256"]["capture_start_chronology"] = ref["sha256"]
        anch_ref = write(self.root, "postcapture_anchor.json", anchor)
        args = dict(self.f.args)
        args.update(request_path=self.root / request["path"], request_sha=request["sha256"],
                    anchor_path=self.root / anch_ref["path"], anchor_sha=anch_ref["sha256"])
        return args

    def audited_admit(self, args=None):
        with patch.object(broad, "audit_encoding", side_effect=self.encoding_auditor) as encoding, \
             patch.object(broad, "audit_capture", side_effect=self.capture_auditor) as capture:
            result = fresh.admit(run_receipt_path=self.run_ref["path"], run_receipt_sha=self.run_ref["sha256"],
                                 **(args or self.post_args()))
            self.assertEqual(encoding.call_count, 4); self.assertEqual(capture.call_count, 4)
            return result

    def test_fresh_seal_derives_32_real_png_encoding_entries(self):
        ref = self.seal(); body, plans, _, _ = fresh.validate_bundle(ref["path"], ref["sha256"])
        self.assertEqual(sum(len(p["entries"]) for p in plans), 32)
        self.assertFalse(body["accepted"])
        self.assertEqual(plans[0]["entries"][0]["seed"], broad.calibration_seed(self.f.req["plans"][0]["sha256"],
            json.loads((self.root / "plan1.json").read_text())["entries"][0]))

    def test_legacy_still_refuses_new_generation_plans(self):
        refs = fresh._references(self.root, self.f.req["plans"])
        receipts = fresh._references(self.root, self.receipts)
        batches = [{"number": i, "generation_plan": gp, "parent_receipt": rp} for i, (gp, rp) in enumerate(zip(refs, receipts), 1)]
        with self.assertRaisesRegex(ValueError, "committed broad batch"):
            broad.derive_plans(batches, fresh._references(self.root, self.exclusions))

    def test_full_seal_separate_authorization_run_real_pin_gates(self):
        self.seal(); before = (self.root / "anchor.json").read_bytes(); self.authorize(); ref = self.mocked_run()
        self.assertEqual((self.root / "anchor.json").read_bytes(), before)
        self.assertEqual(len(self.dispatches), 8)
        self.assertEqual([Path(a[1]).name for a in self.dispatches], ["c16_qimg_prepare_inputs.py"]*4 + ["c16_qimg_capture.py"]*4)
        body = json.loads(Path(ref["path"]).read_text()); self.assertFalse(body["accepted"]); self.assertFalse(body["acceptance_enabled"])

    def test_generation_receipt_png_hash_mismatch_refused(self):
        body = json.loads((self.root / "parent1/receipt.json").read_text()); path = self.root / "parent1" / body["images"][0]["file"]
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "PNG bytes changed"): self.seal()

    def test_false_parent_runtime_refused(self):
        body = json.loads((self.root / "parent1/receipt.json").read_text()); body["runtime"]["full_residency_verified"] = False
        ref = write(self.root, "parent1/receipt.json", body); self.receipts[0] = ref
        for e in self.f.images[:8]: e["parent_receipt_sha256"] = ref["sha256"]
        self.f.refresh("precapture")
        with self.assertRaisesRegex(ValueError, "native parent runtime"): self.seal()

    def test_dispatch_rechecks_changed_request(self):
        self.seal(); self.authorize()
        def mutate():
            path = self.root / "request.json"; path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"): self.mocked_run(mutate)
        self.assertEqual(len(self.dispatches), 1)
        failure = json.loads((self.root / "run/broad_capture_failure.json").read_text())
        self.assertEqual(failure["status"], "incomplete")

    def test_runtime_anchor_must_bind_frozen_source_anchor(self):
        self.seal(); _, _, ref = self.authorize()
        body = json.loads((self.root / ref["path"]).read_text()); body["source_authority_anchor_sha256"] = sha("other")
        altered = write(self.root, ref["path"], body); self.run_args["runtime_anchor_sha"] = altered["sha256"]
        with self.assertRaisesRegex(ValueError, "frozen source authority"): self.mocked_run()
        self.assertEqual(len(self.dispatches), 0)

    def test_runtime_authorization_wrong_bundle_refused(self):
        self.seal(); body, ref, anchor = self.authorize(); body["bundle_sha256"] = sha("different")
        new = write(self.root, ref["path"], body)
        anch = json.loads((self.root / anchor["path"]).read_text()); anch["allowed_runtime_authorization_sha256"] = new["sha256"]
        newer = write(self.root, anchor["path"], anch)
        self.run_args.update(runtime_authorization_sha=new["sha256"], runtime_anchor_sha=newer["sha256"])
        with self.assertRaisesRegex(ValueError, "scope/identity differs"): self.mocked_run()

    def test_legacy_output_rejection_precedes_bundle_read(self):
        with self.assertRaisesRegex(ValueError, "run output exists"):
            broad.run("not-present.json", sha("x"), self.root, None, None, None, None, 2)

    def test_legacy_invalid_thread_rejection_precedes_bundle_read(self):
        with self.assertRaisesRegex(ValueError, "CPU thread count"):
            broad.run("not-present.json", sha("x"), self.root / "unused", None, None, None, None, True)

    def test_legacy_invalid_batch_rejection_precedes_canonical_read(self):
        with patch.object(broad.parent, "read_plan", side_effect=AssertionError("must not be read")):
            with self.assertRaisesRegex(ValueError, "four complete broad batches"): broad.derive_plans([], [])

    def test_legacy_exclusion_rejection_precedes_canonical_read(self):
        with patch.object(broad.parent, "read_plan", side_effect=AssertionError("must not be read")):
            with self.assertRaisesRegex(ValueError, "exclusions required"): broad.derive_plans([{}]*4, [])

    def test_postcapture_admission_reaudits_all_batches(self):
        self.seal(); self.authorize(); self.mocked_run(); result = self.audited_admit()
        self.assertTrue(result["actual_native_capture_audits_revalidated"])
        self.assertEqual(result["actual_forwards"], 128); self.assertFalse(result["accepted"])

    def test_postcapture_forged_complete_minimal_run_refused(self):
        self.seal(); self.authorize(); self.mocked_run()
        body = json.loads(Path(self.run_ref["path"]).read_text()); body.pop("native_audited_run")
        changed = write(self.root, "forged_run.json", body)
        self.run_ref = {"path": str(self.root / changed["path"]), "sha256": changed["sha256"]}
        args = self.post_args()
        with self.assertRaisesRegex(ValueError, "fresh run receipt fields differ"): self.audited_admit(args)

    def test_postcapture_changed_native_forward_count_refused(self):
        self.seal(); self.authorize(); self.mocked_run()
        body = json.loads(Path(self.run_ref["path"]).read_text()); ref = body["native_audited_run"]
        native = json.loads(Path(ref["path"]).read_text()); native["actual_forwards"] = 127
        changed = write(Path(ref["path"]).parent, "altered_native.json", native)
        body["native_audited_run"] = {"path": str(Path(ref["path"]).parent / changed["path"]), "sha256": changed["sha256"]}
        changed = write(self.root, "altered_run.json", body); self.run_ref = {"path": str(self.root / changed["path"]), "sha256": changed["sha256"]}
        with self.assertRaisesRegex(ValueError, "actual capture receipts/forward identities/rows"): self.audited_admit()

    def test_postcapture_start_binding_refused(self):
        self.seal(); self.authorize(); self.mocked_run()
        args = self.post_args(alter_start=lambda x: x.update(run_receipt_sha256=sha("different")))
        with self.assertRaisesRegex(ValueError, "capture-start commitments differ"): self.audited_admit(args)

    def test_postcapture_historical_authorization_identity_refused(self):
        self.seal(); self.authorize(); self.mocked_run()
        body = json.loads(Path(self.run_ref["path"]).read_text()); body["coordinator_physical_gpu_uuid"] = "GPU-different"
        changed = write(self.root, "altered_run.json", body); self.run_ref = {"path": str(self.root / changed["path"]), "sha256": changed["sha256"]}
        with self.assertRaisesRegex(ValueError, "historical runtime authorization/declared identities"): self.audited_admit()

    def test_postcapture_runtime_expired_at_admission_is_historical(self):
        self.seal(); self.authorize(); self.mocked_run()
        # Validation uses recorded run start/end rather than wall-clock admission.
        args = self.post_args()
        with patch.object(gate, "datetime") as wall:
            wall.now.side_effect = AssertionError("admission must not require current authorization")
            self.audited_admit(args)


if __name__ == "__main__": unittest.main()
