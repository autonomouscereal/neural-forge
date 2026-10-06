"""Explicit synthetic authority/metadata fixtures; no model/GPU/subprocess calls."""
from __future__ import annotations
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import c16_qimg_fresh_preregistration as gate

TOOLS = Path(__file__).resolve().parent
CODE_ROOT = TOOLS.parents[3]
PREFIX = "reports/omni-campaign-2026-09/composite/tools/"


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def write(root, name, body):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"path": name, "sha256": gate.meta.raw_sha(path)}


class Fixture:
    """Every authority/document/image identity here is synthetic, not real QA."""
    def __init__(self, root):
        self.root = root
        names = ["c16_qimg_parent_images.py", "c16_qimg_prepare_inputs.py", "c16_qimg_broad_capture.py",
                 "c16_qimg_capture.py", "c16_qimg_preflight.py", "c16_qimg_fold.py", "c16_qimg_joint_replay.py",
                 "c16_qimg_development_refold.py", "qimg_fold_policy_broad_sep07.json", "qimg_fold_policy_sep07.json",
                 "c16_qimg_fresh_metadata.py", "c16_qimg_fresh_preregistration.py", "c16_qimg_fresh_capture.py"]
        paths = [PREFIX + name for name in names] + ["reports/omni-campaign-2026-09/media/mediafold/" + name
                 for name in ("__init__.py", "linalg.py", "experimental_fit_selector.py")]
        pins = [{"path": name, "bytes": (CODE_ROOT / name).stat().st_size, "sha256": gate.meta.raw_sha(CODE_ROOT / name)} for name in paths]
        self.history = {"schema": "chimera_omni.qimg_known_history.v1", "scope": "synthetic-only test history",
                        "authoritative_complete": False, "recipe_sha256": [sha("oldrecipe")], "capture_sha256": [sha("oldcapture")],
                        "image_sha256": [sha("oldimage" + str(i)) for i in range(40)],
                        "prompt_sha256": [sha("oldprompt" + str(i)) for i in range(40)],
                        "image_ids": ["old" + str(i) for i in range(40)], "generation_seeds": list(range(40)),
                        "evidence_pins": [write(root, "history_evidence.json", {"synthetic_test_only": True})]}
        self.recipe = {"schema": gate.meta.RECIPE_SCHEMA, "id": "synthetic-new-staged-recipe", "state": "recipe_only_frozen_draft",
                       "source_commit": "e" * 40, "sampling": copy.deepcopy(gate.meta.SAMPLING), "sigmas": gate.meta.SIGMAS[:],
                       "block": 30, "streams": {s: copy.deepcopy(gate.meta.GATES) for s in ("img", "txt")},
                       "rules": copy.deepcopy(gate.meta.RULES), "diffusion_header_sha256": gate.meta.DIFFUSION_HEADER,
                       "source_tool_pins": pins, "native_source_code_pins": dict(gate.NATIVE_SOURCE_PINS),
                       "historical_exclusions": write(root, "history.json", self.history), "integration_boundary": copy.deepcopy(gate.meta.BOUNDARY)}
        self.req = {"schema": gate.REQUEST_SCHEMA, "contract_sha256": None, "recipe_sha256": None, "plans": [], "assignments": [],
                    "family_registry": None, "qa_registry": None, "qa_result": None, "freeze_receipt": None,
                    "generated_images": None, "generation_chronology": None, "capture_start_chronology": None}
        for batch in range(1, 5):
            entries = [{"id": f"synthetic_b{batch}_{i}", "split": "fit" if i < 6 else "holdout",
                        "seed": 1000 + batch * 8 + i, "prompt": f"Synthetic CPU test metadata {batch} {i}"} for i in range(8)]
            self.req["plans"].append(write(root, f"plan{batch}.json", {"schema": gate.meta.PARENT_SCHEMA,
                "scope": "synthetic test; no actual parent generation", "sampling": gate.meta.SAMPLING, "entries": entries}))
            self.req["assignments"].extend({"id": e["id"], "split": e["split"], "generation_seed": e["seed"],
                "prompt_sha256": sha(e["prompt"]), "batch": batch, "family_id": "family_" + e["id"], "source_uid": "uid_" + e["id"]} for e in entries)
        now = datetime.now(timezone.utc)
        self.times = [(now - timedelta(minutes=10-i)).isoformat().replace("+00:00", "Z") for i in range(6)]
        self.family = {"schema": "chimera_omni.qimg_family_registry_response.v1", "registry_id": "synthetic",
                       "revision": "test1", "assignments_sha256": None, "assignments": self.req["assignments"]}
        self.registry = {"schema": "chimera_omni.qimg_exclusion_registry_response.v1", "registry_id": "synthetic",
                         "revision": "test1", "scope_sha256": sha("synthetic scope"), "coverage": dict(gate.AUTHORITY_SCOPE)}
        self.qa = {"schema": "chimera_omni.qimg_hash_only_qa_response.v1", "recipe_sha256": None, "assignments_sha256": None,
                   "known_history_sha256": None, "family_registry_sha256": None, "qa_registry_sha256": None,
                   "checked_at_utc": self.times[0], "coverage": dict(gate.AUTHORITY_SCOPE), "comparison_method": "synthetic-test-only",
                   "collisions": {k: 0 for k in gate.AUTHORITY_SCOPE}}
        self.freeze = {"schema": gate.FREEZE_SCHEMA, "event_id": "synthetic_freeze", "recorded_at_utc": self.times[1]}
        self.images = [{"id": a["id"], "image_sha256": sha("syntheticimage" + a["id"]),
                        "parent_receipt_sha256": sha("parent" + str(a["batch"]))} for a in self.req["assignments"]]
        self.post = {"schema": "chimera_omni.qimg_postgeneration_hash_only_qa_response.v1", "recipe_sha256": None,
                     "generated_images_sha256": None, "qa_registry_sha256": None, "checked_at_utc": self.times[4],
                     "coverage": dict(gate.AUTHORITY_SCOPE), "comparison_method": "synthetic-test-only", "collisions": {k: 0 for k in gate.AUTHORITY_SCOPE}}
        self.generation = {"schema": gate.GENERATION_SCHEMA, "event_id": "synthetic_generation", "started_at_utc": self.times[2],
                           "completed_at_utc": self.times[3], "source_headers": {role: {"header_sha256": gate.meta.DIFFUSION_HEADER if role == "diffusion" else sha(role),
                           "file_bytes": 1000, "header_bytes": 100, "full_file_sha256": None, "scope": "header only; tensor bodies not read"} for role in ("diffusion", "vae", "text_encoder")}}
        self.anchor = {"schema": "chimera_omni.qimg_owner_authority_anchor.v1", "authority_id": "synthetic-test-owner",
                       "revision": "test1", "allowed_qa_methods": ["synthetic-test-only"], "allowed_raw_sha256": {}}
        self.refresh("pregeneration")

    def refresh(self, stage):
        recipe_ref = write(self.root, "recipe.json", self.recipe)
        contract = {"schema": gate.CONTRACT_SCHEMA, "id": "synthetic-test-contract", "recipe": recipe_ref,
                    "authority_scope": dict(gate.AUTHORITY_SCOPE), "authority_limit": gate.AUTHORITY_LIMIT,
                    "accepted": False, "acceptance_enabled": False}
        contract_ref = write(self.root, "contract.json", contract)
        self.req.update(contract_sha256=contract_ref["sha256"], recipe_sha256=recipe_ref["sha256"])
        assignment_sha = gate.meta.canonical_sha(self.req["assignments"])
        self.family.update(assignments_sha256=assignment_sha, assignments=self.req["assignments"])
        self.req["family_registry"] = write(self.root, "family.json", self.family)
        self.req["qa_registry"] = write(self.root, "registry.json", self.registry)
        self.qa.update(recipe_sha256=recipe_ref["sha256"], assignments_sha256=assignment_sha,
                       known_history_sha256=self.recipe["historical_exclusions"]["sha256"],
                       family_registry_sha256=self.req["family_registry"]["sha256"], qa_registry_sha256=self.req["qa_registry"]["sha256"])
        self.req["qa_result"] = write(self.root, "qa.json", self.qa)
        self.commitment = gate.meta.canonical_sha({k: self.req[k] for k in ("contract_sha256", "recipe_sha256", "plans", "assignments")})
        self.freeze.update(contract_sha256=contract_ref["sha256"], recipe_sha256=recipe_ref["sha256"],
                           preparation_commitment_sha256=self.commitment, qa_result_sha256=self.req["qa_result"]["sha256"])
        self.req["freeze_receipt"] = write(self.root, "freeze.json", self.freeze)
        if stage != "pregeneration":
            self.req["generated_images"] = self.images
            self.post.update(recipe_sha256=recipe_ref["sha256"], generated_images_sha256=gate.meta.canonical_sha(self.images),
                             qa_registry_sha256=self.req["qa_registry"]["sha256"])
            self.post_ref = write(self.root, "post.json", self.post)
            self.generation.update(contract_sha256=contract_ref["sha256"], recipe_sha256=recipe_ref["sha256"],
                preparation_commitment_sha256=self.commitment, freeze_receipt_sha256=self.req["freeze_receipt"]["sha256"],
                generated_images_sha256=gate.meta.canonical_sha(self.images), postgeneration_qa=self.post_ref)
            self.req["generation_chronology"] = write(self.root, "generation.json", self.generation)
        for key in ("family_registry", "qa_registry", "qa_result", "freeze_receipt", "generation_chronology", "capture_start_chronology"):
            if self.req[key] is not None: self.anchor["allowed_raw_sha256"][key] = self.req[key]["sha256"]
        if stage != "pregeneration": self.anchor["allowed_raw_sha256"]["postgeneration_qa"] = self.post_ref["sha256"]
        request_ref, anchor_ref = write(self.root, "request.json", self.req), write(self.root, "anchor.json", self.anchor)
        self.args = dict(contract_path=self.root / "contract.json", contract_sha=contract_ref["sha256"],
                         request_path=self.root / "request.json", request_sha=request_ref["sha256"], code_root=CODE_ROOT,
                         evidence_root=self.root, anchor_path=self.root / "anchor.json", anchor_sha=anchor_ref["sha256"])
        return self.args


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name))

    def refused(self, stage="pregeneration"):
        with self.assertRaises((gate.meta.Refused, TypeError, KeyError)):
            getattr(gate, stage)(**self.f.args)

    def test_pregeneration_valid_distinct_contract(self):
        _, report = gate.pregeneration(**self.f.args)
        self.assertTrue(report["stage_prerequisites_satisfied"])
        self.assertFalse(report["accepted"]); self.assertFalse(report["acceptance_enabled"])
        self.assertFalse(report["external_authority_provenance_independently_verified"])
        self.assertTrue(report["actual_execution_blocked_without_pinned_coordinator_authorization"])
        self.assertEqual(self.f.recipe["integration_boundary"], gate.meta.BOUNDARY)

    def test_precapture_needs_no_future_start(self):
        self.f.refresh("precapture")
        _, result = gate.precapture(**self.f.args)
        self.assertEqual(result["stage"], "precapture")
        self.assertIsNone(self.f.req["capture_start_chronology"])

    def test_precapture_refuses_missing_generation(self): self.refused("precapture")

    def test_pregeneration_refuses_postgeneration_evidence(self):
        self.f.refresh("precapture"); self.refused()

    def test_precapture_refuses_start_assertion(self):
        self.f.req["capture_start_chronology"] = {"path": "fake.json", "sha256": sha("fake")}
        self.f.refresh("precapture"); self.refused("precapture")

    def test_missing_full_exclusion_coverage(self):
        self.f.registry["coverage"]["hidden_gold"] = False; self.f.refresh("pregeneration"); self.refused()

    def test_collision_rejected(self):
        self.f.qa["collisions"]["media"] = 1; self.f.refresh("pregeneration"); self.refused()

    def test_bool_collision_rejected(self):
        self.f.qa["collisions"]["media"] = False; self.f.refresh("pregeneration"); self.refused()

    def test_owner_pin_not_self_established(self):
        self.f.anchor["allowed_raw_sha256"]["qa_result"] = sha("wrong")
        ref = write(self.f.root, "anchor.json", self.f.anchor); self.f.args["anchor_sha"] = ref["sha256"]; self.refused()

    def test_plan_row_changed_after_freeze(self):
        path = self.f.root / "plan1.json"; body = json.loads(path.read_text()); body["entries"][0]["seed"] += 500
        write(self.f.root, "plan1.json", body); self.refused()

    def test_family_leak(self):
        self.f.req["assignments"][6]["family_id"] = self.f.req["assignments"][0]["family_id"]
        self.f.refresh("pregeneration"); self.refused()

    def test_freeze_before_qa_rejected(self):
        self.f.freeze["recorded_at_utc"] = "2020-01-01T00:00:00Z"; self.f.refresh("pregeneration"); self.refused()

    def test_generation_before_freeze_rejected(self):
        self.f.generation["started_at_utc"] = "2020-01-01T00:00:00Z"; self.f.refresh("precapture"); self.refused("precapture")

    def test_postgeneration_qa_before_completion_rejected(self):
        self.f.post["checked_at_utc"] = "2020-01-01T00:00:00Z"; self.f.refresh("precapture"); self.refused("precapture")

    def test_historical_image_rejected(self):
        self.f.images[0]["image_sha256"] = self.f.history["image_sha256"][0]
        self.f.refresh("precapture"); self.refused("precapture")

    def test_native_source_union_exact(self):
        self.f.recipe["native_source_code_pins"].pop("comfy/sd.py"); self.f.refresh("pregeneration"); self.refused()

    def test_source_pin_completeness(self):
        self.f.recipe["source_tool_pins"] = [p for p in self.f.recipe["source_tool_pins"] if not p["path"].endswith("c16_qimg_fresh_capture.py")]
        self.f.refresh("pregeneration"); self.refused()

    def test_package_initializer_required(self):
        self.f.recipe["source_tool_pins"] = [p for p in self.f.recipe["source_tool_pins"] if not p["path"].endswith("__init__.py")]
        self.f.refresh("pregeneration"); self.refused()

    def test_imported_gate_shadow_rejected(self):
        target = CODE_ROOT / "shadow/c16_qimg_fresh_preregistration.py"
        target.parent.mkdir(exist_ok=True); target.write_bytes(Path(gate.__file__).read_bytes())
        self.addCleanup(target.unlink); self.addCleanup(lambda: None)
        for pin in self.f.recipe["source_tool_pins"]:
            if pin["path"].endswith("c16_qimg_fresh_preregistration.py"): pin["path"] = "shadow/c16_qimg_fresh_preregistration.py"
        self.f.refresh("pregeneration"); self.refused()

    def test_old_proposal_shape_not_launch_authority(self):
        self.f.req["schema"] = gate.meta.SCHEMA; self.f.refresh("pregeneration"); self.refused()

    def test_ready_flag_assertion_rejected(self):
        self.f.req["ready_for_capture"] = True; self.f.refresh("pregeneration"); self.refused()

    def test_changed_contract_bytes_refused(self):
        path = self.f.root / "contract.json"; path.write_bytes(path.read_bytes() + b" "); self.refused()


if __name__ == "__main__": unittest.main()
