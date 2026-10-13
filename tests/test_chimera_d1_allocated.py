"""Negative admission and actual production-source optimizer-boundary coverage."""
import argparse
import ast
import asyncio
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
import uuid

from experiments.chimera_recovery import d1_allocated as B
from experiments.chimera_recovery import d1_allocated_contract as R
from experiments.chimera_recovery import d1_allocated_entry as E
from experiments.chimera_recovery import d1_allocated_root as H

V4 = Path(os.environ.get("D1_REVIEWED_V4", "C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/d1-candidate-v4-source-successor"))


def fixture():
    value = B.plan(B.PREPARATION + "/d1-candidate-v5-allocated-successor",
                   "/home/cereal/omni-campaign/distill/runtime_leases/D1-test",
                   "/home/cereal/omni-campaign/distill/d1/test-allocated", "/mnt/lexar/omni-campaign/deltas/test-allocated")
    value.update(schema=R.SCHEMA, active=True, container_id="1" * 64, container_name="codex-d1-test",
                 allocation_group_id="00000000-0000-0000-0000-000000000003", torch_version="2.11.0+cu130",
                 candidate_receipt_sha256="a" * 64, chimera_source_sha256="b" * 64,
                 root_controller_sha256=R.sha(H.__file__),
                 gpus=[dict(uuid=f"GPU-00000000-0000-0000-0000-00000000000{i+1}", host_index=index, container_ordinal=i)
                       for i, index in enumerate([0, 4])])
    uuids = [gpu["uuid"] for gpu in value["gpus"]]
    cuda = dict(device_count=2, ordinals=[0, 1], uuids=uuids, torch_version=value["torch_version"], bf16_matmul_finite=True)
    value["initial_cuda_evidence"] = cuda
    rows = [dict(allocation_id=str(uuid.uuid4()), group_id=value["allocation_group_id"], server="ai", owner=B.OWNER,
                 state="running", kind="generic", name=value["container_name"], container_id=value["container_id"],
                 gpu_uuid=gpu["uuid"], gpu_index=gpu["host_index"], details=dict(gpu_uuids=uuids)) for gpu in value["gpus"]]
    docker = dict(Id=value["container_id"], Image=value["image_id"], Name="/" + value["container_name"], State=dict(Running=True),
                  Config=dict(Labels={"ai.local.gpu-controller": "v1", "ai.local.allocation": value["allocation_group_id"],
                                      "ai.local.owner": B.OWNER, "ai.local.kind": "generic", "ai.local.gpu-uuids": json.dumps(uuids)}),
                  HostConfig=dict(DeviceRequests=[dict(Driver="nvidia", DeviceIDs=uuids)]),
                  Mounts=[dict(Type="bind", Source=m["source"], Destination=m["target"], RW=not m["read_only"]) for m in value["mounts"]])
    host = dict(phase="launch", mem_available_gib=120, k3_running=False, selected_gpu_queues_running=False,
                k3_passes_complete=True, k3_sealed_manifests=4, foreign_gpu_processes=[], chimera_source_sha256=value["chimera_source_sha256"])
    return value, rows, docker, cuda, host


class AllocatedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.value, self.rows, self.docker, self.cuda, self.host = fixture()
        self.now = datetime(2026, 10, 13, 20, 10, tzinfo=timezone.utc)
        candidate = dict(schema="chimera_recovery.d1_allocated_candidate.v1",
                         files={"tools/d1_allocated_contract.py": R.sha(R.__file__)}, successor=dict(manifest_sha256="9" * 64))
        (self.root / "candidate_receipt.json").write_bytes(R.canonical(candidate) + b"\n")
        self.value["candidate_receipt_sha256"] = R.sha(self.root / "candidate_receipt.json")
        self.save_assignment()

    def save_assignment(self):
        path = self.root / "assignment.json"
        path.write_bytes(R.canonical(self.value) + b"\n")
        self.digest = R.sha(path)
        self.args = argparse.Namespace(runtime_assignment=str(path), runtime_assignment_sha256=self.digest)

    def proof(self):
        return R.attest(self.value, self.digest, self.rows, self.docker, self.cuda, self.host, self.now)

    def step(self, proof, now=None):
        path = self.root / "live.json"
        path.write_bytes(R.canonical(proof))
        return R.validate_step(self.args, now=now or self.now, proof_path=path, candidate_root=self.root)

    def test_exact_binding_and_fresh_proof_accept_and_identity_stable(self):
        R.assignment_record(self.args.runtime_assignment, self.digest)
        original = self.step(self.proof())
        self.now += timedelta(seconds=10)
        self.assertEqual(original, self.step(self.proof()))

    def test_source_code_and_readonly_lease_aliases_refused(self):
        for role in ("lease", "candidate", "child"):
            value = copy.deepcopy(self.value)
            next(m for m in value["mounts"] if m["role"] == "run")["source"] = next(m["source"] for m in value["mounts"] if m["role"] == role)
            (self.root / "bad.json").write_bytes(R.canonical(value))
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "aliases"):
                R.assignment_record(self.root / "bad.json", R.sha(self.root / "bad.json"))

    def test_real_container_pg_mismatch_not_json_active_flag(self):
        for field, bad in (("state", "released"), ("owner", "claude"), ("container_id", "2" * 64), ("gpu_index", 6)):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "PostgreSQL"):
                R.attest(self.value, self.digest, rows, self.docker, self.cuda, self.host, self.now)
        with self.assertRaises(ValueError):
            R.attest(self.value, self.digest, self.rows[:1], self.docker, self.cuda, self.host, self.now)

    def test_container_mount_image_and_uuid_changes_refused(self):
        for mutation in (lambda d: d.update(Image="sha256:" + "2" * 64),
                         lambda d: d["Mounts"][0].update(RW=True),
                         lambda d: d["HostConfig"]["DeviceRequests"][0].update(DeviceIDs=["foreign"]),
                         lambda d: d["State"].update(Running=False)):
            docker = copy.deepcopy(self.docker)
            mutation(docker)
            with self.assertRaises(ValueError):
                R.attest(self.value, self.digest, self.rows, docker, self.cuda, self.host, self.now)

    def test_k3_foreign_gpu_and_native_source_fail_closed(self):
        for field, bad in (("k3_running", True), ("k3_passes_complete", False), ("k3_sealed_manifests", 3),
                           ("selected_gpu_queues_running", True), ("foreign_gpu_processes", [dict(pid=17)]),
                           ("chimera_source_sha256", "c" * 64)):
            host = dict(self.host, **{field: bad})
            with self.subTest(field=field), self.assertRaises(ValueError):
                R.attest(self.value, self.digest, self.rows, self.docker, self.cuda, host, self.now)

    def test_memory_floor_before_start_not_self_block_after_buffers_allocate(self):
        host = dict(self.host, mem_available_gib=20)
        with self.assertRaisesRegex(ValueError, "MemAvailable"):
            R.attest(self.value, self.digest, self.rows, self.docker, self.cuda, host, self.now)
        host["phase"] = "training"
        self.assertTrue(R.attest(self.value, self.digest, self.rows, self.docker, self.cuda, host, self.now)["valid"])

    def test_stale_future_naive_revoked_and_foreign_proof_reject(self):
        proof = self.proof()
        for now in (self.now + timedelta(seconds=46), self.now - timedelta(seconds=1)):
            with self.assertRaises(ValueError):
                self.step(proof, now)
        for field, bad in (("valid", False), ("postgres_live", "true"), ("container_id", "f" * 64),
                           ("chimera_source_sha256", "d" * 64), ("phase", "unknown"), ("captured_at", "2026-10-13T20:10:00")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.step(dict(proof, **{field: bad}))

    def test_namespace_mount_access_rejects_actual_readwrite_input(self):
        lines = [f"{i+1} 0 0:1 / {m['target']} {'ro' if m['read_only'] else 'rw'},relatime - ext4 /dev/x rw" for i, m in enumerate(self.value["mounts"])]
        E.mount_access(self.value["mounts"], "\n".join(lines))
        with self.assertRaises(ValueError):
            E.mount_access(self.value["mounts"], "\n".join(lines).replace("/models/child ro,", "/models/child rw,"))

    def test_no_mode_unknown_and_conflicting_mode_refuse_without_runtime_imports(self):
        for argv in ([], ["--run", "--check-only"], ["--r"], ["--host-gpu", "0"]):
            with self.subTest(argv=argv), contextlib_stderr(), self.assertRaises(SystemExit) as caught:
                E.parser().parse_args(argv)
            self.assertEqual(caught.exception.code, 2)

    def test_direct_trainer_host_device_fallback_refused(self):
        self.args.devices = "cuda:0,cuda:4"
        with self.assertRaisesRegex(ValueError, "direct trainer"):
            self.step(self.proof())

    def test_exact_resume_refuses_other_container_assignment(self):
        directory = self.root / "out"
        directory.mkdir()
        (directory / "d1_receipt.json").write_text(json.dumps(dict(identity=dict(allocated_runtime=dict(container="other")))))
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            E.resume_guard(directory, self.step(self.proof()))

    def test_executable_production_source_gate_prevents_optimizer_apply(self):
        path = V4 / "tools/d1_trunk_recovery.py"
        if not path.exists():
            self.skipTest("set D1_REVIEWED_V4 to exact staged reviewed v4 source")
        raw = path.read_bytes()
        published = R.patch_trainer(raw).decode()
        tree = ast.parse(published)
        layer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_layer")
        # Execute the actual production gate immediately preceding apply, rather than a duplicate helper.
        statements = next((nodes[i-2:i+1] for node in ast.walk(layer) if hasattr(node, "body")
                           for nodes in [node.body] for i, statement in enumerate(nodes)
                           if isinstance(statement, ast.Assign) and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                              and n.func.attr == "apply" for n in ast.walk(statement))), None)
        self.assertIsNotNone(statements)
        optimizer = mock.Mock()
        runtime = mock.Mock()
        runtime.validate_step.side_effect = ValueError("revoked actual lease")
        namespace = dict(__import__=lambda name: runtime, rs=argparse.Namespace(args=self.args), trainer=optimizer, host={}, lr=1e-5)
        code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), "published-apply-boundary", "exec")
        with self.assertRaisesRegex(ValueError, "revoked"):
            exec(code, namespace)
        optimizer.apply.assert_not_called()
        runtime.validate_step.side_effect = None
        exec(code, namespace)
        optimizer.apply.assert_called_once_with({}, 1e-5)
        with self.assertRaisesRegex(ValueError, "exact reviewed"):
            R.patch_trainer(raw + b"\n# unreviewed append\n")

    def test_remote_occupancy_patterns_match_real_commands(self):
        tree = ast.parse(H.REMOTE)
        patterns = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "alive" and isinstance(n.args[0], ast.Constant)]
        self.assertEqual(len(patterns), 2)
        self.assertTrue(re.search(patterns[0], "python /queues/frozen/k3_teacher_cache-gpu6.py run --foo"))
        self.assertTrue(re.search(patterns[1], "bash /queues/gpu4-v2.sh"))
        self.assertFalse(re.search(patterns[1], "bash /queues/gpu6.sh"))

    def test_root_monitor_released_pg_group_revokes_after_start(self):
        calls = []
        started = False
        class Conn:
            async def fetchval(inner, *args, **kw): return True
            async def execute(inner, *args, **kw): calls.append(("sql_unlock", args)); return None
            async def close(inner): return None
            async def fetch(inner, sql, server, owner, group, **kw):
                self.assertEqual((server, owner, str(group)), ("ai", B.OWNER, self.value["allocation_group_id"]))
                self.assertNotIn("CREATE", sql)
                rows = copy.deepcopy(self.rows)
                if started: rows[0]["state"] = "released"
                return rows
        def transport(req):
            nonlocal started
            calls.append((req["action"], req))
            if req["action"] == "snapshot": return dict(docker=self.docker, cuda=self.cuda, host=dict(self.host, phase=req["phase"]))
            if req["action"] == "start": started = True
            return {}
        args = argparse.Namespace(assignment=Path(self.args.runtime_assignment), sha256=self.digest, run=True)
        with mock.patch.object(H, "connection", mock.AsyncMock(return_value=Conn())), mock.patch.object(H, "remote", transport):
            with self.assertRaisesRegex(ValueError, "PostgreSQL"):
                asyncio.run(H.execute(args))
        self.assertEqual(sum(action == "start" for action, _ in calls), 1)
        self.assertFalse([req["records"]["live.json"] for action, req in calls if action == "publish"][-1]["valid"])
        self.assertFalse(any(action in {"stop", "allocate", "kill"} for action, _ in calls))

    def test_revoked_or_foreign_group_never_initializes_cuda(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["state"] = "released"
        connection = mock.Mock()
        connection.fetch = mock.AsyncMock(return_value=rows)
        transport = mock.Mock()
        with mock.patch.object(H, "remote", transport), self.assertRaisesRegex(ValueError, "PostgreSQL"):
            asyncio.run(H.guarded_cuda_probe(connection, self.value))
        transport.assert_not_called()
        connection.fetch = mock.AsyncMock(return_value=self.rows)
        metadata = dict(docker=self.docker, host=dict(self.host, foreign_gpu_processes=[dict(pid=13)]))
        transport = mock.Mock(return_value=metadata)
        with mock.patch.object(H, "remote", transport), self.assertRaises(ValueError):
            asyncio.run(H.guarded_cuda_probe(connection, self.value))
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(transport.call_args.args[0]["probe_cuda"])

    def test_native_source_digest_detects_changed_real_file(self):
        directory = self.root / "tree"
        (directory / "src").mkdir(parents=True)
        source = directory / "src/real_native.py"
        source.write_text("weight_scale = 1\n")
        before = R.tree_digest(directory)
        source.write_text("weight_scale = 2\n")
        self.assertNotEqual(before, R.tree_digest(directory))

    def test_pinned_runtime_validator_tamper_rejects_update(self):
        record = json.loads((self.root / "candidate_receipt.json").read_text())
        record["files"]["tools/d1_allocated_contract.py"] = "f" * 64
        (self.root / "candidate_receipt.json").write_bytes(R.canonical(record))
        self.value["candidate_receipt_sha256"] = R.sha(self.root / "candidate_receipt.json")
        self.save_assignment()
        with self.assertRaisesRegex(ValueError, "validator code"):
            self.step(self.proof())


def contextlib_stderr():
    import contextlib
    import io
    return contextlib.redirect_stderr(io.StringIO())


if __name__ == "__main__":
    unittest.main()
