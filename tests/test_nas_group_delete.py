#!/usr/bin/env python3
"""nas_group_delete 1.1.0 tests: the unreadable_in_place and volume plan fields (pure, no database).

The unreadable cases need a POSIX host and a non-root user (chmod 000 must refuse open()); they are skipped on the
workstation and run on the AI server as cereal:
    /home/cereal/.venvs/chimera-intake/bin/python -m unittest tests.test_nas_group_delete -v
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cleanup"))
sys.path.insert(0, str(ROOT))
import cleanup_exec as CE  # noqa: E402
import nas_group_delete as N  # noqa: E402

POSIX_USER = os.name == "posix" and os.geteuid() != 0
WHY = "0600 nobody-owned on the share; cereal cannot open it"


def unit_with(files: dict[str, bytes]) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    for rel, data in files.items():
        p = os.path.join(td.name, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
    return td


def walk(path: str) -> tuple[list[dict], dict]:
    files, _dirs = CE.walk_unit(path)
    return files, {f["rel"]: f for f in files}


class UnreadableCheck(unittest.TestCase):
    def test_readable_file_listed_as_unreadable_blocks(self):
        with unit_with({"a.dump": b"x" * 10, "a.json": b"{}"}) as d:
            _files, by_rel = walk(d)
            u = {"path": d, "planned": {"a.dump": 10}, "unreadable_in_place": {"a.json": WHY}}
            reasons = N.unreadable_check(u, by_rel)
            self.assertEqual(len(reasons), 1)
            self.assertIn("readable now", reasons[0])

    def test_planned_missing_and_unreasoned_entries_block(self):
        with unit_with({"a.dump": b"x"}) as d:
            _files, by_rel = walk(d)
            u = {"path": d, "planned": {"a.dump": 1}, "unreadable_in_place": {"a.dump": WHY, "gone.json": "short"}}
            text = " | ".join(N.unreadable_check(u, by_rel))
            self.assertIn("also planned", text)
            self.assertIn("not in the unit", text)
            self.assertIn("no written reason", text)

    def test_no_field_no_reasons(self):
        with unit_with({"a.dump": b"x"}) as d:
            _files, by_rel = walk(d)
            self.assertEqual(N.unreadable_check({"path": d, "planned": {"a.dump": 1}}, by_rel), [])

    @unittest.skipUnless(POSIX_USER, "needs a POSIX non-root user")
    def test_really_unreadable_file_passes(self):
        with unit_with({"a.dump": b"x", "b.list": b"secret"}) as d:
            os.chmod(os.path.join(d, "b.list"), 0)
            try:
                _files, by_rel = walk(d)
                u = {"path": d, "planned": {"a.dump": 1}, "unreadable_in_place": {"b.list": WHY}}
                self.assertEqual(N.unreadable_check(u, by_rel), [])
            finally:
                os.chmod(os.path.join(d, "b.list"), 0o600)


class MeasureAndArchive(unittest.TestCase):
    def test_measure_unit_without_field_equals_ce_measure(self):
        with unit_with({"a.dump": b"abc", "a.json": b"{}"}) as d:
            files, _ = walk(d)
            log = io.StringIO()
            a = N.measure_unit({"path": d, "planned": {}}, files, log)
            b = CE.measure({"path": d}, files, "index+headers+listing", log)
            self.assertEqual(a["listing_sha256"], b["listing_sha256"])
            self.assertEqual(a["digests"], b["digests"])

    @unittest.skipUnless(POSIX_USER, "needs a POSIX non-root user")
    def test_unreadable_recorded_not_hashed_not_archived(self):
        with unit_with({"a.dump": b"abc", "keep.json": b"{}", "locked.list": b"zz"}) as d, \
                tempfile.TemporaryDirectory() as rec:
            locked = os.path.join(d, "locked.list")
            os.chmod(locked, 0)
            try:
                files, _ = walk(d)
                u = {"path": d, "group": "G", "what": "w", "reason": "r", "evidence": "e", "rebuild": "b",
                     "planned": {"a.dump": 3}, "unreadable_in_place": {"locked.list": WHY}}
                m = N.measure_unit(u, files, io.StringIO())
                d_locked = m["digests"]["locked.list"]
                self.assertEqual(d_locked["unreadable"], WHY)
                self.assertEqual(d_locked["size"], 2)
                self.assertFalse(any(k in d_locked for k in ("sha256", "header_sha256", "head64m_sha256")))
                self.assertIn("locked.list\t2\tunreadable", m["listing_lines"])
                want = hashlib.sha256(("\n".join(m["listing_lines"]) + "\n").encode()).hexdigest()
                self.assertEqual(m["listing_sha256"], want)
                self.assertEqual(m["total_bytes"], 3 + 2 + 2)
                arch = N.archive(u, m, rec, ["a.dump"], ["keep.json", "locked.list"])
                self.assertEqual(arch["recipe_files"], ["a.dump", "keep.json"])
                with tarfile.open(os.path.join(rec, "tree.tar.gz"), "r:gz") as tf:
                    self.assertEqual(sorted(x.name for x in tf if x.isfile()), ["a.dump", "keep.json"])
                rows = {r.split("\t")[0]: r.split("\t")
                        for r in Path(rec, "MANIFEST.tsv").read_text(encoding="utf-8").splitlines()[1:]}
                self.assertEqual(rows["locked.list"][3:5], ["unreadable", WHY])
                self.assertEqual(rows["locked.list"][5:], ["no", "yes", "no"])
                self.assertEqual(rows["a.dump"][5:], ["yes", "no", "yes"])
            finally:
                os.chmod(locked, 0o600)


class VolumeCheck(unittest.TestCase):
    def test_unknown_volume_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("unknown volume", N.volume_check({"path": d, "volume": "bogus"})[0])

    @unittest.skipUnless(os.name == "posix" and os.path.isdir("/mnt/lexar"), "AI-server layout")
    def test_wrong_filesystem_blocks_and_right_one_passes(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as d:
            self.assertEqual(N.volume_check({"path": d, "volume": "home"}), [])
            self.assertIn("not on the lexar filesystem", N.volume_check({"path": d, "volume": "lexar"})[0])
            self.assertIn("not on the nas filesystem", N.volume_check({"path": d})[0])


if __name__ == "__main__":
    unittest.main()
