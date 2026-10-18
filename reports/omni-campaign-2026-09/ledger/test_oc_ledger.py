#!/usr/bin/env python3
"""Tests for the Chimera Omni ledger, checkpoint registry, package and dashboard.

Standard library `unittest` only; no fixtures outside the temp directory each
test builds.  Run::

    python3 test_oc_ledger.py -v

Every test that matters here is a REFUSAL test.  The tooling's whole job is to
refuse -- a merge it cannot disambiguate, a status the counts do not imply, a
clone whose bytes moved, a delete that would take a floor holder or a clone
source with it -- and a refusal that is not tested is a refusal that will be
discovered the wave after it silently stopped happening.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oc_checkpoint  # noqa: E402
import oc_dashboard  # noqa: E402
import oc_ledger  # noqa: E402
import oc_package  # noqa: E402
import oc_seal  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def write_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    """A real (tiny) safetensors file: 8-byte header length, JSON header, data."""

    header: dict[str, object] = {}
    offset = 0
    payload = bytearray()
    for name, shape in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        size = elements * 2  # BF16
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload += bytes(size)
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))


def make_checkpoint_dir(root: Path, name: str, tensors: dict[str, list[int]]) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(directory / "model.safetensors", tensors)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "hidden_size": 8}), encoding="utf-8"
    )
    (directory / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    return directory


def review_payload(
    *,
    useful: int = 2,
    partial: int = 3,
    signal: int = 4,
    artifact: int = 6,
    content: int = 5,
    column_b: int | None = None,
    media: list[dict] | None = None,
    package: str | None = None,
    package_sha256: str | None = None,
    bank2: dict | None = None,
) -> dict:
    return {
        "reviewer": "test-reader",
        "dossier_sha256": "a" * 64,
        "verdict_receipt_sha": "b" * 64,
        "bank1": {
            "lanes_total": 16,
            "useful": useful,
            "partial": partial,
            "observed_miss": 15 - useful - partial,
            "parent_useful_or_partial": 12,
            "sealed_verdict": "alive-degraded",
            "canonical_signal": signal,
            "canonical_artifact_miss": artifact,
            "canonical_content_miss": content,
        },
        "columns": {
            "column_a": useful + partial,
            "column_b": useful + partial if column_b is None else column_b,
        },
        "bank2": bank2,
        "media_out": media or [],
        "smoke_verdict": "coherent",
        "safety": "safe-refusal/redirect",
        "degeneracy_reading": "none",
        "package_path": package,
        "package_sha256": package_sha256,
        "notes": "test review",
    }


def _capture(call) -> tuple[int, str]:
    """Run a command with stdout captured; return ``(exit code, text)``."""

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = call()
    return code, buffer.getvalue()


def _delete_args(checkpoint_id: str, root: Path) -> argparse.Namespace:
    """The argparse namespace `_delete_conditions` reads, without the CLI."""

    return argparse.Namespace(
        checkpoint_id=checkpoint_id,
        reason="x" * 80,
        rebuild_command="ocq.sh X",
        retained_source=str(root / "parents" / "Q35-4B"),
    )


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = self.root / "registry"
        self.ledger = self.root / "ledger.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_checkpoint(self, *argv: str) -> int:
        return oc_checkpoint.main(
            [*argv, "--registry", str(self.registry), "--ledger", str(self.ledger)]
        )

    def register_child(
        self,
        checkpoint_id: str = "OC01-F-Q35-4B-C2S4L03",
        *,
        parent_child: str | None = None,
        stage: str = "fold",
        tensors: dict[str, list[int]] | None = None,
        extra: tuple[str, ...] = (),
    ) -> Path:
        parent_dir = self.root / "parents" / "Q35-4B"
        if not parent_dir.exists():
            make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child_dir = make_checkpoint_dir(
            self.root / "children", checkpoint_id, tensors or {"w": [16, 16]}
        )
        argv = [
            "register",
            "--checkpoint-id", checkpoint_id,
            "--stage", stage,
            "--directory", str(child_dir),
            "--parent-directory", str(parent_dir),
            "--parent-code", "Q35-4B",
            "--recipe-command", f"ocq.sh {checkpoint_id}",
            "--confirm",
            *extra,
        ]
        if stage == "fold" and "--component" not in extra:
            # a fold must NAME what it folded (the >=4x rule is per component),
            # so the fixture names one and lets the headers check it wherever the
            # whole model is itself >=4x smaller.
            parent_n = oc_seal.safetensors_parameter_count(parent_dir)[0]
            child_n = oc_seal.safetensors_parameter_count(child_dir)[0]
            if child_n / parent_n <= oc_checkpoint.MAX_COMPONENT_RATIO:
                argv += ["--component", f"folded:{parent_n}:{child_n}:w"]
            else:  # only part of the model was folded; the rest passed through
                folded = 4 * (parent_n - child_n) // 5
                argv += [
                    "--component", f"folded:{folded}:{folded // 4}",
                    "--passed-through", "everything-else",
                ]
        if parent_child:
            argv += ["--parent-checkpoint-id", parent_child]
        self.assertEqual(0, self.run_checkpoint(*argv))
        return child_dir

    def review(self, checkpoint_id: str, payload: dict, *extra: str) -> int:
        path = self.root / f"rev-{checkpoint_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return self.run_checkpoint(
            "record-review", "--checkpoint-id", checkpoint_id,
            "--review-json", str(path), "--confirm", *extra,
        )

    def build_package(self, checkpoint_id: str, out: Path, *extra: str) -> dict:
        self.assertEqual(
            0,
            oc_package.main(
                ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
                 "--checkpoint-id", checkpoint_id, "--out", str(out), *extra]
            ),
        )
        return json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. canonical bytes and the seal
# --------------------------------------------------------------------------


class TestSeal(Base):
    def test_canonical_bytes_pinned_by_value(self) -> None:
        """The exact byte string the LFM2 campaign's digests are taken over.

        If a later edit "tidies" the dumps call -- adds a space after the comma,
        drops `ensure_ascii`, stops sorting -- every digest in this campaign
        forks from every digest in the last one, silently.  So the bytes are
        pinned here by value and not by re-deriving them from the same call.
        """

        value = {"b": 1, "a": [1, 2, {"z": None, "y": True}], "u": "café", "n": 1.5}
        self.assertEqual(
            b'{"a":[1,2,{"y":true,"z":null}],"b":1,"n":1.5,"u":"caf\\u00e9"}',
            oc_seal.canonical_json_bytes(value),
        )
        self.assertEqual(
            "37bd59a947c37a331f55f2dffb3de98861a24e28fd1f0b235f1f83af74d06752",
            oc_seal.canonical_sha256(value),
        )
        # NaN is not JSON and must not become one.
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.canonical_json_bytes({"x": float("nan")})

    def test_seal_verify_and_tamper(self) -> None:
        body = oc_seal.seal({"schema": "t", "value": 1})
        self.assertEqual(body["receipt_sha"], oc_seal.verify_seal(body))
        tampered = dict(body)
        tampered["value"] = 2
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.verify_seal(tampered)

    def test_write_sealed_refuses_non_canonical_bytes_on_read(self) -> None:
        path = self.root / "r.json"
        oc_seal.write_sealed(path, oc_seal.seal({"schema": "t", "value": 1}))
        self.assertEqual(1, oc_seal.read_sealed(path)["value"])
        # A hand-edit that only re-indents still verifies as JSON but is refused.
        body = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.read_sealed(path)

    def test_directory_digest_and_hardlink_refusal(self) -> None:
        directory = make_checkpoint_dir(self.root, "d", {"w": [4, 4]})
        digest, entries = oc_seal.directory_digest(directory)
        self.assertEqual(3, len(entries))
        self.assertEqual(digest, oc_seal.directory_digest(directory)[0])
        (directory / "config.json").write_text('{"model_type":"x"}', encoding="utf-8")
        self.assertNotEqual(digest, oc_seal.directory_digest(directory)[0])
        try:
            os.link(directory / "config.json", directory / "config-link.json")
        except OSError:  # pragma: no cover - filesystem without hardlinks
            self.skipTest("no hardlink support here")
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.directory_digest(directory)
        self.assertTrue(oc_seal.directory_digest(directory, allow_hardlinks=True)[0])

    def test_a_symlinked_subdirectory_is_refused(self) -> None:
        """A symlinked DIR used to be invisible to both the digest and the count.

        `os.walk(followlinks=False)` does not descend into one and neither does
        `Path.rglob`, so a checkpoint whose shards lived under a symlinked
        `weights/` sealed a directory_sha256 and a parameter_count covering
        none of those bytes -- and `clone` then copied the same visible subset,
        re-digested it, found the digests equal and wrote a receipt asserting a
        byte-identical clone that was missing the weights.  Every path that
        walks a checkpoint must refuse instead.
        """

        outside = make_checkpoint_dir(self.root / "away", "real", {"w": [16, 16]})
        directory = make_checkpoint_dir(self.root / "sym", "ckpt", {"a": [2, 2]})
        try:
            os.symlink(outside, directory / "weights", target_is_directory=True)
        except (OSError, NotImplementedError):  # pragma: no cover - needs privilege
            self.skipTest("no symlink privilege here")
        for call in (
            lambda: oc_seal.directory_digest(directory),
            lambda: oc_seal.safetensors_parameter_count(directory),
            lambda: oc_seal.safetensors_tensor_counts(directory),
            lambda: oc_seal.directory_digest(directory, allow_hardlinks=True),
        ):
            with self.assertRaises(oc_seal.OcSealError) as caught:
                call()
            self.assertIn("symlinked directory", str(caught.exception))
        # and it is the symlink that is refused, not the checkpoint: remove it
        # and the same directory digests and counts fine (4 params, not 260).
        (directory / "weights").unlink()
        self.assertEqual(4, oc_seal.safetensors_parameter_count(directory)[0])
        self.assertTrue(oc_seal.directory_digest(directory)[0])

    def test_a_symlinked_shard_file_is_refused_by_the_counts_too(self) -> None:
        """The file branch already refused in the digest; the counts did not."""

        outside = make_checkpoint_dir(self.root / "away2", "real", {"w": [8, 8]})
        directory = make_checkpoint_dir(self.root / "sym2", "ckpt", {"a": [2, 2]})
        try:
            os.symlink(outside / "model.safetensors", directory / "shard2.safetensors")
        except (OSError, NotImplementedError):  # pragma: no cover
            self.skipTest("no symlink privilege here")
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.safetensors_parameter_count(directory)
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.directory_digest(directory)

    def test_the_count_and_the_digest_walk_the_same_files(self) -> None:
        """One enumerator, so they cannot disagree about what is in a checkpoint."""

        directory = make_checkpoint_dir(self.root / "same", "c", {"a": [2, 3]})
        nested = directory / "shards"
        nested.mkdir()
        write_safetensors(nested / "model-00002.safetensors", {"b": [4]})
        digested = {
            e["path"] for e in oc_seal.directory_manifest(directory)
            if e["path"].endswith(".safetensors")
        }
        counted = {
            p.relative_to(directory).as_posix()
            for p in oc_seal.checkpoint_shards(directory)
        }
        self.assertEqual(digested, counted)
        self.assertEqual(10, oc_seal.safetensors_parameter_count(directory)[0])

    def test_parameter_count_from_headers(self) -> None:
        directory = make_checkpoint_dir(self.root, "p", {"a": [3, 5], "b": [7]})
        count, per_file = oc_seal.safetensors_parameter_count(directory)
        self.assertEqual(22, count)
        self.assertEqual(2, per_file[0]["tensors"])

    def test_a_truncated_safetensors_is_refused(self) -> None:
        """`_DTYPE_BITS` was dead code and the "sanity check" it named did not
        exist, so a half-written shard still yielded a confident count."""

        directory = make_checkpoint_dir(self.root / "t", "c", {"w": [64, 64]})
        shard = directory / "model.safetensors"
        raw = shard.read_bytes()
        shard.write_bytes(raw[: len(raw) - 4096])          # the tail never landed
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.safetensors_parameter_count(directory)
        # a header that lies about its own tensor's width is refused too
        blob = json.dumps(
            {"w": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 99]}},
            separators=(",", ":"),
        ).encode("utf-8")
        shard.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(8192))
        with self.assertRaises(oc_seal.OcSealError):
            oc_seal.safetensors_parameter_count(directory)

    def test_tensor_counts_by_prefix(self) -> None:
        directory = make_checkpoint_dir(
            self.root / "x", "c", {"mlp.up": [4, 4], "mlp.down": [4, 4], "attn.q": [2, 2]}
        )
        counts = oc_seal.safetensors_tensor_counts(directory)
        self.assertEqual((32, 2), oc_seal.prefix_parameter_count(counts, "mlp."))
        self.assertEqual((4, 1), oc_seal.prefix_parameter_count(counts, "attn."))

    def test_writer_lock_is_exclusive(self) -> None:
        lock = self.root / "l.lock"
        with oc_seal.WriterLock(lock):
            with self.assertRaises(oc_seal.OcSealError):
                with oc_seal.WriterLock(lock):
                    pass
        with oc_seal.WriterLock(lock):  # released, so it can be taken again
            pass


# --------------------------------------------------------------------------
# 2. the ledger
# --------------------------------------------------------------------------


class TestLedger(Base):
    def _row(self, instance: str, digest: str, stamp: str) -> dict:
        row = {column: None for column in oc_ledger.CHECKPOINT_ROW_COLUMNS}
        row.update(
            {
                "schema": oc_ledger.CHECKPOINT_ROW_SCHEMA,
                "campaign": oc_seal.CAMPAIGN,
                "checkpoint_id": instance.split("-0")[0],
                "record_instance_id": instance,
                "record_seq": 1,
                "event": "register",
                "stage": "fold",
                "status": "candidate",
                "status_reason": "-",
                "sealed_at": stamp,
                "receipt_sha": digest,
                "notes": "",
                "media_out_artifacts": 0,
                "media_out_scored": 0,
            }
        )
        return row

    def test_append_is_binary_and_canonical(self) -> None:
        oc_ledger.append_rows(self.ledger, [self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:00.000Z")])
        raw = self.ledger.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(1, len(oc_ledger.read_rows(self.ledger)))

    def test_unknown_schema_and_column_drift_refused(self) -> None:
        with self.assertRaises(oc_ledger.OcLedgerError):
            oc_ledger.append_rows(self.ledger, [{"schema": "something-else"}])
        drifted = self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:00.000Z")
        drifted["surprise"] = 1
        with self.assertRaises(oc_ledger.OcLedgerError):
            oc_ledger.append_rows(self.ledger, [drifted])

    def test_read_refuses_a_reformatted_row(self) -> None:
        oc_ledger.append_rows(self.ledger, [self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:00.000Z")])
        row = json.loads(self.ledger.read_text(encoding="utf-8").strip())
        self.ledger.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        with self.assertRaises(oc_ledger.OcLedgerError):
            oc_ledger.read_rows(self.ledger)

    def test_merge_dedupes_and_refuses_an_ambiguous_group(self) -> None:
        shard = self.root / "shard.jsonl"
        report = self.root / "report.json"
        rows = [
            self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z"),
            self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z"),
            self._row("B-0001-register", "b" * 64, "2026-09-01T00:00:02.000Z"),
        ]
        oc_ledger.append_rows(shard, rows)
        self.assertEqual(
            0,
            oc_ledger.main(
                ["merge", "--authority", str(self.ledger), "--shard", str(shard),
                 "--report", str(report), "--apply"]
            ),
        )
        self.assertEqual(2, len(oc_ledger.read_rows(self.ledger)))
        before = self.ledger.read_bytes()

        # a second, different receipt for the same instance and no verdict to
        # disambiguate it: refuse rather than guess
        shard2 = self.root / "shard2.jsonl"
        oc_ledger.append_rows(
            shard2,
            [
                self._row("C-0001-register", "c" * 64, "2026-09-01T00:00:03.000Z"),
                self._row("C-0001-register", "d" * 64, "2026-09-01T00:00:04.000Z"),
            ],
        )
        self.assertEqual(
            3,
            oc_ledger.main(
                ["merge", "--authority", str(self.ledger), "--shard", str(shard2),
                 "--report", str(report), "--apply"]
            ),
        )
        self.assertEqual(before, self.ledger.read_bytes())

    def test_merge_dry_run_writes_nothing(self) -> None:
        """Named for what it tests.

        It used to be called `test_merge_refuses_to_rewrite_the_prefix` and
        tested no refusal at all -- it ran a merge WITHOUT `--apply` and
        asserted the dry run wrote nothing.  No test could have constructed the
        refusal it named, because the check was `result = before + payload` then
        `result.startswith(before)`, which is a tautology.  The real property
        (idempotence, and refusing an instance reused under a new receipt) is
        tested below.
        """

        shard, report = self.root / "s.jsonl", self.root / "r.json"
        oc_ledger.append_rows(
            self.ledger, [self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z")]
        )
        oc_ledger.append_rows(
            shard, [self._row("B-0001-register", "b" * 64, "2026-09-01T00:00:02.000Z")]
        )
        code = oc_ledger.main(
            ["merge", "--authority", str(self.ledger), "--shard", str(shard),
             "--report", str(report)]
        )
        self.assertEqual(0, code)
        self.assertEqual(1, len(oc_ledger.read_rows(self.ledger)))  # dry run

    def test_merge_twice_is_a_no_op(self) -> None:
        """Merging the same shard twice used to duplicate every row.

        Permanently: rows are never rewritten, and `verify` then reported
        `duplicate (...)` and exited 1 for the rest of the campaign.
        """

        shard, report = self.root / "s.jsonl", self.root / "r.json"
        oc_ledger.append_rows(
            shard,
            [
                self._row("Z-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z"),
                self._row("Z-0002-review", "b" * 64, "2026-09-01T00:00:02.000Z"),
            ],
        )
        argv = ["merge", "--authority", str(self.ledger), "--shard", str(shard),
                "--report", str(report), "--apply"]
        self.assertEqual(0, oc_ledger.main(argv))
        after_first = self.ledger.read_bytes()
        self.assertEqual(2, len(oc_ledger.read_rows(self.ledger)))

        self.assertEqual(0, oc_ledger.main(argv))
        self.assertEqual(after_first, self.ledger.read_bytes())   # byte-identical
        self.assertEqual(2, len(oc_ledger.read_rows(self.ledger)))
        report_body = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(0, report_body["rows_appended"])
        self.assertEqual(2, len(report_body["rows_skipped_already_present"]))
        self.assertEqual(0, oc_ledger.main(["verify", "--ledger", str(self.ledger)]))

    def test_merge_refuses_a_second_receipt_for_an_authority_instance(self) -> None:
        shard, report = self.root / "s.jsonl", self.root / "r.json"
        oc_ledger.append_rows(
            self.ledger, [self._row("Z-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z")]
        )
        before = self.ledger.read_bytes()
        oc_ledger.append_rows(
            shard, [self._row("Z-0001-register", "f" * 64, "2026-09-01T00:00:09.000Z")]
        )
        self.assertEqual(
            3,
            oc_ledger.main(
                ["merge", "--authority", str(self.ledger), "--shard", str(shard),
                 "--report", str(report), "--apply"]
            ),
        )
        self.assertEqual(before, self.ledger.read_bytes())

    def test_merge_apply_appends_and_never_rewrites_the_authority(self) -> None:
        """`--apply` used to call `authority.write_bytes(before + payload)`.

        `write_bytes` opens 'wb', which TRUNCATES the file to zero before the
        first byte lands.  A kill, an ENOSPC or a stalled disk in that window
        loses every prior row, and the two prefix checks that follow fire AFTER
        the file has already been destroyed -- they can report the loss and have
        nothing to restore from.  The write is an append now, so the prefix
        invariant is structural instead of asserted.
        """

        shard, report = self.root / "s.jsonl", self.root / "r.json"
        oc_ledger.append_rows(
            self.ledger,
            [
                self._row("Y-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z"),
                self._row("Y-0002-review", "b" * 64, "2026-09-01T00:00:02.000Z"),
            ],
        )
        before = self.ledger.read_bytes()
        oc_ledger.append_rows(
            shard, [self._row("Z-0001-register", "c" * 64, "2026-09-01T00:00:03.000Z")]
        )
        self.assertEqual(
            0,
            oc_ledger.main(
                ["merge", "--authority", str(self.ledger), "--shard", str(shard),
                 "--report", str(report), "--apply"]
            ),
        )
        after = self.ledger.read_bytes()
        self.assertEqual(before, after[: len(before)])   # every prior byte, in place
        self.assertEqual(3, len(oc_ledger.read_rows(self.ledger)))

    def test_merge_holds_no_truncating_write(self) -> None:
        """A source pin, for the same reason the canonical bytes are pinned.

        The prefix assertions cannot tell an append from a rewrite that happened
        to succeed, so the one thing that distinguishes them -- the mode the
        file is opened in -- is checked here.  A "tidy-up" back to `write_bytes`
        is then a test failure and not a silent reintroduction of the only
        operation in this lane that can lose the whole ledger.
        """

        source = inspect.getsource(oc_ledger.cmd_merge)
        self.assertNotIn("authority.write_bytes(", source)
        self.assertIn('open("ab")', source)

    def test_a_row_with_no_receipt_is_a_refusal_not_a_keyerror(self) -> None:
        """`verify` and `merge` index rows by `receipt_sha` without guarding it."""

        row = self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z")
        for column in ("receipt_sha", "sealed_at"):
            broken = dict(row)
            broken[column] = None
            with self.assertRaises(oc_ledger.OcLedgerError):
                oc_ledger.validate_row(broken, where="test")

    def test_a_cell_row_written_here_must_name_this_campaign(self) -> None:
        """R9: both schemas carry `campaign`, not just the checkpoint schema.

        A ported ARCH sealer emitting a cell row for THIS campaign without the
        column would otherwise be accepted and then be indistinguishable from
        one of the ancestor campaign's 1,056 rows.  Reading stays permissive --
        the ancestor rows really have no `campaign` and must still parse.
        """

        cell = {
            "schema": oc_ledger.CELL_ROW_SCHEMA,
            "cell_instance_id": "ARCH.C2S4L03-0001",
            "sealed_at": "2026-09-01T00:00:01.000Z",
            "receipt_sha": "d" * 64,
        }
        oc_ledger.validate_row(cell, where="read")            # ancestor-shaped: fine
        with self.assertRaises(oc_ledger.OcLedgerError):
            oc_ledger.append_rows(self.ledger, [cell])        # written here: refused
        cell["campaign"] = oc_seal.CAMPAIGN
        self.assertEqual(1, oc_ledger.append_rows(self.ledger, [cell]))

    def test_append_refuses_a_ledger_with_no_final_newline(self) -> None:
        oc_ledger.append_rows(
            self.ledger, [self._row("A-0001-register", "a" * 64, "2026-09-01T00:00:01.000Z")]
        )
        self.ledger.write_bytes(self.ledger.read_bytes().rstrip(b"\n"))
        with self.assertRaises(oc_ledger.OcLedgerError):
            oc_ledger.append_rows(
                self.ledger,
                [self._row("B-0001-register", "b" * 64, "2026-09-01T00:00:02.000Z")],
            )

    def test_verify_sees_a_record_whose_row_never_landed(self) -> None:
        """The registry->ledger direction, which nothing checked.

        `oc_ledger verify --registry` walks ROWS and resolves them to records;
        an orphan RECORD (its append died on the lock timeout, or Ctrl-C, or
        ENOSPC) was invisible to it and to `oc_checkpoint verify --deep`, and
        both printed a clean result over a ledger that had lost a checkpoint.
        """

        self.register_child()
        self.ledger.write_bytes(b"")            # the append never landed
        self.assertEqual(0, oc_ledger.main(
            ["verify", "--ledger", str(self.ledger), "--registry", str(self.registry)]
        ))                                       # the OLD check still says fine
        self.assertEqual(1, self.run_checkpoint("verify"))          # this one does not

        self.assertEqual(0, self.run_checkpoint(
            "repair-row", "--checkpoint-id", "OC01-F-Q35-4B-C2S4L03"
        ))                                       # dry run
        self.assertEqual(b"", self.ledger.read_bytes())
        self.assertEqual(0, self.run_checkpoint(
            "repair-row", "--checkpoint-id", "OC01-F-Q35-4B-C2S4L03", "--confirm"
        ))
        self.assertEqual(0, self.run_checkpoint("verify"))
        # a ledger that fails its own byte check is a refusal, not a traceback
        self.ledger.write_text(
            json.dumps(json.loads(self.ledger.read_text(encoding="utf-8")), indent=1) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(3, self.run_checkpoint("verify"))
        self.ledger.write_bytes(b"")
        self.assertEqual(0, self.run_checkpoint(
            "repair-row", "--checkpoint-id", "OC01-F-Q35-4B-C2S4L03", "--confirm"
        ))
        row = oc_ledger.read_rows(self.ledger)[0]
        self.assertEqual(
            oc_checkpoint.current(self.registry, "OC01-F-Q35-4B-C2S4L03")["receipt_sha"],
            row["receipt_sha"],
        )

    def test_the_ancestor_ledger_still_reads(self) -> None:
        """The 1,056 LFM2 rows must parse under this reader, byte-canonically.

        This is the compatibility claim the whole "compatible in spirit" line
        rests on: if the canonical dumps call here differed from the one that
        wrote those rows by one space, every row would fail the byte check.
        The file's digest is pinned by the R3 spec at `d2027831…`; when the
        repo is not on this host the test skips rather than passing vacuously.
        """

        candidates = [
            Path(__file__).resolve().parents[2] / "matrix-campaign-20260810" / "ledger.jsonl",
            Path("/home/cereal/omni-campaign/lfm2-precedent/ledger.jsonl"),
        ]
        for path in candidates:
            if path.is_file():
                break
        else:
            self.skipTest("the ancestor ledger is not on this host")
        self.assertEqual(oc_seal.ANCESTOR["lfm2_ledger_sha256"], oc_seal.file_sha256(path))
        rows = oc_ledger.read_rows(path)
        self.assertEqual(1056, len(rows))
        self.assertEqual({oc_ledger.CELL_ROW_SCHEMA}, {r["schema"] for r in rows})
        # and the ancestor rows carry no `campaign` column: that is how a reader
        # tells the two campaigns apart inside one file.
        self.assertFalse(any("campaign" in row for row in rows))

    def test_verify_resolves_rows_to_records(self) -> None:
        self.register_child()
        self.assertEqual(
            0,
            oc_ledger.main(
                ["verify", "--ledger", str(self.ledger), "--registry", str(self.registry)]
            ),
        )


# --------------------------------------------------------------------------
# 3. the checkpoint registry
# --------------------------------------------------------------------------


class TestRegister(Base):
    def test_dry_run_writes_nothing(self) -> None:
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X1", {"w": [8, 8]})
        code = self.run_checkpoint(
            "register", "--checkpoint-id", "OC01-F-Q35-4B-X1", "--stage", "fold",
            "--directory", str(child), "--parent-directory", str(parent),
            "--component", "w:4096:64:w", "--recipe-command", "ocq.sh X1",
        )
        self.assertEqual(0, code)
        self.assertFalse(self.registry.exists())
        self.assertFalse(self.ledger.exists())

    def test_register_seals_a_record_and_a_row(self) -> None:
        self.register_child()
        record = oc_checkpoint.current(self.registry, "OC01-F-Q35-4B-C2S4L03")
        self.assertEqual("candidate", record["status"])
        self.assertEqual("oc01", record["checkpoint"]["wave"])
        self.assertEqual("C2S4L03", record["checkpoint"]["tag"])
        self.assertAlmostEqual(256 / 4096, record["checkpoint"]["net_ratio"])
        rows = oc_ledger.read_rows(self.ledger)
        self.assertEqual(1, len(rows))
        self.assertEqual(record["receipt_sha"], rows[0]["receipt_sha"])
        self.assertEqual(oc_seal.CAMPAIGN, rows[0]["campaign"])

    def test_bad_name_refused(self) -> None:
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "junk", {"w": [8, 8]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "not-a-name", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--recipe-command", "x", "--confirm",
            ),
        )

    def test_component_over_quarter_is_refused_at_the_fold_stage(self) -> None:
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X2", {"w": [8, 8]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X2", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--recipe-command", "x", "--component", "mlp:1000:300", "--confirm",
            ),
        )
        self.assertEqual(
            0,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X2", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--recipe-command", "x", "--component", "mlp:1000:200", "--confirm",
            ),
        )

    def test_quantized_fold_refused(self) -> None:
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X3", {"w": [8, 8]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X3", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "w:4096:64:w",
                "--recipe-command", "x", "--quantization", "q4_k_m", "--confirm",
            ),
        )

    def test_a_fold_must_name_a_component(self) -> None:
        """`--component` defaulted to `[]`, so the >=4x rule was optional.

        A fold that names nothing has not been checked against the per-component
        rule at all, and the rule is the campaign's whole storage claim.
        """

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X4", {"w": [8, 8]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X4", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--recipe-command", "x", "--confirm",
            ),
        )
        self.assertFalse(self.registry.exists())

    def test_a_fold_that_stored_everything_is_refused(self) -> None:
        """net_ratio 1.0 registered clean and appeared under 'Stage: fold'."""

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X5", {"w": [64, 64]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X5", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "mlp:1000:200", "--recipe-command", "x", "--confirm",
            ),
        )

    def test_components_may_not_claim_more_than_the_model_lost(self) -> None:
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X6", {"w": [60, 60]})
        # the model lost 496 parameters; the component claims 3,000 saved
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X6", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "mlp:4000:1000", "--recipe-command", "x", "--confirm",
            ),
        )
        # ... and is accepted when the addition is stated rather than hidden
        self.assertEqual(
            0,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X6", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "mlp:4000:1000", "--accounting-slack", "2504",
                "--passed-through", "an-added-buffer", "--recipe-command", "x",
                "--confirm",
            ),
        )

    def test_component_counts_are_checked_against_the_headers(self) -> None:
        """With a tensor prefix the two counts are DERIVED, not asserted."""

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "OC01-F-Q35-4B-X7", {"w": [8, 8]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X7", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "w:4096:32:w",  # the headers say 64, not 32
                "--recipe-command", "x", "--confirm",
            ),
        )
        self.assertEqual(
            0,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-X7", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--component", "w:4096:64:w", "--recipe-command", "x", "--confirm",
            ),
        )
        record = oc_checkpoint.current(self.registry, "OC01-F-Q35-4B-X7")
        component = record["checkpoint"]["per_component_ratios"][0]
        self.assertEqual("derived-from-safetensors-headers", component["counts_source"])

    def test_a_typed_parameter_count_the_headers_contradict_is_refused(self) -> None:
        """net_ratio is the campaign's central number, and it could be TYPED.

        Reproduced on the delivered code: a fold child that really stored 3,840
        of its parent's 4,096 parameters (true ratio 0.9375) registered exit=0
        with `net_ratio 0.0625 (16.0000x)` sealed into the record, into the
        ledger row, into SCORECARD.md and into REVIEW_<TAG>.md -- and
        `verify --deep` then returned 0, because it only re-divided the same two
        sealed integers.  When there are shards on disk, the bytes decide.
        """

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "big", {"w": [60, 64]})
        self.assertEqual(4096, oc_seal.safetensors_parameter_count(parent)[0])
        self.assertEqual(3840, oc_seal.safetensors_parameter_count(child)[0])
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-LIED1",
                "--stage", "fold", "--directory", str(child),
                "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                "--parameter-count", "256", "--parent-parameter-count", "4096",
                "--component", "folded:4096:256",
                "--recipe-command", "ocq.sh LIED1", "--confirm",
            ),
        )
        self.assertEqual(
            [], oc_checkpoint.history(self.registry, "OC01-F-Q35-4B-LIED1")
        )

    def test_a_count_that_cannot_be_measured_is_labelled_as_asserted(self) -> None:
        """A checkpoint with no safetensors may still be registered -- loudly."""

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = self.root / "children" / "no-shards"
        child.mkdir(parents=True)
        (child / "weights.bin").write_bytes(bytes(64))
        argv = [
            "register", "--checkpoint-id", "OC01-F-Q35-4B-NOST1",
            "--stage", "fold", "--directory", str(child),
            "--parent-directory", str(parent), "--parent-code", "Q35-4B",
            "--parameter-count", "256", "--component", "folded:4096:256",
            "--recipe-command", "ocq.sh NOST1", "--confirm",
        ]
        self.assertEqual(3, self.run_checkpoint(*argv))       # not silently typed
        self.assertEqual(
            0, self.run_checkpoint(*argv, "--allow-asserted-parameter-counts")
        )
        checkpoint = oc_checkpoint.current(
            self.registry, "OC01-F-Q35-4B-NOST1"
        )["checkpoint"]
        self.assertEqual("asserted-by-operator", checkpoint["parameter_count_source"])
        self.assertEqual(
            "derived-from-safetensors-headers",
            checkpoint["parent_parameter_count_source"],
        )

    def test_the_record_and_the_row_say_where_the_counts_came_from(self) -> None:
        self.register_child()
        checkpoint = oc_checkpoint.current(
            self.registry, "OC01-F-Q35-4B-C2S4L03"
        )["checkpoint"]
        self.assertEqual(
            "derived-from-safetensors-headers", checkpoint["parameter_count_source"]
        )
        row = oc_ledger.read_rows(self.ledger)[0]
        self.assertEqual(
            "derived-from-safetensors-headers", row["parameter_count_source"]
        )
        self.assertEqual(
            "derived-from-safetensors-headers", row["parent_parameter_count_source"]
        )

    def test_verify_rereads_the_headers_and_catches_a_wrong_count(self) -> None:
        """The cheap check `verify --deep` was skipping while it re-hashed bytes.

        Re-dividing two sealed integers can never disagree with the sealed
        quotient.  Re-reading the headers can, and costs 0.00 s on a real 4B.
        """

        child = self.register_child()
        self.assertEqual(0, self.run_checkpoint("verify"))
        # the same file set, one tensor bigger: the sealed count is now wrong
        write_safetensors(child / "model.safetensors", {"w": [32, 16]})
        code, text = _capture(lambda: self.run_checkpoint("verify"))
        self.assertEqual(1, code)
        self.assertIn("safetensors headers", text)

    def test_register_refuses_a_protected_directory(self) -> None:
        """R6, enforced by the only path that may ever delete a checkpoint.

        One `--directory /mnt/ai_data/...` typo made `register` and
        `verify --deep` walk and byte-hash the CIFS share, which reads at under
        3 MB/s -- and a checkpoint registered there, reviewed bad and packaged,
        would have `rmtree`'d part of the share.
        """

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        for where in ("/mnt/ai_data/models/whatever", "/home/cereal/models/Qwen3.5-4B"):
            self.assertEqual(
                3,
                self.run_checkpoint(
                    "register", "--checkpoint-id", "OC01-F-Q35-4B-SHAR1",
                    "--stage", "fold", "--directory", where,
                    "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                    "--recipe-command", "ocq.sh SHAR1", "--confirm",
                ),
                where,
            )

    def test_a_registered_parent_may_live_in_the_model_library(self) -> None:
        """`--stage parent` is the one exception, and only under the library.

        A registered parent is a provenance-verified local copy of a source
        model, and `delete` refuses stage `parent` outright, so nothing under it
        is reachable by an unlink.  The share is refused even for a parent.
        """

        self.assertIsNone(
            oc_checkpoint._protected_prefix(Path("/home/cereal/omni-campaign/oc01/x"))
        )
        oc_checkpoint.assert_not_protected(
            Path("/home/cereal/models/Qwen3.5-4B"), what="--directory", stage="parent"
        )
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.assert_not_protected(
                Path("/mnt/ai_data/models/x"), what="--directory", stage="parent"
            )
        # by PARTS, never by string prefix
        self.assertEqual(
            "/mnt", oc_checkpoint._protected_prefix(Path("/mnt/ai_data_scratch/x"))
        )

    def test_register_may_place_a_checkpoint_under_mnt_lexar(self) -> None:
        """/mnt/lexar is the local USB SSD, not the CIFS share (2026-10-09).

        The Sep-07 S3 trunk child was built on Lexar because root was full, and
        the `/mnt` rule refused its registration although nothing about the
        rule's two reasons (a 3 MB/s walk of the share; an `rmtree` on the
        share) applies to a local disk.  register/clone may now place a
        checkpoint there, of any stage; `/mnt` itself stays the protected
        prefix, so `delete` (iv-c) still refuses it -- see TestDelete.
        """

        for stage in ("fold", "distill", "finetune", None):
            oc_checkpoint.assert_not_protected(
                Path("/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07"),
                what="--directory",
                stage=stage,
            )
        # the clone destination goes through the same door
        oc_checkpoint.assert_not_protected(
            Path("/mnt/lexar/omni-campaign/clones/X.clone-pre-T1"),
            what="the clone destination",
        )
        # ... and it is still a PROTECTED prefix as far as delete is concerned
        self.assertEqual(
            "/mnt", oc_checkpoint._protected_prefix(Path("/mnt/lexar/omni-campaign/x"))
        )
        # by PARTS: a sibling whose name merely starts "lexar" is not Lexar
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.assert_not_protected(
                Path("/mnt/lexarfoo/x"), what="--directory", stage="fold"
            )
        # `..` out of Lexar into the share is the share
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.assert_not_protected(
                Path("/mnt/lexar/omni-campaign/../../ai_data/x"),
                what="--directory",
                stage="fold",
            )

    def test_register_still_refuses_the_cifs_share(self) -> None:
        """Narrowing for Lexar did not open `/mnt/ai_data` or the rest of `/mnt`."""

        for where in ("/mnt/ai_data/omni-campaign/x", "/mnt/ai_data", "/mnt/other/x"):
            for stage in ("fold", "parent", None):
                with self.assertRaises(oc_checkpoint.OcCheckpointError, msg=where):
                    oc_checkpoint.assert_not_protected(
                        Path(where), what="--directory", stage=stage
                    )
        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-SHAR2",
                "--stage", "fold", "--directory", "/mnt/ai_data/omni-campaign/x",
                "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                "--recipe-command", "ocq.sh SHAR2", "--confirm",
            ),
        )
        self.assertFalse(oc_checkpoint.history(self.registry, "OC01-F-Q35-4B-SHAR2"))

    def test_a_dotdot_spelling_walks_past_no_guard(self) -> None:
        """`--directory $OC_ROOT/../models/Qwen3.5-4B` used to be unprotected.

        The guard matched the PARTS of a path made absolute but never resolved,
        and `(..., 'omni-campaign', '..', 'models', 'Qwen3.5-4B')` does not
        start `('/', 'home', 'cereal', 'models')` -- so the source model library
        came back clean, and `..` out of the campaign root into `/mnt` was
        invisible the same way.  That is one typo away from `directory_digest`
        byte-hashing the CIFS share, and from a `bad` review authorising an
        `rmtree` of a source model.
        """

        self.assertEqual(
            "/home/cereal/models",
            oc_checkpoint._protected_prefix(
                Path("/home/cereal/omni-campaign/../models/Qwen3.5-4B")
            ),
        )
        self.assertEqual(
            "/mnt",
            oc_checkpoint._protected_prefix(
                Path("/home/cereal/omni-campaign/oc01/../../../../mnt/ai_data/models/x")
            ),
        )
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.assert_not_protected(
                Path("/home/cereal/omni-campaign/../models/Qwen3.5-4B"),
                what="--directory",
                stage="fold",
            )
        # the share is refused even for a parent, spelled any way
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.assert_not_protected(
                Path("/home/cereal/omni-campaign/oc01/../../../../mnt/ai_data/x"),
                what="--directory",
                stage="parent",
            )
        # resolution is not a string trick in the other direction either: `..`
        # that lands back inside the campaign root is still a free path.
        self.assertIsNone(
            oc_checkpoint._protected_prefix(
                Path("/home/cereal/models/../omni-campaign/oc01/x")
            )
        )

    def test_a_symlink_into_a_protected_root_is_refused(self) -> None:
        """A campaign-looking path whose real location is the model library.

        `.absolute()` does not follow a symlink, so `<root>/library ->
        /home/cereal/models` read as an ordinary campaign directory: the guard
        passed it, and the same link pointed at `/mnt/ai_data` would have had
        `verify --deep` byte-hashing the CIFS share.  What the registry is about
        to walk -- and may one day unlink -- is the link's TARGET, so the target
        is what the guard has to be shown.
        """

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        try:
            os.symlink(
                "/home/cereal/models", self.root / "library", target_is_directory=True
            )
            os.symlink("/mnt/ai_data", self.root / "share", target_is_directory=True)
        except (OSError, NotImplementedError):  # pragma: no cover - needs privilege
            self.skipTest("no symlink privilege here")
        self.assertEqual(
            "/home/cereal/models",
            oc_checkpoint._protected_prefix(self.root / "library" / "Qwen3.5-4B"),
        )
        self.assertEqual(
            "/mnt",
            oc_checkpoint._protected_prefix(self.root / "share" / "models" / "x"),
        )
        # never point these at a directory that EXISTS: on the server the old
        # code would have digested a real 4B parent on the way to the refusal.
        for where in (self.root / "library" / "OC-Q", self.root / "share" / "OC-Q"):
            self.assertEqual(
                3,
                self.run_checkpoint(
                    "register", "--checkpoint-id", "OC01-F-Q35-4B-LINK1",
                    "--stage", "fold", "--directory", str(where),
                    "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                    "--recipe-command", "ocq.sh LINK1", "--confirm",
                ),
                where,
            )
        self.assertFalse(oc_checkpoint.history(self.registry, "OC01-F-Q35-4B-LINK1"))

    def test_the_record_seals_the_resolved_directory(self) -> None:
        """A record names a LOCATION, not the spelling someone typed to reach it.

        `delete` clause (iv-c) re-reads `checkpoint["directory"]` off a record
        sealed long ago and is the last stop before `rmtree`; a `..` or a
        symlink alias sealed verbatim would have that clause re-run against the
        same blind spelling that got the path registered in the first place.
        """

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(
            self.root / "children", "OC01-F-Q35-4B-DOTS1", {"w": [8, 8]}
        )
        self.assertEqual(
            0,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-DOTS1", "--stage", "fold",
                "--directory",
                str(self.root / "children" / ".." / "children" / child.name),
                "--parent-directory",
                str(self.root / "parents" / ".." / "parents" / "Q35-4B"),
                "--parent-code", "Q35-4B", "--component", "w:4096:64:w",
                "--recipe-command", "ocq.sh DOTS1", "--confirm",
            ),
        )
        checkpoint = oc_checkpoint.current(
            self.registry, "OC01-F-Q35-4B-DOTS1"
        )["checkpoint"]
        self.assertNotIn("..", Path(checkpoint["directory"]).parts)
        self.assertNotIn("..", Path(checkpoint["parent_directory"]).parts)
        self.assertEqual(str(oc_checkpoint.resolve_path(child)), checkpoint["directory"])
        self.assertEqual(
            str(oc_checkpoint.resolve_path(parent)), checkpoint["parent_directory"]
        )

    def test_a_dotdot_spelling_is_not_a_second_directory(self) -> None:
        """Two ids on one directory: a DROP of either deletes the other's bytes.

        `_assert_directory_is_free` compared unresolved paths, so one checkpoint
        registered as `children/X` and again as `children/../children/X` was two
        live records over one set of weights -- exactly the state that check
        exists to refuse, reached with no symlink and no privilege at all.
        """

        first = self.register_child("OC01-F-Q35-4B-SPELL1")
        parent = self.root / "parents" / "Q35-4B"
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-SPELL2", "--stage", "fold",
                "--directory", str(first.parent / ".." / first.parent.name / first.name),
                "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                "--component", "w:4096:256:w", "--recipe-command", "x", "--confirm",
            ),
        )
        # and the nesting arm: `<child>/sub/..` IS the registered child
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-SPELL3", "--stage", "fold",
                "--directory", str(first / "sub" / ".."),
                "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                "--component", "w:4096:256:w", "--recipe-command", "x", "--confirm",
            ),
        )

    def test_a_symlink_alias_is_not_a_second_directory(self) -> None:
        """The same weights reached through an alias link were two checkpoints.

        `register --directory <root>/alias/X` after `<root>/children/X` was
        already registered used to be accepted, because `.absolute()` never
        looked through `alias`.  A DROP of either id would then have `rm -rf`'d
        the other id's weights with the registry reporting both as live.
        """

        first = self.register_child("OC01-F-Q35-4B-ALIAS1")
        parent = self.root / "parents" / "Q35-4B"
        try:
            os.symlink(first.parent, self.root / "alias", target_is_directory=True)
        except (OSError, NotImplementedError):  # pragma: no cover - needs privilege
            self.skipTest("no symlink privilege here")
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-ALIAS2", "--stage", "fold",
                "--directory", str(self.root / "alias" / first.name),
                "--parent-directory", str(parent), "--parent-code", "Q35-4B",
                "--component", "w:4096:256:w", "--recipe-command", "x", "--confirm",
            ),
        )
        self.assertFalse(oc_checkpoint.history(self.registry, "OC01-F-Q35-4B-ALIAS2"))

    def test_an_id_with_no_tag_is_refused(self) -> None:
        """`OC01-F-Q35-4B` used to parse as parent `Q35` / tag `4B`."""

        parent = make_checkpoint_dir(self.root / "parents", "Q35-4B", {"w": [64, 64]})
        child = make_checkpoint_dir(self.root / "children", "untagged", {"w": [8, 8]})
        for extra in ((), ("--parent-code", "Q35-4B")):
            self.assertEqual(
                3,
                self.run_checkpoint(
                    "register", "--checkpoint-id", "OC01-F-Q35-4B", "--stage", "fold",
                    "--directory", str(child), "--parent-directory", str(parent),
                    "--component", "w:4096:64:w", "--recipe-command", "x",
                    "--confirm", *extra,
                ),
            )
        self.assertEqual(
            {"parent_code": "Q35-4B", "tag": "X1", "wave": "oc01", "stage_letter": "F"},
            oc_checkpoint.parse_checkpoint_id("OC01-F-Q35-4B-X1", "fold"),
        )

    def test_two_checkpoints_may_not_share_one_directory(self) -> None:
        """A DROP of one would rm -rf the other's bytes (R4 iv-b sees links only)."""

        first = self.register_child("OC01-F-Q35-4B-SHARED1")
        parent = self.root / "parents" / "Q35-4B"
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-SHARED2", "--stage", "fold",
                "--directory", str(first), "--parent-directory", str(parent),
                "--component", "w:4096:256:w", "--recipe-command", "x", "--confirm",
            ),
        )
        # and nesting is refused too: deleting the outer takes the inner
        inner = make_checkpoint_dir(first, "nested", {"w": [4, 4]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-SHARED3", "--stage", "fold",
                "--directory", str(inner), "--parent-directory", str(parent),
                "--component", "w:4096:16:w", "--recipe-command", "x", "--confirm",
            ),
        )

    def test_a_clone_may_not_be_registered_by_hand(self) -> None:
        """R2 was enforced by reading a STRING off the parent record.

        `register --stage clone` sealed status=cloned with no byte copy, no
        digest assertion against a sealed source and no clone receipt, and a
        stage-D child could then name it as its parent and pass.
        """

        self.register_child("OC01-F-Q35-4B-C2S4L03")
        parent = self.root / "parents" / "Q35-4B"
        fake = make_checkpoint_dir(self.root / "children", "fake-clone", {"w": [16, 16]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-C2S4L03.clone-pre-D1",
                "--stage", "clone", "--directory", str(fake),
                "--parent-directory", str(parent), "--recipe-command", "x", "--confirm",
            ),
        )
        self.assertEqual(
            [], oc_checkpoint.history(self.registry, "OC01-F-Q35-4B-C2S4L03.clone-pre-D1")
        )

    def test_a_registered_parent_needs_its_provenance_digest(self) -> None:
        directory = make_checkpoint_dir(self.root / "parents", "Q35-9B", {"w": [64, 64]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "PARENT-Q35-9B", "--stage", "parent",
                "--directory", str(directory), "--parent-directory", str(directory),
                "--recipe-command", "download", "--confirm",
            ),
        )
        self.assertEqual(
            0,
            self.run_checkpoint(
                "register", "--checkpoint-id", "PARENT-Q35-9B", "--stage", "parent",
                "--directory", str(directory), "--parent-directory", str(directory),
                "--parent-provenance-sha256", "e" * 64,
                "--recipe-command", "download", "--confirm",
            ),
        )

    def test_a_second_register_is_refused(self) -> None:
        self.register_child()
        parent = self.root / "parents" / "Q35-4B"
        child = self.root / "children" / "OC01-F-Q35-4B-C2S4L03"
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-C2S4L03", "--stage", "fold",
                "--directory", str(child), "--parent-directory", str(parent),
                "--recipe-command", "x", "--confirm",
            ),
        )

    def test_training_stage_must_come_from_a_clone(self) -> None:
        self.register_child()
        parent = self.root / "parents" / "Q35-4B"
        child = make_checkpoint_dir(self.root / "children", "OC01-D-Q35-4B-C2S4L03", {"w": [16, 16]})
        self.assertEqual(
            3,
            self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-D-Q35-4B-C2S4L03", "--stage", "distill",
                "--directory", str(child), "--parent-directory", str(parent),
                "--parent-checkpoint-id", "OC01-F-Q35-4B-C2S4L03",
                "--recipe-command", "x", "--confirm",
            ),
        )


class TestReview(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cid = "OC01-F-Q35-4B-C2S4L03"
        self.register_child(self.cid)

    def _review(self, payload: dict, *extra: str) -> int:
        path = self.root / "rev.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return self.run_checkpoint(
            "record-review", "--checkpoint-id", self.cid, "--review-json", str(path),
            "--confirm", *extra,
        )

    def test_signal_makes_it_partially_capable(self) -> None:
        self.assertEqual(0, self._review(review_payload(signal=1, artifact=9, content=5)))
        record = oc_checkpoint.current(self.registry, self.cid)
        self.assertEqual("partially_capable", record["status"])
        self.assertEqual(2, record["record_seq"])
        self.assertEqual(2, len(oc_ledger.read_rows(self.ledger)))

    def test_no_signal_anywhere_makes_it_bad(self) -> None:
        self.assertEqual(
            0, self._review(review_payload(useful=0, partial=0, signal=0, artifact=9, content=6))
        )
        self.assertEqual("bad", oc_checkpoint.current(self.registry, self.cid)["status"])

    def test_all_artifact_misses_is_unmeasured_and_never_bad(self) -> None:
        """The estate's rule, structural: an ARTIFACT may not condemn a child.

        Zero signal with fifteen artifact misses (pure loop, cap-ride, missing
        EOS) used to seal as `bad` -- the one deletable status -- and then pass
        all twelve R4 clauses.  A child nobody managed to measure is
        `unmeasured`: it goes back for a re-run, and no path can delete it.
        """

        self.assertEqual(
            0,
            self._review(
                review_payload(useful=0, partial=0, signal=0, artifact=15, content=0)
            ),
        )
        record = oc_checkpoint.current(self.registry, self.cid)
        self.assertEqual("unmeasured", record["status"])
        self.assertIn("artifact-miss", record["status_reason"])
        # one content miss and it IS a grade
        self.assertEqual(
            0,
            self._review(
                review_payload(useful=0, partial=0, signal=0, artifact=14, content=1)
            ),
        )
        self.assertEqual("bad", oc_checkpoint.current(self.registry, self.cid)["status"])

    def test_an_empty_media_artifact_is_an_artifact_miss(self) -> None:
        self.assertEqual(
            0,
            self._review(
                review_payload(
                    useful=0, partial=0, signal=0, artifact=15, content=0,
                    media=[{"prompt_id": "av-gen-kettle-A", "non_empty": False,
                            "cyril_score": None}],
                )
            ),
        )
        self.assertEqual(
            "unmeasured", oc_checkpoint.current(self.registry, self.cid)["status"]
        )

    def test_a_malformed_review_is_a_refusal_not_a_traceback(self) -> None:
        payload = review_payload()
        del payload["bank1"]["useful"]
        self.assertEqual(3, self._review(payload))
        payload = review_payload()
        del payload["columns"]["column_a"]
        self.assertEqual(3, self._review(payload))

    def test_a_silent_wav_is_not_a_signal_lane(self) -> None:
        """R5's wording is non-zero CONTENT; `non_empty` is only st_size > 0.

        A 44-byte WAV header, or a full-length track of pure silence, read as
        intelligence signal, flipped the child to `partially_capable` and
        blocked its deletion.  When the runner measured an `audio_rms` -- and
        `oc_package.media_diagnostics` computes one for every WAV -- silence may
        not also claim to be non-empty.
        """

        silent = [{"prompt_id": "av-A", "non_empty": True, "audio_rms": 0.0,
                   "cyril_score": None}]
        self.assertEqual(3, self.review(self.cid, review_payload(media=silent)))
        heard = [{"prompt_id": "av-A", "non_empty": True, "audio_rms": 0.31,
                  "cyril_score": None}]
        self.assertEqual(0, self.review(self.cid, review_payload(media=heard)))

    def test_a_bank_with_no_intelligence_lanes_is_refused(self) -> None:
        """`lanes_total: 1` made intelligence 0, so every count summed to 0.

        Every consistency check passed and `status_for` fell through to `bad`
        with zero content misses -- delete was still refused by clause (i-e),
        but the scorecard labelled an unmeasured child `bad`.
        """

        payload = review_payload()
        payload["bank1"].update(
            {"lanes_total": 1, "useful": 0, "partial": 0, "observed_miss": 0,
             "canonical_signal": 0, "canonical_artifact_miss": 0,
             "canonical_content_miss": 0}
        )
        payload["columns"] = {"column_a": 0, "column_b": 0}
        self.assertEqual(3, self.review(self.cid, payload))

    def test_a_bank2_that_is_not_an_object_is_a_refusal(self) -> None:
        payload = review_payload()
        payload["bank2"] = []
        self.assertEqual(3, self.review(self.cid, payload))

    def test_a_package_path_needs_its_digest(self) -> None:
        self.assertEqual(3, self._review(review_payload(package="/pkg")))
        self.assertEqual(
            0, self._review(review_payload(package="/pkg", package_sha256="c" * 64))
        )

    def test_media_with_content_alone_is_partially_capable(self) -> None:
        payload = review_payload(
            useful=0, partial=0, signal=0, artifact=9, content=6,
            media=[{"prompt_id": "av-gen-kettle-A", "non_empty": True, "cyril_score": None}],
        )
        self.assertEqual(0, self._review(payload))
        self.assertEqual(
            "partially_capable", oc_checkpoint.current(self.registry, self.cid)["status"]
        )

    def test_status_may_only_assert_agreement(self) -> None:
        self.assertEqual(
            3, self._review(review_payload(signal=1, artifact=9, content=5), "--status", "bad")
        )

    def test_counts_that_do_not_add_up_are_refused(self) -> None:
        payload = review_payload()
        payload["bank1"]["observed_miss"] = 99
        self.assertEqual(3, self._review(payload))
        payload = review_payload()
        payload["bank1"]["canonical_signal"] = 14
        self.assertEqual(3, self._review(payload))

    def test_column_a_must_equal_the_sealed_count(self) -> None:
        payload = review_payload()
        payload["columns"]["column_a"] = 99
        self.assertEqual(3, self._review(payload))

    def test_column_b_may_not_exceed_a(self) -> None:
        payload = review_payload(useful=2, partial=3, column_b=9)
        self.assertEqual(3, self._review(payload))

    def test_parent_band_has_no_default(self) -> None:
        payload = review_payload()
        del payload["bank1"]["parent_useful_or_partial"]
        self.assertEqual(3, self._review(payload))


class TestClone(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cid = "OC01-F-Q35-4B-C2S4L03"
        self.child_dir = self.register_child(self.cid)

    def _review(self, **kwargs: object) -> None:
        path = self.root / "rev.json"
        path.write_text(json.dumps(review_payload(**kwargs)), encoding="utf-8")  # type: ignore[arg-type]
        self.assertEqual(
            0,
            self.run_checkpoint(
                "record-review", "--checkpoint-id", self.cid,
                "--review-json", str(path), "--confirm",
            ),
        )

    def test_clone_of_an_unreviewed_source_is_refused(self) -> None:
        self.assertEqual(
            3,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D", "--confirm"
            ),
        )

    def test_clone_dry_run_then_confirm(self) -> None:
        self._review()
        self.assertEqual(
            0, self.run_checkpoint("clone", "--source", self.cid, "--stage-about-to-run", "D")
        )
        clone_id = f"{self.cid}.clone-pre-D1"
        # the default destination is the campaign's own clones/, NOT beside the
        # source: a registered parent lives in the model library and a clone of
        # it must not drop a full-size copy there.
        clone_dir = self.root / "clones" / clone_id
        self.assertFalse(clone_dir.exists())

        self.assertEqual(
            0,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D", "--confirm"
            ),
        )
        self.assertTrue(clone_dir.is_dir())
        record = oc_checkpoint.current(self.registry, clone_id)
        self.assertEqual("cloned", record["status"])
        self.assertEqual("clone", record["checkpoint"]["stage"])
        self.assertEqual(
            oc_seal.directory_digest(self.child_dir)[0],
            record["checkpoint"]["directory_sha256"],
        )
        receipt = oc_seal.read_sealed(
            self.root / "clones" / f"{clone_id}.clone-receipt.json"
        )
        self.assertEqual(receipt["source_directory_sha256"], receipt["clone_directory_sha256"])
        self.assertEqual("D", receipt["stage_about_to_run"])

    def test_a_trained_child_needs_a_real_clone_receipt(self) -> None:
        """The clone path end to end, and then the receipt taken away."""

        self._review()
        clone_id = f"{self.cid}.clone-pre-D1"
        self.assertEqual(
            0,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D", "--confirm"
            ),
        )
        trained = make_checkpoint_dir(
            self.root / "children", "OC01-D-Q35-4B-C2S4L03", {"w": [16, 16]}
        )
        argv = [
            "register", "--checkpoint-id", "OC01-D-Q35-4B-C2S4L03", "--stage", "distill",
            "--directory", str(trained),
            "--parent-directory", str(self.root / "parents" / "Q35-4B"),
            "--parent-code", "Q35-4B", "--parent-checkpoint-id", clone_id,
            "--recipe-command", "distill.sh", "--confirm",
        ]
        self.assertEqual(0, self.run_checkpoint(*argv))

        # and with the sealed receipt gone, the same registration is refused:
        # the clone's own evidence is what R2 rests on, not the word "clone"
        receipt = self.root / "clones" / f"{clone_id}.clone-receipt.json"
        receipt.unlink()
        second = make_checkpoint_dir(
            self.root / "children", "OC01-D-Q35-4B-SECOND", {"w": [16, 16]}
        )
        argv[argv.index("OC01-D-Q35-4B-C2S4L03")] = "OC01-D-Q35-4B-SECOND"
        argv[argv.index(str(trained))] = str(second)
        self.assertEqual(3, self.run_checkpoint(*argv))

    def test_clone_refuses_when_the_source_bytes_moved(self) -> None:
        self._review()
        (self.child_dir / "config.json").write_text('{"model_type":"moved"}', encoding="utf-8")
        self.assertEqual(
            3,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "T", "--confirm"
            ),
        )

    def test_clone_refuses_a_protected_destination(self) -> None:
        """A clone is a working checkpoint a trainer overwrites in place.

        `--stage parent` is not an escape here: a clone is never a parent, so
        the model library is refused as well as the share.
        """

        self.review(self.cid, review_payload(signal=1, artifact=9, content=5))
        for where in ("/mnt/ai_data/clones/x", "/home/cereal/models/clone-x"):
            self.assertEqual(
                3,
                self.run_checkpoint(
                    "clone", "--source", self.cid, "--stage-about-to-run", "D",
                    "--dest", where, "--confirm",
                ),
                where,
            )

    def test_clone_refuses_a_destination_that_resolves_into_a_protected_root(self) -> None:
        """The destination is where the bytes LAND, not what was typed.

        `--dest $OC_ROOT/../models/x`, or a `clones` symlink whose target is the
        model library, dropped a full-size working copy into the source model
        library while the guard reported a clean campaign path.  Checked as a
        DRY RUN, because the refusal has to happen before `print_plan`: after
        `--confirm` the very next thing the command does is `mkdir` there.
        """

        self._review()
        self.assertEqual(
            3,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D",
                "--dest", "/home/cereal/omni-campaign/../models/clone-x",
            ),
        )
        try:
            os.symlink(
                "/mnt/ai_data/clones", self.root / "elsewhere", target_is_directory=True
            )
        except (OSError, NotImplementedError):  # pragma: no cover - needs privilege
            return
        self.assertEqual(
            3,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D",
                "--dest", str(self.root / "elsewhere" / "clone-x"),
            ),
        )

    def test_hardlink_mode_needs_the_explicit_flag(self) -> None:
        self._review()
        self.assertEqual(
            3,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D",
                "--mode", "hardlink", "--confirm",
            ),
        )


class TestDelete(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cid = "OC01-F-Q35-4B-DEAD1"
        self.child_dir = self.register_child(self.cid)
        self.package = self.root / "package"

    def _review_bad(self, **kwargs: object) -> None:
        payload = review_payload(
            useful=0, partial=0, signal=0, artifact=9, content=6,
            **kwargs,  # type: ignore[arg-type]
        )
        self.assertEqual(0, self.review(self.cid, payload))

    def _build_package(self) -> None:
        """The honest R4(v) order: review, package the review, then record the
        package's own digest on a NEW review record (records are append-only,
        so it supersedes without editing anything)."""

        self._review_bad()
        manifest = self.build_package(self.cid, self.package)
        self._review_bad(
            package=str(self.package), package_sha256=manifest["package_sha256"]
        )

    def _delete(self, *extra: str) -> int:
        return self.run_checkpoint(
            "delete", "--checkpoint-id", self.cid,
            "--reason",
            "0/15 sealed and 0/15 canonical at 0.0625x; every character a reader "
            "needs is in the committed dossier and the checkpoint adds nothing "
            "to it.",
            "--rebuild-command", f"ocq.sh {self.cid}",
            "--retained-source", str(self.root / "parents" / "Q35-4B"),
            *extra,
        )

    def test_delete_refused_when_the_retained_source_does_not_exist(self) -> None:
        self._build_package()
        self.assertEqual(
            4,
            self.run_checkpoint(
                "delete", "--checkpoint-id", self.cid, "--reason", "x" * 80,
                "--rebuild-command", "ocq.sh X",
                "--retained-source", "OC99-F-NOBODY-X", "--confirm",
            ),
        )
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refused_without_a_review(self) -> None:
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refused_while_it_holds_signal(self) -> None:
        path = self.root / "rev.json"
        path.write_text(json.dumps(review_payload(signal=1, artifact=9, content=5)), encoding="utf-8")
        self.run_checkpoint(
            "record-review", "--checkpoint-id", self.cid, "--review-json", str(path), "--confirm"
        )
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refused_without_a_verified_package(self) -> None:
        self._review_bad()
        self.assertEqual(4, self._delete("--confirm"))

    def test_delete_refused_for_a_frontier_holder(self) -> None:
        self._build_package()
        self.assertEqual(
            0,
            self.run_checkpoint(
                "set-status", "--checkpoint-id", self.cid, "--frontier-holder", "1", "--confirm"
            ),
        )
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refused_for_a_clone_source(self) -> None:
        self._build_package()
        self.assertEqual(
            0,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "T",
                "--allow-unreviewed-source", "--confirm",
            ),
        )
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refused_for_a_clone_itself(self) -> None:
        self._build_package()
        self.run_checkpoint(
            "clone", "--source", self.cid, "--stage-about-to-run", "T",
            "--allow-unreviewed-source", "--confirm",
        )
        clone_id = f"{self.cid}.clone-pre-T1"
        code = self.run_checkpoint(
            "delete", "--checkpoint-id", clone_id, "--reason", "x" * 80,
            "--rebuild-command", "x", "--retained-source", self.cid, "--confirm",
        )
        self.assertEqual(4, code)

    def test_delete_refused_when_the_reason_is_thin(self) -> None:
        self._build_package()
        self.assertEqual(
            4,
            self.run_checkpoint(
                "delete", "--checkpoint-id", self.cid, "--reason", "bad",
                "--rebuild-command", "x", "--retained-source", str(self.child_dir),
                "--confirm",
            ),
        )

    def test_delete_dry_run_then_confirm(self) -> None:
        self._build_package()
        self.assertEqual(0, self._delete())          # plan only
        self.assertTrue(self.child_dir.is_dir())
        self.assertEqual(0, self._delete("--confirm"))
        self.assertFalse(self.child_dir.exists())
        record = oc_checkpoint.current(self.registry, self.cid)
        self.assertEqual("delete", record["event"])
        self.assertIn("deleted", record["checkpoint"])
        self.assertEqual(
            f"ocq.sh {self.cid}", record["checkpoint"]["deleted"]["rebuild_command"]
        )
        # the record chain and the ledger still describe it
        self.assertEqual(4, len(oc_checkpoint.history(self.registry, self.cid)))
        self.assertEqual(
            0,
            oc_ledger.main(
                ["verify", "--ledger", str(self.ledger), "--registry", str(self.registry)]
            ),
        )
        # ... in both directions, and no in-flight marker survived
        self.assertEqual(0, self.run_checkpoint("verify"))
        self.assertFalse(
            oc_checkpoint.pending_marker(self.registry, self.cid).exists()
        )

    def test_the_protected_path_clause_is_evaluated_on_every_delete(self) -> None:
        """R6 as clause (iv-c), the last stop before `rmtree`.

        `register` refuses these paths now, but a record sealed before that
        guard existed still names one, so the clause is checked where the unlink
        actually happens and not only at the door.
        """

        self._build_package()
        args = _delete_args(self.cid, self.root)
        clauses = oc_checkpoint._delete_conditions(self.registry, self.cid, args)
        matching = [(ok, text) for ok, text in clauses if text.startswith("(iv-c)")]
        self.assertEqual(1, len(matching))
        self.assertTrue(matching[0][0])          # a campaign path passes
        self.assertEqual(14, len(clauses))       # R4 is fourteen clauses

    def test_delete_refuses_a_record_under_mnt_lexar(self) -> None:
        """Register may now seal a /mnt/lexar directory; delete may still not unlink it.

        The 2026-10-09 narrowing touched the register/clone door only.  A
        record whose sealed directory is on Lexar must fail clause (iv-c) like
        any other `/mnt` path.  The record is substituted in memory and only
        the clause list is evaluated -- no `--confirm` delete is run against a
        `/mnt` path from a test.
        """

        self._build_package()
        real_current = oc_checkpoint.current
        lexar = "/mnt/lexar/omni-campaign/oc-ledger-test-never-exists/DEAD1"

        def on_lexar(registry: Path, checkpoint_id: str):  # type: ignore[no-untyped-def]
            record = json.loads(json.dumps(real_current(registry, checkpoint_id)))
            if checkpoint_id == self.cid:
                record["checkpoint"]["directory"] = lexar
            return record

        args = _delete_args(self.cid, self.root)
        oc_checkpoint.current = on_lexar  # type: ignore[assignment]
        try:
            clauses = oc_checkpoint._delete_conditions(self.registry, self.cid, args)
        finally:
            oc_checkpoint.current = real_current  # type: ignore[assignment]
        matching = [(ok, text) for ok, text in clauses if text.startswith("(iv-c)")]
        self.assertEqual(1, len(matching))
        self.assertFalse(matching[0][0], matching[0][1])
        self.assertIn("'/mnt'", matching[0][1])
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refuses_another_checkpoints_package(self) -> None:
        """R4(v) called `verify_manifest`, which is self-referential.

        It recomputes the package digest from the manifest's own file list, so
        ANY well-formed package verifies -- including one built for a different
        checkpoint.  DEAD1 pointed at GOOD1's package and the clause printed
        PASS.
        """

        good = "OC01-F-Q35-4B-GOOD1"
        self.register_child(good)
        self.assertEqual(0, self.review(good, review_payload(signal=3, artifact=7, content=5)))
        foreign = self.build_package(good, self.root / "package-good")

        self._review_bad(
            package=str(self.root / "package-good"),
            package_sha256=foreign["package_sha256"],
        )
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refuses_a_package_that_changed_after_the_review(self) -> None:
        self._build_package()
        (self.package / "README.md").write_text("edited after review", encoding="utf-8")
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_delete_refuses_an_artifact_only_child(self) -> None:
        """Zero signal, zero content misses: nothing was measured.

        Every clause but this one passed, so `--confirm` would have deleted the
        weights on the strength of fifteen loops.
        """

        self._review_bad()
        manifest = self.build_package(self.cid, self.package)
        self.assertEqual(
            0,
            self.review(
                self.cid,
                review_payload(
                    useful=0, partial=0, signal=0, artifact=15, content=0,
                    package=str(self.package),
                    package_sha256=manifest["package_sha256"],
                ),
            ),
        )
        self.assertEqual(
            "unmeasured", oc_checkpoint.current(self.registry, self.cid)["status"]
        )
        self.assertEqual(4, self._delete("--confirm"))
        self.assertTrue(self.child_dir.is_dir())

    def test_an_interrupted_delete_does_not_poison_the_chain(self) -> None:
        """The marker used to live in the record directory, and `glob('*.json')`
        matches dotfiles: one interrupted delete made `history()` raise for the
        WHOLE registry -- dashboard, list, verify and package with it."""

        self._build_package()
        marker = oc_checkpoint.pending_marker(self.registry, self.cid)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"interrupted": True}), encoding="utf-8")
        # a stray non-record file inside the record directory too
        stray = oc_checkpoint.record_dir(self.registry, self.cid) / ".pending-delete.json"
        stray.write_text(json.dumps({"not": "sealed"}), encoding="utf-8")

        self.assertEqual(3, len(oc_checkpoint.history(self.registry, self.cid)))
        self.assertEqual([self.cid], sorted(oc_checkpoint.all_current(self.registry)))
        self.assertEqual(
            0,
            oc_dashboard.main(
                ["--registry", str(self.registry), "--ledger", str(self.ledger),
                 "--out", str(self.root / "S.md"), "--queue", str(self.root / "Q.md")]
            ),
        )
        # verify still runs, and REPORTS the interrupted delete rather than dying
        self.assertEqual(1, self.run_checkpoint("verify"))


# --------------------------------------------------------------------------
# 4. package and dashboard
# --------------------------------------------------------------------------


class TestChainRace(Base):
    def test_two_writers_cannot_both_take_the_next_sequence(self) -> None:
        """Two mutating commands reading the same head both wrote `-0002-`.

        Different event names, so neither file collided, both succeeded, and
        `history()` raised "record sequence [1,2,2] has a gap or a duplicate"
        from then on -- unrecoverable without hand-deleting a sealed record.
        The head is now re-read under the chain lock and the loser is refused.
        """

        cid = "OC01-F-Q35-4B-C2S4L03"
        self.register_child(cid)
        head = oc_checkpoint.current(self.registry, cid)
        stale = oc_checkpoint.build_record(
            checkpoint_id=cid, seq=2, event="status",
            supersedes=head["receipt_sha"], checkpoint=head["checkpoint"],
            review=None, status="candidate", status_reason="-", notes="",
            created_at=oc_seal.now_utc(),
        )
        # the other writer gets there first
        self.assertEqual(0, self.review(cid, review_payload()))
        with self.assertRaises(oc_checkpoint.OcCheckpointError):
            oc_checkpoint.commit_record(
                registry=self.registry, ledger=self.ledger, record=stale
            )
        self.assertEqual(2, len(oc_checkpoint.history(self.registry, cid)))
        self.assertEqual(0, self.run_checkpoint("verify"))


class TestPackage(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cid = "OC01-F-Q35-4B-C2S4L03"
        self.register_child(self.cid)
        self.out = self.root / "human-review" / self.cid

    def _dossier(self) -> Path:
        path = self.root / "dossier.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "neural-forge-lfm2-matrix-composition-frozen-review-dossier-v1",
                    "cell_id": "ARCH.C2S4L03",
                    "dossier_sha256": "f" * 64,
                    "standard_shape_child": {"net_ratio": 0.0625, "config_diff": [{"field": "x"}]},
                    "child_responses": [
                        {
                            "id": "summarization-priorities",
                            "category": "summarization",
                            "prompt_messages": [{"role": "user", "content": "Summarise."}],
                            "response_verbatim": "Supplier B was chosen. The one-week "
                                                 "expansion risk is unresolved.",
                            "response_sha256": "1" * 64,
                            "generated_tokens": 21,
                            "termination": "complete",
                        },
                        {
                            "id": "arithmetic-multi-step",
                            "category": "arithmetic",
                            "prompt_messages": [{"role": "user", "content": "102 kits."}],
                            "response_verbatim": "loop loop loop loop",
                            "response_sha256": "2" * 64,
                            "generated_tokens": 1024,
                            "termination": "censored-at-cap",
                        },
                    ],
                    "parent_responses": [
                        {
                            "id": "summarization-priorities",
                            "response_verbatim": "Supplier B; expansion risk open.",
                            "generated_tokens": 12,
                            "termination": "complete",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_package_builds_and_verifies(self) -> None:
        dossier = self._dossier()
        per_case = self.root / "per-case.json"
        per_case.write_text(
            json.dumps(
                {
                    "summarization-priorities": "Partial",
                    "arithmetic-multi-step": "Observed miss",
                }
            ),
            encoding="utf-8",
        )
        # TWO cells, and the WRONG one first.  The wave-level re-read file is
        # keyed by cell id, which never equals a checkpoint id, so the old
        # `next(iter(payload))` fallback fired every time and would print
        # ARCH.OTHER1's boxes as this checkpoint's.  With one cell in the
        # fixture that defect passed the test; with two it does not.
        canonical = self.root / "canon.json"
        canonical.write_text(
            json.dumps(
                {
                    "ARCH.OTHER1": {
                        "summarization-priorities": "content-miss",
                        "arithmetic-multi-step": "content-miss",
                    },
                    "ARCH.C2S4L03": {
                        "summarization-priorities": "signal",
                        "arithmetic-multi-step": "artifact-miss",
                    },
                }
            ),
            encoding="utf-8",
        )
        media = self.root / "media"
        media.mkdir()
        (media / "av-gen-kettle-A.wav").write_bytes(b"")  # empty: a real diagnostic
        code = oc_package.main(
            ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
             "--checkpoint-id", self.cid, "--out", str(self.out),
             "--dossier", str(dossier), "--per-case", str(per_case),
             "--canonical-reread", str(canonical), "--media", str(media)]
        )
        self.assertEqual(0, code)
        for name in (
            "README.md", "MANIFEST.json", f"REVIEW_C2S4L03.md",
            "READ-C2S4L03.txt", "READ-PARENT-C2S4L03.txt",
            "dossier-C2S4L03.json", "record-C2S4L03.json",
            "ledger-rows-C2S4L03.jsonl", "media/av-gen-kettle-A.wav",
            "media/media-diagnostics.json",
        ):
            self.assertTrue((self.out / name).exists(), name)
        review = (self.out / "REVIEW_C2S4L03.md").read_text(encoding="utf-8")
        self.assertIn("Supplier B was chosen", review)          # child verbatim
        self.assertIn("Supplier B; expansion risk open.", review)  # parent verbatim
        self.assertIn("| summarization-priorities | Partial | signal |", review)
        # the OTHER cell's boxes must not appear anywhere, and the document
        # must name the block it read so a reader can check the attribution.
        self.assertNotIn("content-miss", review)
        self.assertIn("read from block `ARCH.C2S4L03`", review)
        self.assertIn("ANNOTATIONS", review)
        diagnostics = json.loads(
            (self.out / "media" / "media-diagnostics.json").read_text(encoding="utf-8")
        )
        self.assertFalse(diagnostics["artifacts"][0]["non_empty"])
        self.assertEqual([], oc_package.verify_manifest(self.out))
        self.assertEqual(0, oc_package.main(["verify", "--package", str(self.out)]))

    def test_an_unmatched_canonical_file_is_marked_unread_not_guessed(self) -> None:
        """No block for this checkpoint means the column is UNREAD.

        The old code returned the first dict-valued key in the file, so a
        wave-level re-read that simply did not cover this cell printed some
        OTHER cell's three-box readings in the column Cyril decides on.
        """

        dossier = self._dossier()
        canonical = self.root / "canon-nomatch.json"
        canonical.write_text(
            json.dumps(
                {
                    "ARCH.OTHER1": {"summarization-priorities": "content-miss"},
                    "ARCH.OTHER2": {"summarization-priorities": "signal"},
                }
            ),
            encoding="utf-8",
        )
        code = oc_package.main(
            ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
             "--checkpoint-id", self.cid, "--out", str(self.out),
             "--dossier", str(dossier), "--canonical-reread", str(canonical)]
        )
        self.assertEqual(0, code)
        review = (self.out / "REVIEW_C2S4L03.md").read_text(encoding="utf-8")
        self.assertIn("names no block for this checkpoint", review)
        self.assertIn("UNREAD", review)
        self.assertNotIn("content-miss", review)
        self.assertIn("| summarization-priorities | - | - |", review)

    def test_canonical_lanes_never_guesses(self) -> None:
        payload = {
            "ARCH.OTHER1": {"lane-a": "content-miss"},
            "ARCH.C2S4L03": {"lane-a": "signal"},
        }
        self.assertEqual(
            ({"lane-a": "signal"}, "ARCH.C2S4L03"),
            oc_package._canonical_lanes(
                payload, ["OC01-F-Q35-4B-C2S4L03", "C2S4L03", "ARCH.C2S4L03"]
            ),
        )
        # no candidate matches -> empty and unsourced, NOT ARCH.OTHER1
        self.assertEqual(
            ({}, None),
            oc_package._canonical_lanes(payload, ["OC02-F-X", "X", None]),
        )
        # the flat single-cell form is still accepted and is self-identifying
        self.assertEqual(
            ({"lane-a": "signal"}, "<flat>"),
            oc_package._canonical_lanes({"lane-a": "signal"}, ["anything"]),
        )

    def test_a_missing_input_is_a_refusal_not_a_traceback(self) -> None:
        """`shutil.copy2` on a path that is not there raised FileNotFoundError.

        `main` does not catch it, so a mistyped `--dossier` came out of the
        build as a traceback rather than `REFUSED: <why>` and exit 3 -- and it
        did so after part of the package had already been written.
        """

        self.assertEqual(
            3,
            oc_package.main(
                ["build", "--registry", str(self.registry),
                 "--ledger", str(self.ledger), "--checkpoint-id", self.cid,
                 "--out", str(self.root / "pkg-missing"),
                 "--dossier", str(self.root / "nope.json")]
            ),
        )

    def test_package_verify_catches_a_changed_file(self) -> None:
        oc_package.main(
            ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
             "--checkpoint-id", self.cid, "--out", str(self.out)]
        )
        (self.out / "README.md").write_text("edited", encoding="utf-8")
        problems = oc_package.verify_manifest(self.out)
        self.assertTrue(any("digest mismatch" in p for p in problems))
        self.assertEqual(1, oc_package.main(["verify", "--package", str(self.out)]))

    def test_package_verify_catches_an_added_file(self) -> None:
        oc_package.main(
            ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
             "--checkpoint-id", self.cid, "--out", str(self.out)]
        )
        (self.out / "smuggled.txt").write_text("hi", encoding="utf-8")
        self.assertTrue(any("undeclared" in p for p in oc_package.verify_manifest(self.out)))

    def test_building_into_a_used_folder_is_refused(self) -> None:
        """A second build used to overwrite in place and let stale files join
        the new manifest -- which changes `package_sha256`, the number a review
        commits to."""

        self.build_package(self.cid, self.out)
        first = json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(
            3,
            oc_package.main(
                ["build", "--registry", str(self.registry), "--ledger", str(self.ledger),
                 "--checkpoint-id", self.cid, "--out", str(self.out)]
            ),
        )
        self.assertEqual(
            first,
            json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8")),
        )
        # --overwrite MOVES the old package aside; it never writes over it
        self.build_package(self.cid, self.out, "--overwrite")
        superseded = sorted(self.out.parent.glob(f"{self.out.name}.superseded-*"))
        self.assertEqual(1, len(superseded))
        self.assertEqual([], oc_package.verify_manifest(superseded[0]))

    def test_media_scores_join_on_the_filename_too(self) -> None:
        media = self.root / "media"
        media.mkdir()
        (media / "av-gen-kettle-A.take3.wav").write_bytes(b"RIFFnope")
        self.assertEqual(
            0,
            self.review(
                self.cid,
                review_payload(
                    signal=1, artifact=9, content=5,
                    media=[{"prompt_id": "av-gen-kettle-A", "non_empty": True,
                            "filename": "av-gen-kettle-A.take3.wav", "cyril_score": 4}],
                ),
            ),
        )
        self.build_package(self.cid, self.out, "--media", str(media))
        review = (self.out / "REVIEW_C2S4L03.md").read_text(encoding="utf-8")
        self.assertIn("| 4 |", review)
        self.assertNotIn("No `media_out` entry matched", review)

    def test_wav_diagnostics_measure_a_real_tone(self) -> None:
        import wave as wavemod

        path = self.root / "tone.wav"
        with wavemod.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(struct.pack("<8000h", *([8000] * 8000)))
        diagnostics = oc_package.media_diagnostics(path)
        self.assertTrue(diagnostics["non_empty"])
        self.assertAlmostEqual(1.0, diagnostics["duration_s"], places=3)
        self.assertGreater(diagnostics["audio_rms"], 0.0)


class TestDashboard(Base):
    def test_dashboard_renders_every_stage_and_the_queue(self) -> None:
        cid = "OC01-F-Q35-4B-C2S4L03"
        self.register_child(cid)
        self.assertEqual(
            0,
            self.review(
                cid,
                review_payload(
                    signal=2, artifact=8, content=5,
                    package="/pkg", package_sha256="d" * 64,
                ),
            ),
        )
        dead = "OC01-F-Q35-4B-DEAD1"
        self.register_child(dead)
        self.assertEqual(
            0,
            self.review(
                dead,
                review_payload(useful=0, partial=0, signal=0, artifact=9, content=6),
            ),
        )
        lost = "OC01-F-Q35-4B-LOOP1"
        self.register_child(lost)
        self.assertEqual(
            0,
            self.review(
                lost,
                review_payload(useful=0, partial=0, signal=0, artifact=15, content=0),
            ),
        )
        out, queue = self.root / "SCORECARD.md", self.root / "REVIEW_QUEUE.md"
        code = oc_dashboard.main(
            ["--registry", str(self.registry), "--ledger", str(self.ledger),
             "--out", str(out), "--queue", str(queue)]
        )
        self.assertEqual(0, code)
        scorecard = out.read_text(encoding="utf-8")
        self.assertIn("## Stage: fold", scorecard)
        self.assertIn(cid, scorecard)
        self.assertIn(oc_seal.file_sha256(self.ledger), scorecard)
        text = queue.read_text(encoding="utf-8")
        self.assertIn("## For Cyril (1)", text)
        self.assertIn(cid, text.split("## Unmeasured")[0])
        self.assertIn(lost, text.split("## Unmeasured")[1].split("## Bad")[0])
        self.assertIn(dead, text.split("## Bad")[1])
        # the loop-only child is NOT in the retention section
        self.assertNotIn(lost, text.split("## Bad")[1])

    def test_the_scorecard_says_whether_the_ratio_was_measured(self) -> None:
        """A ratio column that does not say where its terms came from is a claim
        printed as a measurement."""

        self.register_child()
        rows = oc_dashboard.rows_for(oc_checkpoint.all_current(self.registry))
        text = oc_dashboard.scorecard_markdown(
            rows, ledger=self.ledger, ledger_rows=1
        )
        self.assertIn("ratio from", text)
        self.assertIn("| headers |", text)
        self.assertNotIn("**asserted**", text.split("Legend:")[0])

    def test_ordering_is_intelligence_per_compression(self) -> None:
        shallow, deep = "OC01-F-Q35-4B-SHALLOW", "OC01-F-Q35-4B-DEEP"
        self.register_child(shallow, tensors={"w": [40, 40]})   # 1600/4096
        self.register_child(deep, tensors={"w": [10, 10]})      # 100/4096
        path = self.root / "rev.json"
        for cid, signal in ((shallow, 5), (deep, 2)):
            path.write_text(
                json.dumps(review_payload(signal=signal, artifact=10 - signal, content=5)),
                encoding="utf-8",
            )
            self.run_checkpoint(
                "record-review", "--checkpoint-id", cid, "--review-json", str(path), "--confirm"
            )
        rows = oc_dashboard.rows_for(oc_checkpoint.all_current(self.registry))
        ranked = sorted(rows, key=lambda r: -r["sort_key"])
        self.assertEqual(deep, ranked[0]["checkpoint_id"])


# --------------------------------------------------------------------------
# 9. delta checkpoints (oc_checkpoint 1.1.0)
# --------------------------------------------------------------------------


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def write_typed_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Like `write_safetensors`, but each tensor names its own dtype."""

    header: dict[str, object] = {}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        size = elements * _DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payload += bytes(size)
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))


class TestDelta(Base):
    """`register --delta-base`, `delta-compose`, `package-delta`, delta clones.

    The base is a real R2 clone of a reviewed fold child whose one tensor is
    `w` [16, 16] BF16 (256 parameters).  The standard delta below replaces rows
    [0, 4) of `w`, puts a rank-2 LoRA on all of it and a per-column gain on
    columns [8, 16): 64 replaced elements, 2*16 + 16*2 + 8 = 72 ADDED, so the
    composed model carries 256 + 72 = 328 parameters.
    """

    D1 = "OC01-D-Q35-4B-C2S4L03"

    def setUp(self) -> None:
        super().setUp()
        self.cid = "OC01-F-Q35-4B-C2S4L03"
        self.register_child(self.cid)
        self.assertEqual(0, self.review(self.cid, review_payload()))
        self.assertEqual(
            0,
            self.run_checkpoint(
                "clone", "--source", self.cid, "--stage-about-to-run", "D", "--confirm"
            ),
        )
        self.base = f"{self.cid}.clone-pre-D1"
        self.base_sha = oc_checkpoint.current(self.registry, self.base)["checkpoint"][
            "directory_sha256"
        ]

    def standard_entries(self) -> list[dict]:
        return [
            {"op": "replace", "target": "w", "file": "delta.safetensors",
             "tensor": "w.rows", "slice": [[0, 0, 4]]},
            {"op": "lora", "target": "w", "file": "delta.safetensors",
             "a": "w.lora_a", "b": "w.lora_b", "alpha": 4.0, "rank": 2},
            {"op": "scale", "target": "w", "file": "delta.safetensors",
             "tensor": "w.gain", "dim": 1, "slice": [[1, 8, 16]]},
        ]

    def standard_tensors(self) -> dict[str, tuple[str, list[int]]]:
        return {
            "w.rows": ("BF16", [4, 16]),
            "w.lora_a": ("BF16", [2, 16]),
            "w.lora_b": ("BF16", [16, 2]),
            "w.gain": ("F32", [8]),
        }

    def make_delta(
        self,
        name: str,
        *,
        base_id: str | None = None,
        base_sha: str | None = None,
        entries: list[dict] | None = None,
        tensors: dict[str, tuple[str, list[int]]] | None = None,
    ) -> Path:
        directory = self.root / "deltas" / name
        write_typed_safetensors(
            directory / "delta.safetensors",
            self.standard_tensors() if tensors is None else tensors,
        )
        (directory / oc_checkpoint.DELTA_RECIPE_NAME).write_text(
            json.dumps(
                {
                    "schema": oc_checkpoint.DELTA_RECIPE_SCHEMA,
                    "base_checkpoint_id": base_id or self.base,
                    "base_directory_sha256": base_sha or self.base_sha,
                    "entries": self.standard_entries() if entries is None else entries,
                }
            ),
            encoding="utf-8",
        )
        return directory

    def register_delta(self, checkpoint_id: str, directory: Path, *extra: str,
                       base: str | None = None) -> int:
        return self.run_checkpoint(
            "register", "--checkpoint-id", checkpoint_id, "--stage", "distill",
            "--directory", str(directory), "--delta-base", base or self.base,
            "--parent-directory", str(self.root / "parents" / "Q35-4B"),
            "--parent-code", "Q35-4B", "--recipe-command", "d1.sh", "--confirm", *extra,
        )

    def refused(self, *extra: str, **delta_kwargs: object) -> None:
        directory = self.make_delta("bad", **delta_kwargs)  # type: ignore[arg-type]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(3, self.register_delta(self.D1, directory, *extra))
        self.assertEqual([], oc_checkpoint.history(self.registry, self.D1))

    def test_version_string(self) -> None:
        self.assertTrue(oc_checkpoint.TOOL_VERSION.startswith("oc_checkpoint 1.1"))

    def test_register_delta_seals_pin_chain_and_composed_count(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        record = oc_checkpoint.current(self.registry, self.D1)
        checkpoint = record["checkpoint"]
        delta = checkpoint["delta"]
        self.assertEqual(self.base, delta["base_checkpoint_id"])
        self.assertEqual(self.base_sha, delta["base_directory_sha256"])
        self.assertEqual([self.base], delta["chain"])
        self.assertEqual(self.base, checkpoint["parent_checkpoint_id"])
        self.assertEqual(64, delta["replaced_elements"])
        self.assertEqual(72, delta["added_parameters"])
        self.assertEqual(256, delta["base_composed_parameter_count"])
        self.assertEqual(328, checkpoint["parameter_count"])
        self.assertEqual("derived-from-safetensors-headers", checkpoint["parameter_count_source"])
        self.assertEqual({"replace": 1, "lora": 1, "scale": 1}, delta["ops"])
        self.assertEqual(oc_checkpoint.TOOL_VERSION, delta["tool_version"])
        self.assertEqual(
            oc_seal.file_sha256(directory / "DELTA.json"), delta["recipe_file_sha256"]
        )
        code, _text = _capture(lambda: self.run_checkpoint("verify"))
        self.assertEqual(0, code, _text)

    def test_base_digest_mismatch_in_the_recipe_is_refused(self) -> None:
        self.refused(base_sha="c" * 64)

    def test_base_digest_mismatch_on_the_command_line_is_refused(self) -> None:
        self.refused("--delta-base-sha256", "d" * 64)

    def test_matching_command_line_pin_is_accepted(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(
            0, self.register_delta(self.D1, directory, "--delta-base-sha256", self.base_sha)
        )

    def test_recipe_naming_another_base_is_refused(self) -> None:
        self.refused(base_id=self.cid)

    def test_base_that_is_not_a_clone_is_refused(self) -> None:
        directory = self.make_delta(
            "d1", base_id=self.cid,
            base_sha=oc_checkpoint.current(self.registry, self.cid)["checkpoint"]["directory_sha256"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(3, self.register_delta(self.D1, directory, base=self.cid))

    def test_parent_other_than_the_base_is_refused(self) -> None:
        self.refused("--parent-checkpoint-id", self.cid)

    def test_fold_stage_delta_is_refused(self) -> None:
        directory = self.make_delta("d1")
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.run_checkpoint(
                "register", "--checkpoint-id", "OC01-F-Q35-4B-DELTAF", "--stage", "fold",
                "--directory", str(directory), "--delta-base", self.base,
                "--parent-directory", str(self.root / "parents" / "Q35-4B"),
                "--parent-code", "Q35-4B", "--recipe-command", "x", "--confirm",
            )
        self.assertEqual(3, code)

    def test_typed_parameter_count_must_equal_the_composed_count(self) -> None:
        self.refused("--parameter-count", "256")
        directory = self.make_delta("d1ok")
        self.assertEqual(0, self.register_delta(self.D1, directory, "--parameter-count", "328"))

    def test_structural_refusals(self) -> None:
        cases = {
            "unknown op": [{"op": "add", "target": "w", "tensor": "w.rows"}],
            "missing target": [{"op": "replace", "target": "nope", "tensor": "w.rows",
                                "slice": [[0, 0, 4]]}],
            "replace shape": [{"op": "replace", "target": "w", "tensor": "w.rows",
                               "slice": [[0, 0, 5]]}],
            "slice out of bounds": [{"op": "replace", "target": "w", "tensor": "w.rows",
                                     "slice": [[0, 14, 18]]}],
            "slice dim": [{"op": "replace", "target": "w", "tensor": "w.rows",
                           "slice": [[2, 0, 4]]}],
            "lora shape": [{"op": "lora", "target": "w", "a": "w.lora_b", "b": "w.lora_a",
                            "alpha": 4.0}],
            "lora alpha": [{"op": "lora", "target": "w", "a": "w.lora_a", "b": "w.lora_b",
                            "alpha": float("nan")}],
            "lora rank": [{"op": "lora", "target": "w", "a": "w.lora_a", "b": "w.lora_b",
                           "alpha": 4.0, "rank": 3}],
            "scale length": [{"op": "scale", "target": "w", "tensor": "w.gain", "dim": 1}],
            "empty": [],
        }
        for label, entries in cases.items():
            with self.subTest(label):
                directory = self.make_delta(f"bad-{len(label)}-{label[:4]}", entries=entries,
                                            tensors={"w.rows": ("BF16", [4, 16]),
                                                     "w.lora_a": ("BF16", [2, 16]),
                                                     "w.lora_b": ("BF16", [16, 2]),
                                                     "w.gain": ("F32", [8])})
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(3, self.register_delta(self.D1, directory))
                self.assertEqual([], oc_checkpoint.history(self.registry, self.D1))

    def test_orphan_tensor_overlap_dtype_and_reuse_are_refused(self) -> None:
        tensors = dict(self.standard_tensors(), **{"stray": ("BF16", [3])})
        self.refused(tensors=tensors)
        overlap = self.standard_entries() + [
            {"op": "replace", "target": "w", "tensor": "w.rows2", "slice": [[0, 2, 6]]}
        ]
        self.refused(entries=overlap,
                     tensors=dict(self.standard_tensors(), **{"w.rows2": ("BF16", [4, 16])}))
        wrong_dtype = dict(self.standard_tensors(), **{"w.rows": ("F32", [4, 16])})
        self.refused(tensors=wrong_dtype)
        reuse = self.standard_entries() + [
            {"op": "scale", "target": "w", "tensor": "w.gain", "dim": 1, "slice": [[1, 0, 8]]}
        ]
        self.refused(entries=reuse)

    def test_delta_without_recipe_is_refused(self) -> None:
        directory = self.make_delta("d1")
        (directory / "DELTA.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(3, self.register_delta(self.D1, directory))

    def _tamper_base(self) -> None:
        shard = Path(oc_checkpoint.current(self.registry, self.base)["checkpoint"]["directory"]) / "model.safetensors"
        os.chmod(shard, 0o644)
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 0x01  # one payload bit; the header still parses
        shard.write_bytes(bytes(raw))

    def test_delta_compose_writes_a_sealed_plan_and_deep_catches_moved_base_bytes(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        plan_path = self.root / "plan.json"
        code, text = _capture(lambda: self.run_checkpoint(
            "delta-compose", "--checkpoint-id", self.D1, "--deep", "--plan-out", str(plan_path)
        ))
        self.assertEqual(0, code, text)
        plan = oc_seal.read_sealed(plan_path)
        self.assertEqual(oc_checkpoint.DELTA_PLAN_SCHEMA, plan["schema"])
        self.assertEqual(self.base, plan["root"]["checkpoint_id"])
        self.assertEqual([self.D1], [d["checkpoint_id"] for d in plan["deltas"]])
        self.assertEqual(328, plan["composed_parameter_count"])
        self.assertIn("sha256", plan["root"]["files"][0])
        self.assertEqual(self.base_sha, plan["deltas"][0]["base_directory_sha256"])

        self._tamper_base()
        # the SEALED pins still agree with each other, so a shallow compose passes;
        # only the re-hash sees that the bytes under the seal moved.
        code, _text = _capture(lambda: self.run_checkpoint(
            "delta-compose", "--checkpoint-id", self.D1))
        self.assertEqual(0, code)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = self.run_checkpoint("delta-compose", "--checkpoint-id", self.D1, "--deep")
        self.assertEqual(3, code)
        self.assertIn("BASE DIGEST MISMATCH", err.getvalue())

    def test_plan_out_is_never_silently_overwritten(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        plan_path = self.root / "plan.json"
        plan_path.write_text("{}", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(3, self.run_checkpoint(
                "delta-compose", "--checkpoint-id", self.D1, "--plan-out", str(plan_path)))

    def test_clone_of_a_delta_and_a_second_delta_on_top(self) -> None:
        """D1 -> clone-pre-D2 -> D2: the R2 chain the design runs, end to end."""

        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        code, text = _capture(lambda: self.run_checkpoint(
            "clone", "--source", self.D1, "--stage-about-to-run", "D", "--clone-index", "2",
            "--allow-unreviewed-source", "--confirm",
        ))
        self.assertEqual(0, code, text)
        self.assertIn("delta              yes", text)
        d1_clone = f"{self.D1}.clone-pre-D2"
        clone = oc_checkpoint.current(self.registry, d1_clone)["checkpoint"]
        self.assertEqual(
            oc_checkpoint.current(self.registry, self.D1)["checkpoint"]["directory_sha256"],
            clone["directory_sha256"],
        )
        self.assertEqual(self.base, clone["delta"]["base_checkpoint_id"])

        d2 = self.make_delta(
            "d2", base_id=d1_clone, base_sha=clone["directory_sha256"],
            entries=[{"op": "lora", "target": "w", "file": "delta.safetensors",
                      "a": "a2", "b": "b2", "alpha": 1.0}],
            tensors={"a2": ("BF16", [1, 16]), "b2": ("BF16", [16, 1])},
        )
        self.assertEqual(0, self.register_delta("OC01-D-Q35-4B-C2S4L03D2", d2, base=d1_clone))
        record = oc_checkpoint.current(self.registry, "OC01-D-Q35-4B-C2S4L03D2")
        self.assertEqual([self.base, d1_clone], record["checkpoint"]["delta"]["chain"])
        self.assertEqual(328 + 32, record["checkpoint"]["parameter_count"])
        plan = oc_checkpoint.delta_plan(self.registry, "OC01-D-Q35-4B-C2S4L03D2")
        self.assertEqual([d1_clone, "OC01-D-Q35-4B-C2S4L03D2"],
                         [d["checkpoint_id"] for d in plan["deltas"]])
        self.assertEqual(360, plan["composed_parameter_count"])
        code, text = _capture(lambda: self.run_checkpoint("verify"))
        self.assertEqual(0, code, text)
        # the base of a delta can never be deleted out from under it
        args = _delete_args(d1_clone, self.root)
        conditions = oc_checkpoint._delete_conditions(self.registry, d1_clone, args)
        iv_b = [ok for ok, text in conditions if text.startswith("(iv-b)")]
        self.assertEqual([False], iv_b)

    def test_clone_refuses_when_the_base_pin_broke(self) -> None:
        """A sealed pin that disagrees with its base is caught before a byte is copied."""

        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        # forge the head of the D1 chain with a different pin, sealed properly
        head = oc_checkpoint.current(self.registry, self.D1)
        checkpoint = dict(head["checkpoint"])
        checkpoint["delta"] = dict(checkpoint["delta"], base_directory_sha256="e" * 64)
        record = oc_checkpoint.build_record(
            checkpoint_id=self.D1, seq=2, event="status", supersedes=head["receipt_sha"],
            checkpoint=checkpoint, review=None, status="candidate", status_reason="t",
            notes="forged pin", created_at=oc_seal.now_utc(),
        )
        oc_checkpoint.commit_record(registry=self.registry, ledger=self.ledger, record=record)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = self.run_checkpoint(
                "clone", "--source", self.D1, "--stage-about-to-run", "D",
                "--allow-unreviewed-source", "--confirm",
            )
        self.assertEqual(3, code)
        self.assertIn("BASE DIGEST MISMATCH", err.getvalue())
        self.assertFalse((self.root / "clones" / f"{self.D1}.clone-pre-D1").exists())
        code, text = _capture(lambda: self.run_checkpoint("verify"))
        self.assertEqual(1, code)
        self.assertIn("BASE DIGEST MISMATCH", text)

    def test_verify_sees_an_edited_recipe(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        recipe = json.loads((directory / "DELTA.json").read_text(encoding="utf-8"))
        recipe["notes"] = "edited after registration"
        (directory / "DELTA.json").write_text(json.dumps(recipe), encoding="utf-8")
        code, text = _capture(lambda: self.run_checkpoint("verify"))
        self.assertEqual(1, code)
        self.assertIn("recipe was edited", text)

    def test_package_delta_verifies_and_detects_tampering(self) -> None:
        directory = self.make_delta("d1")
        self.assertEqual(0, self.register_delta(self.D1, directory))
        out = self.root / "pkg" / self.D1
        code, _text = _capture(lambda: self.run_checkpoint(
            "package-delta", "--checkpoint-id", self.D1, "--out", str(out)))
        self.assertEqual(0, code)
        self.assertFalse(out.exists())  # dry run
        code, text = _capture(lambda: self.run_checkpoint(
            "package-delta", "--checkpoint-id", self.D1, "--out", str(out), "--confirm"))
        self.assertEqual(0, code, text)
        self.assertEqual([], oc_package.verify_manifest(out))
        manifest = oc_package.read_manifest(out)
        self.assertEqual(self.D1, manifest["checkpoint_id"])
        self.assertEqual(
            oc_checkpoint.current(self.registry, self.D1)["receipt_sha"],
            oc_seal.read_sealed(out / "RECORD.json")["receipt_sha"],
        )
        self.assertEqual(328, oc_seal.read_sealed(out / "PLAN.json")["composed_parameter_count"])
        self.assertEqual(
            oc_seal.directory_digest(directory)[0],
            oc_seal.directory_digest(out / "delta")[0],
        )
        (out / "delta" / "DELTA.json").write_text("{}", encoding="utf-8")
        self.assertTrue(oc_package.verify_manifest(out))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(3, self.run_checkpoint(
                "package-delta", "--checkpoint-id", self.D1, "--out", str(out), "--confirm"))

    def test_package_delta_refuses_a_full_checkpoint(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(3, self.run_checkpoint(
                "package-delta", "--checkpoint-id", self.cid,
                "--out", str(self.root / "pkg"), "--confirm"))

    def test_a_non_delta_record_is_unchanged(self) -> None:
        """1.0 records carry no `delta` key: their sealed body did not move."""

        record = oc_checkpoint.current(self.registry, self.cid)
        self.assertNotIn("delta", record["checkpoint"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
