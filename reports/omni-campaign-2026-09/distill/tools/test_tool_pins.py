"""Tests for tool_pins.py 1.1.0 (card X-T3, lane infra, Claude, 2026-10-14; 1.1.0 --only + speech dirs 2026-10-15).  CPU,
temp trees only:

    cd /home/cereal/omni-campaign/distill/tools
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider -q test_tool_pins.py

What is pinned:
  * versions are READ from the files (TOOL_VERSION / VERSION= / LAUNCH_VERSION= / J6_VERSION=..., never
    NUMERICS_VERSION or ENGINE_EXPECTED_VERSION); a file without one is pinned by sha only;
  * superseded copies attach to their live file by name (F.<x>, S_<digit>, S.<sha16>.ext, F inside a superseded dir),
    a REFREEZE archive in queues/frozen is superseded rather than installed, _staging_* is never scanned;
  * an installed copy names the tools/ file with the same sha;
  * verify: 0 on a fresh generate; 1 with a DRIFT line after an edit, MISSING after a delete, UNPINNED for a new file
    (0 with --allow-unpinned); generate keeps the previous pins under tools/_superseded/;
  * get returns one entry; the real tools directory's own files parse (smoke over every *.py / *.sh beside this test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tool_pins as tp  # noqa: E402


def w(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture
def tree(tmp_path):
    r = tmp_path / "distill"
    w(r / "tools/trainer.py", 'TOOL_NAME = "trainer"\nNUMERICS_VERSION = "n.v1"\nTOOL_VERSION = "1.3.3"  # note\n')
    w(r / "tools/engine.py", 'ENGINE_EXPECTED_VERSION = "9.9.9"\nTOOL_VERSION = "engine 2.0.1"\n')
    w(r / "tools/queue_gpu6.sh", '#!/bin/bash\nset -u\nVERSION="queue_gpu6.sh 1.3.1"\n')
    w(r / "tools/d1_launch.sh", "#!/bin/bash\nLAUNCH_VERSION=d1_hardened_launch-1.2.0   # c\n")
    w(r / "tools/plain.sh", "#!/bin/bash\necho hi\n")
    w(r / "tools/test_trainer.py", "def test_x():\n    pass\n")
    w(r / "tools/_superseded/trainer.py.1.3.2", 'TOOL_VERSION = "1.3.2"\n')
    w(r / "tools/_superseded/trainer_1.3.1.py", 'TOOL_VERSION = "1.3.1"\n')
    w(r / "tools/_superseded/queue_gpu6_1.3.0.sh", 'VERSION="queue_gpu6.sh 1.3.0"\n')
    w(r / "tools/_superseded/old_dir/test_trainer.py", "def test_y():\n    pass\n")
    w(r / "tools/_superseded/unrelated_tool_1.0.0.py", 'TOOL_VERSION = "1.0.0"\n')
    w(r / "tools/_staging_1.0.9/trainer.py", 'TOOL_VERSION = "9.0.0"\n')
    w(r / "tools/__pycache__/trainer.cpython-312.pyc", "x")
    w(r / "queues/gpu6.sh", '#!/bin/bash\nVERSION="queue_gpu6.sh 1.3.0"\n')
    w(r / "queues/gpu6.sh.1.2.0-live-20261014", 'VERSION="queue_gpu6.sh 1.2.0"\n')
    w(r / "queues/j6_vl.sh", 'J6_VERSION="j6_vl.sh 1.1.0"\n')
    w(r / "queues/frozen/engine-gpu6.py", 'TOOL_VERSION = "engine 2.0.1"\n')
    w(r / "queues/frozen/engine-gpu6.0123456789abcdef.py", 'TOOL_VERSION = "engine 2.0.0"\n')
    w(r / "queues/plain_copy.sh", "#!/bin/bash\necho hi\n")
    return r


def test_build(tree):
    pins = tp.build(tree)
    t, i = pins["tools"], pins["installed"]
    assert set(t) == {"trainer.py", "engine.py", "queue_gpu6.sh", "d1_launch.sh", "plain.sh", "test_trainer.py"}
    assert (t["trainer.py"]["version"], t["trainer.py"]["version_source"]) == ("1.3.3", "TOOL_VERSION")
    assert (t["engine.py"]["version"], t["engine.py"]["version_string"]) == ("2.0.1", "engine 2.0.1")
    assert t["queue_gpu6.sh"]["version"] == "1.3.1" and t["d1_launch.sh"]["version"] == "1.2.0"
    assert t["plain.sh"]["version"] is None and t["plain.sh"]["version_source"] == "none"
    assert sorted(s["version"] for s in t["trainer.py"]["superseded"]) == ["1.3.1", "1.3.2"]
    assert [s["path"] for s in t["test_trainer.py"]["superseded"]] == ["tools/_superseded/old_dir/test_trainer.py"]
    assert [s["version"] for s in t["queue_gpu6.sh"]["superseded"]] == ["1.3.0"]
    assert set(i) == {"queues/gpu6.sh", "queues/j6_vl.sh", "queues/frozen/engine-gpu6.py", "queues/plain_copy.sh"}
    assert i["queues/j6_vl.sh"]["version"] == "1.1.0"
    assert [s["path"] for s in i["queues/gpu6.sh"]["superseded"]] == ["queues/gpu6.sh.1.2.0-live-20261014"]
    assert [s["version"] for s in i["queues/frozen/engine-gpu6.py"]["superseded"]] == ["2.0.0"]
    assert i["queues/plain_copy.sh"]["same_as_tools"] == ["plain.sh"] and i["queues/gpu6.sh"]["same_as_tools"] == []
    assert [o["path"] for o in pins["superseded_unmatched"]] == ["tools/_superseded/unrelated_tool_1.0.0.py"]
    assert all("_staging" not in e["path"] for e in t.values())
    assert pins["counts"] == {"tools": 6, "installed": 4, "superseded": 7, "superseded_unmatched": 1, "unversioned": 3}


def test_generate_verify_get(tree, capsys):
    assert tp.main(["generate", "--root", str(tree)]) == 0
    pins_file = tree / "tools/TOOL_PINS.json"
    assert json.loads(pins_file.read_text())["schema"] == tp.SCHEMA
    assert tp.main(["verify", "--root", str(tree)]) == 0
    assert "TOOL_PINS VERIFY: 10 ok, 0 drift, 0 missing, 0 unpinned" in capsys.readouterr().out
    assert tp.main(["get", "trainer.py", "--root", str(tree)]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "1.3.3"
    assert tp.main(["get", "queues/gpu6.sh", "--root", str(tree)]) == 0
    capsys.readouterr()
    assert tp.main(["get", "nope.py", "--root", str(tree)]) == 1

    w(tree / "tools/trainer.py", 'TOOL_VERSION = "1.3.4"\n')
    (tree / "tools/plain.sh").unlink()
    w(tree / "tools/new_tool.py", 'TOOL_VERSION = "0.1.0"\n')
    assert tp.main(["verify", "--root", str(tree)]) == 1
    out = capsys.readouterr().out
    assert "DRIFT tools/trainer.py: pinned 1.3.3" in out and "on disk 1.3.4" in out
    assert "MISSING tools/plain.sh" in out and "UNPINNED tools/new_tool.py" in out
    assert "8 ok, 1 drift, 1 missing, 1 unpinned" in out
    w(tree / "tools/plain.sh", "#!/bin/bash\necho hi\n")
    w(tree / "tools/trainer.py", 'TOOL_NAME = "trainer"\nNUMERICS_VERSION = "n.v1"\nTOOL_VERSION = "1.3.3"  # note\n')
    assert tp.main(["verify", "--root", str(tree), "--allow-unpinned"]) == 0
    capsys.readouterr()

    assert tp.main(["generate", "--root", str(tree)]) == 0
    assert "previous pins kept as" in capsys.readouterr().out
    assert len(list((tree / "tools/_superseded").glob("TOOL_PINS.*.json"))) == 1
    assert tp.main(["verify", "--root", str(tree)]) == 0


def test_every_real_tool_parses():
    for p in sorted(HERE.glob("*")):
        if p.is_file() and p.suffix in tp.CODE_EXT:
            version, vstring, source = tp.read_version(p)
            assert source != "unreadable", p
            assert (version is None) == (source == "none"), p
    assert tp.read_version(HERE / "tool_pins.py")[0] == tp.TOOL_VERSION


# ---- 1.1.0: speech sub-directories and the deployed-only re-pin (intake review item 7) ------------------------------


@pytest.fixture
def lanes_tree(tmp_path):
    r = tmp_path / "distill"
    w(r / "tools/intake.py", 'VERSION = "1.1.1"\n')
    w(r / "tools/conftest.py", 'CONFTEST_VERSION = "oc_cuda_guard 1.0.1"\n')
    w(r / "tools/d2clean_recache.py", 'TOOL_VERSION = "1.0.1"\n')          # another lane's tool
    w(r / "tools/speech/asr_two_judge.py", 'VERSION = "1.0.0"\n')
    w(r / "tools/speech/tests/test_asr_two_judge.py", "def test_x():\n    pass\n")
    w(r / "tools/speech/tests/conftest.py", 'CONFTEST_VERSION = "oc_cuda_guard 1.0.1"\n')
    w(r / "tools/speech/_superseded/asr_two_judge_0.9.0.py", 'VERSION = "0.9.0"\n')
    w(r / "tools/speech/__pycache__/asr_two_judge.cpython-312.pyc", "x")
    w(r / "queues/gpu6.sh", 'VERSION="queue_gpu6.sh 1.3.0"\n')
    return r


def test_speech_subdirectories_are_pinned_by_relative_name(lanes_tree, capsys):
    pins = tp.build(lanes_tree)
    t = pins["tools"]
    assert set(t) == {"intake.py", "conftest.py", "d2clean_recache.py", "speech/asr_two_judge.py",
                      "speech/tests/test_asr_two_judge.py", "speech/tests/conftest.py"}
    assert t["speech/asr_two_judge.py"]["path"] == "tools/speech/asr_two_judge.py" and t["speech/asr_two_judge.py"]["version"] == "1.0.0"
    assert [s["version"] for s in t["speech/asr_two_judge.py"]["superseded"]] == ["0.9.0"]
    assert t["speech/tests/conftest.py"]["sha256"] == t["conftest.py"]["sha256"]          # same name, both pinned
    assert tp.main(["generate", "--root", str(lanes_tree)]) == 0
    assert tp.main(["get", "speech/asr_two_judge.py", "--root", str(lanes_tree)]) == 0
    capsys.readouterr()
    assert tp.main(["verify", "--root", str(lanes_tree)]) == 0
    w(lanes_tree / "tools/speech/new_helper.py", 'VERSION = "0.1.0"\n')
    assert tp.main(["verify", "--root", str(lanes_tree)]) == 1
    assert "UNPINNED tools/speech/new_helper.py" in capsys.readouterr().out


def test_generate_only_repins_the_named_files_and_nothing_else(lanes_tree, capsys):
    assert tp.main(["generate", "--root", str(lanes_tree)]) == 0
    before = json.loads((lanes_tree / "tools/TOOL_PINS.json").read_text())
    capsys.readouterr()
    # my deploy: intake 1.2.0 + asr_two_judge 1.1.0 + a new speech file; meanwhile another lane edits its tool
    # (in flight, not deployed), adds an unpinned file and changes an installed queue
    w(lanes_tree / "tools/intake.py", 'VERSION = "1.2.0"\n')
    w(lanes_tree / "tools/speech/asr_two_judge.py", 'VERSION = "1.1.0"\n')
    w(lanes_tree / "tools/speech/speech_extra.py", 'VERSION = "1.0.0"\n')
    w(lanes_tree / "tools/d2clean_recache.py", 'TOOL_VERSION = "1.0.2"  # in flight\n')
    w(lanes_tree / "tools/d2_next.py", 'TOOL_VERSION = "0.1.0"\n')
    w(lanes_tree / "queues/gpu6.sh", 'VERSION="queue_gpu6.sh 1.3.1"\n')
    assert tp.main(["generate", "--root", str(lanes_tree), "--only", "intake.py", "speech/asr_two_judge.py",
                    "speech/speech_extra.py"]) == 0
    out = capsys.readouterr().out
    assert "GENERATE --only intake.py speech/asr_two_judge.py speech/speech_extra.py" in out and "previous pins kept" in out
    after = json.loads((lanes_tree / "tools/TOOL_PINS.json").read_text())
    assert after["tools"]["intake.py"]["version"] == "1.2.0" and after["tools"]["speech/asr_two_judge.py"]["version"] == "1.1.0"
    assert after["tools"]["speech/speech_extra.py"]["version"] == "1.0.0"
    assert after["tools"]["d2clean_recache.py"] == before["tools"]["d2clean_recache.py"]     # NOT re-pinned
    assert after["installed"]["queues/gpu6.sh"] == before["installed"]["queues/gpu6.sh"]
    assert "d2_next.py" not in after["tools"]
    assert after["mode"] == {"only": ["intake.py", "speech/asr_two_judge.py", "speech/speech_extra.py"],
                             "base_pins_sha256": before["pins_sha256"], "base_generated_utc": before["generated_utc"]}
    assert after["pins_sha256"] == tp.hashlib.sha256(json.dumps(tp.core_of(after), sort_keys=True).encode()).hexdigest()
    assert after["counts"]["tools"] == len(after["tools"]) == 7
    assert tp.main(["verify", "--root", str(lanes_tree)]) == 1                               # the other lane's drift stays
    out = capsys.readouterr().out
    assert "DRIFT tools/d2clean_recache.py: pinned 1.0.1" in out and "DRIFT queues/gpu6.sh" in out
    assert "UNPINNED tools/d2_next.py" in out and "intake.py" not in out.split("TOOL_PINS VERIFY")[0]
    assert "3 drift" not in out and "2 drift" in out
    # refusals: a name not on disk, and --only without current pins
    with pytest.raises(SystemExit, match="not found on disk"):
        tp.main(["generate", "--root", str(lanes_tree), "--only", "nope.py"])
    (lanes_tree / "tools/TOOL_PINS.json").unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        tp.main(["generate", "--root", str(lanes_tree), "--only", "intake.py"])
