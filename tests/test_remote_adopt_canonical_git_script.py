from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "remote_adopt_canonical_git_20260712.sh"
)


def test_remote_adoption_uses_the_existing_source_root_without_clean_or_reset() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ROOT="/home/cereal/model-surgery-lab/neural-forge-src"' in text
    assert "git checkout-index --all --force" in text
    assert "git clean" not in text
    assert "git reset" not in text
    assert "rm -" not in text


def test_remote_adoption_pins_the_canonical_origin_and_branch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ORIGIN="https://github.com/autonomouscereal/neural-forge.git"' in text
    assert 'BRANCH="codex/lrc-blank-compression"' in text
    assert 'git config "branch.$BRANCH.remote" origin' in text


def test_remote_adoption_supports_a_verified_secretless_git_bundle() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'MODE" == "--adopt-bundle"' in text
    assert 'git bundle verify "$BUNDLE"' in text
    assert 'git fetch "$BUNDLE" "refs/heads/$BRANCH"' in text
    assert 'git bundle list-heads "$BUNDLE"' in text
    assert "refs/tags/*:refs/tags/*" in text
