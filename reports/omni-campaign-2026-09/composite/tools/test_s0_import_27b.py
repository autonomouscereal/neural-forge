"""Tests for s0_import_27b.py (run on the AI server, CPU only).

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_c5e6052 \
    PYTHONPATH=$CHIMERA_OMNI_SRC/src /home/cereal/model-surgery-lab/venv/bin/python -m pytest \
        -o addopts="" -p no:cacheprovider -q test_s0_import_27b.py \
        --basetemp /home/cereal/omni-campaign/composite/s0/scratch_tests

Run-book row tests (tool table, s0_import_27b.py, as amended):
  (1) on Qwen3.5-0.8B WITHOUT the band the tool reproduces the import lane's
      `import-lane/s4-fix/qwen35_0p8b_chimera_bf16` byte-for-byte on every
      source-derived tensor, every band row and the tied alias; the random-init
      interface tensors are reproducible only from this tool's seed, and the receipt's
      `lane_comparison` block (--compare-to) names exactly which tensors differ and why;
      re-running the tool reproduces its own output byte-for-byte;
  (2) with `music_codes:65536` declared the header element count grows by exactly
      65,536 x hidden x tables (tables = 2 untied / 1 tied) and the band rows are zero;
  (3) a `--chimera-commit` that is not the tree's commit is refused (exit 3, nothing written).
Review fixes (one test each): --overwrite clears the previous layout and the receipt
records directory == index shard set; --compare-to lane_comparison; --profile auto and
profile-independent tensor bytes; --shard-bytes 0 refused; post-write refusal cleanup +
REFUSED.json; fp16 narrowing waiver + range record; spec commit c5e6052 (not 5a87018).
The synthetic fixture (tiny untied Qwen-named checkpoint) covers everything in seconds;
the two real-0.8B tests skip when the model or the lane checkpoint is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "s0_import_27b.py"
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"))
QWEN_0P8B = Path(os.environ.get("QWEN_0P8B_DIR", "/home/cereal/models/Qwen3.5-0.8B"))
LANE_CKPT = Path(
    os.environ.get(
        "IMPORT_LANE_0P8B_BF16",
        "/home/cereal/omni-campaign/import-lane/s4-fix/qwen35_0p8b_chimera_bf16",
    )
)
MUSIC_BAND = 65536
INDEX = "model.safetensors.index.json"
ONE_SHARD = "model-00001-of-00001.safetensors"

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _load_tool():
    spec = importlib.util.spec_from_file_location("s0_import_27b", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


@pytest.fixture(scope="session")
def tree_commit() -> str:
    if not (CHIMERA_SRC / "src" / "chimera_omni").is_dir():
        pytest.skip(f"no chimera-omni tree at {CHIMERA_SRC}")
    commit, _how = tool.resolve_tree_commit(CHIMERA_SRC)
    return commit


def run_tool(args: list[str], *, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CHIMERA_OMNI_SRC"] = str(CHIMERA_SRC)
    env.pop("PYTHONPATH", None)  # the tool binds the tree itself; nothing may shadow it
    proc = subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    assert proc.returncode == expect, f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc


def summary_of(proc: subprocess.CompletedProcess) -> dict:
    return json.loads(proc.stdout.strip().splitlines()[-1])


def receipt_of(out: Path) -> dict:
    return json.load(open(out / "import_receipt.json", encoding="utf-8"))


# --------------------------------------------------------------------------
# synthetic untied Qwen-named checkpoint (geometry of tests/test_import_qwen35_synthetic_torch_optional.py)
# --------------------------------------------------------------------------

HIDDEN = 32
LAYERS = 4
INTERMEDIATE = 24
KEY_HEADS = 2
VALUE_HEADS = 4
KEY_HEAD_DIM = 8
VALUE_HEAD_DIM = 8
CONV_KERNEL = 4
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
ROPE_DIM = 2
VOCAB = 64


def _hf_config(*, tie: bool, output_gate_type: str | None = None) -> dict:
    text: dict = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "linear_num_key_heads": KEY_HEADS,
        "linear_num_value_heads": VALUE_HEADS,
        "linear_key_head_dim": KEY_HEAD_DIM,
        "linear_value_head_dim": VALUE_HEAD_DIM,
        "linear_conv_kernel_dim": CONV_KERNEL,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "rope_parameters": {"rope_theta": 1.0e7, "partial_rotary_factor": ROPE_DIM / HEAD_DIM},
        "rms_norm_eps": 1.0e-6,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": tie,
        "hidden_act": "silu",
        "mtp_num_hidden_layers": 1,
    }
    if output_gate_type is not None:
        text["output_gate_type"] = output_gate_type
    return {"text_config": text}


def _write_synthetic_qwen(root: Path, *, tie: bool = False, output_gate_type: str | None = None) -> Path:
    from chimera_omni.importing.qwen35 import QWEN_MEDIA_PREFIX, QWEN_MTP_PREFIX, QWEN_TEXT_PREFIX

    generator = torch.Generator().manual_seed(0)

    def normal(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16)

    key_dim = KEY_HEADS * KEY_HEAD_DIM
    value_dim = VALUE_HEADS * VALUE_HEAD_DIM
    prefix = QWEN_TEXT_PREFIX
    state: dict[str, torch.Tensor] = {
        f"{prefix}embed_tokens.weight": normal(VOCAB, HIDDEN),
        f"{prefix}norm.weight": normal(HIDDEN),
    }
    if not tie:
        state["lm_head.weight"] = normal(VOCAB, HIDDEN)
    for index in range(LAYERS):
        layer = f"{prefix}layers.{index}."
        state[f"{layer}input_layernorm.weight"] = normal(HIDDEN)
        state[f"{layer}post_attention_layernorm.weight"] = normal(HIDDEN)
        state[f"{layer}mlp.gate_proj.weight"] = normal(INTERMEDIATE, HIDDEN)
        state[f"{layer}mlp.up_proj.weight"] = normal(INTERMEDIATE, HIDDEN)
        state[f"{layer}mlp.down_proj.weight"] = normal(HIDDEN, INTERMEDIATE)
        if index % 4 == 3:
            state[f"{layer}self_attn.q_proj.weight"] = normal(Q_HEADS * 2 * HEAD_DIM, HIDDEN)
            state[f"{layer}self_attn.k_proj.weight"] = normal(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{layer}self_attn.v_proj.weight"] = normal(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{layer}self_attn.o_proj.weight"] = normal(HIDDEN, Q_HEADS * HEAD_DIM)
            state[f"{layer}self_attn.q_norm.weight"] = normal(HEAD_DIM)
            state[f"{layer}self_attn.k_norm.weight"] = normal(HEAD_DIM)
        else:
            state[f"{layer}linear_attn.in_proj_qkv.weight"] = normal(2 * key_dim + value_dim, HIDDEN)
            state[f"{layer}linear_attn.in_proj_z.weight"] = normal(value_dim, HIDDEN)
            state[f"{layer}linear_attn.in_proj_a.weight"] = normal(VALUE_HEADS, HIDDEN)
            state[f"{layer}linear_attn.in_proj_b.weight"] = normal(VALUE_HEADS, HIDDEN)
            state[f"{layer}linear_attn.conv1d.weight"] = normal(2 * key_dim + value_dim, 1, CONV_KERNEL)
            state[f"{layer}linear_attn.dt_bias"] = normal(VALUE_HEADS, scale=1.0)
            state[f"{layer}linear_attn.A_log"] = torch.log(
                torch.empty(VALUE_HEADS).uniform_(1.0, 16.0, generator=generator)
            ).to(torch.bfloat16)
            state[f"{layer}linear_attn.norm.weight"] = normal(VALUE_HEAD_DIM)
            state[f"{layer}linear_attn.out_proj.weight"] = normal(HIDDEN, value_dim)
    state[f"{QWEN_MTP_PREFIX}fc.weight"] = normal(HIDDEN, 2 * HIDDEN)
    state[f"{QWEN_MEDIA_PREFIX}patch_embed.proj.weight"] = normal(HIDDEN, 3)

    root.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(
        json.dumps(_hf_config(tie=tie, output_gate_type=output_gate_type), indent=2), encoding="utf-8"
    )
    return root


def _band_rows_zero(ckpt: Path, base: int) -> tuple[bool, dict[str, int]]:
    from chimera_omni.importing.safetensors_io import read_safetensors

    views = read_safetensors(ckpt, ["embed.weight"], verify=True)
    embed = views["embed.weight"].torch()
    zero = bool((embed[base:] == 0).all().item())
    index = json.load(open(ckpt / INDEX, encoding="utf-8"))
    if "lm_head.weight" in index["weight_map"]:
        head = read_safetensors(ckpt, ["lm_head.weight"], verify=True)["lm_head.weight"].torch()
        zero = zero and bool((head[base:] == 0).all().item())
    return zero, index["metadata"]["tied_weights"]


def _safetensors_under(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.safetensors"))


# --------------------------------------------------------------------------
# run-book row tests
# --------------------------------------------------------------------------


def test_synthetic_untied_band_grows_headers_by_65536_x_hidden_x_2_and_rows_are_zero(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    plain = tmp_path / "plain"
    banded = tmp_path / "banded"
    common = ["--src", str(src), "--chimera-commit", tree_commit]
    s_plain = summary_of(run_tool([*common, "--out", str(plain)]))
    s_band = summary_of(run_tool([*common, "--out", str(banded), "--ledger-band", f"music_codes:{MUSIC_BAND}"]))

    counts_plain = tool.header_element_counts(plain)
    counts_band = tool.header_element_counts(banded)
    growth = sum(counts_band.values()) - sum(counts_plain.values())
    assert growth == MUSIC_BAND * HIDDEN * 2, growth
    assert s_band["header_element_count"] - s_plain["header_element_count"] == MUSIC_BAND * HIDDEN * 2
    assert s_band["extra_band_elements_declared"] == MUSIC_BAND * HIDDEN * 2
    # only the two vocab tables grew
    for name in counts_plain:
        if name in {"embed.weight", "lm_head.weight"}:
            assert counts_band[name] - counts_plain[name] == MUSIC_BAND * HIDDEN
        else:
            assert counts_band[name] == counts_plain[name], name

    zero, tied = _band_rows_zero(banded, VOCAB)
    assert zero and tied == {}
    receipt = receipt_of(banded)
    assert receipt["schema"] == "chimera_omni.import.qwen35.v1"
    assert receipt["ledger_bands"]["declared_extra"] == [["music_codes", MUSIC_BAND]]
    assert receipt["ledger_bands"]["band_rows_zero"] is True
    assert receipt["ledger_bands"]["extra_band_rows_zero"] is True
    assert receipt["ledger_bands"]["tables"] == 2
    manifest = json.load(open(banded / "config.json", encoding="utf-8"))
    assert manifest["config"]["ledger_bands"][-1] == ["music_codes", MUSIC_BAND]
    assert manifest["config"]["vocab_size"] == VOCAB + 4096 + MUSIC_BAND
    assert manifest["chimera_commit"] == tree_commit
    # receipt carries what the row asks for
    if tool.peak_rss_bytes() is None:
        assert receipt["peak_rss_bytes"] is None
    else:
        assert isinstance(receipt["peak_rss_bytes"], int) and receipt["peak_rss_bytes"] > 0
    assert receipt["tool"]["sha256"] == tool.file_sha256(TOOL)
    assert receipt["chimera"]["executable_source"] == tool.executable_source_manifest(CHIMERA_SRC)
    assert {item["name"] for item in receipt["skipped"]["mtp"]} == {"mtp.fc.weight"}
    assert {item["name"] for item in receipt["skipped"]["media"]} == {"model.visual.patch_embed.proj.weight"}
    assert all("sha256" in item for item in receipt["skipped"]["mtp"] + receipt["skipped"]["media"])
    assert len(receipt["interface_tensors"]) == 16
    sidecar = (banded / "import_receipt.json.sha256").read_text().split()[0]
    assert len(sidecar) == 64
    assert receipt["checkpoint"]["tensor_sha256_count"] == receipt["checkpoint"]["tensors"]
    assert receipt["checkpoint"]["directory_shard_files_match_index"] is True
    assert receipt["checkpoint"]["directory_shard_files"] == receipt["checkpoint"]["index_shard_files"] == [ONE_SHARD]
    assert receipt["lane_comparison"] is None and receipt["narrowing"] is None


def test_synthetic_rerun_is_byte_identical(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    a = tmp_path / "a"
    b = tmp_path / "b"
    for out in (a, b):
        run_tool(["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out)])
    for name in (ONE_SHARD, INDEX, "config.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_commit_mismatch_is_refused_and_writes_nothing(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    out = tmp_path / "refused"
    wrong = "deadbeefcafe0000"
    assert not tool.commits_match(wrong, tree_commit)
    proc = run_tool(["--src", str(src), "--chimera-commit", wrong, "--out", str(out)], expect=3)
    assert "REFUSED" in proc.stderr and "does not name the tree" in proc.stderr
    assert not out.exists()
    # prefix match is what a short hash means
    assert tool.commits_match(tree_commit[:7], tree_commit)
    assert tool.commits_match(tree_commit, tree_commit[:9])
    assert not tool.commits_match("abc", tree_commit)


def test_commit_file_is_the_git_archive_authority(tmp_path):
    fake = tmp_path / "tree"
    (fake / "src" / "chimera_omni").mkdir(parents=True)
    (fake / "src" / "chimera_omni" / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(tool.Refusal):
        tool.resolve_tree_commit(fake)
    (fake / "COMMIT").write_text("# extracted from git archive\nC5E6052d31350ce9267e68c49f924abc2b608c89\n", encoding="utf-8")
    commit, how = tool.resolve_tree_commit(fake)
    assert commit == "c5e6052d31350ce9267e68c49f924abc2b608c89" and how == "COMMIT file"


def test_declared_swish_gate_needs_the_waiver(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "qwen_swish", tie=False, output_gate_type="swish")
    out = tmp_path / "swish"
    proc = run_tool(["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out)], expect=3)
    assert "REFUSED" in proc.stderr and "allow_unverified_output_gate" in proc.stderr
    assert not out.exists()
    run_tool(
        ["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out), "--allow-unverified-output-gate"]
    )
    receipt = receipt_of(out)
    waiver = receipt["output_gate_waiver"]
    assert waiver["config_metadata_attention_output_gate"] == "sigmoid_assumed_declared_swish"
    assert "Q1 resolved" in waiver["q1_observation"]


# --------------------------------------------------------------------------
# review fixes, one test each
# --------------------------------------------------------------------------


def test_overwrite_clears_the_previous_layout_and_receipt_records_directory_matches_index(tmp_path, tree_commit):
    """Major: a re-run with --overwrite under a new shard layout must not leave stale shards
    that oc_seal (which walks the directory, not the index) would count."""

    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    out = tmp_path / "relayout"
    common = ["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out)]
    run_tool([*common, "--shard-bytes", "20000"])
    first = receipt_of(out)
    many = first["checkpoint"]["index_shard_files"]
    assert len(many) > 1 and _safetensors_under(out) == many
    assert first["checkpoint"]["shard_bytes"] == 20000

    # without --overwrite: refused, nothing touched
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    proc = run_tool(common, expect=3)
    assert "pass --overwrite" in proc.stderr
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before

    # with --overwrite under the default layout: exactly one shard remains, the stale ones are gone
    summary = summary_of(run_tool([*common, "--overwrite"]))
    receipt = receipt_of(out)
    assert _safetensors_under(out) == [ONE_SHARD]
    assert receipt["checkpoint"]["index_shard_files"] == [ONE_SHARD]
    assert receipt["checkpoint"]["directory_shard_files"] == [ONE_SHARD]
    assert receipt["checkpoint"]["directory_shard_files_match_index"] is True
    assert summary["directory_shard_files_match_index"] is True
    assert receipt["checkpoint"]["overwrite"] is True
    removed = receipt["checkpoint"]["overwrite_removed"]
    assert set(many) <= set(removed)
    assert {INDEX, "config.json", "import_receipt.json", "import_receipt.json.sha256"} <= set(removed)
    assert not (out / "REFUSED.json").exists()
    # the header count is the single-shard one, not the 12-shard one plus stale files
    assert receipt["checkpoint"]["header_element_count"] == first["checkpoint"]["header_element_count"]
    assert sum(tool.header_element_counts(out).values()) == receipt["checkpoint"]["header_element_count"]

    # a stray shard in a subdirectory is not the tool's to delete: refused even with --overwrite,
    # and the present checkpoint is left exactly as it was
    stray = out / "extra" / "stray.safetensors"
    stray.parent.mkdir()
    stray.write_bytes(b"not a checkpoint")
    before = {p.relative_to(out).as_posix(): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    proc = run_tool([*common, "--overwrite"], expect=3)
    assert "subdirectory" in proc.stderr and "extra/stray.safetensors" in proc.stderr
    assert {p.relative_to(out).as_posix(): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == before


def test_compare_to_records_a_lane_comparison_in_the_receipt(tmp_path, tree_commit):
    """Major: the receipt itself must name the compared checkpoint, which tensors match,
    which differ and why (random-init vs source-derived), and both state_sha256 values."""

    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    common = ["--src", str(src), "--chimera-commit", tree_commit]
    lane = tmp_path / "lane"
    run_tool([*common, "--out", str(lane)])
    lane_receipt = receipt_of(lane)

    same = tmp_path / "same"
    summary = summary_of(run_tool([*common, "--out", str(same), "--compare-to", str(lane)]))
    block = receipt_of(same)["lane_comparison"]
    assert block["checkpoint"] == str(lane.resolve())
    assert block["tensors_in_lane"] == block["tensors_in_mine"] == block["shared_count"] == block["identical_count"]
    assert block["differing"] == [] and block["only_in_lane"] == [] and block["only_in_mine"] == []
    assert block["state_sha256_lane"] == block["state_sha256_mine"] == lane_receipt["checkpoint"]["state_sha256"]
    assert block["state_sha256_equal"] is True and block["byte_identical"] is True
    assert block["weight_map_equal"] and block["tied_weights_equal"] and block["total_size_equal"]
    assert block["config_json_differing_paths"] == []
    assert block["index_sha256"] == lane_receipt["checkpoint"]["index_sha256"]
    assert summary["lane_byte_identical"] is True

    # a different seed re-draws exactly the random-init interface tensors and nothing else
    reseeded = tmp_path / "reseeded"
    summary = summary_of(run_tool([*common, "--out", str(reseeded), "--seed", "1", "--compare-to", str(lane)]))
    receipt = receipt_of(reseeded)
    block = receipt["lane_comparison"]
    random_init = {item["name"] for item in receipt["interface_tensors"]}
    assert len(random_init) == 16
    assert block["differing_source_derived"] == []
    assert set(block["differing"]) == set(block["differing_random_init"]) <= random_init
    assert block["differing_count"] > 0
    assert block["identical_count"] + block["differing_count"] == block["shared_count"]
    assert block["state_sha256_lane"] == lane_receipt["checkpoint"]["state_sha256"]
    assert block["state_sha256_mine"] == receipt["checkpoint"]["state_sha256"] != block["state_sha256_lane"]
    assert block["state_sha256_equal"] is False and block["byte_identical"] is False
    assert summary["lane_byte_identical"] is False
    assert block["config_json_differing_paths"] == []  # the seed is not in the manifest

    # a --compare-to without an index is refused before anything is written
    proc = run_tool([*common, "--out", str(tmp_path / "never"), "--compare-to", str(tmp_path)], expect=3)
    assert "has no model.safetensors.index.json" in proc.stderr and not (tmp_path / "never").exists()


def test_profile_auto_selects_the_27b_profile_by_geometry_and_tensor_bytes_are_profile_independent(
    tmp_path, tree_commit
):
    """Major: the S0 spec input is chimera_omni_27b_profile; `auto` must pick it for the 27B
    source, and the choice must change only config.json's config.metadata, never the bytes."""

    from chimera_omni.importing.profiles import QWEN38_27B_VERIFIED_GEOMETRY

    geometry = dict(QWEN38_27B_VERIFIED_GEOMETRY)
    assert geometry["hidden_size"] == 5120 and geometry["num_hidden_layers"] == 64
    assert tool.resolve_profile({"text_config": geometry}, "auto", geometry) == "chimera_omni_27b"
    assert tool.resolve_profile({"text_config": {**geometry, "hidden_size": 1024}}, "auto", geometry) == "qwen35"
    assert tool.resolve_profile(_hf_config(tie=False), "auto", geometry) == "qwen35"
    assert tool.resolve_profile(_hf_config(tie=False), "chimera_omni_27b", geometry) == "chimera_omni_27b"
    assert tool.resolve_profile(_hf_config(tie=False), "qwen35", geometry) == "qwen35"

    src = _write_synthetic_qwen(tmp_path / "qwen_swish", tie=False, output_gate_type="swish")
    common = ["--src", str(src), "--chimera-commit", tree_commit, "--allow-unverified-output-gate"]
    auto = tmp_path / "auto"
    explicit = tmp_path / "explicit27b"
    s_auto = summary_of(run_tool([*common, "--out", str(auto)]))
    s_27b = summary_of(run_tool([*common, "--out", str(explicit), "--profile", "chimera_omni_27b"]))
    r_auto, r_27b = receipt_of(auto), receipt_of(explicit)
    assert s_auto["profile"] == r_auto["profile"] == "qwen35" and r_auto["profile_requested"] == "auto"
    assert s_27b["profile"] == r_27b["profile"] == "chimera_omni_27b" and r_27b["profile_requested"] == "chimera_omni_27b"

    # bytes: identical shards and index; state_sha256 equal
    for name in (ONE_SHARD, INDEX):
        assert (auto / name).read_bytes() == (explicit / name).read_bytes(), name
    assert s_auto["state_sha256"] == s_27b["state_sha256"]
    # config.json: only config.metadata differs, and the 27B stamps are there beside the waiver
    cfg_auto = json.load(open(auto / "config.json", encoding="utf-8"))
    cfg_27b = json.load(open(explicit / "config.json", encoding="utf-8"))
    diff = tool.json_diff_paths(cfg_auto, cfg_27b)
    assert diff and all(path.startswith("config.metadata.") for path in diff), diff
    meta = cfg_27b["config"]["metadata"]
    assert meta["profile"] == "chimera_omni_27b" and meta["geometry_source"] == "checkpoint_config_json"
    assert meta["attention_output_gate"] == "sigmoid_assumed_declared_swish"
    assert "profile" not in cfg_auto["config"]["metadata"]
    assert r_27b["output_gate_waiver"]["config_metadata_attention_output_gate"] == "sigmoid_assumed_declared_swish"


def test_shard_bytes_zero_or_negative_is_refused_before_anything_is_written(tmp_path, tree_commit):
    """Minor: `--shard-bytes 0` used to fall back silently to the default layout."""

    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    for value in ("0", "-5"):
        out = tmp_path / f"shards_{value}"
        proc = run_tool(["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out), "--shard-bytes", value], expect=3)
        assert "--shard-bytes must be positive" in proc.stderr
        assert not out.exists()
    # a positive value is passed through unchanged
    out = tmp_path / "shards_ok"
    run_tool(["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out), "--shard-bytes", "20000"])
    assert receipt_of(out)["checkpoint"]["shard_bytes"] == 20000


def test_post_write_refusal_unlinks_the_products_and_leaves_a_refused_marker(tmp_path, tree_commit):
    """Minor: a refusal raised after write_safetensors must not leave a sealable checkpoint
    with no receipt beside it. The verifier and the cleanup are exercised directly (no
    end-to-end trigger exists: every known precondition refuses before the write)."""

    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    out = tmp_path / "ckpt"
    argv = ["--src", str(src), "--chimera-commit", tree_commit, "--out", str(out)]
    run_tool(argv)
    index = json.load(open(out / INDEX, encoding="utf-8"))
    cfg = types.SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB + 4096, tie_word_embeddings=False)
    facts = tool.verify_written_checkpoint(out, index, cfg)
    assert facts["directory_shard_files"] == facts["index_shard_files"] == [ONE_SHARD]
    assert facts["directory_shard_files_match_index"] is True

    # the verifier sees what oc_seal would count: a stray shard anywhere under --out
    stray = out / "nested" / "stray.safetensors"
    stray.parent.mkdir()
    stray.write_bytes(b"stray")
    with pytest.raises(tool.Refusal, match="different safetensors set"):
        tool.verify_written_checkpoint(out, index, cfg)
    stray.unlink()
    stray.parent.rmdir()
    # ... and a header that disagrees with the config
    with pytest.raises(tool.Refusal, match="embed.weight header holds"):
        tool.verify_written_checkpoint(out, index, types.SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB, tie_word_embeddings=False))

    # cleanup after such a refusal: the run's products go, REFUSED.json names the reason
    marker = tool.cleanup_after_refusal(out, index, "synthetic post-write refusal", argv)
    assert not (out / ONE_SHARD).exists() and not (out / INDEX).exists() and not (out / "config.json").exists()
    assert _safetensors_under(out) == []
    assert set(marker["removed"]) == {ONE_SHARD, INDEX, "config.json", "import_receipt.json", "import_receipt.json.sha256"}
    on_disk = json.load(open(out / "REFUSED.json", encoding="utf-8"))
    assert on_disk["schema"] == "chimera_omni.import.refused.v1"
    assert on_disk["reason"] == "synthetic post-write refusal"
    assert on_disk["tool"]["argv"] == argv and on_disk["remaining_safetensors"] == []
    assert sorted(p.name for p in out.iterdir()) == ["REFUSED.json"]  # nothing sealable, no receipt beside nothing

    # a fresh run into that directory proceeds (no shards) and replaces the marker
    run_tool(argv)
    assert not (out / "REFUSED.json").exists()
    assert "REFUSED.json" in receipt_of(out)["checkpoint"]["overwrite_removed"]
    assert _safetensors_under(out) == [ONE_SHARD]


def test_fp16_needs_the_narrowing_waiver_and_the_receipt_records_the_range(tmp_path, tree_commit):
    """Minor: fp16 narrows bf16-range weights; refuse by default, range-check when waived."""

    src = _write_synthetic_qwen(tmp_path / "qwen_untied", tie=False)
    common = ["--src", str(src), "--chimera-commit", tree_commit]
    out = tmp_path / "fp16"
    proc = run_tool([*common, "--out", str(out), "--dtype-out", "fp16"], expect=3)
    assert "--allow-fp16-narrowing" in proc.stderr and not out.exists()

    run_tool([*common, "--out", str(out), "--dtype-out", "fp16", "--allow-fp16-narrowing"])
    receipt = receipt_of(out)
    assert receipt["dtype_out"] == "float16"
    narrowing = receipt["narrowing"]
    assert narrowing["target_dtype"] == "torch.float16" and narrowing["target_max"] == 65504.0
    assert narrowing["nonfinite_after_cast"] == 0 and narrowing["overflow_count"] == 0
    per_tensor = narrowing["per_tensor"]
    assert set(per_tensor) == set(json.load(open(out / INDEX, encoding="utf-8"))["weight_map"])
    assert all(
        entry.get("alias") or (entry["max_abs_fp32"] < 65504.0 and entry["nonfinite_after_cast"] == 0)
        for entry in per_tensor.values()
    )
    from chimera_omni.importing.safetensors_io import checkpoint_dtypes

    assert set(checkpoint_dtypes(out)) == {"F16"}
    # bf16 (the campaign dtype) and fp32 need no waiver and record no narrowing
    plain = tmp_path / "bf16"
    run_tool([*common, "--out", str(plain)])
    assert receipt_of(plain)["narrowing"] is None and set(checkpoint_dtypes(plain)) == {"BF16"}


def test_commit_binding_uses_the_actual_tree_even_when_package_bytes_are_identical(tree_commit):
    """A runner may use a newer checkout; its own commit must be named accurately."""

    assert tool.commits_match(tree_commit[:7], tree_commit)
    assert tool.commits_match("c5e6052", tree_commit) == tree_commit.startswith("c5e6052")
    assert not tool.commits_match("5a87018", "c5e6052")


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("relation", ["same", "output_below_source", "source_below_output"])
def test_source_output_overlap_preserves_every_existing_file(tmp_path, tree_commit, relation):
    parent = tmp_path / "parent"
    src = parent / "nested" if relation == "source_below_output" else parent
    _write_synthetic_qwen(src)
    # Use the sharded filename that the old --overwrite loop would delete.
    (src / "model.safetensors").rename(src / ONE_SHARD)
    out = src / "child" if relation == "output_below_source" else parent
    before = _file_bytes(parent)
    proc = run_tool([
        "--src", str(src), "--out", str(out), "--overwrite", "--chimera-commit", tree_commit,
    ], expect=3)
    assert "overlaps --src" in proc.stderr
    assert _file_bytes(parent) == before


def test_symlinked_output_alias_cannot_overwrite_source(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "parent")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(src, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable for this host/account")
    before = _file_bytes(src)
    proc = run_tool([
        "--src", str(src), "--out", str(alias), "--overwrite", "--chimera-commit", tree_commit,
    ], expect=3)
    assert "overlaps --src" in proc.stderr and _file_bytes(src) == before


@pytest.mark.parametrize("invalid", [
    ["--ledger-band", "control:10"],
    ["--ledger-band", "music_codes:10", "--ledger-band", "music_codes:10"],
    [],  # declared swish with no waiver: the profile itself refuses
])
def test_invalid_profile_or_band_preserves_previous_output(tmp_path, tree_commit, invalid):
    src = _write_synthetic_qwen(tmp_path / "parent", output_gate_type="swish" if not invalid else None)
    out = tmp_path / "old-output"
    out.mkdir()
    (out / "config.json").write_text('{"schema":"chimera_omni.checkpoint.v1"}', encoding="utf-8")
    (out / ONE_SHARD).write_bytes(b"previous checkpoint must survive")
    before = _file_bytes(out)
    run_tool([
        "--src", str(src), "--out", str(out), "--overwrite", "--chimera-commit", tree_commit, *invalid,
    ], expect=3)
    assert _file_bytes(out) == before


def test_unrelated_hf_checkpoint_is_not_an_overwrite_target(tmp_path):
    other_parent = tmp_path / "another-parent"
    other_parent.mkdir()
    (other_parent / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    (other_parent / ONE_SHARD).write_bytes(b"registered parent")
    before = _file_bytes(other_parent)
    with pytest.raises(tool.Refusal, match="non-Chimera checkpoint"):
        tool.prepare_out_dir(other_parent, overwrite=True)
    assert _file_bytes(other_parent) == before


def test_unknown_top_level_shard_refuses_before_clearing_known_products(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / ONE_SHARD).write_bytes(b"previous known shard")
    (out / "unrelated.safetensors").write_bytes(b"unrelated data")
    before = _file_bytes(out)
    with pytest.raises(tool.Refusal, match="not the tool's files"):
        tool.prepare_out_dir(out, overwrite=True)
    assert _file_bytes(out) == before


def test_custom_receipt_cannot_replace_source_or_output_config(tmp_path, tree_commit):
    src = _write_synthetic_qwen(tmp_path / "parent")
    out = tmp_path / "unwritten-output"
    before = _file_bytes(src)
    for receipt in (src / "config.json", out / "config.json"):
        proc = run_tool([
            "--src", str(src), "--out", str(out), "--receipt", str(receipt),
            "--chimera-commit", tree_commit,
        ], expect=3)
        assert "REFUSED" in proc.stderr and not out.exists()
        assert _file_bytes(src) == before


def test_archive_source_manifest_tracks_actual_bytes_not_the_commit_label(tmp_path):
    package = tmp_path / "src" / "chimera_omni"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "COMMIT").write_text("c5e6052\n", encoding="utf-8")
    before = tool.executable_source_manifest(tmp_path)
    assert before["files"] == {"src/chimera_omni/__init__.py": tool.file_sha256(module)}
    assert before["checkout_executable_files_clean"] is None
    module.write_text("VALUE = 2\n", encoding="utf-8")
    after = tool.executable_source_manifest(tmp_path)
    assert before["sha256"] != after["sha256"]
    assert tool.resolve_tree_commit(tmp_path)[0] == "c5e6052"


def test_dirty_executable_checkout_is_refused_but_document_edits_are_allowed(tmp_path):
    package = tmp_path / "src" / "chimera_omni"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("Original\n", encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True)

    git("init")
    git("add", ".")
    git("-c", "user.name=S0 test", "-c", "user.email=s0@example.invalid", "commit", "-m", "fixture")
    before = tool.executable_source_manifest(tmp_path)
    assert before["checkout_executable_files_clean"] is True
    readme.write_text("User documentation edit\n", encoding="utf-8")
    assert tool.executable_source_manifest(tmp_path) == before
    module.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(tool.Refusal, match="dirty executable source"):
        tool.executable_source_manifest(tmp_path)
    git("add", "src/chimera_omni/__init__.py")
    with pytest.raises(tool.Refusal, match="dirty executable source"):
        tool.executable_source_manifest(tmp_path)


# --------------------------------------------------------------------------
# real 0.8B (skip when absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (QWEN_0P8B / "config.json").is_file(), reason="Qwen3.5-0.8B not on this host")
@pytest.mark.skipif(not (LANE_CKPT / INDEX).is_file(), reason="import-lane s4-fix checkpoint absent")
def test_0p8b_without_band_reproduces_the_import_lane_checkpoint(tmp_path, tree_commit):
    """Every source-derived tensor, every band row and the tied alias are byte-identical to the
    lane's s4-fix checkpoint, and the receipt's `lane_comparison` block says so itself.

    The lane checkpoint was written by scripts/import_receipt_qwen35.py at 605dbe9, which
    loaded the HF reference BEFORE the import; the reference's construction consumed the
    CPU RNG, so the random-init interface tensors (no Qwen source) carry that run's RNG
    history and are reproducible only from this tool's seed. The comparison is therefore
    recorded, not assumed: `differing_source_derived` must be empty, every difference must
    be a random-init interface tensor, and both state_sha256 values are in the receipt.
    Downstream sealing keys on this tool's state_sha256, never on the R-Q receipt's.
    """

    out = tmp_path / "q08b_plain"
    summary = summary_of(
        run_tool(["--src", str(QWEN_0P8B), "--chimera-commit", tree_commit, "--out", str(out), "--compare-to", str(LANE_CKPT)])
    )
    receipt = receipt_of(out)
    assert receipt["profile"] == "qwen35" and receipt["profile_requested"] == "auto"  # 0.8B geometry is not the 27B's
    random_init = {item["name"] for item in receipt["interface_tensors"]}
    assert len(random_init) == 16

    block = receipt["lane_comparison"]
    assert block["checkpoint"] == str(LANE_CKPT.resolve())
    assert block["tensors_in_lane"] == block["tensors_in_mine"] == block["shared_count"] == 390
    assert block["only_in_lane"] == [] and block["only_in_mine"] == []
    assert block["weight_map_equal"] and block["tied_weights_equal"] and block["total_size_equal"]
    assert block["differing_source_derived"] == [], block["differing_source_derived"]
    assert set(block["differing"]) == set(block["differing_random_init"]) <= random_init
    assert block["identical_count"] == 390 - block["differing_count"] >= 390 - 16
    assert block["config_json_differing_paths"] == ["chimera_commit"]  # the lane left it null
    assert len(block["state_sha256_lane"]) == 64 and block["state_sha256_mine"] == summary["state_sha256"]
    assert block["state_sha256_equal"] == (block["differing_count"] == 0) == block["byte_identical"]

    lane_index = json.load(open(LANE_CKPT / INDEX, encoding="utf-8"))
    mine_index = json.load(open(out / INDEX, encoding="utf-8"))
    shards = sorted(set(lane_index["weight_map"].values()))
    assert shards == receipt["checkpoint"]["index_shard_files"] == receipt["checkpoint"]["directory_shard_files"]
    for name in shards:
        assert (out / name).stat().st_size == (LANE_CKPT / name).stat().st_size, name
    assert mine_index["metadata"]["tied_weights"] == lane_index["metadata"]["tied_weights"] == {"lm_head.weight": "embed.weight"}
    lane_sha = lane_index["metadata"]["tensor_sha256"]
    mine_sha = mine_index["metadata"]["tensor_sha256"]
    source_derived = set(lane_sha) - random_init
    assert len(source_derived) == 390 - 16
    assert all(lane_sha[name] == mine_sha[name] for name in source_derived)
    assert sorted(name for name in lane_sha if lane_sha[name] != mine_sha[name]) == block["differing"]

    mine = json.load(open(out / "config.json", encoding="utf-8"))
    lane = json.load(open(LANE_CKPT / "config.json", encoding="utf-8"))
    assert lane["chimera_commit"] is None and mine["chimera_commit"] == tree_commit
    mine.pop("chimera_commit")
    lane.pop("chimera_commit")
    assert mine == lane

    # the tool reproduces ITSELF byte-for-byte
    again = tmp_path / "q08b_plain_again"
    summary2 = summary_of(run_tool(["--src", str(QWEN_0P8B), "--chimera-commit", tree_commit, "--out", str(again)]))
    for name in (*shards, INDEX, "config.json"):
        assert tool.file_sha256(out / name) == tool.file_sha256(again / name), name
    assert summary["state_sha256"] == summary2["state_sha256"]


@pytest.mark.skipif(not (QWEN_0P8B / "config.json").is_file(), reason="Qwen3.5-0.8B not on this host")
def test_0p8b_with_music_band_grows_headers_by_65536_x_hidden_x_tables(tmp_path, tree_commit):
    plain = LANE_CKPT if (LANE_CKPT / INDEX).is_file() else tmp_path / "q08b_plain"
    if plain != LANE_CKPT:
        run_tool(["--src", str(QWEN_0P8B), "--chimera-commit", tree_commit, "--out", str(plain)])
    banded = tmp_path / "q08b_music"
    summary = summary_of(
        run_tool(
            [
                "--src", str(QWEN_0P8B), "--chimera-commit", tree_commit, "--out", str(banded),
                "--ledger-band", f"music_codes:{MUSIC_BAND}",
            ]
        )
    )
    manifest = json.load(open(banded / "config.json", encoding="utf-8"))["config"]
    hidden = int(manifest["hidden_size"])
    base = int(manifest["ledger_base_vocab_size"])
    tables = 1 if manifest["tie_word_embeddings"] else 2
    growth = sum(tool.header_element_counts(banded).values()) - sum(tool.header_element_counts(plain).values())
    assert growth == MUSIC_BAND * hidden * tables, (growth, hidden, tables)
    assert summary["extra_band_elements_declared"] == MUSIC_BAND * hidden * tables
    zero, _tied = _band_rows_zero(banded, base)
    assert zero
    assert manifest["ledger_bands"][-1] == ["music_codes", MUSIC_BAND]
    assert manifest["vocab_size"] == base + 4096 + MUSIC_BAND
