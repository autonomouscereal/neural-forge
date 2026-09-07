"""Source/recipe checks only; these do not establish a built FPS runtime."""
import ast
from pathlib import Path
import shlex


ROOT = Path(__file__).parent


def test_exact_official_wheel_is_the_only_pinned_install():
    rows = [row for row in (ROOT / "requirements-hy3d-fps.txt").read_text().splitlines()
            if row and not row.startswith("#")]
    assert len(rows) == 1
    assert shlex.split(rows[0]) == [
        "torch-cluster", "@",
        "https://data.pyg.org/whl/torch-2.6.0%2Bcu124/torch_cluster-1.6.3%2Bpt26cu124-cp310-cp310-linux_x86_64.whl",
        "--hash=sha256:707bc36cccd885cb18155f3e1718866c4f4e6df4160a00ee17b9457a41bf530b"]
    docker = (ROOT / "Dockerfile.hy3d-capture").read_text()
    assert "sha256:735020f27cb59da7ca4ea62284154ad97fcbc1a765fa8ffc6c0abd9a2d283c0a" in docker
    assert "FROM local/hy3d-base:735020f27cb59da7" in docker
    runs = [row for row in docker.splitlines() if row.startswith("RUN ")]
    assert len(runs) == 2
    for flag in ("--require-hashes", "--no-deps", "--no-index", "--no-cache-dir"):
        assert flag in runs[0]
    assert "--upgrade" not in runs[0] and "xformers" not in runs[0]
    assert "CUDA_VISIBLE_DEVICES=-1" in runs[1] and "hy3d_fps_self_test.py" in runs[1]
    assert "ENTRYPOINT []" in docker and "HEALTHCHECK NONE" in docker


def test_fps_canary_calls_actual_native_api_and_compiled_operator_on_cpu():
    source = (ROOT / "hy3d_fps_self_test.py").read_text()
    tree = ast.parse(source)
    compile(tree, "hy3d_fps_self_test.py", "exec")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    fps = [node for node in calls if ast.unparse(node.func) == "torch_cluster.fps"]
    assert len(fps) == 1 and len(fps[0].args) == 6
    assert any(ast.unparse(node.func) == "torch.ops.torch_cluster.fps" for node in calls)
    assert "torch.cuda.is_initialized()" in source
    assert "device=\"cpu\"" in source and ".cuda(" not in source
    assert "\"gpu_tested\": False" in source and "\"native_vae_tested\": False" in source
    assert "\"folded_child_proven\": False" in source
