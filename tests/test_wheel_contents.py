import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel_members():
    if shutil.which("uv") is None:
        pytest.skip("uv is not available")

    subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = sorted(
        (REPO_ROOT / "dist").glob("*.whl"),
        key=lambda wheel: wheel.stat().st_mtime,
    )
    assert wheels, "uv build --wheel did not produce a wheel"

    with zipfile.ZipFile(wheels[-1]) as wheel:
        return wheel.namelist()


def test_wheel_contains_the_flat_utils_package(wheel_members):
    assert any(member.startswith("utils/") for member in wheel_members)
    assert "utils/__init__.py" in wheel_members
    assert "utils/preprocessing.py" in wheel_members
    assert "utils/classification_evals_and_tuning.py" in wheel_members


def test_wheel_excludes_tests_and_notebooks(wheel_members):
    assert not any(member.startswith("tests/") for member in wheel_members)
    assert not any(member.startswith("notebooks/") for member in wheel_members)
