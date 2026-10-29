import tomllib
from pathlib import Path

from packaging.version import Version


def _project_metadata():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_core_dependency_floors():
    dependencies = set(_project_metadata()["dependencies"])
    expected = {
        "scikit-learn>=1.6",
        "pandas>=2.2",
        "seaborn>=0.13",
        "statsmodels>=0.14",
        "numpy>=1.26",
        "scipy>=1.11",
    }

    assert expected <= dependencies


def test_optional_extra_floors():
    extras = _project_metadata()["optional-dependencies"]
    expected_text = {
        "beautifulsoup4>=4.12",
        "contractions>=0.1",
        "nltk>=3.9",
        "Pillow>=10",
        "textblob>=0.18",
        "wordcloud>=1.9",
    }
    expected_translate = {
        "deep-translator>=1.11",
        "langid>=1.1.6",
        "tenacity>=8.2",
    }

    assert expected_text <= set(extras["text"])
    assert expected_translate <= set(extras["translate"])


def test_project_identity_is_unchanged():
    project = _project_metadata()

    assert project["name"] == "data-science-utils"
    assert project["requires-python"] == ">=3.14"


def test_requirements_txt_is_generated_from_pyproject():
    lines = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "generated" in lines[0].lower()
    assert "pyproject.toml" in lines[0]

    scikit_learn = [line for line in lines if line.startswith("scikit-learn==")]
    assert len(scikit_learn) == 1
    pinned_version = scikit_learn[0].split("==", 1)[1].split(";", 1)[0].strip()
    assert Version(pinned_version) >= Version("1.6")
