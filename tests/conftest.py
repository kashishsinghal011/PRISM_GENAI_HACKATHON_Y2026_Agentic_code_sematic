import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aci.config import load_config  # noqa: E402

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"index.dir": str(tmp_path / "idx"), "index.cache_dir": str(tmp_path / "cache")})


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


PRE_V1 = '''def normalize(text):
    """Normalize input text before forwarding it to main."""
    cleaned = text.strip().lower()
    return forward(cleaned)

def forward(value):
    return main(value)

def main(value):
    print(value)
'''
CHECKS_V1 = 'def check(s):\n    pre = s[:6]\n    return pre == "en-US"\n'
ENGINE_V1 = "def perf(s):\n    if act(A, s):\n        return act(B, s)\n"


@pytest.fixture
def git_repo(tmp_path):
    """A 3-commit repo. Returns (path, [c1, c2, c3])."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "demo_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "preprocessing.py").write_text(PRE_V1)
    (repo / "checks.py").write_text(CHECKS_V1)
    (repo / "engine.py").write_text(ENGINE_V1)
    (repo / "notes.txt").write_text("not source code")
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "c1")
    (repo / "preprocessing.py").write_text(PRE_V1.replace("def normalize(", "def preprocess_input("))
    (repo / "checks.py").write_text(CHECKS_V1 + '\ndef validate_token(token):\n    """Validate the JWT token before accessing the database."""\n    return jwt.decode(token, SECRET)\n')
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "c2")
    (repo / "preprocessing.py").write_text(PRE_V1.replace("def normalize(", "def sanitize_input("))
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "c3")
    out = subprocess.run(["git", "-C", str(repo), "rev-list", "--reverse", "HEAD"], capture_output=True, text=True, check=True).stdout.split()
    return repo, out
