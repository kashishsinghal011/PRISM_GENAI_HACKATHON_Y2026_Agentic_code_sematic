"""Repository / commit access. Git repos use the git CLI (ls-tree, cat-file --batch, diff);
plain directories are treated as a single 'WORKTREE' version."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

from ..errors import InvalidCommitError, RepositoryNotFoundError

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", "build", "dist", ".mypy_cache", ".idea"}
WORKTREE = "WORKTREE"


class Repo:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise RepositoryNotFoundError(f"Repository path not found or not a directory: {path}")
        self.name = self.path.name
        self.is_git = self._git("rev-parse", "--git-dir", check=False).returncode == 0 and (self.path / ".git").exists()

    def _git(self, *args: str, check: bool = True, input: bytes | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, check=check, input=input)
        except subprocess.CalledProcessError as e:
            raise InvalidCommitError(e.stderr.decode(errors="replace").strip() or "git command failed") from e
        except FileNotFoundError:
            return subprocess.CompletedProcess(args, 127, b"", b"git not installed")

    # -------------------------------------------------------------- commits
    def resolve_commit(self, ref: str | None) -> str:
        if not self.is_git:
            if ref in (None, "", "HEAD", WORKTREE):
                return WORKTREE
            raise InvalidCommitError(f"'{self.path}' is not a git repository; only the working tree can be indexed (got '{ref}').")
        r = self._git("rev-parse", "--verify", "--quiet", f"{ref or 'HEAD'}^{{commit}}", check=False)
        if r.returncode != 0:
            raise InvalidCommitError(f"Invalid commit/ref '{ref}' in {self.path}")
        return r.stdout.decode().strip()

    def commit_time(self, commit: str) -> float:
        if commit == WORKTREE:
            return time.time()
        return float(self._git("show", "-s", "--format=%ct", commit).stdout.decode().strip() or 0)

    # -------------------------------------------------------------- files
    def list_files(self, commit: str) -> dict[str, tuple[str, int]]:
        """path -> (blob id, size)."""
        out: dict[str, tuple[str, int]] = {}
        if commit == WORKTREE:
            for root, dirs, files in os.walk(self.path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for f in files:
                    full = Path(root) / f
                    try:
                        data = full.read_bytes()
                    except OSError:
                        continue
                    out[full.relative_to(self.path).as_posix()] = (hashlib.sha1(data).hexdigest(), len(data))
            return out
        raw = self._git("ls-tree", "-r", "-l", "-z", commit).stdout.decode("utf-8", "replace")
        for entry in raw.split("\0"):
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            _mode, typ, sha, size = meta.split(None, 3)
            if typ == "blob" and not (set(Path(path).parts) & SKIP_DIRS):
                out[path] = (sha, int(size) if size.strip().isdigit() else 0)
        return out

    def read_files(self, commit: str, files: dict[str, tuple[str, int]], paths: list[str]) -> dict[str, str]:
        """Read many files efficiently (one `git cat-file --batch` process)."""
        result: dict[str, str] = {}
        if commit == WORKTREE:
            for p in paths:
                try:
                    result[p] = (self.path / p).read_bytes().decode("utf-8", "replace")
                except OSError:
                    continue
            return result
        if not paths:
            return result
        proc = subprocess.Popen(["git", "-C", str(self.path), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for p in paths:
                proc.stdin.write((files[p][0] + "\n").encode())
                proc.stdin.flush()
                header = proc.stdout.readline().split()
                if len(header) < 3:
                    continue
                size = int(header[2])
                data = proc.stdout.read(size)
                proc.stdout.read(1)
                result[p] = data.decode("utf-8", "replace")
        finally:
            proc.stdin.close()
            proc.wait()
        return result

    def changed_files(self, base: str, commit: str) -> tuple[list[str], list[str]]:
        """(added_or_modified, deleted) between two commits using `git diff --name-status`."""
        if not self.is_git or WORKTREE in (base, commit):
            raise InvalidCommitError("git diff needs two real commits")
        raw = self._git("diff", "--name-status", "--no-renames", "-z", base, commit).stdout.decode("utf-8", "replace").split("\0")
        changed, deleted = [], []
        it = iter(raw)
        for status in it:
            if not status:
                continue
            path = next(it, "")
            (deleted if status.startswith("D") else changed).append(path)
        return changed, deleted
