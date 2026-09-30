"""Core data structures shared across the pipeline."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def content_hash(code: str) -> str:
    norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Chunk:
    repository: str
    file: str
    language: str
    code: str
    start_line: int
    end_line: int
    commit: str = ""
    version: str = ""
    timestamp: float = 0.0
    class_name: str = ""
    function: str = ""
    kind: str = "function"       # function | method | class | block | module | file
    parent: str = ""             # parent function when a large function was split into blocks
    part: int = 0                # block index inside the parent function
    signature: str = ""
    doc: str = ""
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    chash: str = ""
    doc_id: str = ""            # benchmark/document id when the chunk stands for a whole corpus document

    def __post_init__(self) -> None:
        if not self.chash:
            self.chash = content_hash(self.code)

    @property
    def qualname(self) -> str:
        name = self.function or self.class_name or "<module>"
        if self.function and self.class_name:
            name = f"{self.class_name}.{self.function}"
        return f"{name}#{self.part}" if self.part else name

    @property
    def identity(self) -> str:
        """Stable across commits: same file + same qualified name = same logical unit."""
        return f"{self.repository}::{self.file}::{self.kind if self.kind in ('class','module') else 'fn'}::{self.qualname}"

    @property
    def cache_key(self) -> str:
        """Content-addressed: unchanged code keeps its key across commits (see indexing/cache.py)."""
        return hashlib.sha1(f"{self.repository}|{self.file}|{self.qualname}|{self.chash}".encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    method: str = "hybrid"
    rank: int = 0
    features: dict[str, float] = field(default_factory=dict)
    also_in: list[str] = field(default_factory=list)   # other versions collapsed into this result
    explanation: str = ""
    vec: Any = field(default=None, repr=False, compare=False)   # chunk embedding, used by cross-version dedup
