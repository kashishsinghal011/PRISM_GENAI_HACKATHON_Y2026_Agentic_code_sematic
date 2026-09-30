"""Chunking entry point + the multiple text representations of a chunk."""
from __future__ import annotations

import logging
import re

from ..models import Chunk
from ..tokenize import split_identifier
from .parser import parse_source

log = logging.getLogger(__name__)
MAX_CODE_CHARS = 2000  # embedding/BM25 text budget per chunk


def chunk_file(source: str, path: str, repository: str, commit: str = "", version: str = "",
               timestamp: float = 0.0, max_chunk_lines: int = 60, max_file_bytes: int = 1_000_000) -> list[Chunk]:
    """Parse one file into chunks. Very large files are skipped (returns [])."""
    if len(source.encode("utf-8", "replace")) > max_file_bytes:
        log.warning("Skipping %s: larger than %d bytes", path, max_file_bytes)
        return []
    base = dict(repository=repository, commit=commit, version=version, timestamp=timestamp)
    chunks = parse_source(source, path, base, max_chunk_lines)
    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:  # duplicate chunks (same identity + content) inside a file are dropped
        key = f"{c.identity}|{c.chash}|{c.start_line}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _words(s: str) -> str:
    return " ".join(w for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s) for w in split_identifier(part))


def raw_text(c: Chunk) -> str:
    return c.code[:MAX_CODE_CHARS]


def metadata_text(c: Chunk) -> str:
    """Keyword-style view: names, path words, docstring, calls. Compact and identifier-dense."""
    parts = [f"function {_words(c.function)}" if c.function else "", f"class {_words(c.class_name)}" if c.class_name else "",
             f"file {_words(c.file)}", c.doc[:300], f"calls {_words(' '.join(c.calls[:15]))}",
             f"imports {_words(' '.join(c.imports[:10]))}"]
    return "\n".join(p for p in parts if p.strip())


def contextual_text(c: Chunk) -> str:
    head = [f"File: {c.file}"]
    if c.class_name:
        head.append(f"Class: {c.class_name}")
    if c.function:
        head.append(f"Function: {c.function}")
        head.append(f"Purpose: {_words(c.function)}")
    if c.signature:
        head.append(f"Signature: {c.signature}")
    if c.doc:
        head.append(f"Doc: {c.doc[:300]}")
    if c.imports:
        head.append("Imports: " + ", ".join(c.imports[:10]))
    if c.calls:
        head.append("Calls: " + ", ".join(c.calls[:15]))
    return "\n".join(head) + "\nCode:\n" + c.code[:MAX_CODE_CHARS]


REPRESENTATIONS = {"raw": raw_text, "metadata": metadata_text, "contextual": contextual_text}


def represent(c: Chunk, mode: str = "contextual") -> str:
    try:
        return REPRESENTATIONS[mode](c)
    except KeyError:
        raise ValueError(f"Unknown representation '{mode}'. Choose from {sorted(REPRESENTATIONS)}") from None


def lexical_text(c: Chunk) -> str:
    """BM25 document: names/path repeated (cheap BM25F-style field boost) + doc + code."""
    names = f"{c.function} {c.class_name} {c.parent}"
    return f"{names} {names} {names} {c.file} {c.file} {c.doc} {c.signature} {c.code[:MAX_CODE_CHARS]}"
