"""SQLite-backed key/value cache (embeddings, parsed files, query embeddings).

Keys are deterministic and *content-addressed*: sha1(repo | file | qualified name | code-hash) plus the
embedding-model version. The commit hash is deliberately NOT part of the key, otherwise an unchanged
function would be re-embedded on every new commit -- the opposite of what incremental indexing needs.
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


class KVCache:
    def __init__(self, path: str | Path | None):
        self.path = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, val BLOB)")
        self.hits = 0
        self.misses = 0

    def get_many(self, keys: Iterable[str]) -> dict[str, bytes]:
        keys = list(keys)
        out: dict[str, bytes] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT key, val FROM kv WHERE key IN ({','.join('?' * len(part))})"
            out.update(self.db.execute(q, part).fetchall())
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def get(self, key: str) -> bytes | None:
        return self.get_many([key]).get(key)

    def set_many(self, items: dict[str, bytes]) -> None:
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO kv (key, val) VALUES (?, ?)", list(items.items()))

    def set(self, key: str, val: bytes) -> None:
        self.set_many({key: val})

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]

    def close(self) -> None:
        self.db.close()


class EmbeddingCache:
    """chunk.cache_key + model_version -> float32 vector."""

    def __init__(self, kv: KVCache, model_version: str, dim: int):
        self.kv, self.model_version, self.dim = kv, model_version, dim

    def _k(self, chunk_key: str) -> str:
        return f"emb|{self.model_version}|{chunk_key}"

    def get_many(self, chunk_keys: list[str]) -> dict[str, np.ndarray]:
        found = self.kv.get_many([self._k(k) for k in chunk_keys])
        out = {}
        for k in chunk_keys:
            v = found.get(self._k(k))
            if v is not None:
                arr = np.frombuffer(v, dtype=np.float32)
                if arr.shape[0] == self.dim:  # guards against a stale entry from a different-dim model
                    out[k] = arr
        return out

    def set_many(self, vecs: dict[str, np.ndarray]) -> None:
        self.kv.set_many({self._k(k): np.asarray(v, dtype=np.float32).tobytes() for k, v in vecs.items()})


def dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads(b: bytes):
    return pickle.loads(b)
