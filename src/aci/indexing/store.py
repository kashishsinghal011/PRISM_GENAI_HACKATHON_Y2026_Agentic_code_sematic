"""A loaded (or freshly assembled) index for one version of one repository."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..errors import IndexNotFoundError, MissingEmbeddingsError
from ..ingestion.chunker import lexical_text
from ..models import Chunk
from ..tokenize import tokenize
from .lexical_index import BM25Index, IdentifierIndex
from .vector_index import VectorIndex


def _vindex(vectors: np.ndarray, cfg: dict) -> VectorIndex:
    ic = cfg["index"]
    return VectorIndex(vectors.shape[1], ic.get("ann", "hnsw"), ic.get("flat_threshold", 2000), ic.get("hnsw_m", 32),
                       ic.get("hnsw_ef_construction", 100), ic.get("hnsw_ef_search", 128))


class VersionIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder, cfg: dict, bm25: BM25Index,
                 vindex: VectorIndex, manifest: dict | None = None, path: Path | None = None):
        self.chunks, self.vectors, self.embedder, self.cfg = chunks, vectors, embedder, cfg
        self.bm25, self.vindex, self.manifest, self.path = bm25, vindex, manifest or {}, path
        self.ident = IdentifierIndex().build(chunks)

    @staticmethod
    def lexical_docs(chunks: list[Chunk], cfg: dict) -> list[list[str]]:
        raw = cfg.get("representation") == "raw"
        return [tokenize(c.code[:2000] if raw else lexical_text(c)) for c in chunks]

    @classmethod
    def assemble(cls, chunks: list[Chunk], vectors: np.ndarray, embedder, cfg: dict, manifest: dict | None = None) -> "VersionIndex":
        if len(chunks) != len(vectors):
            raise MissingEmbeddingsError(f"{len(chunks)} chunks but {len(vectors)} embeddings.")
        bm25 = BM25Index().build(cls.lexical_docs(chunks, cfg))
        vindex = _vindex(vectors, cfg).build(vectors)
        return cls(chunks, vectors, embedder, cfg, bm25, vindex, manifest)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "chunks.jsonl.gz", "wt", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        np.save(d / "vectors.npy", self.vectors)
        self.bm25.save(d / "bm25.pkl")
        self.vindex.save(d)
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
        self.path = d

    @staticmethod
    def load_chunks(d: Path) -> list[Chunk]:
        with gzip.open(d / "chunks.jsonl.gz", "rt", encoding="utf-8") as f:
            return [Chunk.from_dict(json.loads(l)) for l in f]

    @classmethod
    def load(cls, directory: str | Path, cfg: dict, embedder) -> "VersionIndex":
        d = Path(directory)
        need = ["manifest.json", "chunks.jsonl.gz", "vectors.npy", "bm25.pkl"]
        missing = [n for n in need if not (d / n).exists()]
        if missing:
            raise IndexNotFoundError(f"Index at {d} is incomplete or missing (absent: {', '.join(missing)}). Re-run the indexer.")
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest.get("embedder") and manifest["embedder"] != embedder.model_version:
            raise MissingEmbeddingsError(f"Index was built with embedder '{manifest['embedder']}' but '{embedder.model_version}' is configured. "
                                         "Rebuild the index or restore the previous embedding_model config.")
        vectors = np.load(d / "vectors.npy")
        vindex = VectorIndex.load(d, vectors, ann=cfg["index"].get("ann", "hnsw"), flat_threshold=cfg["index"].get("flat_threshold", 2000),
                                  hnsw_m=cfg["index"].get("hnsw_m", 32), ef_construction=cfg["index"].get("hnsw_ef_construction", 100),
                                  ef_search=cfg["index"].get("hnsw_ef_search", 128))
        return cls(cls.load_chunks(d), vectors, embedder, cfg, BM25Index.load(d / "bm25.pkl"), vindex, manifest, d)
