"""FAISS cosine similarity and BM25 sparse retrieval."""
import os
import re
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from src.chunking import DocumentChunk, tokens


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    """Use exactly the same lowercase tokenization for documents and queries."""
    return re.findall(r"\w+", text.lower())


def normalize(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero embedding cannot be normalized.")
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    model = "text-embedding-3-small"

    def __init__(self):
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key == "your_openai_key_here":
            raise ValueError("Set OPENAI_API_KEY in .env or hosting secrets.")
        self.client = OpenAI(api_key=key, timeout=45.0, max_retries=1)

    def embed(self, texts: list[str]) -> np.ndarray:
        output = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            if any(not t.strip() or len(tokens(t)) > 8191 for t in batch):
                raise ValueError("Embedding input must contain 1–8191 tokens.")
            response = self.client.embeddings.create(model=self.model, input=batch,
                                                     encoding_format="float")
            output.extend(row.embedding for row in sorted(response.data, key=lambda row: row.index))
        return np.asarray(output, dtype=np.float32)


class BM25Index:
    def __init__(self, chunks: list[DocumentChunk]):
        from rank_bm25 import BM25Okapi
        if not chunks:
            raise ValueError("Cannot index an empty corpus.")
        self.chunks = chunks
        self.words = [tokenize(c.content) for c in chunks]
        self.word_sets = [set(w) for w in self.words]
        self.index = BM25Okapi(self.words)

    def search(self, query: str, k: int) -> list[SearchResult]:
        words = tokenize(query)
        if not words or k <= 0:
            return []
        scores = self.index.get_scores(words)
        # BM25 may assign zero/negative score to common terms. A candidate
        # must share at least one real token; no-match queries return nothing.
        candidates = [i for i, w in enumerate(self.word_sets) if set(words) & w]
        ranked = sorted(candidates, key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))[:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=r)
                for r, i in enumerate(ranked, 1)]


class DenseIndex:
    def __init__(self, chunks: list[DocumentChunk], embedder: Embedder):
        import faiss
        if not chunks:
            raise ValueError("Cannot index an empty corpus.")
        self.chunks = chunks
        self.embedder = embedder
        vectors = normalize(embedder.embed([c.content for c in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunks.")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search_vector(self, vector, k: int) -> list[SearchResult]:
        if k <= 0:
            return []
        query = normalize(np.asarray(vector).reshape(1, -1))
        if query.shape[1] != self.index.d:
            raise ValueError("Embedding dimension mismatch.")
        # Exhaustive FAISS index: all scores allow a stable tie-break by ID.
        scores, ids = self.index.search(query, len(self.chunks))
        ranked = sorted(zip(scores[0], ids[0]),
                        key=lambda row: (-float(row[0]), self.chunks[int(row[1])].chunk_id))[:k]
        return [SearchResult(chunk=self.chunks[int(i)], score=float(np.clip(score, -1, 1)), rank=r)
                for r, (score, i) in enumerate(ranked, 1)]

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not query.strip() or k <= 0:
            return []
        return self.search_vector(self.embedder.embed([query])[0], k)
