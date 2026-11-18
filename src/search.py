"""Unified search interface; only source documents are indexed."""
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker, StructureAwareChunker, load_corpus
from src.indexer import BM25Index, DenseIndex, Embedder, OpenAIEmbedder, SearchResult

ROOT = Path(__file__).resolve().parents[1]
BM25 = "BM25 baseline"
SMALL = "Dense 256"
LARGE = "Dense 1024"
STRUCTURAL = "Dense structural"
CONFIGS = [BM25, SMALL, LARGE, STRUCTURAL]


class SearchWorkbench:
    def __init__(self, docs: dict[str, str], embedder: Embedder, overlap: int = 32):
        self.docs = docs
        self.embedder = embedder
        structural = [c for source, text in docs.items()
                      for c in StructureAwareChunker().split(text, source)]
        small = [c for source, text in docs.items()
                 for c in FixedSizeChunker(256, overlap).split(text, source)]
        large = [c for source, text in docs.items()
                 for c in FixedSizeChunker(1024, overlap).split(text, source)]
        self.bm25 = BM25Index(structural)
        self.dense = {SMALL: DenseIndex(small, embedder), LARGE: DenseIndex(large, embedder),
                      STRUCTURAL: DenseIndex(structural, embedder)}
        self.chunks = {BM25: structural, SMALL: small, LARGE: large, STRUCTURAL: structural}

    def search(self, query: str, configuration: str, k: int = 5) -> list[SearchResult]:
        if configuration == BM25:
            return self.bm25.search(query, k)
        return self.dense[configuration].search(query, k)


def create_workbench() -> SearchWorkbench:
    load_dotenv(ROOT / ".env")
    return SearchWorkbench(load_corpus(ROOT / "corpus"), OpenAIEmbedder())
