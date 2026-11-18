import os
import numpy as np
import pytest
from dotenv import load_dotenv

from src.chunking import DocumentChunk
from src.indexer import BM25Index, DenseIndex, OpenAIEmbedder, normalize, tokenize
from src.search import ROOT


def chunk(identifier, text):
    return DocumentChunk(chunk_id=identifier, parent_doc="test.md", content=text,
                         metadata={"char_start": 0, "char_end": len(text)})


def test_exact_keyword_rank_one():
    pytest.importorskip("rank_bm25")
    docs = [chunk("a", "CUSTOM_CONFIG_FLAG_XYZ controls routing."),
            chunk("b", "Server configuration settings."),
            chunk("c", "Socket pool tuning."), chunk("d", "Database connection details.")]
    results = BM25Index(docs).search("custom_config_flag_xyz!", 3)
    assert results[0].chunk.chunk_id == "a"
    assert BM25Index(docs).search("unmatchedword", 3) == []


def test_normalization_and_tokenization():
    assert tokenize("CUSTOM_CONFIG_FLAG_XYZ!") == ["custom_config_flag_xyz"]
    assert np.allclose(normalize([[3, 4], [10, 0]]), [[.6, .8], [1, 0]])
    with pytest.raises(ValueError):
        normalize([[0, 0]])


class FixtureEmbedder:
    """Artificial vectors only test ranking; never used for real benchmarks."""
    def embed(self, texts):
        vectors = {"car": [3, 0], "violin": [0, 2], "volcano": [-1, 0],
                   "automobile": [9, 0], "tied": [1, 0]}
        return np.array([vectors[t] for t in texts], dtype=np.float32)


def test_dense_ranks_synonym_fixture():
    pytest.importorskip("faiss")
    docs = [chunk("a", "car"), chunk("b", "violin"), chunk("c", "volcano")]
    assert not (set(tokenize("automobile")) & set(tokenize("car")))
    results = DenseIndex(docs, FixtureEmbedder()).search("automobile", 2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score >= results[1].score


def test_dense_ties_stable():
    pytest.importorskip("faiss")
    docs = [chunk("z", "car"), chunk("a", "tied")]
    results = DenseIndex(docs, FixtureEmbedder()).search("automobile", 2)
    assert [r.chunk.chunk_id for r in results] == ["a", "z"]


def test_live_semantic_paraphrase():
    """Opt-in real provider integration: no exact shared words in target."""
    if os.getenv("RUN_LIVE_EMBEDDING_TEST") != "1":
        pytest.skip("Set RUN_LIVE_EMBEDDING_TEST=1 to make real API calls.")
    load_dotenv(ROOT / ".env")
    target = "car vehicle road transportation"
    query = "automobile"
    assert not set(tokenize(target)) & set(tokenize(query))
    docs = [chunk("relevant", target), chunk("music", "violin orchestra strings"),
            chunk("geology", "volcano magma eruption"), chunk("food", "banana apple fruit"),
            chunk("astronomy", "galaxy telescope stars")]
    results = DenseIndex(docs, OpenAIEmbedder()).search(query, 2)
    assert "relevant" in [r.chunk.chunk_id for r in results]
