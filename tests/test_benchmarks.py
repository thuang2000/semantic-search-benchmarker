import pytest
import numpy as np

from src.benchmarker import (metrics_at_k, load_queries, resolve_targets, matched_targets,
                             run_benchmark, markdown_report, failure_analysis)
from src.chunking import load_corpus, FixedSizeChunker, StructureAwareChunker
from src.search import ROOT, SearchWorkbench


def test_metric_math_and_missed_ranks():
    ranked = [set(), {0}, {1}, {0}]
    assert metrics_at_k(ranked, {0, 1}, 1) == {"Hit Rate": 0, "Recall": 0, "MRR": 0}
    assert metrics_at_k(ranked, {0, 1}, 2) == {"Hit Rate": 1, "Recall": .5, "MRR": .5}
    assert metrics_at_k(ranked, {0, 1}, 4) == {"Hit Rate": 1, "Recall": 1, "MRR": .5}
    assert metrics_at_k([{0}, {0}], {0, 1}, 2)["Recall"] == .5
    assert metrics_at_k([], {0}, 5)["MRR"] == 0


def test_all_ground_truth_anchors_exist_and_are_unique():
    docs = load_corpus(ROOT / "corpus")
    records = load_queries()
    assert len(records) >= 15
    assert {r.type for r in records} == {"exact", "conceptual"}
    assert all(resolve_targets(r, docs) for r in records)


def test_ground_truth_coverage_across_chunk_sizes():
    pytest.importorskip("tiktoken")
    docs = load_corpus(ROOT / "corpus")
    for chunker in [FixedSizeChunker(256, 32), FixedSizeChunker(1024, 32), StructureAwareChunker()]:
        chunks = [c for source, text in docs.items() for c in chunker.split(text, source)]
        for r in load_queries():
            spans = resolve_targets(r, docs)
            matched = set().union(*(matched_targets(c, spans) for c in chunks))
            assert matched == set(range(len(spans))), r.id


def test_benchmark_pipeline_with_fixture_embeddings():
    pytest.importorskip("tiktoken")
    pytest.importorskip("rank_bm25")
    pytest.importorskip("faiss")
    class TinyEmbedder:
        def embed(self, texts):
            # Deterministic nonzero vectors for plumbing tests, not semantics.
            return np.array([[1., float(len(t) % 11 + 1)] for t in texts])
    wb = SearchWorkbench(load_corpus(ROOT / "corpus"), TinyEmbedder())
    summary, details = run_benchmark(wb, load_queries()[:2])
    assert len(summary) == 12
    assert len(details) == 24
    assert summary["Hit Rate"].between(0, 1).all()
    assert summary["Recall"].between(0, 1).all()
    assert summary["MRR"].between(0, 1).all()
    assert (summary["Average ms"] >= 0).all()
    assert "highest MRR@5" in markdown_report(summary, details)
    assert "Winner" in failure_analysis(details).columns
