"""Run labeled retrieval benchmarks and export actual measured findings."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.chunking import DocumentChunk, sections
from src.indexer import SearchResult
from src.search import BM25, CONFIGS, ROOT, SearchWorkbench, create_workbench


class Target(BaseModel):
    source: str
    section: str
    anchor: str = Field(min_length=1)


class EvaluationQuery(BaseModel):
    id: str
    type: str
    query: str = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)


def load_queries(path: Path = ROOT / "data/eval_queries.json") -> list[EvaluationQuery]:
    records = [EvaluationQuery.model_validate(row) for row in json.loads(path.read_text())]
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError("Evaluation queries must have unique IDs.")
    return records


def resolve_targets(record: EvaluationQuery, docs: dict[str, str]) -> list[tuple[str, int, int]]:
    """Map a human-readable source label to an exact character span."""
    resolved = []
    for target in record.targets:
        text = docs[target.source]
        ranges = [(a, b) for a, b, headers in sections(text)
                  if headers and headers[-1] == target.section]
        if len(ranges) != 1:
            raise ValueError(f"Missing or ambiguous section: {target.section}")
        a, b = ranges[0]
        if text[a:b].count(target.anchor) != 1:
            raise ValueError(f"Missing or ambiguous anchor: {record.id} / {target.section}")
        start = text.index(target.anchor, a, b)
        resolved.append((target.source, start, start + len(target.anchor)))
    return resolved


def matched_targets(chunk: DocumentChunk, targets: list[tuple[str, int, int]]) -> set[int]:
    """An anchor counts only if the whole labeled evidence span is present."""
    return {i for i, (source, a, b) in enumerate(targets)
            if chunk.parent_doc == source and chunk.metadata["char_start"] <= a
            and chunk.metadata["char_end"] >= b}


def metrics_at_k(ranked_matches: list[set[int]], relevant: set[int], k: int) -> dict[str, float]:
    """Recall is source-span recall, not the number of overlapping chunks."""
    if k < 1 or not relevant:
        raise ValueError("K and the ground-truth set must be nonempty.")
    seen = set()
    reciprocal = 0.0
    for rank, matches in enumerate(ranked_matches[:k], 1):
        hits = matches & relevant
        if hits and reciprocal == 0.0:
            reciprocal = 1.0 / rank
        seen.update(hits)
    return {"Hit Rate": float(bool(seen)), "Recall": len(seen) / len(relevant),
            "MRR": reciprocal}


def run_benchmark(workbench: SearchWorkbench, records: list[EvaluationQuery],
                  ks: tuple[int, ...] = (3, 5, 10),
                  progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate identical queries against four already-built indexes.

    Each dense query embedding is computed once. Its measured API time is
    added to each dense configuration's index search time. No query labels
    are passed to the embedding API or the search engine.
    """
    if not records or not ks or min(ks) < 1:
        raise ValueError("At least one query and positive K are required.")
    max_k = max(ks)
    rows = []
    # Resolve the labels solely for evaluation. Ensure all spans are retrievable
    # in every configuration before scoring any ranking.
    labels = {r.id: resolve_targets(r, workbench.docs) for r in records}
    for name in CONFIGS:
        for r in records:
            covered = set().union(*(matched_targets(c, labels[r.id]) for c in workbench.chunks[name]))
            if covered != set(range(len(labels[r.id]))):
                raise ValueError(f"Ground truth span absent from {name}: {r.id}")
    for i, record in enumerate(records, 1):
        start = perf_counter()
        bm25_results = workbench.bm25.search(record.query, max_k)
        bm25_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        vector = workbench.embedder.embed([record.query])[0]
        embedding_ms = (perf_counter() - start) * 1000
        retrieved = {BM25: (bm25_results, bm25_ms)}
        for name, index in workbench.dense.items():
            start = perf_counter()
            results = index.search_vector(vector, max_k)
            retrieved[name] = (results, embedding_ms + (perf_counter() - start) * 1000)
        targets = labels[record.id]
        for name in CONFIGS:
            results, latency = retrieved[name]
            matches = [matched_targets(r.chunk, targets) for r in results]
            first_rank = next((rank for rank, hit in enumerate(matches, 1) if hit), None)
            for k in ks:
                row = {"Query ID": record.id, "Query": record.query, "Type": record.type,
                       "Configuration": name, "K": k, "Latency (ms)": latency,
                       "First relevant rank": first_rank if first_rank is not None and first_rank <= k else None,
                       "Top result": results[0].chunk.chunk_id if results else "No lexical matches"}
                row.update(metrics_at_k(matches, set(range(len(targets))), k))
                rows.append(row)
        if progress:
            progress(i, len(records))
    details = pd.DataFrame(rows)
    summary = []
    for name in CONFIGS:
        for k in ks:
            group = details[(details["Configuration"] == name) & (details["K"] == k)]
            latency = group["Latency (ms)"].to_numpy()
            summary.append({"Configuration": name, "K": k, "Chunks": len(workbench.chunks[name]),
                            "Hit Rate": group["Hit Rate"].mean(), "Recall": group["Recall"].mean(),
                            "MRR": group["MRR"].mean(), "Average ms": float(np.mean(latency)),
                            "p50 ms": float(np.percentile(latency, 50)),
                            "p95 ms": float(np.percentile(latency, 95))})
    return pd.DataFrame(summary), details


def failure_analysis(details: pd.DataFrame, k: int = 5, dense_name: str = "Dense structural") -> pd.DataFrame:
    """Compare reciprocal ranks for every query, not just selected successes."""
    data = details[details["K"] == k]
    bm25 = data[data["Configuration"] == BM25]
    dense = data[data["Configuration"] == dense_name]
    rows = []
    for _, left in bm25.iterrows():
        right = dense[dense["Query ID"] == left["Query ID"]].iloc[0]
        diff = left["MRR"] - right["MRR"]
        if diff == 0:
            continue
        winner = BM25 if diff > 0 else dense_name
        # These are hypotheses based on query category, not causal proofs.
        if diff > 0 and left["Type"] == "exact":
            reason = "Exact identifiers may favor lexical term matching."
        elif diff < 0 and left["Type"] == "conceptual":
            reason = "Paraphrased wording may favor semantic embeddings."
        elif diff > 0:
            reason = "Shared source terms may help BM25; inspect the ranked evidence."
        else:
            reason = "Contextual similarity may help dense retrieval; inspect the ranked evidence."
        rows.append({"Query ID": left["Query ID"], "Query": left["Query"], "Type": left["Type"],
                     "BM25 rank": left["First relevant rank"], "Dense rank": right["First relevant rank"],
                     "Winner": winner, "Possible explanation": reason})
    return pd.DataFrame(rows, columns=["Query ID", "Query", "Type", "BM25 rank", "Dense rank", "Winner", "Possible explanation"])


def markdown_report(summary: pd.DataFrame, details: pd.DataFrame) -> str:
    columns = ["Configuration", "K", "Chunks", "Hit Rate", "Recall", "MRR", "Average ms", "p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, r in summary.iterrows():
        values = [str(r["Configuration"]), str(int(r["K"])), str(int(r["Chunks"]))]
        values += [f"{r[c]:.4f}" for c in columns[3:6]] + [f"{r[c]:.2f}" for c in columns[6:]]
        lines.append("| " + " | ".join(values) + " |")
    at5 = summary[summary["K"] == 5]
    if at5.empty:
        return "\n".join(lines)
    peak = at5["MRR"].max()
    best = ", ".join(at5.loc[np.isclose(at5["MRR"], peak), "Configuration"])
    fast = at5.loc[at5["Average ms"].idxmin()]
    sm = at5[at5["Configuration"] == "Dense 256"].iloc[0]
    lg = at5[at5["Configuration"] == "Dense 1024"].iloc[0]
    st = at5[at5["Configuration"] == "Dense structural"].iloc[0]
    n = details["Query ID"].nunique()
    lines += ["", f"On this run ({n} labeled queries), the highest MRR@5 was {peak:.4f} for {best}. "
              f"Dense 256, Dense 1024, and Dense structural achieved Hit Rate@5 of {sm['Hit Rate']:.4f}, "
              f"{lg['Hit Rate']:.4f}, and {st['Hit Rate']:.4f}, respectively. "
              "Smaller windows can isolate exact evidence, larger windows retain more surrounding context, "
              "and structural sections keep examples together. These are plausible mechanisms rather than proof of causation; "
              "the labeled sample is small, and no one configuration should be assumed best for every workload.", "",
              f"The fastest average query at K=5 was {fast['Configuration']} ({fast['Average ms']:.2f} ms). "
              "Dense latency includes the measured query embedding request plus local FAISS lookup; indexing time is excluded. "
              "The same single embedding request is shared across the three dense configurations for each query. "
              "BM25 executes locally. The trade-off between ranking quality, indexing overhead, and network latency should be "
              "validated on a larger held-out corpus before choosing a production retrieval strategy."]
    return "\n".join(lines) + "\n"


def update_readme(report: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    begin = "<!-- BENCHMARK_START -->"
    end = "<!-- BENCHMARK_END -->"
    a = text.index(begin) + len(begin)
    b = text.index(end, a)
    path.write_text(text[:a] + "\n\n" + report + "\n" + text[b:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    print("Building indexes from source documents (OpenAI embedding requests)...", flush=True)
    wb = create_workbench()
    summary, details = run_benchmark(wb, load_queries(), progress=lambda i, n: print(f"Query {i}/{n}", flush=True))
    report = markdown_report(summary, details)
    (ROOT / "data/benchmark_results.csv").write_text(summary.to_csv(index=False))
    (ROOT / "data/benchmark_details.csv").write_text(details.to_csv(index=False))
    (ROOT / "data/benchmark_report.md").write_text(report)
    if args.write_readme:
        update_readme(report)
    print(report)


if __name__ == "__main__":
    main()
