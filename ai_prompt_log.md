# AI code generation and verification log

## Initial AI prompt

> You are an information retrieval engineer. Generate a Python 3.11 semantic document search engine and benchmark harness. Use Pydantic v2 chunk models, tiktoken fixed windows with configurable overlap, Markdown header splitting, BM25Okapi, normalized OpenAI text-embedding-3-small embeddings in FAISS, and a standardized search result. Compare BM25 against dense 256-token, 1024-token and structural chunking on the same labeled technical questions. Compute Hit Rate, Recall, MRR and average/p50/p95 query latency at K=3, 5 and 10. Provide a Streamlit comparator and benchmark dashboard, plus pytest tests for chunking, search and metric math. Keep the implementation modular and close to the assignment instructions.

## Initial candidate output

The initial AI-generated prototype excerpts are preserved in `audit/initial_draft.py`. They are the specific draft functions audited here; they are not a complete runnable first version. The finalized project does not import them. Review this record and the final code yourself before submission; run and record the test results rather than asserting unperformed validation.

## Issue 1 — Asymmetric BM25 tokenization

The corpus was lowercased and punctuation-normalized, but the original query used raw whitespace splitting. Identifiers with capital letters or trailing punctuation could fail to match.

```diff
- return index.get_scores(query.split())
+ words = tokenize(query)
+ scores = self.index.get_scores(words)
```

`src/indexer.py` now uses `tokenize()` for documents and queries, preserves underscores, and filters candidates with no shared tokens. See `test_exact_keyword_rank_one` and `test_normalization_and_tokenization`.

## Issue 2 — Inverted distance ordering

The prototype sorted cosine distances in descending order even though smaller distances mean more relevant results.

```diff
- return sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)
+ vectors = normalize(embedder.embed([c.content for c in chunks]))
+ self.index = faiss.IndexFlatIP(vectors.shape[1])
+ self.index.add(vectors)
+ query = normalize(np.asarray(vector).reshape(1, -1))
+ scores, ids = self.index.search(query, len(self.chunks))
```

The final backend explicitly returns inner-product similarity rather than distance. With unit vectors, higher inner product means greater cosine similarity. Equal scores are ordered by stable chunk ID. Verify `test_dense_ranks_synonym_fixture`, `test_dense_ties_stable`, and the opt-in live semantic test.

## Issue 3 — Ground-truth indexing leakage

The prototype mapped labels to positions in one chunk array. A larger chunk size would shift those positions and invalidate comparisons.

```diff
- return {chunks[i].chunk_id for i in query_record['target_indices']}
+ spans = resolve_targets(record, workbench.docs)
+ matches = [matched_targets(r.chunk, spans) for r in results]
```

Each query in `data/eval_queries.json` identifies a source file, unique section heading and exact evidence anchor. Only the corpus is indexed. The evaluator resolves stable source character spans and counts a target only when a returned chunk contains the whole span. Overlapping duplicate evidence cannot inflate recall. A coverage check verifies labels under all strategies before scoring. See `test_ground_truth_coverage_across_chunk_sizes`.

## Issue 4 — Code-fence slicing and metadata

The prototype's simple regex could accidentally treat a Python comment inside a fenced example as a Markdown heading.

```diff
- return re.split(r'(?m)(?=^#{1,6} )', text)
+ ranges = sections(text)  # tracks opening/closing fences and heading stack
+ return [make_chunk(text, source, 'structural', a, b, ranges)
+         for a, b, _ in ranges if text[a:b].strip()]
```

`sections()` tracks fence type and length and only recognizes headings outside code. Fixed-window chunks intentionally retain the strict token-budget baseline: they may start/end within a code block. Rather than change the fixed-size experiment or fabricate fence characters, the interface renders these chunks as escaped plain text. The structure-aware alternative preserves the intact examples. Fixed boundaries also maintain valid UTF-8 and exact token overlap. See `test_markdown_fence_preservation`, `test_overlap_integrity` and `test_unicode_no_character_loss`.

## Issue 5 — MRR outside top-K

The initial version scanned the entire ranking, even when the task specified MRR@K.

```diff
- for rank, item in enumerate(ranking, start=1):
+ for rank, matches in enumerate(ranked_matches[:k], 1):
```

`metrics_at_k()` starts reciprocal rank at zero, counts only the first relevant rank inside K, and leaves misses at zero. Hit Rate is binary, and Recall is the fraction of unique labeled evidence spans recovered. See `test_metric_math_and_missed_ranks`.

## Verification and limitations

- Pydantic v2 models preserve typed chunk and result records; no query labels are sent to the embedding model.
- Document vectors and query vectors are both L2-normalized. Zero vectors are rejected.
- No API key is committed. Real API errors are not displayed verbatim to users.
- The same query embedding is reused across dense configurations within each benchmark query; measured embedding latency is included separately in each configuration's end-to-end latency. Index construction is excluded.
- Fixed chunks can cross section boundaries; header hierarchy and offsets retain the original source location. Structural chunks may have variable sizes.
- The 18-query evaluation is a small, curated, non-training benchmark, not a universal result. Review individual winners and actual output data.
- Test fixture embeddings are for deterministic mechanics checks only. Run the optional real-API semantic retrieval test and empirical benchmark to verify actual retrieval quality.

## Student verification notes

Record the date, actual `pytest -v` results, and any additional corrections after you execute the supplied code locally. Do not insert secret keys or fabricate results.
