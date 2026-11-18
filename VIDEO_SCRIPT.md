# Video walkthrough (target: 6:15–6:40)

Keep the webcam visible for the entire recording. Use a working local Docker deployment or hosted application. Record after running the live tests, building indexes, and obtaining actual benchmark findings. Read only the **Narration** portions. Replace bracketed metrics with your measured results. Rehearse to ensure the finished video remains between 5:00 and 7:00.

## 0:00–1:00 — Architecture and corpus

**Screen:** VS Code, project explorer with `corpus`, `data`, `src`, `tests` expanded. Open `corpus/api_spec_sample.md`, show a NumPy heading and a fenced example. Open `src/chunking.py`, find `class FixedSizeChunker`, then `class StructureAwareChunker`.

**Narration:**

Hello, my name is [your name]. This is my Semantic Document Search Engine and Empirical Retrieval Benchmarking project.

The purpose is to compare keyword search and semantic vector search using real technical documentation. My corpus contains 25 NumPy API reference sections, covering topics such as array manipulation, statistics, linear algebra, and file handling.

The project has separate files for chunking, indexing, searching, benchmarking, and the Streamlit interface. I also have an evaluation dataset with 18 technical questions and their ground-truth evidence.

The fixed-size chunker creates windows of 256 or 1,024 tokens with a 32-token overlap. The structure-aware chunker instead uses Markdown headings and preserves fenced code examples. Each chunk stores its source, headers, line numbers, and token count.

## 1:00–2:20 — AI generation and audit

**Screen:** Open `ai_prompt_log.md`, show initial prompt, issues 1 and 2, and issue 3 if time permits. Open `audit/initial_draft.py`; then `src/indexer.py`, highlight `tokenize()`, `normalize()`, `IndexFlatIP`, and sort by negative similarity. Open `data/eval_queries.json` to show source/section/anchor. Only claim personal review after performing it.

**Narration:**

I used ChatGPT to generate the initial prototype. The prompt asked for fixed-size and structural chunking, two search indexes, an evaluation pipeline, and automated tests.

I then reviewed the initial candidate code for mistakes. The first issue was inconsistent BM25 tokenization. Documents were converted to lowercase and punctuation was removed, but queries were split without the same preprocessing. This could cause an exact identifier to be missed. The final version uses the same tokenizer on both sides.

The second issue was reversed distance ordering. The original code treated larger cosine distances as better results. I changed the vector backend to FAISS inner-product search and normalized both document and query vectors. Higher scores now represent higher cosine similarity.

I also corrected the ground-truth mapping. Instead of using chunk positions, which change with chunk size, the labels identify a source section and exact evidence span. This makes the comparison consistent across configurations.

## 2:20–3:45 — Live search comparator

**Screen:** Streamlit, Live Search Comparator, K=5. Before recording, use the benchmark failure panel to identify an exact query where BM25 ranks the correct evidence above dense search, and a conceptual query where dense ranks the evidence above BM25. Select the corresponding dense configuration for each. Paste the exact query verbatim from the evaluation dataset, click Compare searches, point at the labeled evidence and score. Repeat with the conceptual query. If no such winner exists, demonstrate honest ties or use additional technical questions; never fabricate a difference.

Sample starting queries (not guaranteed winners):

- `What is the equal_nan option in numpy.unique?`
- `How do I access only a small portion of an enormous file instead of using all my RAM?`

**Narration:**

Now I will demonstrate the two search methods. I have set the result count to five.

First, I will enter an exact technical question. On the left, BM25 searches for matching terms in the documentation. On the right, the dense method compares the query embedding with the stored document embeddings.

Here, the correct source is [section]. BM25 retrieves it at rank [rank], while dense search [actual observed result]. The application shows the scores and highlights matching words. The two score scales are different and should not be directly compared.

Now I will search with a conceptual question that describes the same idea using different words.

The relevant source is [section]. Dense search [actual observed result], while BM25 [actual observed result]. This demonstrates how semantic similarity and literal keyword matching can produce different rankings.

## 3:45–5:45 — Benchmark and empirical findings

**Screen:** Benchmark & Evaluation tab. Click Run Benchmark live with indexes already loaded. While it runs, explain metrics. Display K=5 table, quality bar chart, average latency chart, and failure analysis. Mention the actual winning configuration(s), differences, and p50/p95 latency. Show the download button if time permits.

**Narration:**

Next, I will run the benchmark on the same 18 labeled questions. It evaluates BM25 and the three dense chunking configurations.

Hit Rate measures whether at least one relevant source is returned. Recall measures how much of the labeled evidence is recovered. Mean Reciprocal Rank rewards results that appear earlier in the ranking. The application also measures average latency and the 50th and 95th percentile query times.

We can now see the measured results. At K equals five, [configuration] achieved the highest MRR of [value]. Its Hit Rate was [value]. Compared with the 1,024-token and structural configurations, the 256-token configuration achieved [actual comparison]. Smaller chunks may isolate specific details, whereas larger or structural chunks may preserve context and complete examples.

The average latency for BM25 was [value] milliseconds, compared with [value] for [dense configuration]. Dense latency includes the OpenAI query embedding request, while BM25 runs locally. Index creation is not included.

This failure-analysis table shows individual questions where one method ranked the relevant information higher than the other. These results help explain which types of searches each method may handle better. Because the dataset is small, the findings do not imply one search method is always superior.

## 5:45–6:20 — Automated tests and conclusion

**Screen:** VS Code terminal in project root. Run `RUN_LIVE_EMBEDDING_TEST=1 uv run pytest -v`. Show test names and passing results. If using Docker, show its running container in another terminal. End with README or GitHub. Report only actual results.

**Narration:**

Finally, I will run the automated test suite. The tests verify token overlap, Markdown fence preservation, exact keyword retrieval, dense similarity ranking, and benchmark calculations. The live semantic test also uses a real embedding request to verify retrieval without exact word overlap.

The project includes the source documentation, evaluation dataset, AI audit log, and measured benchmark findings in the README. It demonstrates why retrieval quality should be measured instead of assuming that one search method or chunk size is always best.

Thank you for watching.
