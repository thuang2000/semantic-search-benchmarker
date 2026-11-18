"""Interactive semantic search and empirical benchmarking dashboard."""
from pathlib import Path
import sys
import html
import re

# Streamlit may execute the script with src/ rather than the root on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.benchmarker import failure_analysis, load_queries, markdown_report, run_benchmark, resolve_targets, matched_targets
from src.indexer import tokenize
from src.search import BM25, SMALL, LARGE, STRUCTURAL, create_workbench

st.set_page_config(page_title="Semantic Search Workbench", layout="wide")
st.title("Semantic Document Search Workbench")
st.caption("NumPy technical reference · BM25 vs. OpenAI embeddings + FAISS")


@st.cache_resource(show_spinner="Building source indexes (first run only)...")
def workbench():
    return create_workbench()


def highlighted(content: str, query: str) -> str:
    """Escape all source/phrase text before adding our own safe mark tags."""
    terms = set(tokenize(query))
    pieces = re.split(r"(\w+)", content)
    return "".join(f"<mark>{html.escape(part)}</mark>" if part.lower() in terms
                   else html.escape(part) for part in pieces)


def render_results(results, query, score_label, target_spans=None):
    if not results:
        st.info("No matching chunks.")
    for result in results:
        chunk = result.chunk
        headers = chunk.metadata["header_hierarchy"]
        heading = " / ".join(headers[0]) if headers else chunk.parent_doc
        st.markdown(f"**#{result.rank} — {html.escape(heading)}**")
        st.caption(f"{score_label}: {result.score:.4f} | {chunk.metadata['token_count']} tokens | "
                   f"Lines {chunk.metadata['line_start']}–{chunk.metadata['line_end']}")
        if target_spans is not None:
            if matched_targets(chunk, target_spans):
                st.success("Contains labeled evidence")
            else:
                st.caption("Does not contain the labeled evidence span")
        with st.expander("View highlighted source chunk", expanded=result.rank == 1):
            # Plain-text display keeps fixed-window partial Markdown fences honest.
            st.markdown('<div style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;">'
                        + highlighted(chunk.content, query) + '</div>', unsafe_allow_html=True)
            st.caption(chunk.chunk_id)


with st.sidebar:
    st.header("Search settings")
    k = st.selectbox("Top K", [3, 5, 10], index=1)
    dense_name = st.selectbox("Dense chunking", [STRUCTURAL, SMALL, LARGE])
    st.caption("Fixed windows use 32 overlapping tokens. BM25 uses structural sections.")
    st.warning("Building indexes and dense queries use the configured OpenAI API key.")
    if st.button("Build / load indexes"):
        try:
            wb = workbench()
            st.success("Indexes ready")
            for name, chunks in wb.chunks.items():
                st.write(f"{name}: {len(chunks)} chunks")
        except Exception:
            st.error("Could not initialize indexes. Check your API key, credits, dependencies, and internet connection.")

search_tab, benchmark_tab = st.tabs(["Live Search Comparator", "Benchmark & Evaluation"])
with search_tab:
    query = st.text_input("Technical query", value="What does allow_pickle do in numpy.load?")
    if st.button("Compare searches", type="primary"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                wb = workbench()
                with st.spinner("Running both retrieval methods..."):
                    left = wb.search(query, BM25, k)
                    right = wb.search(query, dense_name, k)
                labeled = next((r for r in load_queries() if r.query.strip() == query.strip()), None)
                targets = resolve_targets(labeled, wb.docs) if labeled else None
                st.session_state["search_result"] = (query, k, dense_name, left, right, targets)
            except Exception:
                st.error("Search failed. Verify the OpenAI credentials and connection.")
                st.session_state.pop("search_result", None)
    if "search_result" in st.session_state:
        q, old_k, old_dense, left, right, targets = st.session_state["search_result"]
        st.caption(f"Last search: {q} | K={old_k}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("BM25 Keyword Search")
            render_results(left, q, "BM25 score", targets)
        with col2:
            st.subheader(f"{old_dense} — Vector Search")
            render_results(right, q, "Cosine similarity", targets)

with benchmark_tab:
    st.write("Compare four configurations on the same 18 labeled queries at K=3, 5, and 10.")
    st.caption("Dense latency includes the shared query-embedding API time and individual FAISS lookup. Index construction is excluded.")
    if st.button("Run Benchmark", type="primary"):
        try:
            wb = workbench()
            bar = st.progress(0, text="Evaluating labeled queries...")
            def progress(i, n):
                bar.progress(i / n, text=f"Evaluating query {i} of {n}")
            summary, details = run_benchmark(wb, load_queries(), progress=progress)
            st.session_state["benchmark"] = (summary, details)
            bar.empty()
            st.success("Benchmark complete.")
        except Exception:
            st.error("Benchmark failed. Check credentials, dependencies, and labeled corpus files.")
            st.session_state.pop("benchmark", None)
    if "benchmark" in st.session_state:
        summary, details = st.session_state["benchmark"]
        display_k = st.selectbox("Benchmark display K", [3, 5, 10], index=1)
        table = summary[summary["K"] == display_k]
        st.dataframe(table.round(4), hide_index=True, use_container_width=True)
        st.subheader("Retrieval quality")
        st.bar_chart(table.set_index("Configuration")[["Hit Rate", "Recall", "MRR"]])
        st.subheader("Average query latency (ms)")
        st.bar_chart(table.set_index("Configuration")[["Average ms"]])
        st.subheader("Failure analysis")
        compare = st.selectbox("Compare BM25 against", [STRUCTURAL, SMALL, LARGE], key="failure_dense")
        failures = failure_analysis(details, display_k, compare)
        if failures.empty:
            st.info("No reciprocal-rank differences for this configuration at the selected K.")
        else:
            st.dataframe(failures, hide_index=True, use_container_width=True)
            st.caption("Possible explanations are hypotheses; inspect the actual ranked evidence.")
        with st.expander("All per-query measurements"):
            st.dataframe(details[details["K"] == display_k], hide_index=True, use_container_width=True)
        st.download_button("Download benchmark results CSV", summary.to_csv(index=False), "benchmark_results.csv", "text/csv")
        st.download_button("Download per-query results CSV", details.to_csv(index=False), "benchmark_details.csv", "text/csv")
        st.download_button("Download README findings", markdown_report(summary, details), "benchmark_report.md", "text/markdown")
