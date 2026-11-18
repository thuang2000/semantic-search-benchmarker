"""Initial AI prototype excerpts, retained verbatim for the audit.
Not imported or used by the application.
"""
import re


def bm25_query(index, query):
    # The corpus was tokenized with lowercase and punctuation removal.
    return index.get_scores(query.split())


def rank_distances(distances):
    # Intended for a cosine distance backend (smaller is better).
    return sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)


def relevant_ids(query_record, chunks):
    # The first chunking configuration assigned numeric IDs to its chunks.
    return {chunks[i].chunk_id for i in query_record['target_indices']}


def split_headers(text):
    return re.split(r'(?m)(?=^#{1,6} )', text)


def reciprocal_rank(ranking, relevant, k):
    for rank, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1 / rank
    return 0.0
