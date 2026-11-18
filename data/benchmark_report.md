| Configuration | K | Chunks | Hit Rate | Recall | MRR | Average ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| BM25 baseline | 3 | 26 | 0.8889 | 0.8889 | 0.7685 | 0.26 | 0.26 | 0.38 |
| BM25 baseline | 5 | 26 | 0.9444 | 0.9444 | 0.7796 | 0.26 | 0.26 | 0.38 |
| BM25 baseline | 10 | 26 | 0.9444 | 0.9444 | 0.7796 | 0.26 | 0.26 | 0.38 |
| Dense 256 | 3 | 117 | 0.6111 | 0.6111 | 0.3519 | 284.25 | 245.15 | 600.34 |
| Dense 256 | 5 | 117 | 0.8333 | 0.8056 | 0.4046 | 284.25 | 245.15 | 600.34 |
| Dense 256 | 10 | 117 | 0.8889 | 0.8889 | 0.4102 | 284.25 | 245.15 | 600.34 |
| Dense 1024 | 3 | 27 | 0.8333 | 0.8333 | 0.6574 | 283.63 | 244.82 | 600.00 |
| Dense 1024 | 5 | 27 | 0.8889 | 0.8889 | 0.6713 | 283.63 | 244.82 | 600.00 |
| Dense 1024 | 10 | 27 | 1.0000 | 1.0000 | 0.6854 | 283.63 | 244.82 | 600.00 |
| Dense structural | 3 | 26 | 1.0000 | 1.0000 | 0.9722 | 283.60 | 244.81 | 599.98 |
| Dense structural | 5 | 26 | 1.0000 | 1.0000 | 0.9722 | 283.60 | 244.81 | 599.98 |
| Dense structural | 10 | 26 | 1.0000 | 1.0000 | 0.9722 | 283.60 | 244.81 | 599.98 |

On this run (18 labeled queries), the highest MRR@5 was 0.9722 for Dense structural. Dense 256, Dense 1024, and Dense structural achieved Hit Rate@5 of 0.8333, 0.8889, and 1.0000, respectively. Smaller windows can isolate exact evidence, larger windows retain more surrounding context, and structural sections keep examples together. These are plausible mechanisms rather than proof of causation; the labeled sample is small, and no one configuration should be assumed best for every workload.

The fastest average query at K=5 was BM25 baseline (0.26 ms). Dense latency includes the measured query embedding request plus local FAISS lookup; indexing time is excluded. The same single embedding request is shared across the three dense configurations for each query. BM25 executes locally. The trade-off between ranking quality, indexing overhead, and network latency should be validated on a larger held-out corpus before choosing a production retrieval strategy.
