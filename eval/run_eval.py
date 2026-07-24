import argparse
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.config.settings import TOP_K_RETRIEVAL
from app.rag.pipeline import load_existing_index
from app.vectorstore.retriever import HybridRetriever

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def load_eval_set():
    with open(EVAL_SET_PATH, "r") as f:
        return json.load(f)


def chunk_matches(chunk: dict, item: dict) -> bool:
    """
    A retrieved chunk counts as "correct" for a query if it contains at
    least one expected keyword, and (if specified) comes from the right
    source document. This is a proxy for true relevance - good enough to
    catch regressions, not a substitute for a human-labeled relevance set.
    """
    text = chunk["chunk_text"].lower()
    keyword_hit = any(kw.lower() in text for kw in item.get("expected_keywords", []))

    expected_doc = item.get("expected_doc")
    doc_ok = expected_doc is None or chunk["metadata"]["source_pdf"] == expected_doc

    return keyword_hit and doc_ok


def evaluate(retriever: HybridRetriever, eval_set: list, k: int) -> dict:
    """
    Hit Rate@k: fraction of queries where a correct chunk appears anywhere
    in the top k results.
    MRR: mean reciprocal rank of the first correct chunk (rewards ranking
    the right chunk higher, not just including it somewhere in top k).
    """
    hits = []
    reciprocal_ranks = []

    for item in eval_set:
        results = retriever.retrieve(item["query"])[:k]

        rank = None
        for i, chunk in enumerate(results, start=1):
            if chunk_matches(chunk, item):
                rank = i
                break

        hits.append(1 if rank is not None else 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "hit_rate@k": mean(hits) if hits else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "n_queries": len(eval_set),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against one session's saved index."
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help=(
            "The session_id whose index to evaluate. Indexes are per-session "
            "under data/vector_index/<session_id>/ - find yours in the "
            "Streamlit app's URL/session or by listing that directory."
        ),
    )
    args = parser.parse_args()

    vector_store, bm25_store = load_existing_index(args.session_id)
    if vector_store is None:
        print(
            f"No compatible saved index found for session '{args.session_id}'. "
            "Run the Streamlit app (streamlit run app/main.py), upload/ingest "
            "at least one document, then re-run this script with that "
            "session's ID."
        )
        return

    eval_set = load_eval_set()
    if any("REPLACE ME" in item["query"] for item in eval_set):
        print(
            "eval/eval_set.json still has placeholder entries - these numbers "
            "aren't meaningful until you fill it in with real questions about "
            "your own uploaded documents.\n"
        )

    # Three configs of the SAME chunks/index, differing only in retrieval
    # strategy - this isolates whether hybrid search and reranking actually
    # help, rather than conflating them with the chunking/parsing changes.
    configs = {
        "vector_only": dict(use_bm25=False, use_rerank=False),
        "hybrid_no_rerank": dict(use_bm25=True, use_rerank=False),
        "hybrid_plus_rerank": dict(use_bm25=True, use_rerank=True),
    }

    print(f"{'Config':<20}{'Hit Rate@' + str(TOP_K_RETRIEVAL):<15}{'MRR':<10}{'Queries'}")
    print("-" * 60)

    results = {}
    for name, kwargs in configs.items():
        retriever = HybridRetriever(vector_store, bm25_store, top_k=TOP_K_RETRIEVAL, **kwargs)
        metrics = evaluate(retriever, eval_set, TOP_K_RETRIEVAL)
        results[name] = metrics
        print(f"{name:<20}{metrics['hit_rate@k']:<15.3f}{metrics['mrr']:<10.3f}{metrics['n_queries']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
