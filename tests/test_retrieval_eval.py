"""
Retrieval-level evaluation for search_protocols.

Tests the retrieval pipeline in isolation (no agent, no synthesis) to measure
whether the right chunks come back for known queries. This separates retrieval
quality from agent/synthesis quality.

Metrics:
- Recall@k: fraction of ground truth chunks found in top-k results
- MRR: reciprocal rank of the first relevant result
- NDCG@k: normalized discounted cumulative gain (rewards relevant results ranked higher)

Usage:
    pytest tests/test_retrieval_eval.py -v --timeout=120
    pytest tests/test_retrieval_eval.py -v -k "needle"       # only needle cases
    pytest tests/test_retrieval_eval.py -v -k "karhi"         # one case
    pytest tests/test_retrieval_eval.py -v -s                 # show detailed report

    # Compare configurations:
    KNESSY_HYDE=0 KNESSY_RERANK=0 pytest tests/test_retrieval_eval.py -v -s
    KNESSY_HYDE=1 KNESSY_RERANK=1 pytest tests/test_retrieval_eval.py -v -s
"""

import asyncio
import math

import dotenv
dotenv.load_dotenv()

import pytest

from tests.retrieval_cases import RETRIEVAL_CASES


def _search(query: str, top: int, filters: dict) -> list[dict]:
    """Run search_protocols and return raw hits with chunk IDs."""
    result = {}

    async def _run():
        from mcp_server.server import search_protocols

        kwargs = {"query": query, "top": top}
        if filters.get("committee_id"):
            kwargs["committee_id"] = filters["committee_id"]
        if filters.get("from_date"):
            kwargs["from_date"] = filters["from_date"]
        if filters.get("to_date"):
            kwargs["to_date"] = filters["to_date"]

        result["response"] = await search_protocols(**kwargs)

    asyncio.run(_run())
    return result.get("response", "")


def _extract_session_ids(response: str) -> list[str]:
    """Extract session IDs from formatted search_protocols output."""
    import re
    return re.findall(r"Session (\d+)", response)


def _extract_chunk_ids_from_response(response: str, query: str, top: int, filters: dict) -> list[str]:
    """Get chunk IDs by running search at the OpenSearch level directly.

    search_protocols returns formatted text, not raw chunk IDs. We re-run
    the same query against OpenSearch to get the actual _id fields.
    """
    from opensearchpy import OpenSearch
    from openai import OpenAI
    import os

    os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
    openai_client = OpenAI()

    embedding_resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    search_vector = embedding_resp.data[0].embedding

    filter_clauses = []
    if filters.get("committee_id"):
        filter_clauses.append({"term": {"committee_id": filters["committee_id"]}})
    if filters.get("from_date"):
        filter_clauses.append({"range": {"session_date": {"gte": filters["from_date"]}}})
    if filters.get("to_date"):
        filter_clauses.append({"range": {"session_date": {"lte": filters["to_date"]}}})

    knn_clause = {
        "embedding": {
            "vector": search_vector,
            "k": top,
        },
    }
    if filter_clauses:
        knn_clause["embedding"]["filter"] = {"bool": {"must": filter_clauses}}

    bool_query = {
        "should": [
            {"knn": knn_clause},
            {"match": {"text": {"query": query, "boost": 0.3}}},
        ],
    }
    if filter_clauses:
        bool_query["filter"] = filter_clauses
        bool_query["minimum_should_match"] = 1

    result = os_client.search(
        index="knesset-protocols",
        body={
            "query": {"bool": bool_query},
            "size": top,
            "_source": [],
        },
    )
    return [hit["_id"] for hit in result["hits"]["hits"]]


# --- Metrics ---

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of relevant chunks found in retrieved results."""
    if not relevant_ids:
        return 1.0
    found = set(retrieved_ids) & relevant_ids
    return len(found) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Normalized discounted cumulative gain."""
    if not relevant_ids:
        return 1.0

    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # +2 because rank starts at 1, log2(1)=0

    # Ideal DCG: all relevant docs at top positions
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), len(retrieved_ids))))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# --- Test parametrization ---

@pytest.fixture(params=RETRIEVAL_CASES, ids=[c["id"] for c in RETRIEVAL_CASES])
def retrieval_case(request):
    return request.param


def pytest_collection_modifyitems(items):
    for item in items:
        if hasattr(item, "callspec") and "retrieval_case" in item.callspec.params:
            case = item.callspec.params["retrieval_case"]
            for tag in case.get("tags", []):
                item.add_marker(getattr(pytest.mark, tag))


# --- Tests ---

K_VALUES = [5, 10, 20]


@pytest.mark.retrieval
def test_retrieval_eval(retrieval_case):
    """Evaluate retrieval quality for a single case across multiple k values."""
    case_id = retrieval_case["id"]
    query = retrieval_case["query"]
    relevant = retrieval_case["relevant_chunk_ids"]
    filters = retrieval_case.get("filters", {})

    max_k = max(K_VALUES)
    retrieved = _extract_chunk_ids_from_response(None, query, max_k, filters)

    print(f"\n{'=' * 60}")
    print(f"Case: {case_id}")
    print(f"Query: {query}")
    print(f"Filters: {filters}")
    print(f"Expected chunks: {relevant}")
    print(f"Retrieved top-{max_k}: {retrieved}")

    for k in K_VALUES:
        top_k = retrieved[:k]
        r = recall_at_k(top_k, relevant)
        m = mrr(top_k, relevant)
        n = ndcg_at_k(top_k, relevant)
        found = set(top_k) & relevant
        print(f"  @{k:2d}: recall={r:.2f}  mrr={m:.2f}  ndcg={n:.2f}  found={found or '{}'}")

    print(f"{'=' * 60}")

    # No assertion — this is a metrics-gathering test, not pass/fail.
    # Use test_retrieval_summary for aggregate thresholds.


@pytest.mark.retrieval
def test_retrieval_summary():
    """Aggregate retrieval metrics across all cases. Run last for a summary."""
    from openai import OpenAI
    from opensearchpy import OpenSearch

    results = []
    for case in RETRIEVAL_CASES:
        max_k = max(K_VALUES)
        retrieved = _extract_chunk_ids_from_response(
            None, case["query"], max_k, case.get("filters", {})
        )

        row = {"id": case["id"], "tags": case["tags"]}
        for k in K_VALUES:
            top_k = retrieved[:k]
            row[f"recall@{k}"] = recall_at_k(top_k, case["relevant_chunk_ids"])
            row[f"mrr@{k}"] = mrr(top_k, case["relevant_chunk_ids"])
            row[f"ndcg@{k}"] = ndcg_at_k(top_k, case["relevant_chunk_ids"])
        results.append(row)

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"RETRIEVAL EVAL SUMMARY ({len(results)} cases)")
    print(f"{'=' * 80}")
    print(f"{'Case':<30} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR@20':>7} {'NDCG@20':>8}")
    print(f"{'-' * 30} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 8}")

    for r in results:
        print(
            f"{r['id']:<30} "
            f"{r['recall@5']:>6.2f} {r['recall@10']:>6.2f} {r['recall@20']:>6.2f} "
            f"{r['mrr@20']:>7.2f} {r['ndcg@20']:>8.2f}"
        )

    # Averages
    n = len(results)
    for k in K_VALUES:
        avg_recall = sum(r[f"recall@{k}"] for r in results) / n
        avg_mrr = sum(r[f"mrr@{k}"] for r in results) / n
        avg_ndcg = sum(r[f"ndcg@{k}"] for r in results) / n
        print(f"{'AVERAGE':<30} " if k == 5 else f"{'':30} ", end="")
        if k == 5:
            print(f"{avg_recall:>6.2f}", end="")
        elif k == 10:
            print(f"       {avg_recall:>6.2f}", end="")
        elif k == 20:
            print(f"              {avg_recall:>6.2f} {avg_mrr:>7.2f} {avg_ndcg:>8.2f}")

    # Averages by tag
    tag_groups = [
        "needle", "broad",
        "date-filter", "committee-filter", "combined-filter",
        "no-filter", "stress-test",
    ]
    print(f"\n{'Breakdown by tag':<30} {'R@5':>6} {'R@20':>6} {'MRR@20':>7} {'n':>4}")
    print(f"{'-' * 30} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 4}")
    for tag in tag_groups:
        tagged = [r for r in results if tag in r["tags"]]
        if tagged:
            avg_r5 = sum(r["recall@5"] for r in tagged) / len(tagged)
            avg_r20 = sum(r["recall@20"] for r in tagged) / len(tagged)
            avg_mrr = sum(r["mrr@20"] for r in tagged) / len(tagged)
            print(f"  {tag:<28} {avg_r5:>6.2f} {avg_r20:>6.2f} {avg_mrr:>7.2f} {len(tagged):>4}")

    print(f"{'=' * 80}")

    # Soft assertion: average recall@20 should be > 0.5
    avg_recall_20 = sum(r["recall@20"] for r in results) / n
    assert avg_recall_20 > 0.3, (
        f"Average recall@20 is {avg_recall_20:.2f} — below 0.3 threshold"
    )
