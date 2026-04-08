"""
Three-way benchmark comparison: agent pipeline vs one-shot RAG vs no RAG.

Tests whether the multi-step agent pipeline justifies its complexity over
simpler approaches using the same model (gpt-4o).

Strategies:
  - "agent":    Full pipeline (planner → researcher → judge → synthesizer)
  - "oneshot":  1 LLM call for queries + search + 1 LLM call to answer (2 calls)
  - "no_rag":   1 LLM call with just the question, no data (baseline)

Usage:
    pytest tests/test_benchmark_comparison.py -v --timeout=300 -s
    pytest tests/test_benchmark_comparison.py -v -k "oneshot" -s
"""

import asyncio
import json
import logging

import dotenv

dotenv.load_dotenv()

import pytest
from openai import OpenAI

from tests.benchmark_cases import BENCHMARK_CASES

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("comparison")

client = OpenAI()
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Strategy: full agent pipeline
# ---------------------------------------------------------------------------
def _run_agent(question: str) -> str:
    result = {}

    async def _capture():
        from agent.graph import agent

        initial_state = {
            "question": question,
            "messages": [],
            "research_tasks": [],
            "research_results": [],
            "grading_results": [],
            "reformulate": False,
            "is_sufficient": False,
            "eval_feedback": "",
            "iteration": 0,
            "final_answer": "",
        }
        async for event in agent.astream(initial_state):
            for node_name, node_output in event.items():
                if "final_answer" in node_output and node_output["final_answer"]:
                    result["answer"] = node_output["final_answer"]

    asyncio.run(_capture())
    return result.get("answer", "")


# ---------------------------------------------------------------------------
# Strategy: one-shot RAG (search + stuff + answer in 2 LLM calls)
# ---------------------------------------------------------------------------
def _run_oneshot(question: str) -> str:
    # Step 1: one LLM call to generate Hebrew search terms
    plan_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate 3-4 Hebrew search queries for searching Israeli Knesset "
                    "committee protocol transcripts. The queries should cover different "
                    "angles of the topic — synonyms, key people, related legislation. "
                    "Return a JSON array of strings. Respond ONLY with the JSON array."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    raw = plan_response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    queries = json.loads(raw)
    log.info("Oneshot queries: %s", queries)

    # Step 2: search OpenSearch with each query, stuff everything
    async def _search_all():
        from mcp_server.server import search_protocols_for_agent
        all_results = []
        for q in queries:
            result = await search_protocols_for_agent(query=q, top=10)
            all_results.append(f"=== Search: {q} ===\n{result}")
        return "\n\n".join(all_results)

    search_results = asyncio.run(_search_all())
    log.info("Oneshot: gathered %d chars of search results", len(search_results))

    # Step 3: one LLM call to answer from all results
    answer_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Knesset research assistant. Answer the question based "
                    "ONLY on the provided data. Quote directly from protocols when "
                    "possible — use exact Hebrew phrases. Cite session IDs, committee "
                    "names, and dates. Answer in the same language as the question."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nData:\n{search_results[:30000]}",
            },
        ],
        temperature=0,
    )

    return answer_response.choices[0].message.content


# ---------------------------------------------------------------------------
# Strategy: no RAG (just the question, cold)
# ---------------------------------------------------------------------------
def _run_no_rag(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert on Israeli Knesset (parliament) proceedings. "
                    "Answer the question based on your knowledge. If you don't know "
                    "specific details, say so. Answer in the same language as the question."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Scoring (reused from test_benchmark.py)
# ---------------------------------------------------------------------------
def _score_answer(answer: str, case: dict) -> dict:
    results = []
    total_weight = 0
    earned_weight = 0

    for marker in case["expected"]:
        found = marker["text"] in answer
        total_weight += marker["weight"]
        if found:
            earned_weight += marker["weight"]
        results.append({
            "marker": marker["text"],
            "description": marker["description"],
            "weight": marker["weight"],
            "found": found,
        })

    hallucinations = [neg for neg in case.get("negative", []) if neg in answer]
    score = earned_weight / total_weight if total_weight > 0 else 0

    return {
        "case_id": case["id"],
        "score": score,
        "earned": earned_weight,
        "total": total_weight,
        "markers": results,
        "hallucinations": hallucinations,
        "answer_length": len(answer),
    }


STRATEGIES = {
    "agent": _run_agent,
    "oneshot": _run_oneshot,
    "no_rag": _run_no_rag,
}


# ---------------------------------------------------------------------------
# Test parametrization: strategy × case
# ---------------------------------------------------------------------------
def _make_params():
    params = []
    for strategy_name in STRATEGIES:
        for case in BENCHMARK_CASES:
            params.append(
                pytest.param(
                    strategy_name, case,
                    id=f"{strategy_name}__{case['id']}",
                )
            )
    return params


@pytest.mark.comparison
@pytest.mark.parametrize("strategy_name,case", _make_params())
def test_comparison(strategy_name, case):
    """Run a benchmark case with a given strategy and report score."""
    run_fn = STRATEGIES[strategy_name]
    answer = run_fn(case["question"])

    assert answer, f"Empty answer for {strategy_name}/{case['id']}"

    report = _score_answer(answer, case)

    print(f"\n{'=' * 60}")
    print(f"[{strategy_name}] {case['id']}")
    print(f"Score: {report['score']:.0%} ({report['earned']}/{report['total']})")
    print(f"Answer length: {report['answer_length']} chars")
    for m in report["markers"]:
        status = "FOUND" if m["found"] else "MISSING"
        print(f"  [{'*' * m['weight']}] {status}: {m['description']}")
    if report["hallucinations"]:
        print(f"  HALLUCINATIONS: {report['hallucinations']}")
    print(f"{'=' * 60}")

    # No assertions — this is a comparison test, we just collect scores
