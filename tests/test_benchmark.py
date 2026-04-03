"""
Repeatable benchmark tests for the Knesset research agent.

Runs each benchmark case through the agent and checks that expected
evidence markers appear in the answer. Reports pass/fail with scores.

Usage:
    pytest tests/test_benchmark.py -v --timeout=180
    pytest tests/test_benchmark.py -v -k "karhi"           # run one case
    pytest tests/test_benchmark.py -v -k "retrieval"        # run by tag
"""

import asyncio
import logging

import dotenv
dotenv.load_dotenv()

import pytest

from tests.benchmark_cases import BENCHMARK_CASES

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def _run_agent(question: str) -> str:
    """Run the agent and capture the final answer."""
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


def _score_answer(answer: str, case: dict) -> dict:
    """Score an answer against a benchmark case. Returns detailed report."""
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

    # Check negative markers (hallucination detection)
    hallucinations = []
    for neg in case.get("negative", []):
        if neg in answer:
            hallucinations.append(neg)

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


def _format_report(report: dict) -> str:
    """Format a scoring report for readable test output."""
    lines = [
        f"Score: {report['score']:.0%} ({report['earned']}/{report['total']})",
        f"Answer length: {report['answer_length']} chars",
    ]
    for m in report["markers"]:
        status = "FOUND" if m["found"] else "MISSING"
        lines.append(f"  [{'*' * m['weight']}] {status}: {m['description']}")
    if report["hallucinations"]:
        lines.append(f"  HALLUCINATIONS: {report['hallucinations']}")
    return "\n".join(lines)


# Generate one test per benchmark case
@pytest.fixture(params=BENCHMARK_CASES, ids=[c["id"] for c in BENCHMARK_CASES])
def benchmark_case(request):
    return request.param


# Tag-based markers for pytest -k filtering
def pytest_collection_modifyitems(items):
    for item in items:
        if hasattr(item, "callspec") and "benchmark_case" in item.callspec.params:
            case = item.callspec.params["benchmark_case"]
            for tag in case.get("tags", []):
                item.add_marker(getattr(pytest.mark, tag))


@pytest.mark.benchmark
def test_benchmark(benchmark_case):
    """Run a benchmark case and verify expected evidence markers."""
    answer = _run_agent(benchmark_case["question"])

    assert answer, f"Agent returned empty answer for: {benchmark_case['question'][:50]}..."

    report = _score_answer(answer, benchmark_case)

    # Print detailed report for visibility
    print(f"\n{'=' * 60}")
    print(f"Case: {benchmark_case['id']}")
    print(_format_report(report))
    print(f"{'=' * 60}")

    # Critical markers (weight=3) must ALL be found
    critical_missing = [
        m for m in report["markers"]
        if m["weight"] == 3 and not m["found"]
    ]
    assert not critical_missing, (
        f"Critical markers missing: {[m['description'] for m in critical_missing]}"
    )

    # No hallucinations allowed
    assert not report["hallucinations"], (
        f"Hallucinations detected: {report['hallucinations']}"
    )

    # Overall score must be >= 50%
    assert report["score"] >= 0.5, (
        f"Score too low: {report['score']:.0%} (need >= 50%)"
    )
