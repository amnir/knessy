"""CRAG (Corrective RAG) grader node.

Classifies each retrieved chunk as relevant/irrelevant using gpt-4o-mini
structured output. Filters noise before the evaluator and signals the
planner to reformulate when relevance is low.

Uses a single batched API call per research result (not per chunk) to
minimize latency and token spend.
"""

import re

from openai import OpenAI
from pydantic import BaseModel

from agent.state import AgentState, GradingResult, ResearchResult

client = OpenAI()
GRADER_MODEL = "gpt-4o-mini"
RELEVANCE_THRESHOLD = 0.5


class BatchRelevance(BaseModel):
    """Relevance verdict for a batch of chunks. Index-aligned with input."""
    relevant: list[bool]


def _split_chunks(result_text: str) -> list[str]:
    """Split a search_protocols result string into individual chunks."""
    parts = re.split(r"\n\n---\n\n", result_text)
    if parts and parts[0].startswith("Found "):
        first_newline = parts[0].find("\n\n")
        if first_newline != -1:
            parts[0] = parts[0][first_newline + 2:]
    return [p.strip() for p in parts if p.strip()]


def _grade_chunks(question: str, chunks: list[str]) -> list[bool]:
    """Grade all chunks in a single API call. Returns list of bools aligned with input."""
    numbered = "\n\n".join(
        f"[Chunk {i}]\n{chunk[:800]}" for i, chunk in enumerate(chunks)
    )
    response = client.beta.chat.completions.parse(
        model=GRADER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance grader. Given a question and numbered chunks, "
                    "decide for EACH chunk whether it contains information that helps "
                    "answer the question. Be generous — if the chunk discusses the same "
                    "topic, people, or legislation, mark it relevant. "
                    "Return a list of booleans, one per chunk, in order."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\n{numbered}",
            },
        ],
        response_format=BatchRelevance,
        temperature=0,
    )
    verdicts = response.choices[0].message.parsed.relevant
    # Pad or truncate if model returned wrong length
    if len(verdicts) < len(chunks):
        verdicts.extend([True] * (len(chunks) - len(verdicts)))
    return verdicts[:len(chunks)]


def grader(state: AgentState) -> dict:
    """Grade retrieved chunks and filter irrelevant ones."""
    question = state["question"]
    results = state.get("research_results", [])
    iteration = state.get("iteration", 0)

    current_tasks = state.get("research_tasks", [])
    current_task_keys = {f"{t.tool}:{t.args}" for t in current_tasks}

    grading_results = state.get("grading_results", [])
    total_relevant = 0
    total_chunks = 0

    updated_results = []
    for r in results:
        task_key = f"{r.task.tool}:{r.task.args}"

        if r.task.tool != "search_protocols" or task_key not in current_task_keys:
            updated_results.append(r)
            continue

        chunks = _split_chunks(r.result)
        if not chunks:
            updated_results.append(r)
            continue

        verdicts = _grade_chunks(question, chunks)
        relevant_chunks = [c for c, v in zip(chunks, verdicts) if v]

        n_total = len(chunks)
        n_relevant = len(relevant_chunks)
        total_chunks += n_total
        total_relevant += n_relevant
        ratio = n_relevant / n_total if n_total > 0 else 0.0

        grading_results.append(GradingResult(
            task_tool=r.task.tool,
            task_args=r.task.args,
            total_chunks=n_total,
            relevant_chunks=n_relevant,
            filtered_result=r.result if not relevant_chunks else "",
            relevance_ratio=ratio,
        ))

        if relevant_chunks:
            header = f"Found {n_relevant} relevant protocol excerpts (filtered from {n_total}):"
            filtered_text = header + "\n\n" + "\n\n---\n\n".join(relevant_chunks)
            updated_results.append(ResearchResult(task=r.task, result=filtered_text))

    overall_ratio = total_relevant / total_chunks if total_chunks > 0 else 1.0
    should_reformulate = overall_ratio < RELEVANCE_THRESHOLD and iteration < 3

    grading_summary = f"Grader: {total_relevant}/{total_chunks} chunks relevant ({overall_ratio:.0%})"
    if should_reformulate:
        grading_summary += " — triggering reformulation"

    return {
        "research_results": updated_results,
        "grading_results": grading_results,
        "reformulate": should_reformulate,
        "messages": [{"role": "assistant", "content": grading_summary}],
    }
