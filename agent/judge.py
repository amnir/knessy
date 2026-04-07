"""Combined CRAG grader + evaluator node (the "judge").

Grades all retrieved chunks for relevance and decides sufficiency in a
single gpt-4o-mini call. Replaces the separate grader and evaluator nodes
to cut LLM calls per iteration.
"""

import logging
import re

from openai import OpenAI
from pydantic import BaseModel

log = logging.getLogger("agent")

from agent.state import AgentState, GradingResult, ResearchResult

client = OpenAI(max_retries=3)
JUDGE_MODEL = "gpt-4o-mini"
RELEVANCE_THRESHOLD = 0.5
MAX_ITERATIONS = 3


class JudgeVerdict(BaseModel):
    """Structured output from the judge: chunk relevance + sufficiency."""
    relevant: list[bool]
    sufficient: bool
    guidance: str


def _split_chunks(result_text: str) -> list[str]:
    """Split a search_protocols result string into individual chunks."""
    parts = re.split(r"\n\n---\n\n", result_text)
    if parts and parts[0].startswith("Found "):
        first_newline = parts[0].find("\n\n")
        if first_newline != -1:
            parts[0] = parts[0][first_newline + 2:]
    return [p.strip() for p in parts if p.strip()]


def judge(state: AgentState) -> dict:
    """Grade chunks for relevance, filter noise, and decide sufficiency."""
    question = state["question"]
    results = state.get("research_results", [])
    iteration = state.get("iteration", 0)

    # Force stop after MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS:
        return {
            "is_sufficient": True,
            "reformulate": False,
            "messages": [{"role": "assistant", "content": f"Judge: max iterations ({MAX_ITERATIONS}) reached, proceeding to synthesis"}],
        }

    current_tasks = state.get("research_tasks", [])
    current_task_keys = {f"{t.tool}:{t.args}" for t in current_tasks}

    # Collect all chunks from current iteration's search_protocols results
    all_chunks = []
    chunk_sources = []  # track which result each chunk belongs to
    passthrough_results = []

    for r in results:
        task_key = f"{r.task.tool}:{r.task.args}"
        if r.task.tool != "search_protocols" or task_key not in current_task_keys:
            passthrough_results.append(r)
            continue
        chunks = _split_chunks(r.result)
        if not chunks:
            passthrough_results.append(r)
            continue
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_sources.append(r)

    # No search_protocols chunks to grade — just evaluate sufficiency
    if not all_chunks:
        return {
            "is_sufficient": True,
            "reformulate": False,
            "messages": [{"role": "assistant", "content": "Judge: no protocol chunks to grade, proceeding to synthesis"}],
        }

    # Single LLM call: grade all chunks + decide sufficiency
    numbered = "\n\n".join(
        f"[Chunk {i}]\n{chunk[:800]}" for i, chunk in enumerate(all_chunks)
    )

    from agent.nodes import _check_budget
    _check_budget(state)

    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research judge for an Israeli Knesset data assistant. "
                    "You have two jobs:\n\n"
                    "1. GRADE each numbered chunk: does it contain information that helps "
                    "answer the question? Be generous — same topic, people, or legislation "
                    "counts as relevant. Return a boolean per chunk.\n\n"
                    "2. DECIDE SUFFICIENCY: looking only at the relevant chunks, is there "
                    "enough information to answer the question? Mark sufficient=true if the "
                    "relevant chunks address the question. Mark sufficient=false only if "
                    "there is a specific, named gap.\n\n"
                    "3. GUIDANCE: if insufficient, provide concrete next steps — specific "
                    "Hebrew search terms, committee names, date ranges, or bill IDs to try. "
                    "Empty string if sufficient."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\n{numbered}",
            },
        ],
        response_format=JudgeVerdict,
        temperature=0,
    )

    judge_tokens = response.usage.total_tokens if response.usage else 0

    verdict = response.choices[0].message.parsed
    relevance = verdict.relevant
    # Pad or truncate if model returned wrong length
    if len(relevance) < len(all_chunks):
        relevance.extend([True] * (len(all_chunks) - len(relevance)))
    relevance = relevance[:len(all_chunks)]

    # Build filtered results grouped by original research result
    grading_results = state.get("grading_results", [])
    result_chunks: dict[int, list[str]] = {}
    result_totals: dict[int, int] = {}

    for i, (chunk, source, is_relevant) in enumerate(zip(all_chunks, chunk_sources, relevance)):
        source_id = id(source)
        result_totals.setdefault(source_id, 0)
        result_totals[source_id] += 1
        if is_relevant:
            result_chunks.setdefault(source_id, []).append(chunk)

    # Rebuild filtered research results
    updated_results = list(passthrough_results)
    total_relevant = 0
    total_chunks = len(all_chunks)
    seen_sources = set()

    for source in chunk_sources:
        source_id = id(source)
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)

        relevant = result_chunks.get(source_id, [])
        n_total = result_totals[source_id]
        n_relevant = len(relevant)
        total_relevant += n_relevant
        ratio = n_relevant / n_total if n_total > 0 else 0.0

        grading_results.append(GradingResult(
            task_tool=source.task.tool,
            task_args=source.task.args,
            total_chunks=n_total,
            relevant_chunks=n_relevant,
            filtered_result=source.result if not relevant else "",
            relevance_ratio=ratio,
        ))

        if relevant:
            header = f"Found {n_relevant} relevant protocol excerpts (filtered from {n_total}):"
            filtered_text = header + "\n\n" + "\n\n---\n\n".join(relevant)
            updated_results.append(ResearchResult(task=source.task, result=filtered_text))

    overall_ratio = total_relevant / total_chunks if total_chunks > 0 else 1.0
    should_reformulate = overall_ratio < RELEVANCE_THRESHOLD and iteration < MAX_ITERATIONS
    is_sufficient = verdict.sufficient and not should_reformulate

    log.info("Judge: %d/%d chunks relevant (%.0f%%), sufficient=%s, reformulate=%s",
             total_relevant, total_chunks, overall_ratio * 100, is_sufficient, should_reformulate)

    summary = f"Judge: {total_relevant}/{total_chunks} chunks relevant ({overall_ratio:.0%})"
    if should_reformulate:
        summary += " — triggering reformulation"
    elif is_sufficient:
        summary += " — sufficient"
    else:
        summary += f" — need more research: {verdict.guidance[:100]}"

    return {
        "research_results": updated_results,
        "grading_results": grading_results,
        "reformulate": should_reformulate,
        "is_sufficient": is_sufficient,
        "eval_feedback": verdict.guidance,
        "total_tokens": state.get("total_tokens", 0) + judge_tokens,
        "messages": [{"role": "assistant", "content": summary}],
    }
