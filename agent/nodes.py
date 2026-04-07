"""
Agent graph nodes — each is a function that takes state and returns partial state updates.

Nodes:
- planner: Analyzes the question and decides what to research
- researcher: Calls Knesset API and OpenSearch to gather data
- synthesizer: Produces a grounded answer with sources

The judge node (grading + evaluation) lives in agent/judge.py.
"""

import json
import logging

from openai import OpenAI

from agent.state import AgentState, ResearchResult, ResearchTask, check_budget
from mcp_server import knesset_client

log = logging.getLogger("agent")

client = OpenAI(max_retries=3)
MODEL = "gpt-4o"

TOOL_REGISTRY = {
    "search_bills": knesset_client.search_bills,
    "get_bill": knesset_client.get_bill,
    "get_bill_votes": knesset_client.get_bill_votes,
    "get_vote_results": knesset_client.get_vote_results,
    "list_committees": knesset_client.list_committees,
}


def _call_llm(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    """Call OpenAI chat completion. Returns (content, total_tokens)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    tokens = response.usage.total_tokens if response.usage else 0
    content = response.choices[0].message.content or ""
    return content, tokens


def planner(state: AgentState) -> dict:
    """Analyze the question and produce a list of research tasks."""
    check_budget(state)
    question = state["question"]
    iteration = state.get("iteration", 0)

    # On subsequent iterations, include what we already know and evaluator feedback
    prior_context = ""
    if state.get("research_results"):
        prior_context = "\n\nPrevious queries (DO NOT repeat these):\n"
        for r in state["research_results"]:
            prior_context += f"- {r.task.tool}({r.task.args}) → {r.result[:200]}...\n"

        eval_feedback = state.get("eval_feedback", "")
        if eval_feedback:
            prior_context += f"\nEvaluator feedback — what's missing:\n{eval_feedback}\n"

        # Include grading feedback so planner knows what was filtered
        grading_results = state.get("grading_results", [])
        if grading_results:
            prior_context += "\nCRAG grading results (chunks filtered for relevance):\n"
            for g in grading_results:
                prior_context += (
                    f"- {g.task_tool}({g.task_args}): "
                    f"{g.relevant_chunks}/{g.total_chunks} chunks relevant "
                    f"({g.relevance_ratio:.0%})\n"
                )
            if state.get("reformulate", False):
                prior_context += (
                    "\nMost retrieved chunks were IRRELEVANT. "
                    "You MUST use substantially different Hebrew search terms, "
                    "synonyms, or rephrasings. Do not repeat similar queries.\n"
                )

        prior_context += "\nPlan NEW queries with DIFFERENT search terms to fill the gaps above."

    system_prompt = """You are a research planner for an Israeli Knesset (parliament) data assistant.
Your job is to plan which tools to call to answer the user's question.

Available tools:
- search_protocols: Search committee protocol transcripts (hybrid semantic + keyword).
    query (str): Hebrew keywords or phrases to search for
    top (int): max results (always use 10 — irrelevant results are filtered automatically)
    committee_id (int | None): filter by committee ID (from list_committees)
    from_date (str | None): ISO date, e.g. '2024-01-01'
    to_date (str | None): ISO date, e.g. '2024-12-31'
- search_bills: Search bills by name (query: str, knesset_num: int | None, top: int)
- get_bill: Get bill details by ID (bill_id: int)
- get_bill_votes: Get votes for a bill (bill_id: int)
- get_vote_results: Get individual MK votes (vote_id: int)
- list_committees: List committees (knesset_num: int | None, top: int). Returns ID, name, category.

Search strategy (CRITICAL):
1. All Knesset content is in Hebrew. ALWAYS search with Hebrew terms, even if the question is in English.
2. Start with broad search_protocols queries (no committee filter) using varied Hebrew keywords. This is your primary tool.
3. Use multiple DIFFERENT queries for the same topic — synonyms, related terms, names of key people.
   Example for communications law: "חוק התקשורת קרעי", "רפורמה בשידורים", "ועדה מיוחדת תקשורת"
4. Only use list_committees if you specifically need a committee_id to filter. Don't call it speculatively.
5. On retry iterations: DO NOT repeat previous queries. Read the evaluator feedback and try DIFFERENT search terms, different date ranges, or different tools.

Respond with a JSON array of tasks:
[{"tool": "...", "args": {...}, "reason": "..."}]

Respond ONLY with the JSON array."""

    user_prompt = f"Question: {question}{prior_context}"

    raw, tokens_used = _call_llm(system_prompt, user_prompt)

    # Parse the JSON response
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    tasks = []
    try:
        # LLMs sometimes output Python None/True/False instead of JSON null/true/false
        sanitized = cleaned.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
        parsed = json.loads(sanitized)
    except json.JSONDecodeError:
        log.error("Planner returned invalid JSON: %s", cleaned[:200])
        parsed = []

    for item in parsed:
        tasks.append(ResearchTask(
            tool=item["tool"],
            args=item.get("args", {}),
            reason=item.get("reason", ""),
        ))

    log.info("Planner iter %d: %d tasks — %s", iteration + 1, len(tasks),
             ", ".join(f"{t.tool}({t.args.get('query', '')[:30]})" for t in tasks))

    return {
        "research_tasks": tasks,
        "iteration": iteration + 1,
        "total_tokens": state.get("total_tokens", 0) + tokens_used,
        "messages": [{"role": "assistant", "content": f"Planning: identified {len(tasks)} research tasks for iteration {iteration + 1}"}],
    }


async def researcher(state: AgentState) -> dict:
    """Execute research tasks by calling Knesset API and OpenSearch."""
    tasks = state.get("research_tasks", [])
    existing_results = state.get("research_results", [])
    new_results = []

    for task in tasks:
        try:
            if task.tool == "search_protocols":
                # Direct retrieval — query analysis and reranking happen in other nodes.
                from mcp_server.server import search_protocols_for_agent
                result = await search_protocols_for_agent(**task.args)
            elif task.tool in TOOL_REGISTRY:
                result = await TOOL_REGISTRY[task.tool](**task.args)
                # Format raw API results for readability
                if isinstance(result, list):
                    result = json.dumps(result, ensure_ascii=False, indent=2)
                elif isinstance(result, dict):
                    result = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                result = f"Unknown tool: {task.tool}"
        except Exception as e:
            log.error("Tool %s(%s) failed: %s", task.tool, task.args, e, exc_info=True)
            result = f"Error calling {task.tool}: {e}"

        new_results.append(ResearchResult(task=task, result=str(result)))

    all_results = existing_results + new_results
    log.info("Researcher: %d calls done, %d total results", len(new_results), len(all_results))

    return {
        "research_results": all_results,
        "messages": [{"role": "assistant", "content": f"Research: executed {len(new_results)} tool calls, {len(all_results)} total results gathered"}],
    }



def _deduplicate_results(results: list[ResearchResult]) -> list[ResearchResult]:
    """Remove duplicate results (same tool + same args)."""
    seen = set()
    deduped = []
    for r in results:
        key = f"{r.task.tool}:{json.dumps(r.task.args, sort_keys=True)}"
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _truncate_results(results: list[ResearchResult], max_chars: int = 20000) -> str:
    """Build results text, truncating to stay within token limits."""
    results_text = ""
    for r in results:
        entry = f"\n--- {r.task.tool}({r.task.args}) ---\n{r.result}\n"
        if len(results_text) + len(entry) > max_chars:
            # Truncate this entry to fit
            remaining = max_chars - len(results_text)
            if remaining > 200:
                results_text += entry[:remaining] + "\n[truncated]\n"
            break
        results_text += entry
    return results_text


def synthesizer(state: AgentState) -> dict:
    """Produce a final grounded answer with sources."""
    question = state["question"]
    results = _deduplicate_results(state.get("research_results", []))
    results_text = _truncate_results(results)

    system_prompt = """You are a Knesset research assistant. Analyze the research data and answer the question.

Rules:
1. Base your answer ONLY on the provided research data. Do not invent information.
2. Quote directly from committee protocols when possible — exact Hebrew quotes in quotation marks are highly valuable.
3. Cite sources: include session IDs, committee names, and dates.
4. If the data partially answers the question, present what you found and note what's missing.
5. Answer in the same language as the question.
6. Structure long answers with headers for readability.
7. Preserve exact Hebrew terminology from the source data — do not paraphrase Hebrew names, titles, or institutional terms."""

    user_prompt = f"Question: {question}\n\nResearch data:\n{results_text}"

    answer, tokens_used = _call_llm(system_prompt, user_prompt)
    log.info("Synthesizer: produced %d char answer", len(answer))

    return {
        "final_answer": answer,
        "total_tokens": state.get("total_tokens", 0) + tokens_used,
        "messages": [{"role": "assistant", "content": answer}],
    }
