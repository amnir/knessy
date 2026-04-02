"""
Agent graph nodes — each is a function that takes state and returns partial state updates.

Nodes:
- planner: Analyzes the question and decides what to research
- researcher: Calls Knesset API and OpenSearch to gather data
- evaluator: Decides if enough data has been gathered
- synthesizer: Produces a grounded answer with sources
"""

import json

from openai import OpenAI

from agent.state import AgentState, ResearchResult, ResearchTask
from mcp_server import knesset_client

client = OpenAI()
MODEL = "gpt-4o"

MAX_ITERATIONS = 3

# Maps tool names to actual functions
TOOL_REGISTRY = {
    "search_bills": knesset_client.search_bills,
    "get_bill": knesset_client.get_bill,
    "get_bill_votes": knesset_client.get_bill_votes,
    "get_vote_results": knesset_client.get_vote_results,
    "list_committees": knesset_client.list_committees,
}


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI chat completion. Centralized for consistency."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def planner(state: AgentState) -> dict:
    """Analyze the question and produce a list of research tasks."""
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
        prior_context += "\nPlan NEW queries with DIFFERENT search terms to fill the gaps above."

    system_prompt = """You are a research planner for an Israeli Knesset (parliament) data assistant.
Your job is to plan which tools to call to answer the user's question.

Available tools:
- search_protocols: Search committee protocol transcripts (hybrid semantic + keyword).
    query (str): Hebrew keywords or phrases to search for
    top (int): max results (default 5, use 10 for broad questions)
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

    raw = _call_llm(system_prompt, user_prompt)

    # Parse the JSON response
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    tasks = []
    for item in json.loads(cleaned):
        tasks.append(ResearchTask(
            tool=item["tool"],
            args=item.get("args", {}),
            reason=item.get("reason", ""),
        ))

    return {
        "research_tasks": tasks,
        "iteration": iteration + 1,
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
                # search_protocols goes through OpenSearch, not the OData client
                # Import here to avoid circular imports at module level
                from mcp_server.server import search_protocols
                result = await search_protocols(**task.args)
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
            result = f"Error calling {task.tool}: {e}"

        new_results.append(ResearchResult(task=task, result=str(result)))

    all_results = existing_results + new_results

    return {
        "research_results": all_results,
        "messages": [{"role": "assistant", "content": f"Research: executed {len(new_results)} tool calls, {len(all_results)} total results gathered"}],
    }


def evaluator(state: AgentState) -> dict:
    """Decide if enough data has been gathered to answer the question."""
    question = state["question"]
    results = state.get("research_results", [])
    iteration = state.get("iteration", 0)

    # Safety: force stop after MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS:
        return {
            "is_sufficient": True,
            "messages": [{"role": "assistant", "content": f"Evaluation: max iterations ({MAX_ITERATIONS}) reached, proceeding to synthesis"}],
        }

    # Deduplicate before summarizing
    deduped = _deduplicate_results(results)
    results_summary = ""
    for r in deduped:
        results_summary += f"\n- Tool: {r.task.tool}({r.task.args})\n  Result preview: {r.result[:500]}...\n"

    system_prompt = """You are an evaluator for a Knesset research agent.
Given the user's question and the research gathered so far, decide: is there enough to answer?

Rules:
1. If the results contain relevant information that addresses the question, mark SUFFICIENT.
2. Mark INSUFFICIENT only if there is a specific, named gap — e.g. "found the bill but no vote data", "found committee A but not committee B which was also mentioned."
3. NEVER request more research just for "more results" or "more detail" on the same topic.
4. If the same query was already tried and returned results, those results are final — do not retry.

When marking insufficient, you MUST provide concrete guidance for the next search:
- What specific Hebrew search terms to try
- What specific committee or date range to target
- What specific bill ID or vote to look up

Respond with JSON:
{"sufficient": true/false, "reason": "explanation", "guidance": "specific next steps if insufficient, empty string if sufficient"}

Respond ONLY with the JSON object."""

    user_prompt = f"Question: {question}\n\nResearch results so far (iteration {iteration}):\n{results_summary}"

    raw = _call_llm(system_prompt, user_prompt)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    verdict = json.loads(cleaned)
    is_sufficient = verdict.get("sufficient", True)

    return {
        "is_sufficient": is_sufficient,
        "eval_feedback": verdict.get("guidance", ""),
        "messages": [{"role": "assistant", "content": f"Evaluation: {'sufficient' if is_sufficient else 'need more research'} — {verdict.get('reason', '')}"}],
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
6. Structure long answers with headers for readability."""

    user_prompt = f"Question: {question}\n\nResearch data:\n{results_text}"

    answer = _call_llm(system_prompt, user_prompt)

    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }
