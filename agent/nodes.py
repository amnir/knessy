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

    # On subsequent iterations, include what we already know
    prior_context = ""
    if state.get("research_results"):
        prior_context = "\n\nPrevious research results:\n"
        for r in state["research_results"]:
            prior_context += f"\n- Tool: {r.task.tool}({r.task.args})\n  Result: {r.result[:500]}...\n"
        prior_context += "\nThe evaluator determined this is not yet sufficient. Plan additional research to fill the gaps."

    system_prompt = """You are a research planner for an Israeli Knesset (parliament) data assistant.
Your job is to decide which data sources to query to answer the user's question.

Available tools:
- search_bills: Search bills by name (query: str) and/or Knesset number (knesset_num: int, top: int)
- get_bill: Get a specific bill by ID (bill_id: int)
- get_bill_votes: Get votes for a bill (bill_id: int)
- get_vote_results: Get individual MK votes for a vote (vote_id: int)
- search_protocols: Search committee protocol transcripts (query: str, top: int)

Respond with a JSON array of tasks. Each task has:
- "tool": tool name
- "args": dict of arguments
- "reason": why this task is needed

Example:
[
  {"tool": "search_bills", "args": {"query": "חינוך", "knesset_num": 25}, "reason": "Find education-related bills in the 25th Knesset"},
  {"tool": "search_protocols", "args": {"query": "רפורמה בחינוך"}, "reason": "Find committee discussions about education reform"}
]

Respond ONLY with the JSON array, no other text."""

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

    results_summary = ""
    for r in results:
        results_summary += f"\n- Tool: {r.task.tool} | Reason: {r.task.reason}\n  Result preview: {r.result[:300]}...\n"

    system_prompt = """You are an evaluator for a Knesset research agent.
Given the user's question and the research results gathered so far,
decide if there is enough information to provide a good answer.

IMPORTANT: Err on the side of "sufficient". If the results contain relevant
information that directly addresses the question, mark as sufficient.
Only request more research if there are CLEAR, SPECIFIC gaps — for example:
- A bill was mentioned by name but we don't have its details or vote results
- The question asks about multiple topics but results only cover one
- Results returned errors or empty data for a critical query

Do NOT request more research just to get "more" of the same type of data.
Repeating the same search with a higher limit is not useful.

Respond with a JSON object:
{"sufficient": true/false, "reason": "explanation"}

Respond ONLY with the JSON object, no other text."""

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
        "messages": [{"role": "assistant", "content": f"Evaluation: {'sufficient' if is_sufficient else 'need more research'} — {verdict.get('reason', '')}"}],
    }


def synthesizer(state: AgentState) -> dict:
    """Produce a final grounded answer with sources."""
    question = state["question"]
    results = state.get("research_results", [])

    results_text = ""
    for r in results:
        results_text += f"\n--- Source: {r.task.tool}({r.task.args}) ---\n{r.result}\n"

    system_prompt = """You are a Knesset research assistant providing answers to Israeli citizens.

Rules:
1. Answer based ONLY on the research results provided — do not make up information
2. Cite your sources: reference specific bills, votes, committee sessions, or protocol excerpts
3. If the data is insufficient, say what you found and what's missing
4. Answer in the same language as the question (Hebrew or English)
5. Be concise but thorough"""

    user_prompt = f"Question: {question}\n\nResearch data:\n{results_text}"

    answer = _call_llm(system_prompt, user_prompt)

    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }
