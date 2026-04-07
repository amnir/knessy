"""Agent state definition for the research agent graph."""

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


@dataclass
class ResearchTask:
    """A single research sub-task identified by the Planner."""

    tool: str
    args: dict
    reason: str


@dataclass
class ResearchResult:
    """Result from a single research task."""

    task: ResearchTask
    result: str


@dataclass
class GradingResult:
    """CRAG grading result for a single research result."""

    task_tool: str
    task_args: dict
    total_chunks: int
    relevant_chunks: int
    filtered_result: str
    relevance_ratio: float


# Token budget — hard ceiling per agent run to prevent runaway costs.
TOKEN_BUDGET = 200_000


class TokenBudgetExceeded(Exception):
    """Raised when the agent's token budget is exhausted."""


def check_budget(state: dict) -> None:
    """Raise if the token budget has been exhausted."""
    used = state.get("total_tokens", 0)
    if used >= TOKEN_BUDGET:
        raise TokenBudgetExceeded(f"Token budget exhausted ({used:,}/{TOKEN_BUDGET:,})")


class AgentState(TypedDict):
    """State flowing through the agent graph."""

    question: str
    messages: Annotated[list, add_messages]
    research_tasks: list[ResearchTask]
    research_results: list[ResearchResult]
    grading_results: list[GradingResult]
    reformulate: bool
    is_sufficient: bool
    eval_feedback: str
    iteration: int
    final_answer: str
    total_tokens: int
