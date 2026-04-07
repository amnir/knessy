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
