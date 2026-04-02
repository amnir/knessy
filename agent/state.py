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


class AgentState(TypedDict):
    """State flowing through the agent graph."""

    question: str
    messages: Annotated[list, add_messages]
    research_tasks: list[ResearchTask]
    research_results: list[ResearchResult]
    is_sufficient: bool
    eval_feedback: str
    iteration: int
    final_answer: str
