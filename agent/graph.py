"""
LangGraph agent definition — adaptive research loop.

    planner → researcher → grader → [reformulate?]
                                      ├── yes → planner (loop back)
                                      └── no  → evaluator → [sufficient?]
                                                              ├── yes → synthesizer → END
                                                              └── no  → planner (loop back)
"""

from langgraph.graph import END, StateGraph

from agent.grader import grader
from agent.nodes import evaluator, planner, researcher, synthesizer
from agent.state import AgentState


def after_grader(state: AgentState) -> str:
    """Route after grading: reformulate if relevance is too low."""
    if state.get("reformulate", False):
        return "planner"
    return "evaluator"


def should_continue(state: AgentState) -> str:
    """Conditional edge: route based on evaluator's verdict."""
    if state.get("is_sufficient", False):
        return "synthesizer"
    return "planner"


def build_graph() -> StateGraph:
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("grader", grader)
    graph.add_node("evaluator", evaluator)
    graph.add_node("synthesizer", synthesizer)

    # Add edges (the flow)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "grader")

    # Grader decides: reformulate or proceed to evaluation
    graph.add_conditional_edges(
        "grader",
        after_grader,
        {
            "planner": "planner",
            "evaluator": "evaluator",
        },
    )

    # Evaluator decides: synthesize or loop back
    graph.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "synthesizer": "synthesizer",
            "planner": "planner",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Module-level compiled graph — ready to invoke
agent = build_graph()
