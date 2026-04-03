"""
LangGraph agent definition — adaptive research loop.

    planner → researcher → judge → [route]
                                     ├── sufficient → synthesizer → END
                                     ├── reformulate → planner (different terms)
                                     └── insufficient → planner (fill gaps)
"""

from langgraph.graph import END, StateGraph

from agent.judge import judge
from agent.nodes import planner, researcher, synthesizer
from agent.state import AgentState


def after_judge(state: AgentState) -> str:
    """Route after judge: synthesize, reformulate, or research more."""
    if state.get("is_sufficient", False):
        return "synthesizer"
    return "planner"


def build_graph() -> StateGraph:
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("judge", judge)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "judge")

    graph.add_conditional_edges(
        "judge",
        after_judge,
        {
            "synthesizer": "synthesizer",
            "planner": "planner",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Module-level compiled graph — ready to invoke
agent = build_graph()
