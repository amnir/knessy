"""
CLI runner for the Knesset research agent.

Usage:
    python -m agent.run "מה נאמר בוועדות על חדשנות טכנולוגית?"
    python -m agent.run "What education bills were proposed in the 25th Knesset?"
"""

import asyncio
import sys

import dotenv
dotenv.load_dotenv()

from startup import check_env
check_env()

from agent.graph import agent


async def run(question: str):
    print(f"Question: {question}\n")
    print("=" * 60)

    initial_state = {
        "question": question,
        "messages": [],
        "research_tasks": [],
        "research_results": [],
        "grading_results": [],
        "reformulate": False,
        "is_sufficient": False,
        "eval_feedback": "",
        "iteration": 0,
        "final_answer": "",
    }

    # Stream node-by-node so we can see the agent's reasoning
    async for event in agent.astream(initial_state):
        for node_name, node_output in event.items():
            print(f"\n[{node_name}]")

            # Show planner's tasks
            if node_name == "planner" and "research_tasks" in node_output:
                for task in node_output["research_tasks"]:
                    print(f"  → {task.tool}({task.args}) — {task.reason}")

            # Show researcher's progress
            if node_name == "researcher" and "research_results" in node_output:
                for r in node_output["research_results"]:
                    preview = r.result[:150].replace("\n", " ")
                    print(f"  ✓ {r.task.tool}: {preview}...")

            # Show judge's verdict
            if node_name == "judge":
                if "grading_results" in node_output:
                    for g in node_output["grading_results"]:
                        print(f"  {g.relevant_chunks}/{g.total_chunks} chunks relevant ({g.relevance_ratio:.0%})")
                if node_output.get("reformulate"):
                    print("  -> Reformulating query (low relevance)")
                elif node_output.get("is_sufficient"):
                    print("  -> Sufficient")
                else:
                    print("  -> Need more research")

            # Show final answer
            if node_name == "synthesizer" and "final_answer" in node_output:
                print(f"\n{'=' * 60}")
                print(f"\nAnswer:\n{node_output['final_answer']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agent.run \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(run(question))


if __name__ == "__main__":
    main()
