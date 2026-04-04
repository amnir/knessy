"""
Knessy UI — chatbot + direct OpenSearch search in a single Gradio app.

Launch:
    source .venv/bin/activate && PYTHONPATH=. python -m ui.app
"""

import time
from dataclasses import asdict

import dotenv
import gradio as gr
from opensearchpy import OpenSearch

dotenv.load_dotenv()

from agent.graph import agent
from ingest.opensearch_setup import INDEX_NAME

# ---------------------------------------------------------------------------
# OpenSearch client (reused across search requests)
# ---------------------------------------------------------------------------
os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)

# ---------------------------------------------------------------------------
# Tab 1: Agent Chat
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "planner": "Planning research tasks",
    "researcher": "Searching Knesset data",
    "judge": "Evaluating results",
    "synthesizer": "Writing answer",
}


async def respond(message: str, history: list):
    """Stream agent node outputs as collapsible ChatMessage steps."""
    # Append user message
    history.append(gr.ChatMessage(role="user", content=message))
    yield history, gr.update(value="", interactive=False)

    initial_state = {
        "question": message,
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

    start = time.time()

    try:
        async for event in agent.astream(initial_state):
            for node_name, node_output in event.items():
                label = NODE_LABELS.get(node_name, node_name)
                elapsed = round(time.time() - start, 1)

                # Close previous pending step
                for msg in history:
                    if (isinstance(msg, gr.ChatMessage)
                            and msg.metadata
                            and msg.metadata.get("status") == "pending"):
                        msg.metadata["status"] = "done"
                        msg.metadata["duration"] = elapsed

                # Build step content based on node type
                content = ""
                if node_name == "planner" and "research_tasks" in node_output:
                    lines = []
                    for t in node_output["research_tasks"]:
                        lines.append(f"- **{t.tool}**({t.args}) -- {t.reason}")
                    content = "\n".join(lines)

                elif node_name == "researcher" and "research_results" in node_output:
                    lines = []
                    for r in node_output["research_results"]:
                        preview = r.result[:200].replace("\n", " ")
                        lines.append(f"- {r.task.tool}: {preview}...")
                    content = "\n".join(lines)

                elif node_name == "judge":
                    lines = []
                    if "grading_results" in node_output:
                        for g in node_output["grading_results"]:
                            lines.append(
                                f"- {g.relevant_chunks}/{g.total_chunks} chunks relevant "
                                f"({g.relevance_ratio:.0%})"
                            )
                    if node_output.get("reformulate"):
                        lines.append("-> Reformulating query")
                    elif node_output.get("is_sufficient"):
                        lines.append("-> Sufficient evidence collected")
                    else:
                        lines.append("-> Need more research")
                    content = "\n".join(lines)

                elif node_name == "synthesizer" and "final_answer" in node_output:
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=node_output["final_answer"],
                    ))
                    yield history, gr.update(interactive=True)
                    return

                # Add step as collapsible accordion
                history.append(gr.ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={"title": label, "status": "pending"},
                ))
                yield history, gr.update(interactive=False)

    except Exception as e:
        history.append(gr.ChatMessage(
            role="assistant",
            content=f"Agent error: {e}\n\nPlease try again.",
        ))
        yield history, gr.update(interactive=True)
        return

    # Fallback: re-enable input if agent finishes without synthesizer
    yield history, gr.update(interactive=True)


# ---------------------------------------------------------------------------
# Tab 2: Direct OpenSearch search
# ---------------------------------------------------------------------------


def search_fn(query: str, committee: str, date_from: str, date_to: str, top_k: int):
    """BM25 text search against the knesset-protocols index."""
    if not query.strip():
        return []

    must = [{"match": {"text": {"query": query}}}]
    filters = []

    if committee.strip():
        filters.append({"match": {"committee_name": committee.strip()}})
    if date_from.strip():
        filters.append({"range": {"session_date": {"gte": date_from.strip()}}})
    if date_to.strip():
        filters.append({"range": {"session_date": {"lte": date_to.strip()}}})

    body = {
        "query": {"bool": {"must": must, "filter": filters}},
        "size": top_k,
        "_source": [
            "text", "session_id", "committee_name", "session_date", "source_url",
        ],
        "sort": [{"_score": "desc"}],
    }

    try:
        resp = os_client.search(index=INDEX_NAME, body=body)
    except Exception as e:
        return [[str(e), "", "", "", ""]]

    rows = []
    for hit in resp["hits"]["hits"]:
        s = hit["_source"]
        rows.append([
            s.get("text", "")[:500],
            str(s.get("session_id", "")),
            s.get("committee_name", ""),
            str(s.get("session_date", ""))[:10],
            s.get("source_url", ""),
        ])

    return rows if rows else [["No results found", "", "", "", ""]]


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Knessy — Knesset Research Assistant") as app:
    gr.Markdown("# Knessy\nKnesset research assistant — ask questions or search protocols directly.")

    with gr.Tabs():
        # ---- Chat tab ----
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                rtl=True,
                height=600,
                placeholder="Ask a question about the Knesset...",
            )
            msg = gr.Textbox(
                rtl=True,
                placeholder="e.g. מה נאמר בוועדות על חדשנות טכנולוגית?",
                show_label=False,
            )

            msg.submit(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg],
            )

        # ---- Search tab ----
        with gr.Tab("Search"):
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search query",
                    placeholder="e.g. תקציב חינוך",
                    rtl=True,
                    scale=3,
                )
                search_committee = gr.Textbox(
                    label="Committee (optional)",
                    placeholder="e.g. ועדת הכספים",
                    rtl=True,
                    scale=2,
                )
            with gr.Row():
                search_from = gr.Textbox(
                    label="From date",
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )
                search_to = gr.Textbox(
                    label="To date",
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )
                search_top = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Results",
                    scale=1,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)

            search_results = gr.Dataframe(
                headers=["Text", "Session ID", "Committee", "Date", "Source URL"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True,
                interactive=False,
            )

            search_btn.click(
                fn=search_fn,
                inputs=[search_query, search_committee, search_from, search_to, search_top],
                outputs=search_results,
            )

if __name__ == "__main__":
    app.launch()
