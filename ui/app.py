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
# i18n
# ---------------------------------------------------------------------------

STRINGS = {
    "he": {
        "title": "# Knessy\nעוזר מחקר לכנסת — שאלו שאלות או חפשו ישירות בפרוטוקולים.",
        "tab_chat": "צ'אט",
        "tab_search": "חיפוש",
        "chat_placeholder": "שאלו שאלה על הכנסת...",
        "msg_placeholder": "למשל: מה נאמר בוועדות על חדשנות טכנולוגית?",
        "search_query": "שאילתת חיפוש",
        "search_query_placeholder": "למשל: תקציב חינוך",
        "committee": "ועדה (אופציונלי)",
        "committee_placeholder": "למשל: ועדת הכספים",
        "from_date": "מתאריך",
        "to_date": "עד תאריך",
        "results_label": "תוצאות",
        "search_btn": "חיפוש",
        "headers": ["טקסט", "מזהה ישיבה", "ועדה", "תאריך", "קישור"],
        "lang_btn": "English",
        "node_planner": "מתכנן משימות מחקר",
        "node_researcher": "מחפש במאגרי הכנסת",
        "node_judge": "מעריך תוצאות",
        "node_synthesizer": "כותב תשובה",
        "error_prefix": "שגיאת סוכן",
        "try_again": "נסו שוב.",
        "no_results": "לא נמצאו תוצאות",
    },
    "en": {
        "title": "# Knessy\nKnesset research assistant — ask questions or search protocols directly.",
        "tab_chat": "Chat",
        "tab_search": "Search",
        "chat_placeholder": "Ask a question about the Knesset...",
        "msg_placeholder": "e.g. What was said in committees about technological innovation?",
        "search_query": "Search query",
        "search_query_placeholder": "e.g. education budget",
        "committee": "Committee (optional)",
        "committee_placeholder": "e.g. Finance Committee",
        "from_date": "From date",
        "to_date": "To date",
        "results_label": "Results",
        "search_btn": "Search",
        "headers": ["Text", "Session ID", "Committee", "Date", "Source URL"],
        "lang_btn": "עברית",
        "node_planner": "Planning research tasks",
        "node_researcher": "Searching Knesset data",
        "node_judge": "Evaluating results",
        "node_synthesizer": "Writing answer",
        "error_prefix": "Agent error",
        "try_again": "Please try again.",
        "no_results": "No results found",
    },
}

RTL_CSS = """
gradio-app { direction: rtl !important; text-align: right !important; }
gradio-app .tab-wrapper { justify-content: flex-end !important; }
gradio-app [role="tablist"] { flex-direction: row-reverse !important; justify-content: flex-end !important; }
gradio-app .row { flex-direction: row-reverse !important; }
gradio-app input, gradio-app textarea { text-align: right !important; }
gradio-app input[type="range"] { direction: ltr !important; }
"""
LTR_CSS = """
gradio-app { direction: ltr !important; text-align: left !important; }
gradio-app .tab-wrapper { justify-content: space-between !important; }
gradio-app [role="tablist"] { flex-direction: row !important; justify-content: flex-start !important; }
gradio-app .row { flex-direction: row !important; }
gradio-app input, gradio-app textarea { text-align: left !important; }
"""


def node_labels(lang: str) -> dict:
    s = STRINGS[lang]
    return {
        "planner": s["node_planner"],
        "researcher": s["node_researcher"],
        "judge": s["node_judge"],
        "synthesizer": s["node_synthesizer"],
    }


# ---------------------------------------------------------------------------
# Tab 1: Agent Chat
# ---------------------------------------------------------------------------

NODE_LABELS = node_labels("he")


async def respond(message: str, history: list, lang: str):
    """Stream agent node outputs as collapsible ChatMessage steps."""
    labels = node_labels(lang)
    s = STRINGS[lang]
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
                label = labels.get(node_name, node_name)
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
            content=f"{s['error_prefix']}: {e}\n\n{s['try_again']}",
        ))
        yield history, gr.update(interactive=True)
        return

    # Fallback: re-enable input if agent finishes without synthesizer
    yield history, gr.update(interactive=True)


# ---------------------------------------------------------------------------
# Tab 2: Direct OpenSearch search
# ---------------------------------------------------------------------------


def search_fn(query: str, committee: str, date_from: str, date_to: str, top_k: int, lang: str):
    """BM25 text search against the knesset-protocols index."""
    s = STRINGS[lang]
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
        src = hit["_source"]
        rows.append([
            src.get("text", "")[:500],
            str(src.get("session_id", "")),
            src.get("committee_name", ""),
            str(src.get("session_date", ""))[:10],
            src.get("source_url", ""),
        ])

    return rows if rows else [[s["no_results"], "", "", "", ""]]


# ---------------------------------------------------------------------------
# Language switching
# ---------------------------------------------------------------------------


def switch_lang(current_lang: str):
    """Toggle language and return updated components."""
    new_lang = "en" if current_lang == "he" else "he"
    s = STRINGS[new_lang]
    css = RTL_CSS if new_lang == "he" else LTR_CSS
    return (
        new_lang,
        gr.update(value=s["lang_btn"]),
        gr.update(value=f"<style>{css}</style>"),
        gr.update(value=s["title"]),
        gr.update(label=s["tab_chat"]),
        gr.update(label=s["tab_search"]),
        gr.update(placeholder=s["chat_placeholder"]),
        gr.update(placeholder=s["msg_placeholder"]),
        gr.update(label=s["search_query"], placeholder=s["search_query_placeholder"]),
        gr.update(label=s["committee"], placeholder=s["committee_placeholder"]),
        gr.update(label=s["from_date"]),
        gr.update(label=s["to_date"]),
        gr.update(label=s["results_label"]),
        gr.update(value=s["search_btn"]),
        gr.update(headers=s["headers"]),
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

he = STRINGS["he"]

with gr.Blocks(title="Knessy — Knesset Research Assistant") as app:
    lang = gr.State("he")
    css_inject = gr.HTML(f"<style>{RTL_CSS}</style>")

    with gr.Row():
        title_md = gr.Markdown(he["title"])
        lang_btn = gr.Button(he["lang_btn"], size="sm", scale=0, min_width=80)

    with gr.Tabs():
        # ---- Chat tab ----
        with gr.Tab(he["tab_chat"]) as tab_chat:
            chatbot = gr.Chatbot(
                rtl=True,
                height=600,
                placeholder=he["chat_placeholder"],
            )
            msg = gr.Textbox(
                rtl=True,
                placeholder=he["msg_placeholder"],
                show_label=False,
            )

            msg.submit(
                fn=respond,
                inputs=[msg, chatbot, lang],
                outputs=[chatbot, msg],
            )

        # ---- Search tab ----
        with gr.Tab(he["tab_search"]) as tab_search:
            with gr.Row():
                search_query = gr.Textbox(
                    label=he["search_query"],
                    placeholder=he["search_query_placeholder"],
                    rtl=True,
                    scale=3,
                )
                search_committee = gr.Textbox(
                    label=he["committee"],
                    placeholder=he["committee_placeholder"],
                    rtl=True,
                    scale=2,
                )
            with gr.Row():
                search_from = gr.Textbox(
                    label=he["from_date"],
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )
                search_to = gr.Textbox(
                    label=he["to_date"],
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )
                search_top = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label=he["results_label"],
                    scale=1,
                )
                search_btn = gr.Button(he["search_btn"], variant="primary", scale=1)

            search_results = gr.Dataframe(
                headers=he["headers"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True,
                interactive=False,
            )

            search_btn.click(
                fn=search_fn,
                inputs=[search_query, search_committee, search_from, search_to, search_top, lang],
                outputs=search_results,
            )

    lang_btn.click(
        fn=switch_lang,
        inputs=[lang],
        outputs=[
            lang, lang_btn, css_inject, title_md,
            tab_chat, tab_search,
            chatbot, msg,
            search_query, search_committee, search_from, search_to, search_top, search_btn,
            search_results,
        ],
    )

if __name__ == "__main__":
    app.launch()
