# Knesset Helper

Research assistant for Israeli Knesset (parliament) data. Ask questions in Hebrew or English about bills, votes, and committee discussions.

```bash
python3 -m agent.run "What education bills were proposed in the 25th Knesset?"
python3 -m agent.run "מה נאמר בוועדות הכנסת על חדשנות טכנולוגית בחברה הערבית?"
```

> [!NOTE]
> Not affiliated with the Knesset or any Israeli government body. Data is sourced from the Knesset's publicly available OData API. Answers are AI-generated and should not be relied upon for legal, political, or official purposes.

## Setup

**Prerequisites:** Python 3.10+, Docker, OpenAI API key

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY

# Start OpenSearch and restore pre-built index
docker compose up -d
make setup
```

`make setup` downloads a pre-built snapshot from HuggingFace with all indexed protocols — no embedding costs.

To ingest from scratch instead:

```bash
PYTHONPATH=. python3 -m ingest.opensearch_setup
PYTHONPATH=. python3 -m ingest.ingest --knesset-num 25 --limit 20
```

## Usage

### CLI

```bash
PYTHONPATH=. python3 -m agent.run "your question here"
```

### Web UI

```bash
PYTHONPATH=. python -m ui.app
# http://127.0.0.1:7860
```

Two tabs: **Chat** (agent Q&A with streaming progress) and **Search** (direct OpenSearch lookup). Supports Hebrew/English with RTL layout switching.

### MCP Server (Claude Desktop)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knesset-helper": {
      "command": "/path/to/knesset-helper/.venv/bin/python3",
      "args": ["-m", "mcp_server.server"],
      "env": {
        "PYTHONPATH": "/path/to/knesset-helper"
      }
    }
  }
}
```

Exposes `search_bills`, `get_bill_details`, `get_bill_votes`, and `search_protocols` as tools.

## How it works

A [LangGraph](https://github.com/langchain-ai/langgraph) agent runs an adaptive research loop:

```
Planner → Researcher → Judge ─── need more? → back to Planner (up to 3x)
                          │
                     sufficient
                          ↓
                     Synthesizer → Answer
```

The **Planner** breaks questions into research tasks. The **Researcher** executes them against the Knesset OData API and OpenSearch. The **Judge** filters irrelevant results and decides if more data is needed. The **Synthesizer** produces a cited answer.

Committee protocols are indexed into OpenSearch using hybrid search (kNN vectors + BM25 keywords) for retrieval-augmented generation.

## Data source

The [Knesset OData v3 API](https://main.knesset.gov.il/Activity/Info/pages/databases.aspx) provides open access to parliamentary data — bills, votes, MKs, committees, and research documents. All content is in Hebrew.

## License

[MIT](LICENSE)
