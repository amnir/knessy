# Knesset Helper

An agentic research assistant for Israeli Knesset (parliament) data. Ask questions in Hebrew or English about bills, votes, and committee discussions — get grounded, sourced answers.

> [!NOTE]
> **Disclaimer:** This is an educational project. It is not affiliated with, endorsed by, or related to the Knesset or any Israeli government body. Data is sourced from the Knesset's publicly available OData API. Answers are AI-generated and should not be relied upon for legal, political, or official purposes.

## Architecture

```
                        ┌─────────────────────────────┐
                        ▼                             │
User → CLI/Agent →   Planner → Researcher → Evaluator ─── need more
                                                │
                                           sufficient
                                                │
                                                ▼
                                           Synthesizer → Answer
```

**Two interfaces to the same data:**

- **MCP server** — Claude Desktop connects over MCP protocol and calls tools interactively
- **LangGraph agent** — Automated research loop that plans, gathers data, evaluates sufficiency, and synthesizes answers

### LangGraph Agent

The agent uses an adaptive research loop built with [LangGraph](https://github.com/langchain-ai/langgraph). The graph has four nodes:

| Node | Role |
|---|---|
| **Planner** | Breaks the question into research tasks (structured JSON output from the LLM) |
| **Researcher** | Executes tasks against the Knesset API and OpenSearch |
| **Evaluator** | Decides if enough data has been gathered or if more research is needed |
| **Synthesizer** | Produces a grounded answer with source citations |

The conditional edge from Evaluator back to Planner is the core loop — simple questions resolve in one pass, complex analytical questions iterate up to 3 times. The Evaluator prevents both under-researching and infinite loops.

### RAG Pipeline

Committee protocol transcripts are embedded and indexed into OpenSearch for retrieval-augmented generation:

1. **Ingest** — Fetch `.doc` protocol files from the Knesset OData API, convert to text, split into overlapping chunks at paragraph boundaries, embed with OpenAI `text-embedding-3-small`, and bulk-index into OpenSearch
2. **Retrieve** — At query time, embed the question and run hybrid search combining kNN vector similarity with BM25 keyword matching
3. **Augment** — Retrieved chunks are included in the Synthesizer's prompt as grounding context

**Why hybrid search?** Vector search captures semantic similarity ("education reform" matches "school funding"), while BM25 keyword search catches exact terms (Hebrew legal terminology, MK names, bill numbers). Combining both gives better retrieval than either alone — especially important for Hebrew, where morphological variations hurt pure keyword search but exact legal terms need precise matching that embeddings can miss.

### MCP Server

The MCP server exposes four tools that Claude Desktop (or any MCP-compatible client) can call:

- `search_bills` — Search bills by name and Knesset number (OData API)
- `get_bill_details` — Get details for a specific bill (OData API)
- `get_bill_votes` — Get vote results for a bill (OData API)
- `search_protocols` — Hybrid search over embedded committee protocols (OpenSearch)

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector + full-text search | OpenSearch 2.19 (kNN + BM25) |
| Tool protocol | Model Context Protocol (MCP) |
| Data source | Knesset OData v3 API |
| HTTP client | httpx (async) |

## Setup

### Prerequisites

- Python 3.10+
- Docker (for OpenSearch)
- OpenAI API key
- **macOS**: `textutil` (built-in) for `.doc` conversion
- **Linux**: `catdoc` (`apt-get install -y catdoc`) for `.doc` conversion

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Start OpenSearch

```bash
docker compose up -d
```

### Ingest Data

Fetch and index committee protocols (downloads `.doc` files, converts to text, embeds, and indexes):

```bash
# Create the OpenSearch index
PYTHONPATH=. python3 -m ingest.opensearch_setup

# Ingest protocols (adjust --limit for more/fewer)
PYTHONPATH=. python3 -m ingest.ingest --knesset-num 25 --limit 20

# Dry run (download and chunk only, no embeddings/indexing)
PYTHONPATH=. python3 -m ingest.ingest --knesset-num 25 --limit 5 --dry-run
```

### Run the Agent

```bash
PYTHONPATH=. python3 -m agent.run "What education bills were proposed in the 25th Knesset?"
PYTHONPATH=. python3 -m agent.run "מה נאמר בוועדות הכנסת על חדשנות טכנולוגית בחברה הערבית?"
```

### Claude Desktop Integration

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

Restart Claude Desktop. The Knesset tools will appear in the tool picker.

## Data Source

The Knesset OData v3 API provides open access (no auth required) to parliamentary data:

| Service | Data |
|---|---|
| `ParliamentInfo.svc` | Bills, MKs, committees, sessions, factions |
| `Votes.svc` | Vote results, individual MK voting records |
| `MMM.svc` | Research center documents |

All text content is in Hebrew. JSON via `$format=json`.
