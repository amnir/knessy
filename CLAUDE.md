# CLAUDE.md

## Project
Knessy — agentic research assistant for Israeli Knesset data. GitHub: github.com/amnir/knessy (private).

## Running

```bash
# Always needed
source .venv/bin/activate
export PYTHONPATH=.

# OpenSearch must be running for RAG
docker compose up -d

# Agent CLI
python3 -m agent.run "your question"

# Web UI (chat + OpenSearch search)
python -m ui.app

# MCP server (usually launched by Claude Desktop, not manually)
python3 -m mcp_server.server

# Ingest more protocols
python3 -m ingest.ingest --knesset-num 25 --limit 20
python3 -m ingest.opensearch_setup  # reset index
```

## Testing

```bash
# Unit tests (no OpenSearch needed)
OPENAI_API_KEY=test-key-not-used PYTHONPATH=. python -m pytest tests/test_nodes.py -v

# Lint + type check
ruff check .
mypy agent/ mcp_server/ ingest/ startup.py
```

OpenAI clients are created at module import time — any test importing agent modules needs `OPENAI_API_KEY` set (any value works).

## Architecture boundaries

- **MCP server** (`mcp_server/`) — serves Claude Desktop over MCP protocol (stdio transport). This is the ONLY place MCP is used.
- **LangGraph agent** (`agent/`) — calls `knesset_client.py` and OpenSearch directly via Python imports. Does NOT use MCP. Do not reference MCP in agent code.
- **Knesset client** (`mcp_server/knesset_client.py`) — shared data access layer used by both the MCP server and the agent.
- **Web UI** (`ui/`) — Gradio app with chat (streams agent progress) and direct OpenSearch search tabs. Pure presentation layer — does not modify agent behavior.
- **Ingestion** (`ingest/`) — offline batch pipeline. Independent from the agent and MCP server.

## Knesset API

- OData v3 at `knesset.gov.il/Odata/{ParliamentInfo,Votes,MMM}.svc`
- No auth required. JSON via `$format=json`. Default page size 100.
- `WebSiteApi` and `OdataV4` endpoints do NOT work (403 / bot-challenge). Don't try them.
- All text content is Hebrew.

## Package policy

All dependencies are pinned to exact versions released before February 2026. Do not upgrade without verifying the new version via web search (check PyPI publish date, owner, and known incidents). This is a supply chain safety requirement.

## Code style

- Docstrings should be concise and factual — describe what the code does, not why it was chosen over alternatives.
- No tutorial-style comments, no "why X over Y" explanations in code.
- No emojis in code or docs unless explicitly asked.

## .doc conversion

Protocol files from the Knesset are 96% `.doc` (binary Word format). Conversion to text uses:
- **macOS**: `textutil` (built-in)
- **Linux**: `catdoc` (must be installed via `apt-get install catdoc`)

Handled automatically in `ingest/ingest.py:doc_to_text()` based on platform detection.

## Claude Desktop config

Located at `~/Library/Application Support/Claude/claude_desktop_config.json`. The MCP server entry points to this project's venv Python and uses `PYTHONPATH` env var (not `cwd`) to resolve modules.
