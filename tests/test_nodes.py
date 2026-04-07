"""Unit tests for agent graph nodes with mocked LLM calls."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set dummy key so OpenAI client can be created at import time
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

from agent.state import AgentState, ResearchResult, ResearchTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_openai_response(content: str, total_tokens: int = 100):
    """Build a mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = total_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _base_state(**overrides) -> AgentState:
    defaults = {
        "question": "What was discussed about education reform?",
        "messages": [],
        "research_tasks": [],
        "research_results": [],
        "grading_results": [],
        "reformulate": False,
        "is_sufficient": False,
        "eval_feedback": "",
        "iteration": 0,
        "final_answer": "",
        "total_tokens": 0,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    @patch("agent.nodes.client")
    def test_parses_tasks_from_llm_json(self, mock_client):
        tasks_json = json.dumps([
            {"tool": "search_protocols", "args": {"query": "רפורמה בחינוך"}, "reason": "Search for education reform"},
            {"tool": "search_bills", "args": {"query": "חינוך", "knesset_num": 25}, "reason": "Find education bills"},
        ])
        mock_client.chat.completions.create.return_value = _mock_openai_response(tasks_json)

        from agent.nodes import planner
        result = planner(_base_state())

        assert len(result["research_tasks"]) == 2
        assert result["research_tasks"][0].tool == "search_protocols"
        assert result["research_tasks"][1].tool == "search_bills"
        assert result["iteration"] == 1

    @patch("agent.nodes.client")
    def test_handles_markdown_fenced_json(self, mock_client):
        tasks_json = '```json\n[{"tool": "search_protocols", "args": {"query": "test"}, "reason": "r"}]\n```'
        mock_client.chat.completions.create.return_value = _mock_openai_response(tasks_json)

        from agent.nodes import planner
        result = planner(_base_state())

        assert len(result["research_tasks"]) == 1

    @patch("agent.nodes.client")
    def test_includes_prior_context_on_retry(self, mock_client):
        tasks_json = json.dumps([{"tool": "search_protocols", "args": {"query": "new terms"}, "reason": "r"}])
        mock_client.chat.completions.create.return_value = _mock_openai_response(tasks_json)

        state = _base_state(
            iteration=1,
            research_results=[
                ResearchResult(
                    task=ResearchTask(tool="search_protocols", args={"query": "old"}, reason="r"),
                    result="some result text",
                ),
            ],
            eval_feedback="Need more specific terms",
        )

        from agent.nodes import planner
        planner(state)

        # Verify the prompt includes prior context
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "Previous queries" in user_msg
        assert "Need more specific terms" in user_msg


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class TestResearcher:
    @pytest.mark.asyncio
    async def test_calls_registered_tools(self):
        mock_result = [{"BillID": 1, "Name": "Test Bill"}]

        state = _base_state(
            research_tasks=[
                ResearchTask(tool="search_bills", args={"query": "test", "knesset_num": 25}, reason="r"),
            ],
        )

        with patch("agent.nodes.TOOL_REGISTRY", {"search_bills": AsyncMock(return_value=mock_result)}):
            from agent.nodes import researcher
            result = await researcher(state)

        assert len(result["research_results"]) == 1
        assert "Test Bill" in result["research_results"][0].result

    @pytest.mark.asyncio
    async def test_handles_tool_error_gracefully(self):
        state = _base_state(
            research_tasks=[
                ResearchTask(tool="search_bills", args={"query": "fail"}, reason="r"),
            ],
        )

        with patch("agent.nodes.TOOL_REGISTRY", {"search_bills": AsyncMock(side_effect=Exception("API timeout"))}):
            from agent.nodes import researcher
            result = await researcher(state)

        assert len(result["research_results"]) == 1
        assert "Error calling search_bills" in result["research_results"][0].result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_message(self):
        state = _base_state(
            research_tasks=[
                ResearchTask(tool="nonexistent_tool", args={}, reason="r"),
            ],
        )

        from agent.nodes import researcher
        result = await researcher(state)

        assert "Unknown tool" in result["research_results"][0].result

    @pytest.mark.asyncio
    async def test_accumulates_with_existing_results(self):
        existing = ResearchResult(
            task=ResearchTask(tool="search_bills", args={"query": "old"}, reason="r"),
            result="old result",
        )
        state = _base_state(
            research_tasks=[
                ResearchTask(tool="search_bills", args={"query": "new"}, reason="r"),
            ],
            research_results=[existing],
        )

        with patch("agent.nodes.TOOL_REGISTRY", {"search_bills": AsyncMock(return_value=[])}):
            from agent.nodes import researcher
            result = await researcher(state)

        assert len(result["research_results"]) == 2


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class TestJudge:
    @patch("agent.judge.client")
    def test_sufficient_verdict_proceeds_to_synthesis(self, mock_client):
        from agent.judge import JudgeVerdict

        verdict = JudgeVerdict(relevant=[True, True, False], sufficient=True, guidance="")
        parsed_msg = MagicMock()
        parsed_msg.parsed = verdict
        choice = MagicMock()
        choice.message = parsed_msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.beta.chat.completions.parse.return_value = resp

        task = ResearchTask(tool="search_protocols", args={"query": "test"}, reason="r")
        chunks = "Found 3 relevant protocol excerpts:\n\n### Result 1\nchunk1\n\n---\n\n### Result 2\nchunk2\n\n---\n\n### Result 3\nchunk3"
        state = _base_state(
            iteration=1,
            research_tasks=[task],
            research_results=[ResearchResult(task=task, result=chunks)],
        )

        from agent.judge import judge
        result = judge(state)

        assert result["is_sufficient"] is True
        assert result["reformulate"] is False

    @patch("agent.judge.client")
    def test_low_relevance_triggers_reformulation(self, mock_client):
        from agent.judge import JudgeVerdict

        verdict = JudgeVerdict(relevant=[False, False, False, True], sufficient=False, guidance="try different terms")
        parsed_msg = MagicMock()
        parsed_msg.parsed = verdict
        choice = MagicMock()
        choice.message = parsed_msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.beta.chat.completions.parse.return_value = resp

        task = ResearchTask(tool="search_protocols", args={"query": "test"}, reason="r")
        chunks = "Found 4 excerpts:\n\n### R1\nc1\n\n---\n\n### R2\nc2\n\n---\n\n### R3\nc3\n\n---\n\n### R4\nc4"
        state = _base_state(
            iteration=1,
            research_tasks=[task],
            research_results=[ResearchResult(task=task, result=chunks)],
        )

        from agent.judge import judge
        result = judge(state)

        assert result["reformulate"] is True
        assert result["is_sufficient"] is False

    def test_max_iterations_forces_synthesis(self):
        from agent.judge import MAX_ITERATIONS, judge
        state = _base_state(iteration=MAX_ITERATIONS)

        result = judge(state)

        assert result["is_sufficient"] is True
        assert result["reformulate"] is False


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class TestSynthesizer:
    @patch("agent.nodes.client")
    def test_produces_answer_from_research(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_openai_response("Education reform was discussed...")

        state = _base_state(
            research_results=[
                ResearchResult(
                    task=ResearchTask(tool="search_protocols", args={"query": "test"}, reason="r"),
                    result="Protocol text about education...",
                ),
            ],
        )

        from agent.nodes import synthesizer
        result = synthesizer(state)

        assert "Education reform" in result["final_answer"]

    @patch("agent.nodes.client")
    def test_deduplicates_results(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_openai_response("Answer")

        task = ResearchTask(tool="search_protocols", args={"query": "test"}, reason="r")
        duplicate_results = [
            ResearchResult(task=task, result="same result"),
            ResearchResult(task=task, result="same result"),
        ]
        state = _base_state(research_results=duplicate_results)

        from agent.nodes import synthesizer
        synthesizer(state)

        # Check the prompt only includes the result once
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert user_msg.count("same result") == 1


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    @patch("agent.nodes.client")
    def test_planner_tracks_tokens(self, mock_client):
        tasks_json = json.dumps([{"tool": "search_protocols", "args": {"query": "test"}, "reason": "r"}])
        mock_client.chat.completions.create.return_value = _mock_openai_response(tasks_json, total_tokens=500)

        from agent.nodes import planner
        result = planner(_base_state())

        assert result["total_tokens"] == 500

    @patch("agent.nodes.client")
    def test_budget_exceeded_raises(self, mock_client):
        from agent.nodes import planner
        from agent.state import TOKEN_BUDGET, TokenBudgetExceeded
        state = _base_state(total_tokens=TOKEN_BUDGET)

        with pytest.raises(TokenBudgetExceeded):
            planner(state)


# ---------------------------------------------------------------------------
# OData escaping
# ---------------------------------------------------------------------------

class TestODataEscape:
    def test_escapes_single_quotes(self):
        from mcp_server.knesset_client import _odata_escape
        assert _odata_escape("it's") == "it''s"

    def test_no_change_without_quotes(self):
        from mcp_server.knesset_client import _odata_escape
        assert _odata_escape("hello world") == "hello world"

    def test_multiple_quotes(self):
        from mcp_server.knesset_client import _odata_escape
        assert _odata_escape("a'b'c") == "a''b''c"


# ---------------------------------------------------------------------------
# _fetch retry logic
# ---------------------------------------------------------------------------

class TestFetchRetry:
    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        from unittest.mock import AsyncMock as AM

        from mcp_server.knesset_client import _fetch

        mock_client = MagicMock()
        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        mock_client.get = AM(side_effect=[resp_429, resp_200])

        with patch("mcp_server.knesset_client.asyncio.sleep", new_callable=AM):
            result = await _fetch(mock_client, "http://example.com")

        assert result.status_code == 200
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        from unittest.mock import AsyncMock as AM

        from mcp_server.knesset_client import _fetch

        mock_client = MagicMock()
        resp_500 = MagicMock(status_code=500)
        resp_200 = MagicMock(status_code=200)
        mock_client.get = AM(side_effect=[resp_500, resp_200])

        with patch("mcp_server.knesset_client.asyncio.sleep", new_callable=AM):
            result = await _fetch(mock_client, "http://example.com")

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        from unittest.mock import AsyncMock as AM

        from mcp_server.knesset_client import _fetch

        mock_client = MagicMock()
        resp_400 = MagicMock(status_code=400)
        mock_client.get = AM(return_value=resp_400)

        result = await _fetch(mock_client, "http://example.com")

        assert result.status_code == 400
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_returns_last(self):
        from unittest.mock import AsyncMock as AM

        from mcp_server.knesset_client import MAX_RETRIES, _fetch

        mock_client = MagicMock()
        resp_503 = MagicMock(status_code=503)
        mock_client.get = AM(return_value=resp_503)

        with patch("mcp_server.knesset_client.asyncio.sleep", new_callable=AM):
            result = await _fetch(mock_client, "http://example.com")

        assert result.status_code == 503
        assert mock_client.get.call_count == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

class TestStartup:
    def test_check_env_exits_on_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(SystemExit):
                from startup import check_env
                check_env()

    def test_check_env_passes_with_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from startup import check_env
            check_env()  # should not raise

    def test_opensearch_port_validation(self):
        with patch.dict(os.environ, {"OPENSEARCH_PORT": "not-a-number"}):
            with pytest.raises(SystemExit):
                from startup import check_opensearch
                check_opensearch()
