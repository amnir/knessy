"""
MCP server exposing Knesset data as tools.

Run with:  python -m mcp_server.server
Or configure in Claude Desktop's claude_desktop_config.json.
"""

import json
import os

import dotenv

dotenv.load_dotenv()

from startup import check_env, check_opensearch

check_env()

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from mcp_server import knesset_client

openai_client = OpenAI()
os_client = check_opensearch()

RERANKER_MODEL = "gpt-4o-mini"

# Feature flags — set via environment variables
ENABLE_HYDE = os.getenv("KNESSY_HYDE", "1") == "1"
ENABLE_RERANK = os.getenv("KNESSY_RERANK", "1") == "1"


def _analyze_query(query: str) -> dict:
    """Extract structured search constraints from a natural language query.

    Returns date range, committee hints, and optimized Hebrew search terms.
    This pre-filters the corpus before semantic search, which is critical
    for precision at scale.
    """
    response = openai_client.chat.completions.create(
        model=RERANKER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract search constraints from questions about the Israeli Knesset. "
                    "Return a JSON object with:\n"
                    '- "from_date": ISO date string or null (e.g. "2024-01-01")\n'
                    '- "to_date": ISO date string or null\n'
                    '- "committee_hint": Hebrew committee name substring or null (e.g. "כלכלה", "חוקה")\n'
                    '- "search_terms": array of 2-3 Hebrew keyword phrases for protocol search\n\n'
                    "Rules:\n"
                    "- Extract dates from context: '2024' → from_date 2024-01-01, to_date 2024-12-31\n"
                    "- If a specific Knesset number is mentioned (e.g. Knesset 25), use 2022-11-15 as start\n"
                    "- Always generate Hebrew search terms, even for English questions\n"
                    "- Include person names, bill names, and topic keywords as separate terms\n"
                    "- committee_hint: ONLY set this if the question explicitly asks about a specific "
                    "committee (e.g. 'what did the Economics Committee discuss'). "
                    "Do NOT set it for questions about a topic, person, or bill — those may be "
                    "discussed across multiple committees.\n"
                    "Respond ONLY with the JSON object."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"from_date": None, "to_date": None, "committee_hint": None, "search_terms": [query]}


async def _resolve_committee_id(hint: str) -> int | None:
    """Resolve a Hebrew committee name hint to a committee ID."""
    if not hint:
        return None
    committees = await knesset_client.list_committees(knesset_num=25, top=100)
    for c in committees:
        if hint in c.get("Name", ""):
            return c["CommitteeID"]
    return None


def _hyde_expand(query: str) -> str:
    """Generate a hypothetical document that would answer the query (HyDE).

    The hypothetical answer is closer in embedding space to real matching
    documents than the raw question is.
    """
    response = openai_client.chat.completions.create(
        model=RERANKER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Knesset committee protocol writer. "
                    "Given a question, write a short paragraph (in Hebrew) that "
                    "would appear in a committee protocol transcript if this topic "
                    "was discussed. Include realistic speaker names and committee language."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content


def _rerank(query: str, hits: list[dict], top: int) -> list[dict]:
    """Rerank search results using an LLM for relevance scoring.

    Takes the initial retrieval candidates and scores each one against
    the query. Returns the top N by relevance score.
    """
    if len(hits) <= top:
        return hits

    # Build candidates for scoring
    candidates = []
    for i, hit in enumerate(hits):
        text = hit["_source"]["text"][:500]
        candidates.append(f"[{i}] {text}")

    response = openai_client.chat.completions.create(
        model=RERANKER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance judge. Given a query and numbered text passages, "
                    "return a JSON array of the passage numbers most relevant to the query, "
                    "ordered by relevance (most relevant first). "
                    f"Return at most {top} numbers.\n"
                    "Respond ONLY with a JSON array of integers, e.g. [3, 0, 7]"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nPassages:\n" + "\n\n".join(candidates),
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        ranked_indices = json.loads(raw)
        return [hits[i] for i in ranked_indices if i < len(hits)][:top]
    except (json.JSONDecodeError, IndexError):
        return hits[:top]

mcp = FastMCP(
    "Knesset Helper",
    instructions=(
        "You have access to Israeli Knesset (parliament) data. "
        "Use these tools to look up bills, votes, and MK information. "
        "All text content is in Hebrew."
    ),
)


@mcp.tool()
async def search_bills(
    query: str | None = None,
    knesset_num: int | None = None,
    top: int = 10,
) -> str:
    """Search Knesset bills by name (Hebrew) and/or Knesset number.

    Args:
        query: Hebrew text to search for in bill names (substring match).
        knesset_num: Filter by Knesset number (e.g., 25 for the current Knesset).
        top: Maximum number of results to return (default 10).

    Returns:
        Formatted list of matching bills with ID, name, status, and dates.
    """
    bills = await knesset_client.search_bills(query=query, knesset_num=knesset_num, top=top)

    if not bills:
        return "No bills found matching the search criteria."

    results = []
    for bill in bills:
        results.append(
            f"- **Bill {bill['BillID']}**: {bill.get('Name', 'N/A')}\n"
            f"  Knesset: {bill.get('KnessetNum', '?')} | "
            f"  Status: {bill.get('StatusTypeDesc', 'Unknown')}\n"
            f"  Last updated: {bill.get('LastUpdatedDate', 'N/A')}"
        )

    return f"Found {len(bills)} bills:\n\n" + "\n\n".join(results)


@mcp.tool()
async def get_bill_details(bill_id: int) -> str:
    """Get detailed information about a specific Knesset bill.

    Args:
        bill_id: The unique bill ID (from search_bills results).

    Returns:
        Bill details including name, initiators, status, and dates.
    """
    bill = await knesset_client.get_bill(bill_id)

    if not bill:
        return f"Bill {bill_id} not found."

    return (
        f"**Bill {bill['BillID']}**: {bill.get('Name', 'N/A')}\n"
        f"- Knesset: {bill.get('KnessetNum', '?')}\n"
        f"- Type: {bill.get('SubTypeDesc', 'N/A')}\n"
        f"- Status: {bill.get('StatusTypeDesc', 'Unknown')}\n"
        f"- Publication date: {bill.get('PublicationDate', 'N/A')}\n"
        f"- Last updated: {bill.get('LastUpdatedDate', 'N/A')}"
    )


@mcp.tool()
async def get_bill_votes(bill_id: int) -> str:
    """Get voting results for a Knesset bill.

    Searches for plenum votes related to the bill.

    Args:
        bill_id: The unique bill ID.

    Returns:
        Vote summaries with for/against/abstain counts.
    """
    votes = await knesset_client.get_bill_votes(bill_id)

    if not votes:
        return f"No votes found for bill {bill_id}."

    results = []
    for vote in votes:
        accepted = "Accepted" if vote.get("is_accepted") else "Rejected"
        results.append(
            f"- **Vote {vote['vote_id']}** ({vote.get('vote_date', 'N/A')}): {accepted}\n"
            f"  {vote.get('sess_item_dscr', 'N/A')}\n"
            f"  For: {vote.get('total_for', 0)} | "
            f"Against: {vote.get('total_against', 0)} | "
            f"Abstain: {vote.get('total_abstain', 0)}"
        )

    return f"Found {len(votes)} votes for bill {bill_id}:\n\n" + "\n\n".join(results)


@mcp.tool()
async def list_committees(knesset_num: int | None = None, top: int = 50) -> str:
    """List Knesset committees with their IDs.

    Use this to find committee IDs for filtering protocol searches.

    Args:
        knesset_num: Filter by Knesset number (e.g. 25 for current).
        top: Maximum results (default 50).

    Returns:
        List of committees with ID, name, and category.
    """
    committees = await knesset_client.list_committees(knesset_num=knesset_num, top=top)

    if not committees:
        return "No committees found."

    results = []
    for c in committees:
        results.append(
            f"- **{c['CommitteeID']}**: {c.get('Name', 'N/A')} "
            f"({c.get('CategoryDesc', 'N/A')})"
        )

    return f"Found {len(committees)} committees:\n\n" + "\n".join(results)


@mcp.tool()
async def search_protocols(
    query: str,
    top: int = 5,
    committee_id: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> str:
    """Search Knesset committee protocol transcripts using hybrid search.

    Combines semantic (vector) search with keyword (BM25) search for best results.
    Use this to find what was said in committee discussions about a specific topic.

    Args:
        query: What to search for (Hebrew or English). Can be a topic, person name,
               or specific question about committee discussions.
        top: Maximum number of results to return (default 5).
        committee_id: Filter by committee ID (use list_committees to find IDs).
        from_date: Start date filter (ISO format, e.g. '2024-01-01').
        to_date: End date filter (ISO format, e.g. '2024-12-31').

    Returns:
        Relevant excerpts from committee protocol transcripts with source info.
    """
    return await _search_protocols_impl(
        query, top, committee_id, from_date, to_date,
        use_analysis=True, use_rerank=True,
    )


async def search_protocols_for_agent(
    query: str,
    top: int = 5,
    committee_id: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> str:
    """Agent-optimized search: skips query analysis and reranking.

    The planner decides filters explicitly. Query analysis can over-constrain
    (e.g. narrowing to a committee when the topic spans multiple committees).
    The judge node handles relevance filtering post-retrieval.
    """
    return await _search_protocols_impl(
        query, top, committee_id, from_date, to_date,
        use_analysis=False, use_rerank=False,
    )


async def _search_protocols_impl(
    query: str,
    top: int = 5,
    committee_id: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    *,
    use_analysis: bool = True,
    use_rerank: bool = True,
) -> str:
    """Core search implementation shared by MCP tool and agent."""
    # Query analysis: extract structured constraints when caller didn't provide them
    if use_analysis:
        analysis = _analyze_query(query) if not (committee_id or from_date or to_date) else None
        if analysis:
            if not from_date and analysis.get("from_date"):
                from_date = analysis["from_date"]
            if not to_date and analysis.get("to_date"):
                to_date = analysis["to_date"]
            if not committee_id and analysis.get("committee_hint"):
                committee_id = await _resolve_committee_id(analysis["committee_hint"])

    do_rerank = use_rerank and ENABLE_RERANK

    # Embedding: optionally use HyDE for better semantic matching
    if ENABLE_HYDE:
        hyde_doc = _hyde_expand(query)
        embedding_resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query, hyde_doc],
        )
        query_vector = embedding_resp.data[0].embedding
        hyde_vector = embedding_resp.data[1].embedding
        search_vector = [(a + b) / 2 for a, b in zip(query_vector, hyde_vector)]
    else:
        embedding_resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        search_vector = embedding_resp.data[0].embedding

    # Build filter clauses for metadata narrowing
    filter_clauses = []
    if committee_id is not None:
        filter_clauses.append({"term": {"committee_id": committee_id}})
    if from_date:
        filter_clauses.append({"range": {"session_date": {"gte": from_date}}})
    if to_date:
        filter_clauses.append({"range": {"session_date": {"lte": to_date}}})

    # Over-fetch candidates for reranking
    fetch_size = top * 4 if do_rerank else top

    # Build kNN clause with filters inside so Lucene can use efficient
    # filtered search (exact kNN on small filtered sets, approximate on large).
    # Putting filters on the outer bool instead causes kNN to run on the full
    # index first and discard filtered-out results after — missing relevant hits.
    knn_clause = {
        "embedding": {
            "vector": search_vector,
            "k": fetch_size,
        },
    }
    if filter_clauses:
        knn_clause["embedding"]["filter"] = {"bool": {"must": filter_clauses}}

    bool_query = {
        "should": [
            {"knn": knn_clause},
            {
                "match": {
                    "text": {
                        "query": query,
                        "boost": 0.3,
                    },
                },
            },
        ],
    }
    if filter_clauses:
        bool_query["filter"] = filter_clauses
        bool_query["minimum_should_match"] = 1

    result = os_client.search(
        index="knesset-protocols",
        body={
            "query": {"bool": bool_query},
            "size": fetch_size,
            "_source": [
                "text", "session_id", "session_date", "source_url",
                "chunk_index", "committee_id", "committee_name",
            ],
        },
    )

    hits = result["hits"]["hits"]
    if not hits:
        return f"No committee protocol content found for: {query}"

    # Rerank if enabled
    if do_rerank:
        hits = _rerank(query, hits, top)

    results = []
    for i, hit in enumerate(hits):
        src = hit["_source"]
        date = src.get("session_date", "Unknown date")
        if date and "T" in str(date):
            date = str(date).split("T")[0]
        committee = src.get("committee_name", "")

        results.append(
            f"### Result {i+1} (relevance: {hit['_score']:.2f})\n"
            f"**Session {src['session_id']}** | Committee: {committee} | Date: {date}\n"
            f"Source: {src.get('source_url', 'N/A')}\n\n"
            f"{src['text']}"
        )

    return f"Found {len(hits)} relevant protocol excerpts for '{query}':\n\n" + "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
