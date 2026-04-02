"""
MCP server exposing Knesset data as tools.

Run with:  python -m mcp_server.server
Or configure in Claude Desktop's claude_desktop_config.json.
"""

import dotenv
dotenv.load_dotenv()

from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from opensearchpy import OpenSearch

from mcp_server import knesset_client

openai_client = OpenAI()
os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)

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
async def search_protocols(query: str, top: int = 5) -> str:
    """Search Knesset committee protocol transcripts using hybrid search.

    Combines semantic (vector) search with keyword (BM25) search for best results.
    Use this to find what was said in committee discussions about a specific topic.

    Args:
        query: What to search for (Hebrew or English). Can be a topic, person name,
               or specific question about committee discussions.
        top: Maximum number of results to return (default 5).

    Returns:
        Relevant excerpts from committee protocol transcripts with source info.
    """
    embedding_resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vector = embedding_resp.data[0].embedding

    result = os_client.search(
        index="knesset-protocols",
        body={
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top * 2,
                                },
                            },
                        },
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "boost": 0.3,
                                },
                            },
                        },
                    ],
                },
            },
            "size": top,
            "_source": ["text", "session_id", "session_date", "source_url", "chunk_index"],
        },
    )

    hits = result["hits"]["hits"]
    if not hits:
        return f"No committee protocol content found for: {query}"

    results = []
    for i, hit in enumerate(hits):
        src = hit["_source"]
        date = src.get("session_date", "Unknown date")
        if date and "T" in str(date):
            date = str(date).split("T")[0]

        results.append(
            f"### Result {i+1} (relevance: {hit['_score']:.2f})\n"
            f"**Session {src['session_id']}** | Date: {date}\n"
            f"Source: {src.get('source_url', 'N/A')}\n\n"
            f"{src['text']}"
        )

    return f"Found {len(hits)} relevant protocol excerpts for '{query}':\n\n" + "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
