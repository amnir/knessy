"""
Client for the Knesset OData v3 API.

Three services available:
- ParliamentInfo.svc: Bills, MKs, committees, sessions, factions
- Votes.svc: Vote results, individual MK votes
- MMM.svc: Research center documents

All return JSON via $format=json. Pagination is 100 records/page
with server-driven odata.nextLink for more.
"""

import httpx

BASE_URL = "https://knesset.gov.il/Odata"

PARLIAMENT_URL = f"{BASE_URL}/ParliamentInfo.svc"
VOTES_URL = f"{BASE_URL}/Votes.svc"


async def search_bills(
    query: str | None = None,
    knesset_num: int | None = None,
    top: int = 10,
) -> list[dict]:
    """Search Knesset bills by name substring and/or Knesset number.

    Uses OData $filter with substringof for text search (OData v3 syntax).
    """
    filters = []

    if query:
        filters.append(f"substringof('{query}', Name)")

    if knesset_num:
        filters.append(f"KnessetNum eq {knesset_num}")

    params = {
        "$format": "json",
        "$top": str(top),
        "$orderby": "LastUpdatedDate desc",
    }

    if filters:
        params["$filter"] = " and ".join(filters)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{PARLIAMENT_URL}/KNS_Bill", params=params)
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])


async def get_bill(bill_id: int) -> dict | None:
    """Get a single bill by ID."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{PARLIAMENT_URL}/KNS_Bill({bill_id})",
            params={"$format": "json"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

    return resp.json()


async def get_vote_results(vote_id: int) -> list[dict]:
    """Get individual MK votes for a specific vote."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{VOTES_URL}/vote_rslts_kmmbr_shadow",
            params={
                "$format": "json",
                "$filter": f"vote_id eq {vote_id}",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])


async def list_committees(knesset_num: int | None = None, top: int = 50) -> list[dict]:
    """List Knesset committees, optionally filtered by Knesset number.

    Returns committee ID, name, and type for use in protocol filtering.
    """
    params = {
        "$format": "json",
        "$top": str(top),
        "$select": "CommitteeID,Name,CategoryDesc,KnessetNum",
        "$orderby": "Name",
    }
    if knesset_num:
        params["$filter"] = f"KnessetNum eq {knesset_num}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{PARLIAMENT_URL}/KNS_Committee", params=params)
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])


async def get_bill_votes(bill_id: int) -> list[dict]:
    """Get vote headers (summaries) related to a bill.

    Votes are linked to bills through plenum session items.
    This searches vote headers by bill-related session descriptions.
    """
    # First get the bill to know its name
    bill = await get_bill(bill_id)
    if not bill:
        return []

    bill_name = bill.get("Name", "")
    if not bill_name:
        return []

    # Search votes where the session item description contains the bill name
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{VOTES_URL}/View_vote_rslts_hdr_Approved",
            params={
                "$format": "json",
                "$filter": f"substringof('{bill_name}', sess_item_dscr)",
                "$orderby": "vote_date desc",
                "$top": "20",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])
