"""
Client for the Knesset OData v3 API.

Three services available:
- ParliamentInfo.svc: Bills, MKs, committees, sessions, factions
- Votes.svc: Vote results, individual MK votes
- MMM.svc: Research center documents

All return JSON via $format=json. Pagination is 100 records/page
with server-driven odata.nextLink for more.
"""

import asyncio
import logging

import httpx

log = logging.getLogger("knesset_client")

BASE_URL = "https://knesset.gov.il/Odata"

MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds


async def _fetch(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """GET with exponential backoff on 429/5xx errors."""
    for attempt in range(MAX_RETRIES + 1):
        resp = await client.get(url, **kwargs)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE * (2 ** attempt)
                log.warning("Knesset API %d on %s, retrying in %.1fs", resp.status_code, url, delay)
                await asyncio.sleep(delay)
                continue
            log.error("Knesset API %d on %s after %d retries, giving up", resp.status_code, url, MAX_RETRIES)
        return resp
    raise RuntimeError("Unreachable: retry loop completed without returning")


def _odata_escape(value: str) -> str:
    """Escape a string for use in OData v3 filter expressions.

    Single quotes are doubled per the OData v3 spec.
    """
    return value.replace("'", "''")


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
        filters.append(f"substringof('{_odata_escape(query)}', Name)")

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
        resp = await _fetch(client, f"{PARLIAMENT_URL}/KNS_Bill", params=params)
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])


async def get_bill(bill_id: int) -> dict | None:
    """Get a single bill by ID."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _fetch(client, f"{PARLIAMENT_URL}/KNS_Bill({bill_id})", params={"$format": "json"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

    return resp.json()


async def get_vote_results(vote_id: int) -> list[dict]:
    """Get individual MK votes for a specific vote."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _fetch(
            client,
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
        resp = await _fetch(client, f"{PARLIAMENT_URL}/KNS_Committee", params=params)
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
        resp = await _fetch(
            client,
            f"{VOTES_URL}/View_vote_rslts_hdr_Approved",
            params={
                "$format": "json",
                "$filter": f"substringof('{_odata_escape(bill_name)}', sess_item_dscr)",
                "$orderby": "vote_date desc",
                "$top": "20",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    return data.get("value", [])
