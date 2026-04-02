"""
RAG ingestion pipeline for Knesset committee protocols.

Workflow:
1. Fetch protocol document metadata from Knesset OData API
2. Download .doc files to data/protocols/
3. Convert to text using macOS textutil
4. Chunk text with overlap
5. Embed chunks using OpenAI text-embedding-3-small
6. Index into OpenSearch with kNN vector field + raw text (for hybrid search)

Usage:
    python -m ingest.ingest --knesset-num 25 --limit 20
"""

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import httpx
from openai import OpenAI
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

PARLIAMENT_URL = "https://knesset.gov.il/Odata/ParliamentInfo.svc"
DATA_DIR = Path("data/protocols")

# Chunking parameters
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between consecutive chunks


async def fetch_protocol_metadata(
    knesset_num: int,
    limit: int,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:
    """Fetch metadata for committee protocol documents from OData API.

    Paginates with $skip/$top in batches of 100 to collect up to `limit` docs.
    Optional date range filters on LastUpdatedDate (ISO format, e.g. '2024-01-01').
    """
    filter_parts = ["GroupTypeDesc eq 'פרוטוקול ועדה'"]
    if from_date:
        filter_parts.append(f"LastUpdatedDate ge datetime'{from_date}T00:00:00'")
    if to_date:
        filter_parts.append(f"LastUpdatedDate le datetime'{to_date}T23:59:59'")
    odata_filter = " and ".join(filter_parts)

    PAGE_SIZE = 100
    all_docs = []

    async with httpx.AsyncClient(timeout=30) as client:
        skip = 0
        while len(all_docs) < limit:
            batch_size = min(PAGE_SIZE, limit - len(all_docs))
            resp = await client.get(
                f"{PARLIAMENT_URL}/KNS_DocumentCommitteeSession",
                params={
                    "$format": "json",
                    "$top": str(batch_size),
                    "$skip": str(skip),
                    "$filter": odata_filter,
                    "$orderby": "LastUpdatedDate desc",
                },
            )
            resp.raise_for_status()
            batch = resp.json().get("value", [])
            if not batch:
                break
            all_docs.extend(batch)
            skip += len(batch)
            print(f"  Fetched metadata batch: {len(all_docs)} docs so far...")

    # Filter to only .doc/.docx files (skip PDFs for now)
    return [d for d in all_docs if d.get("FilePath", "").endswith((".doc", ".docx"))]


async def download_protocol(doc: dict) -> Path | None:
    """Download a protocol .doc file if not already cached locally."""
    doc_id = doc["DocumentCommitteeSessionID"]
    file_url = doc["FilePath"]
    ext = Path(file_url).suffix
    local_path = DATA_DIR / f"{doc_id}{ext}"

    if local_path.exists():
        return local_path

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        try:
            resp = await client.get(file_url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            print(f"  Failed to download {file_url}: {e}")
            return None

    local_path.write_bytes(resp.content)
    return local_path


def doc_to_text(doc_path: Path) -> str:
    """Convert .doc/.docx to plain text. Uses textutil on macOS, catdoc on Linux."""
    import platform

    if platform.system() == "Darwin":
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-output", tmp_path, str(doc_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"  textutil failed for {doc_path}: {result.stderr.decode()}")
            return ""
        text = Path(tmp_path).read_text(encoding="utf-8")
        Path(tmp_path).unlink()
        return text
    else:
        result = subprocess.run(
            ["catdoc", "-w", str(doc_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"  catdoc failed for {doc_path}: {result.stderr.decode()}")
            return ""
        return result.stdout.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries.

    Splits on double-newlines first (paragraph boundaries), then groups
    paragraphs into chunks of approximately chunk_size characters with
    overlap characters of context from the previous chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk_size, save current and start new
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk)
            # Start next chunk with overlap from end of current
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def embed_chunks(chunks: list[str], openai_client: OpenAI, batch_size: int = 64) -> list[list[float]]:
    """Embed text chunks using OpenAI text-embedding-3-small.

    Sends in batches of batch_size. Kept small (64) to stay under the
    300K token-per-request limit even with long Hebrew chunks.
    """
    if not chunks:
        return []

    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


async def fetch_committee_names() -> dict[int, str]:
    """Fetch committee ID -> name mapping from OData API."""
    mapping = {}
    async with httpx.AsyncClient(timeout=30) as client:
        skip = 0
        while True:
            resp = await client.get(
                f"{PARLIAMENT_URL}/KNS_Committee",
                params={
                    "$format": "json",
                    "$select": "CommitteeID,Name",
                    "$top": "100",
                    "$skip": str(skip),
                },
            )
            resp.raise_for_status()
            items = resp.json().get("value", [])
            if not items:
                break
            for item in items:
                mapping[item["CommitteeID"]] = item["Name"]
            skip += len(items)
    return mapping


async def fetch_session_metadata(session_id: int) -> dict:
    """Fetch committee session details for context."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{PARLIAMENT_URL}/KNS_CommitteeSession({session_id})",
            params={"$format": "json"},
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
    return resp.json()


async def run(
    knesset_num: int,
    limit: int,
    dry_run: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
):
    """Main ingestion pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    date_desc = ""
    if from_date or to_date:
        date_desc = f", {from_date or '...'} to {to_date or '...'}"
    print(f"Fetching protocol metadata (Knesset {knesset_num}, limit {limit}{date_desc})...")
    docs = await fetch_protocol_metadata(knesset_num, limit, from_date, to_date)
    print(f"Found {len(docs)} protocol documents")

    if not docs:
        return

    print("Fetching committee name mapping...")
    committee_names = await fetch_committee_names()
    print(f"Loaded {len(committee_names)} committee names")

    openai_client = OpenAI() if not dry_run else None
    os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False) if not dry_run else None
    total_indexed = 0
    total_errors = 0

    for i, doc in enumerate(docs):
        doc_id = doc["DocumentCommitteeSessionID"]
        session_id = doc["CommitteeSessionID"]
        print(f"\n[{i+1}/{len(docs)}] Processing document {doc_id} (session {session_id})...")

        # Download
        local_path = await download_protocol(doc)
        if not local_path:
            continue

        # Convert to text
        text = doc_to_text(local_path)
        if not text:
            continue
        print(f"  Extracted {len(text)} chars of text")

        # Chunk
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")

        if dry_run:
            continue

        # Embed
        vectors = embed_chunks(chunks, openai_client)
        print(f"  Generated {len(vectors)} embeddings")

        # Get session metadata for context
        session = await fetch_session_metadata(session_id)

        # Build and index records immediately (don't accumulate in memory)
        committee_id = session.get("CommitteeID")
        committee_name = committee_names.get(committee_id, "")
        actions = []
        for j, (chunk, vector) in enumerate(zip(chunks, vectors)):
            actions.append({
                "_index": "knesset-protocols",
                "_id": f"{doc_id}_{j}",
                "_source": {
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "committee_id": committee_id,
                    "committee_name": committee_name,
                    "knesset_num": knesset_num,
                    "session_date": session.get("StartDate"),
                    "chunk_index": j,
                    "text": chunk,
                    "embedding": vector,
                    "source_url": doc["FilePath"],
                },
            })

        success, errors = bulk(os_client, actions)
        total_indexed += success
        total_errors += len(errors)
        print(f"  Indexed {success} chunks (running total: {total_indexed})")

    print(f"\nDone. Total indexed: {total_indexed} chunks ({total_errors} errors)")


def main():
    parser = argparse.ArgumentParser(description="Ingest Knesset committee protocols for RAG")
    parser.add_argument("--knesset-num", type=int, default=25, help="Knesset number (default: 25)")
    parser.add_argument("--limit", type=int, default=20, help="Max protocols to process (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Download and chunk only, skip embedding")
    parser.add_argument("--from-date", type=str, default=None, help="Start date filter (ISO, e.g. 2024-01-01)")
    parser.add_argument("--to-date", type=str, default=None, help="End date filter (ISO, e.g. 2024-12-31)")
    args = parser.parse_args()

    asyncio.run(run(args.knesset_num, args.limit, args.dry_run, args.from_date, args.to_date))


if __name__ == "__main__":
    main()
