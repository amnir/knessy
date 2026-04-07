"""Download a pre-built snapshot from HuggingFace and restore it into OpenSearch.

Uses OpenSearch's native snapshot/restore API for fast, single-operation restore.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

OPENSEARCH_URL = "http://localhost:9200"
CONTAINER_NAME = "knesset-opensearch"
REPO_NAME = "backup"
SNAPSHOT_NAME = "knesset-protocols"
SNAPSHOT_PATH_IN_CONTAINER = "/usr/share/opensearch/snapshots"

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_FILE = SNAPSHOT_DIR / "knesset-protocols-snapshot.tar.gz"

HF_REPO = "amnir/knessy-data"
HF_FILENAME = "knesset-protocols-snapshot.tar.gz"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILENAME}"


def download_snapshot():
    """Download snapshot from HuggingFace if not already present."""
    if SNAPSHOT_FILE.exists():
        size_mb = SNAPSHOT_FILE.stat().st_size / (1024 * 1024)
        print(f"Snapshot already exists ({size_mb:.1f} MB), skipping download.")
        return

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    print("Downloading snapshot from HuggingFace...")

    with httpx.stream("GET", HF_URL, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(SNAPSHOT_FILE, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (1024 * 1024) < 65536:
                    pct = downloaded * 100 // total
                    print(f"  {downloaded // (1024 * 1024)}/{total // (1024 * 1024)} MB ({pct}%)")

    size_mb = SNAPSHOT_FILE.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB")


def wait_for_opensearch(retries: int = 30, delay: float = 2.0):
    """Wait for OpenSearch to be ready."""
    for i in range(retries):
        try:
            resp = httpx.get(f"{OPENSEARCH_URL}/_cluster/health", timeout=5)
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        if i < retries - 1:
            print(f"Waiting for OpenSearch... ({i + 1}/{retries})")
            time.sleep(delay)
    print("OpenSearch not available. Is 'docker compose up -d' running?")
    sys.exit(1)


def api(method: str, path: str, json: dict | None = None) -> dict:
    resp = httpx.request(method, f"{OPENSEARCH_URL}{path}", json=json, timeout=120)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def restore():
    """Restore the snapshot into OpenSearch."""
    wait_for_opensearch()

    # Check if index already has data
    try:
        resp = httpx.get(f"{OPENSEARCH_URL}/knesset-protocols/_count", timeout=10)
        if resp.status_code == 200 and resp.json().get("count", 0) > 0:
            count = resp.json()["count"]
            print(f"Index already has {count} documents. Skipping restore.")
            print("To re-import, first run: python -m ingest.opensearch_setup")
            return
    except (httpx.HTTPStatusError, httpx.ConnectError):
        pass

    # Copy snapshot into the container
    print("Loading snapshot into OpenSearch container...")
    subprocess.run(
        ["docker", "cp", str(SNAPSHOT_FILE), f"{CONTAINER_NAME}:/tmp/snapshot.tar.gz"],
        check=True,
    )
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-c",
         f"mkdir -p {SNAPSHOT_PATH_IN_CONTAINER} && "
         f"tar xzf /tmp/snapshot.tar.gz -C {SNAPSHOT_PATH_IN_CONTAINER}"],
        check=True,
    )

    # Register snapshot repository
    print("Registering snapshot repository...")
    api("PUT", f"/_snapshot/{REPO_NAME}", json={
        "type": "fs",
        "settings": {"location": SNAPSHOT_PATH_IN_CONTAINER},
    })

    # Delete existing index if empty (snapshot restore needs it absent)
    try:
        api("DELETE", "/knesset-protocols")
    except httpx.HTTPStatusError:
        pass

    # Restore
    print("Restoring snapshot...")
    api("POST", f"/_snapshot/{REPO_NAME}/{SNAPSHOT_NAME}/_restore?wait_for_completion=true", json={
        "indices": "knesset-protocols",
        "include_global_state": False,
    })

    # Verify
    resp = httpx.get(f"{OPENSEARCH_URL}/knesset-protocols/_count", timeout=10)
    count = resp.json()["count"]
    print(f"Restored {count} documents. Ready to use!")


def main():
    download_snapshot()
    restore()


if __name__ == "__main__":
    main()
