"""Export the knesset-protocols index using OpenSearch's native snapshot API.

Creates a snapshot inside the container, then copies it out as a tar.gz file
ready for upload to HuggingFace.
"""

import subprocess
from pathlib import Path

import httpx

OPENSEARCH_URL = "http://localhost:9200"
CONTAINER_NAME = "knesset-opensearch"
REPO_NAME = "backup"
SNAPSHOT_NAME = "knesset-protocols"
SNAPSHOT_PATH_IN_CONTAINER = "/usr/share/opensearch/snapshots"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "knesset-protocols-snapshot.tar.gz"


def api(method: str, path: str, json: dict | None = None) -> dict:
    resp = httpx.request(method, f"{OPENSEARCH_URL}{path}", json=json, timeout=120)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def export():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Register a filesystem snapshot repository inside the container
    print("Registering snapshot repository...")
    api("PUT", f"/_snapshot/{REPO_NAME}", json={
        "type": "fs",
        "settings": {"location": SNAPSHOT_PATH_IN_CONTAINER},
    })

    # Delete previous snapshot if it exists
    try:
        api("DELETE", f"/_snapshot/{REPO_NAME}/{SNAPSHOT_NAME}")
        print("Deleted previous snapshot.")
    except httpx.HTTPStatusError:
        pass

    # Create snapshot
    print("Creating snapshot (this may take a moment)...")
    api("PUT", f"/_snapshot/{REPO_NAME}/{SNAPSHOT_NAME}?wait_for_completion=true", json={
        "indices": "knesset-protocols",
        "include_global_state": False,
    })
    print("Snapshot created.")

    # Copy snapshot files out of the container as a tar.gz
    print(f"Copying snapshot to {OUTPUT_FILE}...")
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "tar", "czf", "/tmp/snapshot.tar.gz",
         "-C", SNAPSHOT_PATH_IN_CONTAINER, "."],
        check=True,
    )
    subprocess.run(
        ["docker", "cp", f"{CONTAINER_NAME}:/tmp/snapshot.tar.gz", str(OUTPUT_FILE)],
        check=True,
    )

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Exported snapshot to {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("\nUpload to HuggingFace:")
    print(f"  huggingface-cli upload amnir/knessy-data {OUTPUT_FILE} --repo-type dataset")


if __name__ == "__main__":
    export()
