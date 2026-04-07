"""Startup checks — validate environment before anything runs."""

import logging
import os
import sys

from opensearchpy import OpenSearch


def setup_logging():
    """Configure consistent logging across all modules."""
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("opensearch").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def check_env():
    """Validate required environment variables. Exits with a clear message if missing."""
    setup_logging()
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.example to .env and fill in the values.", file=sys.stderr)
        sys.exit(1)


def check_opensearch(host: str = "localhost", port: int = 9200) -> OpenSearch:
    """Return an OpenSearch client after verifying connectivity.

    Exits with a clear message if OpenSearch is unreachable.
    """
    client = OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False)
    try:
        info = client.info()
        return client
    except Exception as e:
        print(f"Cannot connect to OpenSearch at {host}:{port}: {e}", file=sys.stderr)
        print("Start it with: docker compose up -d", file=sys.stderr)
        sys.exit(1)
