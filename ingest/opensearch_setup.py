"""
OpenSearch index setup for Knesset protocol chunks.

Defines the index mapping for hybrid search (kNN vectors + BM25 full-text)
and provides helpers to create/reset the index.
"""

from opensearchpy import OpenSearch

INDEX_NAME = "knesset-protocols"

# text-embedding-3-small produces 1536-dimensional vectors
EMBEDDING_DIM = 1536

INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
        },
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                },
            },
            "text": {
                "type": "text",
                "analyzer": "standard",
            },
            "doc_id": {"type": "keyword"},
            "session_id": {"type": "integer"},
            "committee_id": {"type": "integer"},
            "committee_name": {"type": "keyword"},
            "knesset_num": {"type": "integer"},
            "session_date": {"type": "date"},
            "chunk_index": {"type": "integer"},
            "source_url": {"type": "keyword"},
        }
    },
}


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
    )


def create_index(client: OpenSearch | None = None):
    """Create the knesset-protocols index, deleting any existing one."""
    if client is None:
        client = get_client()

    if client.indices.exists(index=INDEX_NAME):
        print(f"Deleting existing index '{INDEX_NAME}'...")
        client.indices.delete(index=INDEX_NAME)

    print(f"Creating index '{INDEX_NAME}'...")
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    print("Index created successfully.")


if __name__ == "__main__":
    create_index()
