.PHONY: setup export start

# Download pre-built data and load into OpenSearch
setup:
	docker compose up -d
	@echo "Waiting for OpenSearch to start..."
	python -m scripts.restore_snapshot

# Export current OpenSearch index to data/knesset-protocols.jsonl.gz
export:
	python -m scripts.export_snapshot

# Start the web UI
start:
	python -m ui.app
