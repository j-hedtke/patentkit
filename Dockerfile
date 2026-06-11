# patentkit MCP server — streamable-http transport, built for Cloud Run.
#
# Lean by design:
#   * no secrets baked in — ANTHROPIC_API_KEY / PATENTKIT_MCP_TOKEN arrive as
#     env vars injected from Secret Manager at deploy time
#   * no corpus baked in — it lives in the GCS bucket Cloud Run mounts at
#     /data (PATENTKIT_INDEX_JSONL=/data/corpus.jsonl)
#   * no elasticsearch/viz extras — the default in-memory BM25 store needs
#     neither

FROM python:3.12-slim

# Install the package (NOT editable) from only the files the build needs.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[mcp-http,anthropic,openai,docx,pdf,scrape]"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 patentkit
USER patentkit

# serve_http() reads these from the environment: bind on all interfaces and
# listen on Cloud Run's expected port. Override at deploy time if needed.
ENV PATENTKIT_MCP_HOST=0.0.0.0 \
    PATENTKIT_MCP_PORT=8080
EXPOSE 8080

# Work from / so the repo-relative store defaults (e.g. the concept-graph
# store's "data/graph") resolve to /data/... — exactly where Cloud Run
# mounts the GCS volume. Sessions and the corpus are pointed at /data
# explicitly via PATENTKIT_SESSION_DIR / PATENTKIT_INDEX_JSONL.
WORKDIR /

ENTRYPOINT ["patentkit-mcp", "--http"]
