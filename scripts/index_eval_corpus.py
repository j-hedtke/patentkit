"""Index the scraped eval corpus into managed Elasticsearch.

Auth, in precedence order:
  ELASTICSEARCH_HOST + ELASTICSEARCH_API_KEY      (API key)
  ELASTICSEARCH_HOST + ES_USER + ES_PASSWORD       (basic auth, e.g. ECK 'elastic' user)

Usage:
    python scripts/index_eval_corpus.py [--corpus data/eval_corpus/corpus.jsonl]
        [--index patentkit-eval-corpus] [--recreate]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from patentkit.models import Patent
from patentkit.search.elasticsearch_store import ElasticsearchStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("index_eval_corpus")


def build_client():
    from elasticsearch import Elasticsearch

    host = os.environ["ELASTICSEARCH_HOST"]
    if os.environ.get("ELASTICSEARCH_API_KEY"):
        return Elasticsearch(host, api_key=os.environ["ELASTICSEARCH_API_KEY"],
                             request_timeout=60, verify_certs=False)
    return Elasticsearch(
        host,
        basic_auth=(os.environ["ES_USER"], os.environ["ES_PASSWORD"]),
        request_timeout=60,
        verify_certs=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/eval_corpus/corpus.jsonl")
    parser.add_argument("--index", default="patentkit-eval-corpus")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = build_client()
    info = client.info()
    log.info("connected to %s (es %s)", info["cluster_name"], info["version"]["number"])

    if args.recreate and client.indices.exists(index=args.index):
        client.indices.delete(index=args.index)
        log.info("deleted existing index %s", args.index)

    store = ElasticsearchStore(index=args.index, client=client)
    store.ensure_index()

    patents = []
    with Path(args.corpus).open() as fh:
        for line in fh:
            if line.strip():
                patents.append(Patent.model_validate_json(line))
    count = store.index(patents)
    client.indices.refresh(index=args.index)
    total = client.count(index=args.index)["count"]
    log.info("indexed %d patents; index %s now holds %d docs", count, args.index, total)


if __name__ == "__main__":
    main()
