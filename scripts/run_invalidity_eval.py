"""Evaluate invalidity search against the IPR-example corpus in Elasticsearch.

Runs each queryset's target patent through InvaliditySearchAgent over the
managed-ES corpus and scores predictions against the IPR ground-truth
references (recall@k, MRR). With ANTHROPIC_API_KEY or OPENAI_API_KEY set, the
search is the full agentic loop and the reasoning trace is saved per query;
without keys it runs the degraded keyword-only mode (reported as such).

Usage:
    python scripts/run_invalidity_eval.py [--index patentkit-eval-corpus]
        [--corpus data/eval_corpus/corpus.jsonl] [--final-k 50] [--out data/eval_corpus]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from patentkit.evals.metrics import mrr, recall_at_k
from patentkit.models import Patent, PatentNumber

from index_eval_corpus import build_client  # noqa: E402 - sibling script

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_invalidity_eval")


def pick_llm():
    from patentkit.llm import get_llm

    if os.environ.get("ANTHROPIC_API_KEY"):
        return get_llm("medium", provider="anthropic"), "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return get_llm("medium", provider="openai"), "openai"
    return None, "none (degraded keyword-only mode)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="patentkit-eval-corpus")
    parser.add_argument("--corpus", default="data/eval_corpus/corpus.jsonl")
    parser.add_argument("--final-k", type=int, default=50)
    parser.add_argument("--out", default="data/eval_corpus")
    args = parser.parse_args()

    from patentkit.agents.invalidity_search import InvaliditySearchAgent
    from patentkit.search.elasticsearch_store import ElasticsearchStore

    querysets = json.loads(Path(args.out, "manifest.json").read_text())["querysets"]
    store = ElasticsearchStore(index=args.index, client=build_client())
    llm, llm_desc = pick_llm()
    log.info("LLM: %s", llm_desc)

    # Targets come from the corpus file (full canonical records incl. claims).
    by_number: dict[str, Patent] = {}
    with Path(args.corpus).open() as fh:
        for line in fh:
            if line.strip():
                p = Patent.model_validate_json(line)
                by_number[str(p.patent_number)] = p

    rows = []
    for qs in querysets:
        target = by_number[str(PatentNumber.parse(qs["query_patent"]))]
        agent = InvaliditySearchAgent(keyword_store=store, llm=llm)
        started = time.time()
        # exclude_examiner_art=False: IPR-derived ground truth may be art the
        # examiner cited (e.g. Wong in IPR2020-01018 sits on the face of the
        # '679 patent) — the product-default exclusion would make such
        # references unreachable and silently floor recall.
        out = agent.search(target, claims=qs.get("claims"), final_k=args.final_k,
                           exclude_examiner_art=False)
        elapsed = time.time() - started
        preds = [r["patent_number"] for r in out.results]
        refs = qs["references"]
        row = {
            "query_patent": qs["query_patent"],
            "n_predictions": len(preds),
            "elapsed_s": round(elapsed, 1),
            "recall@5": recall_at_k(preds, refs, 5),
            "recall@10": recall_at_k(preds, refs, 10),
            "recall@25": recall_at_k(preds, refs, 25),
            "recall@50": recall_at_k(preds, refs, 50),
            "mrr": mrr(preds, refs),
            "found": [r for r in refs if any(
                PatentNumber.parse(r).equivalent(PatentNumber.parse(p)) for p in preds)],
            "stop_reason": getattr(out, "stop_reason", None),
        }
        rows.append(row)
        log.info("%s: r@10=%.2f r@25=%.2f mrr=%.3f (%.1fs)",
                 qs["query_patent"], row["recall@10"], row["recall@25"], row["mrr"], elapsed)
        trace = getattr(out, "trace", None)
        if trace is not None:
            trace_path = Path(args.out) / f"trace_{qs['query_patent']}.json"
            try:
                trace_path.write_text(trace.model_dump_json(indent=2))
                log.info("trace saved to %s", trace_path)
            except Exception:  # noqa: BLE001
                log.warning("could not serialize trace")

    n = len(rows)
    report = {
        "llm": llm_desc,
        "exclusions": "examiner art NOT excluded (IPR ground truth can be face-of-patent art)",
        "index": args.index,
        "corpus_size": len(by_number),
        "final_k": args.final_k,
        "per_query": rows,
        "aggregate": {
            key: round(sum(r[key] for r in rows) / n, 4)
            for key in ("recall@5", "recall@10", "recall@25", "recall@50", "mrr")
        },
    }
    out_path = Path(args.out) / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))
    log.info("full report: %s", out_path)


if __name__ == "__main__":
    main()
