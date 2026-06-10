"""Eval harness: run a search function over a QuerySet dataset and report.

:class:`EvalRunner` calls ``search_fn(queryset) -> list[str]`` (ranked
predicted patent numbers) for every query, computes per-query and aggregate
metrics (mean recall@{10,25,50,100}, MRR, MAP, mean recall curve), and
catches per-query exceptions into an errors list instead of aborting the run.

:func:`searchfn_from_stores` adapts any KeywordStore into a basic search_fn
(query patent's title+abstract words as keywords) so the harness runs
end-to-end on in-memory stores with no external services.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from patentkit.evals.datasets import QuerySet
from patentkit.evals.metrics import (
    average_precision,
    mean_recall_curve,
    mrr,
    recall_at_k,
    recall_curve,
)
from patentkit.models import PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery

logger = logging.getLogger(__name__)

#: default recall@k cut points
RECALL_KS = (10, 25, 50, 100)


@dataclass
class EvalReport:
    """Results of one eval run: per-query rows plus aggregates.

    Attributes:
        name: run name.
        rows: one dict per successfully evaluated query (recall@k, mrr, ap, ...).
        aggregates: metric name -> mean value across queries.
        mean_curve: mean recall curve over all queries (index k-1 = recall@k).
        errors: per-query failures as {"query_patent", "error"} dicts.
    """

    name: str
    rows: list[dict] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    mean_curve: list[float] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document (aggregates + per-query table)."""
        lines = [f"# Eval report: {self.name}", ""]
        lines += ["## Aggregates", "", "| metric | value |", "| --- | --- |"]
        for metric, value in self.aggregates.items():
            lines.append(f"| {metric} | {value:.4f} |")
        lines.append(f"| queries | {len(self.rows)} |")
        lines.append(f"| errors | {len(self.errors)} |")
        lines.append("")
        if self.rows:
            metric_columns = [key for key in self.rows[0] if key not in ("query_patent",)]
            lines.append("## Per-query results")
            lines.append("")
            lines.append("| query_patent | " + " | ".join(metric_columns) + " |")
            lines.append("| --- |" + " --- |" * len(metric_columns))
            for row in self.rows:
                cells = [
                    f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
                    for col in metric_columns
                ]
                lines.append(f"| {row['query_patent']} | " + " | ".join(cells) + " |")
            lines.append("")
        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for error in self.errors:
                lines.append(f"- `{error['query_patent']}`: {error['error']}")
            lines.append("")
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> Path:
        """Write the full report (rows, aggregates, curve, errors) as JSON."""
        path = Path(path)
        payload = {
            "name": self.name,
            "aggregates": self.aggregates,
            "mean_recall_curve": self.mean_curve,
            "rows": self.rows,
            "errors": self.errors,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved eval report to %s", path)
        return path


class EvalRunner:
    """Run ``search_fn`` over a dataset of QuerySets and build an EvalReport.

    Args:
        search_fn: maps a QuerySet to a ranked list of predicted patent
            number strings.
        dataset: the QuerySets to evaluate.
        name: report name.
    """

    def __init__(
        self,
        search_fn: Callable[[QuerySet], list[str]],
        dataset: list[QuerySet],
        name: str = "eval",
    ):
        self.search_fn = search_fn
        self.dataset = dataset
        self.name = name

    def run(self, max_k: int = 100, progress: Optional[Callable] = None) -> EvalReport:
        """Evaluate every query; per-query exceptions go into ``report.errors``.

        Args:
            max_k: recall curve depth; recall@k columns are emitted for each
                k in {10, 25, 50, 100} that is <= max_k.
            progress: optional callback ``progress(done, total, query_patent)``.
        """
        ks = [k for k in RECALL_KS if k <= max_k]
        report = EvalReport(name=self.name)
        curves: list[list[float]] = []
        total = len(self.dataset)

        for i, queryset in enumerate(self.dataset):
            try:
                predictions = self.search_fn(queryset)
            except Exception as exc:  # noqa: BLE001 - per-query isolation by design
                logger.exception("search_fn failed for %s", queryset.query_patent)
                report.errors.append({"query_patent": queryset.query_patent, "error": str(exc)})
                if progress:
                    progress(i + 1, total, queryset.query_patent)
                continue
            references = queryset.references
            row: dict = {
                "query_patent": queryset.query_patent,
                "n_references": len(set(references)),
                "n_predictions": len(predictions),
            }
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(predictions, references, k)
            row["mrr"] = mrr(predictions, references)
            row["average_precision"] = average_precision(predictions, references)
            report.rows.append(row)
            curves.append(recall_curve(predictions, references, max_k))
            if progress:
                progress(i + 1, total, queryset.query_patent)

        report.mean_curve = mean_recall_curve(curves)
        if report.rows:
            n = len(report.rows)
            for k in ks:
                report.aggregates[f"mean_recall@{k}"] = (
                    sum(row[f"recall@{k}"] for row in report.rows) / n
                )
            report.aggregates["MRR"] = sum(row["mrr"] for row in report.rows) / n
            report.aggregates["MAP"] = sum(row["average_precision"] for row in report.rows) / n
        return report


#: small stopword list for keyword extraction from titles/abstracts
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with which said wherein thereof method system apparatus
    device one each other than may can such using used use based include includes
    including comprising present invention according embodiment embodiments""".split()
)


def searchfn_from_stores(
    keyword_store: KeywordStore,
    *,
    limit: int = 100,
    max_keywords: int = 15,
) -> Callable[[QuerySet], list[str]]:
    """Build a basic search_fn from a KeywordStore's own content.

    For each QuerySet the query patent is looked up in the store; its
    title+abstract words become the search keywords (minus stopwords), the
    query patent itself is excluded, and the ranked result numbers are
    returned. Lets the harness run end-to-end on in-memory stores.
    """
    from patentkit.search.bm25 import tokenize

    def search_fn(queryset: QuerySet) -> list[str]:
        number = PatentNumber.parse(queryset.query_patent)
        patent = keyword_store.get(number)
        if patent is None:
            logger.warning("Query patent %s not in store; returning no predictions",
                           queryset.query_patent)
            return []
        text = f"{patent.title or ''} {patent.abstract or ''}"
        words = [
            token for token in dict.fromkeys(tokenize(text))
            if token not in _STOPWORDS and len(token) > 2
        ]
        keywords = words[:max_keywords]
        query = SearchQuery(
            keywords=keywords,
            minimum_match=max(1, len(keywords) // 4),
            exclude_numbers=[number],
            limit=limit,
        )
        return [str(result.patent_number) for result in keyword_store.search(query)]

    return search_fn
