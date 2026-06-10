"""Evaluation layer: datasets, metrics, and the eval harness.

Build or load :class:`QuerySet` datasets (query patent + ground-truth prior
art), run any search function over them with :class:`EvalRunner`, and get
recall@k / MRR / MAP reports.
"""

from patentkit.evals.datasets import (
    QuerySet,
    UserEvalSetBuilder,
    default_ipr_toy_dataset,
    load_queryset_jsonl,
    save_queryset_jsonl,
)
from patentkit.evals.harness import EvalReport, EvalRunner, searchfn_from_stores
from patentkit.evals.metrics import (
    average_precision,
    mean_recall_curve,
    mrr,
    normalize_number,
    recall_at_k,
    recall_curve,
)

__all__ = [
    "QuerySet",
    "UserEvalSetBuilder",
    "default_ipr_toy_dataset",
    "load_queryset_jsonl",
    "save_queryset_jsonl",
    "EvalReport",
    "EvalRunner",
    "searchfn_from_stores",
    "average_precision",
    "mean_recall_curve",
    "mrr",
    "normalize_number",
    "recall_at_k",
    "recall_curve",
]
