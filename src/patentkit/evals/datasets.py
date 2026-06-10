"""Eval datasets: QuerySet records, JSONL IO, and the packaged toy IPR set.

A :class:`QuerySet` pairs a query patent (the challenged patent) with its
ground-truth prior-art references — the unit of evaluation for prior-art
search pipelines. The packaged ``data/ipr_toy.jsonl`` is a small illustrative
dataset modeled on real IPR proceedings (toy data, not authoritative).

Set the ``PATENTKIT_IPR_TOY_PATH`` environment variable to point
:func:`default_ipr_toy_dataset` at your own JSONL file instead.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: env var that overrides the packaged toy dataset path
IPR_TOY_PATH_ENV = "PATENTKIT_IPR_TOY_PATH"


class QuerySet(BaseModel):
    """One evaluation query: a patent and its known-relevant prior art.

    Attributes:
        query_patent: the challenged/query patent number, e.g. ``"US6502135B1"``.
        claims: the claim numbers at issue (empty = whole patent).
        references: ground-truth prior-art reference numbers.
        metadata: free-form provenance (proceeding number, notes, ...).
    """

    query_patent: str
    claims: list[int] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def _parse_jsonl(text: str, source: str) -> list[QuerySet]:
    items: list[QuerySet] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(QuerySet.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid QuerySet on line {line_number} of {source}: {exc}") from exc
    return items


def load_queryset_jsonl(path: str | Path) -> list[QuerySet]:
    """Load QuerySets from a JSONL file (one JSON object per line)."""
    path = Path(path)
    return _parse_jsonl(path.read_text(encoding="utf-8"), str(path))


def save_queryset_jsonl(items: Iterable[QuerySet], path: str | Path) -> int:
    """Write QuerySets to a JSONL file; returns the count written."""
    path = Path(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")
            count += 1
    logger.info("Wrote %d QuerySets to %s", count, path)
    return count


def default_ipr_toy_dataset() -> list[QuerySet]:
    """The packaged toy IPR dataset (12 queries).

    Modeled on real IPR proceedings but illustrative only — do not treat the
    reference lists as authoritative. Override with the
    ``PATENTKIT_IPR_TOY_PATH`` env var pointing at your own JSONL file.
    """
    override = os.environ.get(IPR_TOY_PATH_ENV)
    if override:
        logger.info("Loading IPR toy dataset override from %s", override)
        return load_queryset_jsonl(override)
    resource = resources.files("patentkit.evals").joinpath("data/ipr_toy.jsonl")
    return _parse_jsonl(resource.read_text(encoding="utf-8"), "patentkit.evals/data/ipr_toy.jsonl")


class UserEvalSetBuilder:
    """Accumulate user-confirmed relevance judgments into QuerySets.

    Call :meth:`add_judgment` as the user confirms or rejects candidate
    references during review; :meth:`to_querysets` emits one QuerySet per
    query patent that has at least one confirmed-relevant reference (rejected
    references are kept in metadata so they can be reused as hard negatives).
    """

    def __init__(self) -> None:
        #: query patent -> {reference -> relevant?} (insertion-ordered)
        self._judgments: dict[str, dict[str, bool]] = {}

    def add_judgment(self, query_patent: str, reference: str, relevant: bool) -> None:
        """Record (or overwrite) one relevance judgment."""
        self._judgments.setdefault(query_patent, {})[reference] = relevant

    def __len__(self) -> int:
        return len(self._judgments)

    def to_querysets(self, metadata: dict | None = None) -> list[QuerySet]:
        """Emit QuerySets for every query with >= 1 confirmed-relevant reference."""
        out: list[QuerySet] = []
        for query_patent, judgments in self._judgments.items():
            relevant = [ref for ref, ok in judgments.items() if ok]
            rejected = [ref for ref, ok in judgments.items() if not ok]
            if not relevant:
                continue
            out.append(
                QuerySet(
                    query_patent=query_patent,
                    references=relevant,
                    metadata={
                        **(metadata or {}),
                        "source": "user_judgments",
                        "rejected": rejected,
                    },
                )
            )
        return out
