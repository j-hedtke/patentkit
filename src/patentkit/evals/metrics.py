"""Retrieval metrics over (predictions, references) patent-number lists.

All matching is PatentNumber-equivalence-aware: country code + numeric body
must match, but kind codes (B1 vs B2) and leading zeros are ignored, so
``US7654321B2`` matches ``US7654321``. Strings that cannot be parsed as
patent numbers fall back to exact (case/whitespace-insensitive) matching.

Pure functions only — no dependencies beyond the canonical model.
"""

from __future__ import annotations

from patentkit.models import PatentNumber


def normalize_number(raw: str) -> str:
    """Canonical kind-code-insensitive form of a patent number string.

    ``US7654321B2`` -> ``US7654321``; unparseable strings are returned
    stripped and upper-cased so exact matches still work.
    """
    try:
        parsed = PatentNumber.parse(raw)
    except ValueError:
        return raw.strip().upper()
    return f"{parsed.country_code}{parsed.number.lstrip('0') or '0'}"


def _reference_set(references: list[str]) -> set[str]:
    return {normalize_number(ref) for ref in references}


def recall_at_k(predictions: list[str], references: list[str], k: int) -> float:
    """Fraction of unique references found in the top-k predictions."""
    refs = _reference_set(references)
    if not refs:
        return 0.0
    found = {
        normalized
        for normalized in (normalize_number(p) for p in predictions[:k])
        if normalized in refs
    }
    return len(found) / len(refs)


def recall_curve(predictions: list[str], references: list[str], max_k: int) -> list[float]:
    """Recall at every k in 1..max_k (constant beyond the prediction list length)."""
    refs = _reference_set(references)
    curve: list[float] = []
    found: set[str] = set()
    total = len(refs)
    for k in range(1, max_k + 1):
        if k <= len(predictions):
            normalized = normalize_number(predictions[k - 1])
            if normalized in refs:
                found.add(normalized)
        curve.append(len(found) / total if total else 0.0)
    return curve


def mean_recall_curve(curves: list[list[float]]) -> list[float]:
    """Element-wise mean of recall curves; shorter curves extend their last value."""
    if not curves:
        return []
    max_len = max(len(curve) for curve in curves)
    means: list[float] = []
    for k in range(max_len):
        values = [curve[k] if k < len(curve) else (curve[-1] if curve else 0.0) for curve in curves]
        means.append(sum(values) / len(curves))
    return means


def mrr(predictions: list[str], references: list[str]) -> float:
    """Reciprocal rank of the first relevant prediction (0.0 if none)."""
    refs = _reference_set(references)
    for rank, prediction in enumerate(predictions, start=1):
        if normalize_number(prediction) in refs:
            return 1.0 / rank
    return 0.0


def average_precision(predictions: list[str], references: list[str]) -> float:
    """Average precision: mean of precision@hit over all unique references."""
    refs = _reference_set(references)
    if not refs:
        return 0.0
    hits = 0
    total = 0.0
    seen: set[str] = set()
    for rank, prediction in enumerate(predictions, start=1):
        normalized = normalize_number(prediction)
        if normalized in refs and normalized not in seen:
            seen.add(normalized)
            hits += 1
            total += hits / rank
    return total / len(refs)
