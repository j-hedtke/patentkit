"""Harvest :class:`~patentkit.graph.models.MatchPair` rows from the
artifacts the system already emits.

Everything is duck-typed on purpose — no imports from ``patentkit.agents``
or ``patentkit.analysis`` (the graph layer sits below both, and those models
are being reshaped independently):

- **result rows** (``InvaliditySearchResult.results`` etc.): dicts with
  ``patent_number``/``number``/``name``, ``passages`` (dicts with ``text``/
  ``field`` or plain strings), and ``why``;
- **claim charts**: objects exposing ``.references`` -> ``.findings`` ->
  ``.limitation`` (string or ``.text``), ``.status`` and ``.quotes`` —
  ``disclosed`` maps to ``accepted``, ``not_disclosed`` to ``rejected``
  (limitation-level feedback is the most valuable signal);
- **search traces**: objects exposing ``.shortlist_history`` — the final
  shortlist snapshot's ``key_passage``/``why`` entries.

Limitations are accepted as plain strings or objects exposing ``.text``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from patentkit.graph.models import MatchPair

#: cap on passages harvested per result row / quotes per finding
MAX_TEXTS_PER_ITEM = 3


def _limitation_text(limitation: Any) -> str:
    """Plain strings or duck-typed objects exposing ``.text``."""
    text = getattr(limitation, "text", None)
    return str(text) if text is not None else str(limitation)


def _limitation_texts(limitations: Sequence[Any]) -> list[str]:
    return [t for t in (_limitation_text(l).strip() for l in limitations or ()) if t]


# ------------------------------------------------------------- result rows

def harvest_from_results(rows: Any, *, limitations: Sequence[Any] = (),
                         search_id: str = "", created_at: str = "") -> list[MatchPair]:
    """MatchPairs from ranked result rows (``InvaliditySearchResult.results``
    shape, ``AgenticCandidate`` dumps, or ``{"results": [...]}``).

    Result rows do not pair passages to specific limitations, so each
    harvested passage is paired with every given limitation (unreviewed —
    review/feedback later settles which pairings were real). Without
    limitations the pair is recorded with an empty ``query_limitation`` (raw
    residue; promotion ignores blank limitations).
    """
    if isinstance(rows, dict):
        rows = rows.get("results")
    lims = _limitation_texts(limitations) or [""]
    pairs: list[MatchPair] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        patent_id = str(row.get("patent_number") or row.get("number")
                        or row.get("name") or "").strip()
        if not patent_id:
            continue
        texts: list[tuple[str, str]] = []
        for passage in (row.get("passages") or [])[:MAX_TEXTS_PER_ITEM]:
            if isinstance(passage, dict):
                text = str(passage.get("text") or "").strip()
                section = str(passage.get("field") or "")
            else:
                text, section = str(passage).strip(), ""
            if text:
                texts.append((text, section))
        if not texts and str(row.get("why") or "").strip():
            texts = [(str(row["why"]).strip(), "agent_rationale")]
        for text, section in texts:
            for lim in lims:
                pairs.append(MatchPair(
                    query_limitation=lim, matched_text=text, patent_id=patent_id,
                    section=section, search_id=search_id, created_at=created_at,
                ))
    return pairs


# ------------------------------------------------------------- claim charts

_STATUS_TO_OUTCOME = {"disclosed": "accepted", "not_disclosed": "rejected"}


def harvest_from_reference_chart(reference: Any, *, search_id: str = "",
                                 created_at: str = "") -> list[MatchPair]:
    """MatchPairs from one ReferenceChart-shaped object (``.reference_number``
    + ``.findings`` with ``.limitation``/``.status``/``.quotes``)."""
    patent_id = str(getattr(reference, "reference_number", "") or "").strip()
    pairs: list[MatchPair] = []
    for finding in getattr(reference, "findings", None) or []:
        lim = _limitation_text(getattr(finding, "limitation", "")).strip()
        if not lim or not patent_id:
            continue
        status = str(getattr(finding, "status", "") or "")
        outcome = _STATUS_TO_OUTCOME.get(status, "unreviewed")
        feedback_type = "teaches_limitation" if outcome == "accepted" else ""
        section = str(getattr(finding, "citation", "") or "")
        quotes = [str(q).strip() for q in (getattr(finding, "quotes", None) or [])
                  if str(q).strip()][:MAX_TEXTS_PER_ITEM]
        for matched in quotes or [""]:
            pairs.append(MatchPair(
                query_limitation=lim, matched_text=matched, patent_id=patent_id,
                section=section, outcome=outcome, feedback_type=feedback_type,
                search_id=search_id, created_at=created_at,
            ))
    return pairs


def harvest_from_claim_chart(chart: Any, *, search_id: str = "",
                             created_at: str = "") -> list[MatchPair]:
    """MatchPairs from a ClaimChart-shaped object (``.references`` of
    ReferenceChart-shaped objects). Finding statuses map to outcomes:
    disclosed -> accepted, not_disclosed -> rejected, else unreviewed."""
    pairs: list[MatchPair] = []
    for reference in getattr(chart, "references", None) or []:
        pairs += harvest_from_reference_chart(reference, search_id=search_id,
                                              created_at=created_at)
    return pairs


# ------------------------------------------------------------------- traces

def harvest_from_trace(trace: Any, *, limitations: Sequence[Any] = (),
                       search_id: str = "", created_at: str = "") -> list[MatchPair]:
    """MatchPairs from a SearchTrace-shaped object's final shortlist snapshot
    (entries with ``number`` and ``key_passage``/``why``)."""
    history = getattr(trace, "shortlist_history", None) or []
    if not history:
        return []
    rows = [
        {"number": entry.get("number"),
         "passages": [entry["key_passage"]] if entry.get("key_passage") else [],
         "why": entry.get("why")}
        for entry in history[-1] if isinstance(entry, dict)
    ]
    pairs = harvest_from_results(rows, limitations=limitations,
                                 search_id=search_id, created_at=created_at)
    for pair in pairs:
        pair.section = pair.section or "shortlist"
    return pairs


# --------------------------------------------------------------- dispatcher

def harvest_match_pairs(artifact: Any, *, limitations: Sequence[Any] = (),
                        search_id: str = "", created_at: str = "") -> list[MatchPair]:
    """Build MatchPairs from any supported artifact, dispatching on shape.

    Accepts: a list of result-row dicts (or ``{"results": [...]}``), a
    ClaimChart-shaped object, a ReferenceChart-shaped object, a
    SearchTrace-shaped object, or a search-result model exposing
    ``.results`` rows (e.g. ``InvaliditySearchResult``).
    """
    if isinstance(artifact, (list, tuple, dict)):
        return harvest_from_results(artifact, limitations=limitations,
                                    search_id=search_id, created_at=created_at)
    if getattr(artifact, "references", None) is not None:
        return harvest_from_claim_chart(artifact, search_id=search_id,
                                        created_at=created_at)
    if getattr(artifact, "findings", None) is not None:
        return harvest_from_reference_chart(artifact, search_id=search_id,
                                            created_at=created_at)
    if getattr(artifact, "shortlist_history", None) is not None:
        return harvest_from_trace(artifact, limitations=limitations,
                                  search_id=search_id, created_at=created_at)
    results: Optional[Iterable] = getattr(artifact, "results", None)
    if results is not None:
        return harvest_from_results(results, limitations=limitations,
                                    search_id=search_id, created_at=created_at)
    raise TypeError(f"Cannot harvest match pairs from {type(artifact).__name__}")


__all__ = [
    "harvest_match_pairs",
    "harvest_from_results",
    "harvest_from_claim_chart",
    "harvest_from_reference_chart",
    "harvest_from_trace",
    "MAX_TEXTS_PER_ITEM",
]
