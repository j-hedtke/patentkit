"""Notification primitives.

A :class:`Notifier` is anything with ``send(subject, body, **kwargs)``;
:func:`notify_search_complete` formats a compact search-completion message
(type, target, result count, top titles, elapsed time) and fans it out to a
list of notifiers, logging — never raising — on individual failures.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Notifier(Protocol):
    """Anything that can deliver a (subject, body) message."""

    def send(self, subject: str, body: str, **kwargs) -> None: ...


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict, pydantic model, or plain object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _describe(session_or_result: Any) -> dict[str, Any]:
    """Duck-typed extraction from a GuidedSearchSession, an agent result
    model (Invalidity/Fto/Infringement SearchResult), or a plain dict."""
    obj = session_or_result
    search_type = _get(obj, "search_type")
    if search_type is None:  # agent result models encode type in the class name
        name = type(obj).__name__.lower()
        for candidate in ("invalidity", "fto", "infringement"):
            if candidate in name:
                search_type = candidate
                break
    target = _get(obj, "target") or _get(obj, "product_description")
    results = _get(obj, "last_results") or _get(obj, "results") or []

    timing = _get(obj, "timing") or {}
    params = _get(obj, "params") or {}
    elapsed = _get(timing, "total") or _get(params, "elapsed_seconds")
    if target is None:
        plan = _get(obj, "plan")
        if plan is not None:
            target = _get(plan, "target")
    return {
        "search_type": search_type or "patent",
        "target": str(target or "?")[:120],
        "results": results,
        "elapsed": elapsed,
    }


def format_completion_message(session_or_result: Any, link: Optional[str] = None) -> tuple[str, str]:
    """Build the (subject, body) for a completion notification."""
    info = _describe(session_or_result)
    subject = f"patentkit: {info['search_type']} search complete — {info['target']}"
    lines = [
        f"Search type: {info['search_type']}",
        f"Target: {info['target']}",
        f"Results: {len(info['results'])}",
    ]
    for i, result in enumerate(info["results"][:3], start=1):
        number = _get(result, "patent_number") or _get(result, "name") or "?"
        title = _get(result, "title") or _get(result, "description") or ""
        score = _get(result, "score")
        line = f"  {i}. {number} — {str(title)[:80]}"
        if score is not None:
            line += f" (score {float(score):.2f})"
        lines.append(line)
    if info["elapsed"] is not None:
        lines.append(f"Elapsed: {float(info['elapsed']):.1f}s")
    if link:
        lines.append(f"Link: {link}")
    return subject, "\n".join(lines)


def notify_search_complete(
    notifiers: Iterable[Notifier],
    session_or_result: Any,
    link: Optional[str] = None,
) -> int:
    """Send a completion message to every notifier; returns the success count.

    Individual notifier failures are logged and skipped so one broken
    webhook never loses the others' notifications.
    """
    subject, body = format_completion_message(session_or_result, link)
    sent = 0
    for notifier in notifiers:
        try:
            notifier.send(subject, body)
            sent += 1
        except Exception:  # noqa: BLE001 — notification must never break a search
            logger.exception("notifier %r failed", type(notifier).__name__)
    return sent


__all__ = ["Notifier", "notify_search_complete", "format_completion_message"]
