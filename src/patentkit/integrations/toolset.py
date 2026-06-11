"""The single tool surface wrapped by both the MCP server and OpenAI tools.

:class:`PatentToolset` methods all return JSON-serializable dicts and their
docstrings double as tool descriptions. :data:`TOOL_SPECS` hand-writes the
JSON-schema for every method (the full :class:`~patentkit.search.base.
SearchQuery` parameter set is enumerated on ``search_patents``), and
:func:`dispatch` routes a (name, arguments) tool call to the right method.

Optional layers (connectors, analysis, viz, evals) are imported lazily
inside methods and degrade into helpful ``{"error": ...}`` dicts, so the
toolset works keys-free with just the in-memory BM25 store.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from patentkit.agents.feedback import SearchFeedback
from patentkit.agents.guided import GuidedSearch, GuidedSearchSession, SessionStore
from patentkit.agents.planner import (
    PlannedQuery,
    QuerySpec,
    SearchPlan,
    estimate_search_seconds,
    humanize_seconds,
)
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import SearchResult
from patentkit.search.bm25 import BM25Store

logger = logging.getLogger(__name__)

#: cap on prosecution-history text fed to the key-limitations summarizer
_WRAPPER_CHAR_CAP = 60_000

_KEY_LIMITATIONS_PROMPT = """\
You are a patent prosecution analyst. Identify the claim limitations that \
secured allowance of this claim — limitations ADDED BY AMENDMENT or ARGUED \
by the applicant to distinguish the prior art. These are the prime targets \
of an invalidity search.

Patent {patent}, claim {claim_number}:
{claim}

Limitations of the claim:
{limitations}

Prosecution-history excerpts (office actions, applicant remarks, notice of \
allowance), oldest first:
{wrapper}

Respond with JSON only:
{{"key_limitations": [{{"limitation": "<verbatim text of one limitation from \
the list above>", "why": "<one sentence: added/argued and against what \
rejection or art>"}}], "summary": "<2-3 sentence narrative of how the claim \
was allowed>"}}
"""


def _result_dict(result: SearchResult) -> dict:
    return {
        "patent_number": str(result.patent_number),
        "title": result.patent.title if result.patent else None,
        "score": round(float(result.score), 4),
        "passages": [
            {"text": p.text, "field": p.field, "score": round(float(p.score), 4)}
            for p in result.passages
        ],
        "explanation": result.explanation,
    }


def _trace_summary(session: GuidedSearchSession) -> Optional[dict]:
    """Compact summary of the persisted agent trace, when one exists."""
    trace = session.params.get("trace")
    if not trace:
        return None
    return {
        "step_count": len(trace.get("steps", [])),
        "queries_issued": [q.get("arguments") for q in trace.get("queries", [])],
        "shortlist": (trace.get("shortlist_history") or [[]])[-1],
        "stop_reason": trace.get("stop_reason"),
        "elapsed_s": trace.get("elapsed_s"),
    }


def _session_dict(session: GuidedSearchSession) -> dict:
    return {
        "session_id": session.id,
        "search_type": session.search_type,
        "state": session.state,
        "iteration": session.iteration,
        "plan": session.plan.model_dump(mode="json") if session.plan else None,
        "estimated_seconds": session.params.get("estimated_seconds"),
        "estimated_human": session.params.get("estimated_human"),
        "n_results": len(session.last_results),
        "stop_reason": session.params.get("stop_reason"),
        "elapsed_seconds": session.params.get("elapsed_seconds"),
        "trace_summary": _trace_summary(session),
    }


def _progress_markdown(session: GuidedSearchSession) -> str:
    """Short chat-ready progress summary rendered after each execution round.

    Ground-truth-agnostic: candidates are listed with the agent's own "why",
    never with relevance verdicts. Degraded keys-free runs are labeled.
    """
    result = session.params.get("result") or {}
    mode = (result.get("plan_or_params") or {}).get("mode")
    trace = session.params.get("trace") or {}
    queries = trace.get("queries") or []
    round_number = int(session.params.get("executions") or 0) or 1
    lines = [
        f"### Guided search — round {round_number} complete (session `{session.id}`)",
        "",
        f"- Stop reason: `{session.params.get('stop_reason') or '?'}`; "
        f"elapsed {session.params.get('elapsed_seconds')}s",
    ]
    if mode == "degraded_keyword_only":
        lines.append("- Mode: DEGRADED keys-free keyword pass — not agentic-mode performance.")
    if queries:
        lines.append(f"- Queries issued this round ({len(queries)}):")
        for query in queries[:8]:
            arguments = json.dumps(query.get("arguments") or {}, default=str)
            if len(arguments) > 160:
                arguments = arguments[:160] + " …"
            lines.append(f"  - `{query.get('tool', 'search')}` `{arguments}`")
        if len(queries) > 8:
            lines.append(f"  - … and {len(queries) - 8} more")
    if session.last_results:
        lines.append(f"- Current candidates ({len(session.last_results)}):")
        for i, candidate in enumerate(session.last_results[:5], start=1):
            number = candidate.get("patent_number", "?")
            title = candidate.get("title") or ""
            why = (candidate.get("why") or "").strip()
            entry = f"  {i}. **{number}**" + (f" — {title}" if title else "")
            if why:
                entry += f" _({why[:120]})_"
            lines.append(entry)
        if len(session.last_results) > 5:
            lines.append(f"  - … and {len(session.last_results) - 5} more")
    else:
        lines.append("- No candidates yet.")
    lines += [
        "",
        "_Send `guided_search_feedback` to steer the next round, or call "
        "`guided_search_execute` again to continue this same agent "
        "conversation._",
    ]
    return "\n".join(lines)


class PatentToolset:
    """Every patentkit capability behind one JSON-in / JSON-out surface.

    Args:
        keyword_store: keyword store (default: empty in-memory
            :class:`BM25Store` — index patents via :meth:`index_patents`).
        vector_store: optional vector store for semantic reranking.
        llm: optional LLM; when omitted and ``provider`` is set, a default
            HIGH-effort model for that provider is constructed lazily.
        provider: "anthropic" or "openai" (used only when ``llm`` is None).
        session_dir: directory for guided-session JSON persistence.
        notifiers: notifiers used by :meth:`notify` and search completion.
    """

    def __init__(self, keyword_store=None, vector_store=None, llm=None,
                 provider: Optional[str] = None, session_dir: Optional[str] = None,
                 notifiers: Iterable = ()):
        self.keyword_store = keyword_store if keyword_store is not None else BM25Store()
        self.vector_store = vector_store
        self.provider = provider
        self._llm = llm
        self._llm_resolved = llm is not None
        self.notifiers = list(notifiers)
        self.sessions = SessionStore(session_dir)
        self._guided: Optional[GuidedSearch] = None
        #: (normalized patent number, claim number) -> most recent ClaimChart,
        #: so DOCX export never re-runs LLM calls
        self._charts: dict[tuple[str, int], Any] = {}

    # ------------------------------------------------------------ internals
    @property
    def llm(self):
        """The configured LLM, lazily constructed from ``provider`` — or None
        (keys-free degraded mode) when neither is available."""
        if not self._llm_resolved:
            self._llm_resolved = True
            if self.provider:
                try:
                    from patentkit.llm import get_llm  # noqa: PLC0415
                    self._llm = get_llm("high", provider=self.provider)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not build %s LLM (%s); running keys-free", self.provider, exc)
                    self._llm = None
        return self._llm

    @property
    def guided(self) -> GuidedSearch:
        if self._guided is None:
            self._guided = GuidedSearch(
                keyword_store=self.keyword_store, vector_store=self.vector_store,
                llm=self.llm, session_store=self.sessions,
            )
        return self._guided

    def _fetch_patent(self, number: str) -> Patent:
        """Store first, then the Google Patents connector (lazy import)."""
        found = self.keyword_store.get(PatentNumber.parse(number))
        if found is not None:
            return found
        try:
            from patentkit.connectors.inference.google_patents import fetch_patent  # noqa: PLC0415
        except ImportError as exc:
            raise LookupError(
                f"Patent {number} is not in the local index and the Google Patents "
                "connector is unavailable; index it first with index_patents."
            ) from exc
        return fetch_patent(number)

    def _require_session(self, session_id: str) -> GuidedSearchSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise LookupError(f"No guided session with id {session_id!r}")
        return session

    @staticmethod
    def _chart_key(patent_number: str, claim_number: int) -> tuple[str, int]:
        """Cache key, kind-code-agnostic (US7000001B1 == US7000001)."""
        try:
            parsed = PatentNumber.parse(str(patent_number))
            normalized = f"{parsed.country_code}{parsed.number.lstrip('0') or '0'}"
        except ValueError:
            normalized = str(patent_number).strip().upper()
        return normalized, int(claim_number)

    def _cache_chart(self, chart: Any) -> None:
        try:
            key = self._chart_key(chart.query_patent, chart.claim_number)
        except (AttributeError, ValueError, TypeError):
            return
        self._charts[key] = chart

    def _cached_chart(self, patent_number: str, claim_number: int) -> Optional[Any]:
        return self._charts.get(self._chart_key(patent_number, claim_number))

    # ----------------------------------------------------------------- tools
    def search_patents(
        self,
        keywords: Optional[list[str]] = None,
        required_keywords: Optional[list[str]] = None,
        excluded_keywords: Optional[list[str]] = None,
        text: Optional[str] = None,
        minimum_match: Optional[int] = None,
        fields: Optional[list[str]] = None,
        art_classes: Optional[list[str]] = None,
        inventors: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
        before_date: Optional[str] = None,
        after_date: Optional[str] = None,
        countries: Optional[list[str]] = None,
        exclude_numbers: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """Search the indexed patent corpus with the full query parameter set.

        Keyword search (BM25) over title/abstract/claims/specification with
        metadata filters: required/excluded keywords, minimum-match, CPC art
        class prefixes, inventor/assignee substrings, before/after date
        cutoffs (YYYY-MM-DD), country codes, and an exclude list of patent
        numbers. Returns ranked results with highlighted passages.
        """
        spec = QuerySpec(
            keywords=keywords or [],
            required_keywords=required_keywords or [],
            excluded_keywords=excluded_keywords or [],
            text=text,
            minimum_match=minimum_match,
            fields=fields or ["title", "abstract", "claims", "specification"],
            art_classes=art_classes or [],
            inventors=inventors or [],
            assignees=assignees or [],
            before_date=date.fromisoformat(before_date) if before_date else None,
            after_date=date.fromisoformat(after_date) if after_date else None,
            countries=countries or [],
            exclude_numbers=exclude_numbers or [],
            limit=limit,
        )
        results = self.keyword_store.search(spec.to_search_query())
        return {"count": len(results), "results": [_result_dict(r) for r in results]}

    def get_patent(self, number: str) -> dict:
        """Fetch one patent record (title, abstract, claims, dates, citations,
        classifications) by number, e.g. 'US10123456B2'. Looks in the local
        index first, then falls back to the Google Patents connector."""
        try:
            patent = self._fetch_patent(number)
        except (LookupError, ValueError) as exc:
            return {"error": str(exc)}
        return patent.model_dump(mode="json")

    def index_patents(self, numbers: Optional[list[str]] = None,
                      jsonl_path: Optional[str] = None) -> dict:
        """Add patents to the searchable index, either by fetching the given
        patent numbers (requires the Google Patents connector) or by loading
        canonical Patent JSON records from a .jsonl file (one per line)."""
        patents: list[Patent] = []
        errors: list[str] = []
        if jsonl_path:
            path = Path(jsonl_path)
            if not path.exists():
                return {"error": f"No such file: {jsonl_path}"}
            for i, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                try:
                    patents.append(Patent.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"line {i + 1}: {exc}")
        for number in numbers or []:
            try:
                patents.append(self._fetch_patent(number))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{number}: {exc}")
        indexed = self.keyword_store.index(patents) if patents else 0
        if self.vector_store is not None and patents:
            try:
                self.vector_store.index(patents)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vector indexing failed: {exc}")
        return {"indexed": indexed, "corpus_size": len(self.keyword_store), "errors": errors}

    def guided_search_start(self, search_type: str, patent_number: Optional[str] = None,
                            product_description: Optional[str] = None,
                            claims: Optional[list[int]] = None,
                            key_limitations: Optional[Union[list[str], str]] = None) -> dict:
        """Start a guided patent search session. search_type is 'invalidity'
        (prior art against a patent — needs patent_number), 'fto' (freedom to
        operate — needs product_description), or 'infringement' (needs
        patent_number). Optional key_limitations (e.g. from
        summarize_key_limitations) are injected into the executing agent's
        task context. Returns a session_id, a preview of the starting query
        angles (the executing agent generates and refines its own queries),
        and an up-front time estimate; present the preview to the user and
        collect feedback before executing."""
        try:
            session = self.guided.start(
                search_type, target_patent_number=patent_number,  # type: ignore[arg-type]
                product_description=product_description, claims=claims,
                fetch=self._fetch_patent if patent_number else None,
                key_limitations=key_limitations,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return _session_dict(session)

    def guided_search_feedback(self, session_id: str, feedback: dict) -> dict:
        """Apply user feedback to a guided session. The feedback dict may have
        'queries' ([{query_index, verdict: good|too_broad|too_narrow|off_topic,
        note}]), 'results' ([{patent_number, relevant, note}]), 'passages'
        ([{patent_number, passage_text, relevant, note}]), and 'free_text'.
        Before execution it adjusts the plan preview and seeds the agent's
        initial guidance; after execution it is queued and injected as a user
        message when the SAME agent conversation resumes (irrelevant results
        also become hard exclusions). Returns the updated session state."""
        try:
            session = self._require_session(session_id)
            parsed = SearchFeedback.model_validate(feedback)
            if session.state == "awaiting_result_feedback":
                session = self.guided.apply_result_feedback(session, parsed)
            else:
                session = self.guided.apply_plan_feedback(session, parsed)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return _session_dict(session)

    def guided_search_execute(self, session_id: str,
                              budget_seconds: Optional[float] = None,
                              max_steps: Optional[int] = None) -> dict:
        """Execute (or resume) a guided session's agentic search: one LLM
        agent issues and refines its own queries via tool use under a time
        budget; queued feedback is injected into the resumed conversation.
        Optional budget_seconds / max_steps override the per-round budgets —
        pass small values for short rounds the user can steer between with
        guided_search_feedback. Returns ranked results, what was excluded and
        why, timing, the stop reason, a trace summary (full trace via
        get_search_trace), and a chat-ready 'progress' markdown summary.
        Without an LLM it runs a clearly-labeled degraded single keyword
        pass."""
        try:
            session = self._require_session(session_id)
            session = self.guided.execute(session, budget_seconds=budget_seconds,
                                          max_steps=max_steps)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        out = _session_dict(session)
        out["results"] = session.last_results
        result = session.params.get("result") or {}
        out["excluded"] = result.get("excluded", {})
        out["timing"] = result.get("timing", {})
        out["progress"] = _progress_markdown(session)
        if self.notifiers:
            try:
                from patentkit.notify.base import notify_search_complete  # noqa: PLC0415
                notify_search_complete(self.notifiers, session)
            except Exception:  # noqa: BLE001
                logger.exception("completion notification failed")
        return out

    def guided_search_status(self, session_id: str) -> dict:
        """Get the state of a guided session: its plan preview, iteration
        count, time estimate, last results, stop reason, elapsed time, and a
        trace summary (step count, queries issued so far, the agent's
        intermediate shortlist)."""
        try:
            session = self._require_session(session_id)
        except LookupError as exc:
            return {"error": str(exc)}
        out = _session_dict(session)
        out["results"] = session.last_results
        out["feedback_rounds"] = len(session.feedback_history)
        return out

    def get_search_trace(self, session_id: str) -> dict:
        """Get the full reasoning trace of a guided session's last agentic
        execution. The 'markdown' field is a chat-ready narrative — one
        section per agent round with its thinking, queries (inline code),
        result counts, shortlist updates, injected feedback, and the stop
        reason — plus the raw queries and shortlist history. Show the
        markdown to the user so they can give feedback on specific queries
        and results."""
        try:
            session = self._require_session(session_id)
        except LookupError as exc:
            return {"error": str(exc)}
        trace = GuidedSearch.session_trace(session)
        if trace is None:
            return {"error": f"Session {session_id} has no persisted trace yet "
                             "(execute it first; degraded keys-free runs have no trace)."}
        from patentkit.formatting.trace import search_trace_markdown  # noqa: PLC0415
        return {
            "session_id": session_id,
            "markdown": search_trace_markdown(trace),
            "queries": trace.queries,
            "shortlist_history": trace.shortlist_history,
            "feedback": trace.feedback,
            "stop_reason": trace.stop_reason,
            "elapsed_s": trace.elapsed_s,
            "step_count": len(trace.steps),
        }

    def estimate_search_time(self, search_type: str = "invalidity", n_queries: int = 4,
                             corpus_size: Optional[int] = None,
                             charting_claims: int = 0) -> dict:
        """Estimate how long a search will take before running it, from the
        number of planned queries, the corpus size (defaults to the local
        index size), whether LLM reranking is on, and how many claims will be
        charted. Returns seconds and a human-readable duration."""
        if corpus_size is None:
            corpus_size = max(len(self.keyword_store), 1)
        if search_type not in ("invalidity", "fto", "infringement"):
            search_type = "invalidity"
        plan = SearchPlan(
            search_type=search_type,  # type: ignore[arg-type]
            target="estimate", rationale="estimate",
            queries=[PlannedQuery(purpose="q", query=QuerySpec()) for _ in range(max(1, n_queries))],
        )
        seconds = estimate_search_seconds(plan, corpus_size=corpus_size,
                                          with_llm_rerank=self.llm is not None,
                                          charting_claims=charting_claims)
        return {"seconds": seconds, "human": humanize_seconds(seconds),
                "corpus_size": corpus_size, "llm_rerank": self.llm is not None}

    def build_claim_chart(self, patent_number: str, claim_number: int,
                          reference_numbers: list[str],
                          limitations_filter: Optional[list[str]] = None) -> dict:
        """Build an element-by-element invalidity claim chart mapping one claim
        of the target patent against the given prior-art references (works
        with a single reference too). The result includes a ready-to-display
        'markdown' field; an optional limitations_filter (limitation-text
        substrings) restricts the markdown to the matching rows. The chart is
        cached for export_claim_chart_docx. Requires the patentkit analysis
        module; references must be fetchable."""
        try:
            from patentkit.analysis.invalidity import build_claim_chart  # noqa: PLC0415
        except ImportError:
            return {"error": "Claim charting requires patentkit.analysis "
                             "(not available in this installation)."}
        try:
            patent = self._fetch_patent(patent_number)
            references = [self._fetch_patent(n) for n in reference_numbers]
            reference_texts = [(str(p.patent_number), p.text_for_search()) for p in references]
            chart = build_claim_chart(patent, claim_number, reference_texts, self.llm)
            for reference_chart, ref in zip(chart.references, references):
                reference_chart.reference_title = reference_chart.reference_title or ref.title
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        self._cache_chart(chart)
        data: dict[str, Any] = chart.model_dump(mode="json")
        try:
            data["coverage_summary"] = chart.coverage_summary()
        except Exception as exc:  # noqa: BLE001
            data["coverage_summary"] = f"unavailable: {exc}"
        data["markdown"] = self._chart_markdown(chart, limitations_filter)
        return data

    def _chart_markdown(self, chart: Any, limitations_filter: Optional[list[str]]) -> str:
        """Display markdown for a chart, optionally filtered to key rows."""
        try:
            from patentkit.formatting.claim_chart import (  # noqa: PLC0415
                claim_chart_markdown,
                filter_chart,
            )
        except ImportError as exc:  # pragma: no cover — formatting ships with core
            return f"(markdown rendering unavailable: {exc})"
        if limitations_filter:
            filtered = filter_chart(chart, limitations_filter)
            if filtered.limitations:
                note = (f"_Filtered to {len(filtered.limitations)} of "
                        f"{len(chart.limitations)} limitation(s) matching "
                        f"{limitations_filter!r}; coverage figures reflect the "
                        "filtered rows._\n\n")
                return note + claim_chart_markdown(filtered)
            return (f"_limitations_filter {limitations_filter!r} matched no "
                    "limitations; showing the full chart._\n\n"
                    + claim_chart_markdown(chart))
        return claim_chart_markdown(chart)

    def chart_limitation(self, limitation: str, patent: str, claim_number: int,
                         references: list[str]) -> dict:
        """Chart ONE claim limitation across one or more references: returns a
        markdown table with one row per reference (disclosure status,
        reasoning, quotes, citation). Reuses disclosure assessments cached by
        a previous build_claim_chart / chart_limitation call when available;
        only missing (limitation, reference) pairs are assessed. The merged
        chart is cached for export_claim_chart_docx."""
        try:
            from patentkit.analysis.invalidity import (  # noqa: PLC0415
                ClaimChart,
                ReferenceChart,
                assess_reference,
            )
            from patentkit.formatting.claim_chart import limitation_chart_markdown  # noqa: PLC0415
        except ImportError:
            return {"error": "Limitation charting requires patentkit.analysis "
                             "(not available in this installation)."}
        if not references:
            return {"error": "chart_limitation requires at least one reference number."}
        try:
            target = self._fetch_patent(patent)
        except (LookupError, ValueError) as exc:
            return {"error": str(exc)}
        claim = target.get_claim(int(claim_number))
        if claim is None:
            return {"error": f"Claim {claim_number} not found in {target.patent_number}"}

        cached = self._cached_chart(patent, claim_number)
        try:
            # the claim's precomputed structural units (deterministic — no LLM)
            limitations = ((cached.limitations if cached is not None else None)
                           or claim.get_limitations())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Could not split claim {claim_number} into limitations: {exc}"}
        needle = " ".join(str(limitation).split()).lower()
        match = next(
            (lim for lim in limitations
             if needle in " ".join(lim.text.split()).lower()
             or " ".join(lim.text.split()).lower() in needle),
            None,
        )
        if match is None:
            return {"error": f"No limitation of claim {claim_number} matches "
                             f"{limitation!r}. Limitations: "
                             + "; ".join(lim.text for lim in limitations)}

        reference_charts: list[Any] = []
        reused: list[str] = []
        assessed: list[str] = []
        for number in references:
            try:
                normalized = str(PatentNumber.parse(str(number)))
            except ValueError:
                normalized = str(number)
            finding = None
            cached_ref = None
            if cached is not None:
                cached_ref = next(
                    (r for r in cached.references
                     if self._chart_key(r.reference_number, 0)
                     == self._chart_key(normalized, 0)),
                    None,
                )
            if cached_ref is not None:
                finding = next((f for f in cached_ref.findings
                                if f.limitation.text == match.text), None)
            if finding is not None:
                reference_charts.append(ReferenceChart(
                    reference_number=cached_ref.reference_number,
                    reference_title=cached_ref.reference_title,
                    findings=[finding],
                ))
                reused.append(cached_ref.reference_number)
                continue
            try:
                ref_patent = self._fetch_patent(str(number))
                reference_charts.append(assess_reference(
                    [match], ref_patent.text_for_search(), str(ref_patent.patent_number),
                    reference_title=ref_patent.title, llm=self.llm,
                ))
                assessed.append(str(ref_patent.patent_number))
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Assessing {number} failed: {exc}"}

        view = ClaimChart(query_patent=str(target.patent_number),
                          claim_number=int(claim_number),
                          limitations=[match], references=reference_charts)
        self._merge_chart_into_cache(view, match)
        return {
            "patent": str(target.patent_number),
            "claim_number": int(claim_number),
            "limitation": match.text,
            "label": match.label,
            "references": [r.model_dump(mode="json") for r in reference_charts],
            "reused_assessments": reused,
            "new_assessments": assessed,
            "markdown": limitation_chart_markdown(view, match),
        }

    def _merge_chart_into_cache(self, view: Any, limitation: Any) -> None:
        """Merge a single-limitation chart into the cached chart for export."""
        cached = self._cached_chart(view.query_patent, view.claim_number)
        if cached is None:
            self._cache_chart(view)
            return
        if all(lim.text != limitation.text for lim in cached.limitations):
            cached.limitations.append(limitation)
        for reference_chart in view.references:
            key = self._chart_key(reference_chart.reference_number, 0)
            existing = next((r for r in cached.references
                             if self._chart_key(r.reference_number, 0) == key), None)
            if existing is None:
                cached.references.append(reference_chart)
                continue
            existing.findings = (
                [f for f in existing.findings if f.limitation.text != limitation.text]
                + list(reference_chart.findings)
            )

    def export_claim_chart_docx(self, patent: str, claim_number: int,
                                out_path: Optional[str] = None) -> dict:
        """Export the most recent claim chart for (patent, claim_number) —
        built earlier via build_claim_chart or chart_limitation — as a
        color-coded DOCX file, without re-running any LLM calls. Returns the
        absolute path written. Requires the docx extra
        (pip install 'patentkit[docx]')."""
        chart = self._cached_chart(patent, claim_number)
        if chart is None:
            return {"error": f"No cached claim chart for {patent} claim {claim_number}. "
                             "Build one first with build_claim_chart (or chart_limitation), "
                             "then call export_claim_chart_docx again."}
        try:
            from patentkit.formatting.claim_chart import claim_chart_docx  # noqa: PLC0415
        except ImportError as exc:
            return {"error": f"DOCX export requires patentkit.formatting: {exc}"}
        if out_path is None:
            base = (self.sessions.directory or Path.cwd()) / "exports"
            normalized, claim = self._chart_key(patent, claim_number)
            out_path = str(base / f"claim_chart_{normalized}_claim{claim}.docx")
        path = Path(out_path).expanduser().resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            claim_chart_docx(chart, str(path))
        except ImportError as exc:  # python-docx missing at runtime
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"DOCX export failed: {exc}"}
        return {"path": str(path), "patent": str(chart.query_patent),
                "claim_number": int(chart.claim_number),
                "limitations": len(chart.limitations),
                "references": [r.reference_number for r in chart.references]}

    def summarize_key_limitations(self, patent: str, claim_number: int) -> dict:
        """Summarize the KEY limitations of one claim — the ones added by
        amendment or argued for allowance per the prosecution history (the
        prime targets of an invalidity search). With USPTO_ODP_API_KEY set
        (or a file-wrapper-enriched record) the LLM reads the file wrapper;
        without it, degrades to a plain claim split with a clear note. The
        result includes display markdown, and key_limitations can be passed
        straight to guided_search_start."""
        try:
            target = self._fetch_patent(patent)
        except (LookupError, ValueError) as exc:
            return {"error": str(exc)}
        claim = target.get_claim(int(claim_number))
        if claim is None:
            return {"error": f"Claim {claim_number} not found in {target.patent_number}"}
        try:
            # the claim's precomputed structural units (deterministic — no LLM)
            limitations = claim.get_limitations()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Could not split claim {claim_number} into limitations: {exc}"}
        limitation_texts = [lim.text for lim in limitations]

        wrapper_text = target.file_wrapper_text
        wrapper_error: Optional[str] = None
        if not wrapper_text and os.environ.get("USPTO_ODP_API_KEY"):
            wrapper_text, wrapper_error = self._fetch_file_wrapper_text(target)

        if wrapper_text and self.llm is not None:
            return self._summarize_from_wrapper(target, claim, limitation_texts, wrapper_text)

        # Degraded path: no prosecution-history context (or no LLM to read it).
        # No LLM is needed here — the precomputed limitation units are returned.
        if wrapper_text:  # have the history but no LLM to read it
            note = ("No LLM configured to read the prosecution history; "
                    "returning ALL precomputed limitations of the claim.")
        elif wrapper_error:
            note = ("File-wrapper retrieval failed; returning ALL precomputed "
                    f"limitations of the claim. Error: {wrapper_error}")
        elif os.environ.get("USPTO_ODP_API_KEY"):
            note = ("The file wrapper contained no readable documents; "
                    "returning ALL precomputed limitations of the claim.")
        else:
            note = ("File-wrapper (prosecution-history) context is unavailable "
                    "without USPTO_ODP_API_KEY; returning ALL precomputed "
                    "limitations of the claim instead of the allowance-critical "
                    "ones.")
        lines = [f"## Claim limitations — {target.patent_number}, claim {claim_number}",
                 "", f"_{note}_", ""]
        lines += [f"- **{lim.label}** {lim.text}" if lim.label else f"- {lim.text}"
                  for lim in limitations]
        return {
            "patent": str(target.patent_number),
            "claim_number": int(claim_number),
            "mode": "claim_split_only",
            "key_limitations": limitation_texts,
            "limitations": [{"label": lim.label, "text": lim.text} for lim in limitations],
            "note": note,
            "markdown": "\n".join(lines),
        }

    def _fetch_file_wrapper_text(self, target: Patent) -> tuple[Optional[str], Optional[str]]:
        """Pull prosecution-history text via the ODP connector, best-effort."""
        try:
            from patentkit.connectors.inference.file_wrapper import FileWrapperClient  # noqa: PLC0415
        except ImportError as exc:
            return None, f"file-wrapper connector unavailable: {exc}"
        try:
            client = FileWrapperClient()
            app_number = target.application_number or client.app_number_for_patent(
                target.patent_number)
            if not app_number:
                return None, (f"could not resolve an application number for "
                              f"{target.patent_number}")
            return client.get_file_wrapper_text(app_number) or None, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("file-wrapper fetch failed for %s", target.patent_number,
                           exc_info=True)
            return None, str(exc)

    def _summarize_from_wrapper(self, target: Patent, claim: Any,
                                limitation_texts: list[str], wrapper_text: str) -> dict:
        """LLM pass over the prosecution history: which limitations mattered."""
        prompt = _KEY_LIMITATIONS_PROMPT.format(
            patent=target.patent_number,
            claim_number=claim.number,
            claim=claim.text,
            limitations="\n".join(f"- {t}" for t in limitation_texts),
            wrapper=wrapper_text[:_WRAPPER_CHAR_CAP],
        )
        try:
            data = self.llm.complete_json(prompt, max_tokens=4096)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"File-wrapper summarization failed: {exc}"}
        if not isinstance(data, dict):
            data = {}
        details: list[dict] = []
        for item in data.get("key_limitations") or []:
            if not isinstance(item, dict) or not str(item.get("limitation", "")).strip():
                continue
            raw = " ".join(str(item["limitation"]).split())
            # keep the result honest: snap to an actual limitation when possible
            matched = next(
                (t for t in limitation_texts
                 if raw.lower() in " ".join(t.split()).lower()
                 or " ".join(t.split()).lower() in raw.lower()),
                raw,
            )
            details.append({"limitation": matched, "why": str(item.get("why", "")).strip()})
        summary = str(data.get("summary", "")).strip()
        if not details:
            return {"error": "The LLM identified no key limitations from the file "
                             "wrapper; inspect the prosecution history manually or "
                             "fall back to the full claim split."}
        lines = [f"## Key limitations — {target.patent_number}, claim {claim.number}", "",
                 "_Source: USPTO file wrapper (prosecution history) — limitations "
                 "added or argued for allowance._", ""]
        lines += [f"- **{d['limitation']}**" + (f" — {d['why']}" if d["why"] else "")
                  for d in details]
        if summary:
            lines += ["", summary]
        return {
            "patent": str(target.patent_number),
            "claim_number": int(claim.number),
            "mode": "file_wrapper",
            "key_limitations": [d["limitation"] for d in details],
            "details": details,
            "summary": summary or None,
            "markdown": "\n".join(lines),
        }

    def cluster_patents(self, numbers: Optional[list[str]] = None,
                        query: Optional[str] = None) -> dict:
        """Cluster a set of patents (by number list, or the results of a
        keyword query) into technology topics. Requires the viz extra
        (pip install 'patentkit[viz]')."""
        try:
            from patentkit.viz.clustering import cluster_patents as _cluster  # noqa: PLC0415
        except ImportError:
            return {"error": "Clustering requires the viz extra: pip install 'patentkit[viz]'"}
        try:
            patents: list[Patent] = []
            if numbers:
                patents = [self._fetch_patent(n) for n in numbers]
            elif query:
                hits = self.search_patents(keywords=query.split(), limit=100)["results"]
                patents = [self._fetch_patent(h["patent_number"]) for h in hits]
            from patentkit.search.vector import HashingEmbeddings  # noqa: PLC0415
            result = _cluster(patents, HashingEmbeddings(), llm=self.llm)
            return {
                "labels": list(result.labels),
                "topics": {str(k): v for k, v in result.topics.items()},
                "silhouette": result.silhouette,
                "representative": {str(k): v for k, v in result.representative.items()},
            }
        except ImportError as exc:  # numpy/sklearn missing
            return {"error": f"Clustering requires the viz extra (pip install 'patentkit[viz]'): {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def run_eval(self, dataset_path: Optional[str] = None) -> dict:
        """Run the search-performance eval harness over a dataset (default:
        the built-in toy dataset). Requires the patentkit evals module."""
        try:
            from patentkit.evals.datasets import (  # noqa: PLC0415
                default_ipr_toy_dataset,
                load_queryset_jsonl,
            )
            from patentkit.evals.harness import EvalRunner, searchfn_from_stores  # noqa: PLC0415
        except ImportError:
            return {"error": "Evals require the patentkit.evals module "
                             "(not available in this installation)."}
        try:
            dataset = load_queryset_jsonl(dataset_path) if dataset_path else default_ipr_toy_dataset()
            runner = EvalRunner(searchfn_from_stores(self.keyword_store), dataset,
                                name="patentkit-eval")
            report = runner.run()
            return {
                "name": report.name,
                "aggregates": report.aggregates,
                "mean_recall_curve": report.mean_curve,
                "queries": len(report.rows),
                "errors": report.errors,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def notify(self, subject: str, body: str) -> dict:
        """Send a notification (Slack/email, as configured on this server)
        with the given subject and body — e.g. to announce that a long search
        finished and where to find the results."""
        if not self.notifiers:
            return {"sent": 0, "note": "no notifiers configured"}
        sent, errors = 0, []
        for notifier in self.notifiers:
            try:
                notifier.send(subject, body)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("notifier failed")
                errors.append(f"{type(notifier).__name__}: {exc}")
        return {"sent": sent, "errors": errors}


# --------------------------------------------------------------- tool specs

def _arr(description: str, item_type: str = "string") -> dict:
    return {"type": "array", "items": {"type": item_type}, "description": description}


_SEARCH_QUERY_PROPERTIES: dict[str, dict] = {
    "keywords": _arr("Search keywords/phrases (OR semantics with minimum_match)."),
    "required_keywords": _arr("Keywords that MUST all appear (AND semantics)."),
    "excluded_keywords": _arr("Tokens/phrases that must NOT appear anywhere."),
    "text": {"type": "string", "description": "Free-text query for phrase/semantic matching."},
    "minimum_match": {"type": "integer",
                      "description": "Minimum number of `keywords` that must match (default len//3, >=1)."},
    "fields": _arr("Fields to search; subset of title, abstract, claims, specification."),
    "art_classes": _arr("CPC/IPC art-class prefixes, e.g. ['G06F16', 'H04L']."),
    "inventors": _arr("Inventor-name substrings to require."),
    "assignees": _arr("Assignee-name substrings to require."),
    "before_date": {"type": "string", "description": "Only documents effective before this date "
                                                     "(YYYY-MM-DD) — the prior-art cutoff."},
    "after_date": {"type": "string", "description": "Only documents effective after this date (YYYY-MM-DD)."},
    "countries": _arr("Country codes to allow, e.g. ['US', 'EP']."),
    "exclude_numbers": _arr("Patent numbers to exclude from results."),
    "limit": {"type": "integer", "description": "Maximum results to return.", "default": 25},
}

_FEEDBACK_SCHEMA: dict = {
    "type": "object",
    "description": "Structured user feedback.",
    "properties": {
        "queries": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "query_index": {"type": "integer"},
                "verdict": {"type": "string",
                            "enum": ["good", "too_broad", "too_narrow", "off_topic"]},
                "note": {"type": "string"},
            },
            "required": ["query_index", "verdict"],
        }},
        "results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "patent_number": {"type": "string"},
                "relevant": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["patent_number"],
        }},
        "passages": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "patent_number": {"type": "string"},
                "passage_text": {"type": "string"},
                "relevant": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["patent_number", "passage_text"],
        }},
        "free_text": {"type": "string"},
    },
}


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


#: One entry per PatentToolset method; the single source of truth for both
#: the MCP server and the OpenAI function-calling layer.
TOOL_SPECS: list[dict] = [
    _spec(
        "search_patents",
        "Call this for a quick one-off lookup over the indexed patent corpus "
        "(BM25 keyword search with metadata filters) — e.g. to check what is "
        "indexed or sanity-check a query idea. For a real prior-art / FTO / "
        "infringement investigation, prefer guided_search_start, which runs "
        "an agentic multi-query search. Supports the full query parameter "
        "set: keywords, required/excluded keywords, free text, minimum-match, "
        "field selection, CPC art classes, inventors, assignees, date "
        "cutoffs, countries, and exclusions; returns ranked patents with "
        "highlighted passages.",
        dict(_SEARCH_QUERY_PROPERTIES),
    ),
    _spec(
        "get_patent",
        "Call this when you need one patent's full record — title, abstract, "
        "claims, dates, citations, classifications — e.g. before charting it "
        "or discussing its claims. Checks the local index first, then Google "
        "Patents when that connector is available.",
        {"number": {"type": "string", "description": "Patent number, e.g. 'US10123456B2'."}},
        ["number"],
    ),
    _spec(
        "index_patents",
        "Call this BEFORE searching when the corpus is empty or missing "
        "documents: adds patents to the searchable index by patent numbers "
        "(fetched via the Google Patents connector) and/or a JSONL file of "
        "canonical Patent records (one JSON object per line).",
        {
            "numbers": _arr("Patent numbers to fetch and index."),
            "jsonl_path": {"type": "string", "description": "Path to a .jsonl file of Patent records."},
        },
    ),
    _spec(
        "guided_search_start",
        "Call this to begin any serious patent search ('invalidity' needs "
        "patent_number; 'fto' needs product_description; 'infringement' "
        "needs patent_number). Returns a session_id, a preview of the "
        "starting query angles (the executing LLM agent generates and "
        "refines its own queries at run time), and an up-front time "
        "estimate. Present the preview (and estimate) to the user and "
        "collect feedback before calling guided_search_execute. Pass "
        "key_limitations (e.g. from summarize_key_limitations) to focus the "
        "agent on the allowance-critical limitations. Invalidity searches "
        "exclude examiner-cited art and family members by default — enforced "
        "at the tool layer.",
        {
            "search_type": {"type": "string", "enum": ["invalidity", "fto", "infringement"],
                            "description": "Which search workflow to run."},
            "patent_number": {"type": "string", "description": "Target patent number."},
            "product_description": {"type": "string", "description": "Target product (FTO)."},
            "claims": _arr("Claim numbers in focus (default: independent claims).", "integer"),
            "key_limitations": _arr(
                "Key claim limitations to prioritize (e.g. the key_limitations "
                "from summarize_key_limitations); injected into the agent's "
                "task context on first execution."),
        },
        ["search_type"],
    ),
    _spec(
        "guided_search_feedback",
        "Call this whenever the user comments on a guided search's queries "
        "or results (between rounds, or after reviewing the trace). Before "
        "execution it adjusts the plan preview and seeds the agent's initial "
        "guidance; after execution it is queued and injected as a user "
        "message when the SAME agent conversation resumes (irrelevant "
        "results also become hard exclusions enforced at the tool layer). "
        "Returns the updated state.",
        {
            "session_id": {"type": "string", "description": "Session id from guided_search_start."},
            "feedback": _FEEDBACK_SCHEMA,
        },
        ["session_id", "feedback"],
    ),
    _spec(
        "guided_search_execute",
        "Call this to run (or RESUME — same session_id continues the same "
        "agent conversation) a guided session's agentic search: one LLM "
        "agent iteratively generates queries, runs them as tools, reads the "
        "results, refines, and finishes with ranked candidates — under a "
        "step/wall-clock budget, with a full saved reasoning trace. Pass "
        "small budget_seconds/max_steps for short rounds the user can steer "
        "between with guided_search_feedback. Returns ranked results, what "
        "was excluded and why, timing, stop_reason, a trace summary (full "
        "trace via get_search_trace), and a chat-ready 'progress' markdown "
        "summary to show the user.",
        {
            "session_id": {"type": "string", "description": "Session id from guided_search_start."},
            "budget_seconds": {"type": "number",
                               "description": "Wall-clock budget override for THIS round "
                                              "(default 180s). Small values give the user "
                                              "interjection points between rounds."},
            "max_steps": {"type": "integer",
                          "description": "Agent-round budget override for THIS round "
                                         "(default 16)."},
        },
        ["session_id"],
    ),
    _spec(
        "guided_search_status",
        "Call this to check on a guided session without changing it: plan "
        "preview, iteration, time estimate, feedback rounds, the latest "
        "ranked results, stop_reason, elapsed time, and a trace summary "
        "(step count, queries issued, the agent's intermediate shortlist).",
        {"session_id": {"type": "string", "description": "Session id from guided_search_start."}},
        ["session_id"],
    ),
    _spec(
        "get_search_trace",
        "Call this after guided_search_execute when the user should see HOW "
        "the agent searched: the 'markdown' field is a chat-ready narrative "
        "(one section per round — thinking, queries as inline code, result "
        "counts, shortlist updates, injected feedback, stop reason) you can "
        "display verbatim; raw queries and shortlist history are included "
        "for programmatic use. Show it to collect feedback on specific "
        "queries and results.",
        {"session_id": {"type": "string", "description": "Session id from guided_search_start."}},
        ["session_id"],
    ),
    _spec(
        "estimate_search_time",
        "Call this before committing to a search when the user asks how long "
        "it will take (agentic model: expected agent steps x per-step "
        "latency, plus a corpus-size factor and per-claim charting cost). "
        "Returns seconds and a human-readable duration.",
        {
            "search_type": {"type": "string", "enum": ["invalidity", "fto", "infringement"],
                            "default": "invalidity"},
            "n_queries": {"type": "integer", "description": "Planned query count.", "default": 4},
            "corpus_size": {"type": "integer",
                            "description": "Corpus size (default: local index size)."},
            "charting_claims": {"type": "integer",
                                "description": "Claims that will be charted afterwards.", "default": 0},
        },
    ),
    _spec(
        "build_claim_chart",
        "Call this after a search to map one claim element-by-element "
        "against one or more prior-art references. The result's 'markdown' "
        "field is a ready-to-display claim-chart table (pass "
        "limitations_filter to chart only the important limitations); "
        "structured findings and a coverage summary are included for "
        "programmatic use, and the chart is cached for "
        "export_claim_chart_docx. For a single limitation across references "
        "use chart_limitation instead.",
        {
            "patent_number": {"type": "string", "description": "Target patent number."},
            "claim_number": {"type": "integer", "description": "Claim to chart."},
            "reference_numbers": _arr("Prior-art reference patent numbers (one or more)."),
            "limitations_filter": _arr(
                "Optional limitation-text substrings; the markdown chart is "
                "restricted to the matching limitation rows."),
        },
        ["patent_number", "claim_number", "reference_numbers"],
    ),
    _spec(
        "chart_limitation",
        "Call this when the user cares about ONE claim limitation (e.g. the "
        "allowance-critical one) across several references — say results A, "
        "B, C of a search. Returns a markdown table with one row per "
        "reference: disclosure status, reasoning, quotes, and citation. "
        "Reuses assessments cached by build_claim_chart/chart_limitation, so "
        "it is cheap to call after a full chart; the merged chart is cached "
        "for export_claim_chart_docx.",
        {
            "limitation": {"type": "string",
                           "description": "The limitation, verbatim or a distinctive "
                                          "substring of it."},
            "patent": {"type": "string", "description": "Target patent number."},
            "claim_number": {"type": "integer", "description": "Claim the limitation belongs to."},
            "references": _arr("Reference patent numbers to assess (one or more)."),
        },
        ["limitation", "patent", "claim_number", "references"],
    ),
    _spec(
        "export_claim_chart_docx",
        "Call this when the user wants a claim chart as a Word document: "
        "writes the MOST RECENT cached chart for (patent, claim_number) — "
        "from build_claim_chart or chart_limitation — as a color-coded DOCX "
        "and returns the absolute path. No LLM calls are re-run; if no chart "
        "is cached you must build one first. Requires the docx extra.",
        {
            "patent": {"type": "string", "description": "Target patent number."},
            "claim_number": {"type": "integer", "description": "Charted claim number."},
            "out_path": {"type": "string",
                         "description": "Output .docx path (default: an exports/ folder "
                                        "under the session dir or cwd)."},
        },
        ["patent", "claim_number"],
    ),
    _spec(
        "summarize_key_limitations",
        "Call this BEFORE an invalidity search to find which limitations of "
        "a claim were added or argued for allowance (the prosecution-history "
        "'key limitations' — the prime invalidity targets). Uses the USPTO "
        "file wrapper when USPTO_ODP_API_KEY is set (or the record is "
        "already enriched); otherwise degrades to a plain claim split with a "
        "clear note. Returns display markdown plus key_limitations you can "
        "pass to guided_search_start and limitations_filter/chart_limitation.",
        {
            "patent": {"type": "string", "description": "Target patent number."},
            "claim_number": {"type": "integer", "description": "Claim to analyze."},
        },
        ["patent", "claim_number"],
    ),
    _spec(
        "cluster_patents",
        "Call this when the user wants a thematic overview of a patent set: "
        "clusters the patents into technology topics (requires the viz "
        "extra). Provide explicit patent numbers or a keyword query whose "
        "results will be clustered.",
        {
            "numbers": _arr("Patent numbers to cluster."),
            "query": {"type": "string", "description": "Keyword query selecting patents to cluster."},
        },
    ),
    _spec(
        "run_eval",
        "Call this only to measure search performance (precision/recall "
        "against labeled prior art), e.g. after changing the index — not "
        "part of a normal search workflow. Uses the built-in toy dataset "
        "unless a dataset path is given.",
        {"dataset_path": {"type": "string", "description": "Path to an eval dataset file."}},
    ),
    _spec(
        "notify",
        "Call this to alert the user out-of-band through the configured "
        "channels (Slack webhook, SendGrid/SMTP email) — e.g. to announce a "
        "finished long-running search.",
        {
            "subject": {"type": "string", "description": "Short subject/headline."},
            "body": {"type": "string", "description": "Message body."},
        },
        ["subject", "body"],
    ),
]

_TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def dispatch(toolset: PatentToolset, name: str, arguments: dict) -> dict:
    """Route one tool call to the toolset; always returns a JSON-able dict.

    Unknown tools and unexpected method errors come back as
    ``{"error": ...}`` so transport layers never crash on a bad call.
    """
    if name not in _TOOL_NAMES:
        return {"error": f"Unknown tool {name!r}. Known tools: {sorted(_TOOL_NAMES)}"}
    method = getattr(toolset, name)
    try:
        result = method(**(arguments or {}))
    except TypeError as exc:  # bad/missing arguments
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — transport layers must not crash
        logger.exception("tool %s failed", name)
        return {"error": f"{name} failed: {exc}"}
    # Defensive: guarantee JSON serializability for transport layers.
    return json.loads(json.dumps(result, default=str))


__all__ = ["PatentToolset", "TOOL_SPECS", "dispatch"]
